import os
import sys
import threading
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.config.source import Source
from sinner2.io.cv2_unicode import imread_unicode
from sinner2.pipeline.detectors import DetectorModel
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.face_map import FaceMap, UnmatchedPolicy
from sinner2.pipeline.face_map_geometry import FrameGeometry
from sinner2.pipeline.model_cache import (
    get_insightface_swap_model,
    get_model_path,
    record_actual_providers,
)
from sinner2.pipeline.processors.face_swapper_types import (
    RotationAngleSource,
    TargetSex,
)
from sinner2.pipeline.processors.occlusion import (
    FaceParser,
    OccluderModel,
    OcclusionMaskMode,
    apply_occlusion,
    build_occlusion_masker,
)
from sinner2.pipeline.processors.rotation_compensation import (
    compute_roll,
    swap_with_uprighting,
)
from sinner2.pipeline.processors.landmarker import (
    FaceLandmarker,
    landmark_68_to_5,
)
from sinner2.pipeline.processors.swapper_models import (
    FastPasteSwapper,
    GenericOnnxSwapper,
    SwapperModel,
    get_spec,
)
from sinner2.types import Frame

__all__ = [
    "FaceSwapper",
    "FaceSwapperParams",
    "RotationAngleSource",
    "SwapperModel",
    "TargetSex",
]


class FaceSwapperParams(SinnerBaseModel):
    model: SwapperModel = Field(
        default=SwapperModel.INSWAPPER_128,
        description="Swap model: inswapper_128 / reswapper_128 / ghost_* / "
        "simswap_256 / uniface_256",
    )
    detection_interval: int = Field(
        default=1, ge=1, description="Detect faces every Nth frame; >=1"
    )
    detection_size: int = Field(
        default=640,
        ge=64,
        description="Face-detector input size (square, px); smaller = faster "
        "detection but may miss small/distant faces. Aligned to a multiple of 32.",
    )
    detector: DetectorModel = Field(
        default=DetectorModel.BUFFALO_L,
        description="Target-face detector: buffalo_l (full pack, gender/pose) or "
        "the faster detection-only yoloface / scrfd_2.5g",
    )
    many_faces: bool = Field(
        default=True, description="Swap all detected faces (otherwise first only)"
    )
    fast_paste: bool = Field(
        default=True,
        description="Blend the swap back with the fast ROI feather paste "
        "(~2.7x faster per frame) instead of insightface's full-frame "
        "diff-based blend. inswapper/reswapper only — the 256px swappers "
        "always blend this way. Output-affecting.",
    )
    target_sex: TargetSex = Field(
        default=TargetSex.BOTH,
        description="Which detected faces to swap (M/F/B/I — match insightface .sex)",
    )
    # ---- Rotation compensation (experimental) ----
    # For faces tilted past the threshold, upright a crop, (re-)detect clean
    # keypoints, swap there, and composite the result back. Helps when the
    # detector's keypoints degrade at high in-plane roll; does nothing for
    # out-of-plane yaw. Output-affecting → part of the cache key.
    rotation_compensation: bool = Field(
        default=True, description="Upright tilted faces before swapping"
    )
    rotation_threshold_deg: int = Field(
        default=15, ge=0, le=90,
        description="Only compensate faces rolled at least this many degrees",
    )
    rotation_redetect: bool = Field(
        default=True,
        description="Re-detect on the uprighted crop for clean keypoints",
    )
    rotation_angle_source: RotationAngleSource = Field(
        default=RotationAngleSource.POSE,
        description="Measure roll from eye keypoints, the 3D pose estimate, "
        "or the 2dfan4 68-landmark eye-line",
    )
    # ---- Landmark refinement (experimental) ----
    # Replace each detected face's 5 keypoints with the 5 derived from 2dfan4's
    # 68 landmarks before swapping — more accurate alignment on tilted/hard
    # faces (the detector's 5 points degrade there). Output-affecting; the
    # refined geometry also flows to occlusion + enhancer via the ChainContext.
    landmark_refine: bool = Field(
        default=False, description="Refine keypoints with the 2dfan4 landmarker"
    )
    landmark_min_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Skip refinement when 2dfan4's confidence is below this",
    )
    # ---- Occlusion-aware masking ----
    # Mask the swap to the facial-skin region (parser) and/or the visible
    # (unoccluded) face surface (XSeg occluder). Output-affecting.
    occlusion_mask: bool = Field(
        default=False, description="Mask the swap to the real face region"
    )
    occlusion_mode: OcclusionMaskMode = Field(
        default=OcclusionMaskMode.REGION,
        description="Mask source: region (face parser), occluder (XSeg — sees "
        "hands/objects over the face), or both (strictest)",
    )
    occlusion_parser: FaceParser = Field(
        default=FaceParser.BISENET,
        description="Face parser for the region mask: torch bisenet/parsenet "
        "or the thread-safe ONNX exports",
    )
    occluder_model: OccluderModel = Field(
        default=OccluderModel.XSEG_1,
        description="Occluder model: xseg_1/2/3 or xseg_many (all three, "
        "strictest)",
    )


# SINNER2_GEOM_TRACE=1 prints a per-frame diagnostic when the geometry branch
# runs (detection-free face-mapping) — how many faces were rebuilt from the
# precomputed table and, per face, which identity it routed to and whether that
# identity has a usable assigned source. The one tool that distinguishes "the
# geometry is empty" from "the geometry routes to an unassigned identity" when
# mapped faces aren't swapping. Throttled so playback isn't flooded.
_GEOM_TRACE = bool(os.environ.get("SINNER2_GEOM_TRACE"))
_GEOM_TRACE_EVERY = 30


_CROP_THUMB_MAX = 96  # longest side of a comparison thumbnail, px


def _crop_thumb(img: Frame, bbox: Any) -> Frame | None:
    """The face's bbox region from `img`, downsized to a small thumbnail.
    None when the bbox is empty or off-frame."""
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = _CROP_THUMB_MAX / max(ch, cw)
    if scale < 1.0:
        crop = cast(
            Frame,
            cv2.resize(
                crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            ),
        )
    return crop.copy()


def _face_matches(face: Any, target_sex: TargetSex) -> bool:
    """Whether a detected face should be swapped under the given filter.

    BOTH always matches. M/F match only when insightface reports the
    same letter. Faces with no/unknown sex are SKIPPED for M/F filters
    (matches sinner1's behaviour — better to miss one face than swap
    the wrong gender). AS_SOURCE is resolved upstream into M/F/BOTH;
    if it leaks through here we treat it as BOTH to be safe."""
    if target_sex is TargetSex.BOTH or target_sex is TargetSex.AS_SOURCE:
        return True
    face_sex = getattr(face, "sex", None)
    if face_sex == target_sex.value:
        return True
    return False


def _load_inswapper(path: Path, providers: list[str]) -> Any:
    """Loader indirection so tests can stub the insightface call cheaply.

    `providers` is the swapper's ONNX execution-provider list (from its
    OnnxExecution profile). The model is fetched through model_cache so it is
    loaded ONCE and reused across session rebuilds — insightface's get_model
    builds a new ORT session every call, so loading it directly on each
    source/target change stacked inswapper sessions up in VRAM. After load,
    record what ORT actually wired up — `get_available_providers()` advertises
    EPs whose plugin DLL loads, but the EP can still fail at session
    construction (the TensorRT EP DLL loads even when nvinfer is missing; ORT
    then silently falls back). The recorded actual list drives the status-bar
    truth indicator."""
    model = get_insightface_swap_model(path, list(providers))
    # insightface wraps the ORT session in a Model object; the session
    # attribute name is stable for inswapper.
    session = getattr(model, "session", None)
    if session is not None:
        try:
            record_actual_providers(session.get_providers())
        except Exception:
            pass
    return model


class _MappedFace:
    """A face-like object rebuilt from precomputed geometry — enough for the
    swap (``bbox`` + ``kps``) and for the per-identity routing (``normed_embedding``
    set to the identity's centroid, so the existing source lookup maps it without
    change). No detection produced it; the runtime skips detection in mapping
    mode and feeds these instead."""

    def __init__(
        self,
        bbox: tuple[float, ...],
        kps: tuple[tuple[float, float], ...],
        centroid: tuple[float, ...],
        baked_roll: float | None = None,
        *,
        det_score: float = 1.0,
        sex: str | None = None,
        age: int | None = None,
        pose: tuple[float, float, float] | None = None,
    ) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = np.asarray(kps, dtype=np.float32)
        self.normed_embedding = np.asarray(centroid, dtype=np.float32)
        # Carry the matched identity's representative metadata so the detection
        # overlay shows REAL values (score / sex / age / pose) instead of the old
        # hardcoded "1.00" — detection is skipped, so this is what we know.
        self.det_score = float(det_score)
        self.sex = sex
        self.age = age
        self.pose = pose  # (pitch, yaw, roll) degrees, or None
        # Precomputed in-plane roll (degrees) from the geometry, or None to
        # measure it live. compute_roll reads this before any angle source so
        # rotation compensation keeps a steady angle without a pose estimate.
        self.baked_roll = baked_roll


def _mapped_meta(ident: Any, baked_roll: float | None) -> dict[str, Any]:
    """Display metadata (score/sex/age/pose) for a geometry face, taken from its
    matched identity so the overlay shows REAL values. ``pose`` is filled only
    when the identity has a full pose (the full buffalo_l pack ran); its roll uses
    the per-frame baked value when present, else the identity's."""
    if ident is None:
        return {}
    pitch = getattr(ident, "pitch", None)
    yaw = getattr(ident, "yaw", None)
    score = getattr(ident, "det_score", None)
    pose: tuple[float, float, float] | None = None
    if pitch is not None and yaw is not None:
        roll = baked_roll if baked_roll is not None else getattr(ident, "roll", None)
        pose = (float(pitch), float(yaw), float(roll) if roll is not None else 0.0)
    return {
        "det_score": float(score) if score is not None else 1.0,
        "sex": getattr(ident, "sex", None),
        "age": getattr(ident, "age", None),
        "pose": pose,
    }


class _CatalogMatcher:
    """Numpy-backed per-frame router, built once per FaceMap (cached on
    ``set_face_map``). One GEMV — the (N, dim) centroid matrix times the face
    embedding — replaces ``FaceMap.best_match``'s per-identity Python cosine
    loop, and ``by_id`` lets the detection-free geometry path skip rebuilding its
    id index every frame.

    Mirrors ``FaceMap.source_for`` exactly (nearest centroid ≥ threshold wins;
    ties go to the later identity like the pure ``>=`` scan; a matched-but-
    unassigned identity routes to None; no match → the unmatched policy). It
    assumes the query embedding is ALREADY L2-normalized — the runtime always
    passes ``normed_embedding`` / baked normalized centroids — so it drops the
    per-call renormalize ``best_match`` does."""

    def __init__(self, face_map: FaceMap) -> None:
        idents = face_map.identities
        self.by_id: dict[str, Any] = {i.id: i for i in idents}
        self._sources: list[str | None] = [i.source_path for i in idents]
        # Uniform-dim centroids → a real matrix; empty or ragged (never happens
        # with real ArcFace catalogs) → a 0×0 stub so routing falls through to
        # the unmatched policy, matching cosine()'s -1 for a length mismatch.
        dims = {len(i.centroid) for i in idents}
        self._matrix = (
            np.asarray([i.centroid for i in idents], dtype=np.float32)
            if len(dims) == 1 else np.zeros((0, 0), dtype=np.float32)
        )
        self._threshold = float(face_map.threshold)
        self._unmatched = face_map.unmatched
        self._default_source = face_map.default_source
        self._has_identities = bool(idents)
        self._first_source = idents[0].source_path if idents else None

    def source_for(self, embedding: Any) -> str | None:
        e = np.asarray(embedding, dtype=np.float32)
        if (
            self._matrix.shape[0]
            and e.ndim == 1
            and self._matrix.shape[1] == e.shape[0]
        ):
            sims = self._matrix @ e  # GEMV: cosine to every centroid at once
            best_i, best_sim = -1, -2.0  # mirror FaceMap.best_match (first wins ties)
            for i in range(sims.shape[0]):
                if sims[i] >= self._threshold and sims[i] > best_sim:
                    best_i, best_sim = i, float(sims[i])
            if best_i >= 0:
                return self._sources[best_i]  # matched (source may be None → skip)
        if self._unmatched is UnmatchedPolicy.DEFAULT:
            return self._default_source
        if self._unmatched is UnmatchedPolicy.FIRST and self._has_identities:
            return self._first_source
        return None


class FaceSwapper:
    name = "FaceSwapper"
    thread_safe = True  # one ORT session, called concurrently by N workers
    # Publishes this frame's detections into the ChainContext so downstream
    # processors (the enhancer's ONNX backends) reuse them instead of
    # re-detecting — see ChainContext.faces for the contract.
    accepts_context = True

    def __init__(
        self,
        source: Source,
        params: FaceSwapperParams | None = None,
        providers: list[str] | None = None,
        detection_sink: Any = None,
        face_map: FaceMap | None = None,
    ) -> None:
        self._source = source
        self._params = params or FaceSwapperParams()
        # Face mapping: when active, each detected face is routed to a per-
        # identity source (by ArcFace match) instead of the single global
        # source. Only the insightface backends (inswapper/reswapper) take a
        # per-call source, so multi-source is gated to them; generic backends
        # keep the single-source path (deferred). Needs buffalo_l embeddings.
        self._face_map = face_map
        # Numpy matcher over the catalog's centroids (GEMV routing + cached id
        # index) — rebuilt in lock-step with _face_map; see _CatalogMatcher.
        self._matcher = _CatalogMatcher(face_map) if face_map is not None else None
        self._geometry: FrameGeometry | None = None
        self._mapped_sources: dict[str, Any] = {}
        self._supports_multi_source = False
        self._geom_trace_n = 0  # SINNER2_GEOM_TRACE throttle counter
        # ONNX providers from the swapper's OnnxExecution profile. Distinguish
        # None (caller didn't specify → platform default) from an EMPTY list
        # (user explicitly selected no providers → pass it through; ORT then runs
        # on its CPU last-resort). Only None is treated as "unspecified".
        self._providers = list(providers) if providers is not None else None
        # Optional sink for the debug overlay: receives the PRE-swap detections
        # (duck-typed `.publish(faces, w, h)`); None outside the realtime GUI.
        self._detection_sink = detection_sink
        self._analyser: FaceAnalyser | None = None
        self._swapper: Any = None
        self._source_face: Any = None
        # 2dfan4 landmarker — built only when refinement or the landmark-68
        # angle source needs it. Shared thread-safe ONNX session.
        self._landmarker: FaceLandmarker | None = None
        # Occlusion masker — built only when enabled. Torch parsers are NOT
        # thread-safe (serialized via _mask_lock); the ONNX parsers run on a
        # shared thread-safe session and skip the lock (see process()).
        self._masker: Any = None
        self._mask_lock = threading.Lock()

    def setup(self) -> None:
        # None → platform default; an explicit [] stays empty (no GPU the user
        # didn't ask for — ORT falls back to CPU).
        providers = (
            list(DEFAULT_ONNX_PROVIDERS) if self._providers is None else self._providers
        )
        self._analyser = FaceAnalyser(
            detection_interval=self._params.detection_interval,
            providers=providers,
            detection_size=self._params.detection_size,
            detector=self._params.detector,
        )
        spec = get_spec(self._params.model)
        # insightface-compatible models (inswapper / reswapper) load through the
        # INSwapper wrapper; the rest use the facefusion-style generic backend.
        # Both expose .get(img, target, source, paste_back=True), so the swap
        # call sites (plain + rotation) don't branch.
        backend: Any
        if spec.insightface:
            backend = _load_inswapper(get_model_path(spec.model_file), providers)
            if self._params.fast_paste:
                # Replace insightface's internal full-frame diff-blend (~77% of
                # a FullHD swap frame) with the shared ROI feather paste.
                backend = FastPasteSwapper(backend)
        else:
            backend = GenericOnnxSwapper(spec, providers)
            backend.setup()
        source_img = imread_unicode(self._source.path)
        if source_img is None:
            raise ValueError(f"cannot read source image: {self._source.path}")
        faces = self._analyser.analyse_uncached(source_img)
        if not faces:
            raise ValueError(f"no face detected in source: {self._source.path}")
        self._source_face = faces[0]
        if not spec.insightface:
            # Generic backend caches the source identity (converted embedding or
            # aligned crop) — must run after the source face is detected.
            backend.prepare_source(source_img, self._source_face)
        self._swapper = backend
        # Only insightface backends accept a per-call source → multi-source
        # face mapping is theirs; generic backends stay single-source.
        self._supports_multi_source = spec.insightface
        if self._face_map_is_routable():
            self._prepare_mapped_sources(self._face_map)  # type: ignore[arg-type]
        if (
            self._params.landmark_refine
            or self._params.rotation_angle_source is RotationAngleSource.LANDMARK_68
        ):
            self._landmarker = FaceLandmarker(providers=providers)
            self._landmarker.setup()
        if self._params.occlusion_mask:
            # ONNX maskers share the swapper's EP profile; torch parsers
            # resolve their own device ("auto" → CUDA when available).
            self._masker = build_occlusion_masker(
                self._params.occlusion_mode,
                self._params.occlusion_parser,
                self._params.occluder_model,
                providers=providers,
            )
            self._masker.setup()

    def set_source(self, source: Source) -> None:
        """Re-point the swap at a new SOURCE face WITHOUT reloading any model:
        re-analyse the source image with the already-loaded analyser and update
        the source state in place. Cheap (no model load, no chain rebuild) — used
        for live source-face changes so the enhancer/upscaler worker instances
        aren't torn down + rebuilt. Records the source for the next setup() if
        called before setup(). Thread-safe: the new source state is published via
        single assignments (process() snapshots its handles), so concurrent
        workers never see a half-updated source."""
        analyser = self._analyser
        swapper = self._swapper
        if analyser is None or swapper is None:
            self._source = source  # not set up yet → next setup() picks it up
            return
        source_img = imread_unicode(source.path)
        if source_img is None:
            raise ValueError(f"cannot read source image: {source.path}")
        faces = analyser.analyse_uncached(source_img)
        if not faces:
            raise ValueError(f"no face detected in source: {source.path}")
        new_face = faces[0]
        if not get_spec(self._params.model).insightface:
            # Generic backend caches the source identity; prepare_source's final
            # assignment is atomic, so live workers stay consistent.
            swapper.prepare_source(source_img, new_face)
        self._source = source
        self._source_face = new_face  # atomic swap — read via process()'s snapshot

    def _face_map_is_routable(self) -> bool:
        """True when a face map should actually route per-face sources — it's
        active AND the backend supports per-call sources (insightface)."""
        return (
            self._face_map is not None
            and self._face_map.is_active()
            and self._supports_multi_source
        )

    def _prepare_mapped_sources(self, face_map: FaceMap) -> None:
        """Analyse each source image the map assigns into a swap-ready source
        face, keyed by its path. Sources that can't be read / have no face are
        skipped (their identities simply won't swap)."""
        analyser = self._analyser
        if analyser is None:
            return
        prepared: dict[str, Any] = {}
        for path in face_map.assigned_sources():
            img = imread_unicode(Path(path))
            if img is None:
                continue
            faces = analyser.analyse_uncached(img)
            if faces:
                prepared[path] = faces[0]
        self._mapped_sources = prepared

    def set_face_map(self, face_map: FaceMap | None) -> None:
        """Re-point per-face source routing WITHOUT reloading models — re-analyse
        the assigned source images and swap the routing state in place (like
        set_source). Single assignments so concurrent workers never see a
        half-updated map."""
        self._face_map = face_map
        # Rebuild the numpy matcher alongside the map (single assignment, so a
        # concurrent worker sees either the whole old or whole new matcher).
        self._matcher = _CatalogMatcher(face_map) if face_map is not None else None
        if self._face_map_is_routable():
            self._prepare_mapped_sources(face_map)  # type: ignore[arg-type]
        else:
            self._mapped_sources = {}

    def set_geometry(self, geometry: FrameGeometry | None) -> None:
        """Per-frame precomputed geometry for detection-free mapping (hot-applied
        like set_face_map). When set + mapping is active, ``process`` rebuilds
        each frame's faces from it instead of detecting; None reverts to live
        detection. Single assignment — concurrent workers snapshot it."""
        self._geometry = geometry

    def _geometry_faces(
        self, geometry: FrameGeometry | None, ctx: Any,
        frame: Frame | None = None,
    ) -> list[Any] | None:
        """Rebuild this frame's faces from ``geometry`` (no detection), or None to
        fall back to detecting. Each face gets its identity's centroid as the
        embedding so the multi-source routing maps it unchanged. Gated on mapping
        being active + multi-source-capable + a known frame index.

        Geometry is baked at the scan resolution; rescale its bboxes/kps to the
        frame we actually process (a processing_scale < 1 downsizes it) so the
        swap lands correctly. Old sidecars (no bake_size) assume the live frame."""
        face_map = self._face_map
        if (
            geometry is None
            or face_map is None
            or not face_map.is_active()
            or not self._supports_multi_source
            or ctx is None
            or getattr(ctx, "frame_index", None) is None
        ):
            return None
        sx, sy = 1.0, 1.0
        bake = geometry.bake_size
        if frame is not None and bake is not None and bake[0] > 0 and bake[1] > 0:
            fh, fw = frame.shape[:2]
            sx, sy = fw / bake[0], fh / bake[1]
        rescale = sx != 1.0 or sy != 1.0
        # Cached id index from the matcher (built with this map); fall back to a
        # per-frame build if the map was set without one (direct field set).
        matcher = self._matcher
        by_id = (
            matcher.by_id if matcher is not None
            else {ident.id: ident for ident in face_map.identities}
        )
        faces: list[Any] = []
        for gf in geometry.faces_at(ctx.frame_index):
            ident = by_id.get(gf.identity_id)
            # Prefer the baked embedding (routes against the LIVE catalog, so a
            # merged/reassigned/dropped identity is handled with no re-precompute).
            # Older sidecars have none → fall back to the matched identity's
            # centroid by id (skip if that identity is gone from the catalog).
            emb = gf.embedding or (ident.centroid if ident is not None else None)
            if not emb:
                continue
            bbox, kps = gf.bbox, gf.kps
            if rescale:
                bbox = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)
                kps = tuple((x * sx, y * sy) for x, y in kps)
            faces.append(
                _MappedFace(
                    bbox, kps, emb, gf.roll, **_mapped_meta(ident, gf.roll)
                )
            )
        # No precomputed faces for this frame (outside the analysed range, a
        # frame the scan missed, or a stale-only entry) → fall back to live
        # detection so the swap still shows. Geometry stays a fast cache over the
        # frames it DID cover; it never makes a face silently un-swappable.
        return faces or None

    def _trace_geometry(
        self, faces: list[Any], face_map: FaceMap | None, ctx: Any
    ) -> None:
        """SINNER2_GEOM_TRACE diagnostic (read-only). For the geometry branch,
        print — per rebuilt face — which catalog identity it routes to, whether
        that identity has a source assigned, and whether a prepared source face
        exists. Reveals whether non-swapping geometry is empty, routes to an
        unassigned identity, or has an unprepared source. Throttled."""
        self._geom_trace_n += 1
        if self._geom_trace_n % _GEOM_TRACE_EVERY != 1:
            return
        fidx = getattr(ctx, "frame_index", None)
        assigned = sorted(
            i.id[:8]
            for i in (face_map.identities if face_map is not None else ())
            if i.source_path
        )
        rows: list[str] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            match = (
                face_map.best_match(emb)
                if (face_map is not None and emb is not None)
                else None
            )
            src = self._mapped_source_face(f, self._matcher)
            rows.append(
                f"match={match.id[:8] if match is not None else None}"
                f"/srcAssigned={'Y' if (match is not None and match.source_path) else 'n'}"
                f"/srcPrepared={'Y' if src is not None else 'n'}"
            )
        active = face_map.is_active() if face_map is not None else None
        print(
            f"[geom] frame={fidx} built={len(faces)} active={active} "
            f"multiCapable={self._supports_multi_source} "
            f"preparedSources={len(self._mapped_sources)} "
            f"assignedIdentities={assigned} faces=[{'; '.join(rows)}]",
            file=sys.stderr,
            flush=True,
        )

    def _mapped_source_face(self, face: Any, matcher: "_CatalogMatcher | None") -> Any:
        """The prepared source face this detected face routes to, or None to skip
        it (no embedding, no match, or its identity has no assigned source). The
        numpy ``matcher`` does the routing (GEMV); see ``_CatalogMatcher``."""
        if matcher is None:
            return None
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None  # standalone/detection-only detector → can't match
        path = matcher.source_for(embedding)
        if path is None:
            return None
        return self._mapped_sources.get(path)

    def process(self, frame: Frame, ctx: Any = None) -> Frame:
        # Snapshot the backend handles into locals — release() (from a live
        # set_chain/reconfigure) can null self._* concurrently, and the
        # executor's _wait_for_inflight is bounded (5s), so a long in-flight
        # frame can outlive the wait. Holding local refs keeps the backend alive
        # for THIS call regardless. (self._params / _detection_sink aren't nulled
        # by release(), so they're safe to read off self.)
        analyser = self._analyser
        swapper = self._swapper
        source_face = self._source_face
        masker = self._masker
        landmarker = self._landmarker
        if analyser is None or swapper is None or source_face is None:
            raise RuntimeError("FaceSwapper.process called before setup()")
        # Face-mapping's detection-free runtime: when geometry is loaded for this
        # frame, rebuild faces from it (NO detection). Otherwise detect as usual.
        geometry = self._geometry
        geom_faces = self._geometry_faces(geometry, ctx, frame)
        if geom_faces is not None:
            faces = geom_faces
            already_refined = geometry.refined  # type: ignore[union-attr]
            if _GEOM_TRACE:
                self._trace_geometry(faces, self._face_map, ctx)
        elif self._face_map_is_routable() and not analyser.provides_embeddings():
            # Per-identity routing matches each TARGET face's embedding to the
            # catalog, so faces MUST carry one. The full buffalo_l pack provides
            # it; a standalone detector / detection-only `analyse` does NOT →
            # every face would route to nothing. Force det+rec for embeddings.
            faces = analyser.analyse_det_rec(frame)
            already_refined = False
        else:
            faces = analyser.analyse(frame)
            already_refined = False
        # Landmark refinement: replace each face's 5 keypoints with the 5
        # derived from 2dfan4's 68 landmarks (better alignment on tilted
        # faces), and keep the 68 for the landmark-68 roll source. Best-effort
        # per face — a low score or failure leaves the detector keypoints.
        # Skipped when the geometry already baked refined keypoints (self-heal:
        # a raw-geometry frame still refines here if the setting is now on).
        lm68_by_face: dict[int, Any] = {}
        if landmarker is not None and not already_refined:
            for face in faces:
                try:
                    lm68, score = landmarker.detect_68(frame, face.bbox)
                except Exception:  # noqa: BLE001 — never break the swap
                    continue
                if score < self._params.landmark_min_score:
                    continue
                lm68_by_face[id(face)] = lm68
                if self._params.landmark_refine:
                    try:
                        face.kps = landmark_68_to_5(lm68)
                    except Exception:  # noqa: BLE001
                        pass
        if ctx is not None:
            # Publish the PRE-filter list (already refined): the enhancer
            # restores every detected face today (not just swapped ones), so
            # downstream consumers see exactly what a re-detection would have
            # given them — now with the refined keypoints.
            ctx.faces = list(faces)
        # Publish every detected face (before the sex filter) so the debug
        # overlay shows exactly what the detector saw — including faces that
        # won't be swapped. Best-effort; the overlay must never affect the swap.
        if self._detection_sink is not None:
            try:
                # Tag with the frame these faces were detected on so the GUI can
                # reject a click against a stale (advanced-past) snapshot.
                self._detection_sink.publish(
                    faces, frame.shape[1], frame.shape[0],
                    getattr(ctx, "frame_index", None),
                )
            except Exception:
                pass
        # Face mapping (snapshot for a consistent view across this frame): when
        # active it ROUTES each face to a per-identity source and supersedes the
        # sex / many-faces selectors (the map decides what swaps with what).
        face_map = self._face_map
        matcher = self._matcher  # snapshot alongside the map for routing
        multi = (
            face_map is not None
            and face_map.is_active()
            and self._supports_multi_source
        )
        target_sex = self._resolved_target_sex(source_face)
        # The gender filter needs insightface's .sex, which only the buffalo_l
        # pack provides — skip it (swap all) when a detection-only detector is
        # active so M/F doesn't silently drop every face.
        gender_filter = analyser.provides_gender()
        result = frame
        swapped_faces: list[Any] = []
        for face in faces:
            if multi:
                face_source = self._mapped_source_face(face, matcher)
                if face_source is None:
                    continue  # no embedding / no match / identity unassigned
            else:
                if gender_filter and not _face_matches(face, target_sex):
                    continue
                face_source = source_face
            before = result
            result = self._swap_one(
                before, face, swapper, face_source, analyser,
                lm68_by_face.get(id(face)),
            )
            if self._params.occlusion_mask and masker is not None:
                # Revert non-facial pixels (hair/glasses/boundary) to the
                # pre-swap frame. Torch parsers aren't thread-safe → serialize
                # across workers; the ONNX parsers (shared ORT session) run
                # lock-free so N workers can mask in parallel.
                if getattr(masker, "thread_safe", False):
                    result = apply_occlusion(before, result, face, masker)
                else:
                    with self._mask_lock:
                        result = apply_occlusion(before, result, face, masker)
            swapped_faces.append(face)
            if not multi and not self._params.many_faces:
                break
        self._maybe_publish_crops(frame, result, swapped_faces)
        return result

    def _maybe_publish_crops(
        self, original: Frame, result: Frame, swapped_faces: list[Any]
    ) -> None:
        """Publish (bbox, original-crop, swapped-crop) thumbnails for the
        comparison overlay — only when the sink asks for them (zero cost
        otherwise). Best-effort; never affects the swap."""
        sink = self._detection_sink
        if sink is None:
            return
        try:
            if not sink.wants_crops():
                return
            pairs = []
            for face in swapped_faces:
                orig = _crop_thumb(original, face.bbox)
                swap = _crop_thumb(result, face.bbox)
                if orig is not None and swap is not None:
                    bbox = tuple(float(v) for v in face.bbox[:4])
                    pairs.append((bbox, orig, swap))
            sink.publish_crops(pairs, original.shape[1], original.shape[0])
        except Exception:
            pass

    def _swap_one(
        self,
        result: Frame,
        face: Any,
        swapper: Any,
        source_face: Any,
        analyser: FaceAnalyser,
        landmark_68: Any = None,
    ) -> Frame:
        """Swap a single face — uprighting it first when rotation
        compensation is on and the face is tilted past the threshold,
        otherwise a plain in-place swap. Backend handles are passed in (snapshot
        from process()) so a concurrent release() can't null them mid-swap.
        ``landmark_68`` (when the landmarker ran) drives the landmark-68 roll."""
        if self._params.rotation_compensation:
            roll = compute_roll(
                face, self._params.rotation_angle_source, landmark_68
            )
            if abs(roll) >= self._params.rotation_threshold_deg:
                return swap_with_uprighting(
                    result,
                    face,
                    source_face,
                    swapper,
                    analyser,
                    angle_deg=roll,
                    redetect=self._params.rotation_redetect,
                )
        return swapper.get(result, face, source_face, paste_back=True)

    def _resolved_target_sex(self, source_face: Any) -> TargetSex:
        """Resolve AS_SOURCE to the source face's actual sex. BOTH/M/F
        pass through unchanged. Falls back to BOTH if the source face
        has no sex attribute (older insightface model) so we never
        accidentally skip every face."""
        ts = self._params.target_sex
        if ts is not TargetSex.AS_SOURCE:
            return ts
        source_sex = getattr(source_face, "sex", None)
        if source_sex == "M":
            return TargetSex.MALE
        if source_sex == "F":
            return TargetSex.FEMALE
        return TargetSex.BOTH

    def on_seek(self) -> None:
        """Drop the interval-based face-detection cache so the first frame after
        a seek re-detects at the new position. Without this, a detection cached
        at the old position (when detection_interval > 1) is reused, swapping at
        the wrong place and leaving the new face unswapped until the next
        scheduled re-detection."""
        analyser = self._analyser
        if analyser is not None:
            analyser.reset_cache()

    def cache_identity(self) -> str:
        """Output-affecting params, for the realtime cache key."""
        return self._params.model_dump_json()

    def release(self) -> None:
        # Generic ONNX backends (ghost/simswap/uniface) own a cached session;
        # ask them to evict it so disabling/switching the swapper frees VRAM.
        # The insightface backend (inswapper/reswapper) stays resident in
        # model_cache by design (loaded once, reused across session rebuilds so
        # source/target changes don't stack inswapper sessions in VRAM); we just
        # drop our local ref here.
        if isinstance(self._swapper, GenericOnnxSwapper):
            self._swapper.release()
        # The occlusion masker holds a torch model on CUDA — release it (free its
        # VRAM) rather than just dropping the ref, like the swap backend above.
        if self._masker is not None:
            self._masker.release()
        # Landmarker owns a shared (refcounted) ONNX session — evict our hold.
        if self._landmarker is not None:
            self._landmarker.release()
        self._analyser = None
        self._swapper = None
        self._source_face = None
        self._masker = None
        self._landmarker = None
        self._mapped_sources = {}
