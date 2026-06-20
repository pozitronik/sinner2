"""Occlusion-aware face masking for the swapper.

inswapper pastes a roughly-rectangular aligned region over the target, which
overwrites whatever crosses the face — hair on the forehead, glasses, a hat,
the neck/jaw boundary, background in the corners. This masks the swap to the
actual facial-skin region (a BiSeNet face-parse), so those keep the original.

The parser is facexlib's BiSeNet — already installed (a GFPGAN dependency) and
its weights download from a stable facexlib release through our confirmed
download flow (no silent fetch). The composite is the same shape as the
rotation pass: align → mask → warp back → blend. Best-effort: any failure
falls back to the plain swap, so enabling it can never break a render.

Note: handles hair/glasses/hat/neck/boundary well (those are distinct parse
classes); arbitrary occluders like a hand over the cheek (parsed as skin) are a
follow-up needing a dedicated occluder model.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from sinner2.pipeline.model_cache import get_models_dir, model_present
from sinner2.types import Frame


class FaceParser(str, Enum):
    """Which face-parser drives occlusion masking. The first two are torch
    (facexlib) models — NOT thread-safe, so the swapper serializes workers on a
    lock around them. The ONNX pair are facefusion's exports running on a
    shared thread-safe ORT session: no lock, workers parse in parallel."""

    BISENET = "bisenet"    # torch; accurate boundaries, heavier
    PARSENET = "parsenet"  # torch; lighter/faster (GFPGAN's default parser)
    BISENET_ONNX_34 = "bisenet_onnx_34"  # ONNX resnet-34; thread-safe
    BISENET_ONNX_18 = "bisenet_onnx_18"  # ONNX resnet-18; thread-safe, lighter


@dataclass(frozen=True)
class _ParserSpec:
    filename: str
    # Per-parser input normalization (they differ): bisenet expects ImageNet
    # stats; parsenet expects (0.5, 0.5, 0.5) → [-1, 1].
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


_PARSER_SPECS: dict[FaceParser, _ParserSpec] = {
    FaceParser.BISENET: _ParserSpec(
        "parsing_bisenet.pth", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ),
    FaceParser.PARSENET: _ParserSpec(
        "parsing_parsenet.pth", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    ),
}


# ONNX parser model files (facefusion exports), keyed separately from the
# torch _PARSER_SPECS — they have their own normalization, handled inline in
# OnnxParserMasker (contract verified against facefusion's face_masker.py).
_ONNX_PARSER_FILES: dict[FaceParser, str] = {
    FaceParser.BISENET_ONNX_34: "bisenet_resnet_34.onnx",
    FaceParser.BISENET_ONNX_18: "bisenet_resnet_18.onnx",
}


class OcclusionMaskMode(str, Enum):
    """What the occlusion mask is built from.

    REGION   — face-parser classes (skin/brows/eyes/nose/mouth). Handles hair,
               glasses, hats, boundaries — anything that is a PARSE CLASS — but
               cannot see arbitrary occluders (a hand over the cheek parses as
               skin). This is the original behaviour.
    OCCLUDER — a dedicated occluder-segmentation model (XSeg / depth) that
               segments the VISIBLE face surface, excluding any object in
               front of it: hands, microphones, food, hair strands.
    BOTH     — minimum of the two (strictest: a pixel must be facial-region
               AND unoccluded to take the swap).
    """

    REGION = "region"
    OCCLUDER = "occluder"
    BOTH = "both"


class OccluderModel(str, Enum):
    """Occluder-segmentation backend (OCCLUDER / BOTH modes). The XSeg trio
    are DeepFaceLab-lineage community models exported by facefusion — the
    de-facto standard for arbitrary-object face occlusion. XSEG_MANY runs all
    three and takes the minimum (strictest, 3x the inference cost). DEPTH is
    EXPERIMENTAL: monocular depth, masking anything meaningfully closer than
    the face plane — object-agnostic but with softer boundaries than XSeg."""

    XSEG_1 = "xseg_1"
    XSEG_2 = "xseg_2"
    XSEG_3 = "xseg_3"
    XSEG_MANY = "xseg_many"
    DEPTH = "depth"


_XSEG_FILES: dict[OccluderModel, list[str]] = {
    OccluderModel.XSEG_1: ["xseg_1.onnx"],
    OccluderModel.XSEG_2: ["xseg_2.onnx"],
    OccluderModel.XSEG_3: ["xseg_3.onnx"],
    OccluderModel.XSEG_MANY: ["xseg_1.onnx", "xseg_2.onnx", "xseg_3.onnx"],
}

_DEPTH_MODEL_FILE = "depth_anything_v2_small.onnx"


def occluder_model_files(model: OccluderModel) -> list[str]:
    """The weight file(s) an occluder choice needs (so the GUI can confirm +
    download before enabling it). XSEG_MANY needs all three."""
    if model is OccluderModel.DEPTH:
        return [_DEPTH_MODEL_FILE]
    return list(_XSEG_FILES[model])


def parser_model_file(parser: FaceParser) -> str:
    """The weights filename for a parser (so the GUI can confirm + download it
    before enabling occlusion)."""
    if parser in _ONNX_PARSER_FILES:
        return _ONNX_PARSER_FILES[parser]
    return _PARSER_SPECS[parser].filename


_ALIGN_SIZE = 512
_FEATHER_SIGMA = 5.0
# BiSeNet (face-parsing.PyTorch) class ids that ARE the face to swap: skin,
# brows, eyes, nose, mouth, lips. Everything else — hair(17), hat(18),
# glasses(6), ears(7-9), neck(14-15), cloth(16), background(0) — keeps the
# original, which is exactly the occlusion behaviour we want.
_FACE_CLASSES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 10, 11, 12, 13})
# Boolean LUT over the 19 BiSeNet classes (CelebAMask-HQ) so building the mask
# is an O(1)/pixel index (lut[classes]) instead of np.isin's general sort-based
# membership test per face. Byte-identical result.
_FACE_CLASS_LUT: np.ndarray = np.zeros(19, dtype=bool)
_FACE_CLASS_LUT[list(_FACE_CLASSES)] = True

# ArcFace 5-point template (112 space), used to align faces to _ALIGN_SIZE.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def _align_matrix(kps: np.ndarray) -> np.ndarray:
    template = _ARCFACE_DST * (_ALIGN_SIZE / 112.0)
    m, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, np.float32), template, method=cv2.LMEDS
    )
    return m


class OcclusionMasker:
    """facexlib face-parse → a facial-region mask. torch model (not thread-safe;
    the swapper serializes calls on a lock)."""

    thread_safe = False

    def __init__(
        self, parser: FaceParser = FaceParser.BISENET, device: str = "auto"
    ) -> None:
        self._spec = _PARSER_SPECS[parser]
        self._parser = parser
        self._device_str = device
        self._device: Any = None
        self._model: Any = None
        # Whether the resolved device is CUDA, so release() knows to hand the
        # parser's GPU memory back to the driver.
        self._device_is_cuda = False

    def setup(self) -> None:
        from facexlib.parsing import init_parsing_model

        from sinner2.config.execution import resolve_torch_device

        if not model_present(self._spec.filename):
            # The GUI ensures the model (with a download confirmation) before
            # enabling occlusion — never let facexlib fetch it silently.
            raise FileNotFoundError(
                f"face-parsing model not found: "
                f"{get_models_dir() / self._spec.filename}"
            )
        self._device = resolve_torch_device(self._device_str)
        self._device_is_cuda = self._device.type == "cuda"
        self._model = init_parsing_model(
            model_name=self._parser.value,
            device=self._device,
            model_rootpath=str(get_models_dir()),
        )
        self._model.eval()

    def release(self) -> None:
        """Drop the parser model and hand its VRAM back to the driver — torch's
        caching allocator keeps the freed blocks reserved otherwise, so a chain
        rebuild with occlusion on would stack a fresh parser's GPU memory on top
        of the old one's until process exit (mirrors FaceEnhancer.release)."""
        self._model = None
        if self._device_is_cuda:
            import torch

            torch.cuda.empty_cache()

    def face_mask(self, aligned_bgr: Frame) -> np.ndarray:
        """512×512 float mask (1 = facial region) for an aligned BGR crop."""
        import torch

        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(self._spec.mean, np.float32)
        std = np.array(self._spec.std, np.float32)
        chw = np.ascontiguousarray(
            ((rgb - mean) / std).transpose(2, 0, 1), dtype=np.float32
        )
        t = torch.from_numpy(chw)[None].to(self._device)
        with torch.no_grad():
            out = self._model(t)[0]
        classes = out.argmax(dim=1).squeeze().cpu().numpy()
        return _FACE_CLASS_LUT[classes].astype(np.float32)


class OnnxParserMasker:
    """facefusion's BiSeNet ONNX face parsers → the same facial-region mask as
    the torch parsers, on a SHARED thread-safe ORT session — the swapper skips
    its serialization lock for this masker, so N workers parse in parallel.

    Contract (verified against facefusion face_masker.py): 512×512 input, BGR
    flipped to RGB, /255 then ImageNet mean/std, NCHW; output logits over the
    face-parsing.PyTorch 19-class map (same ids as the torch parsers, so
    _FACE_CLASSES is shared)."""

    thread_safe = True

    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_file: str, providers: list[str] | None = None) -> None:
        from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS

        self._model_file = model_file
        self._providers = (
            list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
        )
        self._session: Any = None
        self._in_name = "input"
        self._out_name = "output"

    def setup(self) -> None:
        from sinner2.pipeline.model_cache import get_onnx_session_io

        self._session, self._in_name, self._out_name = get_onnx_session_io(
            self._model_file, providers=self._providers
        )

    def face_mask(self, aligned_bgr: Frame) -> np.ndarray:
        """512×512 float mask (1 = facial region) for an aligned BGR crop."""
        if self._session is None:
            raise RuntimeError("OnnxParserMasker.face_mask called before setup()")
        rgb = aligned_bgr[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.ascontiguousarray(
            ((rgb - self._MEAN) / self._STD).transpose(2, 0, 1)[None], np.float32
        )
        out = self._session.run([self._out_name], {self._in_name: chw})[0]
        if out.ndim == 4:
            out = out[0]
        classes = out.argmax(0)
        return _FACE_CLASS_LUT[classes].astype(np.float32)

    def release(self) -> None:
        from sinner2.pipeline.model_cache import release_onnx_session

        self._session = None
        release_onnx_session(self._model_file, self._providers)


class XsegOccluderMasker:
    """XSeg occluder segmentation → a visible-face mask (1 = unoccluded face,
    0 = occluder/background). Thread-safe (shared ORT sessions).

    Contract VERBATIM from facefusion's create_occlusion_mask: crop resized to
    the model size (256), NHWC, BGR as-is, float32/255; output [0][0], clipped,
    resized back; multiple models min-reduced; then the facefusion hardening
    blur ((GaussianBlur(σ5).clip(0.5,1) - 0.5) * 2) which anchors the boundary
    at probability 0.5 — kept inside this backend so the mask matches the look
    facefusion validated."""

    thread_safe = True
    _MODEL_SIZE = 256

    def __init__(
        self, model: OccluderModel = OccluderModel.XSEG_1,
        providers: list[str] | None = None,
    ) -> None:
        from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS

        self._files = list(_XSEG_FILES[model])
        self._providers = (
            list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
        )
        self._sessions: list[Any] = []
        self._io_names: list[tuple[str, str]] = []

    def setup(self) -> None:
        from sinner2.pipeline.model_cache import get_onnx_session

        self._sessions = [
            get_onnx_session(f, providers=self._providers) for f in self._files
        ]
        self._io_names = [
            (s.get_inputs()[0].name, s.get_outputs()[0].name)
            for s in self._sessions
        ]

    def face_mask(self, aligned_bgr: Frame) -> np.ndarray:
        """Visible-face float mask for an aligned BGR crop (same surface as the
        parser maskers, so the composite + combiner treat all maskers alike)."""
        if not self._sessions:
            raise RuntimeError("XsegOccluderMasker.face_mask called before setup()")
        size = aligned_bgr.shape[0]
        prep = cv2.resize(aligned_bgr, (self._MODEL_SIZE, self._MODEL_SIZE))
        blob = np.expand_dims(prep, axis=0).astype(np.float32) / 255.0
        masks = []
        for session, (in_name, out_name) in zip(self._sessions, self._io_names):
            out = session.run([out_name], {in_name: blob})[0][0]
            mask = np.clip(out, 0.0, 1.0).astype(np.float32)
            masks.append(cv2.resize(mask, (size, size)))
        combined = np.minimum.reduce(masks)
        return (
            cv2.GaussianBlur(combined.clip(0.0, 1.0), (0, 0), 5).clip(0.5, 1.0)
            - 0.5
        ) * 2.0

    def release(self) -> None:
        from sinner2.pipeline.model_cache import release_onnx_session

        self._sessions = []
        self._io_names = []
        for f in self._files:
            release_onnx_session(f, self._providers)


class DepthOccluderMasker:
    """EXPERIMENTAL depth-based occluder: estimates relative depth for the
    aligned crop and masks pixels meaningfully CLOSER than the face plane —
    object-agnostic (anything in front of the face counts) but with softer
    boundaries than XSeg on thin occluders like fingers.

    Depth-Anything V2 small (ONNX, HF onnx-community export). Contract from
    its preprocessor config: 518×518, RGB, /255 then ImageNet mean/std, NCHW;
    output relative INVERSE depth (HIGHER = CLOSER). The face's own depth is
    the median over the crop's central region (the aligned face fills the
    center by construction); pixels closer than that by >15% of the scene's
    depth spread (p95−p5, normalizing the unitless relative scale) are
    occluders. A near-flat depth map (no spread) yields no occlusion rather
    than thresholding noise."""

    thread_safe = True
    _SIZE = 518
    _MARGIN = 0.15
    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, providers: list[str] | None = None) -> None:
        from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS

        self._providers = (
            list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
        )
        self._session: Any = None
        self._in_name = "pixel_values"
        self._out_name = "predicted_depth"

    def setup(self) -> None:
        from sinner2.pipeline.model_cache import get_onnx_session_io

        self._session, self._in_name, self._out_name = get_onnx_session_io(
            _DEPTH_MODEL_FILE, providers=self._providers
        )

    def face_mask(self, aligned_bgr: Frame) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("DepthOccluderMasker.face_mask called before setup()")
        size = aligned_bgr.shape[0]
        prep = cv2.resize(aligned_bgr, (self._SIZE, self._SIZE))
        rgb = prep[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(
            ((rgb - self._MEAN) / self._STD).transpose(2, 0, 1)[None], np.float32
        )
        depth = self._session.run([self._out_name], {self._in_name: blob})[0]
        while depth.ndim > 2:
            depth = depth[0]
        c0, c1 = self._SIZE // 4, self._SIZE * 3 // 4
        face_depth = float(np.median(depth[c0:c1, c0:c1]))
        spread = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        if spread <= 1e-6:
            return np.ones((size, size), np.float32)  # flat scene → no occluders
        visible = (depth <= face_depth + self._MARGIN * spread).astype(np.float32)
        visible = cv2.resize(visible, (size, size))
        # Same boundary treatment as the XSeg backend (facefusion's hardening
        # blur) so the two occluder flavors compose identically downstream.
        return (
            cv2.GaussianBlur(visible.clip(0.0, 1.0), (0, 0), 5).clip(0.5, 1.0)
            - 0.5
        ) * 2.0

    def release(self) -> None:
        from sinner2.pipeline.model_cache import release_onnx_session

        self._session = None
        release_onnx_session(_DEPTH_MODEL_FILE, self._providers)


class CombinedMasker:
    """Minimum of several maskers' masks (BOTH mode: facial-region AND
    unoccluded). Thread-safe only if every part is."""

    def __init__(self, maskers: list[Any]) -> None:
        self._maskers = maskers
        self.thread_safe = all(
            getattr(m, "thread_safe", False) for m in maskers
        )

    def setup(self) -> None:
        for m in self._maskers:
            m.setup()

    def face_mask(self, aligned_bgr: Frame) -> np.ndarray:
        return np.minimum.reduce([m.face_mask(aligned_bgr) for m in self._maskers])

    def release(self) -> None:
        for m in self._maskers:
            m.release()


class CachingMasker:
    """Temporal mask cache (opt-in): skip the parser/occluder forward for a
    near-static face by reusing a recently-computed mask.

    CONTENT-ADDRESSED — the cache key is a coarse signature of the ALIGNED crop
    (an 8×8 luma thumbnail, quantized), so a barely-moved face (near-identical
    aligned crop) HITS the cache while real movement MISSES and recomputes. No
    frame index or face-identity tracking, so it's robust to the realtime
    executor's out-of-order parallel workers — two frames with the face in the
    same pose share the mask whichever order they land.

    Output-affecting: a face that moves WITHIN one signature bucket reuses a
    slightly-stale mask (the occlusion boundary lags a fraction), which is why
    it's opt-in. Thread-safe iff the wrapped masker is; the small LRU is
    lock-guarded and the forward runs OUTSIDE the lock, so the thread-safe ONNX
    maskers still parse in parallel on a miss."""

    # Signature = an 8×8 grayscale thumbnail of the aligned crop, quantized to
    # 16 luma buckets: tolerant of capture noise, sensitive to real movement.
    _THUMB = 8
    _LEVELS = 16

    def __init__(self, inner: Any, max_entries: int = 8) -> None:
        self._inner = inner
        self._max = max(1, max_entries)
        self._cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        # Mirror the wrapped masker so the swapper's lock decision is unchanged.
        self.thread_safe = bool(getattr(inner, "thread_safe", False))

    def setup(self) -> None:
        self._inner.setup()

    def _signature(self, aligned_bgr: Frame) -> bytes:
        thumb = cv2.resize(
            aligned_bgr, (self._THUMB, self._THUMB), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        return (gray // (256 // self._LEVELS)).astype(np.uint8).tobytes()

    def face_mask(self, aligned_bgr: Frame) -> np.ndarray:
        key = self._signature(aligned_bgr)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)  # LRU touch
                return cached
        # Miss: run the real forward OUTSIDE the lock so parallel misses overlap.
        mask = self._inner.face_mask(aligned_bgr)
        with self._lock:
            self._cache[key] = mask
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)  # evict the oldest
        return mask

    def release(self) -> None:
        with self._lock:
            self._cache.clear()
        self._inner.release()


def build_parser_masker(
    parser: FaceParser,
    device: str = "auto",
    providers: list[str] | None = None,
) -> Any:
    """The right masker for a parser choice: torch facexlib for bisenet /
    parsenet (per-instance, lock-serialized by the swapper), shared-session
    ONNX for the facefusion exports (thread-safe, no lock)."""
    if parser in _ONNX_PARSER_FILES:
        return OnnxParserMasker(_ONNX_PARSER_FILES[parser], providers=providers)
    return OcclusionMasker(parser=parser, device=device)


def build_occlusion_masker(
    mode: OcclusionMaskMode,
    parser: FaceParser,
    occluder: OccluderModel,
    device: str = "auto",
    providers: list[str] | None = None,
    cache: bool = False,
) -> Any:
    """Compose the masker for an occlusion-mode choice: the region parser, the
    occluder model, or the min-combination of both. ``cache`` wraps the result
    in a temporal mask cache (reuse a near-static face's mask across frames)."""
    maskers: list[Any] = []
    if mode in (OcclusionMaskMode.REGION, OcclusionMaskMode.BOTH):
        maskers.append(build_parser_masker(parser, device=device, providers=providers))
    if mode in (OcclusionMaskMode.OCCLUDER, OcclusionMaskMode.BOTH):
        if occluder is OccluderModel.DEPTH:
            maskers.append(DepthOccluderMasker(providers=providers))
        else:
            maskers.append(XsegOccluderMasker(occluder, providers=providers))
    masker = maskers[0] if len(maskers) == 1 else CombinedMasker(maskers)
    return CachingMasker(masker) if cache else masker


def apply_occlusion(
    before: Frame, swapped: Frame, face: Any, masker: Any
) -> Frame:
    """Keep only the facial-region pixels from `swapped`; the rest reverts to
    `before` (the pre-swap frame) — so occluders stay original. Best-effort:
    returns `swapped` unchanged on any error.

    The mask warp + float blend run only inside the bounding box of the warped
    aligned square (+2px interpolation bleed) — outside it the warped alpha is
    identically 0, so the result is `before` there by definition, and the full-
    frame float blend this replaces was ~60-80ms of CPU per face at FullHD
    (the same pattern paste_back fixed; see scripts/enhancer_bench.py)."""
    try:
        m = _align_matrix(face.kps)
        aligned = cv2.warpAffine(before, m, (_ALIGN_SIZE, _ALIGN_SIZE))
        mask = masker.face_mask(aligned)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=_FEATHER_SIGMA)
        m_inv = cv2.invertAffineTransform(m)
        h, w = before.shape[:2]
        corners = np.array(
            [[0, 0], [_ALIGN_SIZE, 0], [_ALIGN_SIZE, _ALIGN_SIZE], [0, _ALIGN_SIZE]],
            np.float32,
        ).reshape(1, 4, 2)
        warped = cv2.transform(corners, m_inv)[0]
        x0 = max(int(np.floor(float(warped[:, 0].min()))) - 2, 0)
        y0 = max(int(np.floor(float(warped[:, 1].min()))) - 2, 0)
        x1 = min(int(np.ceil(float(warped[:, 0].max()))) + 2, w)
        y1 = min(int(np.ceil(float(warped[:, 1].max()))) + 2, h)
        if x0 >= x1 or y0 >= y1:
            return before.copy()  # aligned square fully off-frame → all reverts
        m_roi = m_inv.copy()
        m_roi[0, 2] -= x0
        m_roi[1, 2] -= y0
        alpha = cv2.warpAffine(mask, m_roi, (x1 - x0, y1 - y0))[..., None]
        # Outside the warped square alpha is 0 → the blend yields `before`
        # everywhere outside the ROI, so start from a copy of `before`.
        out = before.copy()
        roi_sw = swapped[y0:y1, x0:x1].astype(np.float32)
        roi_bf = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (
            roi_sw * alpha + roi_bf * (1.0 - alpha)
        ).astype(np.uint8)
        return out
    except Exception:
        return swapped
