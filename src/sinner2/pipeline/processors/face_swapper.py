import threading
from pathlib import Path
from typing import Any, cast

import cv2
from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.config.source import Source
from sinner2.io.cv2_unicode import imread_unicode
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.model_cache import get_model_path, record_actual_providers
from sinner2.pipeline.processors.face_swapper_types import (
    RotationAngleSource,
    TargetSex,
)
from sinner2.pipeline.processors.occlusion import (
    FaceParser,
    OcclusionMasker,
    apply_occlusion,
)
from sinner2.pipeline.processors.rotation_compensation import (
    compute_roll,
    swap_with_uprighting,
)
from sinner2.pipeline.processors.swapper_models import (
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
    many_faces: bool = Field(
        default=True, description="Swap all detected faces (otherwise first only)"
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
        description="Measure roll from eye keypoints or the 3D pose estimate",
    )
    # ---- Occlusion-aware masking ----
    # Mask the swap to the facial-skin region (BiSeNet parse) so hair, glasses,
    # hats, and the neck/boundary keep the original. Output-affecting.
    occlusion_mask: bool = Field(
        default=False, description="Mask the swap to the real face region"
    )
    occlusion_parser: FaceParser = Field(
        default=FaceParser.BISENET,
        description="Face parser for the mask: bisenet (accurate) or parsenet (fast)",
    )


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
    OnnxExecution profile). After load, record what ORT actually wired up —
    `get_available_providers()` advertises EPs whose plugin DLL loads, but the
    EP can still fail at session construction (the TensorRT EP DLL loads even
    when nvinfer is missing; ORT then silently falls back). The recorded
    actual list drives the status-bar truth indicator."""
    from insightface.model_zoo import get_model

    model = get_model(str(path), providers=list(providers))
    # insightface wraps the ORT session in a Model object; the session
    # attribute name is stable for inswapper.
    session = getattr(model, "session", None)
    if session is not None:
        try:
            record_actual_providers(session.get_providers())
        except Exception:
            pass
    return model


class FaceSwapper:
    name = "FaceSwapper"
    thread_safe = True  # one ORT session, called concurrently by N workers

    def __init__(
        self,
        source: Source,
        params: FaceSwapperParams | None = None,
        providers: list[str] | None = None,
        detection_sink: Any = None,
    ) -> None:
        self._source = source
        self._params = params or FaceSwapperParams()
        # ONNX providers from the swapper's OnnxExecution profile; None falls
        # back to the platform-default EP order (CUDA then CPU).
        self._providers = list(providers) if providers else None
        # Optional sink for the debug overlay: receives the PRE-swap detections
        # (duck-typed `.publish(faces, w, h)`); None outside the realtime GUI.
        self._detection_sink = detection_sink
        self._analyser: FaceAnalyser | None = None
        self._swapper: Any = None
        self._source_face: Any = None
        # Occlusion masker (torch BiSeNet) — built only when enabled. Not
        # thread-safe, so the shared swapper serializes its calls.
        self._masker: OcclusionMasker | None = None
        self._mask_lock = threading.Lock()

    def setup(self) -> None:
        providers = self._providers or list(DEFAULT_ONNX_PROVIDERS)
        self._analyser = FaceAnalyser(
            detection_interval=self._params.detection_interval,
            providers=providers,
        )
        spec = get_spec(self._params.model)
        # insightface-compatible models (inswapper / reswapper) load through the
        # INSwapper wrapper; the rest use the facefusion-style generic backend.
        # Both expose .get(img, target, source, paste_back=True), so the swap
        # call sites (plain + rotation) don't branch.
        backend: Any
        if spec.insightface:
            backend = _load_inswapper(get_model_path(spec.model_file), providers)
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
        if self._params.occlusion_mask:
            self._masker = OcclusionMasker(parser=self._params.occlusion_parser)
            self._masker.setup()

    def process(self, frame: Frame) -> Frame:
        if self._analyser is None or self._swapper is None or self._source_face is None:
            raise RuntimeError("FaceSwapper.process called before setup()")
        faces = self._analyser.analyse(frame)
        # Publish every detected face (before the sex filter) so the debug
        # overlay shows exactly what the detector saw — including faces that
        # won't be swapped. Best-effort; the overlay must never affect the swap.
        if self._detection_sink is not None:
            try:
                self._detection_sink.publish(faces, frame.shape[1], frame.shape[0])
            except Exception:
                pass
        target_sex = self._resolved_target_sex()
        result = frame
        swapped_faces: list[Any] = []
        for face in faces:
            if not _face_matches(face, target_sex):
                continue
            before = result
            result = self._swap_one(before, face)
            if self._params.occlusion_mask and self._masker is not None:
                # Revert non-facial pixels (hair/glasses/boundary) to the
                # pre-swap frame. Torch parser → serialize across workers.
                with self._mask_lock:
                    result = apply_occlusion(before, result, face, self._masker)
            swapped_faces.append(face)
            if not self._params.many_faces:
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

    def _swap_one(self, result: Frame, face: Any) -> Frame:
        """Swap a single face — uprighting it first when rotation
        compensation is on and the face is tilted past the threshold,
        otherwise a plain in-place swap."""
        if self._params.rotation_compensation:
            roll = compute_roll(face, self._params.rotation_angle_source)
            if abs(roll) >= self._params.rotation_threshold_deg:
                return swap_with_uprighting(
                    result,
                    face,
                    self._source_face,
                    self._swapper,
                    self._analyser,
                    angle_deg=roll,
                    redetect=self._params.rotation_redetect,
                )
        return self._swapper.get(result, face, self._source_face, paste_back=True)

    def _resolved_target_sex(self) -> TargetSex:
        """Resolve AS_SOURCE to the source face's actual sex. BOTH/M/F
        pass through unchanged. Falls back to BOTH if the source face
        has no sex attribute (older insightface model) so we never
        accidentally skip every face."""
        ts = self._params.target_sex
        if ts is not TargetSex.AS_SOURCE:
            return ts
        source_sex = getattr(self._source_face, "sex", None)
        if source_sex == "M":
            return TargetSex.MALE
        if source_sex == "F":
            return TargetSex.FEMALE
        return TargetSex.BOTH

    def release(self) -> None:
        # Generic ONNX backends (ghost/simswap/uniface) own a cached session;
        # ask them to evict it so disabling/switching the swapper frees VRAM.
        # The insightface backend (inswapper/reswapper) has no release().
        if isinstance(self._swapper, GenericOnnxSwapper):
            self._swapper.release()
        self._analyser = None
        self._swapper = None
        self._source_face = None
        self._masker = None
