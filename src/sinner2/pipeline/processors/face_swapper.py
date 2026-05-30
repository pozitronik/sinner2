from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.config.source import Source
from sinner2.io.cv2_unicode import imread_unicode
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.model_cache import get_model_path, record_actual_providers
from sinner2.types import Frame


class TargetSex(str, Enum):
    """Which detected faces to swap based on insightface's sex
    classification. Single-letter values match sinner1's CLI tokens
    so settings files round-trip between versions."""

    BOTH = "B"          # Swap every detected face regardless of sex.
    MALE = "M"          # Only swap faces classified male.
    FEMALE = "F"        # Only swap faces classified female.
    AS_SOURCE = "I"     # Match the source face's sex ("as input").


class FaceSwapperParams(SinnerBaseModel):
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


_MODEL_FILE = "inswapper_128.onnx"


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
    ) -> None:
        self._source = source
        self._params = params or FaceSwapperParams()
        # ONNX providers from the swapper's OnnxExecution profile; None falls
        # back to the platform-default EP order (CUDA then CPU).
        self._providers = list(providers) if providers else None
        self._analyser: FaceAnalyser | None = None
        self._swapper: Any = None
        self._source_face: Any = None

    def setup(self) -> None:
        providers = self._providers or list(DEFAULT_ONNX_PROVIDERS)
        self._analyser = FaceAnalyser(
            detection_interval=self._params.detection_interval,
            providers=providers,
        )
        self._swapper = _load_inswapper(get_model_path(_MODEL_FILE), providers)
        source_img = imread_unicode(self._source.path)
        if source_img is None:
            raise ValueError(f"cannot read source image: {self._source.path}")
        faces = self._analyser.analyse_uncached(source_img)
        if not faces:
            raise ValueError(f"no face detected in source: {self._source.path}")
        self._source_face = faces[0]

    def process(self, frame: Frame) -> Frame:
        if self._analyser is None or self._swapper is None or self._source_face is None:
            raise RuntimeError("FaceSwapper.process called before setup()")
        target_sex = self._resolved_target_sex()
        result = frame
        for face in self._analyser.analyse(frame):
            if not _face_matches(face, target_sex):
                continue
            result = self._swapper.get(result, face, self._source_face, paste_back=True)
            if not self._params.many_faces:
                break
        return result

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
        self._analyser = None
        self._swapper = None
        self._source_face = None
