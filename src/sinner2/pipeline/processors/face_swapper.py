from pathlib import Path
from typing import Any

import cv2
from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.config.source import Source
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.model_cache import get_model_path
from sinner2.types import Frame


class FaceSwapperParams(SinnerBaseModel):
    detection_interval: int = Field(
        default=1, ge=1, description="Detect faces every Nth frame; >=1"
    )
    many_faces: bool = Field(
        default=True, description="Swap all detected faces (otherwise first only)"
    )


_MODEL_FILE = "inswapper_128.onnx"


def _load_inswapper(path: Path) -> Any:
    """Loader indirection so tests can stub the insightface call cheaply."""
    from insightface.model_zoo import get_model

    return get_model(str(path))


class FaceSwapper:
    name = "FaceSwapper"

    def __init__(self, source: Source, params: FaceSwapperParams | None = None) -> None:
        self._source = source
        self._params = params or FaceSwapperParams()
        self._analyser: FaceAnalyser | None = None
        self._swapper: Any = None
        self._source_face: Any = None

    def setup(self) -> None:
        self._analyser = FaceAnalyser(detection_interval=self._params.detection_interval)
        self._swapper = _load_inswapper(get_model_path(_MODEL_FILE))
        source_img = cv2.imread(str(self._source.path))
        if source_img is None:
            raise ValueError(f"cannot read source image: {self._source.path}")
        faces = self._analyser.analyse_uncached(source_img)
        if not faces:
            raise ValueError(f"no face detected in source: {self._source.path}")
        self._source_face = faces[0]

    def process(self, frame: Frame) -> Frame:
        if self._analyser is None or self._swapper is None or self._source_face is None:
            raise RuntimeError("FaceSwapper.process called before setup()")
        result = frame
        for face in self._analyser.analyse(frame):
            result = self._swapper.get(result, face, self._source_face, paste_back=True)
            if not self._params.many_faces:
                break
        return result

    def release(self) -> None:
        self._analyser = None
        self._swapper = None
        self._source_face = None
