from typing import Protocol, runtime_checkable

from sinner2.config.target import Target
from sinner2.io.cv2_unicode import imread_unicode
from sinner2.types import Frame, FrameIndex


@runtime_checkable
class TargetReader(Protocol):
    """Source of frames from a Target.

    Implementations: ImageTargetReader (single-image as 1-frame stream) and
    a future VideoTargetReader (ffmpeg subprocess). The executor talks only
    to this protocol — single vs video is invisible above this layer.
    """

    @property
    def fps(self) -> float: ...

    @property
    def frame_count(self) -> int: ...

    def read(self, index: FrameIndex) -> Frame | None: ...

    def release(self) -> None: ...


class ImageTargetReader:
    """Reads a single image file, presented as a 1-frame stream.

    fps=1, frame_count=1; the image is decoded lazily on first read and
    cached. Any non-zero index returns None (out of range).
    """

    def __init__(self, target: Target) -> None:
        self._target = target
        self._frame: Frame | None = None

    @property
    def fps(self) -> float:
        return 1.0

    @property
    def frame_count(self) -> int:
        return 1

    def read(self, index: FrameIndex) -> Frame | None:
        if index != 0:
            return None
        if self._frame is None:
            img = imread_unicode(self._target.path)
            if img is None:
                raise OSError(f"cannot read image: {self._target.path}")
            self._frame = img
        return self._frame

    def release(self) -> None:
        self._frame = None
