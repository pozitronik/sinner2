from typing import Protocol, runtime_checkable

from sinner2.config.target import Target
from sinner2.io.cv2_unicode import imread_unicode
from sinner2.io.frame_resize import resize_frame, scaled_dims
from sinner2.types import Frame, FrameIndex


@runtime_checkable
class TargetReader(Protocol):
    """Source of frames from a Target.

    Implementations: ImageTargetReader (single-image as 1-frame stream) and
    the video readers (ffmpeg subprocess / cv2 capture). The executor talks
    only to this protocol — single vs video is invisible above this layer.

    width/height are the dimensions of the frames actually produced (after
    any processing-scale downscale); native_width/native_height are the
    source's true dimensions, exposed so the GUI can show the resulting size
    for any scale without rebuilding the reader.
    """

    @property
    def fps(self) -> float: ...

    @property
    def frame_count(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def native_width(self) -> int: ...

    @property
    def native_height(self) -> int: ...

    def read(self, index: FrameIndex) -> Frame | None: ...

    def release(self) -> None: ...


class ImageTargetReader:
    """Reads a single image file, presented as a 1-frame stream.

    fps=1, frame_count=1; the image is decoded in __init__ (so dimensions are
    known and a bad path fails at construction, like the video readers probe)
    and cached, downscaled by `scale` if < 1.0. Any non-zero index returns
    None (out of range).
    """

    def __init__(self, target: Target, scale: float = 1.0) -> None:
        self._target = target
        img = imread_unicode(target.path)
        if img is None:
            raise OSError(f"cannot read image: {target.path}")
        self._native_height, self._native_width = img.shape[0], img.shape[1]
        self._width, self._height = scaled_dims(
            self._native_width, self._native_height, scale
        )
        self._frame: Frame | None = resize_frame(img, self._width, self._height)

    @property
    def fps(self) -> float:
        return 1.0

    @property
    def frame_count(self) -> int:
        return 1

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def native_width(self) -> int:
        return self._native_width

    @property
    def native_height(self) -> int:
        return self._native_height

    def read(self, index: FrameIndex) -> Frame | None:
        if index != 0:
            return None
        # Re-decode if a prior release() cleared the cache.
        if self._frame is None:
            img = imread_unicode(self._target.path)
            if img is None:
                raise OSError(f"cannot read image: {self._target.path}")
            self._frame = resize_frame(img, self._width, self._height)
        return self._frame

    def release(self) -> None:
        self._frame = None
