from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2

from sinner2.types import Frame, FrameIndex


@runtime_checkable
class FrameStore(Protocol):
    """Canonical persistent storage for processed frames.

    Treated as the source of truth — the cache is a hot copy. Implementations
    must be safe for concurrent reads from different indices; writes from
    different indices may proceed in parallel but writes to the same index
    are not required to be atomic.
    """

    def write(self, index: FrameIndex, frame: Frame) -> None: ...
    def read(self, index: FrameIndex) -> Frame | None: ...
    def has(self, index: FrameIndex) -> bool: ...
    def clear_from(self, index: FrameIndex) -> None: ...


class DiskFrameStore:
    """FrameStore backed by per-frame image files in a directory.

    Filenames are 8-digit zero-padded 0-based frame indices with an extension
    chosen at construction (PNG default). 8 digits supports up to 100M frames
    (~3.8 days at 30 fps) — plenty for any realistic target.
    """

    _PAD_WIDTH = 8

    def __init__(self, directory: Path, extension: str = "png") -> None:
        if extension.startswith("."):
            extension = extension[1:]
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ext = extension

    def _path(self, index: FrameIndex) -> Path:
        return self._dir / f"{index:0{self._PAD_WIDTH}d}.{self._ext}"

    def write(self, index: FrameIndex, frame: Frame) -> None:
        if not cv2.imwrite(str(self._path(index)), frame):
            raise OSError(f"cv2.imwrite failed for frame {index} at {self._path(index)}")

    def read(self, index: FrameIndex) -> Frame | None:
        path = self._path(index)
        if not path.is_file():
            return None
        return cv2.imread(str(path))

    def has(self, index: FrameIndex) -> bool:
        return self._path(index).is_file()

    def clear_from(self, index: FrameIndex) -> None:
        for f in self._dir.glob(f"*.{self._ext}"):
            try:
                if int(f.stem) >= index:
                    f.unlink()
            except (ValueError, OSError):
                continue
