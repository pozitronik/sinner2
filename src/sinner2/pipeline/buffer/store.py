import shutil
import tempfile
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


class SessionFrameStore:
    """FrameStore that owns a fresh temp directory for the session.

    Solves the stale-leftovers footgun for realtime executors: each
    session gets a brand-new disk store, and close() removes the temp
    directory entirely. Use for realtime mode; batch jobs that want
    resumable shared frames should use plain DiskFrameStore.

    Composition wrapper around DiskFrameStore — no inheritance, no
    Liskov surprises. close() is idempotent and called from __del__
    as a safety net.
    """

    def __init__(self, prefix: str = "sinner2-session-", extension: str = "png") -> None:
        self._scratch_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self._inner = DiskFrameStore(self._scratch_dir / "frames", extension=extension)
        self._closed = False

    @property
    def scratch_dir(self) -> Path:
        return self._scratch_dir

    def write(self, index: FrameIndex, frame: Frame) -> None:
        self._inner.write(index, frame)

    def read(self, index: FrameIndex) -> Frame | None:
        return self._inner.read(index)

    def has(self, index: FrameIndex) -> bool:
        return self._inner.has(index)

    def clear_from(self, index: FrameIndex) -> None:
        self._inner.clear_from(index)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self._scratch_dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
