import shutil
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from sinner2.pipeline.image_writer import ImageWriter, PNGImageWriter
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
    def cached_indices(self) -> list[FrameIndex]: ...
    def clear_from(self, index: FrameIndex) -> None: ...
    def close(self) -> None:
        """Release resources the store owns (temp dirs, handles). Idempotent;
        a store writing to a caller-owned directory implements it as a no-op."""


class DiskFrameStore:
    """FrameStore backed by per-frame image files in a directory.

    Filenames are 8-digit zero-padded 0-based frame indices with the
    extension chosen by the injected ImageWriter. 8 digits supports up to
    100M frames (~3.8 days at 30 fps) — plenty for any realistic target.
    The writer encapsulates format + quality so swapping image formats is
    a constructor change, not store-internals surgery.
    """

    _PAD_WIDTH = 8

    def __init__(self, directory: Path, writer: ImageWriter | None = None) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._writer = writer if writer is not None else PNGImageWriter()

    def _path(self, index: FrameIndex) -> Path:
        return self._dir / f"{index:0{self._PAD_WIDTH}d}.{self._writer.extension}"

    def write(self, index: FrameIndex, frame: Frame) -> None:
        self._writer.write(self._path(index), frame)

    def read(self, index: FrameIndex) -> Frame | None:
        return self._writer.read(self._path(index))

    def has(self, index: FrameIndex) -> bool:
        return self._path(index).is_file()

    def cached_indices(self) -> list[FrameIndex]:
        """Frame indices currently present on disk (one glob, no decode). Lets
        the visualiser show already-cached frames the moment a session opens."""
        out: list[FrameIndex] = []
        for f in self._dir.glob(f"*.{self._writer.extension}"):
            try:
                out.append(int(f.stem))
            except (ValueError, OSError):
                continue
        return out

    def clear_from(self, index: FrameIndex) -> None:
        if index <= 0:
            # Invalidate-all: one rmtree beats globbing + unlinking every file
            # (O(files) syscalls — a stutter on a large cache, worse under
            # Windows AV). Recreate the directory empty so the store stays usable.
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir.mkdir(parents=True, exist_ok=True)
            return
        # index > 0: glob (not a directed walk) so non-contiguous gaps from
        # out-of-order multi-worker writes don't strand higher frames.
        for f in self._dir.glob(f"*.{self._writer.extension}"):
            try:
                if int(f.stem) >= index:
                    f.unlink()
            except (ValueError, OSError):
                continue

    def close(self) -> None:
        # Writes to a caller-owned directory — nothing to clean up (like
        # PersistentFrameStore). Present so DiskFrameStore satisfies the
        # FrameStore lifecycle contract.
        pass


class PersistentFrameStore:
    """FrameStore backed by an explicit caller-provided directory.

    Does NOT clean up on close. Use this for cross-session frame caches
    where the user expects processed frames to survive between runs.
    Combine with a cache-key directory layout (per source/target/chain
    combination) so different setups don't write into each other.
    """

    def __init__(
        self, directory: Path, writer: ImageWriter | None = None
    ) -> None:
        self._dir = Path(directory)
        self._inner = DiskFrameStore(self._dir, writer=writer)

    @property
    def directory(self) -> Path:
        return self._dir

    def write(self, index: FrameIndex, frame: Frame) -> None:
        self._inner.write(index, frame)

    def read(self, index: FrameIndex) -> Frame | None:
        return self._inner.read(index)

    def has(self, index: FrameIndex) -> bool:
        return self._inner.has(index)

    def cached_indices(self) -> list[FrameIndex]:
        return self._inner.cached_indices()

    def clear_from(self, index: FrameIndex) -> None:
        self._inner.clear_from(index)

    def close(self) -> None:
        # Persistent cache: nothing to clean up. Caller manages dir lifetime.
        pass


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

    def __init__(
        self,
        prefix: str = "sinner2-session-",
        writer: ImageWriter | None = None,
    ) -> None:
        self._scratch_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self._inner = DiskFrameStore(self._scratch_dir / "frames", writer=writer)
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

    def cached_indices(self) -> list[FrameIndex]:
        return self._inner.cached_indices()

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
