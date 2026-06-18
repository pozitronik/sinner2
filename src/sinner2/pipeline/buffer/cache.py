import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sinner2.types import Frame, FrameIndex


@runtime_checkable
class FrameCache(Protocol):
    """In-memory frame cache with bounded byte budget."""

    def put(self, index: FrameIndex, frame: Frame) -> None: ...
    def get(self, index: FrameIndex) -> Frame | None: ...
    def evict_at(self, index: FrameIndex) -> None: ...
    def evict_before(self, index: FrameIndex) -> None: ...
    def evict_from(self, index: FrameIndex) -> None: ...
    def clear(self) -> None: ...
    def set_max_bytes(self, max_bytes: int) -> None: ...
    def memory_used_bytes(self) -> int: ...
    def set_evict_listener(
        self, listener: Callable[[FrameIndex], None] | None
    ) -> None: ...


class MemoryFrameCache:
    """Bounded LRU cache for frames.

    LRU rather than oldest-index because frames can arrive out of order from
    a multi-worker pool. `get` counts as a use — the playhead's working set
    naturally stays hot. Frames larger than the entire budget are silently
    skipped (the canonical copy lives in the store).
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0; got {max_bytes}")
        self._max_bytes = max_bytes
        self._frames: OrderedDict[FrameIndex, Frame] = OrderedDict()
        self._sizes: dict[FrameIndex, int] = {}
        self._total_bytes = 0
        self._lock = threading.RLock()
        # Fired (off the lock) for each frame dropped by MEMORY PRESSURE — the
        # LRU/budget pops below. NOT for the explicit evict_*/clear paths (those
        # are invalidation, which the buffer tracks separately). Lets the
        # visualiser flip a frame from in-memory to on-disk.
        self._on_evict: Callable[[FrameIndex], None] | None = None

    def set_evict_listener(
        self, listener: Callable[[FrameIndex], None] | None
    ) -> None:
        self._on_evict = listener

    def _notify_evicted(self, indices: list[FrameIndex]) -> None:
        cb = self._on_evict
        if cb is None:
            return
        for i in indices:
            cb(i)

    def put(self, index: FrameIndex, frame: Frame) -> None:
        evicted: list[FrameIndex] = []
        size = int(frame.nbytes)
        with self._lock:
            if size > self._max_bytes:
                return
            if index in self._frames:
                self._total_bytes -= self._sizes[index]
            self._frames[index] = frame
            self._frames.move_to_end(index)
            self._sizes[index] = size
            self._total_bytes += size
            while self._total_bytes > self._max_bytes:
                lru_index, _ = self._frames.popitem(last=False)
                self._total_bytes -= self._sizes[lru_index]
                del self._sizes[lru_index]
                evicted.append(lru_index)
        self._notify_evicted(evicted)

    def get(self, index: FrameIndex) -> Frame | None:
        with self._lock:
            frame = self._frames.get(index)
            if frame is not None:
                self._frames.move_to_end(index)
            return frame

    def evict_at(self, index: FrameIndex) -> None:
        with self._lock:
            if index not in self._frames:
                return
            self._total_bytes -= self._sizes[index]
            del self._frames[index]
            del self._sizes[index]

    def evict_before(self, index: FrameIndex) -> None:
        with self._lock:
            to_drop = [i for i in self._frames if i < index]
            for i in to_drop:
                self._total_bytes -= self._sizes[i]
                del self._frames[i]
                del self._sizes[i]

    def evict_from(self, index: FrameIndex) -> None:
        with self._lock:
            to_drop = [i for i in self._frames if i >= index]
            for i in to_drop:
                self._total_bytes -= self._sizes[i]
                del self._frames[i]
                del self._sizes[i]

    def clear(self) -> None:
        """Drop every cached frame. Used on a chain swap — the cache is keyed by
        frame index, not by the chain that produced the pixels, so once the chain
        changes every entry is stale."""
        with self._lock:
            self._frames.clear()
            self._sizes.clear()
            self._total_bytes = 0

    def set_max_bytes(self, max_bytes: int) -> None:
        """Resize the byte budget at runtime, evicting LRU frames down to fit.

        Lets the GUI's memory-cache size take effect on the LIVE session without
        rebuilding the buffer (the budget was previously fixed at construction)."""
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0; got {max_bytes}")
        evicted: list[FrameIndex] = []
        with self._lock:
            self._max_bytes = max_bytes
            while self._total_bytes > self._max_bytes:
                lru_index, _ = self._frames.popitem(last=False)
                self._total_bytes -= self._sizes[lru_index]
                del self._sizes[lru_index]
                evicted.append(lru_index)
        self._notify_evicted(evicted)

    def memory_used_bytes(self) -> int:
        with self._lock:
            return self._total_bytes
