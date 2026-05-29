import threading
from collections import OrderedDict
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
    def memory_used_bytes(self) -> int: ...


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

    def put(self, index: FrameIndex, frame: Frame) -> None:
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

    def memory_used_bytes(self) -> int:
        with self._lock:
            return self._total_bytes
