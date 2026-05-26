import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from sinner2.pipeline.buffer.cache import FrameCache
from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.buffer.store import FrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.types import Frame, FrameIndex

_RECENT_INDICES_CAP = 1024


class FrameBuffer:
    """The single seam the executor uses for frame I/O.

    Composes a FrameStore (canonical persistence), a FrameCache (hot
    in-memory copies), and a Timeline (wall-clock → frame index). Workers
    call put(); playback calls get_at_current_time(). Read path is cache,
    then store on miss, with backfill into cache.

    Disk writes are submitted asynchronously to an injected ThreadPoolExecutor
    so workers don't block on filesystem I/O. The executor's lifecycle is
    owned by the caller (typically RealtimeExecutor).
    """

    def __init__(
        self,
        store: FrameStore,
        cache: FrameCache,
        timeline: Timeline,
        write_executor: ThreadPoolExecutor,
    ) -> None:
        self._store = store
        self._cache = cache
        self._timeline = timeline
        self._write_executor = write_executor
        self._lock = threading.RLock()
        self._last_written_index: FrameIndex | None = None
        self._last_displayed_index: FrameIndex | None = None
        self._recent_indices: deque[FrameIndex] = deque(maxlen=_RECENT_INDICES_CAP)
        self._hits = 0
        self._misses = 0
        self._current_frame_miss = 0

    def put(self, index: FrameIndex, frame: Frame) -> None:
        self._cache.put(index, frame)
        self._write_executor.submit(self._store.write, index, frame)
        with self._lock:
            if self._last_written_index is None or index > self._last_written_index:
                self._last_written_index = index
            self._recent_indices.append(index)

    def get(self, index: FrameIndex) -> Frame | None:
        frame = self._cache.get(index)
        if frame is not None:
            with self._lock:
                self._hits += 1
            return frame
        with self._lock:
            self._misses += 1
        frame = self._store.read(index)
        if frame is not None:
            self._cache.put(index, frame)
        return frame

    def get_at_current_time(self) -> tuple[FrameIndex, Frame | None]:
        target = self._timeline.current_frame()
        frame = self.get(target)
        with self._lock:
            if frame is None:
                self._current_frame_miss += 1
            else:
                self._last_displayed_index = target
        return target, frame

    @property
    def last_written_index(self) -> FrameIndex | None:
        with self._lock:
            return self._last_written_index

    def latest_index_at_or_below(self, target: FrameIndex) -> FrameIndex | None:
        """Find the highest recently-written frame index ≤ target.

        Used by the playback fallback so the display never jumps to a frame
        ahead of the timeline — important after a backward seek, where the
        most-recently-written frame is in the seeked-past future and would
        otherwise stick on screen.

        Backed by a bounded deque of recent put() indices; for windows
        larger than the deque cap (1024), older frames are forgotten and
        won't be candidates even if still in the store.
        """
        with self._lock:
            candidates = [i for i in self._recent_indices if i <= target]
        return max(candidates) if candidates else None

    def invalidate_from(self, index: FrameIndex) -> None:
        self._cache.evict_from(index)
        self._store.clear_from(index)
        with self._lock:
            if self._last_written_index is not None and self._last_written_index >= index:
                self._last_written_index = (index - 1) if index > 0 else None

    def metrics(self) -> BufferMetrics:
        with self._lock:
            last_displayed = self._last_displayed_index
            last_written = self._last_written_index
            current = self._timeline.current_frame()

            if last_displayed is None:
                frame_lag = max(0, current)
            else:
                frame_lag = max(0, current - last_displayed)

            if last_displayed is None or last_written is None:
                display_frame_lag = 0
            else:
                display_frame_lag = max(0, last_written - last_displayed)

            frame_time_s = 1.0 / self._timeline.fps
            total_reads = self._hits + self._misses
            ratio = (self._hits / total_reads) if total_reads > 0 else 0.0

            return BufferMetrics(
                frame_lag=frame_lag,
                time_lag_s=frame_lag * frame_time_s,
                display_frame_lag=display_frame_lag,
                display_time_lag_s=display_frame_lag * frame_time_s,
                current_frame_miss=self._current_frame_miss,
                memory_used_bytes=self._cache.memory_used_bytes(),
                cache_hit_ratio=ratio,
            )
