import threading
from collections import deque

from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor
from sinner2.pipeline.buffer.cache import FrameCache
from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.buffer.store import FrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.types import Frame, FrameIndex

_RECENT_INDICES_CAP = 1024


class FrameBuffer:
    """The single seam the executor uses for frame I/O.

    Composes a FrameStore (canonical persistence), a FrameCache (hot
    in-memory copies), a Timeline (wall-clock → frame index), and a
    BoundedWriteExecutor (background disk writes with backpressure).
    Workers call put(); playback calls get_at_current_time(). Read path
    is cache, then store on miss (skipped when cache_mode == OFF), with
    backfill into cache.

    Disk writes go through a BoundedWriteExecutor so the in-flight write
    queue can't grow without limit — when the cap is hit, the write is
    silently dropped and the frame stays in the memory cache until LRU
    evicts it. The drop count is surfaced in metrics so the user can see
    when the disk is the bottleneck.

    cache_mode runtime semantics:
      WRITE_READ: writes submitted, cache misses fall back to store.read
      READ_ONLY:  no writes submitted; cache misses still read from store
      OFF:        no writes, no reads — memory only
    """

    def __init__(
        self,
        store: FrameStore,
        cache: FrameCache,
        timeline: Timeline,
        write_executor: BoundedWriteExecutor,
        cache_mode: CacheMode = CacheMode.WRITE_READ,
    ) -> None:
        self._store = store
        self._cache = cache
        self._timeline = timeline
        self._write_executor = write_executor
        self._cache_mode = cache_mode
        self._lock = threading.RLock()
        self._last_written_index: FrameIndex | None = None
        self._last_displayed_index: FrameIndex | None = None
        self._recent_indices: deque[FrameIndex] = deque(maxlen=_RECENT_INDICES_CAP)
        self._hits = 0
        self._misses = 0
        self._current_frame_miss = 0
        # Tombstones: indices whose cached data is logically invalid
        # (e.g. the executor just swapped the processing chain and
        # reprocessed the index). get() must return None for these
        # until the next put() lands, even if the cache or store still
        # holds the old frame. Without this, the playback duplicate-
        # frame guard would compare the index of the just-reprocessed
        # frame to the same index it last emitted and skip the repaint,
        # leaving stale pixels on screen. Cleared atomically by put().
        self._invalidated: set[FrameIndex] = set()

    @property
    def cache_mode(self) -> CacheMode:
        return self._cache_mode

    def set_cache_mode(self, mode: CacheMode) -> None:
        """Switch cache behaviour at runtime. Pending writes complete; new
        writes start (or stop) honouring the new mode immediately."""
        with self._lock:
            self._cache_mode = mode

    def put(self, index: FrameIndex, frame: Frame) -> None:
        self._cache.put(index, frame)
        if self._cache_mode is CacheMode.WRITE_READ:
            self._write_executor.submit(self._store.write, index, frame)
        with self._lock:
            if self._last_written_index is None or index > self._last_written_index:
                self._last_written_index = index
            self._recent_indices.append(index)
            # A fresh put for an invalidated index supersedes the tombstone.
            self._invalidated.discard(index)

    def invalidate(self, index: FrameIndex) -> None:
        """Mark index's cached/stored data as logically invalid.

        Used when the processing chain changes mid-session and the same
        index is about to be reprocessed: without invalidation, a get()
        before the worker writes the new frame would return the OLD
        cached/disk data, the playback duplicate-frame guard would
        record that index as "shown", and the worker's new frame for
        the same index would then be silently dropped (same index,
        already shown). After this call, get() returns None for the
        index until the next put() clears the tombstone.
        """
        self._cache.evict_at(index)
        with self._lock:
            self._invalidated.add(index)

    def get(self, index: FrameIndex) -> Frame | None:
        with self._lock:
            tombstoned = index in self._invalidated
        if tombstoned:
            # Tombstoned indices behave as cache-and-store misses regardless
            # of what's actually present, until put() supersedes the tombstone.
            with self._lock:
                self._misses += 1
            return None
        frame = self._cache.get(index)
        if frame is not None:
            with self._lock:
                self._hits += 1
            return frame
        with self._lock:
            self._misses += 1
        if self._cache_mode is CacheMode.OFF:
            return None
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

        Tombstoned (invalidated) indices are excluded: get() answers None
        for them, so offering one would stall the fallback for a tick.
        """
        with self._lock:
            candidates = [
                i for i in self._recent_indices
                if i <= target and i not in self._invalidated
            ]
        return max(candidates) if candidates else None

    def invalidate_from(self, index: FrameIndex) -> None:
        self._cache.evict_from(index)
        self._store.clear_from(index)
        with self._lock:
            if self._last_written_index is not None and self._last_written_index >= index:
                self._last_written_index = (index - 1) if index > 0 else None

    def invalidate_all(self) -> None:
        """Drop EVERY cached + stored frame.

        Used on a chain swap: the cache and store are keyed by frame index, not
        by the chain that produced them, so once the chain changes every
        previously-processed frame is stale and must be re-rendered on the next
        read. Without this a frame already in the (potentially large) memory
        cache would be served unchanged after a processor/param change — the
        change would appear not to apply. Also clears tombstones + the recent-
        index tracking so the buffer starts clean for the new chain."""
        self._cache.clear()
        self._store.clear_from(0)
        with self._lock:
            self._last_written_index = None
            self._invalidated.clear()
            self._recent_indices.clear()

    def set_memory_max_bytes(self, max_bytes: int) -> None:
        """Resize the in-memory cache budget at runtime (evicts LRU down to fit).
        Lets the GUI's memory-cache size apply to the live session immediately."""
        self._cache.set_max_bytes(max_bytes)

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

        write_m = self._write_executor.metrics_snapshot()
        return BufferMetrics(
            frame_lag=frame_lag,
            time_lag_s=frame_lag * frame_time_s,
            display_frame_lag=display_frame_lag,
            display_time_lag_s=display_frame_lag * frame_time_s,
            current_frame_miss=self._current_frame_miss,
            memory_used_bytes=self._cache.memory_used_bytes(),
            cache_hit_ratio=ratio,
            write_outstanding=write_m.outstanding,
            write_max_outstanding=write_m.max_outstanding,
            write_submitted=write_m.submitted,
            write_completed=write_m.completed,
            write_dropped=write_m.dropped,
            write_failed=write_m.failed,
            write_latency_p50_ms=write_m.latency_p50_ms,
            write_latency_p95_ms=write_m.latency_p95_ms,
        )
