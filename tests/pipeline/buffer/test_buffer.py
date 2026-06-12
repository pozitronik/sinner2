from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from sinner2.pipeline.buffer.bounded_write_executor import (
    BoundedWriteExecutor,
    WriteExecutorMetrics,
)
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import FrameCache, MemoryFrameCache
from sinner2.pipeline.buffer.store import DiskFrameStore, FrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.types import Frame

_ZERO_WRITE_METRICS = WriteExecutorMetrics(
    outstanding=0,
    max_outstanding=8,
    submitted=0,
    completed=0,
    dropped=0,
    latency_p50_ms=0.0,
    latency_p95_ms=0.0,
)


def _frame() -> Frame:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def _mock_buffer(
    cache_get_returns: Frame | None = None,
    store_read_returns: Frame | None = None,
    timeline_current: int = 0,
    timeline_fps: float = 30.0,
) -> tuple[FrameBuffer, MagicMock, MagicMock, MagicMock, MagicMock]:
    store = MagicMock(spec=FrameStore)
    store.read.return_value = store_read_returns
    cache = MagicMock(spec=FrameCache)
    cache.get.return_value = cache_get_returns
    cache.memory_used_bytes.return_value = 0
    timeline = MagicMock(spec=Timeline)
    timeline.current_frame.return_value = timeline_current
    timeline.fps = timeline_fps
    executor = MagicMock(spec=BoundedWriteExecutor)
    executor.metrics_snapshot.return_value = _ZERO_WRITE_METRICS
    buf = FrameBuffer(store=store, cache=cache, timeline=timeline, write_executor=executor)
    return buf, store, cache, timeline, executor


class TestFrameBufferPut:
    def test_writes_to_cache(self):
        buf, store, cache, _, executor = _mock_buffer()
        f = _frame()
        buf.put(5, f)
        cache.put.assert_called_once_with(5, f)

    def test_submits_store_write_to_executor(self):
        buf, store, cache, _, executor = _mock_buffer()
        f = _frame()
        buf.put(5, f)
        executor.submit.assert_called_once_with(store.write, 5, f)

    def test_does_not_block_on_store_write(self):
        buf, store, cache, _, executor = _mock_buffer()
        store.write.side_effect = AssertionError("store.write should not be called directly")
        buf.put(5, _frame())  # should not raise — store.write only invoked via executor


class TestFrameBufferGet:
    def test_cache_hit_skips_store(self):
        f = _frame()
        buf, store, cache, *_ = _mock_buffer(cache_get_returns=f)
        out = buf.get(5)
        assert out is f
        store.read.assert_not_called()

    def test_cache_miss_falls_through_to_store(self):
        f = _frame()
        buf, store, cache, *_ = _mock_buffer(cache_get_returns=None, store_read_returns=f)
        out = buf.get(5)
        assert out is f
        store.read.assert_called_once_with(5)

    def test_cache_miss_backfills_cache(self):
        f = _frame()
        buf, store, cache, *_ = _mock_buffer(cache_get_returns=None, store_read_returns=f)
        buf.get(5)
        cache.put.assert_called_with(5, f)

    def test_full_miss_returns_none(self):
        buf, store, cache, *_ = _mock_buffer(cache_get_returns=None, store_read_returns=None)
        assert buf.get(5) is None


class TestGetAtCurrentTime:
    def test_returns_timeline_frame_index(self):
        f = _frame()
        buf, *_ = _mock_buffer(cache_get_returns=f, timeline_current=42)
        idx, frame = buf.get_at_current_time()
        assert idx == 42
        assert frame is f

    def test_records_miss_when_frame_unavailable(self):
        buf, *_ = _mock_buffer(cache_get_returns=None, store_read_returns=None, timeline_current=10)
        buf.get_at_current_time()
        assert buf.metrics().current_frame_miss == 1


class TestLastWrittenIndex:
    def test_none_initially(self):
        buf, *_ = _mock_buffer()
        assert buf.last_written_index is None

    def test_tracks_highest_put_index(self):
        buf, *_ = _mock_buffer()
        buf.put(5, _frame())
        buf.put(2, _frame())
        buf.put(10, _frame())
        assert buf.last_written_index == 10

    def test_invalidate_from_rewinds(self):
        buf, *_ = _mock_buffer()
        buf.put(5, _frame())
        buf.put(10, _frame())
        buf.invalidate_from(8)
        assert buf.last_written_index == 7


class TestLatestIndexAtOrBelow:
    def test_none_when_nothing_written(self):
        buf, *_ = _mock_buffer()
        assert buf.latest_index_at_or_below(100) is None

    def test_returns_highest_below_or_equal_target(self):
        buf, *_ = _mock_buffer()
        for i in [3, 7, 12, 20]:
            buf.put(i, _frame())
        assert buf.latest_index_at_or_below(15) == 12
        assert buf.latest_index_at_or_below(7) == 7
        assert buf.latest_index_at_or_below(100) == 20

    def test_returns_none_when_all_above_target(self):
        buf, *_ = _mock_buffer()
        for i in [50, 60, 70]:
            buf.put(i, _frame())
        assert buf.latest_index_at_or_below(10) is None

    def test_skips_invalidated_indices(self):
        # An invalidated (tombstoned) index must not be offered as a fallback —
        # get() would return None for it, stalling the playback fallback for a
        # tick (audit rank 28). The next-lower valid index is the right answer.
        buf, *_ = _mock_buffer()
        for i in [3, 7, 12]:
            buf.put(i, _frame())
        buf.invalidate(12)
        assert buf.latest_index_at_or_below(15) == 7
        buf.invalidate(7)
        assert buf.latest_index_at_or_below(15) == 3
        # put() supersedes the tombstone — the index becomes a candidate again.
        buf.put(12, _frame())
        assert buf.latest_index_at_or_below(15) == 12

    def test_handles_out_of_order_puts(self):
        buf, *_ = _mock_buffer()
        for i in [10, 5, 20, 3, 15]:
            buf.put(i, _frame())
        assert buf.latest_index_at_or_below(12) == 10
        assert buf.latest_index_at_or_below(4) == 3


class TestInvalidateFrom:
    def test_calls_cache_evict_from_and_store_clear_from(self):
        buf, store, cache, *_ = _mock_buffer()
        buf.invalidate_from(50)
        cache.evict_from.assert_called_once_with(50)
        store.clear_from.assert_called_once_with(50)


class TestInvalidateAll:
    """Whole-buffer invalidation for a chain swap. The cache + store are keyed by
    frame index, not by chain, so after the chain changes every entry is stale
    and must be dropped — otherwise a tweak appears not to apply on any frame
    that's still cached (acute with a large memory cache that never evicts)."""

    def test_clears_cache_store_and_resets_written_index(self):
        buf, store, cache, *_ = _mock_buffer()
        buf.put(5, _frame())
        buf.put(9, _frame())
        assert buf.last_written_index == 9
        buf.invalidate_all()
        cache.clear.assert_called_once_with()
        store.clear_from.assert_called_with(0)
        assert buf.last_written_index is None

    def test_clears_tombstones(self):
        f = _frame()
        buf, store, cache, *_ = _mock_buffer(cache_get_returns=f)
        buf.invalidate(5)
        assert buf.get(5) is None  # tombstoned
        buf.invalidate_all()
        assert buf.get(5) is f  # tombstone gone, cache hit returns again

    def test_set_memory_max_bytes_delegates_to_cache(self):
        buf, store, cache, *_ = _mock_buffer()
        buf.set_memory_max_bytes(2048)
        cache.set_max_bytes.assert_called_once_with(2048)


class TestInvalidate:
    """Per-index tombstone for chain-swap reprocessing.

    The executor calls invalidate(target) on seek so that a subsequent
    get(target), if it happens before the worker writes the new frame,
    doesn't return stale cache/store data and trick the playback
    duplicate-frame guard into recording target as 'shown' — which would
    silently swallow the worker's actual new frame for the same index.
    """

    def test_get_returns_none_until_next_put(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        timeline = Timeline(fps=30)
        executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
        try:
            buf = FrameBuffer(store, cache, timeline, executor)
            buf.put(5, _frame())
            assert buf.get(5) is not None  # in cache
            buf.invalidate(5)
            # Even though cache + (post-flush) store have a frame, get
            # must return None until put supersedes the tombstone.
            assert buf.get(5) is None
            buf.put(5, _frame())
            assert buf.get(5) is not None
        finally:
            executor.shutdown(wait=True)

    def test_invalidate_evicts_cache_entry(self, tmp_path: Path):
        # Cache must be cleared on invalidate; otherwise a parallel
        # get() that bypasses the tombstone check (none exist today,
        # but defence in depth) would still see the stale frame.
        store = DiskFrameStore(tmp_path)
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        timeline = Timeline(fps=30)
        executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
        try:
            buf = FrameBuffer(store, cache, timeline, executor)
            buf.put(7, _frame())
            assert cache.get(7) is not None
            buf.invalidate(7)
            assert cache.get(7) is None
        finally:
            executor.shutdown(wait=True)

    def test_put_clears_tombstone_for_other_indices_only_if_explicit(
        self, tmp_path: Path
    ):
        # Tombstone is per-index; putting a different index doesn't
        # accidentally clear an unrelated invalidation.
        store = DiskFrameStore(tmp_path)
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        timeline = Timeline(fps=30)
        executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
        try:
            buf = FrameBuffer(store, cache, timeline, executor)
            buf.put(3, _frame())
            buf.invalidate(3)
            buf.put(4, _frame())  # different index
            assert buf.get(3) is None  # tombstone still in effect
            assert buf.get(4) is not None
        finally:
            executor.shutdown(wait=True)


class TestMetrics:
    def test_initial_state(self):
        buf, *_ = _mock_buffer(timeline_current=0)
        m = buf.metrics()
        assert m.frame_lag == 0
        assert m.display_frame_lag == 0
        assert m.cache_hit_ratio == 0.0

    def test_hit_ratio_after_reads(self):
        f = _frame()
        buf, store, cache, *_ = _mock_buffer()
        cache.get.side_effect = [f, f, None]
        store.read.return_value = None
        buf.get(0)
        buf.get(1)
        buf.get(2)
        m = buf.metrics()
        assert abs(m.cache_hit_ratio - 2 / 3) < 1e-6

    def test_frame_lag_when_no_display_yet(self):
        buf, *_ = _mock_buffer(timeline_current=100)
        m = buf.metrics()
        assert m.frame_lag == 100  # nothing displayed yet → fully behind

    def test_time_lag_uses_fps(self):
        buf, *_ = _mock_buffer(timeline_current=30, timeline_fps=30.0)
        m = buf.metrics()
        assert m.frame_lag == 30
        assert abs(m.time_lag_s - 1.0) < 1e-6


class TestIntegrationWithRealComponents:
    """Smoke test using real cache + store + timeline (no mocks). Verifies
    the composition works end-to-end without IO races."""

    def test_put_then_get_via_real_components(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        timeline = Timeline(fps=30)
        executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
        try:
            buf = FrameBuffer(store, cache, timeline, executor)
            f = _frame()
            buf.put(5, f)
            # Cache should have it immediately; store will get it asynchronously.
            assert buf.get(5) is not None
        finally:
            executor.shutdown(wait=True)
        # After executor shutdown, async writes are flushed.
        assert store.has(5)

    def test_invalidate_clears_real_store_and_cache(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        timeline = Timeline(fps=30)
        executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
        try:
            buf = FrameBuffer(store, cache, timeline, executor)
            for i in range(5):
                buf.put(i, _frame())
        finally:
            executor.shutdown(wait=True)
        executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
        try:
            buf = FrameBuffer(store, cache, timeline, executor)
            buf.invalidate_from(3)
        finally:
            executor.shutdown(wait=True)
        assert store.has(0)
        assert store.has(2)
        assert not store.has(3)
        assert not store.has(4)


class TestCacheMode:
    """Cache mode gates which I/O paths the buffer takes:
       WRITE_READ: write submits to executor + cache misses fall back to store
       READ_ONLY:  no write submit + cache misses still fall back to store
       OFF:        no write submit + cache misses skip the store entirely
    """

    def test_write_read_submits_writes(self):
        buf, store, _, _, executor = _mock_buffer()
        buf.set_cache_mode(CacheMode.WRITE_READ)
        buf.put(0, _frame())
        assert executor.submit.call_count == 1
        # The submitted callable is store.write; first arg is the index.
        executor.submit.assert_called_with(store.write, 0, executor.submit.call_args[0][2])

    def test_read_only_skips_writes(self):
        buf, _, _, _, executor = _mock_buffer()
        buf.set_cache_mode(CacheMode.READ_ONLY)
        buf.put(0, _frame())
        executor.submit.assert_not_called()

    def test_off_skips_writes(self):
        buf, _, _, _, executor = _mock_buffer()
        buf.set_cache_mode(CacheMode.OFF)
        buf.put(0, _frame())
        executor.submit.assert_not_called()

    def test_read_only_still_falls_back_to_store_on_miss(self):
        wanted = _frame()
        buf, store, cache, _, _ = _mock_buffer(store_read_returns=wanted)
        buf.set_cache_mode(CacheMode.READ_ONLY)
        result = buf.get(0)
        assert result is wanted
        store.read.assert_called_once_with(0)

    def test_off_does_not_read_store_on_miss(self):
        buf, store, _, _, _ = _mock_buffer(store_read_returns=_frame())
        buf.set_cache_mode(CacheMode.OFF)
        result = buf.get(0)
        assert result is None
        store.read.assert_not_called()

    def test_set_cache_mode_hot_swaps(self):
        # Live switch must take effect on the very next put/get without rebuild.
        buf, store, _, _, executor = _mock_buffer(store_read_returns=_frame())
        buf.set_cache_mode(CacheMode.WRITE_READ)
        buf.put(0, _frame())
        assert executor.submit.call_count == 1
        buf.set_cache_mode(CacheMode.OFF)
        buf.put(1, _frame())
        assert executor.submit.call_count == 1  # no new submission
        assert buf.get(2) is None  # OFF blocks store fallback
        store.read.assert_not_called()
