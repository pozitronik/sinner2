from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import FrameCache, MemoryFrameCache
from sinner2.pipeline.buffer.store import DiskFrameStore, FrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.types import Frame


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
    executor = MagicMock(spec=ThreadPoolExecutor)
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


class TestInvalidateFrom:
    def test_calls_cache_evict_from_and_store_clear_from(self):
        buf, store, cache, *_ = _mock_buffer()
        buf.invalidate_from(50)
        cache.evict_from.assert_called_once_with(50)
        store.clear_from.assert_called_once_with(50)


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
        with ThreadPoolExecutor(max_workers=2) as executor:
            buf = FrameBuffer(store, cache, timeline, executor)
            f = _frame()
            buf.put(5, f)
            # Cache should have it immediately; store will get it asynchronously
            assert buf.get(5) is not None
        # After executor shutdown, async writes are flushed
        assert store.has(5)

    def test_invalidate_clears_real_store_and_cache(self, tmp_path: Path):
        store = DiskFrameStore(tmp_path)
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        timeline = Timeline(fps=30)
        with ThreadPoolExecutor(max_workers=2) as executor:
            buf = FrameBuffer(store, cache, timeline, executor)
            for i in range(5):
                buf.put(i, _frame())
        # Re-open with a fresh executor to verify after-flush state
        with ThreadPoolExecutor(max_workers=2) as executor:
            buf = FrameBuffer(store, cache, timeline, executor)
            buf.invalidate_from(3)
        assert store.has(0)
        assert store.has(2)
        assert not store.has(3)
        assert not store.has(4)
