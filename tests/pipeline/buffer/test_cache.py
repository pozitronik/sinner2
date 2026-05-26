import numpy as np
import pytest

from sinner2.pipeline.buffer.cache import FrameCache, MemoryFrameCache
from sinner2.types import Frame


def _frame() -> Frame:
    return np.zeros((10, 10, 3), dtype=np.uint8)  # 300 bytes


class TestMemoryFrameCache:
    def test_compliant_with_protocol(self):
        assert isinstance(MemoryFrameCache(max_bytes=1024), FrameCache)

    def test_rejects_zero_budget(self):
        with pytest.raises(ValueError):
            MemoryFrameCache(max_bytes=0)

    def test_rejects_negative_budget(self):
        with pytest.raises(ValueError):
            MemoryFrameCache(max_bytes=-1)

    def test_put_then_get_returns_same_frame(self):
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        f = _frame()
        cache.put(5, f)
        assert cache.get(5) is f

    def test_get_missing_returns_none(self):
        cache = MemoryFrameCache(max_bytes=1024)
        assert cache.get(99) is None

    def test_memory_used_reflects_put(self):
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        assert cache.memory_used_bytes() == 0
        f = _frame()
        cache.put(0, f)
        assert cache.memory_used_bytes() == f.nbytes

    def test_outsized_frame_is_skipped(self):
        cache = MemoryFrameCache(max_bytes=100)
        huge = np.zeros((100, 100, 3), dtype=np.uint8)
        cache.put(0, huge)
        assert cache.get(0) is None
        assert cache.memory_used_bytes() == 0

    def test_eviction_keeps_total_under_budget(self):
        cache = MemoryFrameCache(max_bytes=600)
        for i in range(5):
            cache.put(i, _frame())
        assert cache.memory_used_bytes() <= 600

    def test_lru_get_marks_as_recently_used(self):
        cache = MemoryFrameCache(max_bytes=600)
        cache.put(0, _frame())
        cache.put(1, _frame())
        cache.get(0)
        cache.put(2, _frame())
        assert cache.get(0) is not None
        assert cache.get(1) is None

    def test_replace_same_index(self):
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        f1 = _frame()
        f2 = _frame()
        cache.put(0, f1)
        cache.put(0, f2)
        assert cache.get(0) is f2
        assert cache.memory_used_bytes() == f2.nbytes

    def test_evict_before(self):
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        for i in [10, 20, 30]:
            cache.put(i, _frame())
        cache.evict_before(25)
        assert cache.get(10) is None
        assert cache.get(20) is None
        assert cache.get(30) is not None

    def test_evict_from(self):
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        for i in [10, 20, 30]:
            cache.put(i, _frame())
        cache.evict_from(25)
        assert cache.get(10) is not None
        assert cache.get(20) is not None
        assert cache.get(30) is None

    def test_evict_releases_memory(self):
        cache = MemoryFrameCache(max_bytes=10 * 1024)
        cache.put(0, _frame())
        cache.put(1, _frame())
        before = cache.memory_used_bytes()
        cache.evict_before(1)
        assert cache.memory_used_bytes() < before
