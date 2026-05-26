import pytest

from sinner2.pipeline.buffer.metrics import BufferMetrics


class TestBufferMetrics:
    def test_construction(self):
        m = BufferMetrics(
            frame_lag=5,
            time_lag_s=0.17,
            display_frame_lag=2,
            display_time_lag_s=0.06,
            current_frame_miss=3,
            memory_used_bytes=1024,
            cache_hit_ratio=0.85,
        )
        assert m.frame_lag == 5
        assert m.cache_hit_ratio == 0.85

    def test_is_frozen(self):
        m = BufferMetrics(
            frame_lag=0,
            time_lag_s=0,
            display_frame_lag=0,
            display_time_lag_s=0,
            current_frame_miss=0,
            memory_used_bytes=0,
            cache_hit_ratio=0,
        )
        with pytest.raises(Exception):
            m.frame_lag = 1  # type: ignore[misc]
