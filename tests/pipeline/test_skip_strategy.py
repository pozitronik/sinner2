from unittest.mock import MagicMock

import pytest

from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    SkipDecision,
)


def _zero_metrics() -> BufferMetrics:
    return BufferMetrics(
        frame_lag=0,
        time_lag_s=0.0,
        display_frame_lag=0,
        display_time_lag_s=0.0,
        current_frame_miss=0,
        memory_used_bytes=0,
        cache_hit_ratio=0.0,
    )


class TestSkipDecision:
    def test_can_signal_idle(self):
        d = SkipDecision(next_frame=None)
        assert d.next_frame is None

    def test_carries_frame_index(self):
        d = SkipDecision(next_frame=42)
        assert d.next_frame == 42

    def test_is_frozen(self):
        d = SkipDecision(next_frame=0)
        with pytest.raises(Exception):
            d.next_frame = 1  # type: ignore[misc]


class TestBestEffortStrategy:
    def test_compliant_with_protocol(self):
        assert isinstance(BestEffortStrategy(), FrameSkipStrategy)

    def test_first_call_returns_zero_when_initialized_negative(self):
        s = BestEffortStrategy()
        d = s.decide(
            last_submitted=-1,
            last_completed=-1,
            timeline=MagicMock(),
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 0

    def test_always_advances_by_one(self):
        s = BestEffortStrategy()
        for last in [0, 5, 100, 10000]:
            d = s.decide(
                last_submitted=last,
                last_completed=last - 1,
                timeline=MagicMock(),
                metrics=_zero_metrics(),
            )
            assert d.next_frame == last + 1

    def test_ignores_lag(self):
        s = BestEffortStrategy()
        high_lag = BufferMetrics(
            frame_lag=1000,
            time_lag_s=33.3,
            display_frame_lag=500,
            display_time_lag_s=16.6,
            current_frame_miss=200,
            memory_used_bytes=0,
            cache_hit_ratio=0.0,
        )
        d = s.decide(
            last_submitted=10,
            last_completed=5,
            timeline=MagicMock(),
            metrics=high_lag,
        )
        assert d.next_frame == 11
