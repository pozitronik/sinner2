"""Unit tests for the metrics overlay's pure helpers.

The QMetricsOverlay widget itself is paint-heavy and hard to assert on
without exhaustive pixel checks — covered by manual QA. CumulativeRateTracker
and the MetricsSample dataclass are pure logic and worth pinning."""
from __future__ import annotations

import time

import pytest

from sinner2.gui.widgets.metrics_overlay import (
    CumulativeRateTracker,
    MetricsSample,
)


class TestCumulativeRateTracker:
    def test_first_update_returns_zero(self):
        # No baseline yet — first sample seeds the tracker, returns 0.
        t = CumulativeRateTracker()
        assert t.update(0) == 0.0

    def test_computes_delta_per_second(self):
        t = CumulativeRateTracker()
        now = 100.0
        t.update(0, now=now)
        # 30 events over 1 second → 30 per second
        rate = t.update(30, now=now + 1.0)
        assert rate == pytest.approx(30.0, abs=0.01)

    def test_handles_non_unit_time_delta(self):
        t = CumulativeRateTracker()
        now = 100.0
        t.update(10, now=now)
        rate = t.update(50, now=now + 0.5)
        assert rate == pytest.approx(80.0)  # 40 events in 0.5s

    def test_returns_zero_for_zero_delta_time(self):
        t = CumulativeRateTracker()
        now = 100.0
        t.update(0, now=now)
        assert t.update(50, now=now) == 0.0

    def test_returns_zero_for_negative_count_delta(self):
        # Counter resets shouldn't produce a negative rate.
        t = CumulativeRateTracker()
        now = 100.0
        t.update(100, now=now)
        assert t.update(0, now=now + 1.0) == 0.0

    def test_reset_clears_baseline(self):
        t = CumulativeRateTracker()
        now = 100.0
        t.update(0, now=now)
        t.update(30, now=now + 1.0)  # establish baseline
        t.reset()
        # Next update should return 0 (no baseline) — even though
        # cumulative count is the same as before.
        assert t.update(30, now=now + 2.0) == 0.0

    def test_now_defaults_to_real_clock(self):
        # 50ms sleep — Windows time.sleep has ~15ms default scheduler
        # quantum, so anything shorter risks dt collapsing to 0 and
        # tripping the dt==0 guard. We only care that the default-time
        # path works, not the exact rate value.
        t = CumulativeRateTracker()
        t.update(0)
        time.sleep(0.05)
        rate = t.update(10)
        assert rate > 0


class TestMetricsSample:
    def test_constructs_with_all_fields(self):
        s = MetricsSample(
            timestamp=100.0,
            read_fps=30.0,
            process_fps=28.5,
            write_fps=27.0,
            drop_fps=0.0,
            cache_hit_ratio=0.87,
            memory_used_mb=42.0,
            work_outstanding=3,
            work_capacity=32,
            write_outstanding=2,
            write_capacity=8,
            total_drops=0,
            last_completed=120,
        )
        assert s.timestamp == 100.0
        assert s.read_fps == 30.0
        assert s.last_completed == 120

    def test_is_frozen(self):
        s = MetricsSample(
            timestamp=0.0,
            read_fps=0.0,
            process_fps=0.0,
            write_fps=0.0,
            drop_fps=0.0,
            cache_hit_ratio=0.0,
            memory_used_mb=0.0,
            work_outstanding=0,
            work_capacity=0,
            write_outstanding=0,
            write_capacity=0,
            total_drops=0,
            last_completed=-1,
        )
        with pytest.raises(Exception):
            s.timestamp = 1.0  # type: ignore[misc]

    def test_processor_timings_defaults_to_empty_dict(self):
        # Field default lets callers construct a sample without
        # supplying per-processor timing data — for tests, for first
        # snapshot before any processing has happened, etc.
        s = MetricsSample(
            timestamp=0.0,
            read_fps=0.0,
            process_fps=0.0,
            write_fps=0.0,
            drop_fps=0.0,
            cache_hit_ratio=0.0,
            memory_used_mb=0.0,
            work_outstanding=0,
            work_capacity=0,
            write_outstanding=0,
            write_capacity=0,
            total_drops=0,
            last_completed=-1,
        )
        assert s.processor_timings == {}

    def test_processor_timings_preserves_insertion_order(self):
        # The overlay renders left-to-right in the order returned by
        # the executor (chain order). Sample mustn't shuffle it.
        s = MetricsSample(
            timestamp=0.0,
            read_fps=0.0,
            process_fps=0.0,
            write_fps=0.0,
            drop_fps=0.0,
            cache_hit_ratio=0.0,
            memory_used_mb=0.0,
            work_outstanding=0,
            work_capacity=0,
            write_outstanding=0,
            write_capacity=0,
            total_drops=0,
            last_completed=0,
            processor_timings={"FaceSwapper": 25.0, "FaceEnhancer": 1200.0},
        )
        assert list(s.processor_timings.keys()) == ["FaceSwapper", "FaceEnhancer"]


class TestOverlayRendersWithTimings:
    """Smoke test the paint path with non-empty processor_timings — the
    timings row exercises a separate text-draw branch from the rates
    row. Failure mode without coverage: typo in the dict iteration,
    rendering would silently break in production."""

    def test_paints_without_error_when_timings_present(self, qtbot):
        # The overlay's paintEvent is normally driven by Qt; we invoke
        # it manually so the test stays headless. Coverage goal: the
        # processor_timings branch in _draw_header executes without
        # raising.
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QPixmap

        from sinner2.gui.widgets.metrics_overlay import QMetricsOverlay

        sample = MetricsSample(
            timestamp=time.time(),
            read_fps=30.0,
            process_fps=5.0,
            write_fps=5.0,
            drop_fps=0.0,
            cache_hit_ratio=0.5,
            memory_used_mb=64.0,
            work_outstanding=2,
            work_capacity=32,
            write_outstanding=0,
            write_capacity=8,
            total_drops=0,
            last_completed=100,
            processor_timings={"FaceSwapper": 25.0, "FaceEnhancer": 1200.0},
        )
        overlay = QMetricsOverlay(snapshot_fn=lambda: sample)
        qtbot.addWidget(overlay)
        overlay.resize(320, 196)
        # Force a sample into the deque so paintEvent has data.
        overlay._tick()  # noqa: SLF001
        # Paint to an off-screen pixmap — same code path as the on-
        # screen paint but no display required.
        pix = QPixmap(overlay.size())
        pix.fill()
        overlay.render(pix, QPoint(0, 0))
        # If the render path raised inside _draw_header's timings
        # branch, we wouldn't reach here. The rendered pixmap is also
        # non-empty (overlay paints text).
        assert not pix.isNull()
