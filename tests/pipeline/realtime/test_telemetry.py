"""Unit tests for TelemetryCollector — the rate + per-processor-timing
aggregator pulled out of RealtimeExecutor.

The processing-fps stall/decay, face-fps, display-fps and timing-window cases
were migrated verbatim (same assertions) from test_executor.py when the math
moved here, so they continue to pin the behaviour at its new home — now driven
against the collector directly instead of a bypass-init executor + observable.
"""
from __future__ import annotations

import time
from collections import deque

from sinner2.pipeline.realtime.telemetry import TelemetryCollector


class TestProcessingFps:
    """processing_fps reports a decaying estimate during slow-but-alive
    progress instead of a hard 0, so a slow source isn't mistaken for a hang."""

    def _processing_fps(self, *, completion_times, last_completion_time, last_fps):
        t = TelemetryCollector()
        t._completion_times = deque(completion_times)  # noqa: SLF001
        t._last_completion_time = last_completion_time  # noqa: SLF001
        t._last_fps = last_fps  # noqa: SLF001
        return t.processing_fps()

    def test_decays_to_small_positive_during_slow_progress(self):
        now = time.monotonic()
        fps = self._processing_fps(
            completion_times=[], last_completion_time=now - 5.0, last_fps=10.0
        )
        assert 0.1 < fps < 0.4  # ~0.2, not 0

    def test_zero_after_long_stall(self):
        now = time.monotonic()
        fps = self._processing_fps(
            completion_times=[], last_completion_time=now - 60.0, last_fps=10.0
        )
        assert fps == 0.0

    def test_never_completed_is_zero(self):
        fps = self._processing_fps(
            completion_times=[], last_completion_time=None, last_fps=0.0
        )
        assert fps == 0.0

    def test_windowed_rate_when_healthy(self):
        now = time.monotonic()
        fps = self._processing_fps(
            completion_times=[now - 0.3, now - 0.2, now - 0.1, now],
            last_completion_time=now, last_fps=0.0,
        )
        assert fps > 5.0

    def test_cold_start_decay_is_capped(self):
        # First completion just happened (count=1), windowed rate undefined,
        # last_fps still 0 → the 1/tiny-elapsed estimate must be capped, not a
        # bogus thousands-fps spike.
        now = time.monotonic()
        fps = self._processing_fps(
            completion_times=[now], last_completion_time=now, last_fps=0.0
        )
        assert fps <= 120.0


class TestFaceProcessingFps:
    """Throughput counting only FACE frames — the head-start sizes off these."""

    def test_zero_with_fewer_than_two_face_frames(self):
        t = TelemetryCollector()
        assert t.face_processing_fps() == 0.0
        t.record_completion(had_faces=True)
        assert t.face_processing_fps() == 0.0  # only one — no rate yet

    def test_records_only_face_frames(self):
        t = TelemetryCollector()
        t.record_completion(had_faces=True)
        t.record_completion(had_faces=False)  # empty frame
        t.record_completion(had_faces=True)
        assert len(t._face_completion_times) == 2  # noqa: SLF001 — faces only
        assert len(t._completion_times) == 3  # noqa: SLF001 — all frames

    def test_computes_rate_from_face_timestamps(self):
        t = TelemetryCollector()
        now = time.monotonic()
        # 3 face frames spanning 0.2 s → (3-1)/0.2 = 10 fps.
        t._face_completion_times.extend([now - 0.2, now - 0.1, now])  # noqa: SLF001
        assert abs(t.face_processing_fps() - 10.0) < 0.5

    def test_trims_outside_the_window(self):
        t = TelemetryCollector()
        now = time.monotonic()
        # Two stale (>3 s old) + two recent → only the recent pair counts.
        t._face_completion_times.extend(  # noqa: SLF001
            [now - 10.0, now - 9.0, now - 0.1, now]
        )
        rate = t.face_processing_fps()
        assert abs(rate - 10.0) < 0.5  # (2-1)/0.1 from the recent pair


class TestDisplayFps:
    """display_fps = distinct frames actually shown per second, computed over
    the same window as processing_fps but from on_frame timestamps."""

    def _display_rate(self, *, display_times):
        t = TelemetryCollector()
        t._display_times = deque(display_times)  # noqa: SLF001
        return t.display_fps()

    def test_windowed_rate_over_shown_frames(self):
        now = time.monotonic()
        # 5 frames shown across 0.4s → (5-1)/0.4 = 10 fps.
        times = [now - 0.4, now - 0.3, now - 0.2, now - 0.1, now]
        assert 9.0 < self._display_rate(display_times=times) < 11.0

    def test_zero_with_fewer_than_two_frames(self):
        assert self._display_rate(display_times=[time.monotonic()]) == 0.0
        assert self._display_rate(display_times=[]) == 0.0


class TestProcessorTimings:
    """Per-processor rolling average-ms, windowed and trimmed lazily on read."""

    def test_empty_when_nothing_recorded(self):
        assert TelemetryCollector().processor_timings() == {}

    def test_averages_per_processor_in_ms(self):
        t = TelemetryCollector()
        t.record_processor_timing("Fast", 1_000_000)  # 1 ms
        t.record_processor_timing("Fast", 3_000_000)  # 3 ms → avg 2 ms
        t.record_processor_timing("Slow", 20_000_000)  # 20 ms
        timings = t.processor_timings()
        assert abs(timings["Fast"] - 2.0) < 1e-6
        assert abs(timings["Slow"] - 20.0) < 1e-6

    def test_aged_entries_trim_out_of_window(self):
        t = TelemetryCollector()
        t.record_processor_timing("P", 5_000_000)
        assert "P" in t.processor_timings()
        # Re-aim the timestamp to "long ago" so the next read trims it without a
        # real 3-second wait.
        with t._timings_lock:  # noqa: SLF001
            aged = [(0.0, name, ns) for (_ts, name, ns) in t._processor_timings]  # noqa: SLF001
            t._processor_timings.clear()  # noqa: SLF001
            t._processor_timings.extend(aged)  # noqa: SLF001
        assert t.processor_timings() == {}
