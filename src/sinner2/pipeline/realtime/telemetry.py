"""Rate + per-processor-timing aggregation for RealtimeExecutor.

Pulled out of the executor so the FPS windows, the face-frame rate, the
display rate, and per-processor timing live in one cohesive, lock-owning unit
instead of threaded through the executor's many concerns. The executor keeps
the public ObservableValues (``processing_fps`` / ``display_fps`` …); this
collector owns the raw timestamp buffers and the windowing math, and the
executor publishes the computed values from its playback tick.

Threading: workers call ``record_completion`` / ``record_processor_timing`` on
the worker threads (cheap appends under a dedicated lock); the playback thread
calls ``record_display`` and reads the rate methods once per tick. Two locks
(``_fps_lock``, ``_timings_lock``) keep the worker append path off the playback
read path. ``record_display`` / ``display_fps`` are playback-thread-only and so
take no lock, exactly as when this state lived on the executor.
"""
from __future__ import annotations

import threading
import time
from collections import deque

# Time-window for processing_fps. Time-based (not count-based) so the reading
# reflects current throughput rather than a sample-window average that can stay
# stale through pauses. 3s is short enough to update visibly after the user
# changes a setting and long enough to be smooth at low fps.
_FPS_WINDOW_S = 3.0
# When completions are too sparse for a windowed rate, report the rate implied
# by "time since the last completion" so a slow-but-progressing pipeline shows a
# small positive fps instead of a hard 0 (indistinguishable from a hang). After
# this long with no completion at all, fall through to 0 — genuinely stalled.
_FPS_STALL_HOLD_S = 30.0
# Ceiling for the stall decay estimate when there's no prior windowed rate to
# cap it (cold start): keeps the first-completion 1/tiny-elapsed reading from
# spiking to a bogus thousands-fps value before the windowed rate kicks in.
_FPS_DECAY_CAP = 120.0
# Time window for per-processor average-ms readout. Matches _FPS_WINDOW_S so the
# metrics-overlay row and the rates next to it cover the same wall-clock slice.
# Cap deque size so a fast no-op chain (1000+ fps) can't unbounded-grow the
# buffer between window trims.
_TIMING_WINDOW_S = 3.0
_TIMING_DEQUE_CAP = 4096


class TelemetryCollector:
    """Aggregates frame-completion, display, and per-processor timing samples
    into rolling rates for the metrics overlay / status bar."""

    def __init__(self) -> None:
        self._fps_lock = threading.RLock()
        # Timestamps of frame completions in the last _FPS_WINDOW_S seconds.
        # Workers append (cheap, list append under lock). The playback thread
        # trims and reads once per tick so workers never block on Qt signal
        # emission and the GUI gets updates at a sane rate (~30 Hz) instead of
        # per-completion (which scales with worker count and serialises the
        # whole pool on the observable's lock).
        self._completion_times: deque[float] = deque()
        # Completion timestamps of FACE frames only (the swapper detected a
        # face). The preprocessing head-start sizes off how fast these — the
        # expensive frames — render, since empty frames are nearly free and
        # would inflate the overall rate. Trimmed lazily in face_processing_fps.
        self._face_completion_times: deque[float] = deque()
        # Most recent completion timestamp (never trimmed) + last windowed fps,
        # so a slow-but-alive pipeline reports a decaying estimate rather than 0.
        self._last_completion_time: float | None = None
        self._last_fps = 0.0
        # Timestamps of distinct frames shown (playback thread only — no lock).
        # Trimmed + read as display_fps each tick, alongside processing_fps.
        self._display_times: deque[float] = deque()
        # Per-processor timing: append (timestamp, processor_name, ns) per
        # process() call. Readers (overlay) get a time-windowed dict via
        # processor_timings(). Bounded deque so a fast no-op chain can't grow it
        # without limit between trim cycles — at 1000+ fps with no enhancer we'd
        # otherwise leak 3000+ entries between overlay ticks.
        self._timings_lock = threading.RLock()
        self._processor_timings: deque[tuple[float, str, int]] = deque(
            maxlen=_TIMING_DEQUE_CAP
        )

    # ---- worker-thread writers (cheap appends) ----

    def record_completion(self, had_faces: bool = False) -> None:
        """Append a completion timestamp. Cheap by design — no calculation and
        no observable publish in the worker hot path. The playback thread reads
        these timestamps and publishes processing_fps. Face frames are tracked
        separately so preprocessing can size its head-start off the expensive
        frames only."""
        with self._fps_lock:
            now = time.monotonic()
            self._completion_times.append(now)
            self._last_completion_time = now
            if had_faces:
                self._face_completion_times.append(now)

    def record_processor_timing(self, name: str, elapsed_ns: int) -> None:
        """Append one process() wall-clock sample for ``name``."""
        with self._timings_lock:
            self._processor_timings.append((time.monotonic(), name, elapsed_ns))

    # ---- playback-thread writer (no lock; single-threaded by contract) ----

    def record_display(self) -> None:
        """Record a distinct frame reaching the display. Playback thread only."""
        self._display_times.append(time.monotonic())

    # ---- readers ----

    def processing_fps(self) -> float:
        """Windowed processing throughput, with a decaying fallback when
        completions are too sparse for a windowed rate (so a slow-but-alive
        pipeline isn't mistaken for a hang). 0.0 when genuinely stalled or
        never started."""
        now = time.monotonic()
        cutoff = now - _FPS_WINDOW_S
        fps = 0.0
        with self._fps_lock:
            while self._completion_times and self._completion_times[0] < cutoff:
                self._completion_times.popleft()
            count = len(self._completion_times)
            last = self._last_completion_time
            if count >= 2:
                span = self._completion_times[-1] - self._completion_times[0]
                if span > 0:
                    fps = (count - 1) / span
                self._last_fps = fps
            elif last is not None and (now - last) <= _FPS_STALL_HOLD_S:
                # Sparse: report the rate implied by the time since the last
                # completion, decaying toward 0 the longer it's been. Cap with
                # the last windowed value so the instant after a completion
                # doesn't spike to a huge 1/tiny-elapsed reading.
                elapsed = now - last
                decayed = (1.0 / elapsed) if elapsed > 0 else self._last_fps
                cap = self._last_fps if self._last_fps > 0 else _FPS_DECAY_CAP
                fps = min(decayed, cap)
        return fps

    def display_fps(self) -> float:
        """Distinct frames actually shown per second over the same window.
        Single-threaded (playback loop), so no lock. Naturally decays to 0 when
        playback stops and no new frames arrive — unlike processing, there's no
        'slow but alive' estimate to hold."""
        cutoff = time.monotonic() - _FPS_WINDOW_S
        while self._display_times and self._display_times[0] < cutoff:
            self._display_times.popleft()
        if len(self._display_times) < 2:
            return 0.0
        dspan = self._display_times[-1] - self._display_times[0]
        return (len(self._display_times) - 1) / dspan if dspan > 0 else 0.0

    def face_processing_fps(self) -> float:
        """Throughput counting FACE frames only, over the recent window. 0.0
        when fewer than two face frames have completed (no reliable estimate —
        preprocessing then falls back to the overall rate). Used to size the
        head-start for the expensive frames."""
        now = time.monotonic()
        cutoff = now - _FPS_WINDOW_S
        with self._fps_lock:
            times = self._face_completion_times
            while times and times[0] < cutoff:
                times.popleft()
            if len(times) < 2:
                return 0.0
            span = times[-1] - times[0]
            return (len(times) - 1) / span if span > 0 else 0.0

    def processor_timings(self) -> dict[str, float]:
        """Average milliseconds per process() call over the last
        _TIMING_WINDOW_S seconds, per processor name.

        Used by the metrics overlay to surface where each frame's wall-clock is
        going. Aged-out entries are trimmed lazily on read so the deque doesn't
        bloat between ticks; callers should treat the result as the current
        rolling average, not cumulative. Empty dict when no frames have been
        processed in the window (paused or idle)."""
        cutoff = time.monotonic() - _TIMING_WINDOW_S
        sums: dict[str, list[int]] = {}
        with self._timings_lock:
            while self._processor_timings and self._processor_timings[0][0] < cutoff:
                self._processor_timings.popleft()
            for _ts, name, ns in self._processor_timings:
                sums.setdefault(name, []).append(ns)
        return {
            name: (sum(vals) / len(vals)) / 1_000_000.0
            for name, vals in sums.items()
        }
