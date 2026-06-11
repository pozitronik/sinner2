"""Optional A/V sync diagnostic tracer (instrumentation only).

Live file playback is "video-master, open-loop": the Timeline ticks a wall
clock and audio free-runs at 1x with no reconciliation, so a sustained offset
between picture and sound is invisible to the code. When SINNER2_SYNC_TRACE is
set to a truthy value, PlayerController samples the clocks (wall / displayed
video frame / audio position) plus the active skip-strategy mode a few times a
second and logs them, so the desync reported on the "synced" strategy can be
measured on a real GPU+camera box. It NEVER influences playback — read-only
sampling, no feedback into the pipeline.

Enable:  set SINNER2_SYNC_TRACE=1 (or true / yes / on) before launching.
Output:  the "sinner2.sync_trace" logger at INFO, one line per sample, e.g.
    sync t=2.103 frame=63 video=2.100s audio=2.380s offset=+0.280s playing=1 mode=synced
where offset = audio - video (positive = audio ahead of the shown picture).
Compare the offset trajectory under "synced" vs "best-effort" to tell a
cold-start frame jump from a strategy falling into its lagging fallback.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer

_ENV_FLAG = "SINNER2_SYNC_TRACE"
_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_INTERVAL_MS = 100

logger = logging.getLogger("sinner2.sync_trace")


@dataclass(frozen=True)
class SyncSample:
    """One read-only snapshot of the playback clocks."""

    frame: int
    video_seconds: float
    audio_seconds: float
    playing: bool
    strategy_mode: str


def sync_trace_enabled() -> bool:
    """True when SINNER2_SYNC_TRACE is set to a truthy value."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in _TRUTHY


class SyncTracer(QObject):
    """Samples a SyncSample provider on a QTimer and logs each reading.

    Dormant unless SINNER2_SYNC_TRACE is set, so the owner can wire start()/
    stop() unconditionally (e.g. on play/pause) without guarding the env."""

    def __init__(
        self,
        sample: Callable[[], SyncSample | None],
        parent: QObject | None = None,
        interval_ms: int = _DEFAULT_INTERVAL_MS,
    ) -> None:
        super().__init__(parent)
        self._sample = sample
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._t0: float | None = None

    def start(self) -> None:
        """Begin sampling — a no-op unless tracing is enabled or already running."""
        if not sync_trace_enabled() or self._timer.isActive():
            return
        self._t0 = time.monotonic()
        logger.info("sync trace started (interval=%dms)", self._timer.interval())
        self._timer.start()

    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    def _tick(self) -> None:
        sample = self._sample()
        if sample is None:
            return
        t0 = self._t0 if self._t0 is not None else time.monotonic()
        offset = sample.audio_seconds - sample.video_seconds
        logger.info(
            "sync t=%.3f frame=%d video=%.3fs audio=%.3fs "
            "offset=%+.3fs playing=%d mode=%s",
            time.monotonic() - t0,
            sample.frame,
            sample.video_seconds,
            sample.audio_seconds,
            offset,
            1 if sample.playing else 0,
            sample.strategy_mode,
        )
