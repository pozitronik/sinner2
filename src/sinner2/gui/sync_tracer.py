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
    sync t=2.103 target=126 shown=63 video=2.100s audio=2.380s offset=+0.280s lag=63f playing=1 mode=synced
where ``target`` is the wall-clock playhead, ``shown`` is the frame actually on
screen, ``video`` is the shown frame's timestamp, offset = audio - video
(positive = audio ahead of the picture), and lag = target - shown (how many
frames the picture trails the playhead). A growing lag under a slow pipeline is
the desync; ``target`` alone equals the audio clock by construction, so it can't
reveal it — ``shown`` is the signal.
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
    """One read-only snapshot of the playback clocks.

    ``frame`` is the wall-clock TARGET (timeline playhead); ``shown_frame`` is the
    frame whose pixels are actually on screen. ``video_seconds`` is the shown
    frame's timestamp — the picture the eye sees — so ``audio_seconds -
    video_seconds`` is the true A/V offset and ``frame - shown_frame`` is how far
    the picture trails the target when the pipeline can't keep up.
    """

    frame: int
    shown_frame: int
    video_seconds: float
    audio_seconds: float
    playing: bool
    strategy_mode: str


def sync_trace_enabled() -> bool:
    """True when SINNER2_SYNC_TRACE is set to a truthy value."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in _TRUTHY


# Trace file written (alongside stderr) when enabled, so a run launched by
# double-clicking run.bat leaves a capture behind even after the console closes.
# Relative → the launch cwd (the project root under run.bat).
_LOG_FILENAME = "sinner2_sync_trace.log"
_HANDLER_MARK = "_sinner2_sync_trace_handler"


def _ensure_log_output() -> None:
    """Make the trace actually print.

    The app configures no logging, so the root logger sits at WARNING with no
    handler and ``logger.info()`` here is dropped before it is even formatted —
    the tracer stays silent in production even with the env flag set (which is
    why the desync read as "unreproducible": the instrument couldn't speak).
    Lift this logger to INFO and give it its own stderr + file handlers the first
    time tracing starts. Idempotent (handlers are marked + added once);
    propagation stays on so pytest's caplog still captures records in tests.
    """
    logger.setLevel(logging.INFO)
    if any(getattr(h, _HANDLER_MARK, False) for h in logger.handlers):
        return
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    setattr(stream, _HANDLER_MARK, True)
    logger.addHandler(stream)
    try:
        file_handler = logging.FileHandler(_LOG_FILENAME, mode="w", encoding="utf-8")
    except OSError:
        # Read-only cwd or a locked file — stderr alone still carries the trace.
        return
    file_handler.setFormatter(fmt)
    setattr(file_handler, _HANDLER_MARK, True)
    logger.addHandler(file_handler)
    logger.info("writing sync trace to %s", os.path.abspath(_LOG_FILENAME))


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
        _ensure_log_output()
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
        lag_frames = sample.frame - sample.shown_frame
        logger.info(
            "sync t=%.3f target=%d shown=%d video=%.3fs audio=%.3fs "
            "offset=%+.3fs lag=%df playing=%d mode=%s",
            time.monotonic() - t0,
            sample.frame,
            sample.shown_frame,
            sample.video_seconds,
            sample.audio_seconds,
            offset,
            lag_frames,
            1 if sample.playing else 0,
            sample.strategy_mode,
        )
