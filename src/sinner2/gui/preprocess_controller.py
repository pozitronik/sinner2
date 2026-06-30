"""Drives a 'preprocess then play' pass over the live realtime executor.

Two buffering modes, picked from the session's skip strategy:

  - SPARSE warm-start (Synced / Predictive): pre-render only the time-aligned
    ladder that skip strategy will actually show (stride ceil(F/R)), to a fixed
    few-seconds cushion, then hand back to it on release. The ladder reaches
    equilibrium (shown rate <= throughput), so the cushion is seconds, not a
    fraction of the clip, and playback starts almost at once and stays smooth.
  - DENSE render-ahead (BestEffort, or legacy/unknown): the classic full-rate
    path — render every frame, head-start B = N·(1 − R/F), then play at native
    fps showing every frame.

It measures throughput, pre-renders the chosen head-start, then releases. The
processing-visualiser bar is the live progress UI; this also emits
progressChanged for a status readout, and started/finished so the GUI can keep
audio in step (audio stays silent while buffering, then starts on release).

The executor is poked directly (via the injected accessor): the cache is forced
to WRITE_READ and a buffering strategy is installed; on release the user's skip
strategy is restored (sparse mode). A settings change or new target restores the
user's choices on the next rebuild regardless.
"""
from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.realtime.preprocess import required_prefill, sparse_prefill
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    BufferingStrategy,
    FrameSkipStrategy,
)

# Opt-in diagnostic: set SINNER2_PREPROCESS_TRACE=1 before launch to print the
# play-path decision + buffering progress to stderr (helps debug "preprocessing
# didn't engage"). Read once at import — set it before starting the app.
_TRACE_ON = os.environ.get("SINNER2_PREPROCESS_TRACE", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def trace(message: str) -> None:
    if _TRACE_ON:
        print(f"[preprocess] {message}", file=sys.stderr, flush=True)


_POLL_INTERVAL_MS = 100
# Frames to complete before trusting the throughput estimate (a cold first
# frame loads models / runs its first inference, so its rate is meaningless).
_WARMUP_FRAMES = 3
# Seconds of time-aligned ladder to pre-render before releasing in the SPARSE
# warm-start path — also ~the buffering wait. The skip strategy sustains the rest
# (the ladder reaches equilibrium), so a couple of seconds is plenty regardless
# of how slow the pipeline is.
_CUSHION_SECONDS = 2.5


class PreprocessController(QObject):
    """Orchestrates the executor's buffering capability into a smart-head-start
    preprocess. Owns only the render orchestration; the GUI wires the signals to
    audio + status."""

    started = Signal()
    progressChanged = Signal(int, int)  # frames done, head-start target
    finished = Signal(bool)             # released to play (True) or cancelled
    failed = Signal(str)

    def __init__(
        self,
        get_executor: Callable[[], RealtimeExecutor | None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_executor = get_executor
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._active = False
        self._target_fps = 0.0
        self._frame_count = 0
        self._start_frame = 0
        self._head_start: int | None = None
        self._tick_count = 0
        # Sparse warm-start state: whether this pass renders the sparse ladder
        # (vs the dense full-rate fill), and the skip strategy to restore on
        # release so playback resumes the ladder instead of staying on BestEffort.
        self._sparse = False
        self._restore_strategy: FrameSkipStrategy | None = None

    def is_active(self) -> bool:
        return self._active

    def start(
        self,
        target_fps: float,
        runtime_strategy: FrameSkipStrategy | None = None,
    ) -> None:
        """Begin buffering ahead FROM THE CURRENT PLAYHEAD for smooth playback at
        ``target_fps`` (the source/native fps), then release to play.

        ``runtime_strategy`` is the skip strategy the session plays with. With a
        skip strategy (Synced/Predictive) and a known fps the pass lays down only
        the SPARSE ladder that strategy will show — a couple of seconds' cushion —
        and restores it on release, so playback starts almost at once and stays
        smooth. With BestEffort (or None / unknown fps) it falls back to the DENSE
        full-rate render-ahead (every frame, big head-start)."""
        if self._active:
            trace("start() ignored — already active")
            return
        executor = self._get_executor()
        if executor is None:
            trace("start() FAILED — no executor")
            self.failed.emit("no active session to preprocess")
            return
        self._frame_count = executor.frame_count()
        if self._frame_count <= 0:
            trace(f"start() FAILED — frame_count={self._frame_count}")
            self.failed.emit("nothing to preprocess")
            return
        self._target_fps = max(0.0, float(target_fps))
        self._start_frame = max(0, executor.current_frame.get())
        self._head_start = None
        self._tick_count = 0
        self._active = True
        # Sparse warm-start when a skip strategy is active and the fps is known;
        # else the legacy dense path (BestEffort / no strategy / unknown fps).
        self._sparse = (
            runtime_strategy is not None
            and not isinstance(runtime_strategy, BestEffortStrategy)
            and self._target_fps > 0
        )
        self._restore_strategy = runtime_strategy if self._sparse else None
        if self._sparse:
            # Backstop lookahead > the cushion span, so the fill never idles
            # before the controller releases (at the cushion); the strategy's own
            # outstanding cap bounds in-flight work meanwhile.
            lookahead = max(120, math.ceil(_CUSHION_SECONDS * self._target_fps) * 2)
            buffering: FrameSkipStrategy = BufferingStrategy(lookahead_frames=lookahead)
        else:
            buffering = BestEffortStrategy()
        trace(
            f"start() OK — from frame {self._start_frame}, "
            f"frame_count={self._frame_count}, target_fps={self._target_fps}, "
            f"sparse={self._sparse}"
        )
        # Disk-backed fill (not RAM-bound). No seek — start_buffering re-anchors
        # the fill to the current playhead and the buffer.has() fast-path reuses
        # any frames already buffered ahead.
        executor.set_skip_strategy(buffering)
        executor.set_cache_mode(CacheMode.WRITE_READ)
        executor.start_buffering()
        self.started.emit()
        self._timer.start()

    def play_now(self) -> None:
        """Release to playback immediately at the current fill (user override)."""
        if self._active:
            self._release(play=True)

    def cancel(self) -> None:
        """Stop buffering without playing — stay paused at the start."""
        if self._active:
            self._release(play=False)

    def _tick(self) -> None:
        executor = self._get_executor()
        if not self._active or executor is None:
            self._release(play=False)
            return
        # Frames rendered ahead of the LIVE playhead + frames remaining from it,
        # read atomically so a section fast-forward / seek (which moves the
        # playhead AND resets last_completed to that point) reads 0 ahead rather
        # than mistaking the seek's high-water mark for buffered frames.
        done_ahead, remaining = executor.preprocess_progress()
        # Size off the FACE-frame rate (the expensive frames, worst-case: assume
        # every remaining frame could carry a face). Fall back to the overall
        # rate only until a couple of face frames have been measured.
        rate = executor.face_processing_fps()
        if rate <= 0:
            rate = executor.processing_fps.get()
        if rate > 0 and done_ahead >= min(_WARMUP_FRAMES, remaining):
            if self._sparse:
                # A fixed few-seconds cushion (span), not a fraction of the clip.
                self._head_start = sparse_prefill(
                    self._target_fps, rate, _CUSHION_SECONDS
                )
            else:
                self._head_start = required_prefill(
                    remaining, rate, self._target_fps
                )
        target = self._head_start if self._head_start is not None else remaining
        self.progressChanged.emit(min(done_ahead, target), max(1, target))
        reached = self._head_start is not None and done_ahead >= self._head_start
        all_done = done_ahead >= remaining
        self._tick_count += 1
        if self._tick_count % 10 == 0 or reached or all_done:
            trace(
                f"tick: done_ahead={done_ahead} remaining={remaining} "
                f"face_fps={executor.face_processing_fps():.2f} "
                f"proc_fps={executor.processing_fps.get():.2f} "
                f"head_start={self._head_start} target={target}"
            )
        if reached or all_done:
            self._release(play=True)

    def _release(self, play: bool) -> None:
        self._timer.stop()
        if not self._active:
            return
        self._active = False
        trace(f"release(play={play})")
        executor = self._get_executor()
        if executor is not None:
            # Hand back to the user's skip strategy (sparse mode) BEFORE starting
            # the timeline, so playback resumes the ladder this pass pre-rendered
            # instead of staying stuck on BestEffort. Ordered ahead of release on
            # the executor's command queue.
            if self._restore_strategy is not None:
                executor.set_skip_strategy(self._restore_strategy)
            executor.release_buffering(play=play)
        self.finished.emit(play)
