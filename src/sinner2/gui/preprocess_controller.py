"""Drives a 'preprocess then play' pass over the live realtime executor.

Forces a sequential, disk-backed fill (so the head-start renders every frame in
order without blowing RAM), measures throughput, pre-renders the smart
head-start B = N·(1 − R/F), then releases playback. The processing-visualiser
bar is the live progress UI; this also emits progressChanged for a status
readout, and started/finished so the GUI can keep audio in step (audio stays
silent while buffering, then starts from frame 0 on release-to-play).

The executor is poked directly (via the injected accessor): the strategy is
forced to BestEffort and the cache to WRITE_READ for the rest of the session —
a settings change or new target restores the user's choices on the next rebuild.
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.realtime.preprocess import required_prefill
from sinner2.pipeline.skip_strategy import BestEffortStrategy

_POLL_INTERVAL_MS = 100
# Frames to complete before trusting the throughput estimate (a cold first
# frame loads models / runs its first inference, so its rate is meaningless).
_WARMUP_FRAMES = 3


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

    def is_active(self) -> bool:
        return self._active

    def start(self, target_fps: float) -> None:
        """Begin buffering ahead FROM THE CURRENT PLAYHEAD for smooth playback at
        ``target_fps`` (the source/native fps), then release to play."""
        if self._active:
            return
        executor = self._get_executor()
        if executor is None:
            self.failed.emit("no active session to preprocess")
            return
        self._frame_count = executor.frame_count()
        if self._frame_count <= 0:
            self.failed.emit("nothing to preprocess")
            return
        self._target_fps = max(0.0, float(target_fps))
        self._start_frame = max(0, executor.current_frame.get())
        self._head_start = None
        self._active = True
        # Sequential + disk-backed fill: every frame in order, not RAM-bound. No
        # seek — the dispatcher already fills ahead of the playhead (last_submitted
        # >= current), so it reuses any frames already buffered there.
        executor.set_skip_strategy(BestEffortStrategy())
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
        completed = executor.last_completed_frame()
        # Frames rendered AT/AHEAD of where playback will start.
        done_ahead = max(0, completed - self._start_frame + 1)
        remaining = max(1, self._frame_count - self._start_frame)
        # Size off the FACE-frame rate (the expensive frames, worst-case: assume
        # every remaining frame could carry a face). Fall back to the overall
        # rate only until a couple of face frames have been measured.
        rate = executor.face_processing_fps()
        if rate <= 0:
            rate = executor.processing_fps.get()
        if rate > 0 and done_ahead >= min(_WARMUP_FRAMES, remaining):
            self._head_start = required_prefill(remaining, rate, self._target_fps)
        target = self._head_start if self._head_start is not None else remaining
        self.progressChanged.emit(min(done_ahead, target), max(1, target))
        reached = self._head_start is not None and done_ahead >= self._head_start
        all_done = completed >= self._frame_count - 1
        if reached or all_done:
            self._release(play=True)

    def _release(self, play: bool) -> None:
        self._timer.stop()
        if not self._active:
            return
        self._active = False
        executor = self._get_executor()
        if executor is not None:
            executor.release_buffering(play=play)
        self.finished.emit(play)
