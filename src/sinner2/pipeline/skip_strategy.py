from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.types import FrameIndex


@dataclass(frozen=True)
class SkipDecision:
    """Strategy output. next_frame=None means 'idle — no work to submit now'."""

    next_frame: FrameIndex | None


@runtime_checkable
class FrameSkipStrategy(Protocol):
    """Decide what frame the dispatcher should submit next.

    Called once per dispatcher tick. The implementation gets enough state
    (last submitted, last completed, current timeline position, lag metrics)
    to choose between submitting the next sequential frame, skipping ahead
    to catch up with wall-clock, or returning None to wait.
    """

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
        read_latency_ms: float | None = None,
    ) -> SkipDecision: ...

    def current_mode(self) -> str:
        """Human-readable label for the strategy's current behaviour.

        Surfaces in the status bar so the user can see when an adaptive
        strategy has shifted modes (e.g. SyncedStrategy falling back to
        sequential submission when reads can't keep up). Stable strategies
        like BestEffort just return a fixed label."""
        ...


class BestEffortStrategy:
    """Submit every frame in order. Let playback lag if processing is slow.

    Right for tuning workflows where the user wants to see the effect of every
    parameter change on every frame. Wrong for watching content where wall-
    clock sync matters — that's what SyncedStrategy is for.

    Callers should initialize last_submitted to -1 so the first call returns 0.
    """

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
        read_latency_ms: float | None = None,
    ) -> SkipDecision:
        return SkipDecision(next_frame=last_submitted + 1)

    def current_mode(self) -> str:
        return "best effort"


class SyncedStrategy:
    """Skip ahead to the wall-clock target frame when processing falls behind.

    Right for viewing — the displayed frame stays as close to the timeline as
    throughput permits, by dropping intermediate frames from the work queue.
    Never goes backwards: if the timeline somehow reports a lower frame than
    we've already submitted (e.g. clock drift), the next submission still
    advances by one.

    Adaptive fallback. When processing falls catastrophically behind
    (`target - last_completed > max_lag_frames`), keeping up by jumping
    ahead is impossible — each jump is a random-access read, and on slow
    sources (network share, HDD) random reads cost a seek + fresh fetch
    that the underlying I/O can't sustain. Aggressive skipping in that
    regime starves the reader and produces 0 fps. We fall back to
    sequential submission (`last_submitted + 1`) so reads stay sequential,
    the reader's prefetch buffer absorbs them, and throughput recovers
    to whatever the pipeline can actually sustain. Display lags the
    timeline, but progress is made.

    Recovery is implicit: once `last_completed` catches up to within
    `max_lag_frames` of `target`, the next call returns the target again.
    For permanently-slow workloads it stays in sequential mode, which is
    the right behaviour.
    """

    # 60 frames ≈ 2 seconds at 30 fps. Tuned to absorb brief slow-downs
    # without giving up on sync, while triggering quickly enough to
    # prevent the death-spiral on truly slow sources.
    _DEFAULT_MAX_LAG_FRAMES = 60

    # How far ahead of the wall-clock playhead the dispatcher may pre-render.
    # ~4 s at 30 fps: enough cushion to ride out brief slow-downs without a
    # faster-than-target pipeline rendering the whole rest of the clip ahead of
    # the display (wasted compute + memory, all discarded on a seek).
    _DEFAULT_LOOKAHEAD_FRAMES = 120

    # A single read slower than this (ms) means the SOURCE is the bottleneck:
    # skipping then forces random-access reads a slow source can't sustain
    # (network share, HDD, ffmpeg keyframe-seek restart ~100-200 ms), so the
    # sequential fallback is warranted. Below it the disk keeps up and skipping
    # is free — stay synced. ~1.5x a 30 fps frame budget; cheap SSD / cv2 in-
    # place seek reads (single-digit ms) never trip it.
    _DEFAULT_IO_BOUND_READ_MS = 50.0

    def __init__(
        self,
        max_lag_frames: int | None = None,
        recover_lag_frames: int | None = None,
        lookahead_frames: int | None = None,
        io_bound_read_ms: float | None = None,
    ) -> None:
        self._max_lag_frames = (
            max_lag_frames
            if max_lag_frames is not None
            else self._DEFAULT_MAX_LAG_FRAMES
        )
        self._lookahead_frames = (
            lookahead_frames
            if lookahead_frames is not None
            else self._DEFAULT_LOOKAHEAD_FRAMES
        )
        self._io_bound_read_ms = (
            io_bound_read_ms
            if io_bound_read_ms is not None
            else self._DEFAULT_IO_BOUND_READ_MS
        )
        # Hysteresis: enter fallback above max_lag, but don't LEAVE it until lag
        # drops below this lower bound. Without the gap, a lag parked near the
        # threshold flaps the mode (and the read pattern) on every frame. Default
        # to half the enter threshold; clamp so it's never above it.
        self._recover_lag_frames = (
            recover_lag_frames
            if recover_lag_frames is not None
            else self._max_lag_frames // 2
        )
        self._recover_lag_frames = min(self._recover_lag_frames, self._max_lag_frames)
        # Updated on each decide() call so current_mode() can report
        # whether we're keeping up or in fallback. The dispatcher reads
        # current_mode() right after decide(), so the value is fresh.
        self._in_fallback = False

    @property
    def max_lag_frames(self) -> int:
        return self._max_lag_frames

    @property
    def recover_lag_frames(self) -> int:
        return self._recover_lag_frames

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
        read_latency_ms: float | None = None,
    ) -> SkipDecision:
        target = timeline.current_frame()
        # Warm-up: nothing has completed yet (cold start — the first frame is
        # still loading models / running its first inference, which can take
        # seconds). Don't chase the wall-clock target: it climbs the whole time,
        # so max(last_submitted+1, target) would fill the queue with sparse high
        # indices and SKIP the opening of the clip. Submit sequentially from
        # where we are so the opening frames are actually processed.
        if last_completed < 0:
            self._in_fallback = False
            return SkipDecision(next_frame=last_submitted + 1)
        # Fall back to sequential only when we're catastrophically behind AND
        # READS are the bottleneck (I/O-bound): skipping then means random reads
        # a slow source can't sustain (→ ~0 fps death-spiral). When COMPUTE is
        # the bottleneck (fast disk, slow GPU), reads are cheap, so keep skipping
        # to stay synced — sequential there is just slow-motion drift. Hysteresis:
        # enter above max_lag, stay until lag drops below recover_lag, so a lag
        # parked at the boundary doesn't flap the mode/read pattern. No latency
        # signal → assume compute-bound (don't degrade to slow-motion blindly).
        lag = target - last_completed
        io_bound = (
            read_latency_ms is not None and read_latency_ms > self._io_bound_read_ms
        )
        if self._in_fallback:
            if lag <= self._recover_lag_frames:
                self._in_fallback = False
        elif lag > self._max_lag_frames and io_bound:
            self._in_fallback = True
        if self._in_fallback:
            return SkipDecision(next_frame=last_submitted + 1)
        nxt = max(last_submitted + 1, target)
        if nxt > target + self._lookahead_frames:
            # Already rendered the full look-ahead window past the playhead —
            # idle rather than pre-render further. Bounds a faster-than-target
            # pipeline's render-ahead to a fixed cushion.
            return SkipDecision(next_frame=None)
        return SkipDecision(next_frame=nxt)

    def current_mode(self) -> str:
        return "synced (lagging)" if self._in_fallback else "synced"
