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

    def __init__(self, max_lag_frames: int | None = None) -> None:
        self._max_lag_frames = (
            max_lag_frames
            if max_lag_frames is not None
            else self._DEFAULT_MAX_LAG_FRAMES
        )
        # Updated on each decide() call so current_mode() can report
        # whether we're keeping up or in fallback. The dispatcher reads
        # current_mode() right after decide(), so the value is fresh.
        self._in_fallback = False

    @property
    def max_lag_frames(self) -> int:
        return self._max_lag_frames

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
    ) -> SkipDecision:
        target = timeline.current_frame()
        # Fall back to sequential when we're catastrophically behind.
        # `last_completed >= 0` guards the warm-up case where nothing
        # has completed yet — we shouldn't decide we're "behind" before
        # the first frame even processes.
        if (
            last_completed >= 0
            and target - last_completed > self._max_lag_frames
        ):
            self._in_fallback = True
            return SkipDecision(next_frame=last_submitted + 1)
        self._in_fallback = False
        return SkipDecision(next_frame=max(last_submitted + 1, target))

    def current_mode(self) -> str:
        return "synced (lagging)" if self._in_fallback else "synced"
