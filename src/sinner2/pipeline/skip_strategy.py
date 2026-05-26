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


class SyncedStrategy:
    """Skip ahead to the wall-clock target frame when processing falls behind.

    Right for viewing — the displayed frame stays as close to the timeline as
    throughput permits, by dropping intermediate frames from the work queue.
    Never goes backwards: if the timeline somehow reports a lower frame than
    we've already submitted (e.g. clock drift), the next submission still
    advances by one. v1 keeps the math intentionally simple; a more adaptive
    variant that reads metrics.frame_lag for aggressive skip distance can
    layer on later if needed.
    """

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
    ) -> SkipDecision:
        target = timeline.current_frame()
        return SkipDecision(next_frame=max(last_submitted + 1, target))
