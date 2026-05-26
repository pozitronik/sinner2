from dataclasses import dataclass

from sinner2.types import Frame, FrameIndex


@dataclass(frozen=True, eq=False)
class WorkItem:
    """A single task on the work queue: one frame to process.

    Workers use their own indexed chain (built per-worker at start time) so
    the item doesn't carry the chain itself. The frame field is a numpy
    ndarray and intentionally excluded from equality — ndarray __eq__ is
    element-wise, not boolean, which would break dict / set membership.
    """

    frame_index: FrameIndex
    source_frame: Frame
