from dataclasses import dataclass

from sinner2.pipeline.processor import Processor
from sinner2.types import Frame, FrameIndex


@dataclass(frozen=True, eq=False)
class WorkItem:
    """A single task on the work queue: decode + chain-snapshot for one frame.

    Carries an immutable snapshot of the processor chain so workers see a
    consistent view even if the dispatcher swaps the chain mid-flight. The
    frame field is a numpy ndarray and intentionally excluded from equality
    — ndarray __eq__ is element-wise, not boolean, which would break dict /
    set membership.
    """

    frame_index: FrameIndex
    source_frame: Frame
    chain_snapshot: tuple[Processor, ...]
