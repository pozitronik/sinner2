from concurrent.futures import Future
from dataclasses import dataclass

from sinner2.types import Frame, FrameIndex


@dataclass(frozen=True, eq=False)
class WorkItem:
    """A single task on the work queue: one frame to process.

    The source frame is delivered asynchronously via the ReaderPool —
    `source_future.result()` blocks until the reader thread services
    the request. Workers await the future before calling the chain so
    the dispatcher isn't blocked by source I/O. The future is `eq=False`
    (Future identity is what matters, not value equality) and the dataclass
    is intentionally `eq=False` to avoid comparing the carried future.
    """

    frame_index: FrameIndex
    source_future: Future[Frame | None]
    # The executor "world" (source / buffer / chain) this item belongs to. A
    # reconfigure bumps the executor's generation; a worker discards a result
    # whose generation no longer matches (it belongs to the pre-swap world).
    generation: int = 0
