import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.types import FrameIndex

if TYPE_CHECKING:
    from sinner2.io.reader_pool import ReaderPool
    from sinner2.pipeline.buffer.buffer import FrameBuffer
    from sinner2.pipeline.buffer.timeline import Timeline
    from sinner2.pipeline.face_map import FaceMap
    from sinner2.pipeline.face_map_geometry import FrameGeometry
    from sinner2.pipeline.processor import Processor
    from sinner2.pipeline.sections import SectionSet
    from sinner2.pipeline.skip_strategy import FrameSkipStrategy


@dataclass(frozen=True)
class PlayMsg:
    pass


@dataclass(frozen=True)
class PauseMsg:
    pass


@dataclass(frozen=True)
class StopMsg:
    pass


@dataclass(frozen=True)
class SeekMsg:
    target_frame: FrameIndex


@dataclass(frozen=True)
class SetParamsMsg:
    processor_name: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class SetChainMsg:
    """Hot-swap the shared processor chain.

    All workers share one chain — ORT sessions are thread-safe for concurrent
    inference, so multiple workers can call the same swapper.get() in
    parallel and let ORT schedule across the GPU efficiently. Processors
    that aren't thread-safe (e.g. GFPGAN) must serialize internally.
    """

    chain: tuple["Processor", ...]


@dataclass(frozen=True)
class SetSkipStrategyMsg:
    strategy: "FrameSkipStrategy"


@dataclass(frozen=True)
class SetWorkerCountMsg:
    """Scale the worker pool up or down without rebuilding the executor.

    Adding workers spawns new threads against the existing shared chain and
    queue. Removing workers signals the surplus to exit at their next loop
    iteration (after finishing any frame they're currently mid-process on).
    """

    n: int


@dataclass(frozen=True)
class SetPlaybackModeMsg:
    """Change how fast the display thread polls the buffer."""

    mode: PlaybackMode


@dataclass(frozen=True)
class SetFaceMapMsg:
    """Hot-apply a face map (per-identity source routing) to the live chain's
    swapper WITHOUT a rebuild — the swapper re-analyses the assigned source
    images in place. Empty map → the single global source."""

    face_map: "FaceMap"


@dataclass(frozen=True)
class SetGeometryMsg:
    """Hot-apply (or clear) the precomputed per-frame geometry on the live
    swapper WITHOUT a rebuild. When set + mapping is active, the swapper skips
    detection and rebuilds each frame's faces from it; None reverts to detecting."""

    geometry: "FrameGeometry | None"


@dataclass(frozen=True)
class SetSectionsMsg:
    """Restrict playback to a set of timeline sections (an inclusive frame-range
    selection). Empty = no restriction. When non-empty, the dispatcher fast-
    forwards the playhead over the gaps so only the selected frames play, and
    stops at the end of the last section."""

    sections: "SectionSet"


@dataclass(frozen=True)
class RerenderMsg:
    """Re-render from the current playhead forward: drop cached frames at and
    after the current frame so they reprocess through the (possibly newly
    tuned) chain, then resubmit from the playhead. Cached frames BEFORE the
    playhead are left intact."""

    pass


@dataclass(frozen=True)
class ReconfigureMsg:
    """Re-point a RUNNING executor at a new reader pool / buffer / timeline /
    chain WITHOUT tearing down the worker pool.

    Source/target changes use this instead of stop()+new-executor: recreating
    the worker threads leaks GPU memory because ORT's CUDA execution provider
    keeps per-thread state for threads that have since died, so each rebuild
    stacks ~N-workers' worth of CUDA memory. Keeping the same threads across the
    swap avoids that entirely.

    The new chain arrives UN-set-up; the handler calls setup() on the dispatcher
    thread so the source-face detector's ORT call runs on a persistent thread,
    not a freshly spawned one. Coordination is via the carried containers:
      - ``done``: set when the swap completes (success or failure).
      - ``old_out``: receives this executor's PREVIOUS (reader_pool, buffer) so
        the caller can shut them down off the dispatcher thread.
      - ``error_out``: receives a message string if the new chain's setup raised
        (e.g. no face in the new source) — the swap is abandoned and the old
        world stays live.
    """

    reader_pool: "ReaderPool"
    buffer: "FrameBuffer"
    timeline: "Timeline"
    chain: tuple["Processor", ...]
    strategy: "FrameSkipStrategy"
    playback_mode: PlaybackMode
    restore_frame: FrameIndex
    play: bool
    done: threading.Event
    old_out: list
    error_out: list


type Message = (
    PlayMsg
    | PauseMsg
    | StopMsg
    | SeekMsg
    | SetParamsMsg
    | SetChainMsg
    | SetSkipStrategyMsg
    | SetWorkerCountMsg
    | SetPlaybackModeMsg
    | SetSectionsMsg
    | SetFaceMapMsg
    | SetGeometryMsg
    | RerenderMsg
    | ReconfigureMsg
)
