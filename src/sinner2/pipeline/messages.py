from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.types import FrameIndex

if TYPE_CHECKING:
    from sinner2.pipeline.processor import Processor
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
)
