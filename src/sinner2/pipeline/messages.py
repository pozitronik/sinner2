from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    chain: tuple["Processor", ...]


@dataclass(frozen=True)
class SetSkipStrategyMsg:
    strategy: "FrameSkipStrategy"


type Message = (
    PlayMsg
    | PauseMsg
    | StopMsg
    | SeekMsg
    | SetParamsMsg
    | SetChainMsg
    | SetSkipStrategyMsg
)
