from typing import Protocol, runtime_checkable

from sinner2.types import Frame


@runtime_checkable
class Processor(Protocol):
    """A single transformation step in the chain.

    Stateless per process() call. Init/teardown happens in setup()/release().
    Params are immutable on the instance — to change params, construct a new
    instance. This invariant is what lets the executor snapshot a chain into
    a WorkItem without worrying about mid-frame param mutation.
    """

    name: str

    def setup(self) -> None: ...

    def process(self, frame: Frame) -> Frame: ...

    def release(self) -> None: ...
