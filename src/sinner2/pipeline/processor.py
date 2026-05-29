from typing import Protocol, runtime_checkable

from sinner2.types import Frame


@runtime_checkable
class Processor(Protocol):
    """A single transformation step in the chain.

    Stateless per process() call. Init/teardown happens in setup()/release().
    Params are immutable on the instance — to change params, construct a new
    instance. This invariant is what lets the executor snapshot a chain into
    a WorkItem without worrying about mid-frame param mutation.

    Thread safety contract:
      * process() may be called concurrently from multiple worker threads.
        Implementations must either be re-entrant or serialize internally
        (e.g. a Lock around non-thread-safe backends — see FaceEnhancer).
      * setup() and release() are called from the executor's dispatcher
        thread, never concurrently with themselves.
      * release() is normally called after all in-flight workers have drained
        (RealtimeExecutor._wait_for_inflight). However, that wait has a
        bounded timeout (currently 5s); if it expires, release() runs while
        a worker may still be inside process(). Implementations that null out
        backend handles in release() should defend by snapshotting the handle
        into a local at the top of process() — see FaceEnhancer.process for
        the pattern. Implementations whose release() is a no-op or only
        clears Python-level state need no extra care.
    """

    name: str

    def setup(self) -> None: ...

    def process(self, frame: Frame) -> Frame: ...

    def release(self) -> None: ...
