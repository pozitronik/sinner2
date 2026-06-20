from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sinner2.types import Frame


@dataclass
class ChainContext:
    """Per-frame state shared DOWN the chain (one instance per frame pass).

    ``faces``: the swapper's pre-filter detections for this frame. Downstream
    consumers (the ONNX restorer backends) align with these instead of running
    their own detection — one GPU detection pass per frame instead of two, and
    swap + enhance agree on the same geometry (the swap pastes onto these very
    keypoints, so they remain valid on the post-swap frame). ``None`` means no
    upstream detection ran (enhancer-only chain, batch stage, direct call) →
    consumers self-detect as before. An EMPTY list is a real result ("this
    frame has no faces") and is trusted, not re-detected.

    ``swapped_faces``: the SUBSET of ``faces`` the swapper actually swapped this
    frame (after the sex / face-map / single-face filters) — a strict subset, in
    swap order. The enhancer's "only swapped faces" option restores just these
    instead of every detected face (don't alter bystanders you didn't swap).
    ``None`` means no swapper ran (enhancer-only chain, batch enhance stage,
    direct call) → the enhancer falls back to all detected faces. An EMPTY list
    is a real result ("the swapper swapped nothing"), trusted as such.

    ``frame_index``: which frame this is, when the executor knows it (realtime +
    batch both do). Face-mapping's detection-free runtime keys its precomputed
    per-frame geometry on this; ``None`` (direct call / stage without an index)
    falls back to detecting.

    Executors construct one context per frame and pass it only to processors
    whose class sets ``accepts_context = True`` (their ``process`` takes an
    optional second argument); everything else keeps the plain one-argument
    ``process(frame)`` contract below.
    """

    faces: list[Any] | None = field(default=None)
    swapped_faces: list[Any] | None = field(default=None)
    frame_index: int | None = field(default=None)


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
    # True if process() may be called concurrently on ONE shared instance
    # (e.g. an ONNX Runtime session). False if each concurrent worker needs
    # its OWN instance (e.g. GFPGAN, which mutates torch state) — the batch
    # stage runner builds N instances for non-thread-safe processors.
    thread_safe: bool

    def setup(self) -> None: ...

    def process(self, frame: Frame) -> Frame: ...

    def release(self) -> None: ...
