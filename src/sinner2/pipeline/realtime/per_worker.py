"""Per-worker processor wrapper for the realtime executor.

The realtime executor shares ONE chain across all workers. That's the fast
path for the thread-safe swapper (a single ORT session called concurrently —
building N sessions instead created N CUDA contexts and slowed things down).
But the enhancer (GFPGAN) mutates torch state and guards process() with a
lock, so a single shared instance serialises every worker on that lock.

PerWorkerProcessor bridges the gap: it presents to the executor as one
thread-safe processor, but hands each WORKER THREAD its own underlying
instance. N workers then enhance in parallel — each on its own GFPGAN (N
weight copies sharing the process's single torch CUDA context, NOT N
contexts) — instead of serialising. Instances build lazily on a thread's
first process() call and are torn down by release(); the executor calls
release() only after draining in-flight work, so it never races a live
process().

This is the realtime analogue of the batch stage runner's per-worker
instancing (see batch/stage.py _ProcessorPool): same goal (un-serialise a
non-thread-safe processor across workers), different mechanism (the realtime
worker pool isn't pinned, so each thread lazily owns its instance via a
thread-local rather than leasing from a queue).
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from sinner2.pipeline.processor import Processor
from sinner2.types import Frame


class PerWorkerProcessor:
    # Thread-safe FROM THE EXECUTOR'S VIEW: distinct worker threads never touch
    # the same underlying instance, so the shared chain can call process()
    # concurrently without a lock.
    thread_safe = True

    def __init__(self, factory: Callable[[], Processor], name: str) -> None:
        self.name = name
        self._factory = factory
        self._local = threading.local()
        # Every instance ever built, so release() can tear them all down.
        # Guarded because workers append from their own threads.
        self._lock = threading.Lock()
        self._instances: list[Processor] = []

    def setup(self) -> None:
        # No-op: each worker thread builds its OWN instance lazily on its first
        # process() call. setup() runs on the executor's setup thread — not a
        # worker — so anything built here wouldn't be the instance the workers
        # actually use.
        pass

    def process(self, frame: Frame) -> Frame:
        inst: Processor | None = getattr(self._local, "instance", None)
        if inst is None:
            inst = self._factory()
            inst.setup()
            self._local.instance = inst
            with self._lock:
                self._instances.append(inst)
        return inst.process(frame)

    def release(self) -> None:
        # The executor calls this only after in-flight work has drained, so no
        # worker is inside process() here. Tear down every per-thread instance
        # and drop the thread-local refs so a later re-setup starts clean.
        with self._lock:
            instances = self._instances
            self._instances = []
        for inst in instances:
            try:
                inst.release()
            except Exception:
                pass
        self._local = threading.local()
