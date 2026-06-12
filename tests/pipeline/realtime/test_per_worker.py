"""Tests for PerWorkerProcessor — the realtime wrapper that gives each worker
thread its own instance of a non-thread-safe processor."""
from __future__ import annotations

import threading
import time

import numpy as np

from sinner2.pipeline.processor import Processor
from sinner2.pipeline.realtime.per_worker import PerWorkerProcessor
from sinner2.types import Frame


class _Stub:
    """Records setup/process/release counts and the max number of threads
    simultaneously inside process() — which must stay 1 for a correctly
    isolated per-thread instance."""

    name = "Stub"

    def __init__(self) -> None:
        self.setup_calls = 0
        self.process_calls = 0
        self.release_calls = 0
        self._active = 0
        self._guard = threading.Lock()
        self.max_active = 0

    def setup(self) -> None:
        self.setup_calls += 1

    def process(self, frame: Frame) -> Frame:
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.process_calls += 1
        time.sleep(0.005)  # widen the window so any cross-thread sharing shows
        with self._guard:
            self._active -= 1
        return frame

    def release(self) -> None:
        self.release_calls += 1


def _frame() -> Frame:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _recording_factory():
    """Returns (factory, built_list). The factory appends each built instance
    (thread-safely) so a test can inspect how many were created."""
    built: list[_Stub] = []
    lock = threading.Lock()

    def factory() -> _Stub:
        s = _Stub()
        with lock:
            built.append(s)
        return s

    return factory, built


class TestContextForwarding:
    """The wrapper always accepts a ChainContext but forwards it only to
    wrapped instances that declare accepts_context themselves."""

    def test_forwards_ctx_to_context_aware_instance(self):
        from sinner2.pipeline.processor import ChainContext

        class _CtxAware:
            name = "aware"
            thread_safe = False
            accepts_context = True

            def __init__(self):
                self.seen = "unset"

            def setup(self):
                pass

            def process(self, frame, ctx=None):
                self.seen = ctx
                return frame

            def release(self):
                pass

        made: list = []

        def factory():
            made.append(_CtxAware())
            return made[-1]

        pw = PerWorkerProcessor(factory=factory, name="aware")
        assert pw.accepts_context is True
        ctx = ChainContext()
        pw.process(np.zeros((4, 4, 3), np.uint8), ctx)
        assert made[0].seen is ctx

    def test_plain_instance_called_without_ctx(self):
        from sinner2.pipeline.processor import ChainContext

        # _Stub.process takes only (frame) — forwarding ctx would TypeError.
        pw = PerWorkerProcessor(factory=_Stub, name="plain")
        out = pw.process(np.zeros((4, 4, 3), np.uint8), ChainContext())
        assert out.shape == (4, 4, 3)


class TestPerWorkerProcessor:
    def test_is_thread_safe_and_keeps_name(self):
        w = PerWorkerProcessor(factory=_Stub, name="FaceEnhancer")
        assert w.thread_safe is True
        assert w.name == "FaceEnhancer"

    def test_satisfies_processor_protocol(self):
        assert isinstance(PerWorkerProcessor(factory=_Stub, name="x"), Processor)

    def test_setup_builds_nothing(self):
        # Per-thread instances build lazily on first process(), NOT in setup()
        # (which runs on the executor's setup thread, not a worker).
        factory, built = _recording_factory()
        w = PerWorkerProcessor(factory=factory, name="x")
        w.setup()
        assert built == []

    def test_same_thread_reuses_one_instance(self):
        factory, built = _recording_factory()
        w = PerWorkerProcessor(factory=factory, name="x")
        for _ in range(3):
            w.process(_frame())
        assert len(built) == 1  # built once, reused
        assert built[0].setup_calls == 1
        assert built[0].process_calls == 3

    def test_each_thread_gets_its_own_instance(self):
        factory, built = _recording_factory()
        w = PerWorkerProcessor(factory=factory, name="x")

        def run() -> None:
            for _ in range(4):
                w.process(_frame())

        threads = [threading.Thread(target=run) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 3  # one instance per worker thread
        assert all(s.setup_calls == 1 for s in built)
        # The whole point: no instance is ever touched by two threads at once.
        assert all(s.max_active == 1 for s in built)
        assert sum(s.process_calls for s in built) == 12

    def test_release_tears_down_all_then_rebuilds(self):
        factory, built = _recording_factory()
        w = PerWorkerProcessor(factory=factory, name="x")

        threads = [
            threading.Thread(target=lambda: w.process(_frame())) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(built) == 2

        w.release()
        assert all(s.release_calls == 1 for s in built)

        # A process() after release builds a fresh instance (state was dropped).
        w.process(_frame())
        assert len(built) == 3
        assert built[-1].release_calls == 0

    def test_release_with_nothing_built_is_safe(self):
        w = PerWorkerProcessor(factory=_Stub, name="x")
        w.release()  # no instances yet — must not raise

    def test_release_thread_local_frees_and_rebuilds_calling_thread(self):
        factory, built = _recording_factory()
        w = PerWorkerProcessor(factory=factory, name="x")
        w.process(_frame())  # build instance 0 on this thread
        assert len(built) == 1
        w.release_thread_local()  # free this thread's instance
        assert built[0].release_calls == 1
        w.process(_frame())  # next call rebuilds a fresh instance
        assert len(built) == 2
        assert built[1].release_calls == 0

    def test_release_thread_local_leaves_other_threads_instances(self):
        # The shrink path: one worker exits and frees ITS model, while the
        # surviving workers' instances must stay resident.
        factory, built = _recording_factory()
        w = PerWorkerProcessor(factory=factory, name="x")
        built_evt = threading.Event()
        hold_evt = threading.Event()

        def survivor() -> None:
            w.process(_frame())  # builds its own instance
            built_evt.set()
            hold_evt.wait()  # park WITHOUT releasing

        t = threading.Thread(target=survivor)
        t.start()
        built_evt.wait()
        survivor_inst = built[0]

        w.process(_frame())  # main thread builds its own
        main_inst = built[1]
        w.release_thread_local()  # main thread frees only its own

        assert main_inst.release_calls == 1
        assert survivor_inst.release_calls == 0  # survivor untouched
        hold_evt.set()
        t.join()

    def test_release_thread_local_with_nothing_built_is_safe(self):
        w = PerWorkerProcessor(factory=_Stub, name="x")
        w.release_thread_local()  # this thread never built one — no-op
