"""Tests for the N-worker LiveLoop: shared chain across workers, throughput
scaling, ordered (latest-completed-wins, monotonic, drop-stragglers) output,
runtime worker-count changes, per-worker instance release on shrink, hot-swap
drain-before-release, stale-world frame suppression, prompt teardown.

Stubs only — no camera, no HTTP, no models.
"""
from __future__ import annotations

import random
import threading
import time

import numpy as np

from sinner2.pipeline.live.live_loop import LiveLoop
from sinner2.pipeline.realtime.per_worker import PerWorkerProcessor


def _wait_until(predicate, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _tag(frame) -> int:
    return int(frame[0, 0, 0])


class _CountingSource:
    """Yields fresh frames whose [0,0,0] is an increasing counter (capped at 250
    so the uint8 never wraps — the sequence stays non-decreasing)."""

    def __init__(self, shape=(4, 4, 3)):
        self._shape = shape
        self._n = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def read(self):
        f = np.zeros(self._shape, np.uint8)
        f[0, 0, 0] = min(self._n, 250)
        self._n += 1
        return f

    def stop(self):
        self.stopped = True


class _SpyProcessor:
    """Shared, thread-safe spy (one instance called by all workers)."""

    def __init__(self, name="spy", delta=0, sleep=0.0):
        self.name = name
        self._delta = delta
        self._sleep = sleep
        self._lock = threading.Lock()
        self.calls = 0
        self.setup_calls = 0
        self.release_calls = 0

    def setup(self):
        with self._lock:
            self.setup_calls += 1

    def release(self):
        with self._lock:
            self.release_calls += 1

    def process(self, frame):
        with self._lock:
            self.calls += 1
        if self._sleep:
            time.sleep(self._sleep)
        if self._delta:
            return (frame.astype(int) + self._delta).astype(np.uint8)
        return frame


class _SpySink:
    def __init__(self):
        self._lock = threading.Lock()
        self.pushed = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def push(self, frame):
        with self._lock:
            self.pushed.append(frame)

    def stop(self):
        self.stopped = True

    def describe(self):
        return "spy"


def test_multiple_workers_share_one_chain_instance():
    spy = _SpyProcessor("shared", delta=1)
    loop = LiveLoop(_CountingSource(), [spy], [_SpySink()], workers=4, fps=1000)
    loop.start()
    try:
        assert _wait_until(lambda: spy.calls > 8)
        assert _wait_until(lambda: loop._active_worker_count() == 4)
    finally:
        loop.stop()
    assert spy.setup_calls == 1  # one shared instance set up once, not per worker


def test_n_workers_increase_throughput():
    def run(workers):
        spy = _SpyProcessor("slow", sleep=0.02)
        loop = LiveLoop(_CountingSource(), [spy], [_SpySink()],
                        workers=workers, fps=1000)
        loop.start()
        time.sleep(0.4)
        n = loop.frames_processed
        loop.stop()
        return n

    one = run(1)
    four = run(4)
    assert one > 0
    assert four > one * 1.5  # parallel workers clear a slow chain faster


def test_output_is_monotonic_no_backward_frames():
    class _RandomSlowIdentity:
        name = "rnd"

        def setup(self):
            pass

        def release(self):
            pass

        def process(self, frame):
            time.sleep(random.uniform(0.001, 0.015))  # reorder completions
            return frame

    sink = _SpySink()
    loop = LiveLoop(_CountingSource(), [_RandomSlowIdentity()], [sink],
                    workers=4, fps=200)
    loop.start()
    time.sleep(0.3)
    loop.stop()
    tags = [_tag(f) for f in sink.pushed]
    assert len(tags) > 3
    assert tags == sorted(tags)  # never a backward (stale-straggler) frame


def test_set_worker_count_grows_and_shrinks():
    loop = LiveLoop(_CountingSource(), [_SpyProcessor()], [_SpySink()],
                    workers=1, fps=1000)
    loop.start()
    try:
        assert _wait_until(lambda: loop._active_worker_count() == 1)
        loop.set_worker_count(4)
        assert _wait_until(lambda: loop._active_worker_count() == 4)
        loop.set_worker_count(2)
        assert _wait_until(lambda: loop._active_worker_count() == 2)
    finally:
        loop.stop()


def test_set_worker_count_does_not_reload_chain():
    spy = _SpyProcessor("shared")
    loop = LiveLoop(_CountingSource(), [spy], [_SpySink()], workers=1, fps=1000)
    loop.start()
    try:
        assert _wait_until(lambda: spy.calls > 0)
        for n in (4, 2, 8, 1):
            loop.set_worker_count(n)
            assert _wait_until(lambda: loop._active_worker_count() == n)
        assert spy.setup_calls == 1   # never reloaded the shared chain
        assert spy.release_calls == 0  # not released until stop
    finally:
        loop.stop()
    assert spy.release_calls == 1


def test_shrink_releases_thread_local_instances():
    created = []

    class _Inst:
        name = "inst"

        def __init__(self):
            self.released = False
            created.append(self)

        def setup(self):
            pass

        def process(self, frame):
            return frame

        def release(self):
            self.released = True

    pw = PerWorkerProcessor(lambda: _Inst(), name="pw")
    loop = LiveLoop(_CountingSource(), [pw], [_SpySink()], workers=4, fps=1000)
    loop.start()
    try:
        assert _wait_until(lambda: len(created) >= 4)  # each worker built its own
        loop.set_worker_count(1)
        # the 3 surplus workers exit and release their own instances now
        assert _wait_until(lambda: sum(i.released for i in created) >= 3)
    finally:
        loop.stop()
    assert _wait_until(lambda: all(i.released for i in created))  # stop frees the rest


def test_hot_swap_drains_inflight_before_releasing_old():
    entered = threading.Event()
    unblock = threading.Event()

    class _Blocking:
        name = "block"

        def __init__(self):
            self.released = False

        def setup(self):
            pass

        def release(self):
            self.released = True

        def process(self, frame):
            entered.set()
            unblock.wait(5.0)
            return frame

    old = _Blocking()
    loop = LiveLoop(_CountingSource(), [old], [_SpySink()], workers=1, fps=1000)
    loop.start()
    try:
        assert entered.wait(2.0)            # a worker is inside old.process()
        loop.set_chain([_SpyProcessor("new")])
        time.sleep(0.3)                     # setup done; swap installed; draining
        assert not old.released             # NOT released while old process() runs
        unblock.set()                       # let the in-flight old frame finish
        assert _wait_until(lambda: old.released, 3.0)  # drained -> released
    finally:
        unblock.set()
        loop.stop()


def test_hot_swap_discards_stale_world_frames():
    entered = threading.Event()
    unblock = threading.Event()

    class _OldSlow:
        name = "old"

        def setup(self):
            pass

        def release(self):
            pass

        def process(self, frame):
            entered.set()
            unblock.wait(5.0)
            out = frame.copy()
            out[0, 0, 0] = 100
            return out

    class _New:
        name = "new"

        def setup(self):
            pass

        def release(self):
            pass

        def process(self, frame):
            out = frame.copy()
            out[0, 0, 0] = 200
            return out

    sink = _SpySink()
    loop = LiveLoop(_CountingSource(), [_OldSlow()], [sink], workers=1, fps=1000)
    loop.start()
    try:
        assert entered.wait(2.0)            # the one worker is stuck in old.process()
        loop.set_chain([_New()])
        # Add a worker AFTER the swap so it only ever sees new-generation frames
        # (the queued old-gen frames are dropped at the worker's entry check).
        loop.set_worker_count(2)
        assert _wait_until(lambda: any(_tag(f) == 200 for f in sink.pushed), 3.0)
        unblock.set()                       # the blocked old frame completes (stale)
        time.sleep(0.2)
    finally:
        unblock.set()
        loop.stop()
    # the only old-chain frame was the blocked one; completing after the swap it
    # is a stale-world frame and must never be shown.
    assert all(_tag(f) != 100 for f in sink.pushed)
    assert any(_tag(f) == 200 for f in sink.pushed)


def test_set_source_applies_to_active_swapper_without_rebuild():
    class _SwapSpy:
        name = "swap"

        def __init__(self):
            self.sources = []
            self.setup_calls = 0
            self.released = False

        def setup(self):
            self.setup_calls += 1

        def release(self):
            self.released = True

        def process(self, frame):
            return frame

        def set_source(self, source):
            self.sources.append(source)

    swap = _SwapSpy()
    loop = LiveLoop(_CountingSource(), [swap], [_SpySink()], workers=2, fps=1000)
    loop.start()
    try:
        assert _wait_until(lambda: swap.setup_calls == 1)  # set up once
        loop.set_source("FACE2")
        assert _wait_until(lambda: swap.sources == ["FACE2"])  # fast-applied
        assert swap.setup_calls == 1  # NOT re-setup — no chain rebuild
        assert not swap.released      # chain (enhancer/upscaler) survives
    finally:
        loop.stop()


def test_set_source_skips_processors_without_setter():
    loop = LiveLoop(_CountingSource(), [_SpyProcessor("noop")], [_SpySink()],
                    workers=1, fps=1000)
    loop.start()
    try:
        loop.set_source("X")  # processor has no set_source -> skipped, no crash
        assert _wait_until(lambda: loop.frames_processed > 0)
    finally:
        loop.stop()


def test_stop_drains_and_joins_promptly():
    class _NoneSource(_CountingSource):
        def read(self):
            return None

    loop = LiveLoop(_NoneSource(), [_SpyProcessor()], [_SpySink()],
                    workers=4, fps=1000)
    loop.start()
    time.sleep(0.05)
    done = threading.Event()
    threading.Thread(target=lambda: (loop.stop(), done.set()), daemon=True).start()
    assert done.wait(timeout=5.0), "stop() did not return promptly"


def test_throwing_processor_passthrough_still_monotonic():
    class _Flaky:
        name = "flaky"

        def setup(self):
            pass

        def release(self):
            pass

        def process(self, frame):
            raise RuntimeError("boom")

    sink = _SpySink()
    loop = LiveLoop(_CountingSource(), [_Flaky()], [sink], workers=4, fps=200)
    loop.start()
    time.sleep(0.2)
    loop.stop()
    assert loop.errors > 0
    tags = [_tag(f) for f in sink.pushed]
    assert tags                      # raw frames passed through (never blank)
    assert tags == sorted(tags)      # still monotonic despite N workers + errors
