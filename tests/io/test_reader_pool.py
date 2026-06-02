"""Tests for ReaderPool.

Covers the parallelism + lifecycle contract: requests served by N
reader threads with N reader instances, futures resolve, cancellation
honoured, shutdown releases every reader, partial construction
failure doesn't leak."""
from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError

import numpy as np
import pytest

from sinner2.io.reader_pool import ReaderPool
from sinner2.types import Frame, FrameIndex


class _FakeReader:
    """In-memory TargetReader for tests.

    `read_delay` lets tests simulate slow I/O so the parallelism check
    can observe wall-clock speedup. `raise_on_index` makes a specific
    index throw so per-request error handling can be exercised."""

    def __init__(
        self,
        count: int = 100,
        read_delay: float = 0.0,
        raise_on_index: int | None = None,
    ) -> None:
        self._count = count
        self._delay = read_delay
        self._raise_on = raise_on_index
        self._frame = np.full((4, 4, 3), 0, dtype=np.uint8)
        self.release_calls = 0
        self.read_calls = 0
        self._lock = threading.Lock()

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def width(self) -> int:
        return 4

    @property
    def height(self) -> int:
        return 4

    @property
    def native_width(self) -> int:
        return 4

    @property
    def native_height(self) -> int:
        return 4

    def read(self, index: FrameIndex) -> Frame | None:
        with self._lock:
            self.read_calls += 1
        if self._raise_on is not None and index == self._raise_on:
            raise RuntimeError(f"simulated read failure at {index}")
        if self._delay > 0:
            time.sleep(self._delay)
        if index < 0 or index >= self._count:
            return None
        # Encode the index into the frame so tests can verify which
        # frame came back, not just "a frame".
        frame = self._frame.copy()
        frame[0, 0, 0] = index & 0xFF
        return frame

    def release(self) -> None:
        with self._lock:
            self.release_calls += 1


def _make_pool(size: int, **reader_kwargs) -> tuple[ReaderPool, list[_FakeReader]]:
    """Build a pool with `size` fresh readers; return both so tests can
    introspect per-reader counters."""
    readers: list[_FakeReader] = []

    def factory() -> _FakeReader:
        r = _FakeReader(**reader_kwargs)
        readers.append(r)
        return r

    pool = ReaderPool(factory, size=size, name="test")
    return pool, readers


class TestConstruction:
    def test_rejects_zero_size(self):
        with pytest.raises(ValueError):
            ReaderPool(_FakeReader, size=0, name="test")

    def test_size_property(self):
        pool, _ = _make_pool(4)
        try:
            assert pool.size == 4
        finally:
            pool.shutdown()

    def test_fps_and_frame_count_from_probe(self):
        pool, readers = _make_pool(2, count=42)
        try:
            assert pool.frame_count == 42
            assert pool.fps == 30.0
            # Probe reader is the first one — it's been built and is in
            # the pool, not a discarded extra.
            assert len(readers) == 2
        finally:
            pool.shutdown()

    def test_factory_failure_releases_partial_pool(self):
        # A factory that raises on the second call should leave no
        # leaked readers — the partially-built first reader must be
        # released before the exception propagates.
        built: list[_FakeReader] = []
        attempts = [0]

        def factory():
            attempts[0] += 1
            if attempts[0] >= 2:
                raise RuntimeError("simulated factory failure")
            r = _FakeReader()
            built.append(r)
            return r

        with pytest.raises(RuntimeError, match="simulated"):
            ReaderPool(factory, size=4, name="test")
        # The successfully-built reader must have been released.
        assert len(built) == 1
        assert built[0].release_calls == 1


class TestReadAsync:
    def test_read_returns_frame(self):
        pool, _ = _make_pool(1)
        try:
            future = pool.read_async(5)
            frame = future.result(timeout=1.0)
            assert frame is not None
            assert frame[0, 0, 0] == 5
        finally:
            pool.shutdown()

    def test_out_of_range_returns_none(self):
        pool, _ = _make_pool(1, count=10)
        try:
            assert pool.read_async(99).result(timeout=1.0) is None
        finally:
            pool.shutdown()

    def test_fifo_order_with_size_one(self):
        pool, _ = _make_pool(1)
        try:
            futures = [pool.read_async(i) for i in range(5)]
            results = [f.result(timeout=2.0) for f in futures]
            for i, r in enumerate(results):
                assert r[0, 0, 0] == i
        finally:
            pool.shutdown()


class TestReadLatency:
    """recent_read_latency_ms() exposes how expensive reads currently are, so
    the skip strategy can tell an I/O-bound source from a compute-bound one."""

    def test_zero_when_no_reads(self):
        pool, _ = _make_pool(1)
        try:
            assert pool.recent_read_latency_ms() == 0.0
        finally:
            pool.shutdown()

    def test_durations_deque_is_bounded(self):
        # The latency window is trimmed only when recent_read_latency_ms() is
        # called (a PLAYING-only path); reads issued while paused (seeks) would
        # otherwise grow it without bound. A maxlen caps it regardless.
        pool, _ = _make_pool(1)
        try:
            cap = pool._read_durations_ms.maxlen  # noqa: SLF001
            assert cap is not None  # bounded, not an unbounded deque
            with pool._rate_lock:  # noqa: SLF001
                for _ in range(cap + 500):
                    pool._read_durations_ms.append((0.0, 1.0))  # noqa: SLF001
            assert len(pool._read_durations_ms) == cap  # noqa: SLF001
        finally:
            pool.shutdown()

    def test_reflects_slow_reads(self):
        pool, _ = _make_pool(1, read_delay=0.05)  # 50 ms per read
        try:
            for i in range(4):
                pool.read_async(i).result(timeout=2.0)
            lat = pool.recent_read_latency_ms()
            assert 30.0 < lat < 250.0  # ~50 ms, generous for scheduler jitter
        finally:
            pool.shutdown()

    def test_fast_reads_stay_low(self):
        pool, _ = _make_pool(1)  # no delay
        try:
            for i in range(8):
                pool.read_async(i).result(timeout=2.0)
            assert pool.recent_read_latency_ms() < 20.0
        finally:
            pool.shutdown()


class TestReadsPerSecondStallDecay:
    """reads_per_second reports a decaying estimate while a slow read is in
    flight rather than a hard 0 (which is indistinguishable from a hang)."""

    def test_decays_to_small_positive_during_slow_progress(self):
        pool, _ = _make_pool(1)
        try:
            now = time.monotonic()
            with pool._rate_lock:  # noqa: SLF001
                pool._read_times.clear()  # noqa: SLF001
                pool._last_read_time = now - 5.0  # noqa: SLF001
                pool._last_read_rate = 10.0  # noqa: SLF001
            assert 0.1 < pool.reads_per_second() < 0.4
        finally:
            pool.shutdown()

    def test_zero_after_long_stall(self):
        pool, _ = _make_pool(1)
        try:
            now = time.monotonic()
            with pool._rate_lock:  # noqa: SLF001
                pool._read_times.clear()  # noqa: SLF001
                pool._last_read_time = now - 60.0  # noqa: SLF001
                pool._last_read_rate = 10.0  # noqa: SLF001
            assert pool.reads_per_second() == 0.0
        finally:
            pool.shutdown()

    def test_cold_start_decay_is_capped(self):
        # First read just happened, no prior windowed rate → cap the
        # 1/tiny-elapsed estimate instead of spiking absurdly.
        pool, _ = _make_pool(1)
        try:
            now = time.monotonic()
            with pool._rate_lock:  # noqa: SLF001
                pool._read_times.clear()  # noqa: SLF001
                pool._last_read_time = now  # noqa: SLF001  (just now)
                pool._last_read_rate = 0.0  # noqa: SLF001  (no prior rate)
            assert pool.reads_per_second() <= 300.0
        finally:
            pool.shutdown()


class TestParallelism:
    def test_size_n_parallelises(self):
        # 4 readers each sleeping 50ms: 8 requests should complete in
        # well under 200ms (which is the serial-cost lower bound).
        pool, _ = _make_pool(4, read_delay=0.05)
        try:
            start = time.monotonic()
            futures = [pool.read_async(i) for i in range(8)]
            for f in futures:
                f.result(timeout=2.0)
            elapsed = time.monotonic() - start
            # Serial: 8 * 50ms = 400ms. With 4 parallel: ~100ms + overhead.
            # Give generous headroom for CI flakiness.
            assert elapsed < 0.25, (
                f"expected parallel execution, took {elapsed:.3f}s"
            )
        finally:
            pool.shutdown()


class TestErrorHandling:
    def test_reader_exception_propagates_to_future(self):
        pool, _ = _make_pool(1, raise_on_index=3)
        try:
            with pytest.raises(RuntimeError, match="simulated read failure at 3"):
                pool.read_async(3).result(timeout=1.0)
        finally:
            pool.shutdown()

    def test_pool_survives_single_read_failure(self):
        # After a bad read, the same pool should still serve the next
        # request. This is the "reader thread doesn't die on one bad
        # read" contract that keeps slow / blippy networks usable.
        pool, _ = _make_pool(1, raise_on_index=2)
        try:
            with pytest.raises(RuntimeError):
                pool.read_async(2).result(timeout=1.0)
            ok = pool.read_async(5).result(timeout=1.0)
            assert ok is not None
            assert ok[0, 0, 0] == 5
        finally:
            pool.shutdown()


class TestShutdown:
    def test_shutdown_releases_every_reader(self):
        pool, readers = _make_pool(3)
        pool.shutdown()
        assert all(r.release_calls == 1 for r in readers), (
            f"release_calls={[r.release_calls for r in readers]}"
        )

    def test_shutdown_is_idempotent(self):
        pool, readers = _make_pool(2)
        pool.shutdown()
        pool.shutdown()  # must not raise; must not re-release
        assert all(r.release_calls == 1 for r in readers)

    def test_read_after_shutdown_returns_cancelled_future(self):
        pool, _ = _make_pool(1)
        pool.shutdown()
        future = pool.read_async(0)
        with pytest.raises(CancelledError):
            future.result(timeout=0.5)

    def test_shutdown_cancels_pending_futures(self):
        # Slow reader + many requests → queue backs up. Shutdown
        # should drain pending requests and cancel them so callers
        # don't block forever.
        pool, _ = _make_pool(1, read_delay=0.5)
        try:
            futures = [pool.read_async(i) for i in range(10)]
        finally:
            # One read may have already started; the rest queued.
            pool.shutdown()
        # At least the later-submitted requests should be cancelled.
        cancelled_count = sum(1 for f in futures if f.cancelled())
        assert cancelled_count > 0, "expected at least some pending futures to be cancelled"


class TestReadsPerSecond:
    def test_zero_when_idle(self):
        pool, _ = _make_pool(1)
        try:
            assert pool.reads_per_second() == 0.0
        finally:
            pool.shutdown()

    def test_positive_after_successful_reads(self):
        # Small read_delay ensures distinct monotonic timestamps even on
        # platforms with coarse clock granularity (Windows + tight burst
        # can otherwise collapse to span=0 → reported rate of 0.0).
        pool, _ = _make_pool(2, read_delay=0.005)
        try:
            futures = [pool.read_async(i) for i in range(10)]
            for f in futures:
                f.result(timeout=2.0)
            assert pool.reads_per_second() > 0.0
        finally:
            pool.shutdown()

    def test_failed_reads_do_not_count(self):
        # raise_on_index=0 means every read raises; rate should stay 0.
        pool, _ = _make_pool(1, raise_on_index=0)
        try:
            for i in range(5):
                fut = pool.read_async(0)
                with pytest.raises(RuntimeError):
                    fut.result(timeout=1.0)
            assert pool.reads_per_second() == 0.0
        finally:
            pool.shutdown()

    def test_decays_to_zero_after_idle(self):
        # Pump reads, then wait past BOTH the rolling window and the stall-hold
        # window — the rate must drop back to 0 (truly idle, not slow progress).
        from sinner2.io import reader_pool as rp_module

        original_window = rp_module._READ_RATE_WINDOW_S
        original_hold = rp_module._READ_STALL_HOLD_S
        rp_module._READ_RATE_WINDOW_S = 0.3
        rp_module._READ_STALL_HOLD_S = 0.3  # short so idle reaches 0 quickly
        try:
            pool, _ = _make_pool(1, read_delay=0.005)
            try:
                for i in range(5):
                    pool.read_async(i).result(timeout=1.0)
                assert pool.reads_per_second() > 0.0
                time.sleep(0.5)  # exceed the window AND the stall-hold
                assert pool.reads_per_second() == 0.0
            finally:
                pool.shutdown()
        finally:
            rp_module._READ_RATE_WINDOW_S = original_window
            rp_module._READ_STALL_HOLD_S = original_hold
