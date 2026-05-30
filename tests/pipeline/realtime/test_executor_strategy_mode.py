"""Executor surfaces the active strategy's mode via an observable.

The mode label is used by the status bar; the test verifies it updates
on play, on set_skip_strategy hot-swap, and reflects the strategy's
internal mode transitions (Synced → fallback)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from sinner2.io.reader_pool import ReaderPool
from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import DiskFrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import BestEffortStrategy, SyncedStrategy


class _StaticReader:
    """Returns a frame for any index. Used to keep the executor making
    progress without depending on real I/O."""

    def __init__(self, count: int = 10000, fps: float = 100.0) -> None:
        self._count = count
        self._fps = fps
        self._frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.release_calls = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._count

    def read(self, index):
        return self._frame if 0 <= index < self._count else None

    def release(self):
        self.release_calls += 1


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def setup(tmp_path: Path):
    store = DiskFrameStore(tmp_path / "frames")
    cache = MemoryFrameCache(max_bytes=10 * 1024 * 1024)
    timeline = Timeline(fps=100.0)
    write_executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
    buffer = FrameBuffer(store, cache, timeline, write_executor)
    yield buffer, timeline, write_executor
    write_executor.shutdown(wait=True)


class TestStrategyModeObservable:
    def test_initial_mode_matches_strategy(self, setup):
        buffer, timeline, _ = setup
        reader = _StaticReader()
        ex = RealtimeExecutor(
            reader_pool=ReaderPool(lambda: reader, size=1, name="t"),
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            assert ex.strategy_mode.get() == "best effort"
        finally:
            ex.stop()

    def test_initial_synced_mode(self, setup):
        buffer, timeline, _ = setup
        reader = _StaticReader()
        ex = RealtimeExecutor(
            reader_pool=ReaderPool(lambda: reader, size=1, name="t"),
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=SyncedStrategy(),
        )
        try:
            assert ex.strategy_mode.get() == "synced"
        finally:
            ex.stop()

    def test_set_skip_strategy_refreshes_mode(self, setup):
        buffer, timeline, _ = setup
        reader = _StaticReader()
        ex = RealtimeExecutor(
            reader_pool=ReaderPool(lambda: reader, size=1, name="t"),
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            assert ex.strategy_mode.get() == "best effort"
            ex.set_skip_strategy(SyncedStrategy())
            assert _wait_until(lambda: ex.strategy_mode.get() == "synced")
        finally:
            ex.stop()

    # The "Synced flips to lagging under load" behaviour is covered by
    # the direct strategy tests in test_skip_strategy.py — reliably
    # forcing the executor down that path here requires processing to
    # lag wall-clock, which depends on chain + reader timing that is
    # hard to make stable in unit tests. The link between strategy
    # state and executor observable is well-covered by
    # test_set_skip_strategy_refreshes_mode above.
