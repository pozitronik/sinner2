import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from sinner2.config.target import Target
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import DiskFrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import BestEffortStrategy
from sinner2.types import Frame


class _CountingProcessor:
    """Returns the input frame unchanged; counts setup/process/release calls."""

    name = "Counting"

    def __init__(self) -> None:
        self.setup_calls = 0
        self.process_calls = 0
        self.release_calls = 0
        self._lock = threading.Lock()

    def setup(self) -> None:
        self.setup_calls += 1

    def process(self, frame: Frame) -> Frame:
        with self._lock:
            self.process_calls += 1
        return frame

    def release(self) -> None:
        self.release_calls += 1


def _factory(*processors):
    """Build a chain factory that returns the given processor instances.

    With worker_count=1 the factory is called once and the test keeps direct
    references to the processors for assertions.
    """
    return lambda: list(processors)


class _MultiFrameReader:
    """In-memory TargetReader for an N-frame synthetic stream."""

    def __init__(self, count: int, fps: float = 30.0) -> None:
        self._count = count
        self._fps = fps
        self._frame = np.full((8, 8, 3), 128, dtype=np.uint8)
        self.release_calls = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._count

    def read(self, index: int) -> Frame | None:
        if index < 0 or index >= self._count:
            return None
        return self._frame

    def release(self) -> None:
        self.release_calls += 1


@pytest.fixture
def buffer_setup(tmp_path: Path):
    store = DiskFrameStore(tmp_path / "frames")
    cache = MemoryFrameCache(max_bytes=10 * 1024 * 1024)
    timeline = Timeline(fps=30.0)
    write_executor = ThreadPoolExecutor(max_workers=2)
    buffer = FrameBuffer(store, cache, timeline, write_executor)
    yield buffer, timeline, write_executor
    write_executor.shutdown(wait=True)


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll until predicate is truthy or timeout. Returns whether it became true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestRealtimeExecutorLifecycle:
    def test_rejects_zero_workers(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        with pytest.raises(ValueError):
            RealtimeExecutor(
                target_reader=_MultiFrameReader(1),
                buffer=buffer,
                timeline=timeline,
                chain=[],
                strategy=BestEffortStrategy(),
                worker_count=0,
            )

    def test_start_calls_setup_on_each_processor(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            assert p.setup_calls == 1
        finally:
            ex.stop()

    def test_stop_calls_release_on_each_processor(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        ex.stop()
        assert p.release_calls == 1

    def test_stop_releases_target_reader(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        reader = _MultiFrameReader(1)
        ex = RealtimeExecutor(
            target_reader=reader,
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        ex.stop()
        assert reader.release_calls == 1

    def test_double_start_is_noop(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.start()
            assert p.setup_calls == 1
        finally:
            ex.stop()

    def test_double_stop_is_noop(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1),
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        ex.stop()
        ex.stop()

    def test_all_workers_share_one_chain(self, buffer_setup):
        # Sinner1 model: ONE shared chain across N workers. ORT
        # InferenceSession is thread-safe so concurrent .run() calls let
        # the GPU schedule them efficiently. Per-worker chains were tried
        # and were slower because of CUDA context contention.
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
            worker_count=4,
        )
        try:
            ex.start()
            assert p.setup_calls == 1  # only one setup regardless of worker_count
        finally:
            ex.stop()
        assert p.release_calls == 1


class TestRealtimeExecutorPlayback:
    def test_play_advances_through_target(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(5, fps=100.0),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: p.process_calls >= 5, timeout=2.0)
        finally:
            ex.stop()

    def test_pause_stops_advancing(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1000, fps=100.0),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: p.process_calls > 0)
            ex.pause()
            time.sleep(0.05)
            snapshot = p.process_calls
            time.sleep(0.1)
            assert p.process_calls == snapshot
        finally:
            ex.stop()

    def test_end_of_target_pauses(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(3, fps=100.0),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: ex.status.get() == "end of target", timeout=2.0)
            assert ex.is_playing.get() is False
        finally:
            ex.stop()

    def test_seek_repositions(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        target = Timeline(fps=100.0)
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1000),
            buffer=buffer,
            timeline=target,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.seek(500)
            assert _wait_until(lambda: target.current_frame() >= 500)
        finally:
            ex.stop()

    def test_seek_while_paused_processes_target_frame(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        target_timeline = Timeline(fps=100.0)
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1000),
            buffer=buffer,
            timeline=target_timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            # No play() — we are in IDLE/PAUSED. A bare seek must still
            # cause the worker to process the target frame so the user
            # gets visual feedback from the scrub.
            assert p.process_calls == 0
            ex.seek(42)
            assert _wait_until(lambda: p.process_calls >= 1, timeout=2.0)
        finally:
            ex.stop()


class TestRealtimeExecutorFrameDelivery:
    def test_on_frame_ready_fires_after_processing(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        delivered: list[tuple[int, Frame]] = []
        lock = threading.Lock()

        def on_frame(f: Frame, i: int) -> None:
            with lock:
                delivered.append((i, f))

        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(3, fps=100.0),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        ex.on_frame_ready(on_frame)
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: len(delivered) > 0, timeout=2.0)
        finally:
            ex.stop()


class TestRealtimeExecutorObservables:
    def test_is_playing_reflects_play_pause(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1000),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            assert ex.is_playing.get() is False
            ex.play()
            assert _wait_until(lambda: ex.is_playing.get() is True)
            ex.pause()
            assert _wait_until(lambda: ex.is_playing.get() is False)
        finally:
            ex.stop()

    def test_processing_fps_updates(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(20, fps=100.0),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: ex.processing_fps.get() > 0.0, timeout=2.0)
        finally:
            ex.stop()


class TestRealtimeExecutorSetChain:
    def test_set_chain_releases_old_and_sets_up_new(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        old = _CountingProcessor()
        new = _CountingProcessor()
        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(1),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[old],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            assert old.setup_calls == 1
            assert new.setup_calls == 0
            ex.set_chain([new])
            assert _wait_until(lambda: new.setup_calls == 1)
            assert _wait_until(lambda: old.release_calls == 1)
        finally:
            ex.stop()


class TestRealtimeExecutorWorkerError:
    def test_worker_error_stops_executor(self, buffer_setup):
        buffer, timeline, _ = buffer_setup

        class Boom:
            name = "Boom"
            def setup(self) -> None: ...
            def process(self, frame: Frame) -> Frame:
                raise RuntimeError("boom")
            def release(self) -> None: ...

        ex = RealtimeExecutor(
            target_reader=_MultiFrameReader(10, fps=100.0),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[Boom()],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: "worker error" in ex.status.get(), timeout=2.0)
        finally:
            ex.stop()


class TestRealtimeExecutorWithImageTarget:
    def test_image_target_plays_one_frame_and_pauses(self, buffer_setup, tmp_path: Path):
        buffer, timeline, _ = buffer_setup
        img_path = tmp_path / "img.png"
        cv2.imwrite(str(img_path), np.full((16, 16, 3), 200, dtype=np.uint8))
        reader = ImageTargetReader(Target(path=img_path))
        assert isinstance(reader, TargetReader)

        ex = RealtimeExecutor(
            target_reader=reader,
            buffer=buffer,
            timeline=Timeline(fps=1.0),
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: ex.status.get() == "end of target", timeout=2.0)
        finally:
            ex.stop()
