"""Tests for the BatchQueue scheduler."""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sinner2.batch.driver import BatchDriver
from sinner2.batch.queue import BatchQueue
from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.batch.task_store import BatchTaskStore
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.types import Frame


class _PassthroughProcessor:
    name = "Pass"

    def __init__(self) -> None:
        self.setup_calls = 0
        self.process_calls = 0
        self.release_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1

    def process(self, frame: Frame) -> Frame:
        self.process_calls += 1
        return frame

    def release(self) -> None:
        self.release_calls += 1


@pytest.fixture
def stub_chain(monkeypatch):
    procs: list[_PassthroughProcessor] = []

    def fake_build(_source, task):
        p = _PassthroughProcessor()
        procs.append(p)
        return [p]

    monkeypatch.setattr(BatchDriver, "_build_chain", staticmethod(fake_build))
    return procs


def _make_image(path: Path, w: int = 16, h: int = 16) -> Path:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _make_task(tmp_path: Path, suffix: str) -> BatchTask:
    return BatchTask(
        source_path=_make_image(tmp_path / f"src_{suffix}.png"),
        target_path=_make_image(tmp_path / f"tgt_{suffix}.png"),
        output_path=tmp_path / f"out_{suffix}",
        output_format=BatchOutputFormat.FRAMES,
        worker_count=1,
        image_format=ImageFormat.JPEG,
        image_quality=80,
    )


@pytest.fixture
def store(tmp_path: Path) -> BatchTaskStore:
    return BatchTaskStore(tmp_path / "batch")


@pytest.fixture
def queue(qtbot, tmp_path: Path, store: BatchTaskStore):
    q = BatchQueue(store=store, cache_root=tmp_path / "cache")
    yield q
    q.stop()


def _wait_signal(qtbot, signal, timeout: float = 5.0):
    """Helper: wait for a signal with a timeout, return True/False."""
    with qtbot.waitSignal(signal, timeout=int(timeout * 1000)) as blocker:
        pass
    return blocker.signal_triggered


class TestSingleTask:
    def test_runs_to_completion(
        self, qtbot, queue, store, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path, "a")
        store.save(task)
        with qtbot.waitSignal(queue.queueIdle, timeout=5000):
            queue.start()
        # Task moved to COMPLETED in the store.
        assert store.load(task.id).status is BatchTaskStatus.COMPLETED
        assert task.output_path.is_dir()


class TestSequentialOrder:
    def test_processes_tasks_one_at_a_time(
        self, qtbot, queue, store, stub_chain, tmp_path
    ):
        tasks = [_make_task(tmp_path, str(i)) for i in range(3)]
        for t in tasks:
            store.save(t)
        with qtbot.waitSignal(queue.queueIdle, timeout=10_000):
            queue.start()
        # All three landed at COMPLETED.
        for t in tasks:
            assert store.load(t.id).status is BatchTaskStatus.COMPLETED


class TestRefreshTask:
    def test_resets_completed_task_to_pending(
        self, qtbot, queue, store, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path, "a")
        store.save(task)
        with qtbot.waitSignal(queue.queueIdle, timeout=5000):
            queue.start()
        assert store.load(task.id).status is BatchTaskStatus.COMPLETED
        queue.refresh_task(task.id)
        reloaded = store.load(task.id)
        assert reloaded.status is BatchTaskStatus.PENDING
        assert reloaded.last_completed_frame == -1

    def test_refresh_wipes_cache_so_rerun_reprocesses(
        self, qtbot, queue, store, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path, "a")
        store.save(task)
        with qtbot.waitSignal(queue.queueIdle, timeout=5000):
            queue.start()
        task_cache = queue._cache_root / task.id  # noqa: SLF001
        assert task_cache.exists()
        assert len(list(task_cache.glob("*.jpg"))) >= 1
        queue.refresh_task(task.id)
        # Cache discarded → a re-run reprocesses from scratch rather than
        # short-circuiting to re-encode.
        assert not task_cache.exists()
        reloaded = store.load(task.id)
        assert reloaded.status is BatchTaskStatus.PENDING
        assert reloaded.last_completed_frame == -1
        assert reloaded.total_frames == -1

    def test_refresh_on_unknown_id_is_noop(self, queue):
        queue.refresh_task("does-not-exist")  # must not raise


class TestQueuePause:
    def test_pause_then_start_resumes_scheduling(
        self, qtbot, queue, store, stub_chain, tmp_path
    ):
        # Two tasks; pause the queue before starting, then start —
        # both should run normally (pause was a no-op before start).
        for s in ("a", "b"):
            store.save(_make_task(tmp_path, s))
        queue.pause()
        queue.start()  # clears the paused flag and schedules
        with qtbot.waitSignal(queue.queueIdle, timeout=5000):
            pass


class TestStopShutsDown:
    def test_stop_with_no_running_task_is_noop(self, queue):
        queue.stop()
        assert not queue.is_running

    def test_stop_idempotent(self, queue):
        queue.stop()
        queue.stop()
        assert not queue.is_running
