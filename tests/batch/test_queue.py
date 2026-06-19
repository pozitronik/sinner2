"""Tests for the BatchQueue scheduler."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sinner2.batch.driver import BatchDriver, StageSpec
from sinner2.batch.queue import BatchQueue
from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.batch.task_store import BatchTaskStore
from sinner2.config.execution import OnnxExecution
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
        return [
            StageSpec("faceswapper", lambda _p=p: _p, True,
                      task.swapper_execution.workers)
        ]

    monkeypatch.setattr(BatchDriver, "_build_stages", staticmethod(fake_build))
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
        swapper_execution=OnnxExecution(workers=1),
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


class TestGlobalOutputDir:
    def test_set_global_output_dir_updates_value(self, queue, tmp_path):
        out = tmp_path / "global_out"
        queue.set_global_output_dir(out)
        assert queue._global_output_dir == out  # noqa: SLF001
        queue.set_global_output_dir(None)
        assert queue._global_output_dir is None  # noqa: SLF001


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
        assert len(list(task_cache.rglob("*.jpg"))) >= 1  # frames live in stage dirs
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


class TestProgressSignal:
    def test_emits_batch_progress_reaching_total(
        self, qtbot, queue, store, stub_chain, tmp_path
    ):
        from sinner2.batch.task import BatchProgress

        task = _make_task(tmp_path, "a")
        store.save(task)
        received: list[BatchProgress] = []
        queue.taskProgress.connect(lambda _tid, p: received.append(p))
        with qtbot.waitSignal(queue.queueIdle, timeout=5000):
            queue.start()
        assert received
        assert all(isinstance(p, BatchProgress) for p in received)
        assert received[-1].overall_completed == received[-1].overall_total


class TestResumeTask:
    def test_resume_failed_reruns_to_completion(
        self, qtbot, queue, store, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path, "a")
        task.status = BatchTaskStatus.FAILED
        task.error_message = "boom"
        store.save(task)
        with qtbot.waitSignal(queue.queueIdle, timeout=5000):
            queue.resume_task(task.id)
        reloaded = store.load(task.id)
        assert reloaded.status is BatchTaskStatus.COMPLETED
        assert reloaded.error_message is None

    def test_resume_completed_task_is_noop(self, queue, store, tmp_path):
        task = _make_task(tmp_path, "b")
        task.status = BatchTaskStatus.COMPLETED
        store.save(task)
        queue.resume_task(task.id)  # not a resumable state
        assert store.load(task.id).status is BatchTaskStatus.COMPLETED


class TestStopPausesNotCancels:
    """App shutdown (BatchQueue.stop) must PAUSE the running task — leaving its
    rendered frames on disk so it resumes — NOT cancel it, which rmtree's the
    whole per-task cache and resets resume markers (data loss)."""

    def test_stop_pauses_running_task_does_not_cancel(self, queue):
        from unittest.mock import MagicMock
        driver = MagicMock()  # hold a ref — stop() nulls queue._driver
        queue._driver = driver  # noqa: SLF001
        queue._thread = MagicMock()  # noqa: SLF001
        queue._worker = MagicMock()  # noqa: SLF001
        queue._current_task_id = "abc"  # noqa: SLF001
        queue.stop()
        driver.pause.assert_called_once()   # resumable
        driver.cancel.assert_not_called()   # no cache wipe


class TestFailureHaltsQueue:
    """A FAILED task halts the queue by default (so the user sees the error and
    can recover without restarting) — unless it opted into continue-on-error,
    in which case the queue auto-skips it and rolls on to the next pending task.
    Driven at the _on_completed level (the stub chain always succeeds)."""

    def _arm(self, queue, task):
        queue._current_task = task  # noqa: SLF001
        queue._worker = None  # noqa: SLF001
        queue._thread = None  # noqa: SLF001
        queue._paused = False  # noqa: SLF001
        scheduled: list[bool] = []
        queue._schedule_next = lambda: scheduled.append(True)  # noqa: SLF001
        idle: list[bool] = []
        queue.queueIdle.connect(lambda: idle.append(True))
        return scheduled, idle

    def test_failure_without_continue_halts_and_pauses(self, queue, store, tmp_path):
        task = _make_task(tmp_path, "a")
        task.continue_on_error = False
        store.save(task)
        scheduled, idle = self._arm(queue, task)
        queue._on_completed(task.id, BatchTaskStatus.FAILED.value)  # noqa: SLF001
        assert scheduled == []          # did NOT roll on
        assert idle == [True]           # went idle (lock releases)
        assert queue._paused is True    # noqa: SLF001  halted

    def test_failure_with_continue_schedules_next(self, queue, store, tmp_path):
        task = _make_task(tmp_path, "a")
        task.continue_on_error = True
        store.save(task)
        scheduled, idle = self._arm(queue, task)
        queue._on_completed(task.id, BatchTaskStatus.FAILED.value)  # noqa: SLF001
        assert scheduled == [True]      # auto-skipped → next pending
        assert queue._paused is False   # noqa: SLF001  queue keeps running

    def test_failure_emits_taskfailed_with_message(self, queue, store, tmp_path):
        task = _make_task(tmp_path, "a")
        task.error_message = "boom on target frame 3"
        store.save(task)
        failures: list[tuple[str, str]] = []
        queue.taskFailed.connect(lambda tid, msg: failures.append((tid, msg)))
        queue._current_task = task  # noqa: SLF001
        queue._worker = None  # noqa: SLF001
        queue._thread = None  # noqa: SLF001
        queue._on_completed(task.id, BatchTaskStatus.FAILED.value)  # noqa: SLF001
        assert failures == [(task.id, "boom on target frame 3")]

    def test_completed_task_still_schedules_next(self, queue, store, tmp_path):
        task = _make_task(tmp_path, "a")
        store.save(task)
        scheduled, idle = self._arm(queue, task)
        queue._on_completed(task.id, BatchTaskStatus.COMPLETED.value)  # noqa: SLF001
        assert scheduled == [True]      # success rolls on
        assert queue._paused is False   # noqa: SLF001


class TestPauseTaskDoesNotAutostartNext:
    """Pausing the current task (pause_task) must stop the queue, NOT roll on to
    the next pending task — PAUSED was treated like COMPLETED and scheduled the
    next task."""

    def test_paused_terminal_does_not_schedule_next(self, queue):
        queue._paused = False  # noqa: SLF001  (single-task pause leaves queue unpaused)
        queue._worker = None  # noqa: SLF001
        queue._thread = None  # noqa: SLF001
        queue._current_task_id = "abc"  # noqa: SLF001
        scheduled: list[bool] = []
        queue._schedule_next = lambda: scheduled.append(True)  # noqa: SLF001
        idle: list[bool] = []
        queue.queueIdle.connect(lambda: idle.append(True))
        queue._on_completed("abc", BatchTaskStatus.PAUSED.value)  # noqa: SLF001
        assert scheduled == []   # did NOT auto-start the next task
        assert idle == [True]    # went idle instead


class TestDeleteTaskCache:
    """delete_task_cache frees a task's intermediate frames without touching
    its status/output, and resets the resume markers so a re-run is clean."""

    def test_wipes_dir_and_resets_markers_keeps_status(self, tmp_path):
        store = BatchTaskStore(tmp_path / "store")
        cache_root = tmp_path / "cache"
        q = BatchQueue(store=store, cache_root=cache_root)
        try:
            t = BatchTask(
                source_path=tmp_path / "s.png",
                target_path=tmp_path / "t.mp4",
                status=BatchTaskStatus.COMPLETED,
                completed_stages=2,
                last_completed_frame=99,
                total_frames=100,
                cache_fingerprint="abc",
            )
            store.save(t)
            cache_dir = cache_root / t.id
            (cache_dir / "stage0").mkdir(parents=True)
            (cache_dir / "stage0" / "00000000.jpg").write_bytes(b"x" * 10)

            assert q.delete_task_cache(t.id) is True
            assert not cache_dir.exists()
            reloaded = store.load(t.id)
            # Output-bearing status + total are kept; resume markers reset.
            assert reloaded.status is BatchTaskStatus.COMPLETED
            assert reloaded.total_frames == 100
            assert reloaded.completed_stages == 0
            assert reloaded.last_completed_frame == -1
            assert reloaded.cache_fingerprint == ""
        finally:
            q.stop()

    def test_noop_on_running_task(self, tmp_path):
        store = BatchTaskStore(tmp_path / "store")
        q = BatchQueue(store=store, cache_root=tmp_path / "cache")
        try:
            t = BatchTask(
                source_path=tmp_path / "s.png", target_path=tmp_path / "t.mp4",
            )
            store.save(t)
            q._current_task_id = t.id  # noqa: SLF001 — pretend it's running
            assert q.delete_task_cache(t.id) is False
        finally:
            q.stop()
