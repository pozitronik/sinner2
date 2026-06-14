"""Tests for QBatchView: table population, row updates, context menu wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.batch.queue import BatchQueue
from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.batch.task_store import BatchTaskStore
from sinner2.gui.widgets.batch_view import (
    _COL_OUTPUT,
    _COL_PROGRESS,
    _COL_SOURCE,
    _COL_STATUS,
    _COL_TARGET,
    QBatchView,
    _stage_names,
)


@pytest.fixture
def store(tmp_path: Path) -> BatchTaskStore:
    return BatchTaskStore(tmp_path / "batch")


@pytest.fixture
def queue(qtbot, tmp_path: Path, store: BatchTaskStore):
    q = BatchQueue(store=store, cache_root=tmp_path / "cache")
    yield q
    q.stop()


@pytest.fixture
def view(qtbot, store: BatchTaskStore, queue: BatchQueue):
    v = QBatchView(store=store, queue=queue)
    qtbot.addWidget(v)
    return v


def _task(tmp_path: Path, **overrides) -> BatchTask:
    kwargs = {
        "source_path": tmp_path / "src.png",
        "target_path": tmp_path / "tgt.mp4",
    }
    kwargs.update(overrides)
    return BatchTask(**kwargs)


class TestSettingsButton:
    def test_settings_button_emits_request(self, view, qtbot):
        with qtbot.waitSignal(view.settingsRequested, timeout=1000):
            view._settings_btn.click()  # noqa: SLF001


class TestInitialPopulate:
    def test_empty_store_renders_empty_table(self, view):
        assert view._model.rowCount() == 0  # noqa: SLF001

    def test_existing_tasks_load_on_construction(
        self, qtbot, tmp_path, store, queue
    ):
        for _ in range(3):
            store.save(_task(tmp_path))
        v = QBatchView(store=store, queue=queue)
        qtbot.addWidget(v)
        assert v._model.rowCount() == 3  # noqa: SLF001


class TestAppendAndReload:
    def test_append_task_adds_row(self, view, tmp_path, store):
        t = _task(tmp_path)
        store.save(t)
        view.append_task(t)
        assert view._model.rowCount() == 1  # noqa: SLF001

    def test_reload_from_store_repopulates(
        self, view, tmp_path, store
    ):
        for _ in range(2):
            store.save(_task(tmp_path))
        view.reload_from_store()
        assert view._model.rowCount() == 2  # noqa: SLF001


class TestRowContent:
    def test_source_target_columns_show_filename_only(
        self, view, tmp_path, store
    ):
        t = _task(
            tmp_path,
            source_path=tmp_path / "deep" / "face.png",
            target_path=tmp_path / "videos" / "clip.mp4",
        )
        # Save accepts non-existent paths (Path field doesn't validate
        # existence on BatchTask).
        store.save(t)
        view.reload_from_store()
        assert view._model.item(0, _COL_SOURCE).text() == "face.png"  # noqa: SLF001
        assert view._model.item(0, _COL_TARGET).text() == "clip.mp4"  # noqa: SLF001
        # Full path is in the tooltip.
        assert "face.png" in view._model.item(0, _COL_SOURCE).toolTip()  # noqa: SLF001

    def test_progress_blank_before_run(self, view, tmp_path, store):
        store.save(_task(tmp_path))
        view.reload_from_store()
        assert view._model.item(0, _COL_PROGRESS).text() == ""  # noqa: SLF001

    def test_status_reflects_task(self, view, tmp_path, store):
        store.save(_task(tmp_path, status=BatchTaskStatus.COMPLETED))
        view.reload_from_store()
        assert view._model.item(0, _COL_STATUS).text() == "completed"  # noqa: SLF001


class TestQueueSignalUpdates:
    def test_progress_signal_updates_cell(
        self, view, tmp_path, store, queue
    ):
        from sinner2.batch.task import BatchProgress

        t = _task(tmp_path)
        store.save(t)
        view.reload_from_store()
        queue.taskProgress.emit(
            t.id,
            BatchProgress(
                stage_index=1,
                stage_count=2,
                stage_name="faceenhancer",
                stage_completed=5,
                stage_total=10,
                overall_completed=15,
                overall_total=20,
            ),
        )
        text = view._model.item(0, _COL_PROGRESS).text()  # noqa: SLF001
        # Step-scoped: 5/10 of stage 2-of-2 → 50%, with the stage name.
        assert "[2/2]" in text
        assert "50%" in text
        assert "faceenhancer" in text
        assert "5/10" in text

    def test_progress_shows_fps_and_time(
        self, view, tmp_path, store, queue, monkeypatch
    ):
        from sinner2.batch.task import BatchProgress
        from sinner2.gui.widgets import batch_view

        clock = [0.0]
        monkeypatch.setattr(batch_view.time, "monotonic", lambda: clock[0])
        t = _task(tmp_path)
        store.save(t)
        view.reload_from_store()
        queue.taskStarted.emit(t.id)  # resets the step tracker
        progress = dict(
            stage_index=0,
            stage_count=2,
            stage_name="faceswapper",
            stage_total=10,
        )
        clock[0] = 0.0
        queue.taskProgress.emit(
            t.id,
            BatchProgress(stage_completed=0, overall_completed=0, overall_total=20, **progress),
        )
        clock[0] = 1.0  # 5 frames in 1s → 5 fps; remaining (10-5)/5 = 1s
        queue.taskProgress.emit(
            t.id,
            BatchProgress(stage_completed=5, overall_completed=5, overall_total=20, **progress),
        )
        text = view._model.item(0, _COL_PROGRESS).text()  # noqa: SLF001
        assert "5 fps" in text
        # Step elapsed 0:01, expected total ~0:02 (elapsed + remaining).
        assert "0:01" in text
        assert "~0:02" in text

    def test_progress_text_derives_step_for_reloaded_task(
        self, view, tmp_path, store
    ):
        # 3 stages now (swapper + enhancer + combine). Paused mid-enhancer
        # (stage 0 done, 5/10 of stage 1): step 2-of-3 at 50%.
        t = _task(
            tmp_path,
            enhancer_enabled=True,
            total_frames=10,
            completed_stages=1,
            last_completed_frame=4,
            status=BatchTaskStatus.PAUSED,
        )
        store.save(t)
        view.reload_from_store()
        text = view._model.item(0, _COL_PROGRESS).text()  # noqa: SLF001
        assert "[2/3]" in text
        assert "50%" in text
        assert "faceenhancer" in text

    def test_completed_signal_refreshes_status_from_store(
        self, view, tmp_path, store, queue
    ):
        t = _task(tmp_path)
        store.save(t)
        view.reload_from_store()
        # Simulate the queue having mutated + saved the task before
        # emitting completed (that's the real flow).
        t.status = BatchTaskStatus.COMPLETED
        store.save(t)
        queue.taskCompleted.emit(t.id)
        assert view._model.item(0, _COL_STATUS).text() == "completed"  # noqa: SLF001


class TestEditRequest:
    def test_double_click_emits_edit_requested(
        self, view, qtbot, tmp_path, store
    ):
        t = _task(tmp_path)
        store.save(t)
        view.reload_from_store()
        with qtbot.waitSignal(view.editRequested, timeout=1000) as blocker:
            view._emit_edit_for_row(0)  # noqa: SLF001
        assert blocker.args == [t.id]

    def test_double_click_does_not_edit_running_task(
        self, view, tmp_path, store, queue
    ):
        # Editing the RUNNING task races the queue's store writer + reopens its
        # resume state. The context menu hides Edit for it; the double-click path
        # must honour the same guard.
        t = _task(tmp_path)
        store.save(t)
        view.reload_from_store()
        queue._current_task_id = t.id  # noqa: SLF001  this task is "running"
        triggered: list[str] = []
        view.editRequested.connect(triggered.append)
        view._emit_edit_for_row(0)  # noqa: SLF001
        assert triggered == []


class TestOutputColumnRespectsGlobalDir:
    def test_output_uses_global_dir_when_resolver_returns_path(
        self, qtbot, tmp_path, store, queue
    ):
        # The view uses the resolver each time it builds a row; verify
        # a non-None resolver shows the global-dir path.
        t = _task(tmp_path)
        store.save(t)
        global_dir = tmp_path / "global_out"
        v = QBatchView(
            store=store,
            queue=queue,
            global_output_dir_resolver=lambda: global_dir,
        )
        qtbot.addWidget(v)
        # Auto-name is "src+tgt.mp4" (video output default).
        assert global_dir.name in v._model.item(0, _COL_OUTPUT).toolTip()  # noqa: SLF001


class TestResetToPending:
    def test_reset_confirmed_calls_refresh(
        self, view, tmp_path, store, queue, monkeypatch
    ):
        t = _task(tmp_path, status=BatchTaskStatus.COMPLETED)
        store.save(t)
        view.reload_from_store()
        called: list[str] = []
        monkeypatch.setattr(
            queue, "refresh_task", lambda tid: called.append(tid)
        )
        monkeypatch.setattr(
            "sinner2.gui.widgets.batch_view.confirm", lambda *a, **k: True
        )
        view._reset_task_to_pending(t.id)  # noqa: SLF001
        assert called == [t.id]

    def test_reset_declined_does_not_refresh(
        self, view, tmp_path, store, queue, monkeypatch
    ):
        t = _task(tmp_path, status=BatchTaskStatus.COMPLETED)
        store.save(t)
        view.reload_from_store()
        called: list[str] = []
        monkeypatch.setattr(
            queue, "refresh_task", lambda tid: called.append(tid)
        )
        monkeypatch.setattr(
            "sinner2.gui.widgets.batch_view.confirm", lambda *a, **k: False
        )
        view._reset_task_to_pending(t.id)  # noqa: SLF001
        assert called == []


class TestFailureSurfacing:
    def test_failed_task_status_tooltip_shows_error(
        self, view, tmp_path, store
    ):
        t = _task(
            tmp_path,
            status=BatchTaskStatus.FAILED,
            error_message="boom: 5 frames missing",
        )
        store.save(t)
        view.reload_from_store()
        tip = view._model.item(0, _COL_STATUS).toolTip()  # noqa: SLF001
        assert tip == "boom: 5 frames missing"


class TestResumeAction:
    def test_resume_calls_queue_resume(
        self, view, tmp_path, store, queue, monkeypatch
    ):
        t = _task(tmp_path, status=BatchTaskStatus.PAUSED)
        store.save(t)
        view.reload_from_store()
        called: list[str] = []
        monkeypatch.setattr(
            queue, "resume_task", lambda tid: called.append(tid)
        )
        view._resume_task(t.id)  # noqa: SLF001
        assert called == [t.id]


class TestStepTracker:
    def test_fmt_eta_formats(self):
        from sinner2.gui.widgets.batch_view import _fmt_eta

        assert _fmt_eta(30) == "0:30"
        assert _fmt_eta(65) == "1:05"
        assert _fmt_eta(3661) == "1:01:01"

    def test_window_fps_elapsed_and_expected(self, monkeypatch):
        from sinner2.gui.widgets import batch_view

        clock = [0.0]
        monkeypatch.setattr(batch_view.time, "monotonic", lambda: clock[0])
        tracker = batch_view._StepTracker()
        fps, elapsed, expected = tracker.update(0, 0, 100)
        # One sample → no rate yet; elapsed is from the step start.
        assert fps == 0.0 and elapsed == 0.0 and expected is None
        clock[0] = 1.0
        fps, elapsed, expected = tracker.update(0, 10, 100)
        assert fps == pytest.approx(10.0)  # 10 frames in 1s
        assert elapsed == pytest.approx(1.0)
        # remaining = (100 - 10) / 10 = 9s; expected = elapsed + remaining.
        assert expected == pytest.approx(10.0)

    def test_new_step_resets_clock_and_rate(self, monkeypatch):
        from sinner2.gui.widgets import batch_view

        clock = [0.0]
        monkeypatch.setattr(batch_view.time, "monotonic", lambda: clock[0])
        tracker = batch_view._StepTracker()
        tracker.update(0, 0, 100)
        clock[0] = 5.0
        tracker.update(0, 100, 100)  # stage 0 finishes at t=5
        # Stage 1 begins at the same wall-clock — elapsed must restart at 0
        # and the rate window must drop stage 0's samples.
        fps, elapsed, expected = tracker.update(1, 0, 100)
        assert elapsed == 0.0
        assert fps == 0.0
        assert expected is None

    def test_no_expected_when_idle(self, monkeypatch):
        from sinner2.gui.widgets import batch_view

        clock = [0.0]
        monkeypatch.setattr(batch_view.time, "monotonic", lambda: clock[0])
        tracker = batch_view._StepTracker()
        tracker.update(0, 50, 100)
        clock[0] = 1.0
        # No progress → fps 0, no expected; elapsed still advances.
        fps, elapsed, expected = tracker.update(0, 50, 100)
        assert fps == 0.0
        assert expected is None
        assert elapsed == pytest.approx(1.0)


class TestStageNames:
    """_stage_names must mirror BatchDriver's progress stages exactly: the
    processor stages (incl. upscaler, once omitted) plus the trailing
    combine/encode step (so reloaded tasks show the same stage count/labels the
    live signal would)."""

    def test_includes_upscaler_stage(self, tmp_path):
        names = _stage_names(_task(
            tmp_path, swapper_enabled=True, enhancer_enabled=False,
            upscaler_enabled=True, output_format=BatchOutputFormat.VIDEO,
        ))
        assert names == ["faceswapper", "upscaler", "encode"]

    def test_full_order_matches_driver(self, tmp_path):
        names = _stage_names(_task(
            tmp_path, swapper_enabled=True, enhancer_enabled=True,
            upscaler_enabled=True, output_format=BatchOutputFormat.VIDEO,
        ))
        assert names == ["faceswapper", "faceenhancer", "upscaler", "encode"]

    def test_upscaler_only_is_not_passthrough(self, tmp_path):
        names = _stage_names(_task(
            tmp_path, swapper_enabled=False, enhancer_enabled=False,
            upscaler_enabled=True, output_format=BatchOutputFormat.VIDEO,
        ))
        assert names == ["upscaler", "encode"]

    def test_frames_output_combine_step_is_copy(self, tmp_path):
        names = _stage_names(_task(
            tmp_path, swapper_enabled=True, enhancer_enabled=False,
            output_format=BatchOutputFormat.FRAMES,
        ))
        assert names == ["faceswapper", "copy"]

    def test_both_disabled_is_passthrough_then_combine(self, tmp_path):
        names = _stage_names(_task(
            tmp_path, swapper_enabled=False, enhancer_enabled=False,
            upscaler_enabled=False, output_format=BatchOutputFormat.VIDEO,
        ))
        assert names == ["passthrough", "encode"]


class TestStepTrackerReuse:
    def test_tracker_not_reallocated_each_progress_tick(
        self, view, tmp_path, store, queue, monkeypatch
    ):
        # rank 40: the per-task _StepTracker (a deque) must be built once, not
        # eagerly constructed-and-discarded on every progress tick via setdefault.
        from sinner2.batch.task import BatchProgress
        from sinner2.gui.widgets import batch_view

        t = _task(tmp_path)
        store.save(t)
        view.reload_from_store()

        real = batch_view._StepTracker
        count = {"n": 0}

        def counting(*a, **k):
            count["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(batch_view, "_StepTracker", counting)
        queue.taskStarted.emit(t.id)  # builds the one tracker
        prog = dict(
            stage_index=0, stage_count=1, stage_name="faceswapper",
            stage_total=10, overall_total=10,
        )
        queue.taskProgress.emit(
            t.id, BatchProgress(stage_completed=1, overall_completed=1, **prog)
        )
        queue.taskProgress.emit(
            t.id, BatchProgress(stage_completed=2, overall_completed=2, **prog)
        )
        assert count["n"] == 1  # ticks reuse the tracker, don't reallocate
