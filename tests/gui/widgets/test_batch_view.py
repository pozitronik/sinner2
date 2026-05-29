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
        assert "75%" in text
        assert "faceenhancer" in text
        assert "5/10" in text

    def test_progress_text_derives_overall_for_reloaded_task(
        self, view, tmp_path, store
    ):
        # Paused mid stage-1 of 2 (stage 0 done): overall = 10 + 5 = 15/20.
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
        assert "75%" in view._model.item(0, _COL_PROGRESS).text()  # noqa: SLF001

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
        from PySide6.QtWidgets import QMessageBox

        t = _task(tmp_path, status=BatchTaskStatus.COMPLETED)
        store.save(t)
        view.reload_from_store()
        called: list[str] = []
        monkeypatch.setattr(
            queue, "refresh_task", lambda tid: called.append(tid)
        )
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *a, **k: QMessageBox.StandardButton.Yes,
        )
        view._reset_task_to_pending(t.id)  # noqa: SLF001
        assert called == [t.id]

    def test_reset_declined_does_not_refresh(
        self, view, tmp_path, store, queue, monkeypatch
    ):
        from PySide6.QtWidgets import QMessageBox

        t = _task(tmp_path, status=BatchTaskStatus.COMPLETED)
        store.save(t)
        view.reload_from_store()
        called: list[str] = []
        monkeypatch.setattr(
            queue, "refresh_task", lambda tid: called.append(tid)
        )
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *a, **k: QMessageBox.StandardButton.No,
        )
        view._reset_task_to_pending(t.id)  # noqa: SLF001
        assert called == []
