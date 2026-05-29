"""Tests for QBatchTaskDialog: pre-fill from task, OK writes back,
runtime state preserved, and the auto-derived output-path behavior."""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.config.execution import OnnxExecution
from sinner2.gui.widgets.batch_task_dialog import QBatchTaskDialog
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.image_writer import ImageFormat


def _task(tmp_path: Path, **overrides) -> BatchTask:
    kwargs = {
        "source_path": tmp_path / "src.png",
        "target_path": tmp_path / "tgt.mp4",
    }
    kwargs.update(overrides)
    return BatchTask(**kwargs)


class TestPrefill:
    def test_dialog_fields_match_task(self, qtbot, tmp_path):
        t = _task(
            tmp_path,
            output_path=tmp_path / "custom.mp4",
            output_format=BatchOutputFormat.FRAMES,
            swapper_detection_interval=5,
            swapper_many_faces=False,
            swapper_target_sex="F",
            enhancer_enabled=False,
            enhancer_upscale=4,
            enhancer_only_center_face=True,
            swapper_execution=OnnxExecution(workers=8),
            video_backend=VideoBackend.CV2,
            reader_pool_size=4,
            image_format=ImageFormat.PNG,
            image_quality=80,
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._source_edit.text() == str(t.source_path)  # noqa: SLF001
        assert dlg._target_edit.text() == str(t.target_path)  # noqa: SLF001
        # Explicit override shown verbatim (not replaced by the auto value).
        assert dlg._output_edit.text() == str(t.output_path)  # noqa: SLF001
        assert dlg._format_combo.currentData() == "frames"  # noqa: SLF001
        assert dlg._detection_interval.value() == 5  # noqa: SLF001
        assert dlg._many_faces.isChecked() is False  # noqa: SLF001
        assert dlg._target_sex.currentData() == "F"  # noqa: SLF001
        assert dlg._enhancer_box.isChecked() is False  # noqa: SLF001
        assert dlg._upscale.value() == 4  # noqa: SLF001
        assert dlg._only_center_face.isChecked() is True  # noqa: SLF001
        assert dlg._swapper_workers.value() == 8  # noqa: SLF001
        assert dlg._video_backend.currentData() == "cv2"  # noqa: SLF001
        assert dlg._reader_pool_size.value() == 4  # noqa: SLF001

    def test_empty_output_path_prefills_resolved_default(
        self, qtbot, tmp_path
    ):
        # No override → field shows the auto path next to the target:
        # <target.parent>/<source_stem>+<target_stem>.mp4
        t = _task(tmp_path, output_path=None)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._output_edit.text() == str(  # noqa: SLF001
            tmp_path / "src+tgt.mp4"
        )

    def test_global_output_dir_used_in_default(self, qtbot, tmp_path):
        t = _task(tmp_path, output_path=None)
        outdir = tmp_path / "renders"
        dlg = QBatchTaskDialog.from_task(t, global_output_dir=outdir)
        qtbot.addWidget(dlg)
        assert dlg._output_edit.text() == str(  # noqa: SLF001
            outdir / "src+tgt.mp4"
        )


class TestDialogSizing:
    def test_minimum_width_is_readable(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path))
        qtbot.addWidget(dlg)
        # Was auto-sizing too narrow to read file paths.
        assert dlg.minimumWidth() >= 560


class TestAutoOutputBehavior:
    def test_untouched_auto_output_persists_as_none(self, qtbot, tmp_path):
        # Field shows a value, but since the user didn't override it the
        # task must keep output_path=None so it stays auto-derived.
        t = _task(tmp_path, output_path=None)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._output_edit.text() != ""  # noqa: SLF001
        assert dlg.to_task().output_path is None

    def test_custom_output_path_is_kept(self, qtbot, tmp_path):
        t = _task(tmp_path, output_path=None)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        custom = tmp_path / "my" / "out.mp4"
        dlg._output_edit.setText(str(custom))  # noqa: SLF001
        assert dlg.to_task().output_path == custom

    def test_default_tracks_source_change_when_untouched(
        self, qtbot, tmp_path
    ):
        t = _task(tmp_path, output_path=None)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._source_edit.setText(str(tmp_path / "hero.png"))  # noqa: SLF001
        assert dlg._output_edit.text() == str(  # noqa: SLF001
            tmp_path / "hero+tgt.mp4"
        )
        # And it still persists as None (still auto).
        assert dlg.to_task().output_path is None

    def test_custom_output_survives_source_change(self, qtbot, tmp_path):
        t = _task(tmp_path, output_path=None)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        custom = tmp_path / "fixed.mp4"
        dlg._output_edit.setText(str(custom))  # noqa: SLF001
        dlg._source_edit.setText(str(tmp_path / "hero.png"))  # noqa: SLF001
        assert dlg._output_edit.text() == str(custom)  # noqa: SLF001
        assert dlg.to_task().output_path == custom

    def test_default_reflects_format_change(self, qtbot, tmp_path):
        # Video default ends with .mp4; switching to frames yields a
        # directory name (no extension).
        t = _task(tmp_path, output_path=None)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._output_edit.text().endswith(".mp4")  # noqa: SLF001
        idx = dlg._format_combo.findData("frames")  # noqa: SLF001
        dlg._format_combo.setCurrentIndex(idx)  # noqa: SLF001
        assert dlg._output_edit.text() == str(tmp_path / "src+tgt")  # noqa: SLF001


class TestWritebackToTask:
    def test_to_task_returns_edited_copy(self, qtbot, tmp_path):
        t = _task(tmp_path, swapper_execution=OnnxExecution(workers=1))
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._swapper_workers.setValue(8)  # noqa: SLF001
        edited = dlg.to_task()
        assert edited.swapper_execution.workers == 8
        # Original is unchanged (model_copy returns a new instance).
        assert t.swapper_execution.workers == 1

    def test_to_task_preserves_id_and_runtime_state(
        self, qtbot, tmp_path
    ):
        t = _task(
            tmp_path,
            status=BatchTaskStatus.PAUSED,
            last_completed_frame=42,
            total_frames=100,
            started_at=123.0,
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._swapper_workers.setValue(2)  # noqa: SLF001
        edited = dlg.to_task()
        # Runtime state stays put — the dialog is for params, not state.
        assert edited.id == t.id
        assert edited.status is BatchTaskStatus.PAUSED
        assert edited.last_completed_frame == 42
        assert edited.total_frames == 100
        assert edited.started_at == 123.0

    def test_cleared_output_path_becomes_none(self, qtbot, tmp_path):
        t = _task(tmp_path, output_path=tmp_path / "x.mp4")
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._output_edit.setText("")  # noqa: SLF001
        edited = dlg.to_task()
        assert edited.output_path is None


class TestCleanupMode:
    def test_prefills_from_task(self, qtbot, tmp_path):
        from sinner2.batch.task import BatchCleanupMode

        t = _task(tmp_path, cleanup_mode=BatchCleanupMode.DROP_AT_END)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._cleanup_combo.currentData() == "drop_at_end"  # noqa: SLF001

    def test_writeback(self, qtbot, tmp_path):
        from sinner2.batch.task import BatchCleanupMode

        t = _task(tmp_path)  # defaults to Keep
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        idx = dlg._cleanup_combo.findData("auto")  # noqa: SLF001
        dlg._cleanup_combo.setCurrentIndex(idx)  # noqa: SLF001
        assert dlg.to_task().cleanup_mode is BatchCleanupMode.AUTO


class TestSwapperToggle:
    def test_prefills_swapper_enabled(self, qtbot, tmp_path):
        t = _task(tmp_path, swapper_enabled=False)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._swapper_box.isChecked() is False  # noqa: SLF001

    def test_writeback_swapper_enabled(self, qtbot, tmp_path):
        t = _task(tmp_path)  # defaults to enabled
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._swapper_box.setChecked(False)  # noqa: SLF001
        assert dlg.to_task().swapper_enabled is False
