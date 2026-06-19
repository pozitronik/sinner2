"""Tests for QBatchTaskDialog: pre-fill from task, OK writes back,
runtime state preserved, and the auto-derived output-path behavior."""
from __future__ import annotations

from pathlib import Path


from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.config.execution import HybridExecution, OnnxExecution
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


class TestProviderFloor:
    def test_empty_selection_floored_to_cpu(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path))
        qtbot.addWidget(dlg)
        for cb in dlg._swapper_providers_row.checkboxes().values():  # noqa: SLF001
            cb.setChecked(False)
        # You can't run on no provider — the saved task floors to CPU.
        assert dlg._selected_providers() == ["CPUExecutionProvider"]  # noqa: SLF001
        assert dlg.to_task().swapper_execution.providers == ["CPUExecutionProvider"]


class TestDependentRows:
    """Dependent controls' accessibility is linked: occlusion subknobs follow
    the master checkbox + mode, rotation knobs follow the toggle, fast-paste
    follows the swap model, GFPGAN-only knobs follow the enhancer model."""

    def test_occlusion_subcontrols_follow_checkbox_and_mode(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path))  # occlusion off
        qtbot.addWidget(dlg)
        assert not dlg._occlusion_mode.isEnabled()  # noqa: SLF001
        assert not dlg._occlusion_parser.isEnabled()  # noqa: SLF001
        assert not dlg._occluder_model.isEnabled()  # noqa: SLF001
        dlg._occlusion_mask.setChecked(True)  # noqa: SLF001
        assert dlg._occlusion_parser.isEnabled()  # noqa: SLF001 — region default
        assert not dlg._occluder_model.isEnabled()  # noqa: SLF001
        mode = dlg._occlusion_mode  # noqa: SLF001
        mode.setCurrentIndex(mode.findData("occluder"))
        assert not dlg._occlusion_parser.isEnabled()  # noqa: SLF001
        assert dlg._occluder_model.isEnabled()  # noqa: SLF001

    def test_rotation_knobs_follow_toggle(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(
            _task(tmp_path, swapper_rotation_compensation=False)
        )
        qtbot.addWidget(dlg)
        assert not dlg._rotation_threshold.isEnabled()  # noqa: SLF001
        assert not dlg._rotation_redetect.isEnabled()  # noqa: SLF001
        assert not dlg._rotation_source.isEnabled()  # noqa: SLF001
        dlg._rotation_enabled.setChecked(True)  # noqa: SLF001
        assert dlg._rotation_threshold.isEnabled()  # noqa: SLF001

    def test_fast_paste_follows_swap_model(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path, swapper_model="ghost_1_256"))
        qtbot.addWidget(dlg)
        assert not dlg._fast_paste.isEnabled()  # noqa: SLF001
        combo = dlg._swapper_model  # noqa: SLF001
        combo.setCurrentIndex(combo.findData("inswapper_128"))
        assert dlg._fast_paste.isEnabled()  # noqa: SLF001

    def test_gfpgan_only_knobs_follow_enhancer_model(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path, enhancer_model="gfpgan"))
        qtbot.addWidget(dlg)
        assert dlg._enhancer_fp16.isEnabled()  # noqa: SLF001
        assert dlg._enhancer_device.isEnabled()  # noqa: SLF001
        combo = dlg._enhancer_model  # noqa: SLF001
        combo.setCurrentIndex(combo.findData("gfpgan_onnx"))
        assert not dlg._enhancer_fp16.isEnabled()  # noqa: SLF001
        assert not dlg._enhancer_device.isEnabled()  # noqa: SLF001


class TestDebouncedTargetProbe:
    def test_keystrokes_do_not_probe_synchronously(self, qtbot, tmp_path):
        # The native-size probe opens the media file (VideoCapture) on the GUI
        # thread; firing it per textChanged keystroke stalled the dialog
        # (audit rank 36). Typing must only arm the debounce timer; the probe
        # runs once when it fires.
        dlg = QBatchTaskDialog.from_task(_task(tmp_path))
        qtbot.addWidget(dlg)
        probes: list[int] = []
        dlg._probe_native_size = lambda: probes.append(1)  # noqa: SLF001
        for ch in str(tmp_path / "video.mp4"):
            dlg._target_edit.insert(ch)  # noqa: SLF001 — per-keystroke textChanged
        assert probes == []  # nothing probed during typing
        assert dlg._probe_timer.isActive()  # noqa: SLF001 — debounce armed
        dlg._probe_timer.stop()  # noqa: SLF001 — fire deterministically
        dlg._probe_timer.timeout.emit()  # noqa: SLF001
        assert len(probes) == 1


class TestPrefill:
    def test_dialog_fields_match_task(self, qtbot, tmp_path):
        t = _task(
            tmp_path,
            output_path=tmp_path / "custom.mp4",
            output_format=BatchOutputFormat.FRAMES,
            swapper_detection_interval=5,
            swapper_detection_size=320,
            swapper_many_faces=False,
            swapper_target_sex="F",
            enhancer_enabled=False,
            enhancer_upscale=4,
            enhancer_only_center_face=True,
            enhancer_fp16=False,
            swapper_execution=OnnxExecution(
                workers=8, providers=["CPUExecutionProvider"]
            ),
            enhancer_execution=HybridExecution(workers=2, device="cpu"),
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
        assert dlg._detection_size.value() == 320  # noqa: SLF001
        assert dlg.to_task().swapper_detection_size == 320  # round-trips back out
        assert dlg._detector.currentData() == "buffalo_l"  # noqa: SLF001  default
        assert dlg.to_task().swapper_detector == "buffalo_l"
        assert dlg._many_faces.isChecked() is False  # noqa: SLF001
        assert dlg._target_sex.currentData() == "F"  # noqa: SLF001
        assert dlg._enhancer_box.isChecked() is False  # noqa: SLF001
        assert dlg._upscale.value() == 4  # noqa: SLF001
        assert dlg._only_center_face.isChecked() is True  # noqa: SLF001
        assert dlg._enhancer_fp16.isChecked() is False  # noqa: SLF001
        assert dlg.to_task().enhancer_fp16 is False  # round-trips back out
        assert dlg._swapper_workers.value() == 8  # noqa: SLF001
        # Only the persisted provider is checked; CPU is always available.
        assert dlg._selected_providers() == ["CPUExecutionProvider"]  # noqa: SLF001
        assert dlg._enhancer_workers.value() == 2  # noqa: SLF001
        assert dlg._enhancer_device.currentData() == "cpu"  # noqa: SLF001
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

    def test_continue_on_error_round_trips(self, qtbot, tmp_path):
        t = _task(tmp_path, continue_on_error=False)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._continue_on_error.isChecked() is False  # noqa: SLF001
        dlg._continue_on_error.setChecked(True)  # noqa: SLF001
        assert dlg.to_task().continue_on_error is True
        assert t.continue_on_error is False  # original untouched

    def test_to_task_writes_execution_profiles(self, qtbot, tmp_path):
        t = _task(
            tmp_path,
            swapper_execution=OnnxExecution(
                workers=2, providers=["CPUExecutionProvider"]
            ),
            enhancer_execution=HybridExecution(workers=1, device="cpu"),
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._swapper_workers.setValue(5)  # noqa: SLF001
        dlg._enhancer_workers.setValue(3)  # noqa: SLF001
        edited = dlg.to_task()
        assert edited.swapper_execution.workers == 5
        # Providers round-trip from the checked boxes (only CPU was wanted).
        assert edited.swapper_execution.providers == ["CPUExecutionProvider"]
        assert edited.enhancer_execution.workers == 3
        assert edited.enhancer_execution.device == "cpu"

    def test_unknown_device_token_is_preserved(self, qtbot, tmp_path):
        # A persisted cuda:N this machine doesn't expose must survive an edit
        # rather than silently resetting to Auto.
        t = _task(tmp_path, enhancer_execution=HybridExecution(device="cuda:9"))
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._enhancer_device.currentData() == "cuda:9"  # noqa: SLF001
        assert dlg.to_task().enhancer_execution.device == "cuda:9"

    def test_unavailable_requested_provider_is_preserved(
        self, qtbot, tmp_path, monkeypatch
    ):
        # Editing a task on a machine missing a requested EP must round-trip it
        # (and its priority order), not silently drop it — mirrors the unknown-
        # device preservation above. The shared provider row renders the task's
        # `preferred` EPs even when ORT doesn't expose them.
        from sinner2.gui.widgets import onnx_providers_row as opr

        monkeypatch.setattr(
            opr, "available_onnx_providers", lambda: ["CPUExecutionProvider"]
        )
        t = _task(
            tmp_path,
            swapper_execution=OnnxExecution(
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            ),
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg.to_task().swapper_execution.providers == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

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


class TestProcessingScale:
    def test_prefills_slider_from_task(self, qtbot, tmp_path):
        t = _task(tmp_path, processing_scale=0.5)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._scale_slider.value() == 50  # noqa: SLF001

    def test_writeback_scale(self, qtbot, tmp_path):
        t = _task(tmp_path)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._scale_slider.setValue(25)  # noqa: SLF001
        assert dlg.to_task().processing_scale == 0.25

    def test_label_percent_only_when_target_unreadable(self, qtbot, tmp_path):
        # The fixture target (tgt.mp4) doesn't exist → probe fails → percent only.
        t = _task(tmp_path, processing_scale=0.5)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._scale_label.text() == "50%"  # noqa: SLF001

    def test_label_shows_dims_for_real_image_target(self, qtbot, tmp_path):
        import cv2
        import numpy as np

        img = tmp_path / "face.png"
        cv2.imwrite(str(img), np.full((100, 80, 3), 128, dtype=np.uint8))  # 80x100
        t = _task(tmp_path, target_path=img, processing_scale=0.5)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._scale_label.text() == "50% [40x50]"  # noqa: SLF001


class TestRotationCompensation:
    def test_prefills_from_task(self, qtbot, tmp_path):
        t = _task(
            tmp_path,
            swapper_rotation_compensation=True,
            swapper_rotation_threshold_deg=25,
            swapper_rotation_redetect=False,
            swapper_rotation_angle_source="pose",
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._rotation_enabled.isChecked() is True  # noqa: SLF001
        assert dlg._rotation_threshold.value() == 25  # noqa: SLF001
        assert dlg._rotation_redetect.isChecked() is False  # noqa: SLF001
        assert dlg._rotation_source.currentData() == "pose"  # noqa: SLF001

    def test_writeback(self, qtbot, tmp_path):
        t = _task(tmp_path)
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        dlg._rotation_enabled.setChecked(True)  # noqa: SLF001
        dlg._rotation_threshold.setValue(40)  # noqa: SLF001
        edited = dlg.to_task()
        assert edited.swapper_rotation_compensation is True
        assert edited.swapper_rotation_threshold_deg == 40


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


class TestDefaultsMode:
    """DEFAULTS mode reuses the whole config form but swaps the per-task
    Paths group for the two queue-wide folders and leaves source/target
    untouched on writeback."""

    def test_paths_group_shows_queue_wide_folders(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog(
            _task(tmp_path),
            defaults_mode=True,
            store_path="/my/store",
            global_output_path="/my/out",
        )
        qtbot.addWidget(dlg)
        # Per-task path edits don't exist in defaults mode; the queue-wide
        # folder edits do, pre-filled from the constructor.
        assert not hasattr(dlg, "_source_edit")
        assert not hasattr(dlg, "_output_edit")
        assert dlg.store_path() == "/my/store"
        assert dlg.global_output_path() == "/my/out"

    def test_title_reflects_defaults_mode(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog(_task(tmp_path), defaults_mode=True)
        qtbot.addWidget(dlg)
        assert "default" in dlg.windowTitle().lower()

    def test_to_task_edits_config_but_leaves_paths(self, qtbot, tmp_path):
        # Template carries sentinel paths; editing config must not invent a
        # source/target/output for it.
        tmpl = _task(tmp_path, source_path=Path("."), target_path=Path("."))
        dlg = QBatchTaskDialog(tmpl, defaults_mode=True)
        qtbot.addWidget(dlg)
        dlg._swapper_workers.setValue(7)  # noqa: SLF001
        dlg._swapper_box.setChecked(False)  # noqa: SLF001
        edited = dlg.to_task()
        assert edited.swapper_execution.workers == 7
        assert edited.swapper_enabled is False
        # Paths untouched — still the template's sentinels.
        assert edited.source_path == Path(".")
        assert edited.target_path == Path(".")
        assert edited.output_path is None

    def test_scale_label_is_percent_only(self, qtbot, tmp_path):
        # No per-task target to probe → the readout shows the bare percent.
        tmpl = _task(tmp_path, processing_scale=0.5)
        dlg = QBatchTaskDialog(tmpl, defaults_mode=True)
        qtbot.addWidget(dlg)
        assert dlg._scale_label.text() == "50%"  # noqa: SLF001

    def test_empty_store_and_output_paths_strip_to_blank(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog(
            _task(tmp_path),
            defaults_mode=True,
            store_path="  ",
            global_output_path="",
        )
        qtbot.addWidget(dlg)
        assert dlg.store_path() == ""
        assert dlg.global_output_path() == ""


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


class TestEnhancerProviders:
    """The enhancer is hybrid: torch GFPGAN uses the CUDA device, the ONNX
    restorers use the ONNX providers row. The dialog exposes both and gates by
    model — matching the live settings panel."""

    def test_providers_round_trip(self, qtbot, tmp_path):
        t = _task(
            tmp_path,
            enhancer_execution=HybridExecution(providers=["CPUExecutionProvider"]),
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg.to_task().enhancer_execution.providers == ["CPUExecutionProvider"]

    def test_providers_gated_by_model(self, qtbot, tmp_path):
        # ONNX restorer (default gfpgan_onnx) uses providers, not the torch
        # device; torch GFPGAN is the reverse.
        dlg = QBatchTaskDialog.from_task(
            _task(tmp_path, enhancer_model="gfpgan_onnx")
        )
        qtbot.addWidget(dlg)
        assert dlg._enhancer_providers_row.isEnabled()  # noqa: SLF001
        assert not dlg._enhancer_device.isEnabled()  # noqa: SLF001
        combo = dlg._enhancer_model  # noqa: SLF001
        combo.setCurrentIndex(combo.findData("gfpgan"))
        assert not dlg._enhancer_providers_row.isEnabled()  # noqa: SLF001
        assert dlg._enhancer_device.isEnabled()  # noqa: SLF001


class TestUpscalerExecution:
    """The upscaler gained editable workers + an ONNX providers row (it was
    device-only before), matching its hybrid torch/ONNX model set."""

    def test_workers_round_trip(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path, upscaler_enabled=True))
        qtbot.addWidget(dlg)
        dlg._upscaler_workers.setValue(3)  # noqa: SLF001
        assert dlg.to_task().upscaler_execution.workers == 3

    def test_providers_round_trip(self, qtbot, tmp_path):
        t = _task(
            tmp_path,
            upscaler_execution=HybridExecution(providers=["CPUExecutionProvider"]),
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg.to_task().upscaler_execution.providers == ["CPUExecutionProvider"]

    def test_device_vs_providers_gated_by_model(self, qtbot, tmp_path):
        # Torch model (general-x4v3) uses the torch device; an ONNX model
        # (span-x4) uses the providers row.
        dlg = QBatchTaskDialog.from_task(
            _task(tmp_path, upscaler_model="general-x4v3")
        )
        qtbot.addWidget(dlg)
        assert dlg._upscaler_device.isEnabled()  # noqa: SLF001
        assert not dlg._upscaler_providers_row.isEnabled()  # noqa: SLF001
        combo = dlg._upscaler_model  # noqa: SLF001
        combo.setCurrentIndex(combo.findData("span-x4"))
        assert not dlg._upscaler_device.isEnabled()  # noqa: SLF001
        assert dlg._upscaler_providers_row.isEnabled()  # noqa: SLF001


class TestTabbedLayout:
    """The form is tabbed one-stage-per-tab (Task / Recognition / Face swap /
    Enhance / Upscale / Execution), mirroring the live settings groups, and
    capped at 80% of the screen height so it stays operable on laptops."""

    def test_form_has_per_stage_tabs(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path))
        qtbot.addWidget(dlg)
        titles = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]  # noqa: SLF001
        assert titles == [
            "Task", "Recognition", "Face swap", "Enhance", "Upscale", "Execution",
        ]

    def test_height_capped_at_80_percent(self, qtbot, tmp_path):
        from PySide6.QtGui import QGuiApplication

        dlg = QBatchTaskDialog.from_task(_task(tmp_path))
        qtbot.addWidget(dlg)
        screen = dlg.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry().height()
        assert dlg.maximumHeight() <= int(avail * 0.8)


class TestFaceMapOption:
    """The Recognition tab exposes a per-task 'Use face map' switch. It's
    disabled (no usable map) by default; round-trips through to_task."""

    def test_disabled_without_a_map(self, qtbot, tmp_path):
        dlg = QBatchTaskDialog.from_task(_task(tmp_path))  # no store dir
        qtbot.addWidget(dlg)
        assert dlg._use_face_map.isEnabled() is False  # noqa: SLF001

    def test_enabled_and_round_trips_when_map_present(self, qtbot, tmp_path):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.face_map_store import face_map_path, save_face_map

        store = tmp_path / "face_maps"
        tgt = tmp_path / "tgt.mp4"
        save_face_map(
            face_map_path(tgt, store),
            FaceMap(identities=(Identity("a", normalize([1.0, 0.0, 0.0])),)),
        )
        t = _task(
            tmp_path, target_path=tgt,
            face_map_store_dir=str(store), use_face_map=True,
        )
        dlg = QBatchTaskDialog.from_task(t)
        qtbot.addWidget(dlg)
        assert dlg._use_face_map.isEnabled() is True  # noqa: SLF001
        assert dlg._use_face_map.isChecked() is True  # noqa: SLF001
        dlg._use_face_map.setChecked(False)  # noqa: SLF001
        assert dlg.to_task().use_face_map is False
