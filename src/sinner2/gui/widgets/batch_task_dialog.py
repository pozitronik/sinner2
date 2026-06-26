"""Modal dialog for editing a BatchTask's config.

Surfaces the full session config (chain + execution + output) in a TABBED form
so the user can tweak any aspect of how the task will run without fighting a
single tall scroll. The control set, grouping AND order mirror the live settings
panel (QProcessorControls): Recognition / Face swap / Enhance / Upscale are
separate tabs just like the live groups; the Face-swap tab uses the same
Rotation / Occlusion sub-boxes; every ONNX-using stage gets the shared
OnnxProvidersRow selector; and the precalculated face map has a per-task switch.

Open via .from_task(task) → user edits → accept() commits back via
.to_task() — caller persists.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sinner2.batch.task import (
    BatchCleanupMode,
    BatchOutputFormat,
    BatchTask,
    resolve_output_path,
)
from sinner2.config.execution import available_torch_devices
from sinner2.config.target import Target, TargetKind
from sinner2.gui.widgets.model_choices import (
    DETECTOR_MODELS,
    ENHANCER_MODELS,
    OCCLUDER_MODELS,
    OCCLUSION_MODES,
    OCCLUSION_PARSERS,
    ROTATION_SOURCES,
    SWAPPER_MODELS,
    UPSCALER_MODELS,
)
from sinner2.gui.widgets.onnx_providers_row import OnnxProvidersRow
from sinner2.gui.widgets.processor_gating import (
    update_enhancer_rows,
    update_occlusion_rows,
    update_rotation_rows,
    update_swapper_model_rows,
    update_temporal_rows,
    update_upscaler_rows,
)
from sinner2.io.cv2_video_target_reader import CV2VideoTargetReader
from sinner2.io.frame_resize import scaled_dims
from sinner2.io.target_reader import ImageTargetReader
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.face_map_geometry import geometry_path
from sinner2.pipeline.face_map_store import face_map_path, load_face_map
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.processors.face_swapper import TargetSex


_TARGET_SEX_OPTIONS = [
    ("Both (M+F)", TargetSex.BOTH.value),
    ("Male only", TargetSex.MALE.value),
    ("Female only", TargetSex.FEMALE.value),
    ("As source face", TargetSex.AS_SOURCE.value),
]

_CLEANUP_OPTIONS = [
    ("Keep all frames", BatchCleanupMode.KEEP.value),
    ("Auto (drop consumed stages)", BatchCleanupMode.AUTO.value),
    ("Drop all when done", BatchCleanupMode.DROP_AT_END.value),
]


class QBatchTaskDialog(QDialog):
    """Edit-a-task modal."""

    def __init__(
        self,
        task: BatchTask,
        parent: QWidget | None = None,
        global_output_dir: Path | None = None,
        *,
        defaults_mode: bool = False,
        store_path: str = "",
        global_output_path: str = "",
    ) -> None:
        super().__init__(parent)
        # Two faces of one form. PER-TASK mode (default) edits a concrete
        # task's source/target/output. DEFAULTS mode edits the Batch Defaults
        # TEMPLATE — the same chain/execution/output controls, but the Paths
        # group swaps the per-task source/target/output for the two queue-wide
        # folders (task store + global output). Reusing the whole form means
        # the defaults editor can never drift from the per-task editor.
        self._defaults_mode = defaults_mode
        self.setWindowTitle(
            "Batch settings — defaults for new tasks"
            if defaults_mode
            else "Edit batch task"
        )
        # Default auto-size came up too narrow to read full file paths.
        self.setMinimumWidth(600)
        self._task = task
        self._global_output_dir = global_output_dir
        # The auto-derived output path currently shown in the field. Lets
        # to_task() distinguish "left on auto" (persist None) from "user
        # typed a path" (persist verbatim). Recomputed at the end of init.
        self._auto_output = ""

        # ---- Paths group ----
        # PER-TASK: source / target / output for this one task. DEFAULTS: the
        # two queue-wide folders that aren't per-task at all (where queued
        # tasks are stored + where finished outputs land).
        if defaults_mode:
            paths_box = QGroupBox("Batch paths")
            paths_form = QFormLayout(paths_box)
            self._store_edit, store_row = self._path_picker(
                initial=store_path,
                caption="Select task store folder",
                file_filter="",
                dir_mode=True,
            )
            self._store_edit.setToolTip(
                "Folder holding the queued-task files. Empty = the default "
                "(<cache>/../batch). Takes effect after a restart."
            )
            paths_form.addRow("Task store folder:", store_row)
            restart_note = QLabel("Store folder change applies after restart.")
            restart_note.setEnabled(False)  # muted hint styling
            paths_form.addRow("", restart_note)
            self._global_out_edit, global_out_row = self._path_picker(
                initial=global_output_path,
                caption="Select global output folder",
                file_filter="",
                dir_mode=True,
            )
            self._global_out_edit.setToolTip(
                "Folder all finished outputs land in. Empty = next to each "
                "task's target. Applies immediately to new and existing tasks."
            )
            paths_form.addRow("Global output folder:", global_out_row)
        else:
            paths_box = QGroupBox("Paths")
            paths_form = QFormLayout(paths_box)
            self._source_edit, source_row = self._path_picker(
                initial=str(task.source_path),
                caption="Select source",
                file_filter="Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All files (*)",
            )
            paths_form.addRow("Source:", source_row)
            self._target_edit, target_row = self._path_picker(
                initial=str(task.target_path),
                caption="Select target",
                file_filter=(
                    "Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv *.webm);;"
                    "All files (*)"
                ),
            )
            paths_form.addRow("Target:", target_row)
            self._output_edit, output_row = self._path_picker(
                initial=str(task.output_path) if task.output_path else "",
                caption="Output (leave empty for default)",
                file_filter="Output file (*)",
                save_mode=True,
            )
            paths_form.addRow("Output:", output_row)
        self._format_combo = QComboBox()
        self._format_combo.addItem("Video (mp4)", BatchOutputFormat.VIDEO.value)
        self._format_combo.addItem("Frames (directory)", BatchOutputFormat.FRAMES.value)
        self._format_combo.setCurrentIndex(
            0 if task.output_format is BatchOutputFormat.VIDEO else 1
        )
        paths_form.addRow("Output format:", self._format_combo)

        # ---- Faces recognition group (its own tab, like the live panel) ----
        # Order mirrors the live Faces group: face map → detector → detection
        # size/interval → "Which to swap" sub-box (selection).
        faces_box = QGroupBox("Faces recognition")
        faces_form = QFormLayout(faces_box)
        # Per-task switch to route faces through the target's precalculated map
        # (catalog + geometry). Enabled only when a usable map exists for the
        # target; the driver loads it live at render time.
        self._use_face_map = QCheckBox()
        self._use_face_map.setChecked(task.use_face_map)
        fm_available = self._probe_face_map_available(task)
        self._use_face_map.setEnabled(fm_available)
        self._use_face_map.setToolTip(
            "Route each detected face through this target's PRECALCULATED face "
            "map (per-identity source + saved geometry) instead of the single "
            "global source. The map is loaded live at render time. Build / scan "
            "it on the Sources tab first."
            + ("" if fm_available else "\nNo usable face map found for this target yet.")
        )
        faces_form.addRow("Use face map:", self._use_face_map)
        self._detector = QComboBox()
        for value, label in DETECTOR_MODELS:
            self._detector.addItem(label, value)
            if value == task.swapper_detector:
                self._detector.setCurrentIndex(self._detector.count() - 1)
        self._detector.setToolTip(
            "Target detector. yoloface / scrfd are faster (detection-only) but "
            "disable the gender filter and use keypoint-angle rotation."
        )
        self._detector.currentIndexChanged.connect(self._update_detector_rows)
        faces_form.addRow("Detector:", self._detector)
        self._detection_size = QSpinBox()
        # Multiples of 32 (SCRFD strides); 640 default, smaller = faster.
        self._detection_size.setRange(128, 1280)
        self._detection_size.setSingleStep(32)
        self._detection_size.setValue(task.swapper_detection_size)
        self._detection_size.setToolTip(
            "Face-detector input size (px). Smaller = faster detection but may "
            "miss small or distant faces. 640 default; multiples of 32."
        )
        faces_form.addRow("Detection size:", self._detection_size)
        self._detection_interval = QSpinBox()
        self._detection_interval.setRange(1, 30)
        self._detection_interval.setValue(task.swapper_detection_interval)
        self._detection_interval.setToolTip(
            "Detect every Nth frame and reuse the result between (1 = every "
            "frame). Higher = faster on stable shots."
        )
        faces_form.addRow("Detection interval:", self._detection_interval)
        # "Which to swap" sub-box (selection) sits under the detector it depends
        # on — same nesting as the live panel.
        which_box = QGroupBox("Which to swap")
        which_form = QFormLayout(which_box)
        self._many_faces = QCheckBox()
        self._many_faces.setChecked(task.swapper_many_faces)
        which_form.addRow("Many faces:", self._many_faces)
        self._target_sex = QComboBox()
        for label, value in _TARGET_SEX_OPTIONS:
            self._target_sex.addItem(label, value)
            if value == task.swapper_target_sex:
                self._target_sex.setCurrentIndex(self._target_sex.count() - 1)
        which_form.addRow("Gender:", self._target_sex)
        faces_form.addRow(which_box)

        # ---- Face swapper group (checkable: disable for enhancer-only) ----
        # Order mirrors the live Face-swap group: model → fast paste →
        # execution (workers + providers) → landmark refine → Rotation →
        # Occlusion.
        swap_box = QGroupBox("Face swapper")
        swap_box.setCheckable(True)
        swap_box.setChecked(task.swapper_enabled)
        self._swapper_box = swap_box
        swap_form = QFormLayout(swap_box)
        self._swapper_model = QComboBox()
        for value, label in SWAPPER_MODELS:
            self._swapper_model.addItem(label, value)
            if value == task.swapper_model:
                self._swapper_model.setCurrentIndex(self._swapper_model.count() - 1)
        self._swapper_model.setToolTip(
            "Face-swap model. Non-default weights download on first run of the "
            "task."
        )
        self._swapper_model.currentIndexChanged.connect(
            self._update_swapper_model_rows
        )
        swap_form.addRow("Model:", self._swapper_model)
        self._fast_paste = QCheckBox()
        self._fast_paste.setChecked(task.swapper_fast_paste)
        self._fast_paste.setToolTip(
            "Fast ROI feather paste (~2.7x faster). Off = insightface's "
            "original diff-based blend (inswapper/reswapper only)."
        )
        swap_form.addRow("Fast paste:", self._fast_paste)
        self._swapper_workers = QSpinBox()
        self._swapper_workers.setRange(1, 16)
        self._swapper_workers.setValue(task.swapper_execution.workers)
        self._swapper_workers.setToolTip(
            "Worker threads for the swap stage. The swapper shares one ONNX "
            "session across threads, so more workers cost little extra VRAM."
        )
        swap_form.addRow("Workers:", self._swapper_workers)
        # Shared provider strip. ``preferred`` renders the task's saved EPs
        # first (in priority order) and keeps a requested-but-unavailable EP
        # (e.g. a CUDA task edited on a CPU box) so editing round-trips it.
        self._swapper_providers_row = OnnxProvidersRow(
            preferred=list(task.swapper_execution.providers)
        )
        self._swapper_providers_row.set_selected(
            list(task.swapper_execution.providers)
        )
        swap_form.addRow("ONNX providers:", self._swapper_providers_row)
        self._landmark_refine = QCheckBox()
        self._landmark_refine.setChecked(task.swapper_landmark_refine)
        self._landmark_refine.setToolTip(
            "Refine keypoints with the 2dfan4 68-point landmarker (better "
            "alignment on tilted faces). Downloads 2dfan4 on first batch run."
        )
        swap_form.addRow("Landmark refine:", self._landmark_refine)

        # -- Rotation sub-box (experimental upright-then-composite) --
        self._rotation_box = QGroupBox("Rotation compensation")
        self._rotation_box.setCheckable(True)
        self._rotation_box.setChecked(task.swapper_rotation_compensation)
        self._rotation_box.toggled.connect(self._update_rotation_rows)
        self._rotation_box.setToolTip(
            "Experimental: upright faces tilted past the threshold before "
            "swapping, then composite back. Affects output."
        )
        rotation_form = QFormLayout(self._rotation_box)
        self._rotation_threshold = QSpinBox()
        self._rotation_threshold.setRange(0, 90)
        self._rotation_threshold.setSuffix("°")
        self._rotation_threshold.setValue(task.swapper_rotation_threshold_deg)
        rotation_form.addRow("Roll threshold:", self._rotation_threshold)
        self._rotation_redetect = QCheckBox()
        self._rotation_redetect.setChecked(task.swapper_rotation_redetect)
        self._rotation_redetect.setToolTip(
            "Re-detect on the uprighted crop for clean keypoints."
        )
        rotation_form.addRow("Re-detect uprighted:", self._rotation_redetect)
        self._rotation_source = QComboBox()
        for label, value in ROTATION_SOURCES:
            self._rotation_source.addItem(label, value)
            if value == task.swapper_rotation_angle_source:
                self._rotation_source.setCurrentIndex(self._rotation_source.count() - 1)
        rotation_form.addRow("Angle source:", self._rotation_source)
        swap_form.addRow(self._rotation_box)

        # -- Occlusion sub-box --
        self._occlusion_box = QGroupBox("Occlusion")
        self._occlusion_box.setCheckable(True)
        self._occlusion_box.setChecked(task.swapper_occlusion_mask)
        self._occlusion_box.setToolTip(
            "Mask the swap to the real face region (hair/glasses/boundary keep "
            "the original). Parser model downloads on first batch run."
        )
        self._occlusion_box.toggled.connect(self._update_occlusion_rows)
        occlusion_form = QFormLayout(self._occlusion_box)
        # The mode comes FIRST — it decides which of the two dependent rows
        # below it (parser / occluder) apply.
        self._occlusion_mode = QComboBox()
        for value, label in OCCLUSION_MODES:
            self._occlusion_mode.addItem(label, value)
            if value == task.swapper_occlusion_mode:
                self._occlusion_mode.setCurrentIndex(
                    self._occlusion_mode.count() - 1
                )
        self._occlusion_mode.currentIndexChanged.connect(
            self._update_occlusion_rows
        )
        occlusion_form.addRow("Mask source:", self._occlusion_mode)
        self._occlusion_parser = QComboBox()
        for value, label in OCCLUSION_PARSERS:
            self._occlusion_parser.addItem(label, value)
            if value == task.swapper_occlusion_parser:
                self._occlusion_parser.setCurrentIndex(
                    self._occlusion_parser.count() - 1
                )
        occlusion_form.addRow("Mask parser:", self._occlusion_parser)
        self._occluder_model = QComboBox()
        for value, label in OCCLUDER_MODELS:
            self._occluder_model.addItem(label, value)
            if value == task.swapper_occluder_model:
                self._occluder_model.setCurrentIndex(
                    self._occluder_model.count() - 1
                )
        occlusion_form.addRow("Occluder:", self._occluder_model)
        self._occlusion_cache = QCheckBox()
        self._occlusion_cache.setChecked(task.swapper_occlusion_cache)
        self._occlusion_cache.setToolTip(
            "Reuse a near-static face's occlusion mask across frames (faster; "
            "slight boundary lag on motion). Output-affecting."
        )
        occlusion_form.addRow("Cache mask:", self._occlusion_cache)
        swap_form.addRow(self._occlusion_box)

        # -- Temporal stabilization sub-box (needs per-frame face-map geometry) --
        geom_available = self._probe_geometry_available(task)
        self._temporal_box = QGroupBox("Temporal stabilization")
        self._temporal_box.setCheckable(True)
        self._temporal_box.setChecked(task.swapper_temporal_stabilization)
        self._temporal_box.setEnabled(geom_available)
        self._temporal_box.setToolTip(
            "Smooth each face's keypoints over time to reduce swap jitter. Needs "
            "a prebuilt face map with per-frame geometry; a no-op without one."
            if geom_available
            else "Requires a prebuilt face map with per-frame geometry for this "
            "target."
        )
        self._temporal_box.toggled.connect(self._update_temporal_rows)
        temporal_form = QFormLayout(self._temporal_box)
        self._temporal_window = QSpinBox()
        self._temporal_window.setRange(1, 199)
        self._temporal_window.setSingleStep(2)
        self._temporal_window.setValue(task.swapper_temporal_window)
        self._temporal_window.setToolTip(
            "Smoothing span in frames (odd; larger = steadier)."
        )
        temporal_form.addRow("Window (frames):", self._temporal_window)
        self._temporal_strength = QDoubleSpinBox()
        self._temporal_strength.setRange(0.0, 1.0)
        self._temporal_strength.setSingleStep(0.1)
        self._temporal_strength.setValue(task.swapper_temporal_strength)
        self._temporal_strength.setToolTip(
            "Blend from raw (0) to fully smoothed (1) keypoints."
        )
        temporal_form.addRow("Strength:", self._temporal_strength)
        swap_form.addRow(self._temporal_box)

        # ---- FaceEnhancer group (its own tab) ----
        # Order mirrors the live enhancer group: model → upscale → fidelity →
        # center face → half precision → execution (workers + device + providers).
        enh_box = QGroupBox("Face enhancer")
        enh_box.setCheckable(True)
        enh_box.setChecked(task.enhancer_enabled)
        self._enhancer_box = enh_box
        enh_form = QFormLayout(enh_box)
        self._enhancer_model = QComboBox()
        for value, label in ENHANCER_MODELS:
            self._enhancer_model.addItem(label, value)
            if value == task.enhancer_model:
                self._enhancer_model.setCurrentIndex(self._enhancer_model.count() - 1)
        self._enhancer_model.setToolTip(
            "Restoration model. GFPGAN upscales the whole frame (torch device); "
            "the ONNX restorers run on the ONNX providers below."
        )
        self._enhancer_model.currentIndexChanged.connect(self._update_enhancer_rows)
        enh_form.addRow("Model:", self._enhancer_model)
        self._upscale = QSpinBox()
        self._upscale.setRange(1, 4)
        self._upscale.setValue(task.enhancer_upscale)
        enh_form.addRow("Upscale:", self._upscale)
        self._enhancer_fidelity = QDoubleSpinBox()
        self._enhancer_fidelity.setRange(0.0, 1.0)
        self._enhancer_fidelity.setSingleStep(0.1)
        self._enhancer_fidelity.setDecimals(2)
        self._enhancer_fidelity.setValue(task.enhancer_codeformer_fidelity)
        self._enhancer_fidelity.setToolTip(
            "CodeFormer fidelity w: 0 = max restoration, 1 = max fidelity to "
            "the input. Ignored by GFPGAN."
        )
        enh_form.addRow("Fidelity (w):", self._enhancer_fidelity)
        self._only_center_face = QCheckBox()
        self._only_center_face.setChecked(task.enhancer_only_center_face)
        enh_form.addRow("Center face only:", self._only_center_face)
        self._only_swapped = QCheckBox()
        self._only_swapped.setChecked(task.enhancer_only_swapped)
        self._only_swapped.setToolTip(
            "Restore only the faces the swapper actually swapped, not every "
            "detected face. Needs the face swapper enabled (greyed out "
            "otherwise)."
        )
        enh_form.addRow("Swapped faces only:", self._only_swapped)
        # Inert without the swapper (it marks the swapped subset) — gate it.
        self._swapper_box.toggled.connect(self._only_swapped.setEnabled)
        self._only_swapped.setEnabled(self._swapper_box.isChecked())
        self._enhancer_fp16 = QCheckBox()
        self._enhancer_fp16.setChecked(task.enhancer_fp16)
        self._enhancer_fp16.setToolTip(
            "GFPGAN half precision: less VRAM per worker + faster. CUDA only; "
            "ignored by the ONNX restorers."
        )
        enh_form.addRow("Half precision:", self._enhancer_fp16)
        self._enhancer_workers = QSpinBox()
        self._enhancer_workers.setRange(1, 16)
        self._enhancer_workers.setValue(task.enhancer_execution.workers)
        self._enhancer_workers.setToolTip(
            "Worker threads for the enhance stage. GFPGAN isn't thread-safe, "
            "so each worker loads its own model (~1.3 GB VRAM each)."
        )
        enh_form.addRow("Workers:", self._enhancer_workers)
        # GFPGAN torch device (Auto / CPU / each CUDA GPU). Used by the torch
        # GFPGAN backend only; the ONNX restorers use the providers row below.
        self._enhancer_device = QComboBox()
        for value, label in available_torch_devices():
            self._enhancer_device.addItem(label, value)
        current_device = task.enhancer_execution.device
        if self._enhancer_device.findData(current_device) < 0:
            # Preserve a device token this machine doesn't expose (e.g. a
            # cuda:N from another box) so editing doesn't silently reset it.
            self._enhancer_device.addItem(current_device, current_device)
        self._enhancer_device.setCurrentIndex(
            self._enhancer_device.findData(current_device)
        )
        self._enhancer_device.setToolTip(
            "Torch device for GFPGAN. Auto picks CUDA when available, else CPU."
        )
        enh_form.addRow("CUDA device:", self._enhancer_device)
        # ONNX providers for the ONNX restorers (GFPGAN-ONNX / CodeFormer / GPEN
        # / RestoreFormer++). Active only when an ONNX model is chosen.
        self._enhancer_providers_row = OnnxProvidersRow(
            preferred=list(task.enhancer_execution.providers)
        )
        self._enhancer_providers_row.set_selected(
            list(task.enhancer_execution.providers)
        )
        enh_form.addRow("ONNX providers:", self._enhancer_providers_row)

        # ---- Upscaler group (its own tab) ----
        # Order mirrors the live upscaler group: model → tile → half precision →
        # execution (workers + device + providers).
        up_box = QGroupBox("Frame upscaler (Real-ESRGAN)")
        up_box.setCheckable(True)
        up_box.setChecked(task.upscaler_enabled)
        self._upscaler_box = up_box
        up_form = QFormLayout(up_box)
        self._upscaler_model = QComboBox()
        for value, label in UPSCALER_MODELS:
            self._upscaler_model.addItem(label, value)
            if value == task.upscaler_model:
                self._upscaler_model.setCurrentIndex(self._upscaler_model.count() - 1)
        self._upscaler_model.currentIndexChanged.connect(self._update_upscaler_rows)
        up_form.addRow("Model:", self._upscaler_model)
        self._upscaler_tile = QSpinBox()
        self._upscaler_tile.setRange(0, 2048)
        self._upscaler_tile.setSingleStep(64)
        self._upscaler_tile.setValue(task.upscaler_tile)
        self._upscaler_tile.setToolTip("Tile size (px) to bound VRAM; 0 = whole frame.")
        up_form.addRow("Tile size:", self._upscaler_tile)
        self._upscaler_fp16 = QCheckBox()
        self._upscaler_fp16.setChecked(task.upscaler_fp16)
        self._upscaler_fp16.setToolTip("Half precision (faster, less VRAM, CUDA only).")
        up_form.addRow("Half precision:", self._upscaler_fp16)
        self._upscaler_workers = QSpinBox()
        self._upscaler_workers.setRange(1, 16)
        self._upscaler_workers.setValue(task.upscaler_execution.workers)
        self._upscaler_workers.setToolTip(
            "Worker threads for the upscale stage. Heavy — each worker loads its "
            "own model plus large activations, so raise it cautiously."
        )
        up_form.addRow("Workers:", self._upscaler_workers)
        self._upscaler_device = QComboBox()
        for value, label in available_torch_devices():
            self._upscaler_device.addItem(label, value)
        up_current_device = task.upscaler_execution.device
        if self._upscaler_device.findData(up_current_device) < 0:
            self._upscaler_device.addItem(up_current_device, up_current_device)
        self._upscaler_device.setCurrentIndex(
            self._upscaler_device.findData(up_current_device)
        )
        self._upscaler_device.setToolTip(
            "Torch device for the torch upscalers (Real-ESRGAN / SwinIR)."
        )
        up_form.addRow("CUDA device:", self._upscaler_device)
        # ONNX providers for the ONNX upscalers (HAT / SPAN / UltraSharp / fp16
        # exports). Active only when an ONNX model is chosen.
        self._upscaler_providers_row = OnnxProvidersRow(
            preferred=list(task.upscaler_execution.providers)
        )
        self._upscaler_providers_row.set_selected(
            list(task.upscaler_execution.providers)
        )
        up_form.addRow("ONNX providers:", self._upscaler_providers_row)

        # ---- Execution group ----
        exec_box = QGroupBox("Execution")
        exec_form = QFormLayout(exec_box)
        self._video_backend = QComboBox()
        for backend in VideoBackend:
            self._video_backend.addItem(backend.value, backend.value)
            if backend is task.video_backend:
                self._video_backend.setCurrentIndex(
                    self._video_backend.count() - 1
                )
        exec_form.addRow("Video backend:", self._video_backend)
        self._reader_pool_size = QSpinBox()
        self._reader_pool_size.setRange(1, 16)
        self._reader_pool_size.setValue(task.reader_pool_size)
        exec_form.addRow("Reader pool size:", self._reader_pool_size)
        # Processing scale: slider drives the percent, the label shows the
        # resulting WxH for this task's target (probed below). Same control as
        # the realtime panel; here the dimensions are exact for the task.
        self._target_native_size: tuple[int, int] | None = None
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(10, 100)
        self._scale_slider.setValue(round(task.processing_scale * 100))
        self._scale_slider.setToolTip(
            "Downscale frames before processing for speed (output is the "
            "reduced resolution). 100% = full resolution."
        )
        self._scale_label = QLabel()
        self._scale_label.setMinimumWidth(110)
        self._scale_slider.valueChanged.connect(self._update_scale_label)
        scale_row = QWidget()
        scale_row_layout = QHBoxLayout(scale_row)
        scale_row_layout.setContentsMargins(0, 0, 0, 0)
        scale_row_layout.addWidget(self._scale_slider, stretch=1)
        scale_row_layout.addWidget(self._scale_label)
        exec_form.addRow("Processing scale:", scale_row)
        self._cleanup_combo = QComboBox()
        for label, value in _CLEANUP_OPTIONS:
            self._cleanup_combo.addItem(label, value)
            if value == task.cleanup_mode.value:
                self._cleanup_combo.setCurrentIndex(
                    self._cleanup_combo.count() - 1
                )
        self._cleanup_combo.setToolTip(
            "Keep: retain every stage's frames. Auto: delete a stage once "
            "the next has consumed it. Drop all when done: delete all "
            "intermediates after the final output. The output is always kept."
        )
        exec_form.addRow("Intermediate frames:", self._cleanup_combo)
        self._continue_on_error = QCheckBox()
        self._continue_on_error.setChecked(task.continue_on_error)
        self._continue_on_error.setToolTip(
            "If this task fails, keep processing the rest of the queue instead "
            "of halting. Off (default): a failure stops the queue so you can "
            "see the error and decide (then Start / right-click → Resume)."
        )
        exec_form.addRow("Continue queue on error:", self._continue_on_error)

        # ---- Output encoding group ----
        out_box = QGroupBox("Output encoding (frames mode + ffmpeg input)")
        out_form = QFormLayout(out_box)
        self._image_format = QComboBox()
        for fmt in ImageFormat:
            self._image_format.addItem(fmt.value, fmt.value)
            if fmt is task.image_format:
                self._image_format.setCurrentIndex(
                    self._image_format.count() - 1
                )
        out_form.addRow("Image format:", self._image_format)
        self._image_quality = QSpinBox()
        self._image_quality.setRange(1, 100)
        self._image_quality.setValue(task.image_quality)
        out_form.addRow("Image quality:", self._image_quality)
        # Power-user ffmpeg override for VIDEO output. Appended last so it wins
        # over the built-in H.264 defaults; the even-scale + audio mapping stay.
        self._encode_args = QLineEdit(task.encode_args)
        self._encode_args.setPlaceholderText(
            "e.g. -c:v libx265 -crf 24 -preset slow  (leave empty for default H.264)"
        )
        self._encode_args.setToolTip(
            "Extra ffmpeg encode args for VIDEO output, appended to the command "
            "so they OVERRIDE the defaults (ffmpeg uses the last value for an "
            "option). Power-user / unvalidated — a bad string fails the encode. "
            "The even-scale filter and audio mapping are kept. No effect in "
            "Frames mode."
        )
        out_form.addRow("Extra ffmpeg args:", self._encode_args)

        # ---- Standard OK / Cancel ----
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Group the config into TABS — one per stage, mirroring the live
        # settings panel's groups (Recognition / Face swap / Enhance / Upscale).
        # Each pane stays short; a single scroll was awkward to operate. Each tab
        # scroll-wraps as a safety net for very small screens; the button box
        # stays OUTSIDE the tabs so it's always reachable.
        tabs = QTabWidget()
        tabs.addTab(self._tab(paths_box, out_box), "Task")
        tabs.addTab(self._tab(faces_box), "Recognition")
        tabs.addTab(self._tab(swap_box), "Face swap")
        tabs.addTab(self._tab(enh_box), "Enhance")
        tabs.addTab(self._tab(up_box), "Upscale")
        tabs.addTab(self._tab(exec_box), "Execution")
        self._tabs = tabs

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, stretch=1)
        layout.addWidget(button_box)

        # Open at the form's natural size but never taller than 80% of the
        # screen — each tab scroll-wraps to absorb overflow on small displays.
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            avail_h = screen.availableGeometry().height()
            cap_h = int(avail_h * 0.8)
            self.setMaximumHeight(cap_h)
            desired_h = tabs.sizeHint().height() + button_box.sizeHint().height() + 24
            self.resize(640, min(desired_h, cap_h))

        # ---- Output default (per-task only) ----
        # DEFAULTS mode has no per-task source/target to derive an output path
        # from (and no target to probe for the scale readout's dimensions), so
        # the auto-output sync + target probe are wired only for a real task.
        if not defaults_mode:
            # With no explicit override, show the auto-derived path (so the
            # user can see exactly where output lands) and keep it synced to
            # source/target/format edits until they type their own path.
            self._auto_output = str(self._resolve_default_output())
            if task.output_path is None:
                self._output_edit.setText(self._auto_output)
            self._output_edit.setToolTip(
                "Auto-generated from the source + target names. Edit to use a "
                "custom path; clear it to restore the automatic name."
            )
            self._source_edit.textChanged.connect(self._refresh_default_output)
            self._target_edit.textChanged.connect(self._refresh_default_output)
            self._format_combo.currentIndexChanged.connect(
                self._refresh_default_output
            )
            # Re-probe the scale readout's dimensions when the target changes —
            # DEBOUNCED: the probe opens the media file synchronously
            # (VideoCapture for videos), and textChanged fires per keystroke, so
            # probing directly stalled the GUI thread on every character of a
            # pasted/typed path. A single-shot timer restarted on each change
            # probes only after typing settles.
            self._probe_timer = QTimer(self)
            self._probe_timer.setSingleShot(True)
            self._probe_timer.setInterval(300)
            self._probe_timer.timeout.connect(self._refresh_scale_dims)
            self._target_edit.textChanged.connect(self._probe_timer.start)
            self._refresh_scale_dims()  # initial probe for the task's target
        else:
            # No target to probe → the scale readout shows the bare percent.
            self._update_scale_label()
        self._update_detector_rows()  # gender filter follows the detector
        self._update_enhancer_rows()  # gray out the inactive model's knobs
        self._update_upscaler_rows()  # upscaler knobs follow the model runtime
        self._update_occlusion_rows()  # occlusion subknobs follow the checkbox
        self._update_rotation_rows()  # rotation knobs follow the toggle
        self._update_temporal_rows()  # temporal knobs follow the toggle
        self._update_swapper_model_rows()  # fast-paste follows the swap model

    @staticmethod
    def _probe_face_map_available(task: BatchTask) -> bool:
        """Whether a non-empty precalculated face map exists for this task's
        target (so the 'Use face map' option can do anything). Cheap one-file
        read at construction; never raises. False in defaults mode / when the
        task isn't wired to a face-map store."""
        if not task.face_map_store_dir:
            return False
        try:
            fm = load_face_map(
                face_map_path(task.target_path, Path(task.face_map_store_dir))
            )
        except Exception:
            return False
        return fm is not None and not fm.is_empty()

    @staticmethod
    def _probe_geometry_available(task: BatchTask) -> bool:
        """Whether a per-frame geometry sidecar exists for this task's target —
        the precondition for temporal stabilization (it smooths that timeline).
        Cheap path check; False in defaults mode / without a face-map store."""
        if not task.face_map_store_dir:
            return False
        try:
            return geometry_path(
                task.target_path, Path(task.face_map_store_dir)
            ).is_file()
        except Exception:
            return False

    @staticmethod
    def _tab(*boxes: QWidget) -> QScrollArea:
        """Stack one or more group boxes in a scroll-wrapped tab page. The
        scroll area only engages on displays too short for the tab's natural
        height (the 80% cap); on normal screens every control is visible."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        for box in boxes:
            page_layout.addWidget(box)
        page_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _update_detector_rows(self) -> None:
        """Gray the gender filter for detection-only detectors (no .sex)."""
        self._target_sex.setEnabled(self._detector.currentData() == "buffalo_l")

    def _update_enhancer_rows(self) -> None:
        # Gating rules shared with the live Settings panel — see processor_gating.
        update_enhancer_rows(
            self._enhancer_model, self._upscale, self._enhancer_fidelity,
            self._enhancer_fp16, self._enhancer_device,
            self._enhancer_providers_row,
        )

    def _update_upscaler_rows(self) -> None:
        # Gating rules shared with the live Settings panel — see processor_gating.
        update_upscaler_rows(
            self._upscaler_model, self._upscaler_fp16, self._upscaler_device,
            self._upscaler_providers_row,
        )

    def _update_occlusion_rows(self) -> None:
        # Gating rules shared with the live Settings panel — see processor_gating.
        update_occlusion_rows(
            self._occlusion_box, self._occlusion_mode, self._occlusion_parser,
            self._occluder_model, self._occlusion_cache,
        )

    def _update_temporal_rows(self) -> None:
        # Gating rules shared with the live Settings panel — see processor_gating.
        update_temporal_rows(
            self._temporal_box, self._temporal_window,
            self._temporal_strength,
        )

    def _update_rotation_rows(self) -> None:
        # Gating rules shared with the live Settings panel — see processor_gating.
        update_rotation_rows(
            self._rotation_box, self._rotation_threshold,
            self._rotation_redetect, self._rotation_source,
        )

    def _update_swapper_model_rows(self) -> None:
        # Gating rules shared with the live Settings panel — see processor_gating.
        update_swapper_model_rows(self._swapper_model, self._fast_paste)

    @classmethod
    def from_task(
        cls,
        task: BatchTask,
        parent: QWidget | None = None,
        global_output_dir: Path | None = None,
    ) -> "QBatchTaskDialog":
        return cls(task, parent=parent, global_output_dir=global_output_dir)

    def to_task(self) -> BatchTask:
        """Return a new BatchTask with edits applied. Preserves the
        original task's id + runtime state (status, last_completed_frame,
        timing). Runtime state isn't editable here.

        In DEFAULTS mode the source/target/output keys are left untouched (the
        template keeps its sentinel paths) — only the chain/execution/output
        config is edited there; the queue-wide folders come from
        store_path()/global_output_path() instead."""
        format_value = self._format_combo.currentData()
        update: dict[str, object] = {
            "output_format": BatchOutputFormat(format_value),
            "use_face_map": self._use_face_map.isChecked(),
            "swapper_enabled": self._swapper_box.isChecked(),
            "swapper_model": self._swapper_model.currentData(),
            "swapper_detection_interval": self._detection_interval.value(),
            "swapper_detection_size": self._detection_size.value(),
            "swapper_detector": self._detector.currentData(),
            "swapper_many_faces": self._many_faces.isChecked(),
            "swapper_fast_paste": self._fast_paste.isChecked(),
            "swapper_landmark_refine": self._landmark_refine.isChecked(),
            "swapper_temporal_stabilization": self._temporal_box.isChecked(),
            "swapper_temporal_window": self._temporal_window.value(),
            "swapper_temporal_strength": self._temporal_strength.value(),
            "swapper_target_sex": self._target_sex.currentData(),
            "swapper_rotation_compensation": self._rotation_box.isChecked(),
            "swapper_rotation_threshold_deg": self._rotation_threshold.value(),
            "swapper_rotation_redetect": self._rotation_redetect.isChecked(),
            "swapper_rotation_angle_source": self._rotation_source.currentData(),
            "swapper_occlusion_mask": self._occlusion_box.isChecked(),
            "swapper_occlusion_mode": self._occlusion_mode.currentData(),
            "swapper_occlusion_parser": self._occlusion_parser.currentData(),
            "swapper_occluder_model": self._occluder_model.currentData(),
            "swapper_occlusion_cache": self._occlusion_cache.isChecked(),
            "enhancer_enabled": self._enhancer_box.isChecked(),
            "enhancer_model": self._enhancer_model.currentData(),
            "enhancer_upscale": self._upscale.value(),
            "enhancer_only_center_face": self._only_center_face.isChecked(),
            "enhancer_only_swapped": self._only_swapped.isChecked(),
            "enhancer_codeformer_fidelity": self._enhancer_fidelity.value(),
            "enhancer_fp16": self._enhancer_fp16.isChecked(),
            "swapper_execution": self._task.swapper_execution.model_copy(
                update={
                    "workers": self._swapper_workers.value(),
                    "providers": self._swapper_providers_row.selected(),
                }
            ),
            "enhancer_execution": self._task.enhancer_execution.model_copy(
                update={
                    "workers": self._enhancer_workers.value(),
                    "device": self._enhancer_device.currentData(),
                    "providers": self._enhancer_providers_row.selected(),
                }
            ),
            "upscaler_enabled": self._upscaler_box.isChecked(),
            "upscaler_model": self._upscaler_model.currentData(),
            "upscaler_tile": self._upscaler_tile.value(),
            "upscaler_fp16": self._upscaler_fp16.isChecked(),
            "upscaler_execution": self._task.upscaler_execution.model_copy(
                update={
                    "workers": self._upscaler_workers.value(),
                    "device": self._upscaler_device.currentData(),
                    "providers": self._upscaler_providers_row.selected(),
                }
            ),
            "video_backend": VideoBackend(self._video_backend.currentData()),
            "reader_pool_size": self._reader_pool_size.value(),
            "processing_scale": self._scale_slider.value() / 100.0,
            "cleanup_mode": BatchCleanupMode(
                self._cleanup_combo.currentData()
            ),
            "continue_on_error": self._continue_on_error.isChecked(),
            "image_format": ImageFormat(self._image_format.currentData()),
            "image_quality": self._image_quality.value(),
            "encode_args": self._encode_args.text().strip(),
        }
        if not self._defaults_mode:
            # Untouched auto value (or empty) → keep output_path None so it
            # stays auto-derived (and follows source/target renames +
            # global-output-dir changes); a genuine edit is stored verbatim.
            output_str = self._output_edit.text().strip()
            update["source_path"] = Path(self._source_edit.text())
            update["target_path"] = Path(self._target_edit.text())
            update["output_path"] = (
                None
                if (not output_str or output_str == self._auto_output)
                else Path(output_str)
            )
        return self._task.model_copy(update=update)

    # ---- Defaults-mode accessors (queue-wide paths) ----

    def store_path(self) -> str:
        """The edited task-store folder (DEFAULTS mode). "" = use the default."""
        return self._store_edit.text().strip()

    def global_output_path(self) -> str:
        """The edited global-output folder (DEFAULTS mode). "" = next to each
        task's target."""
        return self._global_out_edit.text().strip()

    # ---- helpers ----

    def _selected_providers(self) -> list[str]:
        """Checked swapper ONNX providers in display order (non-empty — the
        provider row floors an empty selection to CPU)."""
        return self._swapper_providers_row.selected()

    def _resolve_default_output(self) -> Path:
        """The auto-derived output path for the current source / target /
        format, ignoring any explicit override."""
        probe = self._task.model_copy(
            update={
                "source_path": Path(self._source_edit.text()),
                "target_path": Path(self._target_edit.text()),
                "output_format": BatchOutputFormat(
                    self._format_combo.currentData()
                ),
                "output_path": None,
            }
        )
        return resolve_output_path(probe, self._global_output_dir)

    def _probe_native_size(self) -> tuple[int, int] | None:
        """Read the current target's native (width, height), or None if it
        can't be determined (empty/missing/unreadable path, unsupported kind).

        Uses cv2 for video so a dimensions readout never depends on ffmpeg
        being installed; native size is backend-independent anyway."""
        try:
            target = Target(path=Path(self._target_edit.text()))
            if target.kind is TargetKind.IMAGE:
                reader: ImageTargetReader | CV2VideoTargetReader = (
                    ImageTargetReader(target)
                )
            elif target.kind is TargetKind.VIDEO:
                reader = CV2VideoTargetReader(target)
            else:
                return None
        except Exception:
            return None
        try:
            return reader.native_width, reader.native_height
        finally:
            reader.release()

    def _refresh_scale_dims(self) -> None:
        self._target_native_size = self._probe_native_size()
        self._update_scale_label()

    def _update_scale_label(self) -> None:
        pct = self._scale_slider.value()
        if self._target_native_size is None:
            self._scale_label.setText(f"{pct}%")
            return
        nw, nh = self._target_native_size
        w, h = scaled_dims(nw, nh, pct / 100.0)
        self._scale_label.setText(f"{pct}% [{w}x{h}]")

    def _refresh_default_output(self) -> None:
        """Re-derive the default; if the field is still showing the old
        default (user hasn't overridden it), update it in place."""
        new_auto = str(self._resolve_default_output())
        current = self._output_edit.text().strip()
        if current == "" or current == self._auto_output:
            self._output_edit.setText(new_auto)
        self._auto_output = new_auto

    def _path_picker(
        self,
        initial: str,
        caption: str,
        file_filter: str,
        save_mode: bool = False,
        dir_mode: bool = False,
    ) -> tuple[QLineEdit, QWidget]:
        """Build a (line-edit + browse-button) row. Returns the line
        edit and the composite container widget. ``dir_mode`` browses for a
        folder (the queue-wide batch paths); ``save_mode`` for a save target;
        otherwise an existing file."""
        edit = QLineEdit(initial)
        btn = QPushButton("…")
        btn.setFixedWidth(28)

        def browse() -> None:
            if dir_mode:
                path = QFileDialog.getExistingDirectory(
                    self, caption, edit.text()
                )
            elif save_mode:
                path, _ = QFileDialog.getSaveFileName(
                    self, caption, edit.text(), file_filter
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, caption, edit.text(), file_filter
                )
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, stretch=1)
        layout.addWidget(btn)
        return edit, container
