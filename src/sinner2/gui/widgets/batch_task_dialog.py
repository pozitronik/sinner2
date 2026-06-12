"""Modal dialog for editing a BatchTask's config.

Surfaces the full session config (chain + execution + output) so the
user can tweak any aspect of how the task will run. v1 keeps the form
fields verbatim — a future refactor could share UI with QProcessorControls.

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
    QVBoxLayout,
    QWidget,
)

from sinner2.batch.task import (
    BatchCleanupMode,
    BatchOutputFormat,
    BatchTask,
    resolve_output_path,
)
from sinner2.config.execution import (
    DEFAULT_ONNX_PROVIDERS,
    available_torch_devices,
)
from sinner2.config.target import Target, TargetKind
from sinner2.io.cv2_video_target_reader import CV2VideoTargetReader
from sinner2.io.frame_resize import scaled_dims
from sinner2.io.target_reader import ImageTargetReader
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.model_cache import available_onnx_providers
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
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit batch task")
        # Default auto-size came up too narrow to read full file paths.
        self.setMinimumWidth(600)
        self._task = task
        self._global_output_dir = global_output_dir
        # The auto-derived output path currently shown in the field. Lets
        # to_task() distinguish "left on auto" (persist None) from "user
        # typed a path" (persist verbatim). Recomputed at the end of init.
        self._auto_output = ""

        # ---- Paths group ----
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

        # ---- FaceSwapper group (checkable: disable for enhancer-only) ----
        swap_box = QGroupBox("Face swapper")
        swap_box.setCheckable(True)
        swap_box.setChecked(task.swapper_enabled)
        self._swapper_box = swap_box
        swap_form = QFormLayout(swap_box)
        self._swapper_model = QComboBox()
        for value, label in (
            ("inswapper_128", "inswapper_128 (default)"),
            ("reswapper_128", "ReSwapper 128"),
            ("ghost_1_256", "Ghost 1 (256)"),
            ("ghost_2_256", "Ghost 2 (256)"),
            ("ghost_3_256", "Ghost 3 (256)"),
            ("simswap_256", "SimSwap (256, non-commercial)"),
            ("uniface_256", "UniFace (256)"),
        ):
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
        self._detection_interval = QSpinBox()
        self._detection_interval.setRange(1, 30)
        self._detection_interval.setValue(task.swapper_detection_interval)
        swap_form.addRow("Detection interval:", self._detection_interval)
        self._detection_size = QSpinBox()
        # Multiples of 32 (SCRFD strides); 640 default, smaller = faster.
        self._detection_size.setRange(128, 1280)
        self._detection_size.setSingleStep(32)
        self._detection_size.setValue(task.swapper_detection_size)
        self._detection_size.setToolTip(
            "Face-detector input size (px). Smaller = faster detection but may "
            "miss small or distant faces. 640 default; multiples of 32."
        )
        swap_form.addRow("Detection size:", self._detection_size)
        self._detector = QComboBox()
        for value, label in (
            ("buffalo_l", "buffalo_l (full pack, gender + pose)"),
            ("yoloface", "YOLOFace 8n (fast, detection-only)"),
            ("scrfd_2.5g", "SCRFD 2.5g (fast, detection-only)"),
        ):
            self._detector.addItem(label, value)
            if value == task.swapper_detector:
                self._detector.setCurrentIndex(self._detector.count() - 1)
        self._detector.setToolTip(
            "Target detector. yoloface / scrfd are faster (detection-only) but "
            "disable the gender filter and use keypoint-angle rotation."
        )
        self._detector.currentIndexChanged.connect(self._update_detector_rows)
        swap_form.addRow("Detector:", self._detector)
        self._many_faces = QCheckBox()
        self._many_faces.setChecked(task.swapper_many_faces)
        swap_form.addRow("Many faces:", self._many_faces)
        self._fast_paste = QCheckBox()
        self._fast_paste.setChecked(task.swapper_fast_paste)
        self._fast_paste.setToolTip(
            "Fast ROI feather paste (~2.7x faster). Off = insightface's "
            "original diff-based blend (inswapper/reswapper only)."
        )
        swap_form.addRow("Fast paste:", self._fast_paste)
        self._target_sex = QComboBox()
        for label, value in _TARGET_SEX_OPTIONS:
            self._target_sex.addItem(label, value)
            if value == task.swapper_target_sex:
                self._target_sex.setCurrentIndex(self._target_sex.count() - 1)
        swap_form.addRow("Swap which:", self._target_sex)
        self._update_detector_rows()
        # Rotation compensation (experimental) — see the realtime panel tooltip.
        self._rotation_enabled = QCheckBox()
        self._rotation_enabled.setChecked(task.swapper_rotation_compensation)
        self._rotation_enabled.toggled.connect(self._update_rotation_rows)
        self._rotation_enabled.setToolTip(
            "Experimental: upright faces tilted past the threshold before "
            "swapping, then composite back. Affects output."
        )
        swap_form.addRow("Rotation comp.:", self._rotation_enabled)
        self._rotation_threshold = QSpinBox()
        self._rotation_threshold.setRange(0, 90)
        self._rotation_threshold.setSuffix("°")
        self._rotation_threshold.setValue(task.swapper_rotation_threshold_deg)
        swap_form.addRow("Roll threshold:", self._rotation_threshold)
        self._rotation_redetect = QCheckBox()
        self._rotation_redetect.setChecked(task.swapper_rotation_redetect)
        self._rotation_redetect.setToolTip(
            "Re-detect on the uprighted crop for clean keypoints."
        )
        swap_form.addRow("Re-detect uprighted:", self._rotation_redetect)
        self._rotation_source = QComboBox()
        for label, value in (("Eye keypoints", "keypoints"), ("3D pose estimate", "pose")):
            self._rotation_source.addItem(label, value)
            if value == task.swapper_rotation_angle_source:
                self._rotation_source.setCurrentIndex(self._rotation_source.count() - 1)
        swap_form.addRow("Angle source:", self._rotation_source)
        self._occlusion_mask = QCheckBox()
        self._occlusion_mask.setChecked(task.swapper_occlusion_mask)
        self._occlusion_mask.setToolTip(
            "Mask the swap to the real face region (hair/glasses/boundary keep "
            "the original). Parser model downloads on first batch run."
        )
        self._occlusion_mask.toggled.connect(self._update_occlusion_rows)
        swap_form.addRow("Occlusion mask:", self._occlusion_mask)
        # The mode comes FIRST in the form — it decides which of the two
        # dependent rows below it (parser / occluder) apply.
        self._occlusion_mode = QComboBox()
        for value, label in (
            ("region", "Region (face parser)"),
            ("occluder", "Occluder (XSeg — sees hands/objects)"),
            ("both", "Both (strictest)"),
        ):
            self._occlusion_mode.addItem(label, value)
            if value == task.swapper_occlusion_mode:
                self._occlusion_mode.setCurrentIndex(
                    self._occlusion_mode.count() - 1
                )
        self._occlusion_mode.currentIndexChanged.connect(
            self._update_occlusion_rows
        )
        swap_form.addRow("Mask source:", self._occlusion_mode)
        self._occlusion_parser = QComboBox()
        for value, label in (
            ("bisenet", "BiSeNet (torch, sharper)"),
            ("parsenet", "ParseNet (torch, GFPGAN default)"),
            ("bisenet_onnx_34", "BiSeNet-34 (ONNX, parallel workers)"),
            ("bisenet_onnx_18", "BiSeNet-18 (ONNX, parallel + faster)"),
        ):
            self._occlusion_parser.addItem(label, value)
            if value == task.swapper_occlusion_parser:
                self._occlusion_parser.setCurrentIndex(
                    self._occlusion_parser.count() - 1
                )
        swap_form.addRow("Mask parser:", self._occlusion_parser)
        self._occluder_model = QComboBox()
        for value, label in (
            ("xseg_1", "XSeg 1"),
            ("xseg_2", "XSeg 2"),
            ("xseg_3", "XSeg 3"),
            ("xseg_many", "XSeg all three (strictest, 3x cost)"),
            ("depth", "Depth (experimental, closer-than-face)"),
        ):
            self._occluder_model.addItem(label, value)
            if value == task.swapper_occluder_model:
                self._occluder_model.setCurrentIndex(
                    self._occluder_model.count() - 1
                )
        swap_form.addRow("Occluder:", self._occluder_model)
        self._swapper_workers = QSpinBox()
        self._swapper_workers.setRange(1, 16)
        self._swapper_workers.setValue(task.swapper_execution.workers)
        self._swapper_workers.setToolTip(
            "Worker threads for the swap stage. The swapper shares one ONNX "
            "session across threads, so more workers cost little extra VRAM."
        )
        swap_form.addRow("Workers:", self._swapper_workers)
        # Swapper ONNX providers (multi-select). ORT tries them in the listed
        # order, falling back through to CPU. Unchecking all = platform default.
        providers_box = QWidget()
        providers_layout = QVBoxLayout(providers_box)
        providers_layout.setContentsMargins(0, 0, 0, 0)
        providers_layout.setSpacing(2)
        self._provider_checkboxes: dict[str, QCheckBox] = {}
        try:
            available = available_onnx_providers()
        except Exception:
            available = list(DEFAULT_ONNX_PROVIDERS)
        wanted = list(task.swapper_execution.providers)
        wanted_set = set(wanted)
        # Show the task's requested providers first (in priority order), then any
        # other provider this machine exposes. This keeps a requested-but-
        # unavailable EP (e.g. a CUDA task edited on a CPU-only box) AND its order
        # instead of silently dropping it — mirrors the torch-device preservation.
        ordered = wanted + [p for p in available if p not in wanted_set]
        for prov in ordered:
            cb = QCheckBox(prov)
            cb.setChecked(prov in wanted_set)
            providers_layout.addWidget(cb)
            self._provider_checkboxes[prov] = cb
        providers_box.setToolTip(
            "ONNX execution providers for the swap + detection models. ORT "
            "tries them in the order shown; uncheck all for platform defaults."
        )
        swap_form.addRow("ONNX providers:", providers_box)

        # ---- FaceEnhancer group ----
        enh_box = QGroupBox("Face enhancer")
        enh_box.setCheckable(True)
        enh_box.setChecked(task.enhancer_enabled)
        self._enhancer_box = enh_box
        enh_form = QFormLayout(enh_box)
        self._enhancer_model = QComboBox()
        for value, label in (
            ("gfpgan", "GFPGAN (whole-frame, Upscale knob)"),
            ("gfpgan_onnx", "GFPGAN (ONNX, much faster)"),
            ("codeformer", "CodeFormer (ONNX, fidelity knob)"),
            ("gpen_512", "GPEN-512 (ONNX, more detail)"),
            ("gpen_1024", "GPEN-1024 (ONNX, higher-res)"),
            ("gpen_2048", "GPEN-2048 (ONNX, highest-res; heavy)"),
            ("restoreformer_pp", "RestoreFormer++ (ONNX)"),
        ):
            self._enhancer_model.addItem(label, value)
            if value == task.enhancer_model:
                self._enhancer_model.setCurrentIndex(self._enhancer_model.count() - 1)
        self._enhancer_model.setToolTip(
            "Restoration model. GFPGAN upscales the whole frame; CodeFormer "
            "(ONNX) restores each face with a fidelity knob."
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
        enh_form.addRow("Only center face:", self._only_center_face)
        self._enhancer_fp16 = QCheckBox()
        self._enhancer_fp16.setChecked(task.enhancer_fp16)
        self._enhancer_fp16.setToolTip(
            "GFPGAN half precision: less VRAM per worker + faster. CUDA only; "
            "ignored by CodeFormer."
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
        # GFPGAN torch device (Auto / CPU / each CUDA GPU). Independent of the
        # swapper's ONNX providers — different framework.
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
        enh_form.addRow("Device:", self._enhancer_device)

        # ---- Upscaler group (Real-ESRGAN whole-frame super-resolution) ----
        up_box = QGroupBox("Frame upscaler (Real-ESRGAN)")
        up_box.setCheckable(True)
        up_box.setChecked(task.upscaler_enabled)
        self._upscaler_box = up_box
        up_form = QFormLayout(up_box)
        self._upscaler_model = QComboBox()
        for value, label in (
            ("general-x4v3", "Real-ESRGAN general x4 v3 (fast)"),
            ("x4plus", "Real-ESRGAN x4plus (higher quality)"),
            ("x2plus", "Real-ESRGAN x2plus"),
            ("swinir-m", "SwinIR x4 (sharp, slow)"),
            ("hat-x4", "HAT x4 (ONNX, very slow — stills only)"),
            ("ultrasharp-x4", "4x-UltraSharp (ONNX)"),
            ("span-x4", "SPAN x4 (ONNX, fast)"),
        ):
            self._upscaler_model.addItem(label, value)
            if value == task.upscaler_model:
                self._upscaler_model.setCurrentIndex(self._upscaler_model.count() - 1)
        up_form.addRow("Model:", self._upscaler_model)
        self._upscaler_tile = QSpinBox()
        self._upscaler_tile.setRange(0, 2048)
        self._upscaler_tile.setSingleStep(64)
        self._upscaler_tile.setValue(task.upscaler_tile)
        self._upscaler_tile.setToolTip("Tile size (px) to bound VRAM; 0 = whole frame.")
        up_form.addRow("Tile size:", self._upscaler_tile)
        self._upscaler_fp16 = QCheckBox()
        self._upscaler_fp16.setChecked(task.upscaler_fp16)
        up_form.addRow("Half precision:", self._upscaler_fp16)
        self._upscaler_device = QComboBox()
        for value, label in available_torch_devices():
            self._upscaler_device.addItem(label, value)
        up_current_device = task.upscaler_execution.device
        if self._upscaler_device.findData(up_current_device) < 0:
            self._upscaler_device.addItem(up_current_device, up_current_device)
        self._upscaler_device.setCurrentIndex(
            self._upscaler_device.findData(up_current_device)
        )
        up_form.addRow("Device:", self._upscaler_device)

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

        # ---- Standard OK / Cancel ----
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Stack the config groups in a SCROLLABLE area: the full form is taller
        # than short displays (~768 px laptops), so without this the dialog
        # opened bigger than the screen and OK/Cancel fell off the bottom. The
        # button box stays OUTSIDE the scroll area so it's always reachable.
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(paths_box)
        content_layout.addWidget(swap_box)
        content_layout.addWidget(enh_box)
        content_layout.addWidget(up_box)
        content_layout.addWidget(exec_box)
        content_layout.addWidget(out_box)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(button_box)

        # Open at the form's natural size but never taller than the screen —
        # the scroll area absorbs any overflow on small displays.
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            avail_h = screen.availableGeometry().height()
            self.setMaximumHeight(avail_h)
            desired_h = content.sizeHint().height() + button_box.sizeHint().height() + 24
            self.resize(640, min(desired_h, int(avail_h * 0.9)))

        # ---- Output default ----
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
        # DEBOUNCED: the probe opens the media file synchronously (VideoCapture
        # for videos), and textChanged fires per keystroke, so probing directly
        # stalled the GUI thread on every character of a pasted/typed path.
        # A single-shot timer restarted on each change probes only after
        # typing settles.
        self._probe_timer = QTimer(self)
        self._probe_timer.setSingleShot(True)
        self._probe_timer.setInterval(300)
        self._probe_timer.timeout.connect(self._refresh_scale_dims)
        self._target_edit.textChanged.connect(self._probe_timer.start)
        self._refresh_scale_dims()  # initial probe for the task's target
        self._update_enhancer_rows()  # gray out the inactive model's knob
        self._update_occlusion_rows()  # occlusion subknobs follow the checkbox
        self._update_rotation_rows()  # rotation knobs follow the toggle
        self._update_swapper_model_rows()  # fast-paste follows the swap model

    def _update_detector_rows(self) -> None:
        """Gray the gender filter for detection-only detectors (no .sex)."""
        self._target_sex.setEnabled(self._detector.currentData() == "buffalo_l")

    def _update_enhancer_rows(self) -> None:
        """Enable only the knobs the selected enhancer model uses — Upscale /
        fp16 / torch device for GFPGAN, Fidelity for CodeFormer; the ONNX
        restorers have none (they run on ORT EPs, not a torch device)."""
        model = self._enhancer_model.currentData()
        is_gfpgan = model == "gfpgan"
        self._upscale.setEnabled(is_gfpgan)
        self._enhancer_fidelity.setEnabled(model == "codeformer")
        self._enhancer_fp16.setEnabled(is_gfpgan)
        self._enhancer_device.setEnabled(is_gfpgan)

    def _update_occlusion_rows(self) -> None:
        """Link the occlusion sub-controls to the master checkbox and to each
        other: all gray when the mask is off; the parser applies to region/
        both, the occluder model to occluder/both."""
        on = self._occlusion_mask.isChecked()
        mode = self._occlusion_mode.currentData()
        self._occlusion_mode.setEnabled(on)
        self._occlusion_parser.setEnabled(on and mode != "occluder")
        self._occluder_model.setEnabled(on and mode != "region")

    def _update_rotation_rows(self) -> None:
        """Gray the rotation knobs when rotation compensation is off."""
        on = self._rotation_enabled.isChecked()
        self._rotation_threshold.setEnabled(on)
        self._rotation_redetect.setEnabled(on)
        self._rotation_source.setEnabled(on)

    def _update_swapper_model_rows(self) -> None:
        """Gray fast-paste for the 256px swappers (always fast-pasted); the
        toggle only applies to inswapper / reswapper."""
        from sinner2.pipeline.processors.swapper_models import (
            SwapperModel,
            is_insightface_model,
        )

        model = SwapperModel(self._swapper_model.currentData())
        self._fast_paste.setEnabled(is_insightface_model(model))

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
        timing). Runtime state isn't editable here."""
        # Untouched auto value (or empty) → keep output_path None so it stays
        # auto-derived (and follows source/target renames + global-output-dir
        # changes); a genuine edit is stored verbatim.
        output_str = self._output_edit.text().strip()
        output_path: Path | None
        if not output_str or output_str == self._auto_output:
            output_path = None
        else:
            output_path = Path(output_str)
        format_value = self._format_combo.currentData()
        return self._task.model_copy(
            update={
                "source_path": Path(self._source_edit.text()),
                "target_path": Path(self._target_edit.text()),
                "output_path": output_path,
                "output_format": BatchOutputFormat(format_value),
                "swapper_enabled": self._swapper_box.isChecked(),
                "swapper_model": self._swapper_model.currentData(),
                "swapper_detection_interval": self._detection_interval.value(),
                "swapper_detection_size": self._detection_size.value(),
                "swapper_detector": self._detector.currentData(),
                "swapper_many_faces": self._many_faces.isChecked(),
                "swapper_fast_paste": self._fast_paste.isChecked(),
                "swapper_target_sex": self._target_sex.currentData(),
                "swapper_rotation_compensation": self._rotation_enabled.isChecked(),
                "swapper_rotation_threshold_deg": self._rotation_threshold.value(),
                "swapper_rotation_redetect": self._rotation_redetect.isChecked(),
                "swapper_rotation_angle_source": self._rotation_source.currentData(),
                "swapper_occlusion_mask": self._occlusion_mask.isChecked(),
                "swapper_occlusion_mode": self._occlusion_mode.currentData(),
                "swapper_occlusion_parser": self._occlusion_parser.currentData(),
                "swapper_occluder_model": self._occluder_model.currentData(),
                "enhancer_enabled": self._enhancer_box.isChecked(),
                "enhancer_model": self._enhancer_model.currentData(),
                "enhancer_upscale": self._upscale.value(),
                "enhancer_only_center_face": self._only_center_face.isChecked(),
                "enhancer_codeformer_fidelity": self._enhancer_fidelity.value(),
                "enhancer_fp16": self._enhancer_fp16.isChecked(),
                "swapper_execution": self._task.swapper_execution.model_copy(
                    update={
                        "workers": self._swapper_workers.value(),
                        "providers": self._selected_providers(),
                    }
                ),
                "enhancer_execution": self._task.enhancer_execution.model_copy(
                    update={
                        "workers": self._enhancer_workers.value(),
                        "device": self._enhancer_device.currentData(),
                    }
                ),
                "upscaler_enabled": self._upscaler_box.isChecked(),
                "upscaler_model": self._upscaler_model.currentData(),
                "upscaler_tile": self._upscaler_tile.value(),
                "upscaler_fp16": self._upscaler_fp16.isChecked(),
                "upscaler_execution": self._task.upscaler_execution.model_copy(
                    update={"device": self._upscaler_device.currentData()}
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
            }
        )

    # ---- helpers ----

    def _selected_providers(self) -> list[str]:
        """Checked ONNX providers in display order. An empty selection is floored
        to CPU — you can't run an ONNX model on no provider."""
        selected = [
            name for name, cb in self._provider_checkboxes.items() if cb.isChecked()
        ]
        return selected or ["CPUExecutionProvider"]

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
    ) -> tuple[QLineEdit, QWidget]:
        """Build a (line-edit + browse-button) row. Returns the line
        edit and the composite container widget."""
        edit = QLineEdit(initial)
        btn = QPushButton("…")
        btn.setFixedWidth(28)

        def browse() -> None:
            if save_mode:
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
