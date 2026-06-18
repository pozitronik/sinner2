from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
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

from sinner2.config.execution import (
    DEFAULT_ONNX_PROVIDERS,
    available_torch_devices,
)
from sinner2.io.frame_resize import scaled_dims
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.model_cache import available_onnx_providers
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    FaceEnhancerParams,
)
from sinner2.pipeline.processors.occlusion import (
    FaceParser,
    OccluderModel,
    OcclusionMaskMode,
)
from sinner2.pipeline.processors.swapper_models import is_insightface_model
from sinner2.pipeline.processors.upscaler import (
    UpscalerModel,
    UpscalerParams,
    model_runtime,
    model_supports_fp16,
)
from sinner2.pipeline.detectors import DetectorModel
from sinner2.pipeline.processors.face_swapper import (
    FaceSwapperParams,
    RotationAngleSource,
    SwapperModel,
    TargetSex,
)
from sinner2.gui.processor_snapshot import ProcessorParamsSnapshot
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    PredictiveStrategy,
    SyncedStrategy,
)

# Responsive settings forms: every caption|control group shares ONE caption-
# column width. When the panel is narrower than (that shared width + room for a
# comfortable control + margins), all groups flip to stacking the control UNDER
# its caption so it gets the full row. _apply_form_density() drives the switch.
_COMFORTABLE_CONTROL_PX = 160
_FORM_OVERHEAD_PX = 40

# Ordered so the user sees the default (predictive) first — its combo index 0 is
# the startup selection when no strategy is persisted.
_STRATEGIES: dict[str, type[FrameSkipStrategy]] = {
    "Predictive (real-time, skip ahead)": PredictiveStrategy,
    "Best effort (process every frame, may lag)": BestEffortStrategy,
    "Synced (skip to match wall-clock)": SyncedStrategy,
}

# Display label → mode value. Ordered so the user sees the default first.
_PLAYBACK_MODES: dict[str, PlaybackMode] = {
    "Display: 30 Hz (smooth)": PlaybackMode.FIXED_30,
    "Display: source framerate": PlaybackMode.SOURCE,
    "Display: unlimited (no cap)": PlaybackMode.UNLIMITED,
}

_CACHE_MODES: dict[str, CacheMode] = {
    "Write + Read (default)": CacheMode.WRITE_READ,
    "Read only (don't write new)": CacheMode.READ_ONLY,
    "Off (memory only)": CacheMode.OFF,
}

_IMAGE_FORMATS: dict[str, ImageFormat] = {
    "JPEG (fast, small, lossy)": ImageFormat.JPEG,
    "PNG (slow, large, lossless)": ImageFormat.PNG,
}

_VIDEO_BACKENDS: dict[str, VideoBackend] = {
    "ffmpeg (subprocess pipe)": VideoBackend.FFMPEG,
    "cv2 (in-place seek; better on slow / network sources)": VideoBackend.CV2,
}

_ROTATION_SOURCES: list[tuple[str, str]] = [
    ("Eye keypoints", RotationAngleSource.KEYPOINTS.value),
    ("3D pose estimate", RotationAngleSource.POSE.value),
    ("2dfan4 landmarks", RotationAngleSource.LANDMARK_68.value),
]

_SWAPPER_MODELS: list[tuple[str, str]] = [
    (SwapperModel.INSWAPPER_128.value, "inswapper_128 (default, InsightFace)"),
    (SwapperModel.RESWAPPER_128.value, "ReSwapper 128 (open reproduction)"),
    (SwapperModel.GHOST_1_256.value, "Ghost 1 (256)"),
    (SwapperModel.GHOST_2_256.value, "Ghost 2 (256)"),
    (SwapperModel.GHOST_3_256.value, "Ghost 3 (256, heaviest)"),
    (SwapperModel.SIMSWAP_256.value, "SimSwap (256, non-commercial)"),
    (SwapperModel.UNIFACE_256.value, "UniFace (256, pose-aware)"),
    (SwapperModel.HYPERSWAP_1A_256.value, "Hyperswap 1a (256, newest)"),
    (SwapperModel.HYPERSWAP_1B_256.value, "Hyperswap 1b (256, newest)"),
]

_DETECTOR_MODELS: list[tuple[str, str]] = [
    (DetectorModel.BUFFALO_L.value, "buffalo_l (full pack, gender + pose)"),
    (DetectorModel.YOLOFACE.value, "YOLOFace 8n (fast, detection-only)"),
    (DetectorModel.SCRFD_2_5G.value, "SCRFD 2.5g (fast, detection-only)"),
]

_ENHANCER_MODELS: list[tuple[str, str]] = [
    (EnhancerModel.GFPGAN.value, "GFPGAN (whole-frame, Upscale knob)"),
    (EnhancerModel.GFPGAN_ONNX.value, "GFPGAN (ONNX, much faster)"),
    (EnhancerModel.CODEFORMER.value, "CodeFormer (ONNX, fidelity knob)"),
    (EnhancerModel.GPEN_512.value, "GPEN-512 (ONNX, more detail)"),
    (EnhancerModel.GPEN_1024.value, "GPEN-1024 (ONNX, higher-res)"),
    (EnhancerModel.GPEN_2048.value, "GPEN-2048 (ONNX, highest-res; heavy)"),
    (EnhancerModel.RESTOREFORMER_PP.value, "RestoreFormer++ (ONNX)"),
]

_UPSCALER_MODELS: list[tuple[str, str]] = [
    (UpscalerModel.GENERAL_X4V3.value, "Real-ESRGAN general x4 v3 (fast, small)"),
    (UpscalerModel.X4PLUS.value, "Real-ESRGAN x4plus (higher quality, heavy)"),
    (UpscalerModel.X2PLUS.value, "Real-ESRGAN x2plus"),
    (UpscalerModel.SWINIR_M.value, "SwinIR x4 (transformer, sharp, slow)"),
    (UpscalerModel.HAT_X4.value, "HAT x4 (ONNX, very slow — stills only)"),
    (UpscalerModel.ULTRASHARP_X4.value, "4x-UltraSharp (ONNX)"),
    (UpscalerModel.SPAN_X4.value, "SPAN x4 (ONNX, fast)"),
    (UpscalerModel.REAL_ESRGAN_X4_FP16.value, "Real-ESRGAN x4 fp16 (ONNX)"),
    (UpscalerModel.REAL_ESRGAN_X2_FP16.value, "Real-ESRGAN x2 fp16 (ONNX)"),
]

_OCCLUSION_PARSERS: list[tuple[str, str]] = [
    (FaceParser.BISENET.value, "BiSeNet (torch, sharper)"),
    (FaceParser.PARSENET.value, "ParseNet (torch, GFPGAN default)"),
    (FaceParser.BISENET_ONNX_34.value, "BiSeNet-34 (ONNX, parallel workers)"),
    (FaceParser.BISENET_ONNX_18.value, "BiSeNet-18 (ONNX, parallel + faster)"),
]

_OCCLUSION_MODES: list[tuple[str, str]] = [
    (OcclusionMaskMode.REGION.value, "Region (face parser)"),
    (OcclusionMaskMode.OCCLUDER.value, "Occluder (XSeg — sees hands/objects)"),
    (OcclusionMaskMode.BOTH.value, "Both (strictest)"),
]

_OCCLUDER_MODELS: list[tuple[str, str]] = [
    (OccluderModel.XSEG_1.value, "XSeg 1"),
    (OccluderModel.XSEG_2.value, "XSeg 2"),
    (OccluderModel.XSEG_3.value, "XSeg 3"),
    (OccluderModel.XSEG_MANY.value, "XSeg all three (strictest, 3x cost)"),
    (OccluderModel.DEPTH.value, "Depth (experimental, closer-than-face)"),
]


def _label_for_playback_mode(mode: PlaybackMode) -> str | None:
    for label, value in _PLAYBACK_MODES.items():
        if value is mode:
            return label
    return None


def _label_for_strategy_name(name: str) -> str | None:
    for label, cls in _STRATEGIES.items():
        if cls.__name__ == name:
            return label
    return None


def _label_for_cache_mode(mode: CacheMode) -> str | None:
    for label, value in _CACHE_MODES.items():
        if value is mode:
            return label
    return None


def _label_for_image_format(fmt: ImageFormat) -> str | None:
    for label, value in _IMAGE_FORMATS.items():
        if value is fmt:
            return label
    return None


def _label_for_video_backend(backend: VideoBackend) -> str | None:
    for label, value in _VIDEO_BACKENDS.items():
        if value is backend:
            return label
    return None


_TRT_PROVIDER = "TensorrtExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"
_TRT_TIP = (
    "TensorRT: compiles a GPU-specific engine for this model — typically 2–3×\n"
    "faster than plain CUDA. The FIRST run after enabling builds the engine\n"
    "(tens of seconds, one-time) and caches it to disk. Needs the TensorRT\n"
    "runtime; falls back to CUDA if it's missing."
)
_GENERIC_PROVIDER_TIP = (
    "ONNX execution provider. Multiple may be checked; ORT tries them in the\n"
    "order shown. You can't run on no provider — unchecking everything forces\n"
    "CPU back on (the floor). Applies immediately (rebuilds the session)."
)


def _short_provider_label(prov: str) -> str:
    """'CUDAExecutionProvider' → 'CUDA'; 'TensorrtExecutionProvider' → 'Tensor'."""
    name = prov.replace("ExecutionProvider", "")
    return "Tensor" if name == "Tensorrt" else name


def _set_combo_silently(combo: QComboBox, value: object) -> None:
    """Select the item whose data == ``value`` WITHOUT firing the combo's signals
    — the shared 'reflect/revert a selection programmatically' helper (used to
    revert a declined model download, etc.)."""
    combo.blockSignals(True)
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            break
    combo.blockSignals(False)


class _OnnxProvidersRow(QWidget):
    """The ``[ ]Tensor [ ]CUDA [ ]CPU`` checkbox strip for one ONNX-using
    processor — added to a QFormLayout with an "ONNX Providers" label so it sits
    in the field column like every other row. Forces CPU on when everything is
    unchecked (an ONNX model can't run on zero providers). ``changed`` fires on
    any user toggle."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._tooltips: dict[str, str] = {}
        try:
            available = available_onnx_providers()
        except Exception:  # noqa: BLE001 — broken ORT install → render the defaults
            available = list(DEFAULT_ONNX_PROVIDERS)
        default_active = set(DEFAULT_ONNX_PROVIDERS)
        for prov in available:
            tip = _TRT_TIP if prov == _TRT_PROVIDER else _GENERIC_PROVIDER_TIP
            self._tooltips[prov] = tip
            cb = QCheckBox(_short_provider_label(prov))
            cb.setToolTip(tip)
            cb.setChecked(prov in default_active)
            cb.toggled.connect(self._on_toggled)
            layout.addWidget(cb)
            self._checkboxes[prov] = cb
        layout.addStretch(1)

    def _force_cpu_if_empty(self) -> None:
        if any(cb.isChecked() for cb in self._checkboxes.values()):
            return
        cpu = self._checkboxes.get(_CPU_PROVIDER)
        if cpu is not None:
            cpu.blockSignals(True)
            cpu.setChecked(True)
            cpu.blockSignals(False)

    def _on_toggled(self) -> None:
        self._force_cpu_if_empty()
        self.changed.emit()

    def selected(self) -> list[str]:
        """Checked providers in the platform's preference order. Non-empty."""
        return [p for p, cb in self._checkboxes.items() if cb.isChecked()]

    def set_selected(self, providers: list[str]) -> None:
        """Reflect a restored selection WITHOUT firing ``changed``."""
        wanted = set(providers)
        for p, cb in self._checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(p in wanted)
            cb.blockSignals(False)
        self._force_cpu_if_empty()

    def checkboxes(self) -> dict[str, QCheckBox]:
        return self._checkboxes

    def mark_failed(self, failed: set[str]) -> None:
        """Red strikethrough on providers ORT couldn't initialise; empty clears."""
        for name, cb in self._checkboxes.items():
            if name in failed:
                cb.setStyleSheet(
                    "QCheckBox { color: #d94545; text-decoration: line-through; }"
                )
                cb.setToolTip(
                    f"{name} failed to initialise — ORT fell back to a\n"
                    "lower-priority provider (its runtime libs are likely missing)."
                )
            else:
                cb.setStyleSheet("")
                cb.setToolTip(self._tooltips.get(name, ""))


class QProcessorControls(QWidget):
    """Param editors for the v1 chain — FaceSwapper + FaceEnhancer.

    Emits configChanged whenever any field changes. The controller listens,
    rebuilds the chain (cheap — ONNX models are cached at module scope), and
    calls executor.set_chain() with new Processor instances.

    Cache storage actions emit their own dedicated signals (browseRequested,
    resetRootRequested, invalidateRequested, clearAllRequested,
    sizeCapChanged, rootChanged) so the main window can route them to the
    controller's cache-management API without going through configChanged.
    """

    configChanged = Signal()
    # View toggles (do NOT rebuild the session, unlike configChanged).
    faceOverlayToggled = Signal(bool)
    faceComparisonToggled = Signal(bool)
    useFaceMapToggled = Signal(bool)  # route playback through the target's face map
    openFaceMapRequested = Signal()   # jump to the Sources-tab face-map editor
    browseRootRequested = Signal()
    resetRootRequested = Signal()
    invalidateRequested = Signal()
    rerenderRequested = Signal()
    clearAllRequested = Signal()
    sizeCapChanged = Signal(object)  # int bytes (object to avoid C int overflow at 2 GB+); 0 = uncapped
    rootChanged = Signal(object)  # Path | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # While face-map routing is active the per-identity map decides what
        # swaps with what, so the gender filter / many-faces / detector choice are
        # superseded and grayed out. Tracked here so _update_detector_rows (which
        # also gates the gender filter) composes the two conditions.
        self._face_map_routing = False
        swapper_defaults = FaceSwapperParams()
        enhancer_defaults = FaceEnhancerParams()

        # Controls are grouped by stage, with SUB-GROUPS that keep each
        # dependency inside its own box (gender under the detector; rotation /
        # occlusion knobs under their master toggles). Built up front so widgets
        # can be added regardless of creation order.
        self._which_to_swap_box = QGroupBox("Which to swap")
        which_to_swap_form = QFormLayout(self._which_to_swap_box)
        self._rotation_box = QGroupBox("Rotation")
        rotation_form = QFormLayout(self._rotation_box)
        self._occlusion_box = QGroupBox("Occlusion")
        occlusion_form = QFormLayout(self._occlusion_box)

        swapper_box = QGroupBox("Face swap")
        swapper_box.setCheckable(True)
        swapper_box.setChecked(True)
        swapper_box.toggled.connect(self.configChanged)
        self._swapper_box = swapper_box
        swapper_form = QFormLayout(swapper_box)
        self._swapper_model = QComboBox()
        for value, label in _SWAPPER_MODELS:
            self._swapper_model.addItem(label, value)
            if value == swapper_defaults.model.value:
                self._swapper_model.setCurrentIndex(self._swapper_model.count() - 1)
        self._swapper_model.setToolTip(
            "Face-swap model. inswapper_128 (default) and ReSwapper are 128px;\n"
            "Ghost / SimSwap / UniFace are 256px alternatives with different\n"
            "trade-offs. Non-default weights download on first selection.\n"
            "SimSwap is CC-BY-NC (non-commercial)."
        )
        self._swapper_model.currentIndexChanged.connect(self.configChanged)
        self._swapper_model.currentIndexChanged.connect(
            self._update_swapper_model_rows
        )
        swapper_form.addRow("Model", self._swapper_model)
        self._many_faces = QCheckBox()
        self._many_faces.setChecked(swapper_defaults.many_faces)
        self._many_faces.toggled.connect(self.configChanged)
        # Face SELECTION (which detected faces to swap) — lives in the Faces
        # group's "Which to swap" sub-box, with the detector it depends on.
        which_to_swap_form.addRow("Many faces", self._many_faces)
        self._fast_paste = QCheckBox()
        self._fast_paste.setChecked(swapper_defaults.fast_paste)
        self._fast_paste.toggled.connect(self.configChanged)
        self._fast_paste.setToolTip(
            "Blend the swapped face back with the fast ROI feather paste\n"
            "(~2.7x faster per frame). Off = insightface's original\n"
            "full-frame diff-based blend (inswapper/reswapper only; the\n"
            "256px swappers always use the fast blend)."
        )
        swapper_form.addRow("Fast paste", self._fast_paste)
        self._target_sex = QComboBox()
        # itemData carries the single-letter token so settings round-trip
        # without coupling to the QComboBox index. Order: most-permissive
        # first.
        self._target_sex.addItem("Both (M+F)", TargetSex.BOTH.value)
        self._target_sex.addItem("Male only", TargetSex.MALE.value)
        self._target_sex.addItem("Female only", TargetSex.FEMALE.value)
        self._target_sex.addItem("As source face", TargetSex.AS_SOURCE.value)
        self._target_sex.setToolTip(
            "Which detected faces to swap based on insightface's sex\n"
            "classification.\n\n"
            "  Both: every face.\n"
            "  Male / Female: only faces classified that way.\n"
            "  As source face: match the gender of the loaded source\n"
            "  (lets you keep the same source across mixed-gender targets\n"
            "  by relying on the source's own classification).\n\n"
            "Faces whose sex insightface can't determine are SKIPPED for\n"
            "Male / Female / As-source (better to miss one face than\n"
            "swap the wrong gender)."
        )
        self._target_sex.currentIndexChanged.connect(self.configChanged)
        which_to_swap_form.addRow("Gender", self._target_sex)
        self._occlusion_mask = QCheckBox()
        self._occlusion_mask.setChecked(swapper_defaults.occlusion_mask)
        self._occlusion_mask.setToolTip(
            "Mask the swap to the real facial region (BiSeNet parse) so hair,\n"
            "glasses, hats and the neck/boundary keep the original. Affects\n"
            "output; the parser model downloads on first enable."
        )
        self._occlusion_mask.toggled.connect(self.configChanged)
        self._occlusion_mask.toggled.connect(self._update_occlusion_rows)
        occlusion_form.addRow("Occlusion mask", self._occlusion_mask)
        # The mode comes FIRST in the form — it decides which of the two
        # dependent rows below it (parser / occluder) apply.
        self._occlusion_mode = QComboBox()
        for value, label in _OCCLUSION_MODES:
            self._occlusion_mode.addItem(label, value)
            if value == swapper_defaults.occlusion_mode.value:
                self._occlusion_mode.setCurrentIndex(
                    self._occlusion_mode.count() - 1
                )
        self._occlusion_mode.setToolTip(
            "What builds the mask. Region: face-parser classes (hair/glasses/\n"
            "hats keep the original, but a hand over the cheek parses as skin\n"
            "and gets swapped over). Occluder: XSeg segmentation of the\n"
            "VISIBLE face — hands, mics, any object in front stays original.\n"
            "Both: strictest (facial region AND unoccluded)."
        )
        self._occlusion_mode.currentIndexChanged.connect(self.configChanged)
        self._occlusion_mode.currentIndexChanged.connect(
            self._update_occlusion_rows
        )
        occlusion_form.addRow("Mask source", self._occlusion_mode)
        self._occlusion_parser = QComboBox()
        for value, label in _OCCLUSION_PARSERS:
            self._occlusion_parser.addItem(label, value)
            if value == swapper_defaults.occlusion_parser.value:
                self._occlusion_parser.setCurrentIndex(
                    self._occlusion_parser.count() - 1
                )
        self._occlusion_parser.setToolTip(
            "Face parser for the occlusion mask. BiSeNet (the academic\n"
            "face-parser) tends to give sharper boundaries; ParseNet is the\n"
            "one GFPGAN/CodeFormer use. Try both. Each downloads on first use."
        )
        self._occlusion_parser.currentIndexChanged.connect(self.configChanged)
        occlusion_form.addRow("Mask parser", self._occlusion_parser)
        self._occluder_model = QComboBox()
        for value, label in _OCCLUDER_MODELS:
            self._occluder_model.addItem(label, value)
            if value == swapper_defaults.occluder_model.value:
                self._occluder_model.setCurrentIndex(
                    self._occluder_model.count() - 1
                )
        self._occluder_model.setToolTip(
            "Occluder model (used by Occluder / Both). The three XSeg\n"
            "trainings differ slightly per footage — try them; 'all three'\n"
            "min-combines them (strictest, 3x inference). ~67MB each,\n"
            "downloads on first use."
        )
        self._occluder_model.currentIndexChanged.connect(self.configChanged)
        occlusion_form.addRow("Occluder", self._occluder_model)

        # ONNX execution providers for the swapper — its own one-line selector in
        # the swap group. Detection runs on a process-WIDE shared insightface
        # model whose EPs are fixed at first load, so the detector necessarily
        # uses these too (it can't have a separate line).
        self._swapper_providers_row = _OnnxProvidersRow()
        self._swapper_providers_row.changed.connect(self.configChanged)
        swapper_form.addRow("ONNX Providers", self._swapper_providers_row)

        enhancer_box = QGroupBox("Face enhancer")
        enhancer_box.setCheckable(True)
        enhancer_box.setChecked(True)
        enhancer_box.toggled.connect(self.configChanged)
        enhancer_form = QFormLayout(enhancer_box)
        # Restoration backend: GFPGAN (PyTorch, whole-frame, Upscale knob) or
        # CodeFormer (ONNX, per-face, fidelity knob). The model-specific rows
        # (Upscale vs Fidelity) enable/disable to match the selection.
        self._enhancer_model = QComboBox()
        for value, label in _ENHANCER_MODELS:
            self._enhancer_model.addItem(label, value)
            if value == enhancer_defaults.model.value:
                self._enhancer_model.setCurrentIndex(
                    self._enhancer_model.count() - 1
                )
        self._enhancer_model.setToolTip(
            "Face-restoration model. GFPGAN restores the whole frame and can\n"
            "upscale; CodeFormer (ONNX) restores each detected face with a\n"
            "fidelity knob. CodeFormer's weights download on first enable."
        )
        self._enhancer_model.currentIndexChanged.connect(self.configChanged)
        self._enhancer_model.currentIndexChanged.connect(
            self._update_enhancer_model_rows
        )
        enhancer_form.addRow("Model", self._enhancer_model)
        self._upscale = QSpinBox()
        self._upscale.setRange(1, 4)
        self._upscale.setValue(enhancer_defaults.upscale)
        self._upscale.valueChanged.connect(self.configChanged)
        enhancer_form.addRow("Upscale", self._upscale)
        self._enhancer_fidelity = QDoubleSpinBox()
        self._enhancer_fidelity.setRange(0.0, 1.0)
        self._enhancer_fidelity.setSingleStep(0.1)
        self._enhancer_fidelity.setDecimals(2)
        self._enhancer_fidelity.setValue(enhancer_defaults.codeformer_fidelity)
        self._enhancer_fidelity.setToolTip(
            "CodeFormer fidelity w: 0 = max restoration (smoother, may drift\n"
            "from the input), 1 = max fidelity to the input (preserves detail\n"
            "but less cleanup). Ignored by GFPGAN."
        )
        self._enhancer_fidelity.valueChanged.connect(self.configChanged)
        enhancer_form.addRow("Fidelity (w)", self._enhancer_fidelity)
        self._only_center_face = QCheckBox()
        self._only_center_face.setChecked(enhancer_defaults.only_center_face)
        self._only_center_face.toggled.connect(self.configChanged)
        enhancer_form.addRow("Center face only", self._only_center_face)
        self._enhancer_fp16 = QCheckBox()
        self._enhancer_fp16.setChecked(enhancer_defaults.fp16)
        self._enhancer_fp16.setToolTip(
            "GFPGAN half precision: less VRAM per worker + faster (tensor "
            "cores).\nCUDA only; ignored by CodeFormer. Disable if you see "
            "artifacts."
        )
        self._enhancer_fp16.toggled.connect(self.configChanged)
        enhancer_form.addRow("Half precision", self._enhancer_fp16)
        # GFPGAN runs on PyTorch, so its device is torch's CUDA — independent
        # of the swapper's ONNX providers. Enumerate the actual devices (Auto,
        # CPU, then each CUDA GPU by name) so the user picks from what exists.
        self._enhancer_device = QComboBox()
        for value, label in available_torch_devices():
            self._enhancer_device.addItem(label, value)
        self._enhancer_device.setToolTip(
            "Torch device for GFPGAN. Auto picks CUDA when available, else\n"
            "CPU (much slower). Independent of the swapper's ONNX providers.\n"
            "Applies immediately — rebuilds the chain (reloads the model)."
        )
        self._enhancer_device.currentIndexChanged.connect(self.configChanged)
        enhancer_form.addRow("CUDA device", self._enhancer_device)
        # ONNX providers for the ONNX restorer backends (CodeFormer / GPEN /
        # RestoreFormer++ / GFPGAN-ONNX). Active only when an ONNX model is
        # chosen (torch GFPGAN uses the CUDA device above instead).
        self._enhancer_providers_row = _OnnxProvidersRow()
        self._enhancer_providers_row.changed.connect(self.configChanged)
        enhancer_form.addRow("ONNX Providers", self._enhancer_providers_row)
        self._enhancer_box = enhancer_box
        self._update_enhancer_model_rows()  # gray out the inactive model's knob

        # ---- Upscaler (Real-ESRGAN) — whole-frame super-resolution ----
        upscaler_defaults = UpscalerParams()
        upscaler_box = QGroupBox("Frame upscaler (Real-ESRGAN)")
        upscaler_box.setCheckable(True)
        upscaler_box.setChecked(False)  # opt-in (heavy; weights download on enable)
        upscaler_box.setToolTip(
            "Whole-frame super-resolution after the face stages. Heavy — at x4\n"
            "it quadruples the frame; best for batch / final output. Weights\n"
            "download on first enable."
        )
        upscaler_box.toggled.connect(self.configChanged)
        upscaler_form = QFormLayout(upscaler_box)
        self._upscaler_model = QComboBox()
        for value, label in _UPSCALER_MODELS:
            self._upscaler_model.addItem(label, value)
            if value == upscaler_defaults.model.value:
                self._upscaler_model.setCurrentIndex(
                    self._upscaler_model.count() - 1
                )
        self._upscaler_model.currentIndexChanged.connect(self.configChanged)
        self._upscaler_model.currentIndexChanged.connect(self._update_upscaler_rows)
        upscaler_form.addRow("Model", self._upscaler_model)
        self._upscaler_tile = QSpinBox()
        self._upscaler_tile.setRange(0, 2048)
        self._upscaler_tile.setSingleStep(64)
        self._upscaler_tile.setValue(upscaler_defaults.tile)
        self._upscaler_tile.setToolTip(
            "Tile size (px) to bound VRAM on large frames. 0 = whole frame at\n"
            "once. Raise (e.g. 256–512) if you hit out-of-memory."
        )
        self._upscaler_tile.valueChanged.connect(self.configChanged)
        upscaler_form.addRow("Tile size", self._upscaler_tile)
        self._upscaler_fp16 = QCheckBox()
        self._upscaler_fp16.setChecked(upscaler_defaults.fp16)
        self._upscaler_fp16.setToolTip("Half precision (faster, less VRAM, CUDA only).")
        self._upscaler_fp16.toggled.connect(self.configChanged)
        upscaler_form.addRow("Half precision", self._upscaler_fp16)
        self._upscaler_device = QComboBox()
        for value, label in available_torch_devices():
            self._upscaler_device.addItem(label, value)
        self._upscaler_device.setToolTip(
            "Torch device for the upscaler (independent of the enhancer's)."
        )
        self._upscaler_device.currentIndexChanged.connect(self.configChanged)
        upscaler_form.addRow("CUDA device", self._upscaler_device)
        # ONNX providers for the ONNX upscalers (HAT, fp16 exports). Active only
        # when an ONNX model is chosen (torch Real-ESRGAN uses the CUDA device).
        self._upscaler_providers_row = _OnnxProvidersRow()
        self._upscaler_providers_row.changed.connect(self.configChanged)
        upscaler_form.addRow("ONNX Providers", self._upscaler_providers_row)
        self._upscaler_box = upscaler_box

        # ---- Faces recognition group: detection + selection + the face-map
        # routing mode, together because they depend on each other (gender needs
        # the detector; the face-map mode overrides the detector + selection). ----
        face_box = QGroupBox("Faces recognition")
        face_form = QFormLayout(face_box)
        # Face-mapping routing switch at the TOP of the Faces group (decoupled
        # from the Sources-tab editor): always shown so the feature is
        # discoverable, enabled only once a map exists for the target. On →
        # playback routes each face to its mapped source; off → single source.
        # When on, it grays the detector + "Which to swap" — both inside THIS box.
        self._use_face_map = QCheckBox("Use face map")
        self._use_face_map.setEnabled(False)
        self._use_face_map.setToolTip(
            "Route playback through this target's face map — each person swapped "
            "with their mapped source — instead of the single global source. "
            "Build a map with 'Open face map' first; enabled once one exists. "
            "Remembered per target."
        )
        self._use_face_map.toggled.connect(self.useFaceMapToggled)
        self._open_face_map = QPushButton("Open face map…")
        self._open_face_map.setToolTip(
            "Open the face-map editor on the Sources tab to discover people and "
            "map each to a source."
        )
        self._open_face_map.clicked.connect(self.openFaceMapRequested)
        face_map_row = QHBoxLayout()
        face_map_row.addWidget(self._use_face_map)
        face_map_row.addWidget(self._open_face_map)
        face_map_row.addStretch(1)
        face_form.addRow("Face map", face_map_row)  # top of the Faces group
        self._detector = QComboBox()
        for value, label in _DETECTOR_MODELS:
            self._detector.addItem(label, value)
            if value == swapper_defaults.detector.value:
                self._detector.setCurrentIndex(self._detector.count() - 1)
        self._detector.setToolTip(
            "Target-face detector. buffalo_l runs InsightFace's full pack "
            "(the only one that yields gender + 3D pose). yoloface / scrfd are "
            "detection-only — faster (they skip the recognition/gender/landmark "
            "models per frame), but the gender filter is unavailable and "
            "rotation falls back to the keypoint angle. Downloads on first use."
        )
        self._detector.currentIndexChanged.connect(self.configChanged)
        self._detector.currentIndexChanged.connect(self._update_detector_rows)
        face_form.addRow("Detector", self._detector)
        self._detection_size = QSpinBox()
        # SCRFD strides are 8/16/32, so the detector input must be a multiple
        # of 32. Step by 32 so every value is valid; 640 is insightface's
        # default, smaller trades reach for speed.
        self._detection_size.setRange(128, 1280)
        self._detection_size.setSingleStep(32)
        self._detection_size.setValue(swapper_defaults.detection_size)
        self._detection_size.setToolTip(
            "Face-detector input size (px) for LIVE playback — separate from the "
            "Faces panel's scan size. Smaller = faster detection but may miss "
            "small or distant faces. 640 is the default; multiples of 32."
        )
        self._detection_size.valueChanged.connect(self.configChanged)
        face_form.addRow("Detection size", self._detection_size)
        self._detection_interval = QSpinBox()
        self._detection_interval.setRange(1, 30)
        self._detection_interval.setValue(swapper_defaults.detection_interval)
        self._detection_interval.setToolTip(
            "Detect every Nth frame and reuse the result on the frames between "
            "(1 = every frame). Higher = faster on stable shots; with multiple "
            "realtime workers prefer 1."
        )
        self._detection_interval.valueChanged.connect(self.configChanged)
        face_form.addRow("Detection interval", self._detection_interval)
        # "Which to swap" sub-box (many-faces + gender) sits under the detector
        # it depends on, inside the Faces group.
        face_form.addRow(self._which_to_swap_box)
        self._rotation_enabled = QCheckBox()
        self._rotation_enabled.setChecked(swapper_defaults.rotation_compensation)
        self._rotation_enabled.setToolTip(
            "Experimental: for faces tilted past the threshold, upright a crop,\n"
            "re-detect clean keypoints, swap, then composite back. Helps when\n"
            "the detector's keypoints degrade at high in-plane roll; does\n"
            "nothing for out-of-plane (profile) turns. Affects output."
        )
        self._rotation_enabled.toggled.connect(self.configChanged)
        self._rotation_enabled.toggled.connect(self._update_rotation_rows)
        # Rotation knobs live in the swap group's "Rotation" sub-box so the
        # angle-source/threshold dependencies stay inside it.
        rotation_form.addRow("Rotation compensation", self._rotation_enabled)

        self._rotation_threshold = QSpinBox()
        self._rotation_threshold.setRange(0, 90)
        self._rotation_threshold.setSuffix("°")
        self._rotation_threshold.setValue(swapper_defaults.rotation_threshold_deg)
        self._rotation_threshold.setToolTip(
            "Only compensate faces rolled at least this many degrees; below it, "
            "a plain swap."
        )
        self._rotation_threshold.valueChanged.connect(self.configChanged)
        rotation_form.addRow("Roll threshold", self._rotation_threshold)

        self._rotation_redetect = QCheckBox()
        self._rotation_redetect.setChecked(swapper_defaults.rotation_redetect)
        self._rotation_redetect.setToolTip(
            "Re-run detection on the uprighted crop for clean keypoints (vs.\n"
            "rotating the existing ones in). The main quality lever."
        )
        self._rotation_redetect.toggled.connect(self.configChanged)
        rotation_form.addRow("Re-detect uprighted", self._rotation_redetect)

        self._rotation_source = QComboBox()
        for label, value in _ROTATION_SOURCES:
            self._rotation_source.addItem(label, value)
            if value == swapper_defaults.rotation_angle_source.value:
                self._rotation_source.setCurrentIndex(
                    self._rotation_source.count() - 1
                )
        self._rotation_source.setToolTip(
            "How to measure in-plane roll: the eye keypoints (robust) or "
            "insightface's 3D pose estimate."
        )
        self._rotation_source.currentTextChanged.connect(
            lambda _: self.configChanged.emit()
        )
        rotation_form.addRow("Angle source", self._rotation_source)
        self._landmark_refine = QCheckBox()
        self._landmark_refine.setChecked(swapper_defaults.landmark_refine)
        self._landmark_refine.setToolTip(
            "Refine each detected face's keypoints with the 2dfan4 68-point\n"
            "landmarker before swapping — more accurate alignment on tilted /\n"
            "hard-pose faces (the detector's 5 points degrade there). The\n"
            "refined geometry also flows to the occlusion mask + enhancer.\n"
            "Experimental; downloads the 2dfan4 model on first enable."
        )
        self._landmark_refine.toggled.connect(self.configChanged)
        swapper_form.addRow("Landmark refine", self._landmark_refine)
        # The Rotation + Occlusion sub-boxes render inside the Face swap group.
        swapper_form.addRow(self._rotation_box)
        swapper_form.addRow(self._occlusion_box)

        # Detection overlay toggle (view-only → its own signal, never
        # configChanged). Boxes + sex/age/score/pose on the preview.
        self._overlay_enabled = QCheckBox()
        self._overlay_enabled.setToolTip(
            "Draw detected faces (box + keypoints + sex/age/score/pose) on the\n"
            "preview. Also toggles with F8."
        )
        self._overlay_enabled.toggled.connect(self.faceOverlayToggled)
        self._overlay_enabled.toggled.connect(self._couple_overlay_to_comparison)
        face_form.addRow("Show detection overlay", self._overlay_enabled)

        # Comparison overlay (a view toggle, NOT a chain param → its own signal,
        # never configChanged). Shows [orig | swapped] thumbnails next to each
        # face; it draws ON the detection overlay, so the two are linked:
        # enabling comparison turns the overlay on, and turning the overlay off
        # turns comparison off (see the couple_* handlers).
        self._comparison_enabled = QCheckBox()
        self._comparison_enabled.setToolTip(
            "Show original vs swapped face thumbnails next to each detected\n"
            "face on the preview. Turns on the face-detection overlay (they're\n"
            "linked); also needs the swapper on."
        )
        # Couple FIRST so the overlay is already on by the time the comparison
        # signal reaches the window — otherwise it transiently warns "needs the
        # overlay" before the coupling enables it.
        self._comparison_enabled.toggled.connect(self._couple_comparison_to_overlay)
        self._comparison_enabled.toggled.connect(self.faceComparisonToggled)
        swapper_form.addRow("Show orig/swapped", self._comparison_enabled)
        self._face_box = face_box
        self._update_rotation_rows()  # reflect the default rotation-on state
        self._update_occlusion_rows()  # occlusion subknobs follow the checkbox
        self._update_swapper_model_rows()  # fast-paste follows the swap model
        self._update_detector_rows()  # gray gender filter for detection-only
        self._update_upscaler_rows()  # gray fp16 for ONNX upscalers

        execution_box = QGroupBox("Execution")
        execution_form = QFormLayout(execution_box)
        self._strategy_combo = QComboBox()
        for label in _STRATEGIES:
            self._strategy_combo.addItem(label)
        self._strategy_combo.currentTextChanged.connect(
            lambda _: (self._update_strategy_param_enabled(), self.configChanged.emit())
        )
        execution_form.addRow("Frame-skip strategy", self._strategy_combo)

        self._synced_max_lag_frames = QSpinBox()
        self._synced_max_lag_frames.setRange(5, 1000)
        self._synced_max_lag_frames.setValue(60)
        self._synced_max_lag_frames.setSuffix(" frames")
        self._synced_max_lag_frames.setToolTip(
            "Synced strategy only. When processing falls this many frames\n"
            "behind the timeline, Synced gives up trying to jump ahead and\n"
            "submits frames sequentially (like BestEffort). Prevents the\n"
            "'death-spiral' on slow sources where each jump-ahead is a\n"
            "random seek the reader can't service in time. Smaller value =\n"
            "fall back sooner; larger = try harder to keep up. Default 60\n"
            "(~2s at 30 fps)."
        )
        self._synced_max_lag_frames.valueChanged.connect(self.configChanged)
        execution_form.addRow("Synced lag threshold", self._synced_max_lag_frames)

        self._predictive_max_lead_seconds = QDoubleSpinBox()
        self._predictive_max_lead_seconds.setRange(0.0, 10.0)
        self._predictive_max_lead_seconds.setSingleStep(0.1)
        self._predictive_max_lead_seconds.setValue(1.0)
        self._predictive_max_lead_seconds.setSuffix(" s")
        self._predictive_max_lead_seconds.setToolTip(
            "Predictive strategy only. The strategy aims each frame at where the\n"
            "playhead will be when it finishes processing, so a slow pipeline\n"
            "plays in real time (showing every Nth frame) instead of slow-motion.\n"
            "This caps how far ahead (seconds) that aim may reach: higher =\n"
            "tighter sync on very slow pipelines, but more aggressive skipping\n"
            "and random reads. Default 1.0 s."
        )
        self._predictive_max_lead_seconds.valueChanged.connect(self.configChanged)
        execution_form.addRow(
            "Predictive max lead", self._predictive_max_lead_seconds
        )

        self._worker_count = QSpinBox()
        # Upper bound matches RealtimeExecutor.MAX_WORKERS; the executor's
        # work queue is pre-sized at MAX_WORKERS * 2 so the pool can grow
        # to this limit without reallocating the queue.
        self._worker_count.setRange(1, 16)
        self._worker_count.setValue(1)
        self._worker_count.setToolTip(
            "Number of parallel pipeline workers sharing one chain. ORT runs\n"
            "concurrent inference on the same models, so more workers raise\n"
            "GPU utilization without multiplying memory. Changes apply\n"
            "immediately — the pool grows or shrinks live, no model reload."
        )
        self._worker_count.valueChanged.connect(self.configChanged)
        execution_form.addRow("Realtime workers", self._worker_count)

        self._reader_pool_size = QSpinBox()
        self._reader_pool_size.setRange(1, 16)
        self._reader_pool_size.setValue(1)
        self._reader_pool_size.setToolTip(
            "Number of parallel source readers (independent of worker count).\n"
            "1 = current behaviour: one reader, sequential. Raise to 4-8 for\n"
            "slow sources (network share, HDD) — each reader serves one\n"
            "concurrent I/O request, so N readers pipeline N reads against\n"
            "the network. Cheap with the cv2 backend (in-place seek);\n"
            "ffmpeg + sequential workloads may regress (decoder restarts) —\n"
            "keep at 1 there. Applies immediately — rebuilds the session."
        )
        self._reader_pool_size.valueChanged.connect(self.configChanged)
        execution_form.addRow("Reader pool size", self._reader_pool_size)

        # Processing scale: downscale frames before the chain for speed. The
        # slider drives a percent; the label shows the percent + the resulting
        # WxH for the loaded target (recomputed live, no session rebuild). The
        # rebuild only fires on release — dragging would otherwise rebuild the
        # session on every pixel of travel.
        self._target_native_size: tuple[int, int] | None = None
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(10, 100)  # 10%..100%
        self._scale_slider.setValue(100)
        self._scale_slider.setToolTip(
            "Downscale frames before processing. Smaller frames decode,\n"
            "detect, swap, enhance and encode faster — the speed lever for\n"
            "weak hardware. Output is the reduced resolution. 100% = full\n"
            "resolution (off). Applies on release — rebuilds the session\n"
            "(current frame and play state preserved)."
        )
        self._scale_label = QLabel()
        self._scale_label.setMinimumWidth(110)
        self._scale_slider.valueChanged.connect(self._update_scale_label)
        self._scale_slider.sliderReleased.connect(self.configChanged)
        scale_row = QWidget()
        scale_row_layout = QHBoxLayout(scale_row)
        scale_row_layout.setContentsMargins(0, 0, 0, 0)
        scale_row_layout.addWidget(self._scale_slider, stretch=1)
        scale_row_layout.addWidget(self._scale_label)
        execution_form.addRow("Processing scale", scale_row)
        self._update_scale_label()

        self._playback_combo = QComboBox()
        for label in _PLAYBACK_MODES:
            self._playback_combo.addItem(label)
        self._playback_combo.setToolTip(
            "How fast the display thread polls for new frames.\n"
            "30 Hz: capped for smooth playback, low CPU (default).\n"
            "Source: match the source video's framerate.\n"
            "Unlimited: poll as fast as possible — better seek response."
        )
        self._playback_combo.currentTextChanged.connect(lambda _: self.configChanged.emit())
        execution_form.addRow("Playback rate", self._playback_combo)

        self._video_backend_combo = QComboBox()
        for label in _VIDEO_BACKENDS:
            self._video_backend_combo.addItem(label)
        self._video_backend_combo.setToolTip(
            "Video reader backend for video targets.\n"
            "ffmpeg: persistent subprocess, fast strictly-sequential reads,\n"
            "  but RESTARTS on any out-of-order read — expensive on slow /\n"
            "  network sources and with SyncedStrategy.\n"
            "cv2: persistent VideoCapture with in-place seek — vastly\n"
            "  faster scrubbing and Synced on slow sources.\n"
            "Applies immediately — rebuilds the session (chain reloads,\n"
            "current frame and play state are preserved)."
        )
        self._video_backend_combo.currentTextChanged.connect(lambda _: self.configChanged.emit())
        execution_form.addRow("Video backend", self._video_backend_combo)

        cache_box = QGroupBox("Cache")
        cache_form = QFormLayout(cache_box)
        # cache_mode is the only knob that hot-swaps live; the others apply
        # on next session start (reload source or target). Tooltips note this.
        self._cache_mode_combo = QComboBox()
        for label in _CACHE_MODES:
            self._cache_mode_combo.addItem(label)
        self._cache_mode_combo.setToolTip(
            "Cache behaviour. Hot-swaps live.\n"
            "Write+Read: normal (cache writes + reads from disk).\n"
            "Read only: don't write new frames, still read existing.\n"
            "Off: memory only — no cross-session warmup, no disk fallback."
        )
        self._cache_mode_combo.currentTextChanged.connect(lambda _: self.configChanged.emit())
        cache_form.addRow("Cache mode", self._cache_mode_combo)

        self._image_format_combo = QComboBox()
        for label in _IMAGE_FORMATS:
            self._image_format_combo.addItem(label)
        self._image_format_combo.setToolTip(
            "Encoding format for cached frames.\n"
            "JPEG: ~10x faster encode than PNG, ~5x smaller files, lossy.\n"
            "PNG: lossless. Slow — can cap throughput on FHD/4K.\n"
            "Changes apply on next session start (reload source or target)."
        )
        self._image_format_combo.currentTextChanged.connect(
            lambda _: (self._update_quality_visibility(), self.configChanged.emit())
        )
        cache_form.addRow("Image format", self._image_format_combo)

        self._image_quality = QSpinBox()
        self._image_quality.setRange(1, 100)
        self._image_quality.setValue(95)
        self._image_quality.setToolTip(
            "JPEG encode quality (1=worst, 100=best). 95 is visually\n"
            "indistinguishable for most content. Has no effect when PNG\n"
            "is selected (PNG uses a fixed fast-compression setting)."
        )
        self._image_quality.valueChanged.connect(self.configChanged)
        cache_form.addRow("JPEG quality", self._image_quality)

        self._memory_cache_mb = QSpinBox()
        self._memory_cache_mb.setRange(16, 8192)
        self._memory_cache_mb.setSuffix(" MB")
        self._memory_cache_mb.setValue(128)
        self._memory_cache_mb.setToolTip(
            "In-memory LRU cache size for processed frames. Bigger cache =\n"
            "more frames hot for scrubbing. FHD ~6 MB/frame, 4K ~25 MB/frame.\n"
            "Changes apply on next session start (reload source or target)."
        )
        self._memory_cache_mb.valueChanged.connect(self.configChanged)
        cache_form.addRow("Memory cache", self._memory_cache_mb)

        self._write_workers = QSpinBox()
        self._write_workers.setRange(1, 16)
        self._write_workers.setValue(4)
        self._write_workers.setToolTip(
            "Disk-write thread pool size. More workers = higher write throughput.\n"
            "JPEG benefits up to ~4-8 workers; PNG benefits up to 8+ on fast disk.\n"
            "Changes apply on next session start (reload source or target)."
        )
        self._write_workers.valueChanged.connect(self.configChanged)
        cache_form.addRow("Write workers", self._write_workers)

        self._write_queue_size = QSpinBox()
        self._write_queue_size.setRange(1, 256)
        self._write_queue_size.setValue(8)
        self._write_queue_size.setToolTip(
            "Max outstanding write tasks. When this many are queued, new\n"
            "writes are dropped (the frame stays in memory cache only,\n"
            "and may need reprocessing on backward seek if evicted). Drop\n"
            "count surfaces in metrics so you can see when disk is the\n"
            "bottleneck. Changes apply on next session start."
        )
        self._write_queue_size.valueChanged.connect(self.configChanged)
        cache_form.addRow("Write queue size", self._write_queue_size)

        self._update_quality_visibility()
        self._update_strategy_param_enabled()

        cache_storage_box = QGroupBox("Cache storage")
        storage_layout = QVBoxLayout(cache_storage_box)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Location:"))
        self._cache_root_edit = QLineEdit()
        self._cache_root_edit.setReadOnly(True)
        self._cache_root_edit.setToolTip(
            "Where processed-frame caches live. Each session creates a\n"
            "subdirectory keyed by source+target+chain+format. Changing\n"
            "the location applies on next session start; existing caches\n"
            "are not migrated."
        )
        root_row.addWidget(self._cache_root_edit, stretch=1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browseRootRequested)
        root_row.addWidget(browse_btn)
        reset_btn = QPushButton("Default")
        reset_btn.clicked.connect(self.resetRootRequested)
        root_row.addWidget(reset_btn)
        storage_layout.addLayout(root_row)

        cap_row = QHBoxLayout()
        self._size_cap_enabled = QCheckBox("Hard cap")
        self._size_cap_enabled.setToolTip(
            "When enabled, evict the oldest cache entries at the start of\n"
            "each session until the total size is under the cap.\n"
            "Off by default — user is in control."
        )
        self._size_cap_enabled.toggled.connect(self._on_size_cap_changed)
        cap_row.addWidget(self._size_cap_enabled)
        self._size_cap_mb = QSpinBox()
        self._size_cap_mb.setRange(100, 1024 * 1024)  # 100 MB .. 1 TB
        self._size_cap_mb.setSuffix(" MB")
        self._size_cap_mb.setValue(2048)
        self._size_cap_mb.setEnabled(False)
        self._size_cap_mb.valueChanged.connect(self._on_size_cap_changed)
        cap_row.addWidget(self._size_cap_mb)
        cap_row.addStretch()
        storage_layout.addLayout(cap_row)

        self._cache_stats_label = QLabel("Stats: (no session)")
        self._cache_stats_label.setToolTip(
            "Total cache entries · combined size · free space on the cache volume.\n"
            "Refreshes when sessions start/stop or when you clear the cache."
        )
        storage_layout.addWidget(self._cache_stats_label)

        button_row = QHBoxLayout()
        self._invalidate_btn = QPushButton("Invalidate current session")
        self._invalidate_btn.setToolTip(
            "Drop the current session's cached frames so they're reprocessed.\n"
            "Useful when you've changed something not captured in the cache key\n"
            "(e.g. an underlying model file) and need a clean reprocess."
        )
        self._invalidate_btn.setEnabled(False)
        self._invalidate_btn.clicked.connect(self.invalidateRequested)
        button_row.addWidget(self._invalidate_btn)
        self._rerender_btn = QPushButton("Re-render from here")
        self._rerender_btn.setToolTip(
            "Reprocess from the current frame forward through the chain, so a\n"
            "parameter change you just made applies to frames you've already\n"
            "passed. Frames before the playhead keep their cached pixels."
        )
        self._rerender_btn.setEnabled(False)
        self._rerender_btn.clicked.connect(self.rerenderRequested)
        button_row.addWidget(self._rerender_btn)
        self._clear_all_btn = QPushButton("Clear all caches")
        self._clear_all_btn.setToolTip(
            "Delete every cache entry under the cache root, sparing the\n"
            "currently active session. Requires confirmation."
        )
        self._clear_all_btn.clicked.connect(self.clearAllRequested)
        button_row.addWidget(self._clear_all_btn)
        storage_layout.addLayout(button_row)

        # Inner container holds every group; the outer layout is just a
        # QScrollArea wrapper so all groups remain reachable on small windows.
        inner = QWidget()
        # Low minimum so the panel can be dragged genuinely narrow; the
        # responsive form layout (_apply_form_density) keeps controls usable
        # at small widths by stacking captions above them.
        inner.setMinimumWidth(240)
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        # Face detector first — detection precedes the swap (can't swap a face
        # you didn't find).
        # Captured so live mode can hide the file-only groups (the camera path
        # has no timeline cache, reader pool, processing scale, or video backend;
        # its worker count lives in the Live tab instead — see set_file_only_visible).
        self._execution_box = execution_box
        self._cache_box = cache_box
        self._cache_storage_box = cache_storage_box
        # Cache + Cache-storage live in a separate "Cache settings…" dialog (kept
        # off the main scroll); a button opens it. Both are file-only.
        self._cache_dialog = QDialog(self)
        self._cache_dialog.setWindowTitle("Cache settings")
        cache_dialog_layout = QVBoxLayout(self._cache_dialog)
        cache_dialog_layout.addWidget(cache_box)
        cache_dialog_layout.addWidget(cache_storage_box)
        cache_dialog_layout.addStretch(1)
        self._cache_settings_btn = QPushButton("Cache settings…")
        self._cache_settings_btn.setToolTip(
            "Open the cache + cache-storage settings (mode, image format/quality, "
            "memory budget, write workers, on-disk usage + clear)."
        )
        self._cache_settings_btn.clicked.connect(self._open_cache_settings)
        # Assembled in pipeline order: Faces (face-map + detection + selection) →
        # Face swap → enhance → upscale → file-only Execution + the cache button.
        inner_layout.addWidget(face_box)              # Faces
        inner_layout.addWidget(swapper_box)           # Face swap
        inner_layout.addWidget(enhancer_box)
        inner_layout.addWidget(upscaler_box)
        inner_layout.addWidget(execution_box)
        inner_layout.addWidget(self._cache_settings_btn)
        inner_layout.addStretch()

        scroll = QScrollArea(self)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # All caption|control groups share one responsive layout: a common
        # caption-column width, and a single width breakpoint that flips them
        # ALL between side-by-side and stacked together (no per-row jank).
        # Include the NESTED subgroup forms too (Which to swap / Rotation /
        # Occlusion) so their caption column lines up at the same width as the
        # top-level groups — otherwise they size to their own short labels and
        # read as a different column width.
        self._forms = [
            swapper_form, enhancer_form, upscaler_form, face_form,
            execution_form, cache_form,
            which_to_swap_form, rotation_form, occlusion_form,
        ]
        self._uniform_label_width = self._compute_uniform_label_width()
        self._apply_form_density()

    def _open_cache_settings(self) -> None:
        """Open the cache-settings dialog (modeless, so cache changes preview
        live)."""
        self._cache_dialog.show()
        self._cache_dialog.raise_()
        self._cache_dialog.activateWindow()

    def set_file_only_visible(self, visible: bool) -> None:
        """Show/hide the file-only surface (Execution group + the Cache-settings
        button). Live mode hides them: a camera has no timeline cache, reader
        pool, processing scale, or video backend, and its worker count lives in
        the Live tab. Closing the cache dialog too keeps it from lingering."""
        self._execution_box.setVisible(visible)
        self._cache_settings_btn.setVisible(visible)
        if not visible:
            self._cache_dialog.hide()

    # ---- Responsive form density (consistent + adaptive caption columns) ----

    def _compute_uniform_label_width(self) -> int:
        """Widest caption across ALL groups, so every group's caption column
        lines up at the same width in side-by-side mode."""
        width = 0
        for form in self._forms:
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item is not None and item.widget() is not None:
                    width = max(width, item.widget().sizeHint().width())
        return width

    def _apply_form_density(self) -> None:
        """Switch ALL groups between side-by-side and stacked at once, by the
        panel width. Wide: caption | control with the shared caption column.
        Narrow: caption above control, so the control gets the full row."""
        breakpoint_px = (
            self._uniform_label_width + _COMFORTABLE_CONTROL_PX + _FORM_OVERHEAD_PX
        )
        stacked = self.width() < breakpoint_px
        policy = (
            QFormLayout.RowWrapPolicy.WrapAllRows
            if stacked
            else QFormLayout.RowWrapPolicy.DontWrapRows
        )
        label_width = 0 if stacked else self._uniform_label_width
        for form in self._forms:
            form.setRowWrapPolicy(policy)
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item is not None and item.widget() is not None:
                    item.widget().setMinimumWidth(label_width)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._apply_form_density()

    def _update_quality_visibility(self) -> None:
        # Quality control is JPEG-only; disable when PNG is selected so
        # the user doesn't think it's doing something.
        is_jpeg = self.image_format() is ImageFormat.JPEG
        self._image_quality.setEnabled(is_jpeg)

    def _update_strategy_param_enabled(self) -> None:
        # Each strategy's tuning row is only meaningful when it's the active one.
        cls = _STRATEGIES[self._strategy_combo.currentText()]
        self._synced_max_lag_frames.setEnabled(cls is SyncedStrategy)
        self._predictive_max_lead_seconds.setEnabled(cls is PredictiveStrategy)

    def _on_size_cap_changed(self, *_: object) -> None:
        # Recompute the effective cap and notify. The cap is 0 (uncapped)
        # when the checkbox is off, regardless of the spinbox value.
        enabled = self._size_cap_enabled.isChecked()
        self._size_cap_mb.setEnabled(enabled)
        bytes_cap = (
            self._size_cap_mb.value() * 1024 * 1024 if enabled else 0
        )
        self.sizeCapChanged.emit(bytes_cap)

    # ---- Cache panel state setters (called by main window) ----

    def set_cache_root_text(self, path: Path) -> None:
        self._cache_root_edit.setText(str(path))

    def set_cache_size_cap_bytes(self, max_bytes: int) -> None:
        """Apply a persisted cap value to the widget without re-emitting
        sizeCapChanged. Used on startup restore."""
        self._size_cap_enabled.blockSignals(True)
        self._size_cap_mb.blockSignals(True)
        try:
            if max_bytes > 0:
                self._size_cap_enabled.setChecked(True)
                self._size_cap_mb.setEnabled(True)
                self._size_cap_mb.setValue(max(100, max_bytes // (1024 * 1024)))
            else:
                self._size_cap_enabled.setChecked(False)
                self._size_cap_mb.setEnabled(False)
        finally:
            self._size_cap_enabled.blockSignals(False)
            self._size_cap_mb.blockSignals(False)

    def set_cache_stats_text(self, text: str) -> None:
        self._cache_stats_label.setText(text)

    def set_invalidate_enabled(self, enabled: bool) -> None:
        # Both session-scoped actions share the "a session is active" gate.
        self._invalidate_btn.setEnabled(enabled)
        self._rerender_btn.setEnabled(enabled)

    def cache_size_cap_bytes(self) -> int:
        if not self._size_cap_enabled.isChecked():
            return 0
        return self._size_cap_mb.value() * 1024 * 1024

    def swapper_params(self) -> FaceSwapperParams:
        # Combo's currentData is the str token, not the enum — Qt's
        # QVariant flattens str-Enum to str (see library sort fix). The
        # FaceSwapperParams model coerces back via str-Enum membership.
        return FaceSwapperParams(
            model=SwapperModel(self._swapper_model.currentData()),
            detection_interval=self._detection_interval.value(),
            detection_size=self._detection_size.value(),
            detector=DetectorModel(self._detector.currentData()),
            many_faces=self._many_faces.isChecked(),
            fast_paste=self._fast_paste.isChecked(),
            target_sex=self._target_sex.currentData(),
            rotation_compensation=self._rotation_enabled.isChecked(),
            rotation_threshold_deg=self._rotation_threshold.value(),
            rotation_redetect=self._rotation_redetect.isChecked(),
            rotation_angle_source=self._rotation_source.currentData(),
            landmark_refine=self._landmark_refine.isChecked(),
            occlusion_mask=self._occlusion_mask.isChecked(),
            occlusion_mode=self._occlusion_mode.currentData(),
            occlusion_parser=self._occlusion_parser.currentData(),
            occluder_model=self._occluder_model.currentData(),
        )

    def swapper_model(self) -> str:
        return self._swapper_model.currentData()

    def set_swapper_model(self, value: str) -> None:
        """Set the swap model WITHOUT firing configChanged — used to revert the
        selection when the user declines the model download."""
        _set_combo_silently(self._swapper_model, value)

    def set_swapper_detector(self, value: str) -> None:
        """Set the detector WITHOUT firing configChanged — used to revert the
        selection when the user declines the detector-model download."""
        _set_combo_silently(self._detector, value)
        self._update_detector_rows()

    def set_occlusion_checked(self, on: bool) -> None:
        """Reflect occlusion-on without firing configChanged — used to revert
        the toggle when the user declines the parser-model download."""
        self._occlusion_mask.blockSignals(True)
        self._occlusion_mask.setChecked(bool(on))
        self._occlusion_mask.blockSignals(False)
        self._update_occlusion_rows()  # blocked signal → refresh manually

    def disable_landmark_refine(self) -> None:
        """Turn landmark refinement off without firing configChanged — used to
        revert when the user declines the 2dfan4 model download. Also drops the
        landmark-68 angle source back to pose (it needs the same model)."""
        self._landmark_refine.blockSignals(True)
        self._landmark_refine.setChecked(False)
        self._landmark_refine.blockSignals(False)
        if self._rotation_source.currentData() == RotationAngleSource.LANDMARK_68.value:
            _set_combo_silently(self._rotation_source, RotationAngleSource.POSE.value)

    def _update_enhancer_model_rows(self) -> None:
        """Enable only the knob that applies to the selected enhancer model —
        Upscale for GFPGAN, Fidelity for CodeFormer; GPEN / RestoreFormer++
        have neither."""
        model = self._enhancer_model.currentData()
        is_gfpgan = model == EnhancerModel.GFPGAN.value
        self._upscale.setEnabled(is_gfpgan)
        self._enhancer_fidelity.setEnabled(model == EnhancerModel.CODEFORMER.value)
        # fp16 + CUDA device are GFPGAN-only (torch). The ONNX restorers
        # (CodeFormer / GPEN / RestoreFormer++ / GFPGAN-ONNX) use the ONNX
        # providers row instead — so the two are mutually exclusive by model.
        self._enhancer_fp16.setEnabled(is_gfpgan)
        self._enhancer_device.setEnabled(is_gfpgan)
        self._enhancer_providers_row.setEnabled(not is_gfpgan)

    def enhancer_model(self) -> str:
        return self._enhancer_model.currentData()

    def set_enhancer_model(self, value: str) -> None:
        """Set the enhancer model WITHOUT firing configChanged — used to revert
        the selection when the user declines the CodeFormer model download."""
        _set_combo_silently(self._enhancer_model, value)
        self._update_enhancer_model_rows()

    def enhancer_params(self) -> FaceEnhancerParams:
        # Rotation compensation is shared config — the Face-detector group's
        # controls drive both the swapper and the enhancer.
        return FaceEnhancerParams(
            model=EnhancerModel(self._enhancer_model.currentData()),
            upscale=self._upscale.value(),
            only_center_face=self._only_center_face.isChecked(),
            codeformer_fidelity=self._enhancer_fidelity.value(),
            fp16=self._enhancer_fp16.isChecked(),
            rotation_compensation=self._rotation_enabled.isChecked(),
            rotation_threshold_deg=self._rotation_threshold.value(),
            rotation_redetect=self._rotation_redetect.isChecked(),
            rotation_angle_source=self._rotation_source.currentData(),
        )

    def enhancer_enabled(self) -> bool:
        return self._enhancer_box.isChecked()

    def upscaler_enabled(self) -> bool:
        return self._upscaler_box.isChecked()

    def set_upscaler_checked(self, on: bool) -> None:
        """Reflect upscaler-enabled WITHOUT firing configChanged — used to
        revert the toggle when the user declines the model download."""
        self._upscaler_box.blockSignals(True)
        self._upscaler_box.setChecked(bool(on))
        self._upscaler_box.blockSignals(False)

    def upscaler_params(self) -> UpscalerParams:
        return UpscalerParams(
            model=self._upscaler_model.currentData(),
            tile=self._upscaler_tile.value(),
            fp16=self._upscaler_fp16.isChecked(),
        )

    def upscaler_device(self) -> str:
        return self._upscaler_device.currentData()

    def swapper_enabled(self) -> bool:
        return self._swapper_box.isChecked()

    def face_overlay_enabled(self) -> bool:
        return self._overlay_enabled.isChecked()

    def set_overlay_checked(self, on: bool) -> None:
        """Reflect overlay-on without firing faceOverlayToggled (restore)."""
        self._overlay_enabled.blockSignals(True)
        self._overlay_enabled.setChecked(bool(on))
        self._overlay_enabled.blockSignals(False)

    def toggle_face_overlay(self) -> None:
        """Flip the overlay checkbox (fires faceOverlayToggled) — for the F8
        shortcut, keeping the checkbox and overlay in lock-step."""
        self._overlay_enabled.toggle()

    def face_comparison_enabled(self) -> bool:
        return self._comparison_enabled.isChecked()

    def set_comparison_checked(self, on: bool) -> None:
        """Reflect persisted comparison state without firing the toggle."""
        self._comparison_enabled.blockSignals(True)
        self._comparison_enabled.setChecked(bool(on))
        self._comparison_enabled.blockSignals(False)

    def use_face_map(self) -> bool:
        return self._use_face_map.isChecked()

    def set_use_face_map(self, on: bool) -> None:
        """Reflect the 'use face map' state without firing useFaceMapToggled."""
        self._use_face_map.blockSignals(True)
        self._use_face_map.setChecked(bool(on))
        self._use_face_map.blockSignals(False)

    def set_face_map_available(self, available: bool) -> None:
        """Enable/disable the 'Use face map' switch. The owner (main_window) owns
        the CHECKED state — it's the single source of truth for routing."""
        self._use_face_map.setEnabled(bool(available))

    def _update_rotation_rows(self) -> None:
        """Gray out the rotation knobs (threshold / re-detect / angle source)
        when rotation compensation is off — they have no effect then."""
        on = self._rotation_enabled.isChecked()
        self._rotation_threshold.setEnabled(on)
        self._rotation_redetect.setEnabled(on)
        self._rotation_source.setEnabled(on)

    def set_face_map_routing_active(self, active: bool) -> None:
        """Gray what the per-identity map supersedes while routing is active — all
        INSIDE the Faces group so the override stays local: the whole detection
        config (detector + size + interval — detection is skipped, read from the
        precomputed geometry) and the "Which to swap" sub-box (the map decides
        what swaps with what). The "Show detection overlay" + comparison toggles
        gray out too — the overlay is now managed by the Faces panel (it shows the
        selected identity), so F8 has nothing to control. The swap's Angle source
        stays (it still applies on frames the geometry didn't cover)."""
        self._face_map_routing = bool(active)
        self._detector.setEnabled(not active)
        self._detection_size.setEnabled(not active)
        self._detection_interval.setEnabled(not active)
        self._which_to_swap_box.setEnabled(not active)
        self._overlay_enabled.setEnabled(not active)
        self._comparison_enabled.setEnabled(not active)
        self._update_detector_rows()  # target_sex also gates on the detector

    def _update_detector_rows(self) -> None:
        """Gray out the gender filter for detection-only detectors (no
        insightface .sex) OR while face-map routing supersedes it."""
        full_pack = self._detector.currentData() == DetectorModel.BUFFALO_L.value
        self._target_sex.setEnabled(full_pack and not self._face_map_routing)

    def _update_occlusion_rows(self) -> None:
        """Link the occlusion sub-controls to the master checkbox and to each
        other: everything grays out when the mask is off; the parser applies
        only to region/both, the occluder model only to occluder/both."""
        on = self._occlusion_mask.isChecked()
        mode = self._occlusion_mode.currentData()
        self._occlusion_mode.setEnabled(on)
        self._occlusion_parser.setEnabled(
            on and mode != OcclusionMaskMode.OCCLUDER.value
        )
        self._occluder_model.setEnabled(
            on and mode != OcclusionMaskMode.REGION.value
        )

    def _update_swapper_model_rows(self) -> None:
        """Gray the fast-paste knob for the 256px swappers — they ALWAYS blend
        through the fast ROI paste; the toggle only applies to the insightface
        models (inswapper / reswapper), whose original blend it replaces."""
        model = SwapperModel(self._swapper_model.currentData())
        self._fast_paste.setEnabled(is_insightface_model(model))

    def _update_upscaler_rows(self) -> None:
        """Gray the fp16 knob for models it doesn't apply to — the ONNX
        upscalers (no effect) and SwinIR (its attention can't run in half) —
        and the torch device for the ONNX upscalers (they run on ORT EPs)."""
        model = UpscalerModel(self._upscaler_model.currentData())
        is_onnx = model_runtime(model) == "onnx"
        self._upscaler_fp16.setEnabled(model_supports_fp16(model))
        # Torch device for torch models; ONNX providers row for ONNX models.
        self._upscaler_device.setEnabled(not is_onnx)
        self._upscaler_providers_row.setEnabled(is_onnx)

    def _couple_comparison_to_overlay(self, on: bool) -> None:
        """The comparison thumbnails draw on the detection overlay, so enabling
        comparison must enable the overlay too (otherwise the toggle looks
        broken — nothing appears)."""
        if on and not self._overlay_enabled.isChecked():
            self._overlay_enabled.setChecked(True)  # fires faceOverlayToggled

    def _couple_overlay_to_comparison(self, on: bool) -> None:
        """Turning the detection overlay off also turns comparison off — there's
        nothing to draw the thumbnails onto."""
        if not on and self._comparison_enabled.isChecked():
            self._comparison_enabled.setChecked(False)  # fires faceComparisonToggled

    def skip_strategy(self) -> FrameSkipStrategy:
        cls = _STRATEGIES[self._strategy_combo.currentText()]
        if cls is SyncedStrategy:
            return cls(max_lag_frames=self._synced_max_lag_frames.value())
        if cls is PredictiveStrategy:
            return cls(max_lead_seconds=self._predictive_max_lead_seconds.value())
        return cls()

    def synced_max_lag_frames(self) -> int:
        return self._synced_max_lag_frames.value()

    def predictive_max_lead_seconds(self) -> float:
        return self._predictive_max_lead_seconds.value()

    def realtime_workers(self) -> int:
        return self._worker_count.value()

    def reader_pool_size(self) -> int:
        return self._reader_pool_size.value()

    def processing_scale(self) -> float:
        """Downscale factor in (0, 1]; 1.0 = full resolution."""
        return self._scale_slider.value() / 100.0

    def set_target_native_size(self, size: object) -> None:
        """Tell the scale readout the loaded target's native dimensions so it
        can show the resulting size. `size` is (width, height) or None (no
        target loaded)."""
        if size is None:
            self._target_native_size = None
        else:
            w, h = size  # type: ignore[misc]
            self._target_native_size = (int(w), int(h))
        self._update_scale_label()

    def _update_scale_label(self) -> None:
        pct = self._scale_slider.value()
        if self._target_native_size is None:
            self._scale_label.setText(f"{pct}%")
            return
        nw, nh = self._target_native_size
        w, h = scaled_dims(nw, nh, pct / 100.0)
        self._scale_label.setText(f"{pct}% [{w}x{h}]")

    def strategy_name(self) -> str:
        return type(self.skip_strategy()).__name__

    def playback_mode(self) -> PlaybackMode:
        return _PLAYBACK_MODES[self._playback_combo.currentText()]

    def video_backend(self) -> VideoBackend:
        return _VIDEO_BACKENDS[self._video_backend_combo.currentText()]

    def swapper_providers(self) -> list[str]:
        """Realtime ONNX providers for the swapper (+ the shared detector), in
        the platform's preference order. Always non-empty (CPU floor)."""
        return self._swapper_providers_row.selected()

    def enhancer_providers(self) -> list[str]:
        """ONNX providers for the enhancer's ONNX restorer backends."""
        return self._enhancer_providers_row.selected()

    def upscaler_providers(self) -> list[str]:
        """ONNX providers for the ONNX upscalers."""
        return self._upscaler_providers_row.selected()

    def enhancer_device(self) -> str:
        """Selected torch device token for the realtime enhancer
        ("auto" / "cpu" / "cuda:N")."""
        return self._enhancer_device.currentData()

    def mark_providers_failed(self, failed: set[str]) -> None:
        """Flag providers ORT couldn't initialise on the SWAPPER row (that's what
        record_actual_providers reports). Empty set clears the marks."""
        self._swapper_providers_row.mark_failed(failed)

    def cache_mode(self) -> CacheMode:
        return _CACHE_MODES[self._cache_mode_combo.currentText()]

    def image_format(self) -> ImageFormat:
        return _IMAGE_FORMATS[self._image_format_combo.currentText()]

    def image_quality(self) -> int:
        return self._image_quality.value()

    def memory_cache_mb(self) -> int:
        return self._memory_cache_mb.value()

    def write_workers(self) -> int:
        return self._write_workers.value()

    def write_queue_size(self) -> int:
        return self._write_queue_size.value()

    def snapshot(self) -> ProcessorParamsSnapshot:
        """Capture the full processor + session parameter surface as one
        immutable value object — the single source the controller / settings /
        batch consumers read from, replacing per-field capture."""
        return ProcessorParamsSnapshot(
            swapper_enabled=self.swapper_enabled(),
            swapper_params=self.swapper_params(),
            swapper_providers=tuple(self.swapper_providers()),
            enhancer_enabled=self.enhancer_enabled(),
            enhancer_params=self.enhancer_params(),
            enhancer_device=self.enhancer_device(),
            enhancer_providers=tuple(self.enhancer_providers()),
            upscaler_enabled=self.upscaler_enabled(),
            upscaler_params=self.upscaler_params(),
            upscaler_device=self.upscaler_device(),
            upscaler_providers=tuple(self.upscaler_providers()),
            strategy_name=self.strategy_name(),
            realtime_workers=self.realtime_workers(),
            playback_mode=self.playback_mode(),
            reader_pool_size=self.reader_pool_size(),
            processing_scale=self.processing_scale(),
            synced_max_lag_frames=self.synced_max_lag_frames(),
            predictive_max_lead_seconds=self.predictive_max_lead_seconds(),
            cache_mode=self.cache_mode(),
            image_format=self.image_format(),
            image_quality=self.image_quality(),
            memory_cache_mb=self.memory_cache_mb(),
            write_workers=self.write_workers(),
            write_queue_size=self.write_queue_size(),
            video_backend=self.video_backend(),
        )

    def apply_snapshot(self, snapshot: ProcessorParamsSnapshot) -> None:
        """Write a snapshot back into the controls (the inverse of snapshot()).

        Reuses the bulk-restore path (so per-widget signals stay suppressed and
        configChanged fires once) AND the snapshot's own flat field map — the
        same one settings persistence consumes — so restore and persist read
        from a single source and can't drift apart."""
        self.apply_restored_settings(**snapshot.to_settings_kwargs())

    def apply_restored_settings(
        self,
        *,
        realtime_workers: int | None,
        strategy_name: str | None,
        enhancer_enabled: bool | None,
        swapper_enabled: bool | None = None,
        swapper_model: str | None = None,
        swapper_detection_interval: int | None,
        swapper_detection_size: int | None,
        swapper_detector: str | None,
        swapper_many_faces: bool | None,
        swapper_fast_paste: bool | None = None,
        swapper_landmark_refine: bool | None = None,
        swapper_target_sex: str | None,
        swapper_occlusion_mask: bool | None,
        swapper_occlusion_mode: str | None = None,
        swapper_occlusion_parser: str | None,
        swapper_occluder_model: str | None = None,
        swapper_rotation_compensation: bool | None,
        swapper_rotation_threshold_deg: int | None,
        swapper_rotation_redetect: bool | None,
        swapper_rotation_angle_source: str | None,
        enhancer_model: str | None = None,
        enhancer_upscale: int | None,
        enhancer_only_center_face: bool | None,
        enhancer_codeformer_fidelity: float | None = None,
        enhancer_fp16: bool | None = None,
        playback_mode: PlaybackMode | None,
        cache_mode: CacheMode | None,
        image_format: ImageFormat | None,
        image_quality: int | None,
        memory_cache_mb: int | None,
        write_workers: int | None,
        write_queue_size: int | None,
        video_backend: VideoBackend | None,
        reader_pool_size: int | None,
        processing_scale: float | None,
        synced_max_lag_frames: int | None,
        predictive_max_lead_seconds: float | None = None,
        swapper_providers: list[str] | None,
        enhancer_device: str | None,
        upscaler_enabled: bool | None = None,
        upscaler_model: str | None = None,
        upscaler_tile: int | None = None,
        upscaler_fp16: bool | None = None,
        upscaler_device: str | None = None,
        enhancer_providers: list[str] | None = None,
        upscaler_providers: list[str] | None = None,
    ) -> None:
        """Apply persisted values without firing configChanged per field.

        Block per-widget signals during the bulk apply, then emit configChanged
        once so the controller sees the final state. None values are skipped —
        widget defaults stay in place when no persisted value exists.
        """
        widgets = (
            self._swapper_box,
            self._swapper_model,
            self._detection_interval,
            self._detection_size,
            self._detector,
            self._many_faces,
            self._fast_paste,
            self._landmark_refine,
            self._target_sex,
            self._occlusion_mask,
            self._occlusion_mode,
            self._occlusion_parser,
            self._occluder_model,
            self._rotation_enabled,
            self._rotation_threshold,
            self._rotation_redetect,
            self._rotation_source,
            self._enhancer_box,
            self._upscaler_box,
            self._upscaler_model,
            self._upscaler_tile,
            self._upscaler_fp16,
            self._upscaler_device,
            self._enhancer_model,
            self._upscale,
            self._enhancer_fidelity,
            self._only_center_face,
            self._enhancer_fp16,
            self._enhancer_device,
            self._strategy_combo,
            self._worker_count,
            self._playback_combo,
            self._cache_mode_combo,
            self._image_format_combo,
            self._image_quality,
            self._memory_cache_mb,
            self._write_workers,
            self._write_queue_size,
            self._video_backend_combo,
            self._reader_pool_size,
            self._scale_slider,
            self._synced_max_lag_frames,
            self._predictive_max_lead_seconds,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            if swapper_detection_interval is not None:
                self._detection_interval.setValue(swapper_detection_interval)
            if swapper_detection_size is not None:
                self._detection_size.setValue(swapper_detection_size)
            if swapper_detector is not None:
                for i in range(self._detector.count()):
                    if self._detector.itemData(i) == swapper_detector:
                        self._detector.setCurrentIndex(i)
                        break
            if swapper_many_faces is not None:
                self._many_faces.setChecked(swapper_many_faces)
            if swapper_fast_paste is not None:
                self._fast_paste.setChecked(swapper_fast_paste)
            if swapper_landmark_refine is not None:
                self._landmark_refine.setChecked(swapper_landmark_refine)
            if swapper_target_sex is not None:
                # Look up by stored token (UserRole data) since the
                # combo's display text differs from the persisted value.
                for i in range(self._target_sex.count()):
                    if self._target_sex.itemData(i) == swapper_target_sex:
                        self._target_sex.setCurrentIndex(i)
                        break
            if swapper_occlusion_mask is not None:
                self._occlusion_mask.setChecked(swapper_occlusion_mask)
            if swapper_occlusion_parser is not None:
                for i in range(self._occlusion_parser.count()):
                    if self._occlusion_parser.itemData(i) == swapper_occlusion_parser:
                        self._occlusion_parser.setCurrentIndex(i)
                        break
            if swapper_occlusion_mode is not None:
                for i in range(self._occlusion_mode.count()):
                    if self._occlusion_mode.itemData(i) == swapper_occlusion_mode:
                        self._occlusion_mode.setCurrentIndex(i)
                        break
            if swapper_occluder_model is not None:
                for i in range(self._occluder_model.count()):
                    if self._occluder_model.itemData(i) == swapper_occluder_model:
                        self._occluder_model.setCurrentIndex(i)
                        break
            if swapper_rotation_compensation is not None:
                self._rotation_enabled.setChecked(swapper_rotation_compensation)
            if swapper_rotation_threshold_deg is not None:
                self._rotation_threshold.setValue(swapper_rotation_threshold_deg)
            if swapper_rotation_redetect is not None:
                self._rotation_redetect.setChecked(swapper_rotation_redetect)
            if swapper_rotation_angle_source is not None:
                for i in range(self._rotation_source.count()):
                    if self._rotation_source.itemData(i) == swapper_rotation_angle_source:
                        self._rotation_source.setCurrentIndex(i)
                        break
            if swapper_enabled is not None:
                self._swapper_box.setChecked(swapper_enabled)
            if swapper_model is not None:
                for i in range(self._swapper_model.count()):
                    if self._swapper_model.itemData(i) == swapper_model:
                        self._swapper_model.setCurrentIndex(i)
                        break
            if enhancer_enabled is not None:
                self._enhancer_box.setChecked(enhancer_enabled)
            if enhancer_model is not None:
                for i in range(self._enhancer_model.count()):
                    if self._enhancer_model.itemData(i) == enhancer_model:
                        self._enhancer_model.setCurrentIndex(i)
                        break
            if enhancer_upscale is not None:
                self._upscale.setValue(enhancer_upscale)
            if enhancer_codeformer_fidelity is not None:
                self._enhancer_fidelity.setValue(enhancer_codeformer_fidelity)
            if enhancer_only_center_face is not None:
                self._only_center_face.setChecked(enhancer_only_center_face)
            if enhancer_fp16 is not None:
                self._enhancer_fp16.setChecked(enhancer_fp16)
            if enhancer_device is not None:
                # Match by stored token; a persisted cuda:N that no longer
                # exists on this machine simply isn't found → keep default.
                for i in range(self._enhancer_device.count()):
                    if self._enhancer_device.itemData(i) == enhancer_device:
                        self._enhancer_device.setCurrentIndex(i)
                        break
            if upscaler_enabled is not None:
                self._upscaler_box.setChecked(upscaler_enabled)
            if upscaler_model is not None:
                for i in range(self._upscaler_model.count()):
                    if self._upscaler_model.itemData(i) == upscaler_model:
                        self._upscaler_model.setCurrentIndex(i)
                        break
            if upscaler_tile is not None:
                self._upscaler_tile.setValue(upscaler_tile)
            if upscaler_fp16 is not None:
                self._upscaler_fp16.setChecked(upscaler_fp16)
            if upscaler_device is not None:
                for i in range(self._upscaler_device.count()):
                    if self._upscaler_device.itemData(i) == upscaler_device:
                        self._upscaler_device.setCurrentIndex(i)
                        break
            if strategy_name is not None:
                label = _label_for_strategy_name(strategy_name)
                if label is not None:
                    self._strategy_combo.setCurrentText(label)
            if realtime_workers is not None:
                self._worker_count.setValue(realtime_workers)
            if playback_mode is not None:
                label = _label_for_playback_mode(playback_mode)
                if label is not None:
                    self._playback_combo.setCurrentText(label)
            if cache_mode is not None:
                label = _label_for_cache_mode(cache_mode)
                if label is not None:
                    self._cache_mode_combo.setCurrentText(label)
            if image_format is not None:
                label = _label_for_image_format(image_format)
                if label is not None:
                    self._image_format_combo.setCurrentText(label)
            if image_quality is not None:
                self._image_quality.setValue(image_quality)
            if memory_cache_mb is not None:
                self._memory_cache_mb.setValue(memory_cache_mb)
            if write_workers is not None:
                self._write_workers.setValue(write_workers)
            if write_queue_size is not None:
                self._write_queue_size.setValue(write_queue_size)
            if video_backend is not None:
                label = _label_for_video_backend(video_backend)
                if label is not None:
                    self._video_backend_combo.setCurrentText(label)
            if reader_pool_size is not None:
                self._reader_pool_size.setValue(reader_pool_size)
            if processing_scale is not None:
                self._scale_slider.setValue(
                    max(10, min(100, round(processing_scale * 100)))
                )
            if synced_max_lag_frames is not None:
                self._synced_max_lag_frames.setValue(synced_max_lag_frames)
            if predictive_max_lead_seconds is not None:
                self._predictive_max_lead_seconds.setValue(predictive_max_lead_seconds)
            # Restore each per-model provider row's selection (set_selected
            # blocks signals + forces CPU on for an empty/all-unknown list).
            if swapper_providers is not None:
                self._swapper_providers_row.set_selected(swapper_providers)
            if enhancer_providers is not None:
                self._enhancer_providers_row.set_selected(enhancer_providers)
            if upscaler_providers is not None:
                self._upscaler_providers_row.set_selected(upscaler_providers)
        finally:
            for w in widgets:
                w.blockSignals(False)
        # The format/strategy may have changed so refresh enabled states.
        self._update_quality_visibility()
        self._update_strategy_param_enabled()
        self._update_scale_label()  # reflect a restored scale (set under blockSignals)
        self._update_enhancer_model_rows()  # reflect a restored enhancer model
        self._update_rotation_rows()  # reflect a restored rotation-compensation state
        self._update_detector_rows()  # reflect a restored detector choice
        self._update_upscaler_rows()  # reflect a restored upscaler model
        self._update_occlusion_rows()  # reflect restored occlusion mask/mode
        self._update_swapper_model_rows()  # reflect a restored swap model
        self.configChanged.emit()
