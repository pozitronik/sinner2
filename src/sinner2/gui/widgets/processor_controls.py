from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.model_cache import (
    available_onnx_providers,
    get_active_providers,
)
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processors.face_enhancer import FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams, TargetSex
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    SyncedStrategy,
)

_STRATEGIES: dict[str, type[FrameSkipStrategy]] = {
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
    browseRootRequested = Signal()
    resetRootRequested = Signal()
    invalidateRequested = Signal()
    clearAllRequested = Signal()
    sizeCapChanged = Signal(object)  # int bytes (object to avoid C int overflow at 2 GB+); 0 = uncapped
    rootChanged = Signal(object)  # Path | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        swapper_defaults = FaceSwapperParams()
        enhancer_defaults = FaceEnhancerParams()

        swapper_box = QGroupBox("FaceSwapper")
        swapper_form = QFormLayout(swapper_box)
        self._detection_interval = QSpinBox()
        self._detection_interval.setRange(1, 30)
        self._detection_interval.setValue(swapper_defaults.detection_interval)
        self._detection_interval.valueChanged.connect(self.configChanged)
        swapper_form.addRow("Detection interval", self._detection_interval)
        self._many_faces = QCheckBox()
        self._many_faces.setChecked(swapper_defaults.many_faces)
        self._many_faces.toggled.connect(self.configChanged)
        swapper_form.addRow("Many faces", self._many_faces)
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
        swapper_form.addRow("Swap which", self._target_sex)

        enhancer_box = QGroupBox("FaceEnhancer (GFPGAN)")
        enhancer_box.setCheckable(True)
        enhancer_box.setChecked(True)
        enhancer_box.toggled.connect(self.configChanged)
        enhancer_form = QFormLayout(enhancer_box)
        self._upscale = QSpinBox()
        self._upscale.setRange(1, 4)
        self._upscale.setValue(enhancer_defaults.upscale)
        self._upscale.valueChanged.connect(self.configChanged)
        enhancer_form.addRow("Upscale", self._upscale)
        self._only_center_face = QCheckBox()
        self._only_center_face.setChecked(enhancer_defaults.only_center_face)
        self._only_center_face.toggled.connect(self.configChanged)
        enhancer_form.addRow("Center face only", self._only_center_face)
        self._enhancer_box = enhancer_box

        execution_box = QGroupBox("Execution")
        execution_form = QFormLayout(execution_box)
        self._strategy_combo = QComboBox()
        for label in _STRATEGIES:
            self._strategy_combo.addItem(label)
        self._strategy_combo.currentTextChanged.connect(
            lambda _: (self._update_synced_threshold_enabled(), self.configChanged.emit())
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
        execution_form.addRow("Worker count", self._worker_count)

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

        # ONNX execution providers — multi-select. Order matters: ORT
        # tries providers in the order listed, falling back through to
        # CPU. We expose every available provider as a checkbox and
        # preserve the platform-default order from get_available_providers
        # (which is already "best first"). Unchecking all reverts to the
        # default order at session-build time.
        providers_box = QWidget()
        providers_layout = QVBoxLayout(providers_box)
        providers_layout.setContentsMargins(0, 0, 0, 0)
        providers_layout.setSpacing(2)
        self._provider_checkboxes: dict[str, QCheckBox] = {}
        try:
            available = available_onnx_providers()
        except Exception:
            # ORT might fail to load on a broken install — fall back to
            # the known-good defaults so the panel still renders.
            available = list(get_active_providers())
        # Pre-check whatever the model_cache currently treats as active —
        # on a fresh launch this is the platform defaults (CUDA + CPU),
        # so the user sees the actual effective state instead of an
        # all-unchecked column that lies about what ORT will use.
        # apply_restored_settings will override this when there's a
        # persisted list.
        default_active = set(get_active_providers())
        for prov in available:
            cb = QCheckBox(prov)
            cb.setToolTip(
                "ONNX execution provider. Multiple may be checked; ORT\n"
                "tries them in the order shown. If you uncheck everything,\n"
                "the system falls back to the platform defaults (CUDA + CPU)\n"
                "so inference still works. Applies immediately — rebuilds\n"
                "the session (chain reloads)."
            )
            cb.setChecked(prov in default_active)
            cb.toggled.connect(self.configChanged)
            providers_layout.addWidget(cb)
            self._provider_checkboxes[prov] = cb
        execution_form.addRow("ONNX providers", providers_box)

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
        self._update_synced_threshold_enabled()

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
        inner.setMinimumWidth(320)
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(swapper_box)
        inner_layout.addWidget(enhancer_box)
        inner_layout.addWidget(execution_box)
        inner_layout.addWidget(cache_box)
        inner_layout.addWidget(cache_storage_box)
        inner_layout.addStretch()

        scroll = QScrollArea(self)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _update_quality_visibility(self) -> None:
        # Quality control is JPEG-only; disable when PNG is selected so
        # the user doesn't think it's doing something.
        is_jpeg = self.image_format() is ImageFormat.JPEG
        self._image_quality.setEnabled(is_jpeg)

    def _update_synced_threshold_enabled(self) -> None:
        # Threshold only meaningful when Synced is the active strategy.
        cls = _STRATEGIES[self._strategy_combo.currentText()]
        self._synced_max_lag_frames.setEnabled(cls is SyncedStrategy)

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
        self._invalidate_btn.setEnabled(enabled)

    def cache_size_cap_bytes(self) -> int:
        if not self._size_cap_enabled.isChecked():
            return 0
        return self._size_cap_mb.value() * 1024 * 1024

    def swapper_params(self) -> FaceSwapperParams:
        # Combo's currentData is the str token, not the enum — Qt's
        # QVariant flattens str-Enum to str (see library sort fix). The
        # FaceSwapperParams model coerces back via str-Enum membership.
        return FaceSwapperParams(
            detection_interval=self._detection_interval.value(),
            many_faces=self._many_faces.isChecked(),
            target_sex=self._target_sex.currentData(),
        )

    def enhancer_params(self) -> FaceEnhancerParams:
        return FaceEnhancerParams(
            upscale=self._upscale.value(),
            only_center_face=self._only_center_face.isChecked(),
        )

    def enhancer_enabled(self) -> bool:
        return self._enhancer_box.isChecked()

    def skip_strategy(self) -> FrameSkipStrategy:
        cls = _STRATEGIES[self._strategy_combo.currentText()]
        if cls is SyncedStrategy:
            return cls(max_lag_frames=self._synced_max_lag_frames.value())
        return cls()

    def synced_max_lag_frames(self) -> int:
        return self._synced_max_lag_frames.value()

    def worker_count(self) -> int:
        return self._worker_count.value()

    def reader_pool_size(self) -> int:
        return self._reader_pool_size.value()

    def strategy_name(self) -> str:
        return type(self.skip_strategy()).__name__

    def playback_mode(self) -> PlaybackMode:
        return _PLAYBACK_MODES[self._playback_combo.currentText()]

    def video_backend(self) -> VideoBackend:
        return _VIDEO_BACKENDS[self._video_backend_combo.currentText()]

    def onnx_providers(self) -> list[str]:
        """Currently selected providers in the order they appear in
        the checkbox column (the platform's default preference order).
        Empty list = use the default (caller decides what that means)."""
        return [
            name
            for name, cb in self._provider_checkboxes.items()
            if cb.isChecked()
        ]

    def mark_providers_failed(self, failed: set[str]) -> None:
        """Visually flag providers that were requested but ORT couldn't
        initialise. Red strikethrough on the checkbox label + a tooltip
        explaining what happened. Call with an empty set to clear all
        marks (e.g. when a new selection succeeds clean)."""
        for name, cb in self._provider_checkboxes.items():
            if name in failed:
                cb.setStyleSheet(
                    "QCheckBox { color: #d94545; text-decoration: line-through; }"
                )
                cb.setToolTip(
                    f"{name} failed to initialise — ORT fell back to a\n"
                    "lower-priority provider. Usually means the runtime\n"
                    "libraries this EP depends on (e.g. nvinfer for\n"
                    "TensorRT) aren't installed or aren't on PATH.\n"
                    "Status bar shows what ORT is actually using."
                )
            else:
                cb.setStyleSheet("")
                cb.setToolTip(
                    "ONNX execution provider. Multiple may be checked; ORT\n"
                    "tries them in the order shown. If you uncheck everything,\n"
                    "the system falls back to the platform defaults (CUDA + CPU)\n"
                    "so inference still works. Applies immediately — rebuilds\n"
                    "the session (chain reloads)."
                )

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

    def apply_restored_settings(
        self,
        *,
        worker_count: int | None,
        strategy_name: str | None,
        enhancer_enabled: bool | None,
        swapper_detection_interval: int | None,
        swapper_many_faces: bool | None,
        swapper_target_sex: str | None,
        enhancer_upscale: int | None,
        enhancer_only_center_face: bool | None,
        playback_mode: PlaybackMode | None,
        cache_mode: CacheMode | None,
        image_format: ImageFormat | None,
        image_quality: int | None,
        memory_cache_mb: int | None,
        write_workers: int | None,
        write_queue_size: int | None,
        video_backend: VideoBackend | None,
        reader_pool_size: int | None,
        synced_max_lag_frames: int | None,
        onnx_providers: list[str] | None,
    ) -> None:
        """Apply persisted values without firing configChanged per field.

        Block per-widget signals during the bulk apply, then emit configChanged
        once so the controller sees the final state. None values are skipped —
        widget defaults stay in place when no persisted value exists.
        """
        widgets = (
            self._detection_interval,
            self._many_faces,
            self._target_sex,
            self._enhancer_box,
            self._upscale,
            self._only_center_face,
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
            self._synced_max_lag_frames,
            *self._provider_checkboxes.values(),
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            if swapper_detection_interval is not None:
                self._detection_interval.setValue(swapper_detection_interval)
            if swapper_many_faces is not None:
                self._many_faces.setChecked(swapper_many_faces)
            if swapper_target_sex is not None:
                # Look up by stored token (UserRole data) since the
                # combo's display text differs from the persisted value.
                for i in range(self._target_sex.count()):
                    if self._target_sex.itemData(i) == swapper_target_sex:
                        self._target_sex.setCurrentIndex(i)
                        break
            if enhancer_enabled is not None:
                self._enhancer_box.setChecked(enhancer_enabled)
            if enhancer_upscale is not None:
                self._upscale.setValue(enhancer_upscale)
            if enhancer_only_center_face is not None:
                self._only_center_face.setChecked(enhancer_only_center_face)
            if strategy_name is not None:
                label = _label_for_strategy_name(strategy_name)
                if label is not None:
                    self._strategy_combo.setCurrentText(label)
            if worker_count is not None:
                self._worker_count.setValue(worker_count)
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
            if synced_max_lag_frames is not None:
                self._synced_max_lag_frames.setValue(synced_max_lag_frames)
            if onnx_providers is not None:
                # Restore exact selection — only providers in the
                # persisted list become checked; unknown providers in
                # the list are ignored (a different machine may not
                # have the same ORT build).
                wanted = set(onnx_providers)
                for name, cb in self._provider_checkboxes.items():
                    cb.setChecked(name in wanted)
        finally:
            for w in widgets:
                w.blockSignals(False)
        # The format/strategy may have changed so refresh enabled states.
        self._update_quality_visibility()
        self._update_synced_threshold_enabled()
        self.configChanged.emit()
