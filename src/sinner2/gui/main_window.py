import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QElapsedTimer, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sinner2.batch.queue import BatchQueue
from sinner2.batch.task import (
    DEFAULT_ENHANCER_WORKERS,
    DEFAULT_SWAPPER_WORKERS,
    DEFAULT_UPSCALER_WORKERS,
    BatchCleanupMode,
    BatchOutputFormat,
    BatchProgress,
)
from sinner2.batch.task_store import BatchTaskStore
from sinner2.config import media_extensions
from sinner2.config import settings as user_settings
from sinner2.config.execution import OnnxExecution, TorchExecution
from sinner2.gui.face_detection_probe import FaceDetectionProbe, FaceDetectionSink
from sinner2.gui.icon import app_icon
from sinner2.gui.model_download import ensure_models
from sinner2.pipeline.detectors import DETECTOR_MODEL_FILES, DetectorModel
from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    enhancer_onnx_model_file,
)
from sinner2.pipeline.processors.face_swapper import (
    RotationAngleSource,
    SwapperModel,
)
from sinner2.pipeline.processors.landmarker import (
    MODEL_FILE as LANDMARKER_MODEL_FILE,
)
from sinner2.pipeline.processors.occlusion import (
    OcclusionMaskMode,
    occluder_model_files,
    parser_model_file,
)
from sinner2.pipeline.processors.swapper_models import (
    model_files as swapper_model_files,
)
from sinner2.pipeline.processors.upscaler import model_filename
from sinner2.gui.cache_controller import default_cache_root
from sinner2.gui.confirm import (
    SuppressionStore,
    confirm,
    set_default_suppression_store,
)
from sinner2.gui.live_controller import LiveController
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.session_capabilities import (
    CameraConfig,
    FileTarget,
    SessionCapabilities,
    SessionKind,
)
from sinner2.gui.session_facade import SessionFacade
from sinner2.gui.widgets.batch_task_dialog import QBatchTaskDialog
from sinner2.gui.widgets.batch_view import QBatchView
from sinner2.gui.widgets.models_view import QModelsView
from sinner2.gui.widgets.face_detection_overlay import QFaceDetectionOverlay
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.fullscreen_control_bar import FullscreenControlBar
from sinner2.gui.widgets.live_view import QLiveView
from sinner2.gui.widgets.metrics_overlay import (
    CumulativeRateTracker,
    MetricsSample,
    QMetricsOverlay,
)
from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.gui.widgets.side_panel import QSidePanel
from sinner2.gui.widgets.source_target_panel import QSourceTargetPanel
from sinner2.gui.widgets.status_action_bar import QStatusActionBar
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.types import Frame

_log = logging.getLogger(__name__)

# Detection-probe thread join budget at close. The first detection lazily
# builds the buffalo_l pack (5 ONNX models + prepare(), can exceed 2s and may
# download), so a single wait(2000) can time out and destroy a running thread.
_THREAD_JOIN_WAIT_MS = 2000
_THREAD_JOIN_MAX_WAITS = 15  # ~30s worst case before giving up and logging


def _join_qthread(thread, per_wait_ms: int, max_waits: int) -> bool:  # type: ignore[no-untyped-def]
    """Quit a QThread and block until it actually finishes, in bounded
    increments. Returns True if it stopped, False if it's still running after
    `max_waits` attempts — in which case the caller should log rather than
    destroy a running thread (which crashes on exit)."""
    thread.quit()
    waits = 0
    while thread.isRunning() and waits < max_waits:
        thread.wait(per_wait_ms)
        waits += 1
    return not thread.isRunning()


def _path_is_file(path: Path) -> bool:
    """`Path.is_file()` that reports an UNAVAILABLE location as "not a file"
    rather than raising.

    On Windows a path on a detached drive (unmapped letter, ejected media, a
    disconnected network share) makes `is_file()`/`stat()` raise OSError
    ("the device is not ready" — WinError 21/53) instead of returning False.
    Run during startup restore that exception would abort the whole launch, so
    a persisted-but-now-unreachable path must simply be skipped. We swallow the
    error and treat the path as absent (it self-heals out of settings on the
    next save)."""
    try:
        return path.is_file()
    except OSError:
        return False


def _fmt_size(b: int) -> str:
    """Human-readable bytes: 1024 → 1.0 KB → 1.0 MB → 1.0 GB → 1.0 TB.

    Used by the cache stats / cache management UI. Stays compact (one
    decimal) so the status line / panel doesn't reflow on each refresh.
    """
    n = float(b)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{int(b)} B"


class SinnerMainWindow(QMainWindow):
    """The player surface: frame display on top, transport, then source/target.

    All real work lives on PlayerController; this class is layout, keyboard
    shortcuts, and error dialogs. Closing the window tears down the player.
    """

    # Cross-thread request to the detection probe (runs on its own QThread).
    _requestDetection = Signal(object, int, int)  # frame, width, height

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("sinner2")
        self.setWindowIcon(app_icon())
        self._settings = user_settings.load()
        # Install the process-wide store backing every "Don't ask me again"
        # checkbox. Reads/writes the persisted suppression map through the same
        # authoritative _update_settings path, so child widgets that show a
        # confirm() dialog need no Settings reference of their own.
        set_default_suppression_store(
            SuppressionStore(
                load=lambda: dict(self._settings.confirm_suppressed or {}),
                save=lambda m: self._update_settings(confirm_suppressed=m),
            )
        )
        # Apply configurable accepted file extensions app-wide (library accept
        # filters, file-dialog filters, Target.kind) BEFORE building the side
        # panel, which reads them to build its libraries + dialog filters.
        media_extensions.configure(
            self._settings.library_image_extensions,
            self._settings.library_video_extensions,
        )
        # True while a batch task renders — locks the live-editing surface so
        # the display acts purely as a render preview (DaVinci-style).
        self._batch_active = False
        # Failures collected during the CURRENT batch run (label, message),
        # reported in one consolidated dialog when the queue goes idle so a
        # continue-on-error run doesn't spam a modal per failed task.
        self._batch_failures: list[tuple[str, str]] = []
        # Guards _wait_for_tensorrt_build against re-entry: during a FIRST
        # engine build none of its other guards trip (TRT not recorded yet, no
        # engine on disk), so a swap completing mid-build would stack a second
        # dialog + polling timer over the active one.
        self._trt_wait_active = False
        # Non-modal poll that refreshes the failed-provider highlight AFTER an
        # async chain rebuild records the real providers (see
        # _schedule_provider_highlight_refresh). One at a time — a newer toggle
        # replaces it.
        self._provider_highlight_timer: QTimer | None = None
        # While a batch renders we repurpose the position bar to track the
        # render's last frame; this caches the slider range we set so we only
        # reset it when the stage's frame count actually changes (set_frame_count
        # snaps the value to 0). -1 = "not set for the current batch".
        self._batch_slider_total = -1
        if not self._restore_geometry_from_settings():
            self.resize(960, 720)

        self._display = QFrameDisplayWidget()
        self._transport = QTransportControls()
        self._pickers = QSourceTargetPanel()
        self._processors = QProcessorControls()
        self._live_view = QLiveView()
        # Side panel hosts the processors plus the source/target libraries
        # in a tab widget. Thumbnails get cached under the project temp dir
        # so they survive restarts; same directory as the processed-frame
        # cache root so users only have one folder to clean up.
        # Batch infrastructure — store + queue + view. Built BEFORE
        # the side panel so we can hand the view to QSidePanel.
        batch_store_root = Path(
            self._settings.batch_store_path or default_cache_root().parent / "batch"
        )
        self._batch_store = BatchTaskStore(batch_store_root)
        self._batch_queue = BatchQueue(
            store=self._batch_store,
            cache_root=default_cache_root() / "batch_cache",
            global_output_dir=self._global_output_dir(),
        )
        self._batch_view = QBatchView(
            store=self._batch_store,
            queue=self._batch_queue,
            global_output_dir_resolver=self._global_output_dir,
        )
        self._models_view = QModelsView()
        # Per-panel zoom: fall back to the legacy shared value, then 128.
        _legacy_dim = self._settings.library_display_dim or 128
        self._side_panel = QSidePanel(
            thumbnail_cache_dir=default_cache_root() / "thumbnails",
            processors=self._processors,
            batch_view=self._batch_view,
            models_view=self._models_view,
            live_view=self._live_view,
            sources_display_dim=self._settings.library_sources_display_dim or _legacy_dim,
            targets_display_dim=self._settings.library_targets_display_dim or _legacy_dim,
        )
        # Metrics overlay: child of the frame display so it floats on top
        # of the rendered frame and inherits its z-order. Position is
        # managed manually via resizeEvent (no layout takes child widgets
        # for QFrameDisplayWidget).
        self._write_rate = CumulativeRateTracker()
        self._drop_rate = CumulativeRateTracker()
        self._metrics_overlay = QMetricsOverlay(
            snapshot_fn=self._sample_metrics,
            parent=self._display,
        )
        # Face-detection debug overlay: a transparent full-cover child of the
        # display, fed by a detection probe running on its own thread (so the
        # live preview never stalls). Off by default; toggled with F8.
        self._face_overlay_on = False
        self._comparison_on = False
        self._last_probe_feed = 0.0
        # Last frame handed to the display, kept so enabling the overlay can
        # detect the current frame immediately (e.g. while paused) instead of
        # waiting for the next rendered frame.
        self._last_displayed_frame: Frame | None = None
        self._face_overlay = QFaceDetectionOverlay(parent=self._display)
        self._display.set_face_overlay(self._face_overlay)
        # Auto-hiding playback bar for fullscreen. A child of the display so
        # it floats over the frame; it takes custody of the transport row
        # while fullscreen is active and reveals when the cursor nears the
        # bottom edge. Idle (timer stopped) outside fullscreen.
        self._fs_controls = FullscreenControlBar(self._display)
        # When the swapper is running, the overlay shows ITS pre-swap
        # detections (published to this sink) rather than re-detecting the
        # swapped output. A timer polls the sink while the overlay is on.
        self._detection_sink = FaceDetectionSink()
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setInterval(int(self._PROBE_INTERVAL_S * 1000))
        self._overlay_timer.timeout.connect(self._overlay_tick)
        self._detection_probe = FaceDetectionProbe(
            providers=self._settings.swapper_providers,
            detection_size=self._settings.swapper_detection_size or 640,
        )
        self._detection_thread = QThread(self)
        self._detection_probe.moveToThread(self._detection_thread)
        self._detection_thread.start()
        self._requestDetection.connect(
            self._detection_probe.analyze, Qt.ConnectionType.QueuedConnection
        )
        self._detection_probe.detectionsReady.connect(
            self._on_detections, Qt.ConnectionType.QueuedConnection
        )
        self._display.frameDisplayed.connect(self._feed_detection_probe)

        central = QWidget()
        # Resizable divider between the frame display and the side panel
        # so the user can trade viewer real-estate for more library tile
        # columns. State is persisted across launches the same way as
        # the window geometry. A minimum width on the side panel keeps
        # the controls usable when dragged narrow.
        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._top_splitter.addWidget(self._display)
        self._top_splitter.addWidget(self._side_panel)
        self._top_splitter.setStretchFactor(0, 1)  # display absorbs window growth
        self._top_splitter.setStretchFactor(1, 0)
        self._top_splitter.setChildrenCollapsible(False)
        self._side_panel.setMinimumWidth(280)
        layout = QVBoxLayout(central)
        # Tight gaps between display / transport / pickers / status bar.
        layout.setSpacing(2)
        layout.addWidget(self._top_splitter, stretch=1)
        layout.addWidget(self._transport)
        layout.addWidget(self._pickers)
        self._central_layout = layout

        # Custom bottom status bar: view/window action buttons (left), status
        # message (middle), persistent indicators (right). Replaces
        # QMainWindow's QStatusBar, which hides left widgets behind temporary
        # messages. Each button mirrors a keyboard shortcut and routes through
        # the matching handler, so button state / action / persisted setting
        # never drift.
        self._status_bar = QStatusActionBar()
        layout.addWidget(self._status_bar)
        self.setCentralWidget(central)

        self._status_bar.on_top_button.toggled.connect(self._set_stays_on_top)
        self._status_bar.stats_button.toggled.connect(self._set_stats_visible)
        self._status_bar.rotate_button.clicked.connect(self._cycle_rotation)
        self._status_bar.fullscreen_button.toggled.connect(self._set_fullscreen)
        self._status_bar.side_panel_button.toggled.connect(
            self._set_side_panel_visible
        )
        self._status_bar.save_button.clicked.connect(self._save_current_frame)

        self._status_bar.show_message("ready")
        self._scratch_label = QLabel("cache: —")
        self._scratch_label.setToolTip(
            "Persistent processed-frame cache directory for this session "
            "(survives between runs; keyed by source+target+chain config)"
        )
        self._status_bar.add_permanent_widget(self._scratch_label)
        self._fps_label = QLabel("--- fps")
        self._fps_label.setToolTip(
            "Real cross-worker throughput — frames completed per wall-clock "
            "second across all workers (3-second rolling window)."
        )
        self._status_bar.add_permanent_widget(self._fps_label)
        self._metrics_label = QLabel("")
        self._metrics_label.setToolTip(
            "cache: hit-ratio / memory used. "
            "writes: outstanding/cap, total dropped (cap-hit skips), p50/p95 ms latency."
        )
        self._status_bar.add_permanent_widget(self._metrics_label)
        self._strategy_mode_label = QLabel("")
        self._strategy_mode_label.setToolTip(
            "Current frame-skip strategy mode. 'synced (lagging)' means\n"
            "SyncedStrategy has fallen back to sequential submission\n"
            "because processing can't keep up — display will trail the\n"
            "timeline but throughput stays at the pipeline's max rate."
        )
        self._status_bar.add_permanent_widget(self._strategy_mode_label)
        self._providers_label = QLabel("")
        self._providers_label.setToolTip(
            "ONNX execution providers currently in use, in ORT's try-order.\n"
            "Differs from the checkbox column when the user has unchecked\n"
            "everything (system falls back to defaults so inference still works)."
        )
        self._status_bar.add_permanent_widget(self._providers_label)

        self._controller = PlayerController(self._display, self._transport, parent=self)
        # Wire the swapper's pre-swap detections to the overlay sink (set before
        # any session so every built chain picks it up).
        self._controller.set_detection_sink(self._detection_sink)
        self._controller.errorOccurred.connect(self._show_error)
        self._controller.processingFpsChanged.connect(self._update_fps_label)
        self._controller.sessionScratchDirChanged.connect(self._update_scratch_label)
        # Also drives the TensorRT compile dialog on session START (the launch
        # case: TRT persisted + no cached engine → build runs at first load, with
        # no config-change event to hook).
        self._controller.sessionScratchDirChanged.connect(self._on_session_scratch_dir)
        self._controller.targetNativeSizeChanged.connect(
            self._processors.set_target_native_size
        )
        self._controller.bufferMetricsChanged.connect(self._update_metrics_label)
        self._controller.strategyModeChanged.connect(self._update_strategy_mode_label)
        self._controller.sessionSwitching.connect(self._on_session_switching)

        # Live-camera engine: webcam -> chain -> MJPEG sink. Its preview frames
        # drive the same display; activation + transport are owned by the facade.
        self._live = LiveController(parent=self)
        # Same detection sink the file path uses, so the live swap publishes to
        # the GUI overlay + comparison-crop probe.
        self._live.set_detection_sink(self._detection_sink)
        self._live.frameReady.connect(lambda f: self._display.show_frame(f))
        self._live.runningChanged.connect(self._on_live_running)
        self._live.errorOccurred.connect(self._show_error)
        self._live.processingFpsChanged.connect(self._update_live_fps_label)
        self._live_view.startRequested.connect(self._on_use_camera)
        self._live_view.stopRequested.connect(self._live.stop)
        self._live_view.workersChanged.connect(self._live.set_worker_count)
        self._live_view.configChanged.connect(self._persist_camera_config)

        # Single-session facade: ONE active session (file or camera) the
        # transport + keyboard + target picker bind to. The camera is just
        # another target; the facade pulls the current settings snapshot when it
        # starts/updates the camera chain.
        self._session = SessionFacade(
            self._controller, self._live,
            snapshot_provider=self._processors.snapshot, parent=self,
        )
        self._session.capabilitiesChanged.connect(self._on_capabilities_changed)
        self._pickers.cameraRequested.connect(self._on_use_camera)
        self._transport.playRequested.connect(self._session.play)
        self._transport.pauseRequested.connect(self._session.pause)
        self._transport.seekRequested.connect(self._session.seek_to)

        self._pickers.sourceChanged.connect(self._on_source_changed)
        self._pickers.targetChanged.connect(self._on_target_changed)
        self._pickers.sourceChanged.connect(self._persist_source_path)
        self._pickers.targetChanged.connect(self._persist_target_path)
        self._pickers.sourceRecentsChanged.connect(self._persist_source_recents)
        self._pickers.targetRecentsChanged.connect(self._persist_target_recents)
        # "Add to batch" now lives in the transport row; enable it only when
        # both a source and target are loaded.
        self._transport.addToBatchRequested.connect(self._on_add_to_batch)
        self._pickers.sourceChanged.connect(self._refresh_transport_enabled)
        self._pickers.targetChanged.connect(self._refresh_transport_enabled)
        # Batch queue → realtime preview coordination. Mutually
        # exclusive: when the queue starts a task, pause the live
        # preview; when the queue empties, drop a status-bar note so
        # the user knows it's safe to play again.
        self._batch_queue.taskStarted.connect(self._on_batch_task_started)
        self._batch_queue.taskProgress.connect(self._on_batch_progress)
        self._batch_queue.queueIdle.connect(self._on_batch_queue_idle)
        self._batch_queue.taskFailed.connect(self._on_batch_task_failed)
        self._batch_queue.taskPreview.connect(self._on_batch_preview)
        self._batch_view.editRequested.connect(self._on_edit_batch_task)
        self._processors.configChanged.connect(self._on_processor_config_changed)
        self._processors.configChanged.connect(self._persist_processor_settings)
        self._processors.faceOverlayToggled.connect(self._set_face_overlay_visible)
        self._processors.faceComparisonToggled.connect(self._set_comparison_visible)
        # Cache-management actions (own signals so they don't go through
        # configChanged, which is for runtime tuning of the chain).
        self._processors.browseRootRequested.connect(self._on_browse_cache_root)
        self._processors.resetRootRequested.connect(self._on_reset_cache_root)
        self._processors.invalidateRequested.connect(self._on_invalidate_session)
        self._processors.rerenderRequested.connect(self._on_rerender_from_current)
        self._processors.clearAllRequested.connect(self._on_clear_all_caches)
        self._processors.sizeCapChanged.connect(self._on_size_cap_changed)
        # Audio: transport emits, main_window persists. The controller's
        # own slots already update the backend; we only need to write
        # settings back here.
        self._transport.volumeChanged.connect(self._persist_audio_volume)
        self._controller.cacheStorageStatsChanged.connect(self._refresh_cache_stats)
        # Library wiring: click a thumbnail to load it as source/target,
        # use the same controller hooks as the file picker so frame state
        # and play state are preserved on switch.
        self._side_panel.sources_library().pathSelected.connect(
            self._on_library_source_selected
        )
        self._side_panel.targets_library().pathSelected.connect(
            self._on_library_target_selected
        )
        self._side_panel.sources_library().rootsChanged.connect(
            self._persist_library_sources
        )
        self._side_panel.targets_library().rootsChanged.connect(
            self._persist_library_targets
        )
        # Zoom + sort are kept PER PANEL (source and target independent), so
        # each persists only its own value.
        self._side_panel.sources_library().displayDimChanged.connect(
            lambda d: self._update_settings(library_sources_display_dim=int(d))
        )
        self._side_panel.targets_library().displayDimChanged.connect(
            lambda d: self._update_settings(library_targets_display_dim=int(d))
        )
        self._side_panel.sources_library().sortChanged.connect(
            self._persist_sources_sort
        )
        self._side_panel.targets_library().sortChanged.connect(
            self._persist_targets_sort
        )
        # Apply persisted processor settings before the first session starts —
        # apply_restored_settings emits configChanged once at the end, which
        # both seeds the controller and persists the (now equal) values back.
        self._restore_processor_settings()
        self._restore_cache_management_state()
        self._restore_audio_state()
        self._refresh_cache_stats()
        self._refresh_providers_label()
        self._restore_side_panel_state()
        self._restore_top_splitter_from_settings()
        self._restore_metrics_overlay_state()
        self._restore_face_overlay_state()
        self._restore_comparison_state()
        self._restore_stays_on_top()
        self._restore_rotation()
        # Restoring a complete source+target pair would otherwise trigger a
        # synchronous, model-loading session build inside the constructor,
        # delaying the window's first paint. Defer that build to the first
        # showEvent so the GUI appears immediately ("GUI first, models after").
        self._restoring_paths = False
        self._pending_initial_target: Path | None = None
        self._initial_session_started = False
        self._restore_paths_from_settings()
        # Transport starts disabled and only enables once a source AND target
        # are present (restore above may have supplied them).
        self._refresh_transport_enabled()
        # After paths are restored a session may have auto-started.
        # Flag any persisted-but-broken providers (e.g. user previously
        # checked TensorRT on a machine without the libs).
        self._highlight_failed_providers()
        # Fullscreen state is per-launch (not persisted) — we always
        # start windowed regardless of how the last session ended.
        self._is_fullscreen = False
        # Saved widget visibility for restoration when leaving fullscreen.
        self._pre_fullscreen_visibility: dict[QWidget, bool] = {}
        # Whether the window was maximized before going fullscreen, so exit
        # restores THAT rather than dropping to a smaller "normal" geometry.
        self._pre_fullscreen_maximized = False

    def _on_source_changed(self, source_path: Path) -> None:
        """Source picker fired. Route the new face to the active session — the
        facade hot-applies it to a running file/camera session, or seeds the next
        build."""
        self._session.set_source(source_path)

    def _on_target_changed(self, target_path: Path) -> None:
        """Target picker fired. The facade builds on first load (once a source is
        present), swaps the target on a running session, and tears down the camera
        if one was active.

        During startup restore the heavy first build is deferred: record the
        target and let showEvent kick the build off AFTER the window paints, so
        the GUI is up immediately instead of waiting on model loading."""
        if self._restoring_paths:
            self._pending_initial_target = target_path
            return
        self._session.set_target(FileTarget(target_path))

    def _update_fps_label(self, fps: float) -> None:
        # File-session throughput; ignored while the camera is the active target
        # so a late paused-executor emission can't overwrite the live reading.
        if self._session.active_kind() is not SessionKind.CAMERA:
            self._fps_label.setText(f"{fps:.1f} fps")

    def _update_live_fps_label(self, fps: float) -> None:
        if self._session.active_kind() is SessionKind.CAMERA:
            self._fps_label.setText(f"{fps:.1f} fps")

    def _update_scratch_label(self, scratch_dir: object) -> None:
        self._scratch_label.setText(f"cache: {scratch_dir}" if scratch_dir else "cache: —")

    # ---- Cache management slots ----

    def _restore_cache_management_state(self) -> None:
        # Cache root: settings → controller → widget display
        if self._settings.cache_root_path:
            self._controller.set_cache_root(Path(self._settings.cache_root_path))
        self._processors.set_cache_root_text(self._controller.cache_root())
        # Size cap: settings → controller (state) + widget (display)
        cap_mb = self._settings.cache_size_cap_mb or 0
        cap_bytes = cap_mb * 1024 * 1024 if cap_mb > 0 else 0
        self._controller.set_cache_size_cap_bytes(cap_bytes)
        self._processors.set_cache_size_cap_bytes(cap_bytes)

    def _refresh_cache_stats(self) -> None:
        manager = self._controller.cache_manager()
        entries = manager.list_entries()
        total = sum(e.size_bytes for e in entries)
        free = manager.free_disk_bytes()
        if free == 0:
            free_text = "free: ?"
        else:
            free_text = f"free: {_fmt_size(free)}"
        text = f"{len(entries)} entries · {_fmt_size(total)} · {free_text}"
        self._processors.set_cache_stats_text(text)
        self._processors.set_invalidate_enabled(
            self._controller.executor() is not None
        )

    def _on_browse_cache_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose cache root directory",
            str(self._controller.cache_root()),
        )
        if not chosen:
            return
        self._controller.set_cache_root(Path(chosen))
        self._processors.set_cache_root_text(self._controller.cache_root())
        self._update_settings(cache_root_path=str(self._controller.cache_root()))

    def _on_reset_cache_root(self) -> None:
        self._controller.set_cache_root(None)
        self._processors.set_cache_root_text(self._controller.cache_root())
        self._update_settings(cache_root_path=None)

    def _on_invalidate_session(self) -> None:
        if self._controller.executor() is None:
            return
        if not confirm(
            self,
            "invalidate_session",
            "Invalidate current session",
            "Drop all cached frames for this session and reprocess from scratch?",
        ):
            return
        self._controller.invalidate_current_session()

    def _on_rerender_from_current(self) -> None:
        # No confirmation: it only reprocesses from the playhead forward and is
        # the natural "apply my param change retroactively" gesture.
        self._controller.rerender_from_current()

    def _on_clear_all_caches(self) -> None:
        manager = self._controller.cache_manager()
        entries = manager.list_entries()
        protected = self._controller.session_cache_dir()
        deletable = [e for e in entries if e.path != protected]
        if not deletable:
            QMessageBox.information(
                self,
                "Clear all caches",
                "Nothing to delete — only the current session's cache is present.",
            )
            return
        total = sum(e.size_bytes for e in deletable)
        if not confirm(
            self,
            "clear_all_caches",
            "Clear all caches",
            f"Delete {len(deletable)} cache entries ({_fmt_size(total)})?\n"
            "The currently active session will be spared.",
        ):
            return
        self._controller.clear_all_caches()

    def _on_size_cap_changed(self, bytes_cap: int) -> None:
        self._controller.set_cache_size_cap_bytes(bytes_cap)
        cap_mb = bytes_cap // (1024 * 1024) if bytes_cap > 0 else 0
        self._update_settings(cache_size_cap_mb=cap_mb or None)

    # ---- Audio ----

    def _restore_audio_state(self) -> None:
        volume = self._settings.audio_volume if self._settings.audio_volume is not None else 100
        # Push silently into the widget so we don't echo back as a user
        # "change" (which would re-save settings unnecessarily). Mute was
        # dropped — volume 0 is silence.
        self._transport.set_volume_silently(volume)
        # Push into the controller so the backend (constructed lazily on
        # first audio_backend() call) starts with the right value.
        self._controller.apply_initial_audio_state(volume=volume)

    def _persist_audio_volume(self, value: int) -> None:
        self._update_settings(audio_volume=int(value))

    def _refresh_transport_enabled(self, *_: object) -> None:
        # Per-control gating from the active session's capabilities (a file is
        # seekable/finite/maybe-audio; a camera disables seek/volume but keeps
        # play=stop/start). A batch render holds the editing surface → all off.
        caps = (
            SessionCapabilities.none()
            if self._batch_active
            else self._session.capabilities()
        )
        self._transport.apply_capabilities(caps)

    def _on_capabilities_changed(self, caps: object) -> None:
        # The active session (re)built or switched targets — reflect it on the
        # transport + hide the file-only Settings groups when a camera is active.
        if not self._batch_active:
            self._transport.apply_capabilities(caps)  # type: ignore[arg-type]
        is_camera = self._session.active_kind() is SessionKind.CAMERA
        self._processors.set_file_only_visible(not is_camera)

    def _update_strategy_mode_label(self, mode: object) -> None:
        text = str(mode) if mode else ""
        self._strategy_mode_label.setText(f"strategy: {text}" if text else "")

    def _refresh_providers_label(self) -> None:
        # Trim the trailing "ExecutionProvider" suffix so the label
        # stays short — "CUDAExecutionProvider, CPUExecutionProvider"
        # is too noisy in the status bar.
        providers = self._controller.effective_onnx_providers()
        short = [p.removesuffix("ExecutionProvider") or p for p in providers]
        self._providers_label.setText(f"EP: {', '.join(short)}" if short else "")

    def _highlight_failed_providers(self) -> None:
        """Mark requested-but-not-loaded providers red on the widget
        and pop a transient status-bar message. The mismatch is the
        signal that a provider's runtime libs are missing (TensorRT
        EP loaded but no nvinfer, etc.). Pulled here instead of
        emitted by the controller so the styling is purely a view
        concern."""
        requested = set(self._processors.swapper_providers())
        actual = set(self._controller.effective_onnx_providers())
        # Empty `requested` = user unchecked everything → we use
        # defaults; nothing to flag as failed in that case.
        if not requested:
            self._processors.mark_providers_failed(set())
            return
        failed = requested - actual
        self._processors.mark_providers_failed(failed)
        if failed:
            short_failed = ", ".join(
                p.removesuffix("ExecutionProvider") for p in failed
            )
            short_actual = ", ".join(
                p.removesuffix("ExecutionProvider")
                for p in self._controller.effective_onnx_providers()
            )
            self._status_bar.show_message(
                f"ONNX provider(s) failed to load: {short_failed}. "
                f"ORT is using: {short_actual}",
                7000,
            )

    def _schedule_provider_highlight_refresh(self) -> None:
        """Refresh the failed-provider highlight AFTER the async chain rebuild
        records what ORT actually wired up.

        ``set_chain`` is asynchronous — the rebuild + provider recording run on
        the executor's dispatcher thread — so highlighting synchronously here
        would compare the new request against the PREVIOUS session's providers
        and flash a spurious red until the next toggle (the reported bug: a
        re-checked provider goes red, then clears one toggle later). Poll
        ``get_actual_providers()`` (non-modal) until it changes from the
        pre-rebuild snapshot, a newer toggle supersedes this request, the
        session goes away, or a short timeout backstops a same-providers
        rebuild — then highlight against the truth. A genuine fallback still
        shows red, just after the rebuild instead of before it.
        """
        from sinner2.pipeline import model_cache

        requested = tuple(self._processors.swapper_providers())
        if not requested or self._controller.executor() is None:
            # Defaults in use, or no live session to rebuild — nothing to wait
            # for; highlight against the current truth immediately.
            self._highlight_failed_providers()
            return
        before = model_cache.get_actual_providers()
        # Replace any in-flight refresh so rapid toggles don't stack timers.
        if self._provider_highlight_timer is not None:
            self._provider_highlight_timer.stop()
        elapsed = QElapsedTimer()
        elapsed.start()
        timer = QTimer(self)
        timer.setInterval(150)
        self._provider_highlight_timer = timer

        def _poll() -> None:
            actual = model_cache.get_actual_providers()
            rebuilt = actual != before  # the rebuilt session recorded its EPs
            superseded = tuple(self._processors.swapper_providers()) != requested
            gone = self._controller.executor() is None
            if rebuilt or superseded or gone or elapsed.elapsed() > 8000:
                timer.stop()
                if self._provider_highlight_timer is timer:
                    self._provider_highlight_timer = None
                # A newer toggle owns the highlight now — don't fight it.
                if not superseded:
                    self._highlight_failed_providers()

        timer.timeout.connect(_poll)
        timer.start()

    def _update_metrics_label(self, metrics: object) -> None:
        # `metrics` is BufferMetrics. Compact one-liner: cache hit% / memory MB,
        # write queue depth, total drops, write latency p50/p95.
        ratio = getattr(metrics, "cache_hit_ratio", 0.0)
        mem_mb = getattr(metrics, "memory_used_bytes", 0) / 1024 / 1024
        wq_out = getattr(metrics, "write_outstanding", 0)
        wq_max = getattr(metrics, "write_max_outstanding", 0)
        wq_drop = getattr(metrics, "write_dropped", 0)
        wl_p50 = getattr(metrics, "write_latency_p50_ms", 0.0)
        wl_p95 = getattr(metrics, "write_latency_p95_ms", 0.0)
        self._metrics_label.setText(
            f"cache {ratio * 100:.0f}% / {mem_mb:.0f}M  "
            f"writes {wq_out}/{wq_max} drops {wq_drop} "
            f"p50/p95 {wl_p50:.0f}/{wl_p95:.0f}ms"
        )

    def _on_processor_config_changed(self) -> None:
        if self._batch_active:
            return  # editing is locked while a batch renders

        # If the upscaler / occlusion mask was just enabled and its weights
        # aren't present, ask to download them (never silently). Decline →
        # revert the toggle so the chain isn't rebuilt with a missing model.
        if self._processors.upscaler_enabled() and not self._ensure_upscaler_model():
            self._processors.set_upscaler_checked(False)
        # A non-default swap model needs its weights (and, for ghost/simswap,
        # an embedding converter) present. Confirm the download; decline reverts
        # to inswapper_128 (a required model, always present).
        if self._processors.swapper_enabled():
            model = SwapperModel(self._processors.swapper_model())
            if model is not SwapperModel.INSWAPPER_128 and not ensure_models(
                self, swapper_model_files(model)
            ):
                self._processors.set_swapper_model(SwapperModel.INSWAPPER_128.value)
        swapper_cfg = self._processors.swapper_params()
        # 2dfan4 weight is needed when refinement OR the landmark-68 angle
        # source is selected; decline reverts both to their non-landmark state.
        if swapper_cfg.landmark_refine or (
            swapper_cfg.rotation_angle_source is RotationAngleSource.LANDMARK_68
        ):
            if not ensure_models(self, [LANDMARKER_MODEL_FILE]):
                self._processors.disable_landmark_refine()
        if swapper_cfg.occlusion_mask:
            # Collect every weight the selected mask mode needs: the parser
            # for region/both, the occluder model(s) for occluder/both.
            needed: list[str] = []
            if swapper_cfg.occlusion_mode is not OcclusionMaskMode.OCCLUDER:
                needed.append(parser_model_file(swapper_cfg.occlusion_parser))
            if swapper_cfg.occlusion_mode is not OcclusionMaskMode.REGION:
                needed.extend(occluder_model_files(swapper_cfg.occluder_model))
            if not ensure_models(self, needed):
                self._processors.set_occlusion_checked(False)
        # A detection-only detector (yoloface / scrfd) needs its weight; confirm
        # the download. Decline reverts to buffalo_l (the always-present pack).
        if swapper_cfg.detector is not DetectorModel.BUFFALO_L and not ensure_models(
            self, [DETECTOR_MODEL_FILES[swapper_cfg.detector]]
        ):
            self._processors.set_swapper_detector(DetectorModel.BUFFALO_L.value)
        # ONNX enhancers (CodeFormer / GPEN-512 / RestoreFormer++) each need
        # their own weight — confirm the download when one is selected (and the
        # enhancer's on); decline reverts to GFPGAN so the chain isn't rebuilt
        # against a missing model.
        if self._processors.enhancer_enabled():
            onnx_file = enhancer_onnx_model_file(
                EnhancerModel(self._processors.enhancer_model())
            )
            if onnx_file is not None and not ensure_models(self, [onnx_file]):
                self._processors.set_enhancer_model(EnhancerModel.GFPGAN.value)

        # Every download/revert guard above has settled the widget state; capture
        # it once as a snapshot and route all consumers through that single value
        # object instead of re-reading each field by hand (which used to drift).
        snap = self._processors.snapshot()
        # Route the whole settings bundle to the active session (the facade
        # hot-applies to the camera chain when one is active, else applies the
        # chain hot-swap + video-backend / reader-pool / processing-scale
        # rebuilds to the file engine).
        self._session.apply_settings(snap)
        # Keep the overlay's detection probe on the SAME providers/size: a
        # providers change resets the shared face analysis, and a probe stuck
        # on its construction-time list could rebuild it on the stale EPs.
        self._detection_probe.configure(
            list(snap.swapper_providers),
            snap.swapper_params.detection_size,
        )
        # Swapper-provider / enhancer-device rebuilds are folded into
        # apply_settings above; just refresh the status-bar EP label
        # and the failed-provider highlight afterwards.
        self._refresh_providers_label()
        # A TensorRT-enable triggers a one-time engine compile on the executor
        # (blocks the dispatcher ~25s). Show a modal "compiling" dialog until it
        # finishes rather than leaving a frozen preview + a (prematurely) red
        # provider checkbox; the highlight is refreshed when the wait ends.
        # Otherwise defer the highlight until the async rebuild records the real
        # providers (set_chain is async), so re-checking a provider doesn't flash
        # a spurious red against the previous session's provider list.
        if not self._wait_for_tensorrt_build():
            self._schedule_provider_highlight_refresh()

    _TRT_PROVIDER = "TensorrtExecutionProvider"

    def _on_session_scratch_dir(self, scratch_dir: object) -> None:
        """A session was (re)installed (non-None dir). If it's about to compile a
        TensorRT engine, surface the modal wait — this covers the launch case
        (TRT persisted, no cached engine, build happens at the first session
        start with no config-change event to hook)."""
        if scratch_dir is not None:
            self._wait_for_tensorrt_build()

    def _wait_for_tensorrt_build(self) -> bool:
        """If TensorRT is requested but no session has actually loaded it yet, a
        (possibly slow, one-time) engine build is about to run. Show a modal busy
        dialog until TensorRT shows up in the ACTUAL recorded providers (build
        done) or we give up, then refresh the provider highlight. Returns True if
        it took over the wait (caller must NOT also highlight); False otherwise.

        Uses model_cache.get_actual_providers() — the truly-loaded list — NOT the
        controller's effective_onnx_providers(), which falls back to the REQUESTED
        list before any session loads (so at launch it would wrongly report TRT as
        already active and skip the dialog). No-op when TRT isn't requested,
        there's no session, or a session has already recorded TRT (cached)."""
        from sinner2.pipeline import model_cache

        trt = self._TRT_PROVIDER
        if trt not in self._processors.swapper_providers():
            return False
        if self._controller.executor() is None:
            return False
        if self._trt_wait_active:
            # A wait is already showing (re-entered via sessionScratchDirChanged
            # while the build runs) — don't stack a second dialog + timer.
            return True
        before = model_cache.get_actual_providers()
        if before is not None and trt in before:
            return False  # a session already built + loaded TRT this run
        if model_cache.tensorrt_engine_cached():
            return False  # engine already compiled on disk → fast load, no modal
        self._trt_wait_active = True
        dialog = QProgressDialog(
            "Compiling the TensorRT engine for the swap model.\n"
            "One-time step (about 30 seconds); cached for next time.",
            "", 0, 0, self,
        )
        dialog.setWindowTitle("TensorRT")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setCancelButton(None)  # the compile can't be interrupted
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.show()
        elapsed = QElapsedTimer()
        elapsed.start()
        timer = QTimer(self)
        timer.setInterval(400)

        def _poll() -> None:
            actual = model_cache.get_actual_providers()
            built = actual is not None and trt in actual
            # A DIFFERENT session recorded providers without TRT → it fell back
            # (engine failed to load) → stop waiting and let the red highlight
            # show the truth. (Can't detect a same-providers fallback; the 75s
            # timeout backstops that rare case.)
            fell_back = actual is not None and actual != before and trt not in actual
            gone = (
                self._controller.executor() is None
                or trt not in self._processors.swapper_providers()
            )
            if built or fell_back or gone or elapsed.elapsed() > 75_000:
                self._trt_wait_active = False
                timer.stop()
                dialog.close()
                self._highlight_failed_providers()

        timer.timeout.connect(_poll)
        timer.start()
        return True

    def _ensure_upscaler_model(self) -> bool:
        """Confirm + download the selected upscaler's weights if missing.
        Returns True if present (or downloaded), False if declined/failed."""
        name = model_filename(self._processors.upscaler_params().model)
        return ensure_models(self, [name])

    def _on_session_switching(self, switching: bool) -> None:
        """Disable transport + the processor controls + show a notice while an
        async source/target swap drains the old session and builds the new one,
        so the user can't drive a half-torn-down session OR change chain config
        mid-swap (which reconfigure_from would silently overwrite with the
        swap-time snapshot). Re-enabled when the new session is ready."""
        if switching:
            self._transport.apply_capabilities(SessionCapabilities.none())
        else:
            self._refresh_transport_enabled()
        self._processors.setEnabled(not switching)
        if switching:
            self._status_bar.show_message("Switching session…")
        else:
            self._status_bar.show_message("ready", 2000)

    def _show_error(self, message: str) -> None:
        self._status_bar.show_message(message, 5000)
        QMessageBox.critical(self, "sinner2", message)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        # View toggles don't need an executor — handle them before the
        # early-return so the user can adjust window state pre-load too.
        # Each shortcut drives the matching status-bar action button, so the
        # button's pressed state and the action stay in lock-step.
        if key == Qt.Key.Key_F9:
            self._status_bar.side_panel_button.toggle()
            return
        if key == Qt.Key.Key_F4:
            self._status_bar.stats_button.toggle()
            return
        if key == Qt.Key.Key_F11:
            self._status_bar.fullscreen_button.toggle()
            return
        if (
            key in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            # Alt+Enter mirrors F11 — toggle fullscreen via the same action
            # button so button state / shortcut / persisted view stay in sync.
            self._status_bar.fullscreen_button.toggle()
            return
        if key == Qt.Key.Key_F12:
            self._status_bar.on_top_button.toggle()
            return
        if key == Qt.Key.Key_F8:
            self._processors.toggle_face_overlay()
            return
        if key == Qt.Key.Key_R:
            self._status_bar.rotate_button.click()
            return
        if (
            key == Qt.Key.Key_S
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._status_bar.save_button.click()
            return
        if (
            key in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            # Send current source + target + settings to the batch queue
            # (same as the transport's + button; the handler no-ops if a
            # source/target isn't loaded).
            self._on_add_to_batch()
            return
        if key == Qt.Key.Key_Escape and self._is_fullscreen:
            # Escape only consumed when fullscreen is active so it doesn't
            # eat the dialog-cancel keypress in normal use.
            self._status_bar.fullscreen_button.toggle()
            return
        if not self._session.is_active():
            super().keyPressEvent(event)
            return
        # Route through the facade so the keys reach the ACTIVE session (file or
        # camera); the player keeps the audio-aware play/seek logic. Space is
        # play/pause for any session; the seek keys apply only to seekable
        # (finite) targets.
        if key == Qt.Key.Key_Space:
            self._session.toggle_playback()
            return
        executor = self._controller.executor()
        if not self._session.capabilities().seekable or executor is None:
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_Left:
            self._session.seek_to(max(0, executor.current_frame.get() - 1))
            return
        if key == Qt.Key.Key_Right:
            self._session.seek_to(executor.current_frame.get() + 1)
            return
        if key == Qt.Key.Key_Home:
            self._session.seek_to(0)
            return
        if key == Qt.Key.Key_End:
            self._session.seek_to(max(0, executor.frame_count() - 1))
            return
        super().keyPressEvent(event)

    # ---- View toggles (driven by the status-bar action buttons) ----

    @staticmethod
    def _set_button_checked(button: object, on: bool) -> None:
        """Reflect state on a toggle button WITHOUT re-emitting toggled
        (which would re-run its handler and re-persist)."""
        button.blockSignals(True)  # type: ignore[attr-defined]
        button.setChecked(bool(on))  # type: ignore[attr-defined]
        button.blockSignals(False)  # type: ignore[attr-defined]

    def _set_side_panel_visible(self, on: bool) -> None:
        self._side_panel.setVisible(on)
        # Persist immediately so the next launch matches the last user choice.
        self._update_settings(side_panel_visible=on)

    def _restore_side_panel_state(self) -> None:
        # Default True (panel shown) if the setting is missing.
        visible = (
            True
            if self._settings.side_panel_visible is None
            else bool(self._settings.side_panel_visible)
        )
        self._side_panel.setVisible(visible)
        self._set_button_checked(self._status_bar.side_panel_button, visible)

    def _set_stays_on_top(self, on: bool) -> None:
        # Flip the WindowStaysOnTopHint and re-show — Qt requires the widget
        # to be re-shown after window-flag changes so the window manager picks
        # them up. show() preserves geometry and content.
        already_on = bool(
            self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        )
        if already_on != on:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
            self.show()
        self._update_settings(window_stays_on_top=on)

    def _restore_stays_on_top(self) -> None:
        on = bool(self._settings.window_stays_on_top)
        if on:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            # show() called by the caller chain; no need here.
        self._set_button_checked(self._status_bar.on_top_button, on)

    def _cycle_rotation(self) -> None:
        new_rot = self._display.cycle_rotation()
        self._update_settings(display_rotation=new_rot)
        self._status_bar.show_message(f"Rotation: {new_rot}°", 2000)

    def _restore_rotation(self) -> None:
        if self._settings.display_rotation:
            self._display.set_rotation(self._settings.display_rotation)

    def _save_current_frame(self) -> None:
        """Write the currently-displayed pixmap (rotated as on screen)
        to a user-picked file. Defaults to PNG; JPEG also accepted via
        the file extension. No-op when nothing's on screen."""
        pix = self._display.current_pixmap()
        if pix is None:
            self._status_bar.show_message("No frame to save", 2000)
            return
        # Default filename embeds the timeline frame index so the user
        # can save many frames in succession without overwriting.
        frame_idx = 0
        executor = self._controller.executor()
        if executor is not None:
            frame_idx = executor.current_frame.get()
        default_name = f"sinner2-frame-{frame_idx:08d}.png"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save current frame",
            default_name,
            "PNG image (*.png);;JPEG image (*.jpg);;All files (*)",
        )
        if not path_str:
            return
        # QPixmap.save handles the format from the extension and
        # supports Unicode paths through Qt's I/O (no need for our
        # cv2_unicode wrapper here).
        ok = pix.save(path_str)
        if ok:
            self._status_bar.show_message(f"Saved {path_str}", 3000)
        else:
            QMessageBox.warning(
                self, "Save failed", f"Could not write image to:\n{path_str}"
            )

    def _set_fullscreen(self, on: bool) -> None:
        # Driven by the fullscreen action button (and F11 / Esc, which toggle
        # it). Guard against redundant calls so the button-toggled signal
        # can't double-enter/exit.
        if on == self._is_fullscreen:
            return
        if on:
            self._enter_fullscreen()
        else:
            self._exit_fullscreen()

    def _enter_fullscreen(self) -> None:
        # Snapshot visibility of every chrome widget — the custom status bar
        # included (it's a normal widget in the central layout now) — so
        # exit_fullscreen can restore exactly what was showing. The transport
        # is NOT hidden: it's moved into the auto-hiding fullscreen bar below.
        chrome: list[QWidget] = [
            self._side_panel,
            self._pickers,
            self._status_bar,
        ]
        self._pre_fullscreen_visibility = {w: w.isVisible() for w in chrome}
        # Capture maximized state BEFORE showFullScreen() clears it, so exit
        # can return to maximized rather than a smaller restored geometry.
        self._pre_fullscreen_maximized = self.isMaximized()
        for w in chrome:
            w.setVisible(False)
        # Hand the transport to the fullscreen bar so the playback controls
        # stay reachable (revealed on cursor-near-bottom) without permanently
        # covering the frame. removeWidget first so the central layout drops
        # its slot cleanly before the bar reparents it.
        self._central_layout.removeWidget(self._transport)
        self._fs_controls.attach(self._transport)
        self._is_fullscreen = True
        self.showFullScreen()
        self._fs_controls.begin()

    def _exit_fullscreen(self) -> None:
        # Stop the cursor watch, take the transport back out of the bar, and
        # re-home it into its normal slot (index 1: below the display
        # splitter, above the pickers row).
        self._fs_controls.end()
        self._fs_controls.detach(self._transport)
        # Re-home the transport just below the display splitter. Resolved by the
        # splitter's current index (the mode toggle above it shifts the slots),
        # so it lands correctly whether or not the toggle is present.
        slot = self._central_layout.indexOf(self._top_splitter) + 1
        self._central_layout.insertWidget(slot, self._transport)
        self._transport.show()
        for w, was_visible in self._pre_fullscreen_visibility.items():
            w.setVisible(was_visible)
        self._pre_fullscreen_visibility = {}
        self._is_fullscreen = False
        # Restore the pre-fullscreen window state. showNormal() alone would
        # drop a window that was maximized down to its restored size.
        if self._pre_fullscreen_maximized:
            self.showMaximized()
        else:
            self.showNormal()

    # ---- Metrics overlay ----

    def _reset_metrics_rates(self) -> None:
        # Reset the cumulative write/drop rate trackers so the first reading
        # after the overlay is (re-)shown is a fresh baseline, not a delta
        # smeared over the whole interval the overlay was hidden (its timer is
        # stopped while hidden, freezing the trackers' prev count/timestamp).
        self._write_rate.reset()
        self._drop_rate.reset()

    def _set_stats_visible(self, on: bool) -> None:
        if on:
            self._reset_metrics_rates()
        self._metrics_overlay.setVisible(on)
        if on:
            self._reposition_metrics_overlay()
        self._update_settings(metrics_overlay_visible=on)

    def _restore_metrics_overlay_state(self) -> None:
        visible = bool(self._settings.metrics_overlay_visible)
        if visible:
            self._reset_metrics_rates()
            self._reposition_metrics_overlay()
        self._metrics_overlay.setVisible(visible)
        self._set_button_checked(self._status_bar.stats_button, visible)

    def _reposition_metrics_overlay(self) -> None:
        # Anchor top-left of the frame display with an 8 px margin.
        # Called on toggle-on and whenever the display resizes.
        hint = self._metrics_overlay.sizeHint()
        margin = 8
        self._metrics_overlay.setGeometry(
            margin,
            margin,
            hint.width(),
            hint.height(),
        )

    def _sample_metrics(self) -> "MetricsSample | None":
        # Called by the overlay's QTimer (~10 Hz). Returns None when no
        # session is active so the overlay shows the placeholder.
        executor = self._controller.executor()
        if executor is None:
            self._write_rate.reset()
            self._drop_rate.reset()
            return None
        import time as _time

        now = _time.monotonic()
        buf_metrics = executor.metrics.get()
        write_fps = self._write_rate.update(buf_metrics.write_completed, now)
        drop_fps = self._drop_rate.update(buf_metrics.write_dropped, now)
        return MetricsSample(
            timestamp=now,
            read_fps=executor.reads_per_second(),
            process_fps=executor.processing_fps.get(),
            write_fps=write_fps,
            drop_fps=drop_fps,
            cache_hit_ratio=buf_metrics.cache_hit_ratio,
            memory_used_mb=buf_metrics.memory_used_bytes / 1024 / 1024,
            work_outstanding=0,  # not surfaced by executor today; placeholder
            work_capacity=0,
            write_outstanding=buf_metrics.write_outstanding,
            write_capacity=buf_metrics.write_max_outstanding,
            total_drops=buf_metrics.write_dropped,
            last_completed=executor.last_completed_frame(),
            # Per-processor average ms over the last few seconds —
            # the overlay surfaces this in a "P:" row so the user can
            # see which processor in the chain owns the wall-clock.
            processor_timings=executor.processor_timings(),
        )

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        # Keep the overlay anchored to the frame display's top-right
        # corner as the window resizes.
        super().resizeEvent(event)
        if self._metrics_overlay.isVisible():
            self._reposition_metrics_overlay()
        if self._is_fullscreen:
            # Keep the playback bar pinned to the bottom edge as the
            # fullscreen surface settles to its final size.
            self._fs_controls.reposition()

    # ---- Face-detection overlay ----

    _PROBE_INTERVAL_S = 0.15  # ~6 Hz: enough to track, cheap enough to stay smooth

    def _apply_face_overlay_visible(self, on: bool) -> None:
        self._face_overlay_on = on
        self._refresh_overlay_modes()
        if on:
            self._face_overlay.setGeometry(self._display.rect())
            self._face_overlay.show()
            self._overlay_timer.start()
            # Show something immediately rather than waiting for a poll tick or
            # the next rendered frame (key when paused). Swapper on → its
            # published detections; swapper off → probe the current frame.
            if self._processors.swapper_enabled():
                self._overlay_tick()
            elif self._last_displayed_frame is not None:
                self._submit_to_probe(self._last_displayed_frame)
        else:
            self._overlay_timer.stop()
            self._face_overlay.hide()
            self._face_overlay.clear()
            # Keep the sink's latest detection: the swapper keeps publishing
            # whether or not the overlay is shown, so re-enabling (even while
            # paused) can display the current frame's boxes at once instead of
            # waiting for the next rendered frame.

    def _refresh_overlay_modes(self) -> None:
        """Comparison crops are only wanted (and drawn) when BOTH the face
        overlay and the comparison toggle are on — so the swapper extracts
        them only then (zero cost otherwise)."""
        comparison = self._face_overlay_on and self._comparison_on
        self._detection_sink.set_wants_crops(comparison)
        self._face_overlay.set_comparison(comparison)

    def _overlay_tick(self) -> None:
        # Swapper-on path: poll the swapper's published PRE-swap detections
        # (and, in comparison mode, its orig/swapped crops). The swapper-off
        # path runs through the probe, fed by displayed frames.
        if not self._face_overlay_on or not self._processors.swapper_enabled():
            return
        latest = self._detection_sink.latest_detections()
        if latest is not None:
            detections, w, h = latest
            self._face_overlay.set_detections(detections, w, h)
        if self._comparison_on:
            crops = self._detection_sink.latest_crops()
            if crops is not None:
                pairs, w, h = crops
                self._face_overlay.set_crop_pairs(pairs, w, h)

    def _set_face_overlay_visible(self, on: bool) -> None:
        """Toggle handler for the face button (and F8). Applies + persists."""
        self._apply_face_overlay_visible(on)
        if on:
            # With the swapper ON the overlay shows the swapper's own pre-swap
            # detections; with it OFF, a re-detection of the (raw) frame.
            if self._processors.swapper_enabled():
                msg = "Face-detection overlay on — showing the face swapper's detections"
            else:
                msg = "Face-detection overlay on (F8)"
            self._status_bar.show_message(msg, 4000)
        self._update_settings(face_overlay_visible=on)

    def _restore_face_overlay_state(self) -> None:
        visible = bool(self._settings.face_overlay_visible)
        self._apply_face_overlay_visible(visible)
        self._processors.set_overlay_checked(visible)

    def _set_comparison_visible(self, on: bool) -> None:
        """Toggle handler for the comparison checkbox. Persists + applies."""
        self._comparison_on = on
        self._refresh_overlay_modes()
        if on:
            # Force one reprocess so the current (possibly paused) frame's crops
            # publish now, instead of only after the next rendered frame.
            executor = self._controller.executor()
            if executor is not None:
                current = executor.current_frame.get()
                if current >= 0:
                    executor.seek(current)
            if not (self._face_overlay_on and self._processors.swapper_enabled()):
                self._status_bar.show_message(
                    "Comparison needs the face overlay (F8) and the swapper on",
                    4000,
                )
        self._update_settings(face_comparison_visible=on)

    def _restore_comparison_state(self) -> None:
        on = bool(self._settings.face_comparison_visible)
        self._comparison_on = on
        self._processors.set_comparison_checked(on)
        self._refresh_overlay_modes()

    def _feed_detection_probe(self, frame: Frame) -> None:
        # Tap each displayed frame. Always remember the latest (so enabling the
        # overlay can detect it at once). Only run the probe when the overlay
        # is on AND the swapper is off — when the swapper runs, the overlay
        # uses its published pre-swap detections instead (the _overlay_tick
        # poll), so re-detecting the swapped output would be both wrong and
        # wasteful. Zero detection cost when the overlay is off.
        self._last_displayed_frame = frame
        if not self._face_overlay_on or self._processors.swapper_enabled():
            return
        import time as _time

        if _time.monotonic() - self._last_probe_feed < self._PROBE_INTERVAL_S:
            return
        self._submit_to_probe(frame)

    def _submit_to_probe(self, frame: Frame) -> None:
        import time as _time

        self._last_probe_feed = _time.monotonic()
        h, w = frame.shape[:2]
        # Copy so the producer can't mutate the buffer under the probe thread.
        self._requestDetection.emit(frame.copy(), w, h)

    def _on_detections(self, detections: object, width: int, height: int) -> None:
        if self._face_overlay_on:
            self._face_overlay.set_detections(detections, width, height)  # type: ignore[arg-type]

    def _on_use_camera(self) -> None:
        """Make the camera the active target (from the picker button or the Live
        tab). The facade tears down the file session + auto-starts the camera;
        needs a source face to build the chain."""
        if self._pickers.source_path() is None:
            self._status_bar.show_message("Load a source face first", 4000)
            return
        self._persist_camera_config()
        self._session.set_target(CameraConfig(
            device=self._live_view.device(),
            width=self._live_view.width(),
            height=self._live_view.height(),
            fps=self._live_view.fps(),
            workers=self._live_view.workers(),
            mjpeg_port=self._live_view.port(),
        ))

    def _persist_camera_config(self) -> None:
        self._update_settings(
            camera_device=self._live_view.device(),
            camera_width=self._live_view.width(),
            camera_height=self._live_view.height(),
            camera_fps=self._live_view.fps(),
            camera_workers=self._live_view.workers(),
            camera_mjpeg_port=self._live_view.port(),
        )

    def _on_live_running(self, running: bool) -> None:
        self._live_view.set_running(running)
        self._live_view.set_url(self._live.sink_url() if running else None)
        # Reflect the camera session on the transport (play = stop/start; the
        # capability gate keeps seek/volume disabled).
        self._refresh_transport_enabled()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        # Kick the deferred initial session build ONCE, on first show, one
        # event-loop tick later (singleShot(0)) so the window paints before
        # model loading starts.
        if not self._initial_session_started:
            self._initial_session_started = True
            QTimer.singleShot(0, self._start_deferred_initial_session)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._persist_geometry_to_settings()
        # Stop the batch queue FIRST so its runner thread joins
        # before the controller / side panel start tearing down
        # shared resources (models, etc.).
        self._batch_queue.stop()
        self._live.stop()
        self._controller.shutdown()
        # Stop the detection probe thread (debug overlay) so it doesn't
        # outlive Qt during shutdown. Wait in bounded increments rather than a
        # single wait(2000): the first detection can be lazily building the
        # buffalo_l pack, which exceeds 2s — destroying the thread mid-load
        # crashes on exit ('QThread: Destroyed while thread is still running').
        if not _join_qthread(
            self._detection_thread, _THREAD_JOIN_WAIT_MS, _THREAD_JOIN_MAX_WAITS
        ):
            _log.warning(
                "detection thread still running at close after %d×%dms; "
                "proceeding without destroying it",
                _THREAD_JOIN_MAX_WAITS,
                _THREAD_JOIN_WAIT_MS,
            )
        # Cancel any in-flight model download + join its thread.
        self._models_view.shutdown()
        # Stop the thumbnail thread pool; without this the daemon
        # workers occasionally outlive Qt and emit GUI-warning noise
        # during interpreter shutdown.
        self._side_panel.shutdown()
        super().closeEvent(event)

    def _restore_processor_settings(self) -> None:
        self._processors.apply_restored_settings(
            realtime_workers=self._settings.realtime_workers,
            strategy_name=self._settings.strategy_name,
            enhancer_enabled=self._settings.enhancer_enabled,
            swapper_enabled=self._settings.swapper_enabled,
            swapper_model=self._settings.swapper_model,
            swapper_detection_interval=self._settings.swapper_detection_interval,
            swapper_detection_size=self._settings.swapper_detection_size,
            swapper_detector=self._settings.swapper_detector,
            swapper_many_faces=self._settings.swapper_many_faces,
            swapper_fast_paste=self._settings.swapper_fast_paste,
            swapper_landmark_refine=self._settings.swapper_landmark_refine,
            swapper_target_sex=self._settings.swapper_target_sex,
            swapper_rotation_compensation=self._settings.swapper_rotation_compensation,
            swapper_rotation_threshold_deg=self._settings.swapper_rotation_threshold_deg,
            swapper_rotation_redetect=self._settings.swapper_rotation_redetect,
            swapper_rotation_angle_source=self._settings.swapper_rotation_angle_source,
            swapper_occlusion_mask=self._settings.swapper_occlusion_mask,
            swapper_occlusion_mode=self._settings.swapper_occlusion_mode,
            swapper_occlusion_parser=self._settings.swapper_occlusion_parser,
            swapper_occluder_model=self._settings.swapper_occluder_model,
            enhancer_model=self._settings.enhancer_model,
            enhancer_upscale=self._settings.enhancer_upscale,
            enhancer_only_center_face=self._settings.enhancer_only_center_face,
            enhancer_codeformer_fidelity=self._settings.enhancer_codeformer_fidelity,
            enhancer_fp16=self._settings.enhancer_fp16,
            playback_mode=self._settings.playback_mode,
            cache_mode=self._settings.cache_mode,
            image_format=self._settings.image_format,
            image_quality=self._settings.image_quality,
            memory_cache_mb=self._settings.memory_cache_mb,
            write_workers=self._settings.write_workers,
            write_queue_size=self._settings.write_queue_size,
            video_backend=self._settings.video_backend,
            reader_pool_size=self._settings.reader_pool_size,
            processing_scale=self._settings.processing_scale,
            synced_max_lag_frames=self._settings.synced_max_lag_frames,
            swapper_providers=self._settings.swapper_providers,
            enhancer_device=self._settings.enhancer_device,
            upscaler_enabled=self._settings.upscaler_enabled,
            upscaler_model=self._settings.upscaler_model,
            upscaler_tile=self._settings.upscaler_tile,
            upscaler_fp16=self._settings.upscaler_fp16,
            upscaler_device=self._settings.upscaler_device,
        )

    def _persist_processor_settings(self) -> None:
        # One capture, one flat map — the same to_settings_kwargs() the widget's
        # restore path consumes, so persisted + restored settings can't diverge.
        self._update_settings(**self._processors.snapshot().to_settings_kwargs())

    def _restore_paths_from_settings(self) -> None:
        # Restore recents BEFORE the current path, so the persist
        # triggered by set_source/set_target (which prepends current
        # to recents) operates on top of the loaded list rather than
        # an empty one. Stale entries (files since deleted) stay in
        # the menu — user can pick "Clear recents" if they want.
        if self._settings.recent_sources:
            self._pickers.set_source_recents(
                [Path(p) for p in self._settings.recent_sources]
            )
        if self._settings.recent_targets:
            self._pickers.set_target_recents(
                [Path(p) for p in self._settings.recent_targets]
            )
        # _restoring_paths defers the heavy first session build: setting the
        # target picker fires _on_target_changed, which records the target
        # instead of building now (the build runs after first paint — see
        # _start_deferred_initial_session). _path_is_file keeps an unreachable
        # persisted path (detached drive) from aborting startup.
        self._restoring_paths = True
        try:
            if self._settings.source_path:
                p = Path(self._settings.source_path)
                if _path_is_file(p):
                    self._pickers.set_source(p)
            if self._settings.target_path:
                p = Path(self._settings.target_path)
                if _path_is_file(p):
                    self._pickers.set_target(p)
        finally:
            self._restoring_paths = False
        # Restore the persisted camera target config into the Live tab (silent —
        # set_config doesn't fire configChanged, so it won't re-persist).
        s = self._settings
        if s.camera_device is not None:
            self._live_view.set_config(
                device=s.camera_device,
                width=s.camera_width or self._live_view.width(),
                height=s.camera_height or self._live_view.height(),
                fps=s.camera_fps or self._live_view.fps(),
                workers=s.camera_workers or self._live_view.workers(),
                mjpeg_port=s.camera_mjpeg_port or self._live_view.port(),
            )
        # Library roots — set_roots is silent (doesn't fire rootsChanged)
        # so restoring won't re-persist the same list. Stale folder
        # roots (parent dir since deleted) and stale file roots are
        # silently dropped inside set_roots; folder contents are
        # re-discovered via the background scanner so a folder root
        # that gained files since last save picks them up automatically.
        if self._settings.library_sources:
            self._side_panel.sources_library().set_roots(
                [Path(p) for p in self._settings.library_sources]
            )
        if self._settings.library_targets:
            self._side_panel.targets_library().set_roots(
                [Path(p) for p in self._settings.library_targets]
            )
        # Per-panel sort (silent — set_sort doesn't emit sortChanged).
        self._side_panel.sources_library().set_sort(
            self._settings.library_sources_sort_field,
            self._settings.library_sources_sort_order,
        )
        self._side_panel.targets_library().set_sort(
            self._settings.library_targets_sort_field,
            self._settings.library_targets_sort_order,
        )

    def _start_deferred_initial_session(self) -> None:
        """Build the session for the restored source+target, deferred to after
        the first paint (scheduled by showEvent).

        Keeping this out of the constructor means the window is visible and
        responsive before the multi-second, model-loading build begins — "GUI
        first, models after". No-op when restore supplied no complete
        source+target pair (the common case: nothing persisted, or a missing /
        detached path was skipped)."""
        target = self._pending_initial_target
        self._pending_initial_target = None
        if target is None:
            return
        self._session.set_target(FileTarget(target))
        # The build records what ORT actually loaded and may flip capabilities —
        # re-run the same chrome refresh the constructor did right after restore.
        self._refresh_transport_enabled()
        self._highlight_failed_providers()

    def _persist_source_path(self, path: Path) -> None:
        self._update_settings(source_path=str(path))

    def _persist_target_path(self, path: Path) -> None:
        self._update_settings(target_path=str(path))

    def _persist_source_recents(self, paths: list) -> None:
        self._update_settings(recent_sources=[str(p) for p in paths])

    def _persist_target_recents(self, paths: list) -> None:
        self._update_settings(recent_targets=[str(p) for p in paths])

    def _persist_library_sources(self, paths: list) -> None:
        self._update_settings(library_sources=[str(p) for p in paths])

    def _persist_library_targets(self, paths: list) -> None:
        self._update_settings(library_targets=[str(p) for p in paths])

    def _persist_sources_sort(self) -> None:
        lib = self._side_panel.sources_library()
        self._update_settings(
            library_sources_sort_field=lib.sort_field(),
            library_sources_sort_order=lib.sort_order(),
        )

    def _persist_targets_sort(self) -> None:
        lib = self._side_panel.targets_library()
        self._update_settings(
            library_targets_sort_field=lib.sort_field(),
            library_targets_sort_order=lib.sort_order(),
        )

    # ---- Batch ----

    def _global_output_dir(self) -> Path | None:
        """Configured global batch-output folder, or None (→ output goes
        next to each task's target)."""
        if self._settings.batch_global_output_path:
            return Path(self._settings.batch_global_output_path)
        return None

    def _on_add_to_batch(self) -> None:
        """Snapshot the currently-loaded source + target + processor
        settings into a new BatchTask, persist it, and append it to
        the Batch tab. No-op if source or target is unset."""
        source = self._pickers.source_path()
        target = self._pickers.target_path()
        if source is None or target is None:
            return
        default_format_value = (
            self._settings.batch_default_format or BatchOutputFormat.VIDEO.value
        )
        try:
            default_format = BatchOutputFormat(default_format_value)
        except ValueError:
            default_format = BatchOutputFormat.VIDEO
        default_cleanup_value = (
            self._settings.batch_default_cleanup or BatchCleanupMode.KEEP.value
        )
        try:
            default_cleanup = BatchCleanupMode(default_cleanup_value)
        except ValueError:
            default_cleanup = BatchCleanupMode.KEEP
        # Capture the processor surface once; the batch task is built from it.
        snap = self._processors.snapshot()
        # Per-processor execution profiles. Carry the captured ONNX providers into
        # the swapper profile (CPU vs GPU is a meaningful choice); workers default
        # to the batch throughput defaults rather than the realtime pool size
        # (live latency tuning ≠ batch throughput tuning). These execution objects
        # are batch-only, so they're built here rather than in the shared snapshot.
        providers = snap.swapper_providers
        swapper_execution = (
            OnnxExecution(workers=DEFAULT_SWAPPER_WORKERS, providers=list(providers))
            if providers
            else OnnxExecution(workers=DEFAULT_SWAPPER_WORKERS)
        )
        enhancer_execution = TorchExecution(
            workers=DEFAULT_ENHANCER_WORKERS, device=snap.enhancer_device,
        )
        upscaler_execution = TorchExecution(
            workers=DEFAULT_UPSCALER_WORKERS, device=snap.upscaler_device,
        )
        task = snap.to_batch_task(
            source_path=source,
            target_path=target,
            output_format=default_format,
            cleanup_mode=default_cleanup,
            swapper_execution=swapper_execution,
            enhancer_execution=enhancer_execution,
            upscaler_execution=upscaler_execution,
        )
        self._batch_store.save(task)
        self._batch_view.append_task(task)
        self._status_bar.show_message(
            f"Added to batch: {source.name} → {target.name}", 3000
        )

    def _set_editing_locked(self, locked: bool) -> None:
        """Lock/unlock the whole live-editing surface (transport, pickers,
        settings + libraries). The Batch tab stays interactive so the queue
        can still be driven; the display becomes a read-only render preview."""
        # Transport also depends on source+target presence, so route through
        # the refresh (which honours both the lock and the picker state).
        self._refresh_transport_enabled()
        self._pickers.setEnabled(not locked)
        self._side_panel.set_editing_locked(locked)

    def _on_batch_task_started(self, _task_id: str) -> None:
        # DaVinci-style: while a batch renders, pause the live executor and
        # lock the ENTIRE editing surface. Two simultaneous ORT sessions
        # contend for the GPU (OOM risk), and — more importantly — the
        # display must act purely as a render preview, not a live edit.
        if not self._batch_active:
            self._batch_failures = []  # first task of a fresh run
        self._batch_active = True
        self._batch_slider_total = -1  # re-arm the position bar for this task
        if self._controller.executor() is not None:
            self._controller.executor().pause()
        self._set_editing_locked(True)
        self._status_bar.show_message("Batch running — editing locked", 5000)

    def _on_batch_progress(self, _task_id: str, progress: BatchProgress) -> None:
        # The editing surface is locked during a render, so repurpose the
        # position bar to track the batch: set the slider range to the stage's
        # frame count once (set_frame_count snaps the value to 0), then advance
        # the playhead to the last rendered frame each tick.
        if progress.stage_total != self._batch_slider_total:
            self._batch_slider_total = progress.stage_total
            self._transport.set_frame_count(progress.stage_total)
        self._transport.set_current_frame(max(0, progress.stage_completed - 1))

    def _on_batch_queue_idle(self) -> None:
        self._batch_active = False
        self._batch_slider_total = -1
        # Restore the position bar to the live session we hijacked it from.
        self._controller.resync_transport()
        self._set_editing_locked(False)
        self._status_bar.show_message(
            "Batch queue idle — editing unlocked", 3000
        )
        # Surface the real failure reason(s) prominently now the run is done —
        # one dialog for the whole run (a continue-on-error run can fail many).
        self._report_batch_failures()

    def _on_batch_task_failed(self, task_id: str, message: str) -> None:
        # Collect for the consolidated dialog at queue-idle (avoids modal spam
        # mid-run); a short status note flags it immediately. The row itself
        # shows "failed" with the reason on hover (batch_view._refresh_row).
        label = self._batch_task_label(task_id)
        self._batch_failures.append((label, message or "unknown error"))
        self._status_bar.show_message(f"Batch task failed: {label}", 8000)

    def _batch_task_label(self, task_id: str) -> str:
        """A readable 'source → target' label for a task id, or the id if the
        task can't be loaded."""
        try:
            if self._batch_store.exists(task_id):
                task = self._batch_store.load(task_id)
                return f"{task.source_path.name} → {task.target_path.name}"
        except Exception:
            pass
        return task_id

    def _report_batch_failures(self) -> None:
        """Show one error dialog summarising every task that failed this run,
        then clear the list. No-op when nothing failed."""
        failures = self._batch_failures
        self._batch_failures = []
        if not failures:
            return
        if len(failures) == 1:
            label, msg = failures[0]
            self._show_error(f"Batch task failed — {label}:\n\n{msg}")
            return
        lines = "\n\n".join(f"• {label}:\n  {msg}" for label, msg in failures)
        self._show_error(f"{len(failures)} batch tasks failed:\n\n{lines}")

    def _on_batch_preview(self, _task_id: str, frame: Frame) -> None:
        # Show what the batch is producing on the (idle) preview surface.
        self._display.show_frame(frame)

    def _on_edit_batch_task(self, task_id: str) -> None:
        if not self._batch_store.exists(task_id):
            return
        task = self._batch_store.load(task_id)
        dlg = QBatchTaskDialog.from_task(
            task, parent=self, global_output_dir=self._global_output_dir()
        )
        if dlg.exec() == dlg.DialogCode.Accepted:
            edited = dlg.to_task()
            self._batch_store.save(edited)
            self._batch_view.reload_from_store()

    def _on_library_source_selected(self, path: Path) -> None:
        """Library tile click → route through the same picker pipeline
        as the file dialog. Setting the picker fires its sourceChanged
        signal which wires straight into the controller."""
        if self._batch_active:
            return  # editing locked during a render
        self._pickers.set_source(path)

    def _on_library_target_selected(self, path: Path) -> None:
        if self._batch_active:
            return
        self._pickers.set_target(path)

    def _update_settings(self, **fields: object) -> None:
        # Keep the in-memory copy authoritative even if the disk write fails:
        # assign BEFORE save() so a transient OSError (full / read-only disk)
        # can't leave self._settings stale while the UI shows the new value —
        # that stale base would silently corrupt every later model_copy. Log the
        # failure instead of swallowing it.
        updated = self._settings.model_copy(update=fields)
        self._settings = updated
        try:
            user_settings.save(updated)
        except Exception as exc:  # noqa: BLE001 - never crash the GUI on a settings write
            _log.warning("failed to persist settings: %s", exc)

    def _restore_geometry_from_settings(self) -> bool:
        hex_str = self._settings.window_geometry_hex
        if not hex_str:
            return False
        try:
            return self.restoreGeometry(QByteArray.fromHex(hex_str.encode()))
        except Exception:
            return False

    def _persist_geometry_to_settings(self) -> None:
        try:
            geom_hex = self.saveGeometry().toHex().data().decode()
            splitter_hex = self._top_splitter.saveState().toHex().data().decode()
            self._update_settings(
                window_geometry_hex=geom_hex,
                top_splitter_state_hex=splitter_hex,
            )
        except Exception:
            pass

    def _restore_top_splitter_from_settings(self) -> bool:
        hex_str = self._settings.top_splitter_state_hex
        if not hex_str:
            return False
        try:
            return self._top_splitter.restoreState(
                QByteArray.fromHex(hex_str.encode())
            )
        except Exception:
            return False
