import inspect
import logging
import os
import threading
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sinner2.batch import defaults as batch_defaults
from sinner2.batch.queue import BatchQueue
from sinner2.batch.task import BatchProgress
from sinner2.batch.task_store import BatchTaskStore
from sinner2.config import media_extensions
from sinner2.config import settings as user_settings
from sinner2.gui.face_map_controller import FaceMapController
from sinner2.gui.icon import app_icon
from sinner2.gui.model_download import ensure_models
from sinner2.pipeline.detectors import DETECTOR_MODEL_FILES, DetectorModel
from sinner2.pipeline.face_map_store import (
    canonical_target,
    load_use_map,
    use_map_path,
)
from sinner2.pipeline.sections import SectionSet
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
from sinner2.gui.bridges.thread_safe_callback import ThreadSafeCallback
from sinner2.gui.cache_controller import default_cache_root
from sinner2.gui.confirm import (
    SuppressionStore,
    confirm,
    set_default_suppression_store,
)
from sinner2.gui.live_controller import LiveController
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.preprocess_controller import PreprocessController, trace as _pp_trace
from sinner2.gui.session_capabilities import (
    CameraConfig,
    FileTarget,
    SessionCapabilities,
    SessionKind,
)
from sinner2.gui.session_facade import SessionFacade
from sinner2.gui.project import PROJECT_SUFFIX, Project
from sinner2.gui.widgets.batch_task_dialog import QBatchTaskDialog
from sinner2.gui.widgets.batch_view import QBatchView
from sinner2.gui.widgets.models_view import QModelsView
from sinner2.gui.widgets.face_detection_overlay import FaceDetection
from sinner2.gui.widgets.face_map_panel import QFaceMapPanel
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.batch_coordinator import BatchCoordinator
from sinner2.gui.cache_management_controller import CacheManagementController
from sinner2.gui.face_overlay_controller import FaceOverlayController
from sinner2.gui.fullscreen_controller import FullscreenController
from sinner2.gui.metrics_overlay_controller import MetricsOverlayController
from sinner2.gui.provider_status_controller import ProviderStatusController
from sinner2.gui.settings_binder import SettingsBinder
from sinner2.gui.widgets.fullscreen_control_bar import FullscreenControlBar
from sinner2.gui.widgets.live_view import QLiveView
from sinner2.gui.widgets.metrics_overlay import QMetricsOverlay
from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.gui.widgets.settings_dialog import QSettingsDialog
from sinner2.gui.widgets.side_panel import QSidePanel
from sinner2.gui.widgets.source_target_panel import QSourceTargetPanel
from sinner2.gui.widgets.memory_monitor import MemoryMonitor
from sinner2.gui.widgets.status_action_bar import QStatusActionBar
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.types import Frame

_log = logging.getLogger(__name__)

# Detection-probe thread join budget at close. The first detection lazily
# builds the buffalo_l pack (5 ONNX models + prepare(), can exceed 2s and may
# download), so a single wait(2000) can time out and destroy a running thread.
_THREAD_JOIN_WAIT_MS = 2000
_THREAD_JOIN_MAX_WAITS = 15  # ~30s worst case before giving up and logging

# Set SINNER2_OVERLAY_TRACE=1 to log the detection-overlay poll state (flags +
# sink/overlay counts + the displayed frame) on every tick. Diagnoses an
# intermittent "stuck box" — whether the sink is stale vs the displayed frame,
# and which flag (F8 vs face-map pick mode) is keeping the overlay up.
_OVERLAY_TRACE = bool(os.environ.get("SINNER2_OVERLAY_TRACE"))


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

    # Relays the face-analyser's buffalo_l download start/end across the thread
    # it fires on (a session-setup worker) onto the GUI thread.
    _modelLoadEvent = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("sinner2")
        self.setWindowIcon(app_icon())
        self._settings_binder = SettingsBinder(user_settings.load())
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
        # the display acts purely as a render preview (DaVinci-style). Read across
        # the window; the BatchCoordinator flips it (failures + slider range live
        # on the coordinator).
        self._batch_active = False
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
        # Batch Defaults template — the source of EVERY field of a new task
        # except its source/target. Batch is decoupled from the live preview:
        # "Add to batch" no longer inherits the preview's chain/scale/etc.
        self._batch_defaults_path = batch_defaults.batch_defaults_path()
        self._batch_defaults = batch_defaults.load_defaults(
            self._batch_defaults_path
        )
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
        self._batch_view.settingsRequested.connect(self._on_batch_settings)
        self._models_view = QModelsView()
        self._face_map_panel = QFaceMapPanel()
        # Seed the scan settings from disk (None fields keep the panel defaults).
        self._face_map_panel.restore_settings(
            stride=self._settings.face_analyze_stride,
            workers=self._settings.face_analyze_workers,
            preview=self._settings.face_analyze_preview,
            demographics=self._settings.face_analyze_demographics,
            precompute=self._settings.face_analyze_precompute,
            detector=self._settings.face_analyze_detector,
            detection_size=self._settings.face_analyze_detection_size,
            landmark_refine=self._settings.face_analyze_landmark_refine,
            landmark_min_score=self._settings.face_analyze_landmark_min_score,
            bake_angle=self._settings.face_analyze_bake_angle,
            batch_recognition=self._settings.face_analyze_batch_recognition,
        )
        # Per-panel zoom: fall back to the legacy shared value, then 128.
        _legacy_dim = self._settings.library_display_dim or 128
        self._side_panel = QSidePanel(
            thumbnail_cache_dir=default_cache_root() / "thumbnails",
            processors=self._processors,
            batch_view=self._batch_view,
            face_map_panel=self._face_map_panel,
            sources_display_dim=self._settings.library_sources_display_dim or _legacy_dim,
            targets_display_dim=self._settings.library_targets_display_dim or _legacy_dim,
        )
        # Metrics overlay: child of the frame display so it floats on top
        # of the rendered frame and inherits its z-order. Position is
        # managed manually via resizeEvent (no layout takes child widgets
        # for QFrameDisplayWidget).
        self._metrics_overlay_ctl = MetricsOverlayController(
            controller_getter=lambda: self._controller,
            update_settings=self._update_settings,
            settings_getter=lambda: self._settings,
        )
        self._metrics_overlay = QMetricsOverlay(
            snapshot_fn=self._metrics_overlay_ctl.sample,
            parent=self._display,
        )
        self._metrics_overlay_ctl.set_overlay(self._metrics_overlay)
        # Face-detection debug overlay (+ its detection probe thread + sink) is
        # owned by FaceOverlayController; it reads shared face-map/scan state
        # back through this window. Built here (needs the display + settings).
        self._face_overlay_ctl = FaceOverlayController(self)
        # Auto-hiding playback bar for fullscreen. A child of the display so
        # it floats over the frame; it takes custody of the transport row
        # while fullscreen is active and reveals when the cursor nears the
        # bottom edge. Idle (timer stopped) outside fullscreen.
        self._fs_controls = FullscreenControlBar(self._display)
        # Surface the one-time buffalo_l pack download (insightface fetches it
        # silently inside the first session build, on a worker thread). The
        # notifier hops onto the GUI thread via _modelLoadEvent and shows an
        # indeterminate progress dialog; cleared in closeEvent.
        self._model_load_dialog: QProgressDialog | None = None
        self._modelLoadEvent.connect(self._on_model_load_event)
        from sinner2.pipeline import face_analyser
        face_analyser.set_load_notifier(self._modelLoadEvent.emit)
        # Processing visualiser: polls the live executor's per-frame state and
        # feeds the transport's heatmap bar while the bar is shown (~20 Hz).
        self._visualiser_timer = QTimer(self)
        self._visualiser_timer.setInterval(50)
        self._visualiser_timer.timeout.connect(self._visualiser_tick)
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
        # Project menu lives on a 📂 button in the button bar, right before the
        # pin button (not a top menu bar). _project_path tracks the open file so
        # Save writes back to it instead of re-prompting.
        self._project_path: Path | None = None
        self._menu_button = self._build_project_menu_button()
        self._status_bar = QStatusActionBar()
        self._status_bar.add_leading_button(self._menu_button)
        layout.addWidget(self._status_bar)
        self.setCentralWidget(central)
        # Fullscreen enter/exit is owned by a dedicated controller; the window
        # keeps the keyboard/button wiring and delegates (see _enter_fullscreen).
        self._fullscreen = FullscreenController(
            self,
            fs_controls=self._fs_controls,
            chrome=[self._side_panel, self._pickers, self._status_bar],
            central_layout=self._central_layout,
            transport=self._transport,
            top_splitter=self._top_splitter,
        )
        # ⚙️ Settings dialog (modeless): consolidates Cache + Models + Camera,
        # hosting the existing widget instances so their wiring stays intact.
        # Built once; the button shows/raises it.
        self._settings_dialog = QSettingsDialog(
            cache_widgets=self._processors.cache_widgets(),
            models_view=self._models_view,
            camera_view=self._live_view,
            parent=self,
        )
        self._status_bar.settings_button.clicked.connect(self._open_settings)
        # Drag a media file onto the window (the preview) to load it: videos →
        # target, images → source. The picker ROWS accept their own drops too
        # (forcing the destination regardless of type); this catches drops
        # anywhere else. Disabled while a batch render locks editing.
        self.setAcceptDrops(True)

        self._status_bar.on_top_button.toggled.connect(self._set_stays_on_top)
        self._status_bar.stats_button.toggled.connect(self._set_stats_visible)
        self._status_bar.visualiser_button.toggled.connect(
            self._set_visualiser_visible
        )
        self._status_bar.rotate_button.clicked.connect(self._cycle_rotation)
        self._status_bar.fullscreen_button.toggled.connect(
            self._fullscreen.set_fullscreen
        )
        self._status_bar.side_panel_button.toggled.connect(
            self._set_side_panel_visible
        )
        self._status_bar.save_button.clicked.connect(self._save_current_frame)

        self._status_bar.show_message("ready")
        # Right-side indicator panels (each a discrete cell — icon + value,
        # divider, fixed min-width, hidden while empty). Order: cache · resolution
        # · processing-fps · display-fps · workers · buffer · drops · strategy · EP.
        self._cache_panel = self._status_bar.add_panel(
            "🗄",
            "Persistent processed-frame cache directory for this session "
            "(survives between runs; keyed by source+target+chain config)",
            min_width=56,
            key="cache",
            label="Cache directory",
        )
        self._resolution_panel = self._status_bar.add_panel(
            "📐",
            "Target resolution: native size @ source frame rate, and the\n"
            "processed size when downscaled (processing scale < 100%).",
            min_width=120,
            key="resolution",
            label="Resolution",
        )
        self._fps_panel = self._status_bar.add_panel(
            "⏱",
            "Processing throughput — frames the pipeline COMPLETES per wall-clock "
            "second across all workers (3-second rolling window).",
            min_width=64,
            key="fps",
            label="Processing FPS",
        )
        self._display_fps_panel = self._status_bar.add_panel(
            "🖥",
            "Effective display rate — distinct frames actually SHOWN per second.\n"
            "Lags processing fps when the sync strategy skips frames to keep up.",
            min_width=64,
            key="display_fps",
            label="Display FPS",
        )
        self._workers_panel = self._status_bar.add_panel(
            "👷",
            "Realtime worker threads in effect for this session (after clamping "
            "the configured count to the CPU / per-processor limits).",
            min_width=40,
            key="workers",
            label="Workers",
        )
        self._metrics_panel = self._status_bar.add_panel(
            "▦",
            "cache: hit-ratio / memory used. "
            "writes: outstanding/cap, p50/p95 ms latency.",
            min_width=170,
            key="buffer",
            label="Buffer metrics",
        )
        self._drops_panel = self._status_bar.add_panel(
            "⊘",
            "Frames lost under load. 'skip' = frames the sync strategy never\n"
            "processed (dropped to stay synced with wall-clock); 'drop' =\n"
            "processed frames the write buffer discarded (disk couldn't keep up).",
            min_width=64,
            key="drops",
            label="Dropped frames",
        )
        self._strategy_panel = self._status_bar.add_panel(
            "⏭",
            "Current frame-skip strategy mode. 'synced (lagging)' means\n"
            "SyncedStrategy has fallen back to sequential submission\n"
            "because processing can't keep up — display will trail the\n"
            "timeline but throughput stays at the pipeline's max rate.",
            min_width=72,
            key="strategy",
            label="Strategy mode",
        )
        self._providers_panel = self._status_bar.add_panel(
            "⚡",
            "ONNX execution providers currently in use, in ORT's try-order.\n"
            "Differs from the checkbox column when the user has unchecked\n"
            "everything (system falls back to defaults so inference still works).",
            min_width=96,
            key="providers",
            label="Execution providers",
        )
        self._memory_panel = self._status_bar.add_panel(
            "🧠",
            "Live memory use. VRAM = the GPU device total used / capacity (ALL\n"
            "processes — what 'how close to OOM' depends on); RAM = this process's\n"
            "resident size. Needs nvidia-ml-py for VRAM / psutil for RAM; shows\n"
            "only what's available.",
            min_width=170,
            key="memory",
            label="Memory use",
        )
        self._memory_monitor = MemoryMonitor(self._memory_panel, parent=self)
        # Restore which panels the user hid via the right-click menu, then keep
        # the choice persisted on every toggle.
        for key in self._settings.status_panels_hidden or []:
            self._status_bar.set_panel_user_visible(key, False)
        self._status_bar.panelVisibilityChanged.connect(
            self._on_panel_visibility_changed
        )
        # Session-scoped indicator state (composed from several signals).
        self._native_size: tuple[int, int] | None = None
        self._frames_skipped = 0
        self._write_dropped = 0

        self._controller = PlayerController(self._display, self._transport, parent=self)
        # Wire the swapper's pre-swap detections to the overlay sink (set before
        # any session so every built chain picks it up).
        self._controller.set_detection_sink(self._face_overlay_ctl.sink)
        self._controller.errorOccurred.connect(self._show_error)
        self._controller.processingFpsChanged.connect(self._update_fps_label)
        self._controller.displayFpsChanged.connect(self._update_display_fps_label)
        self._controller.framesSkippedChanged.connect(self._on_frames_skipped)
        self._controller.sessionScratchDirChanged.connect(self._update_scratch_label)
        # Also drives the TensorRT compile dialog on session START (the launch
        # case: TRT persisted + no cached engine → build runs at first load, with
        # no config-change event to hook).
        self._controller.sessionScratchDirChanged.connect(self._on_session_scratch_dir)
        self._controller.targetNativeSizeChanged.connect(
            self._processors.set_target_native_size
        )
        # Second slot: the resolution panel needs the native size too, and the
        # None emit on teardown clears the session-scoped indicator panels.
        self._controller.targetNativeSizeChanged.connect(self._on_native_size_changed)
        self._controller.bufferMetricsChanged.connect(self._update_metrics_label)
        self._controller.strategyModeChanged.connect(self._update_strategy_mode_label)
        self._controller.sessionSwitching.connect(self._on_session_switching)

        # Face mapping: the Faces panel + the analysis job + per-target catalog,
        # coordinated by FaceMapController. The overlay's pick clicks and the
        # Sources-library clicks route in via the handlers below.
        #
        # Two overlays share one widget, with SEPARATE logic (see
        # _refresh_overlay_state): the FACE-MAP overlay (editor open → boxes +
        # selected-identity highlight + click-to-pick, independent of F8/swapper)
        # and the F8 DIAGNOSTIC overlay (boxes only, ONLY while the editor is
        # closed). _faces_mode drives the former; _face_overlay_on (F8) the latter.
        # _face_analyzing forces the overlay fully down while a scan owns the
        # display.
        self._face_analyzing = False
        # _faces_mode = the editor panel is open (pick faces, highlight).
        # _use_face_map = the routing switch in the Face detector group — the
        # SINGLE SOURCE OF TRUTH for whether the map routes playback. Opening the
        # editor turns it on; it persists after the editor closes.
        # _map_available = a usable catalog exists for the current target.
        self._faces_mode = False
        self._use_face_map = False
        self._map_available = False
        self._face_map_ctl = FaceMapController(
            panel=self._face_map_panel,
            player=self._controller,
            detection_sink=self._face_overlay_ctl.sink,
            store_dir=default_cache_root() / "face_maps",
            target_path=self._pickers.target_path,
            providers=lambda: list(self._processors.swapper_providers()) or None,
            # Detection-quality settings come from the Faces panel's own Detection
            # group (D1) — decoupled from the live swapper so the scan can use a
            # different size / refinement than playback.
            detector=lambda: self._face_map_panel.detector(),
            detection_size=lambda: self._face_map_panel.detection_size(),
            current_frame=self._current_display_frame,
            sections=self._controller.sections,
            landmark_refine=lambda: self._face_map_panel.landmark_refine(),
            landmark_min_score=lambda: self._face_map_panel.landmark_min_score(),
            show_preview=self._display.show_frame,
            set_position=self._transport.set_current_frame,
            # Deferred: the session facade is built a few lines below. Route
            # through _on_seek_requested so a row-click jump clears stale boxes.
            navigate=lambda frame, bbox: self._on_face_navigate(frame, bbox),
            status=self._status_bar.show_message,
            parent=self,
        )
        self._face_map_ctl.analyzingChanged.connect(self._on_face_analysis_active)
        self._face_map_panel.resetRequested.connect(self._on_face_map_reset)
        self._face_map_panel.selectionChanged.connect(self._on_face_selection_changed)
        self._face_map_panel.settingsChanged.connect(
            self._persist_face_analyze_settings
        )
        # The in-panel "Use face map" toggle is the SAME routing switch as the one
        # in the Face-recognition settings — same handler keeps them in sync.
        self._face_map_panel.useFaceMapToggled.connect(self._on_use_face_map_toggled)
        # The Faces panel's "Show overlay" toggle controls the face-map overlay.
        self._face_map_panel.showOverlayToggled.connect(
            self._face_overlay_ctl._on_show_overlay_toggled
        )
        # The Sources-tab "Faces" toggle drives face-mapping MODE: reveal the
        # panel, lock the global source picker, enable preview face-picking, and
        # route source-tile clicks to the selected face(s).
        self._side_panel.facesModeToggled.connect(self._on_faces_mode_toggled)

        # Live-camera engine: webcam -> chain -> MJPEG sink. Its preview frames
        # drive the same display; activation + transport are owned by the facade.
        self._live = LiveController(parent=self)
        # Same detection sink the file path uses, so the live swap publishes to
        # the GUI overlay + comparison-crop probe.
        self._live.set_detection_sink(self._face_overlay_ctl.sink)
        self._live.frameReady.connect(lambda f: self._display.show_frame(f))
        self._live.runningChanged.connect(self._on_live_running)
        self._live.errorOccurred.connect(self._show_error)
        self._live.processingFpsChanged.connect(self._update_live_fps_label)
        self._live_view.allowCameraToggled.connect(self._on_allow_camera_toggled)
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
        self._pickers.cameraToggled.connect(self._on_camera_toggled)
        # Preprocessing: render a smart head-start before releasing playback.
        self._preprocess = PreprocessController(
            get_executor=self._controller.executor, parent=self
        )
        self._preprocess.started.connect(self._on_preprocess_started)
        self._preprocess.progressChanged.connect(self._on_preprocess_progress)
        self._preprocess.finished.connect(self._on_preprocess_finished)
        self._preprocess.failed.connect(self._on_preprocess_failed)
        # Play/pause are preprocess-aware: with the "Buffer ahead before playback"
        # execution option on, Play buffers a head-start first; Play again while
        # buffering releases early; Pause while buffering cancels.
        self._transport.playRequested.connect(self._on_play_requested)
        self._transport.pauseRequested.connect(self._on_pause_requested)
        # All user seeks funnel through _on_seek_requested so the overlay drops
        # its stale boxes before the playhead jumps (otherwise a box from the old
        # position lingers "stuck" over the new frame until a fresh detection).
        self._transport.seekRequested.connect(self._on_seek_requested)
        # Timeline section selection ([ / ] in/out) → restrict live playback +
        # carry into the next "Add to batch".
        self._transport.sectionsChanged.connect(self._on_sections_changed)

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
        self._batch_coordinator = BatchCoordinator(
            controller=self._controller,
            transport=self._transport,
            status_bar=self._status_bar,
            display=self._display,
            batch_store=self._batch_store,
            is_active=lambda: self._batch_active,
            set_active=lambda a: setattr(self, "_batch_active", a),
            # Late-bound through self (not captured bound methods) so a test that
            # monkeypatches window._show_error / _set_editing_locked is honored.
            set_editing_locked=lambda locked: self._set_editing_locked(locked),
            show_error=lambda msg: self._show_error(msg),
        )
        self._batch_queue.taskStarted.connect(self._on_batch_task_started)
        self._batch_queue.taskProgress.connect(self._on_batch_progress)
        self._batch_queue.queueIdle.connect(self._on_batch_queue_idle)
        self._batch_queue.taskFailed.connect(self._on_batch_task_failed)
        self._batch_queue.taskPreview.connect(self._on_batch_preview)
        self._batch_view.editRequested.connect(self._on_edit_batch_task)
        self._processors.configChanged.connect(self._on_processor_config_changed)
        self._processors.configChanged.connect(self._persist_processor_settings)
        self._processors.faceOverlayToggled.connect(
            self._face_overlay_ctl._set_face_overlay_visible
        )
        self._processors.faceComparisonToggled.connect(
            self._face_overlay_ctl._set_comparison_visible
        )
        self._processors.useFaceMapToggled.connect(self._on_use_face_map_toggled)
        self._processors.openFaceMapRequested.connect(
            self._side_panel.open_face_map_editor
        )
        self._face_map_ctl.mapAvailabilityChanged.connect(
            self._on_map_availability_changed
        )
        self._face_map_ctl.useForPlaybackRestored.connect(
            self._on_use_for_playback_restored
        )
        self._face_map_ctl.analysisProducedMap.connect(
            self._on_analysis_produced_map
        )
        # Cache-management actions (own signals so they don't go through
        # configChanged, which is for runtime tuning of the chain).
        self._cache_mgmt = CacheManagementController(
            window=self,
            controller=self._controller,
            processors=self._processors,
            update_settings=self._update_settings,
            settings_getter=lambda: self._settings,
        )
        self._provider_status = ProviderStatusController(
            window=self,
            controller=self._controller,
            processors=self._processors,
            providers_panel=self._providers_panel,
            status_bar=self._status_bar,
        )
        self._processors.browseRootRequested.connect(self._cache_mgmt.on_browse_root)
        self._processors.resetRootRequested.connect(self._cache_mgmt.on_reset_root)
        self._processors.invalidateRequested.connect(
            self._cache_mgmt.on_invalidate_session
        )
        self._processors.rerenderRequested.connect(
            self._cache_mgmt.on_rerender_from_current
        )
        self._processors.clearAllRequested.connect(self._cache_mgmt.on_clear_all)
        self._processors.sizeCapChanged.connect(self._cache_mgmt.on_size_cap_changed)
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
        self._side_panel.sources_library().folderModeChanged.connect(
            lambda on: self._update_settings(library_sources_folder_mode=bool(on))
        )
        self._side_panel.targets_library().folderModeChanged.connect(
            lambda on: self._update_settings(library_targets_folder_mode=bool(on))
        )
        # Cache stats (entry count + total size) are a stat-walk of every cache
        # dir — seconds on a large cache. Compute them OFF the GUI thread and
        # apply via a queued callback so session start / teardown / close never
        # freeze the UI; a generation counter drops a slow walk superseded by a
        # newer trigger, and `_closing` skips the walk entirely during shutdown.
        self._closing = False
        self._cache_stats_gen = 0
        self._cache_stats_cb = ThreadSafeCallback(self)
        self._cache_stats_cb.fired.connect(self._apply_cache_stats)
        # The restore below applies saved processor settings; its configChanged
        # must NOT pop a blocking model-download confirm (headless that hangs,
        # for a user it's a startup nag). `_restoring_settings` suppresses the
        # prompt during restore and `_models_confirmed` gates the one deferred
        # confirm that runs before the first session build.
        self._restoring_settings = False
        self._models_confirmed = False
        # Apply persisted processor settings before the first session starts —
        # apply_restored_settings emits configChanged once at the end, which
        # both seeds the controller and persists the (now equal) values back.
        self._restore_processor_settings()
        self._cache_mgmt.restore_state()
        self._restore_audio_state()
        self._refresh_cache_stats()
        self._refresh_providers_label()
        self._restore_side_panel_state()
        self._restore_top_splitter_from_settings()
        self._restore_metrics_overlay_state()
        self._restore_visualiser_state()
        self._face_overlay_ctl._restore_face_overlay_state()
        self._face_overlay_ctl._restore_comparison_state()
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
        # A new target is a new timeline — load that target's remembered
        # selection (or clear, if none), so the old video's regions can't apply
        # to the new one. Same for its face-map catalog.
        self._restore_sections_for_target(target_path)
        self._face_map_ctl.restore_for_target(target_path)
        self._ensure_models_confirmed_before_build()
        self._session.set_target(FileTarget(target_path))

    def _on_sections_changed(self, sections: SectionSet) -> None:
        """Transport [ / ] edit → restrict live playback to the selection,
        remember it for this target, and capture it into the next 'Add to
        batch'."""
        self._controller.set_sections(sections)
        target = self._pickers.target_path()
        if target is not None:
            self._persist_sections(target, sections)

    def _persist_sections(self, target: Path, sections: SectionSet) -> None:
        """Remember (or forget) the selection for ``target`` in settings."""
        existing = dict(self._settings.sections_by_target or {})
        # Canonicalize the key like the face-map sidecars so the SAME file via a
        # different path string maps to one entry (see canonical_target).
        key = canonical_target(target)
        if sections.is_empty():
            existing.pop(key, None)
        else:
            existing[key] = sections.to_pairs()
        self._update_settings(sections_by_target=existing or None)

    def _restore_sections_for_target(self, target: Path) -> None:
        """Apply the selection remembered for ``target`` (empty if none) to the
        timeline AND the executor. set_sections on the transport is silent, so
        push to the controller explicitly."""
        saved = (self._settings.sections_by_target or {}).get(canonical_target(target))
        sections = SectionSet.of(saved) if saved else SectionSet.empty()
        self._transport.set_sections(sections)
        self._controller.set_sections(sections)

    def _update_fps_label(self, fps: float) -> None:
        # File-session throughput; ignored while the camera is the active target
        # so a late paused-executor emission can't overwrite the live reading.
        if self._session.active_kind() is not SessionKind.CAMERA:
            self._fps_panel.set_value(f"{fps:.1f} fps")

    def _update_live_fps_label(self, fps: float) -> None:
        if self._session.active_kind() is SessionKind.CAMERA:
            self._fps_panel.set_value(f"{fps:.1f} fps")

    def _update_display_fps_label(self, fps: float) -> None:
        # File-session only (the live MJPEG path has no separate display rate);
        # ignored while the camera is active so a late emit can't show a stale
        # value over the live reading.
        if self._session.active_kind() is not SessionKind.CAMERA:
            self._display_fps_panel.set_value(f"{fps:.1f} fps")

    def _update_scratch_label(self, scratch_dir: object) -> None:
        # The cell shows just the session cache-key dir name (the full path is
        # long and cluttered the bar); the panel tooltip explains what it is.
        self._cache_panel.set_value(Path(str(scratch_dir)).name if scratch_dir else "")

    def _on_native_size_changed(self, size: object) -> None:
        # (w, h) on session start, None on teardown. Drives the resolution panel
        # and clears the session-scoped cells when the session ends.
        self._native_size = size if size else None  # type: ignore[assignment]
        self._refresh_session_indicators()
        if not size:
            self._display_fps_panel.set_value("")
            self._frames_skipped = 0
            self._write_dropped = 0
            self._update_drops_panel()

    def _refresh_session_indicators(self) -> None:
        # Resolution + worker count read live state, so each hides itself when no
        # session is active. Called on session start/teardown (native-size signal)
        # and on a config change (processing scale / worker count may move).
        self._update_resolution_panel()
        active = self._controller.executor() is not None
        self._workers_panel.set_value(
            str(self._controller.applied_worker_count()) if active else ""
        )

    def _update_resolution_panel(self) -> None:
        size = self._native_size
        if not size:
            self._resolution_panel.set_value("")
            return
        w, h = int(size[0]), int(size[1])
        text = f"{w}×{h}"
        fps = self._controller.target_fps()
        if fps > 0:
            text += f" @{fps:.0f}"
        scale = self._processors.processing_scale()
        if scale and scale < 1.0:
            text += f" → {round(w * scale)}×{round(h * scale)}"
        self._resolution_panel.set_value(text)

    def _on_frames_skipped(self, count: object) -> None:
        self._frames_skipped = count if isinstance(count, int) else 0
        self._update_drops_panel()

    def _update_drops_panel(self) -> None:
        # 'skip' = strategy skips (never processed); 'drop' = write-buffer drops.
        # Cell hides when both are zero.
        parts = []
        if self._frames_skipped:
            parts.append(f"{self._frames_skipped} skip")
        if self._write_dropped:
            parts.append(f"{self._write_dropped} drop")
        self._drops_panel.set_value(" · ".join(parts))

    def _on_panel_visibility_changed(self, _key: str, _visible: bool) -> None:
        # Persist the full hidden set on every toggle (None when nothing hidden,
        # so a fresh install / cleared list defaults all panels visible).
        hidden = self._status_bar.hidden_panel_keys()
        self._update_settings(status_panels_hidden=hidden or None)

    # ---- Cache management slots ----

    def _refresh_cache_stats(self) -> None:
        # Skip during shutdown — closeEvent tears the session down (firing
        # cacheStorageStatsChanged), and walking a large cache root on the GUI
        # thread would freeze the close.
        if self._closing:
            return
        # The invalidate button reflects live GUI state (cheap) — keep it here.
        self._processors.set_invalidate_enabled(
            self._controller.executor() is not None
        )
        # The size/count walk runs off the GUI thread; a generation counter lets
        # _apply_cache_stats drop a slow result that a newer refresh superseded.
        manager = self._controller.cache_manager()
        self._cache_stats_gen += 1
        gen = self._cache_stats_gen

        def _walk() -> None:
            try:
                entries = manager.list_entries()
                payload = (
                    gen,
                    len(entries),
                    sum(e.size_bytes for e in entries),
                    manager.free_disk_bytes(),
                )
            except Exception:  # noqa: BLE001 — a cache read must never crash the worker
                return
            if not self._closing:
                self._cache_stats_cb(payload)

        threading.Thread(target=_walk, name="cache-stats", daemon=True).start()

    def _apply_cache_stats(self, payload: object) -> None:
        # Runs on the GUI thread (queued from the walk worker). Ignore a result
        # superseded by a newer refresh, or one arriving during close.
        gen, count, total, free = payload  # type: ignore[misc]
        if self._closing or gen != self._cache_stats_gen:
            return
        free_text = "free: ?" if free == 0 else f"free: {_fmt_size(free)}"
        self._processors.set_cache_stats_text(
            f"{count} entries · {_fmt_size(total)} · {free_text}"
        )

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
        # Lock target selection while the camera IS the target (the 📹 toggle
        # stays usable). Skipped during a batch render, which owns its own lock.
        if not self._batch_active:
            self._pickers.set_target_enabled(not is_camera)

    def _update_strategy_mode_label(self, mode: object) -> None:
        self._strategy_panel.set_value(str(mode) if mode else "")

    def _refresh_providers_label(self) -> None:
        self._provider_status.refresh_label()

    def _highlight_failed_providers(self) -> None:
        self._provider_status.highlight_failed()

    def _schedule_provider_highlight_refresh(self) -> None:
        self._provider_status.schedule_highlight_refresh()

    def _update_metrics_label(self, metrics: object) -> None:
        # `metrics` is BufferMetrics. Compact cell: hit% · memory · write-queue
        # depth · p50/p95 latency; the drop count appended only when nonzero.
        ratio = getattr(metrics, "cache_hit_ratio", 0.0)
        mem_mb = getattr(metrics, "memory_used_bytes", 0) / 1024 / 1024
        wq_out = getattr(metrics, "write_outstanding", 0)
        wq_max = getattr(metrics, "write_max_outstanding", 0)
        wq_drop = getattr(metrics, "write_dropped", 0)
        wl_p50 = getattr(metrics, "write_latency_p50_ms", 0.0)
        wl_p95 = getattr(metrics, "write_latency_p95_ms", 0.0)
        self._metrics_panel.set_value(
            f"{ratio * 100:.0f}% · {mem_mb:.0f}MB · "
            f"q{wq_out}/{wq_max} · {wl_p50:.0f}/{wl_p95:.0f}ms"
        )
        # Write-buffer drops live in the dedicated drops panel (combined with
        # strategy skips), not crammed into the buffer cell.
        self._write_dropped = int(wq_drop)
        self._update_drops_panel()

    def _on_processor_config_changed(self) -> None:
        if self._batch_active:
            return  # editing is locked while a batch renders

        # Confirm/download any optional weight the new config needs (declining
        # reverts the control to a present default). Skipped during settings
        # restore — window construction must never block on a modal dialog, so
        # the prompt is deferred to the first session build (see
        # _ensure_models_confirmed_before_build); the saved selection is kept.
        if not self._restoring_settings:
            self._confirm_optional_models()
            self._models_confirmed = True

        # The guard above has settled the widget state; capture
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
        self._face_overlay_ctl.configure_probe(
            list(snap.swapper_providers),
            snap.swapper_params.detection_size,
        )
        # Swapper-provider / enhancer-device rebuilds are folded into
        # apply_settings above; just refresh the status-bar EP label
        # and the failed-provider highlight afterwards.
        self._refresh_providers_label()
        # Processing scale / worker count may have changed → refresh the
        # resolution + workers cells.
        self._refresh_session_indicators()
        # A TensorRT-enable triggers a one-time engine compile on the executor
        # (blocks the dispatcher ~25s). Show a modal "compiling" dialog until it
        # finishes rather than leaving a frozen preview + a (prematurely) red
        # provider checkbox; the highlight is refreshed when the wait ends.
        # Otherwise defer the highlight until the async rebuild records the real
        # providers (set_chain is async), so re-checking a provider doesn't flash
        # a spurious red against the previous session's provider list.
        if not self._wait_for_tensorrt_build():
            self._schedule_provider_highlight_refresh()

    def _confirm_optional_models(self) -> None:
        """Confirm + download (or revert) every OPTIONAL weight the current
        processor config selects. Each guard reverts the offending control to a
        present default on decline so the chain never builds against a missing
        model. Modal — only call from a GUI-thread, user-initiated path (NOT
        during settings restore; the deferred first-build confirm calls it)."""
        # If the upscaler / occlusion mask is enabled and its weights aren't
        # present, ask to download them (never silently). Decline → revert the
        # toggle so the chain isn't rebuilt with a missing model.
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

    def _ensure_models_confirmed_before_build(self) -> None:
        """Run the model-download confirm deferred from settings restore, once,
        right before the first session build ("keep selection, prompt on first
        use"). A decline reverts to a present default and the revert re-applies
        via configChanged. No-op after the first user config change already
        confirmed the models."""
        if self._models_confirmed:
            return
        self._models_confirmed = True
        self._confirm_optional_models()

    def _on_session_scratch_dir(self, scratch_dir: object) -> None:
        """A session was (re)installed (non-None dir). If it's about to compile a
        TensorRT engine, surface the modal wait — this covers the launch case
        (TRT persisted, no cached engine, build happens at the first session
        start with no config-change event to hook)."""
        if scratch_dir is not None:
            self._wait_for_tensorrt_build()

    def _wait_for_tensorrt_build(self) -> bool:
        return self._provider_status.wait_for_tensorrt_build()

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

    # ---- Project save / restore ----

    def _build_project_menu_button(self) -> QToolButton:
        """The 📂 caller button (bottom row, before the button bar) that pops up
        the project menu. Parented to the window so its action shortcuts stay
        active without a top menu bar."""
        menu = QMenu(self)
        open_act = menu.addAction("Open Project…")
        open_act.setShortcut("Ctrl+Shift+O")  # Ctrl+S/O are taken by save-frame
        open_act.triggered.connect(self._on_open_project)
        menu.addSeparator()
        save_act = menu.addAction("Save Project")
        save_act.setShortcut("Ctrl+Shift+S")
        save_act.triggered.connect(self._on_save_project)
        save_as_act = menu.addAction("Save Project As…")
        save_as_act.triggered.connect(self._on_save_project_as)
        # Window-context shortcuts work even though the menu only opens on click.
        self.addActions(menu.actions())

        button = QToolButton()
        button.setText("📂")
        button.setToolTip("Project — open / save")
        button.setAutoRaise(True)  # flat, like the other button-bar actions
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        # Drop the menu-indicator arrow ("v") so it reads as a plain icon button
        # like its neighbours — it still opens the menu on click.
        button.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        self._project_menu = menu
        return button

    def _open_settings(self) -> None:
        """Show the ⚙️ Settings window (Cache / Models / Camera), modeless so its
        changes preview live against the main window."""
        self._settings_dialog.show_and_raise()

    def _set_project_path(self, path: Path | None) -> None:
        self._project_path = path
        self.setWindowTitle(f"sinner2 — {path.name}" if path is not None else "sinner2")

    def _capture_project(self) -> Project:
        """Snapshot the current working state into a Project value object."""
        sections = self._transport.sections()
        return Project(
            source_path=self._pickers.source_path(),
            target_path=self._pickers.target_path(),
            sections=None if sections.is_empty() else sections.to_pairs(),
            processor=self._processors.snapshot().to_settings_kwargs(),
        )

    def _apply_project(self, project: Project) -> None:
        """Restore a project by re-driving the normal load path: apply the chain
        config (via the settings-restore path), set source + target (the pickers
        emit → the session builds), then override the section selection."""
        # Chain config: coerce the stored string tokens back to enums by
        # re-validating through the Settings model, then apply to the controls.
        if project.processor:
            merged = self._settings.model_dump()
            merged.update(project.processor)
            try:
                self._settings = user_settings.Settings.model_validate(merged)
                user_settings.save(self._settings)
            except Exception as exc:  # noqa: BLE001 — a bad field shouldn't abort
                _log.warning("project chain config rejected: %s", exc)
            else:
                self._restore_processor_settings()
        # Media — setting the pickers emits, reusing the file-pick load flow
        # (which builds the session and restores the target's saved sections).
        if project.source_path is not None:
            self._pickers.set_source(project.source_path)
        if project.target_path is not None:
            self._pickers.set_target(project.target_path)
        # The project's selection overrides the target's remembered one (or
        # clears it when the project had none).
        sections = (
            SectionSet.of(project.sections)
            if project.sections
            else SectionSet.empty()
        )
        self._transport.set_sections(sections)
        self._controller.set_sections(sections)
        if project.target_path is not None:
            self._persist_sections(project.target_path, sections)

    def _on_open_project(self) -> None:
        start = str(self._project_path.parent) if self._project_path else ""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open project", start, f"Sinner project (*{PROJECT_SUFFIX})"
        )
        if not path_str:
            return
        try:
            project = Project.load(Path(path_str))
        except (OSError, ValueError) as exc:  # JSONDecodeError ⊂ ValueError
            QMessageBox.warning(
                self, "Open project", f"Couldn't open the project:\n{exc}"
            )
            return
        self._apply_project(project)
        self._set_project_path(Path(path_str))

    def _on_save_project(self) -> None:
        if self._project_path is None:
            self._on_save_project_as()
            return
        self._write_project(self._project_path)

    def _on_save_project_as(self) -> None:
        start = str(self._project_path) if self._project_path else ""
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save project as", start, f"Sinner project (*{PROJECT_SUFFIX})"
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix != PROJECT_SUFFIX:
            path = path.with_suffix(PROJECT_SUFFIX)
        if self._write_project(path):
            self._set_project_path(path)

    def _write_project(self, path: Path) -> bool:
        try:
            self._capture_project().save(path)
        except OSError as exc:
            QMessageBox.warning(
                self, "Save project", f"Couldn't save the project:\n{exc}"
            )
            return False
        self._status_bar.show_message(f"Project saved: {path.name}", 3000)
        return True

    # ---- Drag-and-drop loading ----

    def _dropped_media_paths(self, event: QDropEvent | QDragEnterEvent) -> list[Path]:
        """Local media files in a drag, or [] when there are none / a batch is
        rendering (editing locked). Used by the drag/drop handlers to decide
        whether to accept and how to route."""
        if self._batch_active:
            return []
        md = event.mimeData()
        if not md.hasUrls():
            return []
        out: list[Path] = []
        for url in md.urls():
            local = url.toLocalFile()
            if local and media_extensions.is_media_ext(Path(local)):
                out.append(Path(local))
        return out

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._dropped_media_paths(event):
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if self._dropped_media_paths(event):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = self._dropped_media_paths(event)
        if not paths:
            return
        event.acceptProposedAction()
        # Route by type: a video → target, an image (a face) → source. Setting
        # the picker emits its change signal, so this reuses the exact file-pick
        # load path. Drop ONTO a picker row to force the destination by type.
        target = next(
            (p for p in paths if media_extensions.is_video_ext(p)), None
        )
        source = next(
            (p for p in paths if media_extensions.is_image_ext(p)), None
        )
        if target is not None:
            self._pickers.set_target(target)
        if source is not None:
            self._pickers.set_source(source)

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
        if key == Qt.Key.Key_F6:
            self._status_bar.visualiser_button.toggle()
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
        if (
            key == Qt.Key.Key_Escape
            and self._transport.pending_in() is not None
        ):
            # A half-marked section ([ pressed, no ] yet) → cancel it. Only
            # consumed when there IS a pending mark, so normal Escape still works.
            self._transport.cancel_pending()
            return
        if not self._session.is_active():
            super().keyPressEvent(event)
            return
        # Route through the facade so the keys reach the ACTIVE session (file or
        # camera); the player keeps the audio-aware play/seek logic. Space is
        # play/pause for any session; the seek keys apply only to seekable
        # (finite) targets.
        if key == Qt.Key.Key_Space:
            self._toggle_play()
            return
        executor = self._controller.executor()
        if not self._session.capabilities().seekable or executor is None:
            super().keyPressEvent(event)
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            # Frame stepping with precision modifiers: Ctrl = 100, Shift = 10,
            # plain = 1. Lets the user land exactly on a boundary before [ / ].
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:
                step = 100
            elif mods & Qt.KeyboardModifier.ShiftModifier:
                step = 10
            else:
                step = 1
            current = executor.current_frame.get()
            target = (
                current - step if key == Qt.Key.Key_Left else current + step
            )
            # Floor at 0; the timeline clamps the upper bound to the last frame.
            self._on_seek_requested(max(0, target))
            return
        if key == Qt.Key.Key_Home:
            self._on_seek_requested(0)
            return
        if key == Qt.Key.Key_End:
            self._on_seek_requested(max(0, executor.frame_count() - 1))
            return
        if key == Qt.Key.Key_P:
            # Jump to the next (Shift = previous) processed frame where detection
            # found no face — a likely swap failure. Works whether or not the
            # visualiser bar is shown.
            forward = not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            problem = executor.next_problem_frame(
                executor.current_frame.get(), forward
            )
            if problem is not None:
                self._on_seek_requested(problem)
            else:
                self._status_bar.show_message(
                    "No more no-face frames in that direction.", 2000
                )
            return
        # Section selection: [ sets a section's start at the playhead, ] its
        # end; Delete removes the selected band. The transport owns the state
        # machine (new in-point vs nudge-selected) and emits sectionsChanged.
        if key == Qt.Key.Key_BracketLeft:
            self._transport.mark_in(executor.current_frame.get())
            return
        if key == Qt.Key.Key_BracketRight:
            self._transport.mark_out(executor.current_frame.get())
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._transport.delete_selected()
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

    @property
    def _is_fullscreen(self) -> bool:
        # Backed by FullscreenController; kept as a window alias so the keyboard
        # handlers (Esc), the resize reposition, and the tests read it here.
        return self._fullscreen.is_fullscreen

    def _enter_fullscreen(self) -> None:
        self._fullscreen.enter()

    def _exit_fullscreen(self) -> None:
        self._fullscreen.exit()

    # ---- Metrics overlay ----

    def _set_stats_visible(self, on: bool) -> None:
        self._metrics_overlay_ctl.set_visible(on)

    def _restore_metrics_overlay_state(self) -> None:
        visible = self._metrics_overlay_ctl.restore_state()
        self._set_button_checked(self._status_bar.stats_button, visible)

    def _set_visualiser_visible(self, on: bool) -> None:
        self._transport.set_visualiser_visible(on)
        if on:
            self._visualiser_timer.start()
            self._visualiser_tick()  # paint at once rather than after a tick
        else:
            self._visualiser_timer.stop()
        self._update_settings(visualiser_visible=on)

    def _visualiser_tick(self) -> None:
        """Feed the transport's heatmap the live executor's per-frame state.
        No-op when no file session is active (e.g. camera mode / mid-swap)."""
        executor = self._controller.executor()
        if executor is None:
            return
        self._transport.set_frame_states(
            executor.frame_states_snapshot(),
            executor.frame_count(),
            executor.face_states_snapshot(),
        )

    def _restore_visualiser_state(self) -> None:
        visible = bool(self._settings.visualiser_visible)
        self._transport.set_visualiser_visible(visible)
        if visible:
            self._visualiser_timer.start()
        self._set_button_checked(self._status_bar.visualiser_button, visible)

    # ---- Preprocessing ----

    def _preprocess_on_play_eligible(self) -> bool:
        """The 'buffer ahead before playback' option is on AND this is a file
        session (preprocessing is meaningless for a live camera)."""
        return (
            self._processors.preprocess_before_play()
            and self._controller.executor() is not None
            and self._session.active_kind() is not SessionKind.CAMERA
        )

    def _on_play_requested(self) -> None:
        # While buffering, Play releases the head-start early. With the option on,
        # Play buffers a head-start first; otherwise it's the normal session play.
        _pp_trace(
            f"play requested: option={self._processors.preprocess_before_play()} "
            f"executor={self._controller.executor() is not None} "
            f"kind={self._session.active_kind()} "
            f"buffering={self._preprocess.is_active()}"
        )
        if self._preprocess.is_active():
            self._preprocess.play_now()
        elif self._preprocess_on_play_eligible():
            self._preprocess.start(self._controller.target_fps())
        else:
            _pp_trace("→ normal play (option off or not a file session)")
            self._session.play()

    def _on_pause_requested(self) -> None:
        # Pause while buffering cancels the preprocess pass; otherwise normal.
        if self._preprocess.is_active():
            self._preprocess.cancel()
        else:
            self._session.pause()

    def _toggle_play(self) -> None:
        """Space: release early while buffering; start preprocessing on a paused
        file session with the option on; otherwise the normal play/pause toggle."""
        if self._preprocess.is_active():
            self._preprocess.play_now()
            return
        executor = self._controller.executor()
        paused = executor is not None and not executor.is_playing.get()
        if paused and self._preprocess_on_play_eligible():
            self._preprocess.start(self._controller.target_fps())
        else:
            self._session.toggle_playback()

    def _on_preprocess_started(self) -> None:
        # Show the visualiser so its filling green bar IS the progress display,
        # silence audio while buffering, and mark the play button "Buffering…".
        self._set_button_checked(self._status_bar.visualiser_button, True)
        self._set_visualiser_visible(True)
        self._transport.set_buffering(True)
        self._controller.preprocess_audio_start()
        self._status_bar.show_message("Preprocessing…")

    def _on_preprocess_progress(self, done: int, target: int) -> None:
        pct = int(100 * done / target) if target > 0 else 0
        self._status_bar.show_message(
            f"Preprocessing… {pct}%  ({done}/{target})"
        )

    def _on_preprocess_finished(self, played: bool) -> None:
        self._transport.set_buffering(False)
        if played:
            self._controller.preprocess_audio_release()
            self._status_bar.show_message("Playing (preprocessed)", 3000)
        else:
            self._status_bar.show_message("Preprocess cancelled", 3000)

    def _on_preprocess_failed(self, message: object) -> None:
        self._status_bar.show_message(f"Preprocess: {message}", 4000)

    def _reposition_metrics_overlay(self) -> None:
        self._metrics_overlay_ctl.reposition()

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

    # Face-detection overlay → FaceOverlayController (these thin delegators +
    # aliases keep the window API the face-map handlers + tests call).
    def _overlay_active(self) -> bool:
        return self._face_overlay_ctl._overlay_active()

    def _face_map_overlay_on(self) -> bool:
        return self._face_overlay_ctl._face_map_overlay_on()

    def _diagnostic_overlay_on(self) -> bool:
        return self._face_overlay_ctl._diagnostic_overlay_on()

    def _set_face_overlay_visible(self, on: bool) -> None:
        self._face_overlay_ctl._set_face_overlay_visible(on)

    @property
    def _face_overlay(self):  # type: ignore[no-untyped-def]
        return self._face_overlay_ctl._face_overlay

    @property
    def _detection_sink(self):  # type: ignore[no-untyped-def]
        return self._face_overlay_ctl._detection_sink

    @property
    def _overlay_timer(self):  # type: ignore[no-untyped-def]
        return self._face_overlay_ctl._overlay_timer

    @property
    def _last_displayed_frame(self):  # type: ignore[no-untyped-def]
        return self._face_overlay_ctl._last_displayed_frame

    @_last_displayed_frame.setter
    def _last_displayed_frame(self, value) -> None:  # type: ignore[no-untyped-def]
        self._face_overlay_ctl._last_displayed_frame = value

    @property
    def _face_overlay_on(self) -> bool:
        return self._face_overlay_ctl._face_overlay_on

    @_face_overlay_on.setter
    def _face_overlay_on(self, value: bool) -> None:
        self._face_overlay_ctl._face_overlay_on = value

    @property
    def _comparison_on(self) -> bool:
        return self._face_overlay_ctl._comparison_on

    @_comparison_on.setter
    def _comparison_on(self, value: bool) -> None:
        self._face_overlay_ctl._comparison_on = value

    @property
    def _requestDetection(self):  # type: ignore[no-untyped-def]
        # The probe-request signal now lives on the controller; alias it so tests
        # can connect to verify a frame was submitted to the probe.
        return self._face_overlay_ctl._requestDetection

    def _current_display_frame(self) -> int:
        ex = self._controller.executor()
        return ex.current_frame.get() if ex is not None else 0

    def _on_faces_mode_toggled(self, on: bool) -> None:
        """The Sources-tab "Face map" toggle opens/closes the EDITOR (click-to-pick
        faces + selection highlight). It does NOT touch routing — the "Use face
        map" switch owns that, and only unlocks once a map is built (so the switch
        never sits enabled with nothing to route). Assigning a source flips the
        switch on (see _on_library_source_selected)."""
        self._faces_mode = on
        self._refresh_overlay_state()
        self._refresh_face_highlight()  # clears the highlight when the editor closes

    def _on_use_face_map_toggled(self, on: bool) -> None:
        """The user toggled the Face-detector "Use face map" switch."""
        self._set_use_face_map(bool(on))

    def _on_use_for_playback_restored(self, use: bool) -> None:
        """A target's saved 'use the map' preference loaded — apply it without
        re-persisting."""
        self._set_use_face_map(bool(use), persist=False)

    def _on_map_availability_changed(self, available: bool) -> None:
        """A usable catalog appeared/vanished for the current target."""
        self._map_available = bool(available)
        self._update_face_map_enabled()
        if not available:
            self._set_use_face_map(False)  # nothing to route

    def _on_analysis_produced_map(self, has_map: bool) -> None:
        """A fresh scan built a usable catalog → turn the "Use face map" switch ON
        so the map drives playback at once (the user shouldn't have to flip it
        manually after every analysis). Routing-on pushes the armed map + geometry,
        which reprocesses the current frame — so the mapped swap appears and the
        detection sink repopulates for the selection highlight. Target-restore
        honours the saved preference instead (useForPlaybackRestored)."""
        if has_map and not self._use_face_map:
            self._set_use_face_map(True)

    def _update_face_map_enabled(self) -> None:
        """The switch unlocks ONLY once a map is built for the target — never just
        because the editor is open (a switch you can flip with nothing to route is
        confusing). Both copies of the switch (settings + Face scanner panel)."""
        self._processors.set_face_map_available(self._map_available)
        self._face_map_panel.set_face_map_available(self._map_available)

    def _set_use_face_map(self, on: bool, *, persist: bool = True) -> None:
        """SINGLE SOURCE OF TRUTH for face-map mode: set the switch state, reflect
        it on the checkbox, (optionally) persist per target, apply swap routing,
        AND flip the overlay (face-map ↔ diagnostic). This toggle is the one gate
        between single-source and face-map — overlay included."""
        self._use_face_map = bool(on)
        self._processors.set_use_face_map(on)
        self._face_map_panel.set_use_face_map(on)  # the linked in-panel copy
        if persist:
            self._face_map_ctl.set_use_for_playback(on)
        self._refresh_face_map_routing()
        self._refresh_overlay_state()
        self._refresh_face_highlight()

    def _refresh_face_map_routing(self) -> None:
        """Routing is active ⟺ the "Use face map" switch is on. When active the
        global source picker + the superseded swap controls lock, and the
        controller arms the map + pushes the geometry."""
        active = self._use_face_map
        self._pickers.set_source_enabled(not active)
        self._processors.set_face_map_routing_active(active)
        self._face_map_ctl.set_mode_active(active)

    def _refresh_face_highlight(self) -> None:
        self._face_overlay_ctl._refresh_face_highlight()

    def _on_face_selection_changed(self) -> None:
        """A Faces-list selection changed: highlight the chosen identity's box
        now — and, with the swapper OFF, kick a fresh detection of the current
        frame so the box is current even while paused (the highlight reads the
        detection sink, which the probe fills in the swapper-off case)."""
        self._refresh_face_highlight()
        if (
            self._face_map_overlay_on()
            and not self._face_analyzing
            and not self._processors.swapper_enabled()
            and self._last_displayed_frame is not None
        ):
            self._submit_to_probe(self._last_displayed_frame)

    def _on_face_analysis_active(self, active: bool) -> None:
        """Lock the editing surface while a face-map scan runs (like a batch
        render): pause live playback so the scan owns the device + the preview,
        and disable transport/settings/pickers. The Faces panel stays live so
        Cancel works. On finish, unlock (unless a batch is still running). The
        scan owns the display, so _face_analyzing forces the overlay down (no
        boxes over the scan); finishing restores it via _refresh_overlay_state."""
        self._face_analyzing = active
        if active:
            self._session.pause()
        # A SCAN keeps the Faces panel interactive (lock_faces=False) so its
        # Cancel button is reachable; a BATCH render locks everything. (Scan and
        # batch don't co-occur, so active=True ⇒ not a batch.)
        self._set_editing_locked(
            active or self._batch_active, lock_faces=not active
        )
        self._refresh_overlay_state()

    def _refresh_overlay_state(self) -> None:
        self._face_overlay_ctl._refresh_overlay_state()

    def _clear_overlay_for_seek(self) -> None:
        self._face_overlay_ctl._clear_overlay_for_seek()

    def _on_seek_requested(self, frame: int) -> None:
        """The single funnel for user seeks (transport knob, arrow/Home/End,
        face-row navigate): clear stale overlay boxes, then seek the active
        session."""
        self._clear_overlay_for_seek()
        self._session.seek_to(int(frame))

    def _on_face_navigate(self, frame: int, detection: object) -> None:
        """A found-face row was clicked: seek to it (clearing stale boxes), then
        draw that face's box + scanned age/sex/angle straight from the catalog so
        the overlay shows even on a cached frame the swapper skips — instead of
        relying on a live re-detect that can miss it."""
        self._on_seek_requested(frame)
        if isinstance(detection, FaceDetection) and self._native_size is not None:
            w, h = self._native_size
            self._face_overlay_ctl.show_catalog_face(detection, w, h)

    def _submit_to_probe(self, frame: Frame) -> None:
        self._face_overlay_ctl._submit_to_probe(frame)

    def _on_allow_camera_toggled(self, allowed: bool) -> None:
        """The Camera-tab gate: reveal/hide the 📹 toggle + persist the choice."""
        self._pickers.set_camera_button_visible(allowed)
        self._update_settings(camera_mode_allowed=allowed)

    def _on_camera_toggled(self, on: bool) -> None:
        """The 📹 toggle drove the mode: start the camera, or stop it. Reverts
        the button if the camera can't start (e.g. no source). Stop routes
        through the facade so it leaves CAMERA + restores the file-only chrome."""
        if on:
            if not self._on_use_camera():
                self._pickers.set_camera_active(False)
        else:
            self._session.deactivate_camera()

    def _on_use_camera(self) -> bool:
        """Make the camera the active target. The facade tears down the file
        session + auto-starts the camera; needs a source face to build the
        chain. Returns whether the start was issued."""
        if self._pickers.source_path() is None:
            self._status_bar.show_message("Load a source face first", 4000)
            return False
        self._persist_camera_config()
        self._ensure_models_confirmed_before_build()
        self._session.set_target(CameraConfig(
            device=self._live_view.device(),
            width=self._live_view.width(),
            height=self._live_view.height(),
            fps=self._live_view.fps(),
            workers=self._live_view.workers(),
            mjpeg_port=self._live_view.port(),
        ))
        return True

    def _persist_face_analyze_settings(self) -> None:
        p = self._face_map_panel
        self._update_settings(
            face_analyze_stride=p.stride(),
            face_analyze_workers=p.workers(),
            face_analyze_preview=p.preview_enabled(),
            face_analyze_demographics=p.detect_demographics(),
            face_analyze_precompute=p.precompute_geometry(),
            face_analyze_detector=p.detector().value,
            face_analyze_detection_size=p.detection_size(),
            face_analyze_landmark_refine=p.landmark_refine(),
            face_analyze_landmark_min_score=p.landmark_min_score(),
            face_analyze_bake_angle=p.bake_angle(),
            face_analyze_batch_recognition=p.batch_recognition(),
        )

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
        # The 📹 toggle is the single source of truth — sync it to reality (a
        # failed start or external stop unchecks it).
        self._pickers.set_camera_active(running)
        # Face-mapping is file-only — a camera can't be precomputed; disable the
        # Faces toggle (and clear the mode) while the camera session runs.
        self._side_panel.set_faces_available(not running)
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

    def _on_model_load_event(self, text: str) -> None:
        """Show/hide an indeterminate busy dialog for the buffalo_l download
        (non-empty text = a download is in progress; "" = done). Runs on the
        GUI thread via the _modelLoadEvent relay."""
        if text:
            if self._model_load_dialog is None:
                dialog = QProgressDialog(text, "", 0, 0, self)  # 0,0 = busy spinner
                dialog.setWindowTitle("Preparing models")
                dialog.setCancelButton(None)  # insightface's fetch can't be cancelled
                dialog.setWindowModality(Qt.WindowModality.WindowModal)
                dialog.setMinimumDuration(0)
                dialog.setAutoClose(False)
                dialog.setAutoReset(False)
                self._model_load_dialog = dialog
                dialog.show()
            else:
                self._model_load_dialog.setLabelText(text)
        elif self._model_load_dialog is not None:
            self._model_load_dialog.close()
            self._model_load_dialog = None

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        # Mark closing FIRST so the session teardown below (which fires
        # cacheStorageStatsChanged) doesn't kick off a cache-stat walk on the
        # way out — _refresh_cache_stats / _apply_cache_stats both early-return.
        self._closing = True
        self._persist_geometry_to_settings()
        from sinner2.pipeline import face_analyser
        face_analyser.set_load_notifier(None)  # drop the bound-signal reference
        # Stop the provider-highlight poll FIRST: a queued tick during teardown
        # would touch the processor/controller after they're gone.
        self._provider_status.stop()
        # Stop the batch queue FIRST so its runner thread joins
        # before the controller / side panel start tearing down
        # shared resources (models, etc.).
        self._batch_queue.stop()
        self._live.stop()
        # Stop the face-map analysis thread before the controller/models tear down.
        self._face_map_ctl.shutdown()
        self._controller.shutdown()
        # Stop the detection probe thread (debug overlay) so it doesn't outlive
        # Qt during shutdown — bounded-join (the first detection can be lazily
        # building the buffalo_l pack, exceeding a single 2s wait).
        if not self._face_overlay_ctl.stop():
            _log.warning(
                "detection thread still running at close; "
                "proceeding without destroying it"
            )
        # Cancel any in-flight model download + join its thread.
        self._models_view.shutdown()
        # Stop the thumbnail thread pool; without this the daemon
        # workers occasionally outlive Qt and emit GUI-warning noise
        # during interpreter shutdown.
        self._side_panel.shutdown()
        super().closeEvent(event)

    def _restore_processor_settings(self) -> None:
        # apply_restored_settings emits configChanged synchronously at the end;
        # the flag makes its handler skip the blocking model-download confirm
        # (deferred to the first session build — see
        # _ensure_models_confirmed_before_build). Keeps the saved selection.
        self._restoring_settings = True
        try:
            # Every keyword was `name=self._settings.<name>`, so spread the
            # persisted values straight from the canonical key set — the
            # method's keyword-only params, the same surface to_settings_kwargs()
            # persists. The explicit signature stays the contract; parity is
            # pinned by test_restore_keys_all_exist_on_settings (every key must
            # resolve on Settings) + the to_settings_kwargs keys-match test.
            restore_params = inspect.signature(
                self._processors.apply_restored_settings
            ).parameters
            self._processors.apply_restored_settings(**{
                name: getattr(self._settings, name)
                for name, p in restore_params.items()
                if p.kind is p.KEYWORD_ONLY
            })
        finally:
            self._restoring_settings = False

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
        # Restore the persisted camera target config into the Camera tab (silent
        # — set_config doesn't fire configChanged, so it won't re-persist).
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
        # Restore the "Allow camera mode" gate → 📹 button visibility (silent).
        camera_allowed = bool(s.camera_mode_allowed)
        self._live_view.set_allow_camera(camera_allowed)
        self._pickers.set_camera_button_visible(camera_allowed)
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
        # Per-panel folder-mirror toggle (silent — set_folder_mode doesn't emit).
        self._side_panel.sources_library().set_folder_mode(
            bool(self._settings.library_sources_folder_mode)
        )
        self._side_panel.targets_library().set_folder_mode(
            bool(self._settings.library_targets_folder_mode)
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
        # Restore the selection + face-map catalog remembered for this target
        # before the build, so the live executor starts already configured.
        self._restore_sections_for_target(target)
        self._face_map_ctl.restore_for_target(target)
        self._ensure_models_confirmed_before_build()
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
        """Mint a new BatchTask from the Batch Defaults template — carrying
        over ONLY the currently-loaded source + target — persist it, and
        append it to the Batch tab. No-op if source or target is unset.

        Batch is deliberately decoupled from the live preview: every other
        field (chain look, execution profiles, processing scale, output
        policy) comes from the editable defaults, not from the preview's
        tuning. Each task is still individually editable afterwards."""
        source = self._pickers.source_path()
        target = self._pickers.target_path()
        if source is None or target is None:
            return
        task = batch_defaults.mint_task(self._batch_defaults, source, target)
        # Point the task at the per-target face-map sidecar store so the driver
        # loads the CURRENT catalog + geometry live at render time, and capture
        # the live "use the map" routing preference as the task's own per-task
        # flag (editable later in the task dialog). No map / routing-off → single
        # source.
        store_dir = default_cache_root() / "face_maps"
        # Append to the end of the queue: one past the current max position.
        next_order = max(
            (t.order for t in self._batch_store.list()), default=-1
        ) + 1
        task = task.model_copy(update={
            "face_map_store_dir": str(store_dir),
            "use_face_map": load_use_map(use_map_path(target, store_dir)),
            "order": next_order,
        })
        # Carry the live timeline selection (like source/target, it's specific to
        # THIS target — not part of the reusable defaults template). Empty → the
        # whole video, which is BatchTask's own default.
        sections = self._transport.sections()
        if not sections.is_empty():
            task = task.model_copy(update={"sections": sections.to_pairs()})
        self._batch_store.save(task)
        self._batch_view.append_task(task)
        self._status_bar.show_message(
            f"Added to batch: {source.name} → {target.name}", 3000
        )

    def _on_batch_settings(self) -> None:
        """Edit the Batch Defaults template + the queue-wide paths (task store
        + global output), then persist them. The store-path change applies on
        the next launch (the store is built at startup); the global-output
        change applies immediately to the queue and the displayed output names."""
        dlg = QBatchTaskDialog(
            self._batch_defaults,
            parent=self,
            defaults_mode=True,
            store_path=self._settings.batch_store_path or "",
            global_output_path=self._settings.batch_global_output_path or "",
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        self._batch_defaults = dlg.to_task()
        batch_defaults.save_defaults(self._batch_defaults_path, self._batch_defaults)
        old_store = self._settings.batch_store_path or ""
        new_store = dlg.store_path() or None
        new_global = dlg.global_output_path() or None
        self._update_settings(
            batch_store_path=new_store,
            batch_global_output_path=new_global,
        )
        # Global output feeds the queue's driver (built per run) and the Batch
        # tab's resolved output column — apply both live.
        self._batch_queue.set_global_output_dir(self._global_output_dir())
        self._batch_view.reload_from_store()
        if (new_store or "") != old_store:
            self._status_bar.show_message(
                "Task store folder change takes effect after restart.", 5000
            )
        else:
            self._status_bar.show_message("Batch settings saved.", 3000)

    def _set_editing_locked(self, locked: bool, *, lock_faces: bool = True) -> None:
        """Lock/unlock the whole live-editing surface (transport, pickers,
        settings + libraries). The Batch tab stays interactive so the queue
        can still be driven; the display becomes a read-only render preview.
        ``lock_faces=False`` (a scan, not a batch) keeps the Faces panel live so
        its Cancel button works."""
        # Transport also depends on source+target presence, so route through
        # the refresh (which honours both the lock and the picker state).
        self._refresh_transport_enabled()
        self._pickers.setEnabled(not locked)
        if not locked:
            # Re-honour the face-mapping source lock (re-enabling the whole
            # picker above would otherwise unlock the source the map owns) —
            # the lock follows ROUTING (the "Use face map" switch), not the editor.
            self._pickers.set_source_enabled(not self._use_face_map)
        self._side_panel.set_editing_locked(locked, lock_faces=lock_faces)

    def _on_batch_task_started(self, task_id: str) -> None:
        self._batch_coordinator.on_task_started(task_id)

    def _on_batch_progress(self, task_id: str, progress: BatchProgress) -> None:
        self._batch_coordinator.on_progress(task_id, progress)

    def _on_batch_queue_idle(self) -> None:
        self._batch_coordinator.on_queue_idle()

    def _on_batch_task_failed(self, task_id: str, message: str) -> None:
        self._batch_coordinator.on_task_failed(task_id, message)

    def _batch_task_label(self, task_id: str) -> str:
        return self._batch_coordinator.task_label(task_id)

    def _report_batch_failures(self) -> None:
        self._batch_coordinator.report_failures()

    def _on_batch_preview(self, task_id: str, frame: Frame) -> None:
        self._batch_coordinator.on_preview(task_id, frame)

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
        """A Sources-library tile click. It assigns to the selected face(s) ONLY
        when the editor is open AND a map has actually been built (there's
        something to assign to); otherwise it sets the GLOBAL source like
        single-source mode. So with the Face scanner open but nothing scanned
        yet, a click still picks the global source instead of going nowhere."""
        if self._batch_active:
            return  # editing locked during a render
        if self._faces_mode and self._map_available:
            ids = self._face_map_panel.selected_identities()
            if ids:
                self._face_map_ctl.assign_source(ids, path)
                # Assigning IS entering the map workflow → ensure routing is on so
                # the "Use face map" switch reflects the live mode (item 4).
                if not self._use_face_map:
                    self._set_use_face_map(True)
            else:
                self._status_bar.show_message(
                    "Select one or more faces first, then click a source.", 3000
                )
            return
        self._pickers.set_source(path)

    def _on_face_map_reset(self) -> None:
        """Reset → discard the catalog AND scan progress so Analyze starts over.
        Confirmed (it throws away any manual source assignments), but suppressible
        — the user can tick "Don't ask me again" to skip the prompt next time."""
        if self._batch_active:
            return
        if confirm(
            self,
            "face_map_reset",
            "Reset faces",
            "Clear the discovered faces, their source assignments, and saved scan "
            "progress? Analyze will start fresh.",
            suppressible=True,
        ):
            self._face_map_ctl.reset_catalog()

    def _on_library_target_selected(self, path: Path) -> None:
        if self._batch_active:
            return
        self._pickers.set_target(path)

    @property
    def _settings(self) -> "user_settings.Settings":
        # Backed by SettingsBinder. Kept as a window alias so the ~30 reads and
        # the tests that seed `win._settings = Settings()` on a bare __new__
        # window keep working (the setter lazily builds the binder there).
        return self._settings_binder.settings

    @_settings.setter
    def _settings(self, value: "user_settings.Settings") -> None:
        binder = getattr(self, "_settings_binder", None)
        if binder is None:
            self._settings_binder = SettingsBinder(value)
        else:
            binder.set_settings(value)

    def _update_settings(self, **fields: object) -> None:
        # Assign-before-save lives in the binder (a stale in-memory copy after a
        # failed write would corrupt every later model_copy).
        self._settings_binder.update(**fields)

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
