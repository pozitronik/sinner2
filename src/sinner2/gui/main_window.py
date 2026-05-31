from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sinner2.batch.queue import BatchQueue
from sinner2.batch.task import (
    DEFAULT_ENHANCER_WORKERS,
    DEFAULT_SWAPPER_WORKERS,
    BatchCleanupMode,
    BatchOutputFormat,
    BatchProgress,
    BatchTask,
)
from sinner2.batch.task_store import BatchTaskStore
from sinner2.config import settings as user_settings
from sinner2.config.execution import OnnxExecution, TorchExecution
from sinner2.gui.face_detection_probe import FaceDetectionProbe
from sinner2.gui.player_controller import PlayerController, default_cache_root
from sinner2.gui.widgets.batch_task_dialog import QBatchTaskDialog
from sinner2.gui.widgets.batch_view import QBatchView
from sinner2.gui.widgets.face_detection_overlay import QFaceDetectionOverlay
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
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
        self._settings = user_settings.load()
        # True while a batch task renders — locks the live-editing surface so
        # the display acts purely as a render preview (DaVinci-style).
        self._batch_active = False
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
        self._side_panel = QSidePanel(
            thumbnail_cache_dir=default_cache_root() / "thumbnails",
            processors=self._processors,
            batch_view=self._batch_view,
            thumb_display_dim=self._settings.library_display_dim or 128,
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
        self._last_probe_feed = 0.0
        self._face_overlay = QFaceDetectionOverlay(parent=self._display)
        self._display.set_face_overlay(self._face_overlay)
        self._detection_probe = FaceDetectionProbe(
            providers=self._settings.swapper_providers
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
        self._status_bar.face_button.toggled.connect(self._set_face_overlay_visible)
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
        self._controller.errorOccurred.connect(self._show_error)
        self._controller.processingFpsChanged.connect(self._update_fps_label)
        self._controller.sessionScratchDirChanged.connect(self._update_scratch_label)
        self._controller.targetNativeSizeChanged.connect(
            self._processors.set_target_native_size
        )
        self._controller.bufferMetricsChanged.connect(self._update_metrics_label)
        self._controller.strategyModeChanged.connect(self._update_strategy_mode_label)

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
        # Thumbnail size is shared between source and target libraries
        # so resizing one updates the other. Persist + mirror via the
        # side panel's helper to avoid a feedback loop (set_display_dim
        # short-circuits when the value hasn't changed).
        self._side_panel.sources_library().displayDimChanged.connect(
            self._on_library_display_dim_changed
        )
        self._side_panel.targets_library().displayDimChanged.connect(
            self._on_library_display_dim_changed
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
        self._restore_stays_on_top()
        self._restore_rotation()
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
        """Source picker fired. First-load → set_source_and_target;
        subsequent changes → change_source (preserves frame + play state).
        Both paths must be present for any action."""
        target_path = self._pickers.target_path()
        if target_path is None:
            return
        if self._controller.executor() is None:
            self._controller.set_source_and_target(source_path, target_path)
        else:
            self._controller.change_source(source_path)

    def _on_target_changed(self, target_path: Path) -> None:
        """Target picker fired. First-load → set_source_and_target;
        subsequent changes → change_target (resets to frame 0 but
        keeps play state)."""
        source_path = self._pickers.source_path()
        if source_path is None:
            return
        if self._controller.executor() is None:
            self._controller.set_source_and_target(source_path, target_path)
        else:
            self._controller.change_target(target_path)

    def _update_fps_label(self, fps: float) -> None:
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
        confirmed = QMessageBox.question(
            self,
            "Invalidate current session",
            "Drop all cached frames for this session and reprocess from scratch?",
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        self._controller.invalidate_current_session()

    def _on_rerender_from_current(self) -> None:
        # No confirmation: it only reprocesses from the playhead forward and is
        # the natural "apply my param change retroactively" gesture.
        self._controller.rerender_from_current()

    def _on_clear_all_caches(self) -> None:
        manager = self._controller.cache_manager()
        entries = manager.list_entries()
        protected = (
            self._controller._session_cache_dir  # noqa: SLF001 — UX text needs it
        )
        deletable = [e for e in entries if e.path != protected]
        if not deletable:
            QMessageBox.information(
                self,
                "Clear all caches",
                "Nothing to delete — only the current session's cache is present.",
            )
            return
        total = sum(e.size_bytes for e in deletable)
        confirmed = QMessageBox.question(
            self,
            "Clear all caches",
            f"Delete {len(deletable)} cache entries ({_fmt_size(total)})?\n"
            "The currently active session will be spared.",
        )
        if confirmed != QMessageBox.StandardButton.Yes:
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
        # The transport (play / seek / volume + add-to-batch) is usable only
        # once both a source and target are loaded, and never while a batch
        # render holds the editing surface.
        ready = (
            self._pickers.source_path() is not None
            and self._pickers.target_path() is not None
        )
        self._transport.setEnabled(ready and not self._batch_active)

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
        from sinner2.gui.player_controller import CacheSettings

        self._controller.apply_session_config(
            swapper_params=self._processors.swapper_params(),
            enhancer_params=self._processors.enhancer_params(),
            enhancer_enabled=self._processors.enhancer_enabled(),
            swapper_enabled=self._processors.swapper_enabled(),
            strategy=self._processors.skip_strategy(),
            worker_count=self._processors.realtime_workers(),
            playback_mode=self._processors.playback_mode(),
            cache_settings=CacheSettings(
                mode=self._processors.cache_mode(),
                image_format=self._processors.image_format(),
                image_quality=self._processors.image_quality(),
                memory_max_bytes=self._processors.memory_cache_mb() * 1024 * 1024,
                write_workers=self._processors.write_workers(),
                write_queue_size=self._processors.write_queue_size(),
            ),
            swapper_providers=tuple(self._processors.swapper_providers()),
            enhancer_device=self._processors.enhancer_device(),
        )
        # Video backend isn't part of the session-config bundle because
        # it's used by set_source_and_target rather than the executor;
        # push it directly to the controller so the next session picks
        # up the user's selection.
        self._controller.set_video_backend(self._processors.video_backend())
        # Reader pool size triggers a session rebuild via its own setter
        # (same pattern as video_backend).
        self._controller.set_reader_pool_size(self._processors.reader_pool_size())
        # Processing scale also rebuilds the session via its own setter (it's
        # part of the reader construction + cache key, not the live chain).
        self._controller.set_processing_scale(self._processors.processing_scale())
        # Swapper-provider / enhancer-device rebuilds are folded into
        # apply_session_config above; just refresh the status-bar EP label
        # and the failed-provider highlight afterwards.
        self._refresh_providers_label()
        self._highlight_failed_providers()

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
        if key == Qt.Key.Key_F12:
            self._status_bar.on_top_button.toggle()
            return
        if key == Qt.Key.Key_F8:
            self._status_bar.face_button.toggle()
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
        executor = self._controller.executor()
        if executor is None:
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_Space:
            if executor.is_playing.get():
                executor.pause()
            else:
                executor.play()
            return
        if key == Qt.Key.Key_Left:
            executor.seek(max(0, executor.current_frame.get() - 1))
            return
        if key == Qt.Key.Key_Right:
            executor.seek(executor.current_frame.get() + 1)
            return
        if key == Qt.Key.Key_Home:
            executor.seek(0)
            return
        if key == Qt.Key.Key_End:
            executor.seek(max(0, executor.frame_count() - 1))
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
        # exit_fullscreen can restore exactly what was showing.
        chrome: list[QWidget] = [
            self._side_panel,
            self._transport,
            self._pickers,
            self._status_bar,
        ]
        self._pre_fullscreen_visibility = {w: w.isVisible() for w in chrome}
        # Capture maximized state BEFORE showFullScreen() clears it, so exit
        # can return to maximized rather than a smaller restored geometry.
        self._pre_fullscreen_maximized = self.isMaximized()
        for w in chrome:
            w.setVisible(False)
        self._is_fullscreen = True
        self.showFullScreen()

    def _exit_fullscreen(self) -> None:
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

    def _set_stats_visible(self, on: bool) -> None:
        self._metrics_overlay.setVisible(on)
        if on:
            self._reposition_metrics_overlay()
        self._update_settings(metrics_overlay_visible=on)

    def _restore_metrics_overlay_state(self) -> None:
        visible = bool(self._settings.metrics_overlay_visible)
        if visible:
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

    # ---- Face-detection overlay ----

    _PROBE_INTERVAL_S = 0.15  # ~6 Hz: enough to track, cheap enough to stay smooth

    def _apply_face_overlay_visible(self, on: bool) -> None:
        self._face_overlay_on = on
        if on:
            self._face_overlay.setGeometry(self._display.rect())
            self._face_overlay.show()
        else:
            self._face_overlay.hide()
            self._face_overlay.clear()

    def _set_face_overlay_visible(self, on: bool) -> None:
        """Toggle handler for the face button (and F8). Applies + persists."""
        self._apply_face_overlay_visible(on)
        if on:
            self._status_bar.show_message(
                "Face-detection overlay on (F8) — disable the swapper to "
                "inspect target faces", 4000
            )
        self._update_settings(face_overlay_visible=on)

    def _restore_face_overlay_state(self) -> None:
        visible = bool(self._settings.face_overlay_visible)
        self._apply_face_overlay_visible(visible)
        self._set_button_checked(self._status_bar.face_button, visible)

    def _feed_detection_probe(self, frame: Frame) -> None:
        # Tap each displayed frame; forward a throttled, decoupled copy to the
        # probe thread. Zero cost when the overlay is off.
        if not self._face_overlay_on:
            return
        import time as _time

        now = _time.monotonic()
        if now - self._last_probe_feed < self._PROBE_INTERVAL_S:
            return
        self._last_probe_feed = now
        h, w = frame.shape[:2]
        # Copy so the producer can't mutate the buffer under the probe thread.
        self._requestDetection.emit(frame.copy(), w, h)

    def _on_detections(self, detections: object, width: int, height: int) -> None:
        if self._face_overlay_on:
            self._face_overlay.set_detections(detections, width, height)  # type: ignore[arg-type]

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._persist_geometry_to_settings()
        # Stop the batch queue FIRST so its runner thread joins
        # before the controller / side panel start tearing down
        # shared resources (models, etc.).
        self._batch_queue.stop()
        self._controller.shutdown()
        # Stop the detection probe thread (debug overlay) so it doesn't
        # outlive Qt during shutdown.
        self._detection_thread.quit()
        self._detection_thread.wait(2000)
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
            swapper_detection_interval=self._settings.swapper_detection_interval,
            swapper_many_faces=self._settings.swapper_many_faces,
            swapper_target_sex=self._settings.swapper_target_sex,
            enhancer_upscale=self._settings.enhancer_upscale,
            enhancer_only_center_face=self._settings.enhancer_only_center_face,
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
        )

    def _persist_processor_settings(self) -> None:
        swapper = self._processors.swapper_params()
        enhancer = self._processors.enhancer_params()
        self._update_settings(
            realtime_workers=self._processors.realtime_workers(),
            strategy_name=self._processors.strategy_name(),
            enhancer_enabled=self._processors.enhancer_enabled(),
            swapper_enabled=self._processors.swapper_enabled(),
            swapper_detection_interval=swapper.detection_interval,
            swapper_many_faces=swapper.many_faces,
            # str-Enum .value is the single-letter token, kept stable
            # for settings round-trip and sinner1 compatibility.
            swapper_target_sex=swapper.target_sex.value,
            enhancer_upscale=enhancer.upscale,
            enhancer_only_center_face=enhancer.only_center_face,
            playback_mode=self._processors.playback_mode(),
            cache_mode=self._processors.cache_mode(),
            image_format=self._processors.image_format(),
            image_quality=self._processors.image_quality(),
            memory_cache_mb=self._processors.memory_cache_mb(),
            write_workers=self._processors.write_workers(),
            write_queue_size=self._processors.write_queue_size(),
            video_backend=self._processors.video_backend(),
            reader_pool_size=self._processors.reader_pool_size(),
            processing_scale=self._processors.processing_scale(),
            synced_max_lag_frames=self._processors.synced_max_lag_frames(),
            swapper_providers=self._processors.swapper_providers(),
            enhancer_device=self._processors.enhancer_device(),
        )

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
        if self._settings.source_path:
            p = Path(self._settings.source_path)
            if p.is_file():
                self._pickers.set_source(p)
        if self._settings.target_path:
            p = Path(self._settings.target_path)
            if p.is_file():
                self._pickers.set_target(p)
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

    def _on_library_display_dim_changed(self, dim: int) -> None:
        # Mirror to both libraries so they stay in sync, then persist.
        # set_display_dim no-ops when the value is unchanged, so the
        # sister widget's emit is suppressed and we don't loop.
        self._side_panel.set_display_dim(dim)
        self._update_settings(library_display_dim=int(dim))

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
        swapper = self._processors.swapper_params()
        enhancer = self._processors.enhancer_params()
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
        # Per-processor execution profiles. Carry the current ONNX providers
        # into the swapper profile (CPU vs GPU is a meaningful captured choice);
        # workers default to the batch throughput defaults rather than the
        # realtime pool size (live latency tuning ≠ batch throughput tuning).
        providers = self._processors.swapper_providers()
        swapper_execution = (
            OnnxExecution(workers=DEFAULT_SWAPPER_WORKERS, providers=list(providers))
            if providers
            else OnnxExecution(workers=DEFAULT_SWAPPER_WORKERS)
        )
        enhancer_execution = TorchExecution(
            workers=DEFAULT_ENHANCER_WORKERS,
            device=self._processors.enhancer_device(),
        )
        task = BatchTask(
            source_path=source,
            target_path=target,
            output_format=default_format,
            cleanup_mode=default_cleanup,
            swapper_enabled=self._processors.swapper_enabled(),
            swapper_detection_interval=swapper.detection_interval,
            swapper_many_faces=swapper.many_faces,
            swapper_target_sex=swapper.target_sex.value,
            enhancer_enabled=self._processors.enhancer_enabled(),
            enhancer_upscale=enhancer.upscale,
            enhancer_only_center_face=enhancer.only_center_face,
            swapper_execution=swapper_execution,
            enhancer_execution=enhancer_execution,
            video_backend=self._processors.video_backend(),
            reader_pool_size=self._processors.reader_pool_size(),
            processing_scale=self._processors.processing_scale(),
            image_format=self._processors.image_format(),
            image_quality=self._processors.image_quality(),
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

    def _on_batch_task_failed(self, _task_id: str, message: str) -> None:
        # Failures are otherwise quiet (Status cell + its hover tooltip); a
        # status-bar notice makes sure the user notices something stopped.
        self._status_bar.show_message(f"Batch task failed: {message}", 12000)

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
        try:
            updated = self._settings.model_copy(update=fields)
            user_settings.save(updated)
            self._settings = updated
        except Exception:
            pass

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
