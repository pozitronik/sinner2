from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from sinner2.audio.audio_backend import AudioBackend, AudioBackendName
from sinner2.config.media_extensions import is_video_ext
from sinner2.config.source import Source
from sinner2.config.target import Target
from sinner2.gui.audio_controller import AudioController
from sinner2.gui.session_capabilities import SessionCapabilities
from sinner2.gui.bridges.observable_bridge import ObservableValueBridge
from sinner2.gui.cache_controller import CacheController
from sinner2.gui.session_builder import (
    _DEFAULT_CACHE_SETTINGS,
    CacheSettings,
    SessionBuilder,
    SessionBuildSpec,
    SessionFactory,
    _default_session_factory,
    _SessionBundle,
)
from sinner2.gui.swap_coordinator import SwapCoordinator, _SwapOutcome
from sinner2.gui.sync_tracer import SyncSample, SyncTracer
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor
from sinner2.pipeline.buffer.store import FrameStore
from sinner2.pipeline.cache_manager import CacheManager
from sinner2.pipeline.chain_builder import build_chain
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    FaceEnhancerParams,
)
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams
from sinner2.pipeline.processors.upscaler import UpscalerParams
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    SyncedStrategy,
)


# CodeFormer is a heavy, GPU-bound, SHARED ONNX session: extra realtime workers
# don't add throughput (they serialize on the one GPU session) and only deepen
# the in-flight queue, which adds latency between a seek and the frame showing.
# Cap the EFFECTIVE realtime worker count when it's the active enhancer so the
# preview stays responsive. The user's stored worker count is untouched — this
# only bounds what the executor actually runs with.
_CODEFORMER_REALTIME_WORKER_CAP = 2


class PlayerController(QObject):
    """Owns the realtime executor lifecycle and wires widgets to it.

    Responsibilities:
      - Build / tear down the executor when source+target are both set
      - Bridge executor observables to widget setter slots
      - Forward widget signals (play/pause/seek) to executor commands
      - Surface setup / runtime errors via the errorOccurred signal
      - Clean up scratch directory on shutdown
    """

    errorOccurred = Signal(str)
    processingFpsChanged = Signal(object)  # carries float; declared `object` to match the bridge
    displayFpsChanged = Signal(object)  # carries float — effective shown-frame rate
    framesSkippedChanged = Signal(object)  # carries int — cumulative strategy skips this session
    sessionScratchDirChanged = Signal(object)  # Path | None — emitted on session start/end
    bufferMetricsChanged = Signal(object)  # carries BufferMetrics; routes to status bar
    strategyModeChanged = Signal(object)  # carries str; routes to status bar mode label
    cacheStorageStatsChanged = Signal()  # fired on session start/teardown/clear so the cache panel can refresh
    targetNativeSizeChanged = Signal(object)  # (width, height) on session start, None on teardown
    sessionSwitching = Signal(bool)  # True while an async source/target swap is draining+rebuilding

    def __init__(
        self,
        frame_display: QFrameDisplayWidget,
        transport: QTransportControls,
        session_factory: SessionFactory | None = None,
        audio_backend_factory: Callable[[AudioBackendName], AudioBackend] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._display = frame_display
        self._transport = transport
        self._session_factory = session_factory or _default_session_factory

        self._executor: RealtimeExecutor | None = None
        self._write_executor: BoundedWriteExecutor | None = None
        self._session_store: FrameStore | None = None
        self._session_cache_dir: Path | None = None
        self._bridges: list[ObservableValueBridge] = []
        self._current_target_path: Path | None = None
        self._current_source_path: Path | None = None

        self._current_source: Source | None = None
        self._swapper_params = FaceSwapperParams()
        self._enhancer_params = FaceEnhancerParams()
        self._enhancer_enabled = True
        self._upscaler_params = UpscalerParams()
        self._upscaler_enabled = False
        self._upscaler_device: str = "auto"
        self._swapper_enabled = True
        self._strategy: FrameSkipStrategy = BestEffortStrategy()
        self._worker_count = 1
        # Effective worker count the live executor was last started/set with
        # (may be capped below _worker_count for a heavy enhancer — see
        # _effective_worker_count). Tracked so a config change re-applies it
        # only when the effective value actually moves.
        self._applied_worker_count = 1
        self._playback_mode: PlaybackMode = PlaybackMode.FIXED_30
        self._cache_settings: CacheSettings = _DEFAULT_CACHE_SETTINGS
        # Cache-storage policy (root, per-session subdir, size cap, clear) lives
        # in its own helper; it reports storage changes back through the Qt signal.
        self._cache = CacheController(self.cacheStorageStatsChanged.emit)
        # Session assembly (reader pool + cache dir + store + executor) lives in
        # its own Qt-free builder; the chain + worker count are built here (they
        # are shared with the live hot-swap path) and passed into build().
        self._session_builder = SessionBuilder(self._cache, self._session_factory)
        # Audio playback (backend lifecycle + volume + guarded mirror ops) lives
        # in its own helper; it constructs the backend lazily because some
        # implementations (QtMultimedia) need a QApplication to exist first.
        self._audio = AudioController(audio_backend_factory, self.errorOccurred.emit)
        # Target fps cached on load so seek-by-frame can convert to seconds.
        self._target_fps: float = 0.0
        # Optional A/V sync diagnostic sampler (dormant unless SINNER2_SYNC_TRACE
        # is set). Read-only; started/stopped with playback. See sync_tracer.py.
        self._sync_tracer = SyncTracer(self._sync_sample, parent=self)
        # Video reader backend (applies on next session start).
        self._video_backend: VideoBackend = VideoBackend.FFMPEG
        # Number of parallel readers in the ReaderPool. Default 1 ≈ current
        # single-reader behaviour. Raise for slow sources (network/HDD)
        # with SyncedStrategy. Changes rebuild the session.
        self._reader_pool_size: int = 1
        # Processing scale: downscale frames before the chain for speed.
        # 0 < s <= 1; 1.0 = full resolution. Part of the cache key, so a
        # change rebuilds the session into a distinct cache dir.
        self._processing_scale: float = 1.0
        # Realtime swapper ONNX providers (priority order) + the realtime
        # enhancer torch device. Empty providers → platform-default EP order;
        # "auto" device → CUDA if available else CPU. Both are passed
        # explicitly into the chain at build time — no global provider state.
        self._swapper_providers: tuple[str, ...] = ()
        self._enhancer_device: str = "auto"
        # Optional debug-overlay sink the swapper publishes its pre-swap
        # detections to. Set once at startup, before any session, so every
        # rebuilt chain picks it up via _build_chain.
        self._detection_sink: object | None = None

        # A completed-but-not-yet-adopted swap bundle. Set on the worker thread
        # the moment a swap succeeds and cleared by the GUI slot once adopted;
        # if shutdown() races the queued slot, it's still set and shutdown
        # releases its (new, live) write executor + store to avoid a leak.
        self._last_swap_bundle: _SessionBundle | None = None
        # Desired post-swap position/play state. Held as controller state (not a
        # per-call callback) so coalesced changes during a swap carry forward
        # the latest intent. Applied to the new executor once it's installed.
        self._restore_frame = 0
        self._restore_play = False
        # Async source/target swap. A change on a running session tears down +
        # rebuilds on a worker thread so the slow drain never blocks the GUI;
        # SwapCoordinator owns the coalescing + threading + GUI-hop, and calls
        # back into the controller-coupled work (build/reconfigure/adopt) below.
        self._swap = SwapCoordinator(
            run_job=self._run_swap_job,
            on_complete=self._on_swap_complete,
            on_begin=self._audio.pause_if_loaded,
            on_switching=self.sessionSwitching.emit,
            parent=self,
        )

        # Play/pause/seek are routed by the SessionFacade (so they reach whichever
        # engine owns the active target); the controller keeps the audio-aware
        # play()/pause()/seek_to()/toggle_playback() the facade calls. Volume is
        # file-only audio and stays wired here.
        transport.volumeChanged.connect(self._on_audio_volume_changed)

    def set_source_and_target(self, source_path: Path | None, target_path: Path | None) -> None:
        """Synchronous session (re)build. Used for first-load (no running
        session to drain, so no GUI freeze) and at shutdown. Source/target
        CHANGES on a running session go through the async path (change_source /
        change_target) so the slow teardown doesn't block the UI."""
        if source_path is None or target_path is None:
            return
        self._teardown_session()
        try:
            bundle = self._build_session(source_path, target_path)
        except Exception as exc:
            self.errorOccurred.emit(f"session setup failed: {exc}")
            return
        self._install_session(bundle)

    def _build_session(self, source_path: Path, target_path: Path) -> _SessionBundle:
        """Assemble a session for source+target. The chain + effective worker
        count are built here (they're shared with the live hot-swap path) and
        handed to the Qt-free SessionBuilder, which owns the reader/cache/store/
        executor assembly. Build warnings ride back in the bundle for the GUI
        caller to emit."""
        source = Source(path=source_path)
        target = Target(path=target_path)
        chain = self._build_chain(source)
        effective_workers = self._effective_worker_count()
        self._applied_worker_count = effective_workers
        return self._session_builder.build(
            source, target, source_path, target_path,
            chain, effective_workers, self._build_spec(),
        )

    def _build_spec(self) -> SessionBuildSpec:
        """Snapshot the session-assembly config (reader / cache / executor knobs)
        the builder needs, from the controller's current state."""
        return SessionBuildSpec(
            strategy=self._strategy,
            playback_mode=self._playback_mode,
            cache_settings=self._cache_settings,
            video_backend=self._video_backend,
            reader_pool_size=self._reader_pool_size,
            processing_scale=self._processing_scale,
        )

    def _install_session(self, bundle: _SessionBundle) -> None:
        """Wire a freshly built session into the controller + widgets and start
        it. Qt-touching — MUST run on the GUI thread (it creates observable
        bridges, hooks the display, emits signals, loads audio)."""
        # Surface any warnings the (Qt-free) build collected, now on the GUI thread.
        for warning in bundle.warnings:
            self.errorOccurred.emit(warning)
        executor = bundle.executor
        executor.on_frame_ready(self._display.show_frame)
        self._bind_observables(executor)
        self._current_source = bundle.source
        self._current_source_path = bundle.source_path
        self._current_target_path = bundle.target_path
        self._executor = executor
        self._write_executor = bundle.write_executor
        self._session_store = bundle.session_store
        self._session_cache_dir = bundle.cache_dir
        self._target_fps = bundle.target_fps
        self._transport.set_frame_count(bundle.frame_count)
        self._transport.set_fps(bundle.target_fps)
        self.sessionScratchDirChanged.emit(bundle.cache_dir)
        self.cacheStorageStatsChanged.emit()
        # Native source size for the scale readout ("50% [960x540]").
        self.targetNativeSizeChanged.emit(bundle.native_size)
        # Load the target into the audio backend. Done after the executor is
        # built so a backend-init failure doesn't prevent silent playback.
        backend = self.audio_backend()
        if backend is not None:
            try:
                backend.load(bundle.target_path)
            except Exception as exc:
                self.errorOccurred.emit(f"audio load failed: {exc}")
        try:
            executor.start()
        except Exception as exc:
            self.errorOccurred.emit(f"executor.start failed: {exc}")
            self._teardown_session()

    # ---- async session swap (source/target change on a running session) ----

    def _run_swap_job(self, source_path: Path, target_path: Path) -> _SwapOutcome:
        """Worker-thread half of an async source/target change (run via
        SwapCoordinator). Builds the new session world (the slow reader-probe
        runs here, off the GUI thread) as an UNSTARTED executor, then hands its
        world to the LIVE executor to adopt — keeping the live executor's worker
        threads (and their ORT per-thread CUDA state) alive instead of churning
        them, which is what leaked GPU memory. Shuts the displaced old resources
        down here too; returns the outcome for the coordinator to marshal back to
        the GUI thread."""
        executor = self._executor
        if executor is None:
            # No live executor to adopt into (shouldn't happen — change_* guard
            # against it). Fall back to reporting nothing changed.
            return _SwapOutcome(error="no active session")
        try:
            bundle = self._build_session(source_path, target_path)
        except Exception as exc:  # noqa: BLE001 — surfaced on the GUI thread
            return _SwapOutcome(error=str(exc))
        # Capture the OLD write executor + store (controller-owned) to shut down
        # after the swap; the old reader pool comes back from reconfigure_from.
        old_write_executor = self._write_executor
        old_store = self._session_store
        old = executor.reconfigure_from(
            bundle.executor,
            restore_frame=self._restore_frame,
            play=self._restore_play,
        )
        if old is None:
            # Swap failed (e.g. no face in the new source) — the old world is
            # still live. Discard the freshly built (unstarted) world.
            self._discard_unstarted_bundle(bundle)
            return _SwapOutcome(error="could not switch to the new source/target")
        old_reader_pool, _old_buffer = old
        # Shut the displaced resources down off the GUI thread. The old reader
        # pool's threads + the write executor's threads don't touch ORT, so
        # recreating them is harmless (unlike the worker threads we kept).
        old_reader_pool.shutdown()
        if old_write_executor is not None:
            old_write_executor.shutdown(wait=True)
        if old_store is not None:
            old_store.close()
        # Stash before returning: the live executor now writes through this
        # bundle's write executor + store. If shutdown() joins this thread before
        # the queued GUI slot adopts the bundle, shutdown drains it.
        self._last_swap_bundle = bundle
        return _SwapOutcome(bundle=bundle)

    @staticmethod
    def _discard_unstarted_bundle(bundle: _SessionBundle) -> None:
        """Tear down the resources of a freshly built but NEVER-INSTALLED session
        (the executor was never start()ed, so it owns no threads — only the
        reader pool + write executor + store need releasing)."""
        try:
            bundle.executor.reader_pool.shutdown()
        except Exception:
            pass
        try:
            bundle.write_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            bundle.session_store.close()
        except Exception:
            pass

    def _on_swap_complete(self, outcome: _SwapOutcome) -> None:
        """GUI thread (via SwapCoordinator): a background swap finished. The LIVE
        executor has already adopted the new world (or kept the old one on
        failure); re-point the controller's GUI-facing references at the new
        resources and reload + restore audio. The coordinator owns the
        swapping / pending / switching state and runs any coalesced request."""
        if outcome.error is not None:
            self.errorOccurred.emit(f"session switch failed: {outcome.error}")
            # The old session is still live (the failed swap left it untouched)
            # and, if it was playing, still producing frames — but the swap start
            # paused audio. Re-sync audio to it so the video doesn't play on
            # silently (A/V desync) until the user manually toggles play.
            self._restore_audio_to_live_session()
        elif outcome.bundle is not None:
            self._adopt_swapped_bundle(outcome.bundle)
            # Adopted into our refs — _teardown_session now owns it, so clear
            # the shutdown-drain stash to avoid a double teardown.
            self._last_swap_bundle = None

    def _adopt_swapped_bundle(self, bundle: _SessionBundle) -> None:
        """Re-point controller state + widgets at the new world after the live
        executor adopted it. The executor itself (and its observable bridges)
        is unchanged — bridges stay wired to the same executor observables — so
        this only refreshes the controller-owned references, the transport
        range, the cache panel, the native-size readout, and audio."""
        # Surface build warnings now, on the GUI thread (the build ran on a worker).
        for warning in bundle.warnings:
            self.errorOccurred.emit(warning)
        self._current_source = bundle.source
        self._current_source_path = bundle.source_path
        self._current_target_path = bundle.target_path
        self._write_executor = bundle.write_executor
        self._session_store = bundle.session_store
        self._session_cache_dir = bundle.cache_dir
        self._target_fps = bundle.target_fps
        self._transport.set_frame_count(bundle.frame_count)
        self._transport.set_fps(bundle.target_fps)
        # set_frame_count resets the slider to 0. Restore the position we seeked
        # back to, so a target change while PAUSED doesn't show 0 until the user
        # hits play (the continuous current_frame stream that would otherwise
        # correct it only runs during playback).
        self._transport.set_current_frame(self._restore_frame)
        self.sessionScratchDirChanged.emit(bundle.cache_dir)
        self.cacheStorageStatsChanged.emit()
        self.targetNativeSizeChanged.emit(bundle.native_size)
        # Reload audio for the (possibly new) target, then restore the position
        # + play state so sound resumes with the picture. load() switches media
        # when the target changed; on a source-only swap the target is unchanged
        # so load() no-ops — reload() then forces a fresh setSource to re-arm the
        # deferred play/seek, otherwise the restore's resume is a bare play() on a
        # just-paused player and audio stays silent until a manual stop/restart.
        backend = self.audio_backend()
        if backend is not None:
            try:
                backend.load(bundle.target_path)
                backend.reload()
            except Exception as exc:
                self.errorOccurred.emit(f"audio load failed: {exc}")
        self._restore_audio_state()

    def _restore_audio_state(self) -> None:
        """Re-point the audio backend at the restored position + play state after
        an async session swap. The restore intent (frame / play / target fps) is
        controller-owned session state; the audio helper applies it."""
        self._audio.restore_state(
            self._target_fps, self._restore_frame, self._restore_play
        )

    def apply_session_config(
        self,
        swapper_params: FaceSwapperParams,
        enhancer_params: FaceEnhancerParams,
        enhancer_enabled: bool,
        strategy: FrameSkipStrategy,
        worker_count: int,
        playback_mode: PlaybackMode,
        cache_settings: CacheSettings,
        swapper_enabled: bool = True,
        swapper_providers: tuple[str, ...] = (),
        enhancer_device: str = "auto",
        upscaler_params: UpscalerParams | None = None,
        upscaler_enabled: bool = False,
        upscaler_device: str = "auto",
    ) -> None:
        """Update stored params and propagate any changes to the live session.

        Hot-swap surface: chain (on param / providers / device change),
        strategy, worker_count, playback_mode, cache_mode, memory cache size.
        The rest of cache_settings (format, quality, write workers, write queue
        size) is stored and takes effect at the next session start — switching
        them live would require re-creating the buffer + write executor + store
        directory hash, which is what `set_source_and_target` already does.

        A chain change additionally invalidates the whole frame cache inside the
        executor (the cache is keyed by frame index, not chain), so the new
        chain's output reaches the display on EVERY frame — paused or playing —
        not just the one visible while paused.

        Providers (swapper, ONNX) and device (enhancer, torch) are part of the
        chain: changing either rebuilds the chain so the processors reload on
        the new hardware. A swapper-providers change also drops the shared
        insightface model + ONNX session cache, which are bound to the EP list
        they were built with.
        """
        swapper_providers = tuple(swapper_providers)
        upscaler_params = upscaler_params or UpscalerParams()
        providers_changed = swapper_providers != self._swapper_providers
        # det_size is baked into the shared insightface detector at build time
        # (like providers), so a change needs the singleton dropped + rebuilt.
        detection_size_changed = (
            self._swapper_params is not None
            and swapper_params is not None
            and swapper_params.detection_size != self._swapper_params.detection_size
        )
        chain_changed = (
            swapper_params != self._swapper_params
            or enhancer_params != self._enhancer_params
            or enhancer_enabled != self._enhancer_enabled
            or swapper_enabled != self._swapper_enabled
            or providers_changed
            or enhancer_device != self._enhancer_device
            or upscaler_params != self._upscaler_params
            or upscaler_enabled != self._upscaler_enabled
            or upscaler_device != self._upscaler_device
        )
        strategy_changed = type(strategy) is not type(self._strategy)
        # Synced threshold changes don't change the type, but still need
        # a hot-swap so the executor's strategy reflects the new threshold.
        if (
            not strategy_changed
            and isinstance(strategy, SyncedStrategy)
            and isinstance(self._strategy, SyncedStrategy)
            and strategy.max_lag_frames != self._strategy.max_lag_frames
        ):
            strategy_changed = True
        playback_mode_changed = playback_mode is not self._playback_mode
        cache_mode_changed = cache_settings.mode is not self._cache_settings.mode
        memory_bytes_changed = (
            cache_settings.memory_max_bytes != self._cache_settings.memory_max_bytes
        )

        self._swapper_params = swapper_params
        self._enhancer_params = enhancer_params
        self._enhancer_enabled = enhancer_enabled
        self._swapper_enabled = swapper_enabled
        self._strategy = strategy
        self._worker_count = worker_count
        self._playback_mode = playback_mode
        self._cache_settings = cache_settings
        self._swapper_providers = swapper_providers
        self._enhancer_device = enhancer_device
        self._upscaler_params = upscaler_params
        self._upscaler_enabled = upscaler_enabled
        self._upscaler_device = upscaler_device

        if self._executor is None or self._current_source is None:
            return
        if chain_changed:
            if providers_changed or detection_size_changed:
                # The shared insightface detector was built with the OLD
                # providers + det_size; drop it (and, on a provider change, the
                # ONNX session cache too, since those sessions are bound to the
                # EP list) so the rebuilt chain re-creates them.
                from sinner2.pipeline import face_analyser, model_cache

                if providers_changed:
                    model_cache.clear_session_cache()
                face_analyser.reset_shared_face_analysis()
            try:
                new_chain = self._build_chain(self._current_source)
            except Exception as exc:
                self.errorOccurred.emit(f"chain rebuild failed: {exc}")
                return
            # set_chain invalidates the whole frame cache and re-renders the
            # current frame itself (paused or playing), so no seek nudge here —
            # and crucially it refreshes EVERY cached frame, not just the visible
            # one, so a tweak applies across the clip even with a large cache.
            self._executor.set_chain(new_chain)
        if strategy_changed:
            self._executor.set_skip_strategy(strategy)
        # Re-apply the EFFECTIVE worker count whenever it moves — that's the
        # slider changing OR the enhancer flipping the CodeFormer cap on/off
        # (the latter rides in on chain_changed, not a worker_count change).
        effective_workers = self._effective_worker_count()
        if effective_workers != self._applied_worker_count:
            self._executor.set_worker_count(effective_workers)
            self._applied_worker_count = effective_workers
        if playback_mode_changed:
            self._executor.set_playback_mode(playback_mode)
        if cache_mode_changed:
            self._executor.set_cache_mode(cache_settings.mode)
        if memory_bytes_changed and cache_settings.memory_max_bytes > 0:
            self._executor.set_memory_cache_bytes(cache_settings.memory_max_bytes)

    def _effective_worker_count(self) -> int:
        """Realtime worker count actually used, capped for a heavy GPU-bound
        enhancer (CodeFormer) so the preview stays responsive. Falls back to the
        user's requested count for everything else."""
        if (
            self._enhancer_enabled
            and self._enhancer_params.model is EnhancerModel.CODEFORMER
        ):
            return min(self._worker_count, _CODEFORMER_REALTIME_WORKER_CAP)
        return self._worker_count

    def _build_chain(self, source: Source) -> list[Processor]:
        """Compose the realtime chain from current controller state via the
        shared builder (same logic the live-camera path uses)."""
        return build_chain(
            source,
            swapper_enabled=self._swapper_enabled,
            swapper_params=self._swapper_params,
            swapper_providers=self._swapper_providers,
            detection_sink=self._detection_sink,
            enhancer_enabled=self._enhancer_enabled,
            enhancer_params=self._enhancer_params,
            enhancer_device=self._enhancer_device,
            upscaler_enabled=self._upscaler_enabled,
            upscaler_params=self._upscaler_params,
            upscaler_device=self._upscaler_device,
        )

    def deactivate(self) -> None:
        """Tear down the active session — cancel any in-flight swap, stop the
        executor, close the store — WITHOUT destroying the reusable audio
        backend. Used when the session switches to another target kind (e.g. the
        camera); a later file load rebuilds via set_source_and_target. shutdown()
        is this plus the audio backend teardown."""
        # A swap may be mid-flight on a worker thread; drop any coalesced request
        # + wait for it so we don't tear down while it's still building/stopping.
        self._swap.cancel_pending_and_join(30.0)
        self._teardown_session()
        # If a swap completed but its GUI slot never adopted the bundle (close
        # raced the queued signal), _teardown_session tore down the OLD refs —
        # the new bundle's write executor + store are live but unadopted, so
        # release them here. Done AFTER teardown stops the executor that writes
        # through them. (None in the normal case — the slot cleared it.)
        bundle = self._last_swap_bundle
        self._last_swap_bundle = None
        if bundle is not None:
            try:
                bundle.write_executor.shutdown(wait=True)
            except Exception:
                pass
            try:
                bundle.session_store.close()
            except Exception:
                pass

    def shutdown(self) -> None:
        self._sync_tracer.stop()
        self.deactivate()
        self._audio.shutdown()

    def executor(self) -> RealtimeExecutor | None:
        return self._executor

    def capabilities(self) -> SessionCapabilities:
        """The active file session's capabilities — always seekable + finite;
        audio only when the target is a video. NONE when no session is loaded.
        Reads executor() (not the attribute) so it tracks a test/real swap."""
        if self.executor() is None:
            return SessionCapabilities.none()
        has_audio = (
            self._current_target_path is not None
            and is_video_ext(self._current_target_path)
        )
        return SessionCapabilities.for_file(has_audio=has_audio)

    def session_cache_dir(self) -> Path | None:
        """The active session's cache dir (None if no session / cache off) —
        exposed so the GUI doesn't reach the private attribute."""
        return self._session_cache_dir

    def set_detection_sink(self, sink: object | None) -> None:
        """Set the sink the swapper publishes pre-swap detections to. Call
        before any session starts; the chain reads it at build time."""
        self._detection_sink = sink

    def resync_transport(self) -> None:
        """Re-point the position bar at the live session.

        Used after something external (a batch render) has driven the
        transport's slider to follow its own progress: restore the range +
        playhead to the current session so the scrubber matches reality
        again. No session → reset to an empty range.
        """
        if self._executor is None:
            self._transport.set_frame_count(0)
            return
        self._transport.set_frame_count(self._executor.frame_count())
        self._transport.set_current_frame(
            max(0, self._executor.current_frame.get())
        )

    # ---- Cache management ----

    def cache_root(self) -> Path:
        return self._cache.cache_root()

    def cache_manager(self) -> CacheManager:
        return self._cache.cache_manager()

    def set_cache_root(self, path: Path | None) -> None:
        """Switch the cache root (None reverts to the default). Delegated to the
        cache helper, which fires cacheStorageStatsChanged on an actual change."""
        self._cache.set_cache_root(path)

    def cache_size_cap_bytes(self) -> int:
        return self._cache.cache_size_cap_bytes()

    def set_cache_size_cap_bytes(self, max_bytes: int) -> None:
        self._cache.set_cache_size_cap_bytes(max_bytes)

    def invalidate_current_session(self) -> None:
        """Clear the active session's cached frames so they reprocess.

        Pauses, drops the on-disk dir (and memory cache), and resumes if
        it was playing. The chain stays loaded — only the rendered frames
        are discarded.
        """
        if self._executor is None or self._session_cache_dir is None:
            return
        was_playing = self._executor.is_playing.get()
        self._executor.pause()
        # invalidate_from(0) clears everything in cache + on-disk store from
        # frame 0 upward — exactly the "drop all rendered frames" semantic.
        try:
            self._executor.invalidate_from(0)
        except Exception as exc:
            self.errorOccurred.emit(f"cache invalidate failed: {exc}")
        if was_playing:
            self._executor.play()
        self.cacheStorageStatsChanged.emit()

    def rerender_from_current(self) -> None:
        """Reprocess from the playhead forward through the current chain — the
        retroactive 'apply the new params to frames I've already passed'
        action. Frames before the playhead keep their cached pixels."""
        if self._executor is not None:
            self._executor.rerender_from_current()

    def clear_all_caches(self) -> tuple[int, int]:
        """Wipe every cache entry under the current root. Spares the
        currently-active session's directory. Returns (entries_deleted,
        bytes_freed) for the UI to display."""
        protect: list[Path] = []
        if self._session_cache_dir is not None:
            protect.append(self._session_cache_dir)
        return self._cache.clear_all(protect)

    def _bind_observables(self, executor: RealtimeExecutor) -> None:
        current_bridge = ObservableValueBridge(executor.current_frame, self)
        current_bridge.valueChanged.connect(self._transport.set_current_frame)
        playing_bridge = ObservableValueBridge(executor.is_playing, self)
        playing_bridge.valueChanged.connect(self._transport.set_is_playing)
        status_bridge = ObservableValueBridge(executor.status, self)
        status_bridge.valueChanged.connect(self._on_status)
        fps_bridge = ObservableValueBridge(executor.processing_fps, self)
        fps_bridge.valueChanged.connect(self.processingFpsChanged)
        display_fps_bridge = ObservableValueBridge(executor.display_fps, self)
        display_fps_bridge.valueChanged.connect(self.displayFpsChanged)
        metrics_bridge = ObservableValueBridge(executor.metrics, self)
        metrics_bridge.valueChanged.connect(self.bufferMetricsChanged)
        mode_bridge = ObservableValueBridge(executor.strategy_mode, self)
        mode_bridge.valueChanged.connect(self.strategyModeChanged)
        skipped_bridge = ObservableValueBridge(executor.frames_skipped, self)
        skipped_bridge.valueChanged.connect(self.framesSkippedChanged)
        self._bridges = [
            current_bridge,
            playing_bridge,
            status_bridge,
            fps_bridge,
            display_fps_bridge,
            metrics_bridge,
            mode_bridge,
            skipped_bridge,
        ]

    def _on_status(self, message: object) -> None:
        text = str(message)
        if text and text.lower().startswith(
            ("worker error", "executor.start", "session setup", "chain setup")
        ):
            self.errorOccurred.emit(text)

    def _teardown_session(self) -> None:
        for bridge in self._bridges:
            bridge.shutdown()
        self._bridges = []
        # Stop audio before tearing down the executor so the user doesn't
        # hear audio continuing while the frame view freezes.
        self._audio.pause_if_loaded()
        if self._executor is not None:
            self._executor.stop()
            self._executor = None
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)
            self._write_executor = None
        if self._session_store is not None:
            self._session_store.close()
            self._session_store = None
            self._session_cache_dir = None
            self.sessionScratchDirChanged.emit(None)
            self.targetNativeSizeChanged.emit(None)
        self._current_source = None
        self._current_source_path = None
        self._current_target_path = None
        self.cacheStorageStatsChanged.emit()

    def _sync_sample(self) -> SyncSample | None:
        """Read-only snapshot of the playback clocks for the sync tracer.
        None when no session is active. Uses the already-built audio backend
        (never constructs one just to sample)."""
        ex = self._executor
        if ex is None:
            return None
        backend = self._audio.backend
        audio_s = backend.audio_position_seconds() if backend is not None else -1.0
        frame = max(0, ex.current_frame.get())
        fps = self._target_fps
        return SyncSample(
            frame=frame,
            video_seconds=frame / fps if fps > 0 else 0.0,
            audio_seconds=audio_s,
            playing=ex.is_playing.get(),
            strategy_mode=ex.strategy_mode.get(),
        )

    def _on_play(self) -> None:
        if self._executor is not None:
            self._executor.play()
        self._audio.play_if_loaded()
        self._sync_tracer.start()

    def _on_pause(self) -> None:
        if self._executor is not None:
            self._executor.pause()
        self._audio.pause_if_loaded()
        self._sync_tracer.stop()

    def play(self) -> None:
        """Public play — symmetric with pause(); the session facade calls it for
        the transport's play request. No-op when no session is loaded."""
        self._on_play()

    def pause(self) -> None:
        """Public pause — used when live-camera mode takes over the display so
        the file session stops driving it. No-op when nothing is playing."""
        self._on_pause()

    def toggle_playback(self) -> None:
        """Audio-aware play/pause toggle. The spacebar shortcut routes through
        here (not executor.play/pause directly) so audio stays in lock-step with
        the video exactly like the transport button — otherwise pausing left the
        audio playing whenever the button didn't have keyboard focus."""
        if self._executor is None:
            return
        if self._executor.is_playing.get():
            self._on_pause()
        else:
            self._on_play()

    def seek_to(self, frame: int) -> None:
        """Audio-aware seek. The arrow / Home / End shortcuts route through here
        so the audio backend follows the seek instead of desyncing."""
        self._on_seek(frame)

    def _on_seek(self, frame: int) -> None:
        if self._executor is not None:
            self._executor.seek(frame)
        if self._target_fps > 0:
            self._audio.seek_if_loaded(frame / self._target_fps)

    # ---- Audio ----

    def audio_backend(self) -> AudioBackend | None:
        """Lazy accessor — backend is constructed on first request so the
        QApplication exists by then. Returns None if construction failed."""
        return self._audio.ensure_backend()

    def set_audio_backend(self, name: AudioBackendName) -> None:
        # Swap the backend; only when it actually changed, reload the current
        # media + restore seek/play from the executor. Without that, switching
        # mid-session leaves the new backend with no media loaded so play/pause/
        # seek all no-op and audio stays silent for the rest of the session.
        if self._audio.switch_backend(name):
            self._reload_audio_into_backend()

    def _reload_audio_into_backend(self) -> None:
        """Reload the current target into the active audio backend and restore
        seek+play from the live executor. No-op when no backend, no target, or
        no live session — used after the backend is (re)constructed mid-session."""
        if self._audio.backend is None or self._current_target_path is None:
            return
        self._audio.load(self._current_target_path)
        self._restore_audio_to_live_session()

    def _restore_audio_to_live_session(self) -> None:
        """Re-sync the audio backend to the currently-live executor's play/seek
        state (no reload). Used when audio was paused for an operation but the
        old session stays live — e.g. a swap that failed — so the backend
        doesn't sit silent against still-advancing video."""
        if self._executor is not None:
            self._restore_frame = max(0, self._executor.current_frame.get())
            self._restore_play = self._executor.is_playing.get()
        self._restore_audio_state()

    def _on_audio_volume_changed(self, value: int) -> None:
        self._audio.set_volume(value)

    def _rebuild_current_session_async(self) -> None:
        """Re-point the running session at its CURRENT source+target through the
        in-place reconfigure path, preserving frame + play state.

        Used by the structural settings (video backend, reader-pool size,
        processing scale) that each need a fresh reader pool / cache dir but must
        NOT tear the executor down — recreating the worker threads leaks GPU
        memory (see RealtimeExecutor.reconfigure_from). No-op when no session is
        active. The reconfigure path restores position/play (and audio) once the
        new world is live, so callers just set their field and call this."""
        if (
            self._executor is None
            or self._current_source_path is None
            or self._current_target_path is None
        ):
            return
        self._restore_frame = max(0, self._executor.current_frame.get())
        self._restore_play = self._executor.is_playing.get()
        self._swap.request(
            self._current_source_path, self._current_target_path
        )

    def video_backend(self) -> VideoBackend:
        return self._video_backend

    def set_video_backend(self, backend: VideoBackend) -> None:
        """Switch the video reader backend.

        If a session is running, rebuild it in place so the new backend takes
        effect immediately while keeping the worker pool alive. The current
        frame and play state are preserved across the rebuild."""
        if backend is self._video_backend:
            return
        self._video_backend = backend
        self._rebuild_current_session_async()

    def reader_pool_size(self) -> int:
        return self._reader_pool_size

    def set_reader_pool_size(self, n: int) -> None:
        """Change the parallel reader pool size.

        Pool size is structural — the pool can't be resized after construction
        without disrupting in-flight reads, so a change rebuilds the session in
        place (same pattern as set_video_backend). Current frame and play state
        are preserved across the rebuild.
        """
        clamped = max(1, min(16, n))
        if clamped == self._reader_pool_size:
            return
        self._reader_pool_size = clamped
        self._rebuild_current_session_async()

    def processing_scale(self) -> float:
        return self._processing_scale

    def set_processing_scale(self, scale: float) -> None:
        """Change the processing downscale (0 < s <= 1).

        Scale is part of the cache key + reader construction, so a change
        rebuilds the session in place (same pattern as set_reader_pool_size).
        Current frame and play state are preserved across the rebuild.
        """
        clamped = max(0.01, min(1.0, scale))
        if clamped == self._processing_scale:
            return
        self._processing_scale = clamped
        self._rebuild_current_session_async()

    def target_fps(self) -> float:
        """The active target's native frame rate (0.0 when no session) — used
        for the status-bar resolution panel and the transport's time readout."""
        return self._target_fps

    def applied_worker_count(self) -> int:
        """The realtime worker count actually in effect (after clamping) for the
        current session — surfaced in the status bar."""
        return self._applied_worker_count

    def swapper_providers(self) -> tuple[str, ...]:
        return self._swapper_providers

    def effective_onnx_providers(self) -> tuple[str, ...]:
        """Whatever ORT will actually use right now.

        Prefers the most recent session's `get_providers()` (recorded
        by processors when their ONNX session loads) over the user's
        request — because ORT silently falls back when a requested
        provider can't initialise (missing runtime libs, GPU absent),
        and the GUI should show the truth. Falls back to the requested
        swapper providers (or the platform default) when no session has
        loaded yet (pre-startup state).
        """
        from sinner2.pipeline import model_cache

        actual = model_cache.get_actual_providers()
        if actual:
            return actual
        # No session loaded yet: report exactly what the user requested — an
        # empty tuple means "no providers selected", not a hidden GPU default.
        return self._swapper_providers

    def change_source(self, source_path: Path) -> None:
        """Replace the source while preserving frame position + play state.

        The chain holds a reference to the source, so a source swap requires a
        full session rebuild. That teardown can block on uninterruptible
        in-flight inference (e.g. CodeFormer), so it runs ASYNCHRONOUSLY off the
        GUI thread; we capture frame + play state now and re-apply them (seek +
        resume) once the new session is installed. No-op if no session is active
        or no target is loaded yet — first-load is set_source_and_target's job.
        """
        if self._current_target_path is None:
            return
        if self._executor is None and not self._swap.swapping:
            return  # no active or in-flight session
        if self._executor is not None:
            # Capture position + play state from the live session. Mid-swap
            # (executor detached) we carry forward the last-captured intent.
            self._restore_frame = max(0, self._executor.current_frame.get())
            self._restore_play = self._executor.is_playing.get()
        self._current_source_path = source_path
        self._swap.request(source_path, self._current_target_path)

    def change_target(self, target_path: Path) -> None:
        """Replace the target. Position resets to frame 0 and the first frame is
        submitted for processing immediately so the display reflects the new
        target. Play state is preserved. Runs asynchronously (see change_source)
        so the teardown never freezes the UI."""
        if self._current_source_path is None:
            return
        if self._executor is None and not self._swap.swapping:
            return  # no active or in-flight session
        if self._executor is not None:
            self._restore_play = self._executor.is_playing.get()
        self._restore_frame = 0  # new timeline → start at frame 0
        self._current_target_path = target_path
        self._swap.request(self._current_source_path, target_path)

    def apply_initial_audio_state(self, volume: int) -> None:
        """Push persisted audio volume into the audio helper without re-emitting
        transport signals. Called once on startup before any session loads; does
        NOT construct the backend (none exists yet on first launch)."""
        self._audio.cache_initial_volume(volume)
