import hashlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from sinner2.audio.audio_backend import (
    AudioBackend,
    AudioBackendName,
    build_audio_backend,
)
from sinner2.config.source import Source
from sinner2.config.target import Target, TargetKind
from sinner2.gui.bridges.observable_bridge import ObservableValueBridge
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.io.reader_pool import ReaderPool
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.io.video_backend import VideoBackend, build_video_target_reader
from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import FrameStore, PersistentFrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.cache_manager import CacheManager, make_meta
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import (
    ImageFormat,
    ImageWriter,
    build_image_writer,
)
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    FaceEnhancer,
    FaceEnhancerParams,
)
from sinner2.pipeline.processors.face_swapper import FaceSwapper, FaceSwapperParams
from sinner2.pipeline.processors.upscaler import Upscaler, UpscalerParams
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.realtime.per_worker import PerWorkerProcessor
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    SyncedStrategy,
)


@dataclass(frozen=True)
class CacheSettings:
    """Bundle of cache-and-writer knobs the session factory needs.

    Grouped so the factory signature stays readable as the cache surface
    grows: format, quality, in-memory cap, mode, and the disk-write
    executor sizing all flow together.
    """

    mode: CacheMode
    image_format: ImageFormat
    image_quality: int
    memory_max_bytes: int
    write_workers: int
    write_queue_size: int


SessionFactory = Callable[
    [
        ReaderPool,
        list[Processor],
        FrameSkipStrategy,
        int,
        PlaybackMode,
        CacheSettings,
        FrameStore,
    ],
    tuple[RealtimeExecutor, BoundedWriteExecutor],
]


_DEFAULT_CACHE_SETTINGS = CacheSettings(
    mode=CacheMode.WRITE_READ,
    image_format=ImageFormat.JPEG,
    image_quality=95,
    memory_max_bytes=128 * 1024 * 1024,
    write_workers=4,
    write_queue_size=8,
)

# CodeFormer is a heavy, GPU-bound, SHARED ONNX session: extra realtime workers
# don't add throughput (they serialize on the one GPU session) and only deepen
# the in-flight queue, which adds latency between a seek and the frame showing.
# Cap the EFFECTIVE realtime worker count when it's the active enhancer so the
# preview stays responsive. The user's stored worker count is untouched — this
# only bounds what the executor actually runs with.
_CODEFORMER_REALTIME_WORKER_CAP = 2


def default_cache_root() -> Path:
    """Persistent processed-frame cache root used when the user has not
    set a custom path.

    `SINNER2_CACHE_DIR` env var overrides; defaults to `<install>/temp/`.
    Exposed (not `_` prefixed) so the GUI can show the default in tooltips
    and as the file-dialog start path.
    """
    env = os.environ.get("SINNER2_CACHE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "temp"


def _cache_key(
    source: Source,
    target: Target,
    chain: list[Processor],
    writer: ImageWriter,
    scale: float = 1.0,
) -> str:
    """Stable hash of (source path, target path, chain config, writer settings,
    processing scale).

    Two sessions with identical inputs land in the same cache subdirectory
    so processed frames carry over between runs. Different chain params,
    different image format, different quality, or a different processing
    scale go to a different subdirectory — keeps stale frames from a
    different configuration out of view and lets the user toggle formats or
    downscale without colliding with the full-resolution cache.
    """
    parts: list[str] = [
        str(source.path.resolve()),
        str(target.path.resolve()),
        writer.cache_key,
        f"scale={scale:.4f}",
    ]
    for p in chain:
        parts.append(p.name)
        params = getattr(p, "_params", None)
        if params is not None and hasattr(params, "model_dump_json"):
            parts.append(params.model_dump_json())
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _make_reader(
    target: Target, video_backend: VideoBackend, scale: float = 1.0
) -> TargetReader:
    if target.kind == TargetKind.IMAGE:
        return ImageTargetReader(target, scale)
    if target.kind == TargetKind.VIDEO:
        return build_video_target_reader(target, video_backend, scale)
    raise ValueError(f"unsupported target kind: {target.kind}")


def _make_reader_factory(
    target: Target, video_backend: VideoBackend, scale: float = 1.0
) -> Callable[[], TargetReader]:
    """Thunk that builds a fresh reader on each call.

    Used by ReaderPool to construct its N reader instances. Per-backend
    logic stays in _make_reader; this just curries the arguments."""
    return lambda: _make_reader(target, video_backend, scale)


def _default_session_factory(
    reader_pool: ReaderPool,
    chain: list[Processor],
    strategy: FrameSkipStrategy,
    worker_count: int,
    playback_mode: PlaybackMode,
    cache_settings: CacheSettings,
    store: FrameStore,
) -> tuple[RealtimeExecutor, BoundedWriteExecutor]:
    """Build a realtime executor around reader pool + chain + strategy + store.

    Caller owns the store lifecycle. Caller takes ownership of (executor,
    write_executor) and is responsible for stop() → shutdown(wait=True)
    in that order. The chain is shared by all workers — ORT sessions handle
    concurrent inference, so this is the fast path. The write_executor is
    bounded; FrameBuffer.put silently drops writes when its outstanding
    cap is hit so the queue can't pin memory unbounded on slow disks. The
    reader pool's shutdown is owned by the executor (called from stop()).
    """
    timeline = Timeline(fps=reader_pool.fps)
    cache = MemoryFrameCache(max_bytes=cache_settings.memory_max_bytes)
    write_executor = BoundedWriteExecutor(
        max_workers=cache_settings.write_workers,
        max_outstanding=cache_settings.write_queue_size,
    )
    buffer = FrameBuffer(
        store,
        cache,
        timeline,
        write_executor,
        cache_mode=cache_settings.mode,
    )
    executor = RealtimeExecutor(
        reader_pool=reader_pool,
        buffer=buffer,
        timeline=timeline,
        chain=chain,
        strategy=strategy,
        worker_count=worker_count,
        playback_mode=playback_mode,
    )
    return executor, write_executor


def _spawn_daemon(fn: Callable[[], None]) -> threading.Thread:
    """Default swap runner: fire the job on a daemon thread. Swapped out in
    tests for an inline runner so the async flow executes deterministically."""
    thread = threading.Thread(target=fn, name="sinner2-session-swap", daemon=True)
    thread.start()
    return thread


@dataclass
class _SessionBundle:
    """A fully built (but not yet installed) session — the non-Qt product of
    _build_session, ready to be handed to _install_session on the GUI thread."""

    executor: RealtimeExecutor
    write_executor: BoundedWriteExecutor
    session_store: FrameStore
    cache_dir: Path
    source: Source
    source_path: Path
    target_path: Path
    target_fps: float
    frame_count: int
    native_size: tuple[int, int]


@dataclass
class _SwapOutcome:
    """Result of a background session swap, marshaled back to the GUI thread.
    Exactly one of bundle/error is set."""

    bundle: _SessionBundle | None = None
    error: str | None = None


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
    sessionScratchDirChanged = Signal(object)  # Path | None — emitted on session start/end
    bufferMetricsChanged = Signal(object)  # carries BufferMetrics; routes to status bar
    strategyModeChanged = Signal(object)  # carries str; routes to status bar mode label
    cacheStorageStatsChanged = Signal()  # fired on session start/teardown/clear so the cache panel can refresh
    targetNativeSizeChanged = Signal(object)  # (width, height) on session start, None on teardown
    sessionSwitching = Signal(bool)  # True while an async source/target swap is draining+rebuilding
    # Private: a background swap finished; carries a _SwapOutcome. Emitted from
    # the swap worker thread, so the connection is queued onto the GUI thread.
    _sessionSwapReady = Signal(object)

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
        self._audio_backend_factory = audio_backend_factory or build_audio_backend

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
        # User-overridable cache root path. None → fall back to default.
        self._cache_root: Path = default_cache_root()
        # Hard cap on total cache size in bytes. 0 → uncapped.
        self._cache_size_cap_bytes: int = 0
        # Audio playback. Backend is constructed lazily because some
        # implementations (QtMultimedia) need a QApplication to exist
        # first, and tests may build the controller before that.
        self._audio_backend: AudioBackend | None = None
        self._audio_backend_name: AudioBackendName = AudioBackendName.QT
        self._audio_volume: int = 100
        # Target fps cached on load so seek-by-frame can convert to seconds.
        self._target_fps: float = 0.0
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

        # Async session-swap state. A source/target change on a running session
        # tears down + rebuilds on a background thread so the slow worker drain
        # (uninterruptible in-flight inference) never blocks the GUI. Only one
        # swap runs at a time; a request arriving mid-swap is coalesced (latest
        # wins) into _swap_pending and run when the current one finishes.
        self._swapping = False
        self._swap_pending: tuple[Path, Path] | None = None
        self._swap_thread: threading.Thread | None = None
        # Desired post-swap position/play state. Held as controller state (not a
        # per-call callback) so coalesced changes during a swap carry forward
        # the latest intent. Applied to the new executor once it's installed.
        self._restore_frame = 0
        self._restore_play = False
        # Indirection so tests can run the swap inline instead of on a thread.
        self._spawn_swap: Callable[[Callable[[], None]], threading.Thread | None] = (
            _spawn_daemon
        )
        self._sessionSwapReady.connect(self._on_session_swap_ready)

        transport.playRequested.connect(self._on_play)
        transport.pauseRequested.connect(self._on_pause)
        transport.seekRequested.connect(self._on_seek)
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
        """Build all session resources (reader pool, chain, executor) WITHOUT
        touching Qt — safe to run on a background thread. Heavy model loading
        does NOT happen here (the executor loads models asynchronously in its
        own setup thread at start()); the slow part avoided up front is the
        reader-pool probe. Raises on failure after cleaning up a half-built
        pool. The few warning emits here (cache unavailable) are Qt-signal
        emits, which are thread-safe (delivered queued to the GUI thread)."""
        source = Source(path=source_path)
        target = Target(path=target_path)
        reader_pool: ReaderPool | None = None
        try:
            # The pool's eager probe reader surfaces open errors here, so a bad
            # source fails session setup just like the old single-reader path.
            reader_factory = _make_reader_factory(
                target, self._video_backend, self._processing_scale
            )
            reader_pool = ReaderPool(
                reader_factory, size=self._reader_pool_size, name="target",
            )
            chain = self._build_chain(source)
            writer = build_image_writer(
                self._cache_settings.image_format,
                self._cache_settings.image_quality,
            )
            cache_dir = self._cache_root / _cache_key(
                source, target, chain, writer, self._processing_scale
            )
            # Cache root reachable? If not, fall back to OFF for this session
            # so the user sees something rather than a crash.
            cache_settings = self._cache_settings
            manager = CacheManager(self._cache_root)
            if not manager.is_available():
                self.errorOccurred.emit(
                    f"cache root unavailable ({self._cache_root}); running with cache OFF"
                )
                cache_settings = CacheSettings(
                    mode=CacheMode.OFF,
                    image_format=cache_settings.image_format,
                    image_quality=cache_settings.image_quality,
                    memory_max_bytes=cache_settings.memory_max_bytes,
                    write_workers=cache_settings.write_workers,
                    write_queue_size=cache_settings.write_queue_size,
                )
            elif self._cache_size_cap_bytes > 0:
                manager.enforce_size_cap(self._cache_size_cap_bytes)
            session_store = PersistentFrameStore(cache_dir, writer=writer)
            if manager.is_available():
                self._write_session_metadata(
                    manager, cache_dir, source, target,
                    reader_pool.frame_count, chain, writer,
                )
            effective_workers = self._effective_worker_count()
            self._applied_worker_count = effective_workers
            executor, write_executor = self._session_factory(
                reader_pool, chain, self._strategy, effective_workers,
                self._playback_mode, cache_settings, session_store,
            )
        except Exception:
            # Pool may have been partially built; tear it down so its threads +
            # reader handles don't leak, then propagate to the caller.
            if reader_pool is not None:
                try:
                    reader_pool.shutdown()
                except Exception:
                    pass
            raise
        return _SessionBundle(
            executor=executor,
            write_executor=write_executor,
            session_store=session_store,
            cache_dir=cache_dir,
            source=source,
            source_path=source_path,
            target_path=target_path,
            target_fps=float(reader_pool.fps) if reader_pool.fps > 0 else 0.0,
            frame_count=reader_pool.frame_count,
            native_size=(reader_pool.native_width, reader_pool.native_height),
        )

    def _install_session(self, bundle: _SessionBundle) -> None:
        """Wire a freshly built session into the controller + widgets and start
        it. Qt-touching — MUST run on the GUI thread (it creates observable
        bridges, hooks the display, emits signals, loads audio)."""
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

    def _change_session_async(self, source_path: Path, target_path: Path) -> None:
        """Swap the running session to a new source/target WITHOUT blocking the
        GUI. The slow teardown (draining uninterruptible in-flight inference)
        and the rebuild run on a worker thread; the desired position/play state
        (self._restore_frame/_play) is applied once the new executor is
        installed. A request arriving while a swap is already running is
        coalesced (latest wins) — callers update _restore_* + _current_*_path
        before calling so the coalesced build targets the latest selection."""
        if self._swapping:
            self._swap_pending = (source_path, target_path)
            return
        self._begin_swap(source_path, target_path)

    def _begin_swap(self, source_path: Path, target_path: Path) -> None:
        self._swapping = True
        self.sessionSwitching.emit(True)
        # Pause audio for the brief swap window; _on_session_swap_ready restores
        # it (re-seek + play) once the new world is live. The EXECUTOR is NOT
        # torn down — it keeps running the old frames until the in-place swap
        # lands, so the display never blanks.
        if self._audio_backend is not None and self._audio_backend.is_loaded():
            self._audio_backend.pause()
        self._swap_thread = self._spawn_swap(
            lambda: self._run_swap_job(source_path, target_path)
        )

    def _run_swap_job(self, source_path: Path, target_path: Path) -> None:
        """Worker-thread half of an async source/target change. Builds the new
        session world (the slow reader-probe runs here, off the GUI thread) as an
        UNSTARTED executor, then hands its world to the LIVE executor to adopt —
        keeping the live executor's worker threads (and their ORT per-thread CUDA
        state) alive instead of churning them, which is what leaked GPU memory.
        Shuts the displaced old resources down here too, then marshals GUI-ref
        updates back to the main thread."""
        executor = self._executor
        if executor is None:
            # No live executor to adopt into (shouldn't happen — change_* guard
            # against it). Fall back to reporting nothing changed.
            self._sessionSwapReady.emit(_SwapOutcome(error="no active session"))
            return
        try:
            bundle = self._build_session(source_path, target_path)
        except Exception as exc:  # noqa: BLE001 — surfaced on the GUI thread
            self._sessionSwapReady.emit(_SwapOutcome(error=str(exc)))
            return
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
            self._sessionSwapReady.emit(
                _SwapOutcome(error="could not switch to the new source/target")
            )
            return
        old_reader_pool, _old_buffer = old
        # Shut the displaced resources down off the GUI thread. The old reader
        # pool's threads + the write executor's threads don't touch ORT, so
        # recreating them is harmless (unlike the worker threads we kept).
        old_reader_pool.shutdown()
        if old_write_executor is not None:
            old_write_executor.shutdown(wait=True)
        if old_store is not None:
            old_store.close()
        self._sessionSwapReady.emit(_SwapOutcome(bundle=bundle))

    @staticmethod
    def _discard_unstarted_bundle(bundle: _SessionBundle) -> None:
        """Tear down the resources of a freshly built but NEVER-INSTALLED session
        (the executor was never start()ed, so it owns no threads — only the
        reader pool + write executor + store need releasing)."""
        try:
            bundle.executor._reader_pool.shutdown()  # noqa: SLF001
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

    def _on_session_swap_ready(self, outcome: _SwapOutcome) -> None:
        """GUI thread: a background swap finished. The LIVE executor has already
        adopted the new world (or kept the old one on failure); here we just
        re-point the controller's GUI-facing references at the new resources,
        reload + restore audio, and run any swap coalesced while this one ran."""
        self._swapping = False
        self._swap_thread = None
        if outcome.error is not None:
            self.errorOccurred.emit(f"session switch failed: {outcome.error}")
        elif outcome.bundle is not None:
            self._adopt_swapped_bundle(outcome.bundle)
        self.sessionSwitching.emit(False)
        if self._swap_pending is not None:
            source_path, target_path = self._swap_pending
            self._swap_pending = None
            self._begin_swap(source_path, target_path)

    def _adopt_swapped_bundle(self, bundle: _SessionBundle) -> None:
        """Re-point controller state + widgets at the new world after the live
        executor adopted it. The executor itself (and its observable bridges)
        is unchanged — bridges stay wired to the same executor observables — so
        this only refreshes the controller-owned references, the transport
        range, the cache panel, the native-size readout, and audio."""
        self._current_source = bundle.source
        self._current_source_path = bundle.source_path
        self._current_target_path = bundle.target_path
        self._write_executor = bundle.write_executor
        self._session_store = bundle.session_store
        self._session_cache_dir = bundle.cache_dir
        self._target_fps = bundle.target_fps
        self._transport.set_frame_count(bundle.frame_count)
        self.sessionScratchDirChanged.emit(bundle.cache_dir)
        self.cacheStorageStatsChanged.emit()
        self.targetNativeSizeChanged.emit(bundle.native_size)
        # Reload audio for the (possibly new) target — a no-op in the backend
        # when the path is unchanged (source-only swap) — then restore the
        # position + play state so sound resumes with the picture.
        backend = self.audio_backend()
        if backend is not None:
            try:
                backend.load(bundle.target_path)
            except Exception as exc:
                self.errorOccurred.emit(f"audio load failed: {exc}")
        self._restore_audio_state()

    def _restore_audio_state(self) -> None:
        """Re-point the audio backend at the restored position + play state
        after an async session swap.

        _adopt_swapped_bundle has already reloaded the target into the backend
        (a no-op when only the source changed and the target path is the same).
        QtMediaAudioBackend arms a pending seek/play when the media isn't ready
        yet, so issuing these immediately is safe — they apply the moment the
        codec reports LoadedMedia. Mirrors the play/seek the transport handlers
        do, but driven by the controller's restore intent rather than a user
        action."""
        backend = self._audio_backend
        if backend is None or not backend.is_loaded():
            return
        if self._target_fps > 0:
            backend.seek_seconds(self._restore_frame / self._target_fps)
        if self._restore_play:
            backend.play()
        else:
            backend.pause()

    @staticmethod
    def _write_session_metadata(
        manager: CacheManager,
        cache_dir: Path,
        source: Source,
        target: Target,
        target_frame_count: int,
        chain: list[Processor],
        writer: ImageWriter,
    ) -> None:
        chain_summary = ", ".join(p.name for p in chain) or "(empty chain)"
        # build_image_writer round-trip: writer.cache_key starts with the
        # format token ("jpg-q95" / "png-c1") — extract the quality from
        # the writer's own attribute since the surface is open.
        if hasattr(writer, "quality"):
            quality = int(getattr(writer, "quality"))
        elif hasattr(writer, "compression"):
            quality = int(getattr(writer, "compression"))
        else:
            quality = 0
        meta = make_meta(
            source_path=str(source.path),
            target_path=str(target.path),
            target_frame_count=target_frame_count,
            image_format=writer.extension,
            image_quality=quality,
            chain_summary=chain_summary,
        )
        manager.write_meta(cache_dir, meta)

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
        strategy, worker_count, playback_mode, cache_mode. The rest of
        cache_settings (format, quality, memory cache size, write workers,
        write queue size) is stored and takes effect at the next session start
        — switching them live would require re-creating the buffer + write
        executor + store directory hash, which is what `set_source_and_target`
        already does.

        Providers (swapper, ONNX) and device (enhancer, torch) are part of the
        chain: changing either rebuilds the chain so the processors reload on
        the new hardware. A swapper-providers change also drops the shared
        insightface model + ONNX session cache, which are bound to the EP list
        they were built with.
        """
        swapper_providers = tuple(swapper_providers)
        upscaler_params = upscaler_params or UpscalerParams()
        providers_changed = swapper_providers != self._swapper_providers
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
            if providers_changed:
                # The shared insightface model + any cached ONNX sessions were
                # built with the OLD providers; drop them so the rebuilt chain
                # re-creates them on the new EP list.
                from sinner2.pipeline import face_analyser, model_cache

                model_cache.clear_session_cache()
                face_analyser.reset_shared_face_analysis()
            try:
                new_chain = self._build_chain(self._current_source)
            except Exception as exc:
                self.errorOccurred.emit(f"chain rebuild failed: {exc}")
                return
            self._executor.set_chain(new_chain)
            # When paused, seek back to the current frame so the new
            # chain reprocesses the visible pixels. Without this the
            # display stays as the old chain's output until play.
            # Playing case needs no nudge — dispatcher submits next
            # frames through the new chain on its own.
            if not self._executor.is_playing.get():
                current = self._executor.current_frame.get()
                if current >= 0:
                    self._executor.seek(current)
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
        # Both processors are optional. An empty chain is valid — the
        # executor passes frames through unchanged (raw preview). Each
        # processor gets its framework-native execution param: ONNX providers
        # for the swapper, a torch device for the enhancer.
        chain: list[Processor] = []
        if self._swapper_enabled:
            chain.append(FaceSwapper(
                source=source,
                params=self._swapper_params,
                # Pass the selection through verbatim — an EMPTY list means the
                # user unchecked everything ("no providers"); the swapper keeps
                # it empty (ORT → CPU) instead of substituting a GPU default.
                providers=list(self._swapper_providers),
                detection_sink=self._detection_sink,
            ))
        if self._enhancer_enabled:
            # GFPGAN isn't thread-safe, so a single shared instance serialises
            # every worker on its lock. Wrap it so each realtime worker gets
            # its own instance and the live chain enhances in parallel. The
            # swapper stays a shared single instance (thread-safe ORT session).
            params = self._enhancer_params
            device = self._enhancer_device
            chain.append(PerWorkerProcessor(
                factory=lambda p=params, d=device: FaceEnhancer(params=p, device=d),
                name=FaceEnhancer.name,
                # Surface the params so they're part of the frame cache key — a
                # change must invalidate cached frames, not serve stale ones.
                params=params,
            ))
        if self._upscaler_enabled:
            # Whole-frame super-resolution, last in the chain. Torch model →
            # per-worker like the enhancer.
            up_params = self._upscaler_params
            up_device = self._upscaler_device
            chain.append(PerWorkerProcessor(
                factory=lambda p=up_params, d=up_device: Upscaler(params=p, device=d),
                name=Upscaler.name,
                params=up_params,
            ))
        return chain

    def shutdown(self) -> None:
        # A swap may be mid-flight on a worker thread; wait for it so we don't
        # tear down (or exit) while it's still building/stopping a session.
        self._swap_pending = None
        swap_thread = self._swap_thread
        if swap_thread is not None and swap_thread.is_alive():
            swap_thread.join(timeout=30.0)
        self._teardown_session()
        if self._audio_backend is not None:
            self._audio_backend.shutdown()
            self._audio_backend = None

    def executor(self) -> RealtimeExecutor | None:
        return self._executor

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
        return self._cache_root

    def cache_manager(self) -> CacheManager:
        """Fresh CacheManager for the current root. Cheap to construct; we
        rebuild rather than cache so a root change is immediately visible."""
        return CacheManager(self._cache_root)

    def set_cache_root(self, path: Path | None) -> None:
        """Switch the cache root. None reverts to the default. Does NOT
        migrate existing caches — only future sessions land in the new
        location. The current session keeps its existing path until
        teardown so we don't yank the rug out mid-write."""
        new_root = Path(path) if path is not None else default_cache_root()
        if new_root == self._cache_root:
            return
        self._cache_root = new_root
        self.cacheStorageStatsChanged.emit()

    def cache_size_cap_bytes(self) -> int:
        return self._cache_size_cap_bytes

    def set_cache_size_cap_bytes(self, max_bytes: int) -> None:
        """Hard cap on total cache size. 0 = uncapped. Enforced at the
        start of each session; not enforced live (would require periodic
        size walks)."""
        self._cache_size_cap_bytes = max(0, max_bytes)

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
            self._executor._buffer.invalidate_from(0)  # noqa: SLF001
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
        result = self.cache_manager().clear_all(protect=protect)
        self.cacheStorageStatsChanged.emit()
        return result

    def _bind_observables(self, executor: RealtimeExecutor) -> None:
        current_bridge = ObservableValueBridge(executor.current_frame, self)
        current_bridge.valueChanged.connect(self._transport.set_current_frame)
        playing_bridge = ObservableValueBridge(executor.is_playing, self)
        playing_bridge.valueChanged.connect(self._transport.set_is_playing)
        status_bridge = ObservableValueBridge(executor.status, self)
        status_bridge.valueChanged.connect(self._on_status)
        fps_bridge = ObservableValueBridge(executor.processing_fps, self)
        fps_bridge.valueChanged.connect(self.processingFpsChanged)
        metrics_bridge = ObservableValueBridge(executor.metrics, self)
        metrics_bridge.valueChanged.connect(self.bufferMetricsChanged)
        mode_bridge = ObservableValueBridge(executor.strategy_mode, self)
        mode_bridge.valueChanged.connect(self.strategyModeChanged)
        self._bridges = [
            current_bridge,
            playing_bridge,
            status_bridge,
            fps_bridge,
            metrics_bridge,
            mode_bridge,
        ]

    def _on_status(self, message: object) -> None:
        text = str(message)
        if text and text.lower().startswith(("worker error", "executor.start", "session setup")):
            self.errorOccurred.emit(text)

    def _teardown_session(self) -> None:
        for bridge in self._bridges:
            bridge.shutdown()
        self._bridges = []
        # Stop audio before tearing down the executor so the user doesn't
        # hear audio continuing while the frame view freezes.
        if self._audio_backend is not None and self._audio_backend.is_loaded():
            self._audio_backend.pause()
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

    def _on_play(self) -> None:
        if self._executor is not None:
            self._executor.play()
        if self._audio_backend is not None and self._audio_backend.is_loaded():
            self._audio_backend.play()

    def _on_pause(self) -> None:
        if self._executor is not None:
            self._executor.pause()
        if self._audio_backend is not None and self._audio_backend.is_loaded():
            self._audio_backend.pause()

    def _on_seek(self, frame: int) -> None:
        if self._executor is not None:
            self._executor.seek(frame)
        if (
            self._audio_backend is not None
            and self._audio_backend.is_loaded()
            and self._target_fps > 0
        ):
            self._audio_backend.seek_seconds(frame / self._target_fps)

    # ---- Audio ----

    def audio_backend(self) -> AudioBackend | None:
        """Lazy accessor — backend is constructed on first request so the
        QApplication exists by then. Returns None if construction failed
        (logged via errorOccurred)."""
        if self._audio_backend is None:
            try:
                self._audio_backend = self._audio_backend_factory(self._audio_backend_name)
                # Replay cached volume so the backend matches the UI state the
                # user set before any media was loaded.
                self._audio_backend.set_volume(self._audio_volume / 100.0)
            except Exception as exc:
                self.errorOccurred.emit(f"audio backend init failed: {exc}")
                self._audio_backend = None
        return self._audio_backend

    def set_audio_backend(self, name: AudioBackendName) -> None:
        if name is self._audio_backend_name and self._audio_backend is not None:
            return
        if self._audio_backend is not None:
            self._audio_backend.shutdown()
            self._audio_backend = None
        self._audio_backend_name = name
        # Reconstruct so the new backend picks up the cached volume and is
        # ready for the next session's load() call.
        self.audio_backend()

    def _on_audio_volume_changed(self, value: int) -> None:
        self._audio_volume = max(0, min(100, value))
        if self.audio_backend() is not None:
            self._audio_backend.set_volume(self._audio_volume / 100.0)  # type: ignore[union-attr]

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
        self._change_session_async(
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
        if self._executor is None and not self._swapping:
            return  # no active or in-flight session
        if self._executor is not None:
            # Capture position + play state from the live session. Mid-swap
            # (executor detached) we carry forward the last-captured intent.
            self._restore_frame = max(0, self._executor.current_frame.get())
            self._restore_play = self._executor.is_playing.get()
        self._current_source_path = source_path
        self._change_session_async(source_path, self._current_target_path)

    def change_target(self, target_path: Path) -> None:
        """Replace the target. Position resets to frame 0 and the first frame is
        submitted for processing immediately so the display reflects the new
        target. Play state is preserved. Runs asynchronously (see change_source)
        so the teardown never freezes the UI."""
        if self._current_source_path is None:
            return
        if self._executor is None and not self._swapping:
            return  # no active or in-flight session
        if self._executor is not None:
            self._restore_play = self._executor.is_playing.get()
        self._restore_frame = 0  # new timeline → start at frame 0
        self._current_target_path = target_path
        self._change_session_async(self._current_source_path, target_path)

    def apply_initial_audio_state(self, volume: int) -> None:
        """Push persisted audio volume into the controller + (lazy) backend
        without re-emitting transport signals. Called once on startup
        before any session loads."""
        self._audio_volume = max(0, min(100, volume))
        # If the backend is already constructed (it isn't on first launch,
        # but be defensive), reflect the value immediately.
        if self._audio_backend is not None:
            self._audio_backend.set_volume(self._audio_volume / 100.0)
