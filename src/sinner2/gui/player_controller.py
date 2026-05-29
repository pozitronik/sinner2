import hashlib
import os
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
from sinner2.pipeline.processors.face_enhancer import FaceEnhancer, FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapper, FaceSwapperParams
from sinner2.pipeline.realtime.executor import RealtimeExecutor
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
) -> str:
    """Stable hash of (source path, target path, chain config, writer settings).

    Two sessions with identical inputs land in the same cache subdirectory
    so processed frames carry over between runs. Different chain params,
    different image format, or different quality go to a different
    subdirectory — keeps stale frames from a different configuration out
    of view and lets the user toggle formats without losing their PNG
    cache when they switch to JPEG.
    """
    parts: list[str] = [
        str(source.path.resolve()),
        str(target.path.resolve()),
        writer.cache_key,
    ]
    for p in chain:
        parts.append(p.name)
        params = getattr(p, "_params", None)
        if params is not None and hasattr(params, "model_dump_json"):
            parts.append(params.model_dump_json())
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _make_reader(target: Target, video_backend: VideoBackend) -> TargetReader:
    if target.kind == TargetKind.IMAGE:
        return ImageTargetReader(target)
    if target.kind == TargetKind.VIDEO:
        return build_video_target_reader(target, video_backend)
    raise ValueError(f"unsupported target kind: {target.kind}")


def _make_reader_factory(
    target: Target, video_backend: VideoBackend
) -> Callable[[], TargetReader]:
    """Thunk that builds a fresh reader on each call.

    Used by ReaderPool to construct its N reader instances. Per-backend
    logic stays in _make_reader; this just curries the arguments."""
    return lambda: _make_reader(target, video_backend)


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
        self._swapper_enabled = True
        self._strategy: FrameSkipStrategy = BestEffortStrategy()
        self._worker_count = 1
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
        self._audio_muted: bool = False
        # Target fps cached on load so seek-by-frame can convert to seconds.
        self._target_fps: float = 0.0
        # Video reader backend (applies on next session start).
        self._video_backend: VideoBackend = VideoBackend.FFMPEG
        # Number of parallel readers in the ReaderPool. Default 1 ≈ current
        # single-reader behaviour. Raise for slow sources (network/HDD)
        # with SyncedStrategy. Changes rebuild the session.
        self._reader_pool_size: int = 1
        # Selected ONNX execution providers as a tuple in priority order.
        # Empty → use the platform defaults (model_cache.get_active_providers
        # falls back when set_active_providers is called with None).
        self._onnx_providers: tuple[str, ...] = ()

        transport.playRequested.connect(self._on_play)
        transport.pauseRequested.connect(self._on_pause)
        transport.seekRequested.connect(self._on_seek)
        transport.volumeChanged.connect(self._on_audio_volume_changed)
        transport.mutedChanged.connect(self._on_audio_muted_changed)

    def set_source_and_target(self, source_path: Path | None, target_path: Path | None) -> None:
        if source_path is None or target_path is None:
            return
        self._teardown_session()
        try:
            source = Source(path=source_path)
            target = Target(path=target_path)
            # Build the reader factory + pool first. The pool's eager
            # probe reader surfaces open errors here, so a bad source
            # fails session setup just like the old single-reader path.
            reader_factory = _make_reader_factory(target, self._video_backend)
            reader_pool = ReaderPool(
                reader_factory,
                size=self._reader_pool_size,
                name="target",
            )
            chain = self._build_chain(source)
            writer = build_image_writer(
                self._cache_settings.image_format,
                self._cache_settings.image_quality,
            )
            cache_dir = self._cache_root / _cache_key(source, target, chain, writer)
            # Cache root reachable? If not, fall back to OFF for this session
            # so the user sees something rather than a crash. They can change
            # the cache root in the management panel without restarting.
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
                # Evict old entries before adding a new one. The current
                # session's dir doesn't exist yet so no need to protect it.
                manager.enforce_size_cap(self._cache_size_cap_bytes)
            session_store = PersistentFrameStore(cache_dir, writer=writer)
            if manager.is_available():
                self._write_session_metadata(
                    manager,
                    cache_dir,
                    source,
                    target,
                    reader_pool.frame_count,
                    chain,
                    writer,
                )
            executor, write_executor = self._session_factory(
                reader_pool,
                chain,
                self._strategy,
                self._worker_count,
                self._playback_mode,
                cache_settings,
                session_store,
            )
        except Exception as exc:
            # Pool may have been partially built; tear it down so its
            # threads + reader handles don't leak.
            try:
                reader_pool.shutdown()  # type: ignore[possibly-undefined]
            except (NameError, Exception):
                pass
            self.errorOccurred.emit(f"session setup failed: {exc}")
            return

        executor.on_frame_ready(self._display.show_frame)
        self._bind_observables(executor)
        self._current_source = source
        self._current_source_path = source_path
        self._current_target_path = target_path
        self._executor = executor
        self._write_executor = write_executor
        self._session_store = session_store
        self._session_cache_dir = cache_dir
        self._target_fps = float(reader_pool.fps) if reader_pool.fps > 0 else 0.0
        self._transport.set_frame_count(reader_pool.frame_count)
        self.sessionScratchDirChanged.emit(cache_dir)
        self.cacheStorageStatsChanged.emit()
        # Load the target into the audio backend. Done after the executor
        # is built so a backend-init failure doesn't prevent silent playback.
        backend = self.audio_backend()
        if backend is not None:
            try:
                backend.load(target_path)
            except Exception as exc:
                self.errorOccurred.emit(f"audio load failed: {exc}")

        try:
            executor.start()
        except Exception as exc:
            self.errorOccurred.emit(f"executor.start failed: {exc}")
            self._teardown_session()

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
    ) -> None:
        """Update stored params and propagate any changes to the live session.

        Hot-swap surface: chain (on param change), strategy, worker_count,
        playback_mode, cache_mode. The rest of cache_settings (format,
        quality, memory cache size, write workers, write queue size) is
        stored and takes effect at the next session start — switching them
        live would require re-creating the buffer + write executor + store
        directory hash, which is what `set_source_and_target` already does.
        """
        chain_changed = (
            swapper_params != self._swapper_params
            or enhancer_params != self._enhancer_params
            or enhancer_enabled != self._enhancer_enabled
            or swapper_enabled != self._swapper_enabled
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
        worker_count_changed = worker_count != self._worker_count
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

        if self._executor is None or self._current_source is None:
            return
        if chain_changed:
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
        if worker_count_changed:
            self._executor.set_worker_count(worker_count)
        if playback_mode_changed:
            self._executor.set_playback_mode(playback_mode)
        if cache_mode_changed:
            self._executor.set_cache_mode(cache_settings.mode)

    def _build_chain(self, source: Source) -> list[Processor]:
        # Both processors are optional. An empty chain is valid — the
        # executor passes frames through unchanged (raw preview).
        chain: list[Processor] = []
        if self._swapper_enabled:
            chain.append(FaceSwapper(source=source, params=self._swapper_params))
        if self._enhancer_enabled:
            chain.append(FaceEnhancer(params=self._enhancer_params))
        return chain

    def shutdown(self) -> None:
        self._teardown_session()
        if self._audio_backend is not None:
            self._audio_backend.shutdown()
            self._audio_backend = None

    def executor(self) -> RealtimeExecutor | None:
        return self._executor

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
                # Replay cached volume/mute so the backend matches the UI
                # state the user set before any media was loaded.
                self._audio_backend.set_volume(self._audio_volume / 100.0)
                self._audio_backend.set_muted(self._audio_muted)
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
        # Reconstruct so the new backend picks up the cached volume/mute
        # and is ready for the next session's load() call.
        self.audio_backend()

    def _on_audio_volume_changed(self, value: int) -> None:
        self._audio_volume = max(0, min(100, value))
        if self.audio_backend() is not None:
            self._audio_backend.set_volume(self._audio_volume / 100.0)  # type: ignore[union-attr]

    def _on_audio_muted_changed(self, muted: bool) -> None:
        self._audio_muted = bool(muted)
        if self.audio_backend() is not None:
            self._audio_backend.set_muted(self._audio_muted)  # type: ignore[union-attr]

    def video_backend(self) -> VideoBackend:
        return self._video_backend

    def set_video_backend(self, backend: VideoBackend) -> None:
        """Switch the video reader backend.

        If a session is running, rebuild it so the new backend takes
        effect immediately — the chain reload that comes with this is
        the same cost as a source/target change. The current frame and
        play state are preserved across the rebuild so the user lands
        roughly where they were."""
        if backend is self._video_backend:
            return
        self._video_backend = backend
        if (
            self._executor is None
            or self._current_source_path is None
            or self._current_target_path is None
        ):
            return
        was_playing = self._executor.is_playing.get()
        last_frame = self._executor.current_frame.get()
        source_path = self._current_source_path
        target_path = self._current_target_path
        self.set_source_and_target(source_path, target_path)
        if self._executor is None:
            # Rebuild failed (errorOccurred already emitted); nothing to resume.
            return
        if last_frame > 0:
            self._executor.seek(last_frame)
        if was_playing:
            self._executor.play()

    def reader_pool_size(self) -> int:
        return self._reader_pool_size

    def set_reader_pool_size(self, n: int) -> None:
        """Change the parallel reader pool size.

        Pool size is structural — the pool can't be resized after
        construction without disrupting in-flight reads, so a change
        triggers a session rebuild (same pattern as set_video_backend).
        Current frame and play state are preserved across the rebuild.
        """
        clamped = max(1, min(16, n))
        if clamped == self._reader_pool_size:
            return
        self._reader_pool_size = clamped
        if (
            self._executor is None
            or self._current_source_path is None
            or self._current_target_path is None
        ):
            return
        was_playing = self._executor.is_playing.get()
        last_frame = self._executor.current_frame.get()
        source_path = self._current_source_path
        target_path = self._current_target_path
        self.set_source_and_target(source_path, target_path)
        if self._executor is None:
            return
        if last_frame > 0:
            self._executor.seek(last_frame)
        if was_playing:
            self._executor.play()

    def onnx_providers(self) -> tuple[str, ...]:
        return self._onnx_providers

    def effective_onnx_providers(self) -> tuple[str, ...]:
        """Whatever ORT will actually use right now.

        Prefers the most recent session's `get_providers()` (recorded
        by processors when their ONNX session loads) over the user's
        request — because ORT silently falls back when a requested
        provider can't initialise (missing runtime libs, GPU absent),
        and the GUI should show the truth. Falls back to the requested
        list when no session has loaded yet (pre-startup state).
        """
        from sinner2.pipeline import model_cache

        actual = model_cache.get_actual_providers()
        if actual:
            return actual
        return model_cache.get_active_providers()

    def set_onnx_providers(self, providers: list[str] | tuple[str, ...]) -> None:
        """Replace the active ONNX provider list.

        The new list is published globally via model_cache.set_active_providers
        — which also resets the shared insightface model. If a session is
        running, rebuild it so the chain reloads with the new providers
        (same pattern as set_video_backend). Empty input is normalised
        to "use platform defaults" via the model_cache fallback.
        """
        from sinner2.pipeline import model_cache

        new_tuple = tuple(providers)
        if new_tuple == self._onnx_providers:
            return
        self._onnx_providers = new_tuple
        model_cache.set_active_providers(new_tuple if new_tuple else None)
        if (
            self._executor is None
            or self._current_source_path is None
            or self._current_target_path is None
        ):
            return
        was_playing = self._executor.is_playing.get()
        last_frame = self._executor.current_frame.get()
        source_path = self._current_source_path
        target_path = self._current_target_path
        self.set_source_and_target(source_path, target_path)
        if self._executor is None:
            return
        self._executor.seek(max(0, last_frame))
        if was_playing:
            self._executor.play()

    def change_source(self, source_path: Path) -> None:
        """Replace the source while preserving frame position + play state.

        The chain holds a reference to the source, so a source swap
        requires a full session rebuild. We capture frame + play state,
        rebuild with the new source against the same target, then seek
        back to the captured frame (which also triggers processing so
        the display updates immediately) and resume play if it was on.
        No-op if no session is active or no target is loaded yet —
        first-load is the responsibility of set_source_and_target.
        """
        if self._executor is None or self._current_target_path is None:
            return
        was_playing = self._executor.is_playing.get()
        last_frame = self._executor.current_frame.get()
        target_path = self._current_target_path
        self.set_source_and_target(source_path, target_path)
        if self._executor is None:
            return
        # Seek to the captured frame regardless of whether it's > 0 — a
        # seek to 0 also forces the worker to process frame 0 so the
        # display reflects the new source immediately.
        self._executor.seek(max(0, last_frame))
        if was_playing:
            self._executor.play()

    def change_target(self, target_path: Path) -> None:
        """Replace the target. Position resets to frame 0 and the first
        frame is submitted for processing immediately so the display
        reflects the new target. Play state is preserved so a swap
        mid-playback keeps playing the new file from its start."""
        if self._executor is None or self._current_source_path is None:
            return
        was_playing = self._executor.is_playing.get()
        source_path = self._current_source_path
        self.set_source_and_target(source_path, target_path)
        if self._executor is None:
            return
        # seek(0) submits frame 0 for processing so the new target's
        # first frame appears without waiting for the user to press play.
        self._executor.seek(0)
        if was_playing:
            self._executor.play()

    def apply_initial_audio_state(self, volume: int, muted: bool) -> None:
        """Push persisted audio state into the controller + (lazy) backend
        without re-emitting transport signals. Called once on startup
        before any session loads."""
        self._audio_volume = max(0, min(100, volume))
        self._audio_muted = bool(muted)
        # If the backend is already constructed (it isn't on first launch,
        # but be defensive), reflect the values immediately.
        if self._audio_backend is not None:
            self._audio_backend.set_volume(self._audio_volume / 100.0)
            self._audio_backend.set_muted(self._audio_muted)
