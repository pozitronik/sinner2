"""Session construction for the realtime player.

Phase 3.1 extracts the build half of PlayerController here: the value objects
(CacheSettings, the built-but-not-installed _SessionBundle, the SessionFactory
signature) and the reader/executor construction helpers. This module is Qt-free
so the heavy build can run on a background thread; problems surfaced mid-build
are collected (a future SessionBuilder step) and emitted by the GUI caller.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sinner2.config.source import Source
from sinner2.config.target import Target, TargetKind
from sinner2.gui.cache_controller import CacheController
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
from sinner2.pipeline.image_writer import ImageFormat, ImageWriter, build_image_writer
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import FrameSkipStrategy


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


@dataclass
class _SessionBundle:
    """A fully built (but not yet installed) session — the non-Qt product of
    the session build, ready to be handed to _install_session on the GUI thread.

    `warnings` carries any non-fatal problems raised during the (Qt-free,
    possibly background-thread) build — e.g. cache root unavailable; the GUI
    caller emits them on the GUI thread after install/adopt."""

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
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionBuildSpec:
    """The session-assembly knobs SessionBuilder.build() needs beyond the
    source/target and the (controller-built, hot-swap-shared) chain + worker
    count: the frame-skip strategy, playback mode, cache settings, reader
    backend/pool size, and processing scale."""

    strategy: FrameSkipStrategy
    playback_mode: PlaybackMode
    cache_settings: CacheSettings
    video_backend: VideoBackend
    reader_pool_size: int
    processing_scale: float


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


class SessionBuilder:
    """Assembles a realtime session (reader pool, cache dir, store, executor)
    from a source/target + a chain + worker count + spec.

    Qt-free: it runs the slow reader probe and can execute on a background
    thread. Non-fatal problems (cache unavailable) are collected into the
    returned bundle's `warnings`, which the GUI caller emits. The processor
    chain and effective worker count are built by the controller — they're
    shared with the live hot-swap path — and passed in here.
    """

    def __init__(
        self, cache: CacheController, session_factory: SessionFactory
    ) -> None:
        self._cache = cache
        self._session_factory = session_factory

    def build(
        self,
        source: Source,
        target: Target,
        source_path: Path,
        target_path: Path,
        chain: list[Processor],
        effective_workers: int,
        spec: SessionBuildSpec,
    ) -> _SessionBundle:
        """Build all session resources WITHOUT touching Qt. Heavy model loading
        does NOT happen here (the executor loads models asynchronously at
        start()); the slow part avoided up front is the reader-pool probe.
        Raises on failure after cleaning up a half-built pool."""
        warnings: list[str] = []
        reader_pool: ReaderPool | None = None
        try:
            # The pool's eager probe reader surfaces open errors here, so a bad
            # source fails session setup just like the old single-reader path.
            reader_factory = _make_reader_factory(
                target, spec.video_backend, spec.processing_scale
            )
            reader_pool = ReaderPool(
                reader_factory, size=spec.reader_pool_size, name="target",
            )
            writer = build_image_writer(
                spec.cache_settings.image_format,
                spec.cache_settings.image_quality,
            )
            cache_dir = self._cache.cache_dir_for(
                source, target, chain, writer, spec.processing_scale
            )
            # Cache root reachable? If not, fall back to OFF for this session
            # so the user sees something rather than a crash.
            cache_settings = spec.cache_settings
            manager = self._cache.cache_manager()
            if not manager.is_available():
                warnings.append(
                    f"cache root unavailable ({self._cache.cache_root()}); "
                    "running with cache OFF"
                )
                cache_settings = CacheSettings(
                    mode=CacheMode.OFF,
                    image_format=cache_settings.image_format,
                    image_quality=cache_settings.image_quality,
                    memory_max_bytes=cache_settings.memory_max_bytes,
                    write_workers=cache_settings.write_workers,
                    write_queue_size=cache_settings.write_queue_size,
                )
            else:
                # Cache root OK — evict old dirs down to the cap, sparing the
                # active session's dir so a cache-hit reuse isn't evicted out
                # from under us (rank 29).
                self._cache.enforce_cap(manager, cache_dir)
            session_store = PersistentFrameStore(cache_dir, writer=writer)
            if manager.is_available():
                self._write_session_metadata(
                    manager, cache_dir, source, target,
                    reader_pool.frame_count, chain, writer,
                )
            executor, write_executor = self._session_factory(
                reader_pool, chain, spec.strategy, effective_workers,
                spec.playback_mode, cache_settings, session_store,
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
            warnings=warnings,
        )

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
