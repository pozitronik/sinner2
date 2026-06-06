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
from sinner2.io.reader_pool import ReaderPool
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.io.video_backend import VideoBackend, build_video_target_reader
from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import FrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat
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
