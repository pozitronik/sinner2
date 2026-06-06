"""Immutable snapshot of every processor + session parameter the GUI exposes.

`QProcessorControls.snapshot()` captures one; `apply_snapshot()` writes one back.
This is the single value object the formerly-divergent param surfaces converge on
— the controller's `apply_session_config`, settings persistence, and batch-task
construction each read from it (adapters added as those consumers are migrated),
replacing the hand-maintained 14-/39-parameter signatures and the thrice-
duplicated param capture in the main window. One object, one source of truth, so
the surfaces can no longer drift out of lockstep and silently drop a setting.
"""
from __future__ import annotations

from dataclasses import dataclass

from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processors.face_enhancer import FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams
from sinner2.pipeline.processors.upscaler import UpscalerParams


@dataclass(frozen=True)
class ProcessorParamsSnapshot:
    """The full processor + session parameter surface, captured at one instant.

    Composes the three Pydantic param models (which already compare + round-trip
    by value) with the per-processor enables, devices/providers, and the
    session-scalar group. Frozen so it can be compared for change detection and
    passed around without aliasing the live widget state.

    Note: the rotation fields live on BOTH `swapper_params` and `enhancer_params`
    because one shared set of widgets drives them — they always agree.
    """

    # Swapper
    swapper_enabled: bool
    swapper_params: FaceSwapperParams
    swapper_providers: tuple[str, ...]
    # Enhancer
    enhancer_enabled: bool
    enhancer_params: FaceEnhancerParams
    enhancer_device: str
    # Upscaler
    upscaler_enabled: bool
    upscaler_params: UpscalerParams
    upscaler_device: str
    # Session / realtime
    strategy_name: str
    realtime_workers: int
    playback_mode: PlaybackMode
    reader_pool_size: int
    processing_scale: float
    synced_max_lag_frames: int
    # Cache / output
    cache_mode: CacheMode
    image_format: ImageFormat
    image_quality: int
    memory_cache_mb: int
    write_workers: int
    write_queue_size: int
    video_backend: VideoBackend
