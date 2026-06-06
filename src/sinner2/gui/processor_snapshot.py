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
from typing import Any

from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processors.face_enhancer import FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams
from sinner2.pipeline.processors.upscaler import UpscalerParams
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    SyncedStrategy,
)


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

    def to_session_config(self) -> dict[str, Any]:
        """Build the keyword arguments for PlayerController.apply_session_config.

        Rebuilds the frame-skip strategy from its name (+ synced lag) and bundles
        the cache knobs into a CacheSettings — exactly what the main window used
        to assemble inline. CacheSettings is imported lazily to avoid a GUI
        import cycle (player_controller → processor_controls → this module)."""
        from sinner2.gui.player_controller import CacheSettings

        strategy: FrameSkipStrategy = (
            SyncedStrategy(max_lag_frames=self.synced_max_lag_frames)
            if self.strategy_name == SyncedStrategy.__name__
            else BestEffortStrategy()
        )
        return dict(
            swapper_params=self.swapper_params,
            enhancer_params=self.enhancer_params,
            enhancer_enabled=self.enhancer_enabled,
            swapper_enabled=self.swapper_enabled,
            strategy=strategy,
            worker_count=self.realtime_workers,
            playback_mode=self.playback_mode,
            cache_settings=CacheSettings(
                mode=self.cache_mode,
                image_format=self.image_format,
                image_quality=self.image_quality,
                memory_max_bytes=self.memory_cache_mb * 1024 * 1024,
                write_workers=self.write_workers,
                write_queue_size=self.write_queue_size,
            ),
            swapper_providers=self.swapper_providers,
            enhancer_device=self.enhancer_device,
            upscaler_params=self.upscaler_params,
            upscaler_enabled=self.upscaler_enabled,
            upscaler_device=self.upscaler_device,
        )
