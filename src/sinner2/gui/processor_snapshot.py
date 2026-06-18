"""Immutable snapshot of every processor + session parameter the GUI exposes.

`QProcessorControls.snapshot()` captures one; `apply_snapshot()` writes one back.
This is the single value object the formerly-divergent param surfaces converge on
— the controller's `apply_session_config` and settings persistence each read from
it, replacing the hand-maintained 14-/39-parameter signatures and the thrice-
duplicated param capture in the main window. One object, one source of truth, so
the surfaces can no longer drift out of lockstep and silently drop a setting.

(Batch no longer reads from here — it's decoupled from the live preview and mints
each task from its own Batch Defaults template; see sinner2.batch.defaults.)
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
    PredictiveStrategy,
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
    enhancer_providers: tuple[str, ...]
    # Upscaler
    upscaler_enabled: bool
    upscaler_params: UpscalerParams
    upscaler_device: str
    upscaler_providers: tuple[str, ...]
    # Session / realtime
    strategy_name: str
    realtime_workers: int
    playback_mode: PlaybackMode
    reader_pool_size: int
    processing_scale: float
    synced_max_lag_frames: int
    predictive_max_lead_seconds: float
    preprocess_before_play: bool
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
        to assemble inline."""
        from sinner2.gui.session_builder import CacheSettings

        if self.strategy_name == SyncedStrategy.__name__:
            strategy: FrameSkipStrategy = SyncedStrategy(
                max_lag_frames=self.synced_max_lag_frames
            )
        elif self.strategy_name == BestEffortStrategy.__name__:
            strategy = BestEffortStrategy()
        else:  # PredictiveStrategy — the default for viewing
            strategy = PredictiveStrategy(
                max_lead_seconds=self.predictive_max_lead_seconds
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
            enhancer_providers=self.enhancer_providers,
            upscaler_params=self.upscaler_params,
            upscaler_enabled=self.upscaler_enabled,
            upscaler_device=self.upscaler_device,
            upscaler_providers=self.upscaler_providers,
        )

    def to_settings_kwargs(self) -> dict[str, Any]:
        """Flatten the snapshot into the flat keyword surface shared by settings
        persistence (`_update_settings`) and the widget restore path
        (`apply_restored_settings`) — the single source for that 39-field map, so
        persist and restore can't drift. str-Enum model fields become their
        stable `.value` tokens (settings round-trip + sinner1 compatibility); the
        session enums and primitives pass through as-is."""
        sp = self.swapper_params
        ep = self.enhancer_params
        up = self.upscaler_params
        return dict(
            realtime_workers=self.realtime_workers,
            strategy_name=self.strategy_name,
            enhancer_enabled=self.enhancer_enabled,
            swapper_enabled=self.swapper_enabled,
            swapper_model=sp.model.value,
            swapper_detection_interval=sp.detection_interval,
            swapper_detection_size=sp.detection_size,
            swapper_detector=sp.detector.value,
            swapper_many_faces=sp.many_faces,
            swapper_fast_paste=sp.fast_paste,
            swapper_landmark_refine=sp.landmark_refine,
            swapper_target_sex=sp.target_sex.value,
            swapper_rotation_compensation=sp.rotation_compensation,
            swapper_rotation_threshold_deg=sp.rotation_threshold_deg,
            swapper_rotation_redetect=sp.rotation_redetect,
            swapper_rotation_angle_source=sp.rotation_angle_source.value,
            swapper_occlusion_mask=sp.occlusion_mask,
            swapper_occlusion_mode=sp.occlusion_mode.value,
            swapper_occlusion_parser=sp.occlusion_parser.value,
            swapper_occluder_model=sp.occluder_model.value,
            enhancer_model=ep.model.value,
            enhancer_upscale=ep.upscale,
            enhancer_only_center_face=ep.only_center_face,
            enhancer_codeformer_fidelity=ep.codeformer_fidelity,
            enhancer_fp16=ep.fp16,
            playback_mode=self.playback_mode,
            cache_mode=self.cache_mode,
            image_format=self.image_format,
            image_quality=self.image_quality,
            memory_cache_mb=self.memory_cache_mb,
            write_workers=self.write_workers,
            write_queue_size=self.write_queue_size,
            video_backend=self.video_backend,
            reader_pool_size=self.reader_pool_size,
            processing_scale=self.processing_scale,
            synced_max_lag_frames=self.synced_max_lag_frames,
            predictive_max_lead_seconds=self.predictive_max_lead_seconds,
            preprocess_before_play=self.preprocess_before_play,
            swapper_providers=list(self.swapper_providers),
            enhancer_device=self.enhancer_device,
            enhancer_providers=list(self.enhancer_providers),
            upscaler_enabled=self.upscaler_enabled,
            upscaler_model=up.model.value,
            upscaler_tile=up.tile,
            upscaler_fp16=up.fp16,
            upscaler_device=self.upscaler_device,
            upscaler_providers=list(self.upscaler_providers),
        )
