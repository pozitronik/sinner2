"""Build the processor chain from a set of processor params.

Shared by the realtime path (PlayerController) and the live-camera path
(LiveController) so both compose the chain identically — same enable toggles,
same per-worker wrapping for the thread-unsafe (torch) processors, same
provider/device wiring. Kept in the pipeline layer (no GUI imports); callers
pass the raw params they hold (controller state or a widget snapshot).
"""
from __future__ import annotations

from sinner2.config.source import Source
from sinner2.pipeline.face_map import FaceMap
from sinner2.pipeline.face_map_geometry import FrameGeometry
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_enhancer import FaceEnhancer, FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapper, FaceSwapperParams
from sinner2.pipeline.processors.upscaler import Upscaler, UpscalerParams
from sinner2.pipeline.realtime.per_worker import PerWorkerProcessor


def build_chain(
    source: Source,
    *,
    swapper_enabled: bool,
    swapper_params: FaceSwapperParams,
    swapper_providers: tuple[str, ...] | list[str],
    detection_sink: object | None,
    enhancer_enabled: bool,
    enhancer_params: FaceEnhancerParams,
    enhancer_device: str,
    upscaler_enabled: bool,
    upscaler_params: UpscalerParams,
    upscaler_device: str,
    enhancer_providers: tuple[str, ...] | list[str] = (),
    upscaler_providers: tuple[str, ...] | list[str] = (),
    face_map: FaceMap | None = None,
    geometry: FrameGeometry | None = None,
) -> list[Processor]:
    """Compose the chain for the given source + params. Every processor is
    optional; an empty chain is valid (raw passthrough). Each gets its
    framework-native execution param: ONNX providers for the swapper, a torch
    device for the enhancer/upscaler. ``face_map`` (when active) routes each
    detected face to a per-identity source; ``geometry`` (when set) lets the
    swapper skip detection and rebuild faces from the precomputed table."""
    chain: list[Processor] = []
    if swapper_enabled:
        swapper = FaceSwapper(
            source=source,
            params=swapper_params,
            # Pass the selection through verbatim — an EMPTY list means the user
            # unchecked everything ("no providers"); the swapper keeps it empty
            # (ORT → CPU) instead of substituting a GPU default.
            providers=list(swapper_providers),
            detection_sink=detection_sink,
            face_map=face_map,
        )
        swapper.set_geometry(geometry)
        chain.append(swapper)
    if enhancer_enabled:
        # GFPGAN isn't thread-safe, so a single shared instance serialises every
        # worker on its lock. Wrap it so each worker gets its own instance. The
        # swapper stays a shared single instance (thread-safe ORT session).
        chain.append(PerWorkerProcessor(
            factory=lambda p=enhancer_params, d=enhancer_device,
            pr=tuple(enhancer_providers): FaceEnhancer(
                # The ONNX restorer backends run on the ENHANCER's own ONNX
                # providers; torch GFPGAN uses the CUDA device `d`.
                params=p, device=d, providers=list(pr),
            ),
            name=FaceEnhancer.name,
            # Surface the params so they're part of the frame cache key — a change
            # must invalidate cached frames, not serve stale ones.
            params=enhancer_params,
        ))
    if upscaler_enabled:
        # Whole-frame super-resolution, last in the chain. Torch model →
        # per-worker like the enhancer.
        chain.append(PerWorkerProcessor(
            factory=lambda p=upscaler_params, d=upscaler_device,
            pr=tuple(upscaler_providers): Upscaler(
                # ONNX upscalers run on the UPSCALER's own ONNX providers; torch
                # Real-ESRGAN uses the CUDA device `d`.
                params=p, device=d, providers=list(pr),
            ),
            name=Upscaler.name,
            params=upscaler_params,
        ))
    return chain
