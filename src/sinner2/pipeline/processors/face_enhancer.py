import sys
import threading
from pathlib import Path
from typing import Any

from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.model_cache import get_model_path
from sinner2.pipeline.processors.face_swapper_types import RotationAngleSource
from sinner2.pipeline.processors.rotation_compensation import (
    compute_roll,
    enhance_with_uprighting,
)
from sinner2.types import Frame


class FaceEnhancerParams(SinnerBaseModel):
    upscale: int = Field(default=1, ge=1, le=4, description="Output upscale factor")
    only_center_face: bool = Field(
        default=False, description="Enhance only the center face"
    )
    # Rotation compensation — shared config with the swapper (same UI controls).
    # GFPGAN has no rotation handling of its own and mangles tilted faces;
    # uprighting a crop and re-enhancing it fixes them. Output-affecting.
    rotation_compensation: bool = Field(default=True)
    rotation_threshold_deg: int = Field(default=15, ge=0, le=90)
    rotation_redetect: bool = Field(default=True)  # unused here; kept for parity
    rotation_angle_source: RotationAngleSource = Field(
        default=RotationAngleSource.POSE
    )


_MODEL_FILE = "GFPGANv1.4.pth"


def _load_restorer(path: Path, upscale: int, device: Any) -> Any:
    """Loader indirection so tests can stub the gfpgan call cheaply.
    ``device`` is a torch.device chosen by the caller."""
    from gfpgan import GFPGANer

    return GFPGANer(
        model_path=str(path),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        device=device,
    )


class FaceEnhancer:
    name = "FaceEnhancer"
    thread_safe = False  # GFPGAN mutates torch state — each worker needs its own

    def __init__(
        self,
        params: FaceEnhancerParams | None = None,
        device: str = "auto",
    ) -> None:
        self._params = params or FaceEnhancerParams()
        # Torch device from the enhancer's TorchExecution profile
        # ("auto"/"cpu"/"cuda"/"cuda:N"); resolved at setup().
        self._device = device
        self._restorer: Any = None
        # Detector for rotation compensation (finds tilted faces to upright).
        # Built in setup(); reuses the shared insightface model.
        self._analyser: FaceAnalyser | None = None
        # Set at setup(): whether the resolved device is CUDA, so release()
        # knows whether to hand the model's VRAM back to the driver.
        self._device_is_cuda = False
        # GFPGAN's enhance() mutates torch state, so it isn't thread-safe.
        # Both run modes now give each worker its OWN instance (batch via the
        # stage runner's per-worker pool, realtime via PerWorkerProcessor), so
        # this lock is effectively never contended — kept as a safety net for
        # any path that might share an instance.
        self._enhance_lock = threading.Lock()

    def setup(self) -> None:
        from sinner2.config.execution import resolve_torch_device

        # GFPGAN is PyTorch, so its device is torch's CUDA — independent of the
        # ONNX providers the swapper uses. Resolve the requested device and
        # announce it: a CPU fallback pegs every core and is far slower, so
        # surface it loudly instead of letting it look like a "slow GPU".
        device = resolve_torch_device(self._device)
        self._device_is_cuda = device.type == "cuda"
        if device.type == "cuda":
            print(
                f"[sinner2] FaceEnhancer (GFPGAN) device: {device}",
                file=sys.stderr,
            )
        else:
            print(
                "[sinner2] WARNING: FaceEnhancer (GFPGAN) running on CPU "
                f"(requested device={self._device!r}).",
                file=sys.stderr,
            )
        self._restorer = _load_restorer(
            get_model_path(_MODEL_FILE), self._params.upscale, device
        )
        if self._params.rotation_compensation:
            self._analyser = FaceAnalyser()

    def process(self, frame: Frame) -> Frame:
        # Local snapshot — release() can null self._restorer concurrently;
        # holding a local ref keeps the GFPGAN object alive for this call.
        restorer = self._restorer
        if restorer is None:
            raise RuntimeError("FaceEnhancer.process called before setup()")

        def enhance_image(img: Frame, only_center: bool) -> Frame:
            with self._enhance_lock:
                _, _, out = restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=only_center,
                    paste_back=True,
                )
            return out if out is not None else img

        result = enhance_image(frame, self._params.only_center_face)
        if not self._params.rotation_compensation or self._analyser is None:
            return result
        # GFPGAN just mangled any tilted faces (no rotation handling). For each
        # face rolled past the threshold, re-enhance an uprighted crop of the
        # ORIGINAL face and composite it over the cursed result.
        for face in self._analyser.analyse(frame):
            roll = compute_roll(face, self._params.rotation_angle_source)
            if abs(roll) >= self._params.rotation_threshold_deg:
                result = enhance_with_uprighting(
                    result,
                    frame,
                    face,
                    lambda crop: enhance_image(crop, only_center=True),
                    angle_deg=roll,
                )
        return result

    def release(self) -> None:
        self._restorer = None
        self._analyser = None
        # Hand the model's VRAM back to the driver. Torch's caching allocator
        # otherwise keeps the freed blocks reserved, so a realtime worker-count
        # DECREASE (or a chain rebuild) wouldn't visibly free GPU memory —
        # nvidia-smi would still show the per-worker models resident.
        if self._device_is_cuda:
            import torch

            torch.cuda.empty_cache()
