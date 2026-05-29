import sys
import threading
from pathlib import Path
from typing import Any

from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.pipeline.model_cache import get_model_path
from sinner2.types import Frame


class FaceEnhancerParams(SinnerBaseModel):
    upscale: int = Field(default=1, ge=1, le=4, description="Output upscale factor")
    only_center_face: bool = Field(
        default=False, description="Enhance only the center face"
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
        # GFPGAN's enhance() mutates torch state, so it isn't thread-safe. The
        # batch runner gives each worker its OWN instance (thread_safe=False);
        # this lock only matters where an instance is shared (the realtime
        # chain), serializing concurrent enhance() calls there.
        self._enhance_lock = threading.Lock()

    def setup(self) -> None:
        from sinner2.config.execution import resolve_torch_device

        # GFPGAN is PyTorch, so its device is torch's CUDA — independent of the
        # ONNX providers the swapper uses. Resolve the requested device and
        # announce it: a CPU fallback pegs every core and is far slower, so
        # surface it loudly instead of letting it look like a "slow GPU".
        device = resolve_torch_device(self._device)
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

    def process(self, frame: Frame) -> Frame:
        # Local snapshot — release() can null self._restorer concurrently;
        # holding a local ref keeps the GFPGAN object alive for this call.
        restorer = self._restorer
        if restorer is None:
            raise RuntimeError("FaceEnhancer.process called before setup()")
        with self._enhance_lock:
            _, _, restored = restorer.enhance(
                frame,
                has_aligned=False,
                only_center_face=self._params.only_center_face,
                paste_back=True,
            )
        return restored if restored is not None else frame

    def release(self) -> None:
        self._restorer = None
