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


def _load_restorer(path: Path, upscale: int, device: str) -> Any:
    """Loader indirection so tests can stub the gfpgan call cheaply."""
    import torch
    from gfpgan import GFPGANer

    return GFPGANer(
        model_path=str(path),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        device=torch.device(device),
    )


class FaceEnhancer:
    name = "FaceEnhancer"

    def __init__(self, params: FaceEnhancerParams | None = None) -> None:
        self._params = params or FaceEnhancerParams()
        self._restorer: Any = None
        # GFPGAN's restorer.enhance() is not thread-safe — its PyTorch
        # backend mutates internal state during inference. The semaphore
        # serializes concurrent enhance() calls from multiple workers
        # (matches sinner1's pattern). FaceSwapper has no such constraint
        # because ORT InferenceSession is genuinely thread-safe.
        self._enhance_lock = threading.Lock()

    def setup(self) -> None:
        import torch

        # GFPGAN is PyTorch, so its device is torch's CUDA — independent of the
        # ONNX execution providers the swapper uses. Choose it explicitly and
        # announce it: a CPU fallback pegs every core and is far slower, so
        # surface it loudly instead of letting it look like a "slow GPU".
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print("[sinner2] FaceEnhancer (GFPGAN) device: cuda", file=sys.stderr)
        else:
            print(
                "[sinner2] WARNING: FaceEnhancer (GFPGAN) running on CPU — "
                "torch.cuda.is_available() is False.",
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
