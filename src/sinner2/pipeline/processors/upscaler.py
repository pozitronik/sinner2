"""General (non-face) super-resolution via Real-ESRGAN.

A torch processor that upscales the whole frame — distinct from the face
enhancer, which only restores the face region. Runs Real-ESRGAN on the
`basicsr` network architectures (already installed as a GFPGAN dependency), so
no extra package is needed. The model weights download lazily on first use.

Heavy: at x4 it quadruples the frame. Primarily a batch / final-output stage;
usable in realtime but slow. Per-worker (torch isn't thread-safe), with an
optional tiling mode so large frames don't exhaust VRAM, and an fp16 option.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import cv2
import numpy as np
from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.pipeline.model_cache import get_model_path
from sinner2.types import Frame


class UpscalerModel(str, Enum):
    """Selectable Real-ESRGAN models (tokens round-trip via str-Enum)."""

    GENERAL_X4V3 = "general-x4v3"  # SRVGGNetCompact — small, fast, general x4
    X4PLUS = "x4plus"             # RRDBNet — higher quality x4, heavier
    X2PLUS = "x2plus"             # RRDBNet — x2


@dataclass(frozen=True)
class _ModelSpec:
    filename: str
    scale: int
    build: Callable[[], Any]  # lazily constructs the (un-loaded) network


def _build_srvgg_x4v3() -> Any:
    from basicsr.archs.srvgg_arch import SRVGGNetCompact

    return SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
        upscale=4, act_type="prelu",
    )


def _build_rrdb(scale: int) -> Any:
    from basicsr.archs.rrdbnet_arch import RRDBNet

    return RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
        num_grow_ch=32, scale=scale,
    )


_MODEL_SPECS: dict[UpscalerModel, _ModelSpec] = {
    UpscalerModel.GENERAL_X4V3: _ModelSpec(
        "realesr-general-x4v3.pth", 4, _build_srvgg_x4v3
    ),
    UpscalerModel.X4PLUS: _ModelSpec(
        "RealESRGAN_x4plus.pth", 4, lambda: _build_rrdb(4)
    ),
    UpscalerModel.X2PLUS: _ModelSpec(
        "RealESRGAN_x2plus.pth", 2, lambda: _build_rrdb(2)
    ),
}


class UpscalerParams(SinnerBaseModel):
    model: UpscalerModel = Field(default=UpscalerModel.GENERAL_X4V3)
    tile: int = Field(
        default=0, ge=0,
        description="Tile size (px) to bound VRAM; 0 = whole frame at once",
    )
    fp16: bool = Field(default=True, description="Half precision (faster, less VRAM)")


def model_filename(model: UpscalerModel) -> str:
    """The weights filename for a model (so the GUI can ensure it's present
    — with a download confirmation — before the processor needs it)."""
    return _MODEL_SPECS[model].filename


def _load_state_dict(path: Any, device: Any) -> dict:
    import torch

    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("params_ema", "params"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def _load_model(spec: _ModelSpec, device: Any, fp16: bool) -> Any:
    """Build the network, load weights, move to device. Indirected so tests can
    stub it cheaply (no basicsr / weights needed)."""
    net = spec.build()
    # get_model_path raises if missing — the GUI ensures the model is present
    # (with a download confirmation) before enabling the upscaler.
    net.load_state_dict(
        _load_state_dict(get_model_path(spec.filename), device), strict=True
    )
    net.eval()
    if fp16 and device.type == "cuda":
        net = net.half()
    return net.to(device)


def _tiled_forward(model: Any, t: Any, scale: int, tile: int, pad: int = 10) -> Any:
    """Run `model` over `t` (1,C,H,W) in `tile`-sized patches with `pad` overlap,
    stitching the upscaled result — keeps peak VRAM bounded for large frames."""
    import torch

    _, c, h, w = t.shape
    out = torch.zeros(
        (1, c, h * scale, w * scale), dtype=t.dtype, device=t.device
    )
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            yp0, yp1 = max(y0 - pad, 0), min(y1 + pad, h)
            xp0, xp1 = max(x0 - pad, 0), min(x1 + pad, w)
            up = model(t[:, :, yp0:yp1, xp0:xp1])
            oy0, ox0 = (y0 - yp0) * scale, (x0 - xp0) * scale
            oh, ow = (y1 - y0) * scale, (x1 - x0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (
                up[:, :, oy0:oy0 + oh, ox0:ox0 + ow]
            )
    return out


def _upscale(
    model: Any, bgr: Frame, *, scale: int, device: Any, fp16: bool, tile: int
) -> Frame:
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    if fp16 and device.type == "cuda":
        t = t.half()
    with torch.no_grad():
        out = _tiled_forward(model, t, scale, tile) if tile > 0 else model(t)
    arr = (
        out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0)
        .mul(255.0).round().to(torch.uint8).cpu().numpy()
    )
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class Upscaler:
    name = "Upscaler"
    thread_safe = False  # torch model — each worker needs its own instance

    def __init__(
        self, params: UpscalerParams | None = None, device: str = "auto"
    ) -> None:
        self._params = params or UpscalerParams()
        self._device_str = device
        self._model: Any = None
        self._device: Any = None
        self._scale = 1
        self._device_is_cuda = False
        self._lock = threading.Lock()

    def setup(self) -> None:
        from sinner2.config.execution import resolve_torch_device

        device = resolve_torch_device(self._device_str)
        self._device = device
        self._device_is_cuda = device.type == "cuda"
        if device.type != "cuda":
            print(
                "[sinner2] WARNING: Upscaler (Real-ESRGAN) running on CPU "
                f"(requested device={self._device_str!r}) — very slow.",
                file=sys.stderr,
            )
        spec = _MODEL_SPECS[self._params.model]
        self._scale = spec.scale
        self._model = _load_model(spec, device, self._params.fp16)

    def process(self, frame: Frame) -> Frame:
        model = self._model
        if model is None:
            raise RuntimeError("Upscaler.process called before setup()")
        with self._lock:
            return _upscale(
                model, frame,
                scale=self._scale, device=self._device,
                fp16=self._params.fp16, tile=self._params.tile,
            )

    def release(self) -> None:
        self._model = None
        if self._device_is_cuda:
            import torch

            torch.cuda.empty_cache()
