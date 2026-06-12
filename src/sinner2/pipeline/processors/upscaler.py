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
from sinner2.pipeline.model_cache import get_model_path, get_onnx_session
from sinner2.types import Frame


class UpscalerModel(str, Enum):
    """Selectable whole-frame upscaler models (tokens round-trip via str-Enum).

    The first three are Real-ESRGAN on basicsr archs (torch/.pth); SwinIR is
    also torch/.pth; HAT / UltraSharp / SPAN run as ONNX. All x4 except x2plus."""

    GENERAL_X4V3 = "general-x4v3"  # SRVGGNetCompact — small, fast, general x4
    X4PLUS = "x4plus"             # RRDBNet — higher quality x4, heavier
    X2PLUS = "x2plus"             # RRDBNet — x2
    SWINIR_M = "swinir-m"         # SwinIR real-SR M — transformer, sharp, slow
    HAT_X4 = "hat-x4"             # HAT (ONNX) — high detail
    ULTRASHARP_X4 = "ultrasharp-x4"  # 4x-UltraSharp (ONNX) — community favourite
    SPAN_X4 = "span-x4"           # SPAN (ONNX) — tiny + fast


@dataclass(frozen=True)
class _ModelSpec:
    filename: str
    scale: int
    runtime: str  # "torch" (basicsr arch + .pth) | "onnx"
    build: Callable[[], Any] | None = None  # torch arch builder; None for onnx
    align: int = 1  # input H,W must be a multiple of this (SwinIR window = 8)
    fp16_ok: bool = True  # SwinIR's attention errors in half precision → False


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


def _build_swinir() -> Any:
    # SwinIR real-SR "M" x4 config — matches the upstream
    # 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN weights (verified to load
    # strict). img_range 1.0 → [0,1] input, same as the Real-ESRGAN path.
    from basicsr.archs.swinir_arch import SwinIR

    return SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="nearest+conv", resi_connection="1conv",
    )


_MODEL_SPECS: dict[UpscalerModel, _ModelSpec] = {
    UpscalerModel.GENERAL_X4V3: _ModelSpec(
        "realesr-general-x4v3.pth", 4, "torch", _build_srvgg_x4v3
    ),
    UpscalerModel.X4PLUS: _ModelSpec(
        "RealESRGAN_x4plus.pth", 4, "torch", lambda: _build_rrdb(4)
    ),
    UpscalerModel.X2PLUS: _ModelSpec(
        "RealESRGAN_x2plus.pth", 2, "torch", lambda: _build_rrdb(2)
    ),
    UpscalerModel.SWINIR_M: _ModelSpec(
        "swinir_realsr_m_x4.pth", 4, "torch", _build_swinir, align=8, fp16_ok=False
    ),
    UpscalerModel.HAT_X4: _ModelSpec("real_hatgan_x4.onnx", 4, "onnx"),
    UpscalerModel.ULTRASHARP_X4: _ModelSpec("ultra_sharp_x4.onnx", 4, "onnx"),
    UpscalerModel.SPAN_X4: _ModelSpec("span_kendata_x4.onnx", 4, "onnx"),
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


def model_runtime(model: UpscalerModel) -> str:
    """'torch' or 'onnx'."""
    return _MODEL_SPECS[model].runtime


def model_supports_fp16(model: UpscalerModel) -> bool:
    """Whether the fp16 knob does anything for this model — only torch models
    that tolerate half precision (Real-ESRGAN yes; SwinIR no; ONNX n/a). The
    GUI greys the knob when False."""
    spec = _MODEL_SPECS[model]
    return spec.runtime == "torch" and spec.fp16_ok


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


def _run_aligned(model: Any, patch: Any, scale: int, align: int) -> Any:
    """Run `model` on `patch` (1,C,h,w), padding h,w up to a multiple of `align`
    first (SwinIR's window attention requires it) and cropping the scaled output
    back. align=1 is a no-op."""
    import torch.nn.functional as F

    h, w = patch.shape[2], patch.shape[3]
    ah, aw = (align - h % align) % align, (align - w % align) % align
    if ah or aw:
        patch = F.pad(patch, (0, aw, 0, ah), mode="replicate")
    return model(patch)[:, :, : h * scale, : w * scale]


def _tiled_forward(
    model: Any, t: Any, scale: int, tile: int, pad: int = 10, align: int = 1
) -> Any:
    """Run `model` over `t` (1,C,H,W) in `tile`-sized patches with `pad` overlap,
    stitching the upscaled result — keeps peak VRAM bounded for large frames."""
    import torch

    _, c, h, w = t.shape
    # Accumulate the stitched output on CPU (float32). The full upscaled frame
    # can be ~GBs at x4, so holding it on the GPU defeats the point of tiling
    # (bounding peak VRAM to a single tile's activations); each tile is moved to
    # CPU right after it's produced. Mirrors the ONNX tiled path.
    out = torch.zeros(
        (1, c, h * scale, w * scale), dtype=torch.float32, device="cpu"
    )
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            yp0, yp1 = max(y0 - pad, 0), min(y1 + pad, h)
            xp0, xp1 = max(x0 - pad, 0), min(x1 + pad, w)
            up = _run_aligned(model, t[:, :, yp0:yp1, xp0:xp1], scale, align)
            oy0, ox0 = (y0 - yp0) * scale, (x0 - xp0) * scale
            oh, ow = (y1 - y0) * scale, (x1 - x0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (
                up[:, :, oy0:oy0 + oh, ox0:ox0 + ow].float().cpu()
            )
    return out


def _upscale(
    model: Any, bgr: Frame, *, scale: int, device: Any, fp16: bool, tile: int,
    align: int = 1,
) -> Frame:
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    if fp16 and device.type == "cuda":
        t = t.half()
    with torch.no_grad():
        if tile > 0:
            out = _tiled_forward(model, t, scale, tile, align=align)
        else:
            out = _run_aligned(model, t, scale, align)
    arr = (
        out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0)
        .mul(255.0).round().to(torch.uint8).cpu().numpy()
    )
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _onnx_run(session: Any, chw: np.ndarray, in_name: str, out_name: str) -> np.ndarray:
    return session.run([out_name], {in_name: chw.astype(np.float32)})[0]


def _onnx_tiled(
    session: Any, chw: np.ndarray, scale: int, tile: int,
    in_name: str, out_name: str, pad: int = 10,
) -> np.ndarray:
    """Tile an (1,C,H,W) [0,1] array through the ONNX session — numpy twin of
    _tiled_forward, keeping peak memory bounded for large frames."""
    _, c, h, w = chw.shape
    out = np.zeros((1, c, h * scale, w * scale), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            yp0, yp1 = max(y0 - pad, 0), min(y1 + pad, h)
            xp0, xp1 = max(x0 - pad, 0), min(x1 + pad, w)
            up = _onnx_run(session, chw[:, :, yp0:yp1, xp0:xp1], in_name, out_name)
            oy0, ox0 = (y0 - yp0) * scale, (x0 - xp0) * scale
            oh, ow = (y1 - y0) * scale, (x1 - x0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (
                up[:, :, oy0:oy0 + oh, ox0:ox0 + ow]
            )
    return out


def _onnx_fixed_tile(
    session: Any, chw: np.ndarray, scale: int, size: int, in_name: str, out_name: str
) -> np.ndarray:
    """For ONNX models with a FIXED square input (e.g. HAT = 256x256): split the
    frame into exact `size`x`size` tiles, edge-pad partial/edge tiles up to the
    full size, run each, and stitch the (cropped) scaled outputs."""
    _, c, h, w = chw.shape
    out = np.zeros((1, c, h * scale, w * scale), np.float32)
    for y0 in range(0, h, size):
        for x0 in range(0, w, size):
            y1, x1 = min(y0 + size, h), min(x0 + size, w)
            patch = chw[:, :, y0:y1, x0:x1]
            ph, pw = size - (y1 - y0), size - (x1 - x0)
            if ph or pw:
                patch = np.pad(patch, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")
            up = _onnx_run(session, patch, in_name, out_name)
            oh, ow = (y1 - y0) * scale, (x1 - x0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = up[:, :, :oh, :ow]
    return out


def _finalize_bgr(chw: np.ndarray) -> Frame:
    """Fused CHW float RGB [0,1] → HWC BGR uint8, processed in row bands.

    The naive expression (clip → transpose → *255 → round → astype + cvtColor)
    makes five full passes over the upscaled float frame — ~3GB of memory
    traffic for a FullHD x4 output (400MB float32), which measured LARGER than
    the SPAN model's own inference (scripts/upscaler_bench.py: post 502ms vs
    forward 285ms). Band-wise processing keeps each band's intermediates
    cache-resident (one DRAM read of the float frame, one write of the uint8
    result), and assigning channels in reverse order folds the RGB→BGR swap
    in, dropping the separate cvtColor pass.

    Byte-identical to the naive expression: same mul → clip → round-half-even
    per pixel (clip before or after the multiply is equivalent for the 0..255
    bounds), and np.rint produces exact integer-valued floats, so the uint8
    cast cannot truncate differently."""
    _, h, w = chw.shape
    arr = np.empty((h, w, 3), np.uint8)
    band = max(1, (1 << 22) // max(1, w * 12))  # ~4MB float band per channel set
    for y0 in range(0, h, band):
        y1 = min(y0 + band, h)
        b = chw[:, y0:y1, :] * 255.0
        np.clip(b, 0.0, 255.0, out=b)
        np.rint(b, out=b)
        for c in range(3):
            arr[y0:y1, :, 2 - c] = b[c]
    return arr


def _onnx_upscale(
    session: Any, bgr: Frame, *, scale: int, tile: int, in_name: str, out_name: str,
    fixed_size: int | None = None,
) -> Frame:
    """ONNX whole-frame upscale. Same RGB/255 [0,1] contract as _upscale
    (verified against the facefusion upscaler models), output clamped to [0,1].
    `fixed_size` set → the model only accepts that exact square input, so tile
    at it (HAT). Otherwise dynamic-shape: whole-frame, or the user's tile."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    if fixed_size:
        out = _onnx_fixed_tile(session, chw, scale, fixed_size, in_name, out_name)
    elif tile > 0:
        out = _onnx_tiled(session, chw, scale, tile, in_name, out_name)
    else:
        out = _onnx_run(session, chw, in_name, out_name)
    return _finalize_bgr(out[0])


class Upscaler:
    name = "Upscaler"
    thread_safe = False  # torch model — each worker needs its own instance

    def __init__(
        self, params: UpscalerParams | None = None, device: str = "auto"
    ) -> None:
        self._params = params or UpscalerParams()
        self._device_str = device
        self._model: Any = None
        self._session: Any = None  # ONNX runtime models
        self._in_name = "input"
        self._out_name = "output"
        self._runtime = "torch"
        self._align = 1  # torch input alignment (SwinIR = 8)
        self._fp16 = False  # effective fp16 (off for models that can't do half)
        self._onnx_fixed_size: int | None = None  # fixed square ONNX input (HAT)
        self._device: Any = None
        self._scale = 1
        self._device_is_cuda = False
        self._providers: list[str] = []  # EPs used for an ONNX-runtime model
        self._lock = threading.Lock()

    def setup(self) -> None:
        from sinner2.config.execution import resolve_torch_device

        device = resolve_torch_device(self._device_str)
        self._device = device
        self._device_is_cuda = device.type == "cuda"
        if device.type != "cuda":
            print(
                "[sinner2] WARNING: Upscaler running on CPU "
                f"(requested device={self._device_str!r}) — very slow.",
                file=sys.stderr,
            )
        spec = _MODEL_SPECS[self._params.model]
        self._scale = spec.scale
        self._runtime = spec.runtime
        if spec.runtime == "onnx":
            # ONNX models run through onnxruntime; map the torch device to EPs.
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device_is_cuda
                else ["CPUExecutionProvider"]
            )
            self._providers = providers
            self._session = get_onnx_session(spec.filename, providers=providers)
            inp = self._session.get_inputs()[0]
            self._in_name = inp.name
            self._out_name = self._session.get_outputs()[0].name
            # A fixed (non-dynamic) square input means the model only accepts
            # that exact size — tile at it (e.g. HAT = 256). Dynamic → None.
            shape = inp.shape
            if (
                isinstance(shape, (list, tuple)) and len(shape) == 4
                and isinstance(shape[2], int) and isinstance(shape[3], int)
                and shape[2] > 0 and shape[2] == shape[3]
            ):
                self._onnx_fixed_size = shape[2]
        else:
            self._align = spec.align
            # Honour fp16 only when the arch tolerates it (SwinIR doesn't).
            self._fp16 = self._params.fp16 and spec.fp16_ok
            self._model = _load_model(spec, device, self._fp16)

    def process(self, frame: Frame) -> Frame:
        if self._runtime == "onnx":
            if self._session is None:
                raise RuntimeError("Upscaler.process called before setup()")
            with self._lock:
                return _onnx_upscale(
                    self._session, frame, scale=self._scale,
                    tile=self._params.tile, in_name=self._in_name,
                    out_name=self._out_name, fixed_size=self._onnx_fixed_size,
                )
        model = self._model
        if model is None:
            raise RuntimeError("Upscaler.process called before setup()")
        with self._lock:
            return _upscale(
                model, frame,
                scale=self._scale, device=self._device,
                fp16=self._fp16, tile=self._params.tile, align=self._align,
            )

    def cache_identity(self) -> str:
        """Output-affecting params, for the realtime cache key."""
        return self._params.model_dump_json()

    def release(self) -> None:
        self._model = None
        if self._session is not None:
            from sinner2.pipeline.model_cache import release_onnx_session

            self._session = None
            release_onnx_session(_MODEL_SPECS[self._params.model].filename, self._providers)
        if self._device_is_cuda:
            import torch

            torch.cuda.empty_cache()
