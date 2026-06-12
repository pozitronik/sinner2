"""Tests for the Real-ESRGAN upscaler — inference, tiling, and the processor
shell — using a stub network (no weights / basicsr needed)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from types import SimpleNamespace

from sinner2.pipeline.processors import upscaler
from sinner2.pipeline.processors.upscaler import (
    _MODEL_SPECS,
    Upscaler,
    UpscalerModel,
    UpscalerParams,
    _onnx_upscale,
    _upscale,
    model_filename,
)

_CPU = torch.device("cpu")


class _StubNet:
    """Nearest-neighbour upscale — deterministic and local, so tiled and
    whole-frame results must match exactly."""

    def __init__(self, scale: int) -> None:
        self._scale = scale

    def __call__(self, t):
        return F.interpolate(t, scale_factor=self._scale, mode="nearest")

    def eval(self):
        return self

    def half(self):
        return self

    def to(self, _device):
        return self


class TestUpscaleInference:
    def test_scales_dimensions_and_keeps_uint8(self):
        frame = np.random.randint(0, 255, (8, 10, 3), dtype=np.uint8)
        out = _upscale(_StubNet(2), frame, scale=2, device=_CPU, fp16=False, tile=0)
        assert out.shape == (16, 20, 3)
        assert out.dtype == np.uint8

    def test_tiled_matches_whole_frame(self):
        frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        net = _StubNet(2)
        whole = _upscale(net, frame, scale=2, device=_CPU, fp16=False, tile=0)
        tiled = _upscale(net, frame, scale=2, device=_CPU, fp16=False, tile=8)
        assert np.array_equal(whole, tiled)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs CUDA to distinguish GPU vs CPU accumulation",
    )
    def test_tiled_forward_accumulates_on_cpu(self):
        # Tiling exists to bound peak VRAM to a single tile's activations; the
        # stitched output (~GBs at x4) must NOT be held on the GPU.
        from sinner2.pipeline.processors.upscaler import _tiled_forward

        t = torch.zeros((1, 3, 8, 8), device="cuda")
        out = _tiled_forward(_StubNet(2), t, scale=2, tile=4)
        assert out.device.type == "cpu"
        assert tuple(out.shape) == (1, 3, 16, 16)


class _StubOnnxUpscaleSession:
    """Nearest-neighbour upscale via numpy repeat, echoing the [0,1] input."""

    def __init__(self, scale: int) -> None:
        self._scale = scale

    def get_inputs(self):
        # Dynamic spatial dims → not a fixed-size model.
        return [SimpleNamespace(name="input", shape=["batch", 3, "h", "w"])]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, _names, feeds):
        x = feeds["input"]  # (1,3,H,W) float [0,1]
        up = np.repeat(np.repeat(x, self._scale, axis=2), self._scale, axis=3)
        return [up]


class TestModelRegistry:
    def test_every_model_has_a_spec_and_filename(self):
        for model in UpscalerModel:
            assert model in _MODEL_SPECS
            assert model_filename(model)  # non-empty

    def test_rrdb_family_defaults_to_tiling(self):
        # RRDB nets can't run whole-frame at FullHD (VRAM blowout → system-
        # memory fallback, measured 34-69 s/frame); tile=0 falls back to the
        # spec's 256 default — facefusion's own tiling size for them.
        for m in (
            UpscalerModel.X4PLUS,
            UpscalerModel.X2PLUS,
            UpscalerModel.REAL_ESRGAN_X4_FP16,
            UpscalerModel.REAL_ESRGAN_X2_FP16,
        ):
            assert _MODEL_SPECS[m].default_tile == 256
        assert _MODEL_SPECS[UpscalerModel.GENERAL_X4V3].default_tile == 0
        assert _MODEL_SPECS[UpscalerModel.SPAN_X4].default_tile == 0

    def test_fp16_exports_are_onnx_with_right_scales(self):
        # facefusion's upstream-validated fp16 exports: fp32 I/O contract
        # (same as every other ONNX upscaler), fp16 weights inside.
        assert _MODEL_SPECS[UpscalerModel.REAL_ESRGAN_X4_FP16].runtime == "onnx"
        assert _MODEL_SPECS[UpscalerModel.REAL_ESRGAN_X4_FP16].scale == 4
        assert _MODEL_SPECS[UpscalerModel.REAL_ESRGAN_X2_FP16].scale == 2
        assert (
            _MODEL_SPECS[UpscalerModel.REAL_ESRGAN_X4_FP16].filename
            == "real_esrgan_x4_fp16.onnx"
        )

    def test_runtimes_split_torch_vs_onnx(self):
        assert _MODEL_SPECS[UpscalerModel.X4PLUS].runtime == "torch"
        assert _MODEL_SPECS[UpscalerModel.SWINIR_M].runtime == "torch"
        assert _MODEL_SPECS[UpscalerModel.HAT_X4].runtime == "onnx"
        assert _MODEL_SPECS[UpscalerModel.HAT_X4].filename.endswith(".onnx")
        assert _MODEL_SPECS[UpscalerModel.SWINIR_M].filename.endswith(".pth")


class TestOnnxUpscale:
    def test_scales_and_keeps_uint8(self):
        frame = np.random.randint(0, 255, (8, 10, 3), dtype=np.uint8)
        out = _onnx_upscale(
            _StubOnnxUpscaleSession(4), frame, scale=4, tile=0,
            in_name="input", out_name="output",
        )
        assert out.shape == (32, 40, 3)
        assert out.dtype == np.uint8

    def test_preserves_color(self):
        # The echo-upscaler returns the input, so colors round-trip (within
        # uint8 rounding) — proving the RGB/255 normalization is symmetric.
        frame = np.random.randint(0, 255, (6, 6, 3), dtype=np.uint8)
        out = _onnx_upscale(
            _StubOnnxUpscaleSession(2), frame, scale=2, tile=0,
            in_name="input", out_name="output",
        )
        # Each 2x2 block equals the source pixel.
        assert np.array_equal(out[0:2, 0:2], np.broadcast_to(frame[0, 0], (2, 2, 3)))

    def test_tiled_matches_whole_frame(self):
        frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        s = _StubOnnxUpscaleSession(2)
        whole = _onnx_upscale(s, frame, scale=2, tile=0, in_name="input", out_name="output")
        tiled = _onnx_upscale(s, frame, scale=2, tile=8, in_name="input", out_name="output")
        assert np.array_equal(whole, tiled)

    def test_finalize_is_byte_identical_to_naive_expression(self):
        # The fused band-wise finalizer must reproduce the original five-pass
        # expression exactly — including out-of-range values (models can
        # over/undershoot [0,1]) and round-half-even ties. Odd height + a
        # tiny band size exercise the banding seams.
        import cv2

        from sinner2.pipeline.processors.upscaler import _finalize_bgr

        rng = np.random.default_rng(9)
        chw = rng.uniform(-0.2, 1.2, (3, 37, 23)).astype(np.float32)
        chw[0, 0, 0] = 0.5 / 255.0 * 255.0  # exact .5 tie in 0..255 space
        naive = cv2.cvtColor(
            (np.clip(chw, 0.0, 1.0).transpose(1, 2, 0) * 255.0)
            .round().astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        assert np.array_equal(_finalize_bgr(chw), naive)

    def test_finalize_does_not_mutate_input(self):
        from sinner2.pipeline.processors.upscaler import _finalize_bgr

        chw = np.random.default_rng(10).uniform(0, 1, (3, 8, 8)).astype(np.float32)
        original = chw.copy()
        _finalize_bgr(chw)
        assert np.array_equal(chw, original)


class _AlignRequiredNet:
    """Like SwinIR: errors unless H,W are a multiple of `align` (its window)."""

    def __init__(self, scale: int, align: int) -> None:
        self._scale = scale
        self._align = align
        self.seen: list = []

    def __call__(self, t):
        self.seen.append((int(t.shape[2]), int(t.shape[3])))
        if t.shape[2] % self._align or t.shape[3] % self._align:
            raise RuntimeError("window reshape: input not aligned")
        return F.interpolate(t, scale_factor=self._scale, mode="nearest")

    def eval(self):
        return self

    def half(self):
        return self

    def to(self, _device):
        return self


class _FixedSizeOnnxSession:
    """Like HAT: only accepts an exact `size`x`size` input (errors otherwise)."""

    def __init__(self, size: int, scale: int) -> None:
        self._size = size
        self._scale = scale
        self.seen: list = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[1, 3, self._size, self._size])]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, _names, feeds):
        x = feeds["input"]
        self.seen.append((x.shape[2], x.shape[3]))
        if x.shape[2] != self._size or x.shape[3] != self._size:
            raise RuntimeError("INVALID_ARGUMENT: fixed input size")
        up = np.repeat(np.repeat(x, self._scale, axis=2), self._scale, axis=3)
        return [up]


class TestInputAlignmentRegression:
    """SwinIR (window attention) and HAT (fixed 256 input) both crash on
    arbitrary frame sizes — the reported 'frame doesn't change' bug. The torch
    path pads to a multiple of the window; the ONNX path tiles at the fixed
    size."""

    def test_torch_align_pads_and_crops(self):
        net = _AlignRequiredNet(2, 8)
        frame = np.random.randint(0, 255, (6, 10, 3), np.uint8)  # not /8
        out = _upscale(net, frame, scale=2, device=_CPU, fp16=False, tile=0, align=8)
        assert out.shape == (12, 20, 3)             # cropped to 6*2 x 10*2
        assert net.seen[0][0] % 8 == 0 and net.seen[0][1] % 8 == 0  # net saw /8

    def test_torch_align_tiled_also_pads(self):
        net = _AlignRequiredNet(2, 8)
        frame = np.random.randint(0, 255, (20, 20, 3), np.uint8)
        out = _upscale(net, frame, scale=2, device=_CPU, fp16=False, tile=7, align=8)
        assert out.shape == (40, 40, 3)
        assert all(h % 8 == 0 and w % 8 == 0 for h, w in net.seen)

    def test_onnx_fixed_size_tiles_at_exact_size(self):
        s = _FixedSizeOnnxSession(4, 4)
        frame = np.random.randint(0, 255, (6, 10, 3), np.uint8)
        out = _onnx_upscale(
            s, frame, scale=4, tile=0, in_name="input", out_name="output",
            fixed_size=4,
        )
        assert out.shape == (24, 40, 3)              # 6*4 x 10*4
        assert all(sh == (4, 4) for sh in s.seen)    # every call exactly 4x4


class TestFp16Gating:
    def test_swinir_forces_fp32(self, monkeypatch):
        # SwinIR's attention errors in half precision — fp16 must be forced off
        # even when the user requested it.
        seen = {}

        def fake_load(spec, device, fp16):
            seen["fp16"] = fp16
            return _StubNet(spec.scale)

        monkeypatch.setattr(upscaler, "_load_model", fake_load)
        up = Upscaler(
            params=UpscalerParams(model=UpscalerModel.SWINIR_M, fp16=True),
            device="cpu",
        )
        up.setup()
        assert seen["fp16"] is False

    def test_realesrgan_keeps_fp16(self, monkeypatch):
        seen = {}

        def fake_load(spec, device, fp16):
            seen["fp16"] = fp16
            return _StubNet(spec.scale)

        monkeypatch.setattr(upscaler, "_load_model", fake_load)
        up = Upscaler(
            params=UpscalerParams(model=UpscalerModel.X4PLUS, fp16=True),
            device="cpu",
        )
        up.setup()
        assert seen["fp16"] is True


class TestUpscalerProcessor:
    def test_not_thread_safe(self):
        assert Upscaler.thread_safe is False

    def test_process_before_setup_raises(self):
        with pytest.raises(RuntimeError, match="before setup"):
            Upscaler().process(np.zeros((4, 4, 3), np.uint8))

    def test_setup_then_process_upscales(self, monkeypatch):
        monkeypatch.setattr(
            upscaler, "_load_model",
            lambda spec, device, fp16: _StubNet(spec.scale),
        )
        up = Upscaler(
            params=UpscalerParams(model=UpscalerModel.X2PLUS, fp16=False),
            device="cpu",
        )
        up.setup()
        out = up.process(np.zeros((8, 10, 3), np.uint8))
        assert out.shape == (16, 20, 3)

    def test_model_scale_resolved_from_spec(self, monkeypatch):
        monkeypatch.setattr(
            upscaler, "_load_model",
            lambda spec, device, fp16: _StubNet(spec.scale),
        )
        up = Upscaler(params=UpscalerParams(model=UpscalerModel.X4PLUS), device="cpu")
        up.setup()
        out = up.process(np.zeros((5, 5, 3), np.uint8))
        assert out.shape == (20, 20, 3)  # x4

    def test_onnx_model_dispatches_to_onnx_path(self, monkeypatch):
        monkeypatch.setattr(
            upscaler, "get_onnx_session",
            lambda filename, providers: _StubOnnxUpscaleSession(4),
        )
        up = Upscaler(params=UpscalerParams(model=UpscalerModel.ULTRASHARP_X4), device="cpu")
        up.setup()
        out = up.process(np.zeros((5, 5, 3), np.uint8))
        assert out.shape == (20, 20, 3)  # x4 via the ONNX path

    def test_fixed_size_onnx_model_tiles_through_processor(self, monkeypatch):
        # HAT has a fixed 256 input; setup detects it from the session shape and
        # process() tiles at that size (here 4 via the stub) instead of crashing.
        monkeypatch.setattr(
            upscaler, "get_onnx_session",
            lambda filename, providers: _FixedSizeOnnxSession(4, 4),
        )
        up = Upscaler(params=UpscalerParams(model=UpscalerModel.HAT_X4), device="cpu")
        up.setup()
        assert up._onnx_fixed_size == 4  # noqa: SLF001
        out = up.process(np.zeros((6, 10, 3), np.uint8))
        assert out.shape == (24, 40, 3)  # x4, tiled — no INVALID_ARGUMENT
