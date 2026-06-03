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


class _StubOnnxUpscaleSession:
    """Nearest-neighbour upscale via numpy repeat, echoing the [0,1] input."""

    def __init__(self, scale: int) -> None:
        self._scale = scale

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

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
        up = Upscaler(params=UpscalerParams(model=UpscalerModel.HAT_X4), device="cpu")
        up.setup()
        out = up.process(np.zeros((5, 5, 3), np.uint8))
        assert out.shape == (20, 20, 3)  # x4 via the ONNX path
