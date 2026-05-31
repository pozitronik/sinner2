"""Tests for the Real-ESRGAN upscaler — inference, tiling, and the processor
shell — using a stub network (no weights / basicsr needed)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sinner2.pipeline.processors import upscaler
from sinner2.pipeline.processors.upscaler import (
    Upscaler,
    UpscalerModel,
    UpscalerParams,
    _upscale,
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
