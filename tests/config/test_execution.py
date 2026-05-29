"""Tests for per-processor execution profiles + device resolution."""
from __future__ import annotations

from sinner2.config.execution import (
    ExecutionProfile,
    OnnxExecution,
    TorchExecution,
    available_torch_devices,
    resolve_torch_device,
)


class TestProfiles:
    def test_workers_default(self):
        assert ExecutionProfile().workers == 1

    def test_onnx_default_providers(self):
        p = OnnxExecution()
        assert p.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert p.workers == 1

    def test_torch_default_device(self):
        assert TorchExecution().device == "auto"

    def test_onnx_roundtrip(self):
        p = OnnxExecution(providers=["CPUExecutionProvider"], workers=8)
        assert OnnxExecution.model_validate_json(p.model_dump_json()) == p

    def test_torch_roundtrip(self):
        p = TorchExecution(device="cuda:1", workers=2)
        assert TorchExecution.model_validate_json(p.model_dump_json()) == p


class TestResolveTorchDevice:
    def test_cpu(self):
        assert resolve_torch_device("cpu").type == "cpu"

    def test_cuda_unavailable_falls_back_to_cpu(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_torch_device("cuda").type == "cpu"
        assert resolve_torch_device("auto").type == "cpu"

    def test_cuda_available(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_torch_device("cuda").type == "cuda"
        assert resolve_torch_device("auto").type == "cuda"


class TestAvailableDevices:
    def test_always_includes_auto_and_cpu(self):
        vals = [v for v, _ in available_torch_devices()]
        assert "auto" in vals
        assert "cpu" in vals
