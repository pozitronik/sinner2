"""Tests for the shared OnnxProvidersRow checkbox strip — display ordering,
CPU floor, and round-trip of a requested-but-unavailable EP."""
from __future__ import annotations

from sinner2.gui.widgets import onnx_providers_row as opr

_TENSOR = "TensorrtExecutionProvider"
_CUDA = "CUDAExecutionProvider"
_CPU = "CPUExecutionProvider"


def _patch_available(monkeypatch, providers):
    monkeypatch.setattr(opr, "available_onnx_providers", lambda: list(providers))


class TestDisplayOrder:
    def test_renders_in_platform_priority_order(self, qtbot, monkeypatch):
        # ORT reports EPs in priority order (TensorRT first, CPU last); the strip
        # must render that order regardless of which EPs the task saved — so it
        # reads the same as the live panel (the bug: saved EPs were shoved first,
        # pushing TensorRT to the end).
        _patch_available(monkeypatch, [_TENSOR, _CUDA, _CPU])
        row = opr.OnnxProvidersRow(preferred=[_CUDA, _CPU])  # saved subset
        qtbot.addWidget(row)
        assert list(row.checkboxes().keys()) == [_TENSOR, _CUDA, _CPU]

    def test_no_preferred_matches_available(self, qtbot, monkeypatch):
        # The live panel passes no `preferred`; the strip is exactly available().
        _patch_available(monkeypatch, [_TENSOR, _CUDA, _CPU])
        row = opr.OnnxProvidersRow()
        qtbot.addWidget(row)
        assert list(row.checkboxes().keys()) == [_TENSOR, _CUDA, _CPU]


class TestUnavailableRequestedEp:
    def test_inserted_keeping_priority_position(self, qtbot, monkeypatch):
        # CUDA requested but the machine only has CPU: CUDA must render (so it
        # round-trips) AND keep its position before CPU, not get appended last.
        _patch_available(monkeypatch, [_CPU])
        row = opr.OnnxProvidersRow(preferred=[_CUDA, _CPU])
        qtbot.addWidget(row)
        assert list(row.checkboxes().keys()) == [_CUDA, _CPU]
        row.set_selected([_CUDA, _CPU])
        assert row.selected() == [_CUDA, _CPU]


class TestCpuFloor:
    def test_unchecking_everything_forces_cpu(self, qtbot, monkeypatch):
        _patch_available(monkeypatch, [_TENSOR, _CUDA, _CPU])
        row = opr.OnnxProvidersRow()
        qtbot.addWidget(row)
        for cb in row.checkboxes().values():
            cb.setChecked(False)
        assert row.selected() == [_CPU]
