"""Tests for the headless memory probes + per-model footprint registry."""
from __future__ import annotations

import sys
import types

import pytest

from sinner2.pipeline import memory_probe as mp


@pytest.fixture(autouse=True)
def _reset_state():
    """Module-level probe state is global — reset it around every test."""
    mp.reset_footprints()
    mp._nvml_ready = None  # noqa: SLF001 — force re-detection
    mp._measuring = False  # noqa: SLF001
    yield
    mp.reset_footprints()
    mp._nvml_ready = None  # noqa: SLF001
    mp._measuring = False  # noqa: SLF001


def _fake_pynvml(used: int, total: int):
    return types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: "handle",
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(
            used=used, total=total
        ),
    )


class TestDeviceVram:
    def test_reads_used_and_total(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(2_000, 8_000))
        assert mp.device_vram() == (2_000, 8_000)

    def test_none_when_pynvml_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)  # import → ImportError
        assert mp.device_vram() is None

    def test_none_when_nvml_query_raises(self, monkeypatch):
        boom = types.SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda i: (_ for _ in ()).throw(
                RuntimeError("nvml")
            ),
        )
        monkeypatch.setitem(sys.modules, "pynvml", boom)
        assert mp.device_vram() is None

    def test_init_result_is_cached(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        assert mp.device_vram() is None
        assert mp._nvml_ready is False  # noqa: SLF001 — cached, won't retry import


class TestProcessRam:
    def test_reads_rss(self, monkeypatch):
        fake = types.SimpleNamespace(
            Process=lambda: types.SimpleNamespace(
                memory_info=lambda: types.SimpleNamespace(rss=4_096)
            )
        )
        monkeypatch.setitem(sys.modules, "psutil", fake)
        assert mp.process_ram() == 4_096

    def test_none_when_psutil_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert mp.process_ram() is None


_GB = 1024 ** 3


class TestFormatMemory:
    def test_vram_and_ram(self):
        out = mp.format_memory((6 * _GB + _GB // 5, 24 * _GB), 3 * _GB + _GB // 10)
        assert out == "VRAM 6.2 / 24 GB · RAM 3.1 GB"

    def test_ram_only(self):
        assert mp.format_memory(None, 2 * _GB) == "RAM 2.0 GB"

    def test_vram_only(self):
        assert mp.format_memory((_GB, 8 * _GB), None) == "VRAM 1.0 / 8 GB"

    def test_neither_is_empty_string(self):
        assert mp.format_memory(None, None) == ""


class TestMeasureModelLoad:
    def test_records_vram_and_ram_delta(self, monkeypatch):
        vrams = iter([(1_000, 8_000), (1_300, 8_000)])  # before, after
        monkeypatch.setattr(mp, "device_vram", lambda index=0: next(vrams))
        rams = iter([500, 800])
        monkeypatch.setattr(mp, "process_ram", lambda: next(rams))
        with mp.measure_model_load("foo.onnx"):
            pass
        fp = mp.model_footprints()["foo.onnx"]
        assert fp.vram_bytes == 300
        assert fp.ram_bytes == 300
        assert fp.first_load is True  # first GPU-touching load → context tax

    def test_only_first_gpu_load_is_flagged(self, monkeypatch):
        monkeypatch.setattr(mp, "device_vram", lambda index=0: (1_000, 8_000))
        monkeypatch.setattr(mp, "process_ram", lambda: 0)
        with mp.measure_model_load("a"):
            pass
        with mp.measure_model_load("b"):
            pass
        assert mp.model_footprints()["a"].first_load is True
        assert mp.model_footprints()["b"].first_load is False

    def test_concurrent_load_is_not_measured(self, monkeypatch):
        monkeypatch.setattr(mp, "device_vram", lambda index=0: (1_000, 8_000))
        monkeypatch.setattr(mp, "process_ram", lambda: 0)
        with mp.measure_model_load("outer"):
            with mp.measure_model_load("inner"):  # global counter busy → skip
                pass
        fps = mp.model_footprints()
        assert "outer" in fps and "inner" not in fps

    def test_no_gpu_records_ram_only(self, monkeypatch):
        monkeypatch.setattr(mp, "device_vram", lambda index=0: None)
        rams = iter([100, 250])
        monkeypatch.setattr(mp, "process_ram", lambda: next(rams))
        with mp.measure_model_load("cpu_model"):
            pass
        fp = mp.model_footprints()["cpu_model"]
        assert fp.vram_bytes is None
        assert fp.ram_bytes == 150
        assert fp.first_load is False  # no GPU → no context tax

    def test_measurement_error_never_breaks_the_load(self, monkeypatch):
        def boom(index=0):
            raise RuntimeError("nvml exploded")

        monkeypatch.setattr(mp, "device_vram", boom)
        monkeypatch.setattr(mp, "process_ram", lambda: 0)
        ran = []
        with mp.measure_model_load("x"):
            ran.append(1)
        assert ran == [1]  # the wrapped load still ran
        assert mp._measuring is False  # noqa: SLF001 — flag released even on error
