"""Tests for the status-bar memory monitor (poller → panel)."""
from __future__ import annotations

from sinner2.gui.widgets.memory_monitor import MemoryMonitor

_GB = 1024 ** 3


class _FakePanel:
    def __init__(self) -> None:
        self.value: str | None = None

    def set_value(self, text: str) -> None:
        self.value = text


class TestMemoryMonitor:
    def test_writes_formatted_readout_on_init(self, qtbot):
        panel = _FakePanel()
        m = MemoryMonitor(
            panel, vram_fn=lambda: (_GB, 8 * _GB), ram_fn=lambda: 2 * _GB
        )
        # refresh() runs in __init__ — the cell shows a value immediately.
        assert panel.value == "VRAM 1.0 / 8 GB · RAM 2.0 GB"
        m.stop()

    def test_empty_when_no_probes_available(self, qtbot):
        panel = _FakePanel()
        m = MemoryMonitor(panel, vram_fn=lambda: None, ram_fn=lambda: None)
        assert panel.value == ""  # empty → the status cell hides itself
        m.stop()

    def test_refresh_reflects_new_readings(self, qtbot):
        vals = iter([(_GB, 8 * _GB), (2 * _GB, 8 * _GB)])
        panel = _FakePanel()
        m = MemoryMonitor(panel, vram_fn=lambda: next(vals), ram_fn=lambda: None)
        assert panel.value == "VRAM 1.0 / 8 GB"  # initial poll
        m.refresh()                               # next poll
        assert panel.value == "VRAM 2.0 / 8 GB"
        m.stop()

    def test_stop_halts_the_timer(self, qtbot):
        panel = _FakePanel()
        m = MemoryMonitor(panel, vram_fn=lambda: None, ram_fn=lambda: None)
        m.stop()
        assert not m._timer.isActive()  # noqa: SLF001
