"""Drives a status-bar cell with a live VRAM/RAM readout.

Polls the headless memory probes on a timer and writes a formatted string into
a status panel (anything with ``set_value(str)``). The probes are injectable so
the poller is testable without a GPU. Empty readout (no probes available) makes
the panel hide itself.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QTimer

from sinner2.pipeline.memory_probe import (
    device_vram,
    format_memory,
    process_ram,
)

VramFn = Callable[[], "tuple[int, int] | None"]
RamFn = Callable[[], "int | None"]


class MemoryMonitor(QObject):
    """Refreshes ``panel.set_value(...)`` with the live memory readout ~1 Hz."""

    def __init__(
        self,
        panel: Any,
        *,
        interval_ms: int = 1000,
        vram_fn: VramFn = device_vram,
        ram_fn: RamFn = process_ram,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel = panel
        self._vram_fn = vram_fn
        self._ram_fn = ram_fn
        self._timer = QTimer(self)
        self._timer.setInterval(max(100, interval_ms))
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()  # paint a value immediately, don't wait a full interval

    def refresh(self) -> None:
        self._panel.set_value(format_memory(self._vram_fn(), self._ram_fn()))

    def stop(self) -> None:
        self._timer.stop()
