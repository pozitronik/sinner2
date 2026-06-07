"""File / Live mode switch — a small two-button segmented control.

File mode is the normal seekable-target pipeline; Live mode is the camera feed.
The mode is the mutual exclusion: main_window pauses the file session when Live
is selected and stops the camera when File is selected, and shows only the
controls relevant to the active mode.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QWidget,
)


class QModeToggle(QWidget):
    modeChanged = Signal(str)  # "file" or "live", emitted on user selection only

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file_btn = QPushButton("File")
        self._live_btn = QPushButton("Live")
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for btn in (self._file_btn, self._live_btn):
            btn.setCheckable(True)
            self._group.addButton(btn)
        self._file_btn.setChecked(True)
        self._file_btn.clicked.connect(lambda: self.modeChanged.emit("file"))
        self._live_btn.clicked.connect(lambda: self.modeChanged.emit("live"))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addWidget(self._file_btn)
        row.addWidget(self._live_btn)
        row.addStretch(1)

    def mode(self) -> str:
        return "live" if self._live_btn.isChecked() else "file"

    def set_mode(self, mode: str) -> None:
        """Reflect the mode programmatically WITHOUT emitting modeChanged
        (setChecked doesn't fire clicked), so syncing state can't loop."""
        live = mode == "live"
        self._live_btn.setChecked(live)
        self._file_btn.setChecked(not live)
