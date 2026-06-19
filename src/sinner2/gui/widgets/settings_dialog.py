"""The ⚙️ Settings window: a tabbed, modeless dialog consolidating the formerly
scattered Cache settings + Models tab + Live tab.

It HOSTS existing widget instances rather than building its own — the cache
group boxes (owned/wired by QProcessorControls), the models view, and the camera
(live) view — so every signal connection main_window already made stays intact;
this dialog just reparents them under three tabs (Cache / Models / Camera).

Modeless (opened from the button bar's ⚙️) so cache + camera changes preview
live against the main window behind it.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class QSettingsDialog(QDialog):
    """Tabbed settings window: Cache / Models / Camera."""

    def __init__(
        self,
        *,
        cache_widgets: list[QWidget],
        models_view: QWidget,
        camera_view: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(560, 480)

        tabs = QTabWidget()
        # Cache tab: stack the (reparented) cache group boxes.
        cache_page = QWidget()
        cache_layout = QVBoxLayout(cache_page)
        for box in cache_widgets:
            cache_layout.addWidget(box)
        cache_layout.addStretch(1)
        tabs.addTab(cache_page, "Cache")
        tabs.addTab(models_view, "Models")
        tabs.addTab(camera_view, "Camera")
        self._tabs = tabs

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)

    def show_and_raise(self) -> None:
        """Show modelessly and bring to front (re-open re-focuses the window)."""
        self.show()
        self.raise_()
        self.activateWindow()
