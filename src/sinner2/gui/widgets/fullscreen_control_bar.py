"""Auto-hiding playback bar for fullscreen mode.

In fullscreen the normal chrome (transport row, pickers, status bar) is
hidden so the frame fills the screen. This bar gives the playback controls
back without permanent clutter: the window's transport widget is reparented
IN while fullscreen is active — so the single transport stays the one
source of truth (same slider position, same play state, nothing to keep in
sync) — and handed back on exit.

A poll timer watches the cursor. Within ``_REVEAL_MARGIN_PX`` of the host's
bottom edge the bar reveals; anywhere else it hides so it never covers the
picture. Polling ``QCursor.pos()`` (rather than mouse-tracking every child
widget that might sit under the cursor) keeps the reveal logic in one place
and works regardless of which widget is focused.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QVBoxLayout, QWidget


class FullscreenControlBar(QWidget):
    # Cursor this close (in px) to the host's bottom edge reveals the bar.
    _REVEAL_MARGIN_PX = 72
    _POLL_INTERVAL_MS = 100

    def __init__(self, host: QWidget) -> None:
        super().__init__(host)
        self._host = host
        self._revealed = False
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 8, 16, 10)
        # Translucent dark backdrop so the light Qt controls stay legible
        # over an arbitrary video frame.
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: rgba(18, 18, 18, 210);")
        self._timer = QTimer(self)
        self._timer.setInterval(self._POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_cursor)
        self.hide()

    # ---- transport custody ----

    def attach(self, transport: QWidget) -> None:
        """Reparent the transport into the bar (on enter-fullscreen)."""
        self._layout.addWidget(transport)
        transport.show()

    def detach(self, transport: QWidget) -> None:
        """Release the transport so the caller can re-home it into the
        normal layout (on exit-fullscreen)."""
        self._layout.removeWidget(transport)
        transport.setParent(None)

    # ---- lifecycle ----

    def begin(self) -> None:
        """Start watching the cursor. The bar starts hidden."""
        self._revealed = False
        self.hide()
        self.reposition()
        self._timer.start()

    def end(self) -> None:
        """Stop watching the cursor and hide the bar."""
        self._timer.stop()
        self._revealed = False
        self.hide()

    def reposition(self) -> None:
        """Anchor the bar full-width along the host's bottom edge."""
        height = self.sizeHint().height()
        self.setGeometry(
            0, max(0, self._host.height() - height), self._host.width(), height
        )

    # ---- reveal logic ----

    def is_revealed(self) -> bool:
        return self._revealed

    def _poll_cursor(self) -> None:
        self.apply_reveal(self._host.mapFromGlobal(QCursor.pos()))

    def _should_reveal(self, local_pos: QPoint) -> bool:
        x = local_pos.x()
        y = local_pos.y()
        within_x = 0 <= x <= self._host.width()
        near_bottom = 0 <= (self._host.height() - y) <= self._REVEAL_MARGIN_PX
        # Stay revealed while the cursor is over the bar itself, so reaching
        # up to the slider/buttons never makes it vanish mid-gesture.
        over_bar = self._revealed and self.geometry().contains(local_pos)
        return (within_x and near_bottom) or over_bar

    def apply_reveal(self, local_pos: QPoint) -> None:
        """Show/hide the bar for a cursor position in host coordinates.
        Public so it can be driven deterministically in tests."""
        reveal = self._should_reveal(local_pos)
        if reveal == self._revealed:
            return
        self._revealed = reveal
        if reveal:
            self.reposition()
            self.show()
            self.raise_()
        else:
            self.hide()
