"""Bottom status bar with view/window action buttons.

Replaces QMainWindow's QStatusBar. QStatusBar hides its left-side widgets
whenever a temporary message is showing, which would blink the action buttons
off exactly when the user is acting (rotate, save, errors…). This custom bar
keeps the buttons, a status message, and the persistent indicators in one
QHBoxLayout where nothing hides anything else:

    [📌][📊][🔄][⛶][◧][💾]   status message …            <indicators>
     └ action buttons (left)   └ stretchy message          └ permanent (right)

The buttons are exposed as public attributes so the main window wires their
signals to its toggle handlers and reflects state (checked) on them; the
message API mirrors QStatusBar.showMessage(text, timeout).
"""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget


class QStatusActionBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Toggles reflect on/off state; actions just fire. Tooltips carry the
        # keyboard shortcut so the buttons double as shortcut discovery.
        self.on_top_button = self._toggle("📌", "Keep window on top (F12)")
        self.stats_button = self._toggle("📊", "Show stats overlay (F4)")
        self.face_button = self._toggle("👤", "Show face-detection overlay (F8)")
        self.rotate_button = self._action("🔄", "Rotate display (R)")
        self.fullscreen_button = self._toggle("⛶", "Fullscreen (F11)")
        self.side_panel_button = self._toggle("◧", "Toggle side panel (F9)")
        self.save_button = self._action("💾", "Save current frame (Ctrl+S)")

        self._message = QLabel("")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(2)
        for button in (
            self.on_top_button,
            self.stats_button,
            self.face_button,
            self.rotate_button,
            self.fullscreen_button,
            self.side_panel_button,
            self.save_button,
        ):
            layout.addWidget(button)
        # Stretchy message pushes anything added later (the indicators) right.
        layout.addWidget(self._message, stretch=1)
        self._layout = layout

        # Clears a timed message; a timeout of 0 leaves it up indefinitely.
        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(lambda: self._message.setText(""))

    @staticmethod
    def _toggle(text: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCheckable(True)
        button.setAutoRaise(True)
        return button

    @staticmethod
    def _action(text: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    # ---- Message API (mirrors QStatusBar.showMessage) ----

    def show_message(self, text: str, timeout: int = 0) -> None:
        self._message.setText(text)
        if timeout > 0:
            self._clear_timer.start(timeout)
        else:
            self._clear_timer.stop()

    def current_message(self) -> str:
        return self._message.text()

    def add_permanent_widget(self, widget: QWidget) -> None:
        """Append a persistent indicator on the right (after the stretchy
        message label). Mirrors QStatusBar.addPermanentWidget."""
        self._layout.addWidget(widget)
