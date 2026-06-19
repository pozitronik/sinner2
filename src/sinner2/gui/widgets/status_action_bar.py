"""Bottom status bar: view/window action buttons (left), a status message
(middle), and persistent indicator PANELS (right).

Replaces QMainWindow's QStatusBar, which hides its left widgets whenever a
temporary message shows — that would blink the action buttons off exactly when
the user acts (rotate, save, errors…). Here nothing hides anything else:

    [📌][📊][🔄][⛶][◧][💾] │ status message …   🗄 cache │ ⏱ fps │ ▦ buffer │ ⚡ EP
     └ action buttons (left) │ └ stretchy message  └──── indicator panels (right) ────┘

Each indicator is a `_StatusPanel` cell — a thin left divider, an icon prefix
and a value — with a min-width so changing numbers don't shift the layout and
auto-hide while empty (no blank cells). This is the Delphi-VCL "status panels"
idea, flattened to dividers instead of sunken bevels.

The buttons are public attributes so the main window wires their signals to its
toggle handlers and reflects state (checked) on them; the message API mirrors
QStatusBar.showMessage(text, timeout).
"""
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QWidget,
)


def _divider() -> QFrame:
    """A thin flat vertical 1px separator between bar sections."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setLineWidth(1)
    return line


class _StatusPanel(QWidget):
    """One indicator cell: ``[divider │ icon value]``.

    Visibility is two-level: a cell shows only when it has a value AND the user
    hasn't hidden it via the bar's context menu. So an empty cell never shows a
    blank box, and a cell the user switched off stays off even with live data.
    A min-width keeps a changing value (e.g. the FPS number) from shifting
    neighbouring cells left and right.
    """

    def __init__(
        self,
        icon: str = "",
        tooltip: str = "",
        min_width: int = 0,
        key: str = "",
        label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon = icon
        self._text = ""
        self._key = key
        self._label = label or key
        self._user_visible = True
        layout = QHBoxLayout(self)
        # A little vertical inset on the divider; no horizontal margin so the
        # bar's own spacing controls the gap between cells.
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)
        layout.addWidget(_divider())
        self._value = QLabel()
        if min_width > 0:
            self._value.setMinimumWidth(min_width)
        layout.addWidget(self._value)
        if tooltip:
            self.setToolTip(tooltip)
        self.set_value("")

    def set_value(self, text: str) -> None:
        self._text = text or ""
        if self._text and self._icon:
            self._value.setText(f"{self._icon} {self._text}")
        else:
            self._value.setText(self._text)
        self._apply_visibility()

    def set_user_visible(self, visible: bool) -> None:
        """Show/hide per the context-menu toggle (independent of the value)."""
        self._user_visible = bool(visible)
        self._apply_visibility()

    def _apply_visibility(self) -> None:
        self.setVisible(self._user_visible and bool(self._text))

    def value(self) -> str:
        """The current value text (without the icon prefix); "" when hidden."""
        return self._text

    def key(self) -> str:
        return self._key

    def label(self) -> str:
        return self._label

    def user_visible(self) -> bool:
        return self._user_visible


class QStatusActionBar(QWidget):
    # Emitted (panel key, now-visible) when the user toggles a panel via the
    # right-click menu, so the main window can persist the choice.
    panelVisibilityChanged = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panels: list[_StatusPanel] = []
        # Toggles reflect on/off state; actions just fire. Tooltips carry the
        # keyboard shortcut so the buttons double as shortcut discovery.
        self.on_top_button = self._toggle("📌", "Keep window on top (F12)")
        self.stats_button = self._toggle("📊", "Show stats overlay (F4)")
        self.visualiser_button = self._toggle(
            "▦", "Show processing visualiser — per-frame pipeline state (F6)"
        )
        self.rotate_button = self._action("🔄", "Rotate display (R)")
        self.fullscreen_button = self._toggle("⛶", "Fullscreen (F11)")
        self.side_panel_button = self._toggle("◧", "Toggle side panel (F9)")
        self.save_button = self._action("💾", "Save current frame (Ctrl+S)")
        self.settings_button = self._action(
            "⚙️", "Settings — cache, models, camera"
        )

        self._message = QLabel("")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(2)
        for button in (
            self.on_top_button,
            self.stats_button,
            self.visualiser_button,
            self.rotate_button,
            self.fullscreen_button,
            self.side_panel_button,
            self.save_button,
            self.settings_button,
        ):
            layout.addWidget(button)
        # Divider sets the action-button group apart from the message.
        layout.addWidget(_divider())
        # Stretchy message pushes the indicator panels to the right edge.
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

    # ---- Indicator panels (right) ----

    def add_panel(
        self,
        icon: str = "",
        tooltip: str = "",
        min_width: int = 0,
        key: str = "",
        label: str = "",
    ) -> _StatusPanel:
        """Append a persistent indicator cell on the right and return it.

        Call ``panel.set_value(text)`` to update it; an empty value hides the
        whole cell (its divider too). Cells appear in call order, each divided
        from its neighbour — the first one's divider separates the panels from
        the stretchy message. ``key``/``label`` register the cell in the
        right-click "panels" menu so the user can show/hide it."""
        panel = _StatusPanel(icon, tooltip, min_width, key=key, label=label)
        self._layout.addWidget(panel)
        if key:
            self._panels.append(panel)
        return panel

    def set_panel_user_visible(self, key: str, visible: bool) -> None:
        """Apply a persisted show/hide choice without emitting (restore path)."""
        for panel in self._panels:
            if panel.key() == key:
                panel.set_user_visible(visible)
                return

    def hidden_panel_keys(self) -> list[str]:
        """Keys of panels the user has switched off — for persistence."""
        return [p.key() for p in self._panels if not p.user_visible()]

    def add_permanent_widget(self, widget: QWidget) -> None:
        """Append a raw persistent widget on the right (mirrors
        QStatusBar.addPermanentWidget). Prefer ``add_panel`` for indicators."""
        self._layout.addWidget(widget)

    def add_leading_button(self, button: QWidget) -> None:
        """Insert a host-owned button at the FRONT of the action group — before
        the pin button (e.g. the project 📂 menu button)."""
        self._layout.insertWidget(0, button)

    # ---- Panel-visibility context menu ----

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Right-click anywhere on the bar → a menu of checkable panel toggles."""
        if not self._panels:
            return
        menu = QMenu(self)
        menu.addSection("Status bar panels")
        for panel in self._panels:
            action = menu.addAction(panel.label())
            action.setCheckable(True)
            action.setChecked(panel.user_visible())
            # default-arg binds the loop variable; triggered passes the new state.
            action.triggered.connect(
                lambda checked, p=panel: self._toggle_panel(p, checked)
            )
        menu.exec(event.globalPos())

    def _toggle_panel(self, panel: _StatusPanel, visible: bool) -> None:
        panel.set_user_visible(visible)
        self.panelVisibilityChanged.emit(panel.key(), visible)
