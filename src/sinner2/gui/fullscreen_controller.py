"""Fullscreen enter/exit for the main window.

Owns the *behaviour* of going fullscreen: snapshot + hide the chrome, hand the
transport row to the auto-hiding bottom bar (so playback stays reachable), go
fullscreen, and on exit restore exactly what was showing — including a window
that was maximized (showNormal alone would drop it to its restored size).

The window keeps the keyboard/button wiring (F11 / Alt+Enter / Esc toggle the
status-bar button, whose ``toggled`` routes to ``set_fullscreen``) and delegates
here. This holds REFERENCES to window widgets; Qt parents are unchanged (the bar
stays parented to the display). Window-state calls (showFullScreen / showNormal /
showMaximized / isMaximized) go through the window object so tests can stub them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QBoxLayout, QSplitter, QWidget

if TYPE_CHECKING:
    from sinner2.gui.widgets.fullscreen_control_bar import FullscreenControlBar
    from sinner2.gui.widgets.transport_controls import QTransportControls


class FullscreenController:
    def __init__(
        self,
        window: QWidget,
        *,
        fs_controls: "FullscreenControlBar",
        chrome: list[QWidget],
        central_layout: QBoxLayout,
        transport: "QTransportControls",
        top_splitter: QSplitter,
    ) -> None:
        self._window = window
        self._fs_controls = fs_controls
        self._chrome = chrome
        self._central_layout = central_layout
        self._transport = transport
        self._top_splitter = top_splitter
        # Fullscreen state is per-launch (not persisted).
        self._is_fullscreen = False
        # Saved widget visibility + maximized state for restoration on exit.
        self._pre_visibility: dict[QWidget, bool] = {}
        self._pre_maximized = False

    @property
    def is_fullscreen(self) -> bool:
        return self._is_fullscreen

    def set_fullscreen(self, on: bool) -> None:
        # Driven by the fullscreen action button (and F11 / Esc, which toggle
        # it). Guard against redundant calls so the button-toggled signal can't
        # double-enter/exit.
        if on == self._is_fullscreen:
            return
        if on:
            self.enter()
        else:
            self.exit()

    def enter(self) -> None:
        # Snapshot visibility of every chrome widget — the custom status bar
        # included (it's a normal widget in the central layout now) — so exit
        # can restore exactly what was showing. The transport is NOT hidden:
        # it's moved into the auto-hiding fullscreen bar below.
        self._pre_visibility = {w: w.isVisible() for w in self._chrome}
        # Capture maximized state BEFORE showFullScreen() clears it, so exit can
        # return to maximized rather than a smaller restored geometry.
        self._pre_maximized = self._window.isMaximized()
        for w in self._chrome:
            w.setVisible(False)
        # Hand the transport to the fullscreen bar so the playback controls stay
        # reachable (revealed on cursor-near-bottom) without permanently covering
        # the frame. removeWidget first so the central layout drops its slot
        # cleanly before the bar reparents it.
        self._central_layout.removeWidget(self._transport)
        self._fs_controls.attach(self._transport)
        self._is_fullscreen = True
        self._window.showFullScreen()
        self._fs_controls.begin()

    def exit(self) -> None:
        # Stop the cursor watch, take the transport back out of the bar, and
        # re-home it into its normal slot (below the display splitter).
        self._fs_controls.end()
        self._fs_controls.detach(self._transport)
        # Re-home the transport just below the display splitter. Resolved by the
        # splitter's current index (slots above it can shift), so it lands
        # correctly regardless.
        slot = self._central_layout.indexOf(self._top_splitter) + 1
        self._central_layout.insertWidget(slot, self._transport)
        self._transport.show()
        for w, was_visible in self._pre_visibility.items():
            w.setVisible(was_visible)
        self._pre_visibility = {}
        self._is_fullscreen = False
        # Restore the pre-fullscreen window state. showNormal() alone would drop
        # a window that was maximized down to its restored size.
        if self._pre_maximized:
            self._window.showMaximized()
        else:
            self._window.showNormal()
