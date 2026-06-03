"""Tests for FullscreenControlBar — the auto-hiding fullscreen playback bar.

The bar reveals when the cursor nears the host's bottom edge and hides
otherwise; it takes custody of the (single) transport widget while
fullscreen is active so there's no duplicate control to keep in sync.

Reveal decisions are driven through apply_reveal(local_point) directly so
the tests stay deterministic — no dependency on the real cursor position
or a shown window (visibility needs a shown ancestor; we assert on the
explicit is_revealed() state instead).
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QLabel, QWidget

from sinner2.gui.widgets.fullscreen_control_bar import FullscreenControlBar


@pytest.fixture
def host(qtbot):
    w = QWidget()
    w.resize(800, 600)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def bar(qtbot, host):
    b = FullscreenControlBar(host)
    qtbot.addWidget(b)
    return b


class TestRevealLogic:
    def test_hidden_by_default(self, bar):
        assert bar.is_revealed() is False

    def test_cursor_near_bottom_reveals(self, bar, host):
        bar.apply_reveal(QPoint(host.width() // 2, host.height() - 5))
        assert bar.is_revealed() is True

    def test_cursor_far_from_bottom_hides(self, bar, host):
        # First reveal, then move the cursor up — it must hide again.
        bar.apply_reveal(QPoint(host.width() // 2, host.height() - 5))
        bar.apply_reveal(QPoint(host.width() // 2, 50))
        assert bar.is_revealed() is False

    def test_cursor_outside_horizontal_bounds_does_not_reveal(self, bar, host):
        bar.apply_reveal(QPoint(host.width() + 200, host.height() - 5))
        assert bar.is_revealed() is False

    def test_stays_revealed_while_cursor_over_bar(self, bar, host):
        # Reveal via the bottom margin, then move the cursor a little higher —
        # still within the bar's own rect — and it must stay up so reaching
        # for the slider doesn't dismiss it.
        bar.apply_reveal(QPoint(host.width() // 2, host.height() - 5))
        assert bar.is_revealed() is True
        top_of_bar = bar.geometry().top() + 2
        bar.apply_reveal(QPoint(host.width() // 2, top_of_bar))
        assert bar.is_revealed() is True

    def test_end_resets_revealed_state(self, bar, host):
        bar.apply_reveal(QPoint(host.width() // 2, host.height() - 5))
        bar.end()
        assert bar.is_revealed() is False


class TestTransportCustody:
    def test_attach_reparents_into_bar(self, bar, qtbot):
        transport = QLabel("transport")
        qtbot.addWidget(transport)
        bar.attach(transport)
        assert transport.parentWidget() is bar

    def test_detach_releases_widget(self, bar, qtbot):
        transport = QLabel("transport")
        qtbot.addWidget(transport)
        bar.attach(transport)
        bar.detach(transport)
        assert transport.parentWidget() is None
