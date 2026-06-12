"""Tests for the busy caption overlay — visibility follows the message."""
from __future__ import annotations

from sinner2.gui.widgets.busy_caption import QBusyCaption


def test_hidden_by_default(qtbot):
    cap = QBusyCaption()
    qtbot.addWidget(cap)
    assert not cap.isVisible()


def test_message_shows_and_clears(qtbot):
    cap = QBusyCaption()
    qtbot.addWidget(cap)
    cap.show()  # parent would normally be visible
    cap.show_message("Applying settings…")
    assert cap.isVisible()
    cap.show_message("")
    assert not cap.isVisible()


def test_clear_helper_hides(qtbot):
    cap = QBusyCaption()
    qtbot.addWidget(cap)
    cap.show_message("loading models…")
    cap.clear()
    assert not cap.isVisible()


def test_click_through(qtbot):
    from PySide6.QtCore import Qt

    cap = QBusyCaption()
    qtbot.addWidget(cap)
    # The caption must never eat clicks meant for the transport / overlay.
    assert cap.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
