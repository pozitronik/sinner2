"""Tests for the ⚙️ Settings dialog (tabbed Cache / Models / Camera host)."""
from __future__ import annotations

from PySide6.QtWidgets import QWidget

from sinner2.gui.widgets.settings_dialog import QSettingsDialog


def test_tabs_host_the_given_widgets(qtbot):
    cache_a, cache_b = QWidget(), QWidget()
    models, camera = QWidget(), QWidget()
    dlg = QSettingsDialog(
        cache_widgets=[cache_a, cache_b], models_view=models, camera_view=camera
    )
    qtbot.addWidget(dlg)
    titles = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]  # noqa: SLF001
    assert titles == ["Cache", "Models", "Camera"]
    assert dlg._tabs.widget(1) is models  # noqa: SLF001 — same instance, reparented
    assert dlg._tabs.widget(2) is camera  # noqa: SLF001
    # The cache group boxes were reparented under the dialog (the Cache tab).
    assert dlg.isAncestorOf(cache_a)
    assert dlg.isAncestorOf(cache_b)


def test_show_and_raise_makes_it_visible(qtbot):
    dlg = QSettingsDialog(
        cache_widgets=[], models_view=QWidget(), camera_view=QWidget()
    )
    qtbot.addWidget(dlg)
    assert dlg.isVisible() is False
    dlg.show_and_raise()
    assert dlg.isVisible() is True
    dlg.hide()  # don't leak a shown window
