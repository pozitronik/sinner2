"""Tests for QLiveView (Stage 5b): the start/stop toggle emits the right intent
for its current state, getters report the configured values, and set_running /
set_url reflect controller state."""
from __future__ import annotations

import pytest

from sinner2.gui.widgets.live_view import QLiveView


@pytest.fixture
def view(qtbot):
    v = QLiveView()
    qtbot.addWidget(v)
    return v


def test_defaults(view):
    assert view.device() == 0
    assert view.width() == 1280
    assert view.height() == 720
    assert view.fps() == 30
    assert view.port() == 8080


def test_toggle_emits_start_then_stop_by_state(view):
    starts, stops = [], []
    view.startRequested.connect(lambda: starts.append(1))
    view.stopRequested.connect(lambda: stops.append(1))

    view._toggle.click()  # noqa: SLF001 — not running → start
    assert starts == [1] and stops == []

    view.set_running(True)  # controller confirms running
    view._toggle.click()  # noqa: SLF001 — running → stop
    assert stops == [1]


def test_set_running_toggles_button_and_locks_inputs(view):
    view.set_running(True)
    assert "Stop" in view._toggle.text()  # noqa: SLF001
    assert not view._device.isEnabled()  # noqa: SLF001
    assert not view._width.isEnabled()  # noqa: SLF001
    view.set_running(False)
    assert "Start" in view._toggle.text()  # noqa: SLF001
    assert view._device.isEnabled()  # noqa: SLF001


def test_set_url(view):
    view.set_url("http://localhost:8080/")
    assert "8080" in view._url.text()  # noqa: SLF001
    view.set_url(None)
    assert view._url.text() == "—"  # noqa: SLF001
