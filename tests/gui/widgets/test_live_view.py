"""Tests for QLiveView: the Camera settings tab — getters report configured
values, the "Allow camera mode" gate emits, and set_running / set_url reflect
controller state (start/stop itself is the main window's 📹 toggle, not here)."""
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


def test_allow_camera_gate_emits_and_round_trips(qtbot, view):
    assert view.allow_camera() is False  # opt-in: off by default
    with qtbot.waitSignal(view.allowCameraToggled, timeout=1000) as blocker:
        view._allow_camera.setChecked(True)  # noqa: SLF001
    assert blocker.args == [True]
    assert view.allow_camera() is True


def test_set_allow_camera_is_silent(qtbot, view):
    seen: list = []
    view.allowCameraToggled.connect(lambda on: seen.append(on))
    view.set_allow_camera(True)
    assert view.allow_camera() is True
    assert seen == []  # restore must NOT emit


def test_set_running_locks_inputs_and_gate(view):
    view.set_running(True)
    assert not view._device.isEnabled()  # noqa: SLF001
    assert not view._width.isEnabled()  # noqa: SLF001
    assert not view._allow_camera.isEnabled()  # noqa: SLF001 — can't ungate mid-run
    view.set_running(False)
    assert view._device.isEnabled()  # noqa: SLF001
    assert view._allow_camera.isEnabled()  # noqa: SLF001


def test_set_url(view):
    view.set_url("http://localhost:8080/")
    assert "8080" in view._url.text()  # noqa: SLF001
    view.set_url(None)
    assert view._url.text() == "—"  # noqa: SLF001


def test_workers_default_and_range(view):
    assert view.workers() == 1
    assert view._workers.minimum() == 1   # noqa: SLF001
    assert view._workers.maximum() == 16  # noqa: SLF001


def test_workers_getter(view):
    view._workers.setValue(4)  # noqa: SLF001
    assert view.workers() == 4


def test_workers_changed_signal(qtbot, view):
    with qtbot.waitSignal(view.workersChanged, timeout=1000) as blocker:
        view._workers.setValue(3)  # noqa: SLF001
    assert blocker.args == [3]


def test_workers_stays_enabled_while_running(view):
    # The pool resizes live, so the Workers control is NOT locked while running.
    view.set_running(True)
    assert view._workers.isEnabled()  # noqa: SLF001


def test_set_config_restores_values_without_emitting(qtbot, view):
    seen: list = []
    view.configChanged.connect(lambda: seen.append(1))
    view.set_config(device=2, width=640, height=480, fps=24, workers=3,
                    mjpeg_port=9000)
    assert view.device() == 2
    assert view.width() == 640 and view.height() == 480
    assert view.fps() == 24 and view.workers() == 3 and view.port() == 9000
    assert seen == []  # restore must NOT emit configChanged


def test_config_changed_fires_on_user_change(qtbot, view):
    with qtbot.waitSignal(view.configChanged, timeout=1000):
        view._width.setValue(800)  # noqa: SLF001
