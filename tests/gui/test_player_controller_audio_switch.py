"""Switching the audio backend mid-session must not silence audio.

Regression: set_audio_backend() shut down the old backend and built a fresh
one that only re-applied volume — it never reloaded the current media or
restored seek/play. With no media loaded, every transport action no-ops, so
audio stayed silent for the rest of the session. The fix reloads the current
target and restores position+play from the live executor.
"""
from __future__ import annotations

import pytest

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from tests.audio.test_audio_backend import FakeAudioBackend


class _Obs:
    def __init__(self, value: object) -> None:
        self._value = value

    def get(self) -> object:
        return self._value


class _FakeExecutor:
    def __init__(self, frame: int, playing: bool) -> None:
        self.current_frame = _Obs(frame)
        self.is_playing = _Obs(playing)

    def stop(self) -> None:  # called by _teardown_session on shutdown
        pass


@pytest.fixture
def widgets(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return display, transport


@pytest.fixture
def fresh_factory():
    """Returns a brand-new FakeAudioBackend on every call, tracking them all."""
    instances: list[FakeAudioBackend] = []

    def factory(_name: AudioBackendName) -> FakeAudioBackend:
        b = FakeAudioBackend()
        instances.append(b)
        return b

    factory.instances = instances  # type: ignore[attr-defined]
    return factory


def _controller(widgets, factory) -> PlayerController:
    display, transport = widgets
    return PlayerController(
        frame_display=display,
        transport=transport,
        audio_backend_factory=factory,
    )


class TestSetAudioBackendReloadsMedia:
    def test_switch_reloads_target_and_restores_playing(
        self, widgets, fresh_factory, tmp_path
    ):
        ctrl = _controller(widgets, fresh_factory)
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"x")
        ctrl._current_target_path = target  # noqa: SLF001
        ctrl._target_fps = 30.0  # noqa: SLF001
        ctrl._executor = _FakeExecutor(frame=60, playing=True)  # noqa: SLF001

        ctrl.set_audio_backend(AudioBackendName.QT)

        backend = fresh_factory.instances[-1]
        assert backend.loaded == target  # media reloaded into the new backend
        assert backend.position_s == pytest.approx(2.0)  # 60 / 30 fps
        assert backend.is_playing is True  # mirrored the live session
        ctrl.shutdown()

    def test_switch_restores_paused_state(
        self, widgets, fresh_factory, tmp_path
    ):
        ctrl = _controller(widgets, fresh_factory)
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"x")
        ctrl._current_target_path = target  # noqa: SLF001
        ctrl._target_fps = 30.0  # noqa: SLF001
        ctrl._executor = _FakeExecutor(frame=0, playing=False)  # noqa: SLF001

        ctrl.set_audio_backend(AudioBackendName.QT)

        backend = fresh_factory.instances[-1]
        assert backend.loaded == target
        assert backend.is_playing is False
        ctrl.shutdown()

    def test_switch_without_session_does_not_load(
        self, widgets, fresh_factory
    ):
        # No target loaded yet: switching the backend must not attempt a load.
        ctrl = _controller(widgets, fresh_factory)
        ctrl.set_audio_backend(AudioBackendName.QT)
        backend = fresh_factory.instances[-1]
        assert backend.loaded is None
        ctrl.shutdown()
