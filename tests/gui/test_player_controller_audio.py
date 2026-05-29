"""Wiring tests for audio handling in PlayerController.

The Qt backend itself is not exercised — too fragile to test without
real media + audio hardware. Instead we inject a FakeAudioBackend via
the constructor's audio_backend_factory hook and assert the controller
calls it correctly on play/pause/seek/teardown."""
from __future__ import annotations

import pytest

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from tests.audio.test_audio_backend import FakeAudioBackend


@pytest.fixture
def widgets(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return display, transport


@pytest.fixture
def fake_factory():
    """Returns the same FakeAudioBackend instance each call so tests can
    inspect its state. The list is exposed so tests can grab the instance
    after construction."""
    instances: list[FakeAudioBackend] = []

    def factory(_name: AudioBackendName) -> FakeAudioBackend:
        if not instances:
            instances.append(FakeAudioBackend())
        return instances[0]

    factory.instances = instances  # type: ignore[attr-defined]
    return factory


def _make_controller(widgets, fake_factory):
    display, transport = widgets
    ctrl = PlayerController(
        frame_display=display,
        transport=transport,
        audio_backend_factory=fake_factory,
    )
    return ctrl, transport


class TestAudioBackendConstruction:
    def test_lazy_until_accessed(self, widgets, fake_factory):
        ctrl, _ = _make_controller(widgets, fake_factory)
        assert fake_factory.instances == []
        ctrl.audio_backend()
        assert len(fake_factory.instances) == 1
        ctrl.shutdown()

    def test_cached_volume_and_mute_applied_on_first_access(
        self, widgets, fake_factory
    ):
        ctrl, _ = _make_controller(widgets, fake_factory)
        ctrl.apply_initial_audio_state(volume=40, muted=True)
        backend = ctrl.audio_backend()
        assert backend is not None
        # Volume comes in as 0-1 scale after the factory hook runs.
        assert backend.volume == pytest.approx(0.4)
        assert backend.muted is True
        ctrl.shutdown()


class TestTransportSignalsRoutedToBackend:
    def test_volume_slider_change_forwarded(self, widgets, fake_factory):
        ctrl, transport = _make_controller(widgets, fake_factory)
        ctrl.audio_backend()  # force construction
        transport.volumeChanged.emit(60)
        backend = fake_factory.instances[0]
        assert backend.volume == pytest.approx(0.6)
        ctrl.shutdown()

    def test_mute_toggle_forwarded(self, widgets, fake_factory):
        ctrl, transport = _make_controller(widgets, fake_factory)
        ctrl.audio_backend()
        transport.mutedChanged.emit(True)
        assert fake_factory.instances[0].muted is True
        ctrl.shutdown()


class TestShutdown:
    def test_shutdown_calls_backend_shutdown(self, widgets, fake_factory):
        ctrl, _ = _make_controller(widgets, fake_factory)
        backend = ctrl.audio_backend()
        assert backend is not None
        ctrl.shutdown()
        assert backend.shut_down is True
