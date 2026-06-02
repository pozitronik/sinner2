"""Changing the target while paused must keep the transport position, not
reset the slider to 0.

Regression: _adopt_swapped_bundle calls transport.set_frame_count(), which
resets the slider value to 0. While paused there's no continuous current_frame
stream to correct it, so the position read 0 until the user pressed play (which
restored it). The adopt path now restores _restore_frame explicitly.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def _controller(widgets) -> PlayerController:
    display, transport = widgets

    def factory(_name: AudioBackendName) -> FakeAudioBackend:
        return FakeAudioBackend()

    return PlayerController(
        frame_display=display,
        transport=transport,
        audio_backend_factory=factory,
    )


def _bundle(frame_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        source=object(),
        source_path=Path("s.jpg"),
        target_path=Path("t.mp4"),
        write_executor=None,
        session_store=None,
        cache_dir=Path("cache"),
        target_fps=30.0,
        frame_count=frame_count,
        native_size=(64, 48),
    )


class TestAdoptRestoresPosition:
    def test_position_restored_after_target_change(self, widgets):
        display, transport = widgets
        ctrl = _controller(widgets)
        ctrl._restore_frame = 50  # noqa: SLF001  paused at frame 50
        ctrl._restore_play = False  # noqa: SLF001

        ctrl._adopt_swapped_bundle(_bundle(frame_count=100))  # noqa: SLF001

        # Slider must show the restored position, NOT the set_frame_count reset.
        assert transport._slider.value() == 50  # noqa: SLF001
        ctrl.shutdown()
