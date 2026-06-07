"""PlayerController.capabilities() (file caps: seekable/finite; audio only for
video) + deactivate() (session teardown that KEEPS the reusable audio backend,
unlike shutdown() which also tears the backend down)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.gui.player_controller import PlayerController
from sinner2.gui.session_capabilities import SessionKind
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls


# ---- capabilities(): pure attribute logic, no QObject init needed ----

def test_capabilities_none_when_no_session():
    pc = PlayerController.__new__(PlayerController)
    pc._executor = None  # noqa: SLF001
    assert pc.capabilities().kind is SessionKind.NONE


def test_capabilities_video_target_is_seekable_with_audio():
    pc = PlayerController.__new__(PlayerController)
    pc._executor = MagicMock()  # noqa: SLF001 — a live session
    pc._current_target_path = Path("clip.mp4")  # noqa: SLF001
    caps = pc.capabilities()
    assert caps.kind is SessionKind.FILE
    assert caps.seekable and caps.has_timeline and caps.can_play_pause
    assert caps.has_audio  # video → audio control available


def test_capabilities_image_target_has_no_audio():
    pc = PlayerController.__new__(PlayerController)
    pc._executor = MagicMock()  # noqa: SLF001
    pc._current_target_path = Path("pic.png")  # noqa: SLF001
    caps = pc.capabilities()
    assert caps.kind is SessionKind.FILE
    assert caps.seekable
    assert not caps.has_audio  # image → no audio


# ---- deactivate() vs shutdown(): audio backend lifecycle ----

@pytest.fixture
def controller(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return PlayerController(frame_display=display, transport=transport)


def test_deactivate_tears_down_without_audio_shutdown(controller):
    controller._audio = MagicMock()  # noqa: SLF001
    controller._swap = MagicMock()   # noqa: SLF001
    controller.deactivate()
    controller._swap.cancel_pending_and_join.assert_called_once()  # noqa: SLF001
    controller._audio.shutdown.assert_not_called()  # noqa: SLF001 backend reusable


def test_shutdown_also_shuts_the_audio_backend(controller):
    controller._audio = MagicMock()  # noqa: SLF001
    controller._swap = MagicMock()   # noqa: SLF001
    controller.shutdown()
    controller._audio.shutdown.assert_called_once()  # noqa: SLF001
