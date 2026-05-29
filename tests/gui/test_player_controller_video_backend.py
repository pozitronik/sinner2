"""Tests for PlayerController.set_video_backend rebuild behaviour.

We don't exercise a full session (would require real models + a real
video file); instead we patch the internal session state to simulate
"a session is active" and verify set_video_backend triggers the
rebuild path."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.io.video_backend import VideoBackend


@pytest.fixture
def widgets(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return display, transport


def _make_controller(widgets):
    display, transport = widgets
    return PlayerController(frame_display=display, transport=transport)


def _attach_fake_session(
    ctrl: PlayerController, *, playing: bool = False, current_frame: int = 0
) -> MagicMock:
    """Make the controller look like a session is active.

    set_video_backend's rebuild path keys off _executor + the two stored
    paths. We patch all three so the rebuild branch is reached without
    spinning up a real executor."""
    ctrl._current_source_path = Path("/dummy/source.jpg")  # noqa: SLF001
    ctrl._current_target_path = Path("/dummy/target.mp4")  # noqa: SLF001
    fake_executor = MagicMock()
    fake_executor.is_playing.get.return_value = playing
    fake_executor.current_frame.get.return_value = current_frame
    ctrl._executor = fake_executor  # noqa: SLF001
    return fake_executor


class TestSetVideoBackendNoOp:
    def test_same_backend_is_no_op(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        called: list[object] = []
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: called.append(a))
        # Default backend is FFMPEG; setting it again should do nothing.
        ctrl.set_video_backend(VideoBackend.FFMPEG)
        assert called == []
        ctrl.shutdown()

    def test_change_without_session_stores_value_only(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        called: list[object] = []
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: called.append(a))
        ctrl.set_video_backend(VideoBackend.CV2)
        # No session → no rebuild; value is stored for the next session.
        assert called == []
        assert ctrl.video_backend() is VideoBackend.CV2
        ctrl.shutdown()


class TestSetVideoBackendRebuild:
    def test_change_with_session_triggers_rebuild(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl)
        rebuild_calls: list[tuple[Path | None, Path | None]] = []
        monkeypatch.setattr(
            ctrl,
            "set_source_and_target",
            lambda s, t: rebuild_calls.append((s, t)),
        )
        ctrl.set_video_backend(VideoBackend.CV2)
        assert rebuild_calls == [
            (Path("/dummy/source.jpg"), Path("/dummy/target.mp4"))
        ]
        assert ctrl.video_backend() is VideoBackend.CV2
        ctrl.shutdown()

    def test_play_state_resumed_after_rebuild(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=True, current_frame=42)
        # Stub the rebuild so the _executor reference survives (so the
        # play/seek calls land on our fake instead of a None replacement).
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.set_video_backend(VideoBackend.CV2)
        # Rebuild stub left _executor alone; controller should have
        # called seek(42) and play() on it.
        fake.seek.assert_called_once_with(42)
        fake.play.assert_called_once_with()
        ctrl.shutdown()

    def test_no_play_on_paused_rebuild(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=False, current_frame=10)
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.set_video_backend(VideoBackend.CV2)
        # Paused: seek runs, play() is NOT called.
        fake.seek.assert_called_once_with(10)
        fake.play.assert_not_called()
        ctrl.shutdown()

    def test_no_seek_when_at_frame_zero(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=False, current_frame=0)
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.set_video_backend(VideoBackend.CV2)
        # Frame 0 = no need to seek (saves the worker some pointless work).
        fake.seek.assert_not_called()
        fake.play.assert_not_called()
        ctrl.shutdown()
