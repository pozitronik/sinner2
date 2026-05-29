"""Tests for PlayerController.change_source / change_target.

The QOL flow: when only the source path changes, the session rebuilds
under the hood but the current frame and play state are preserved.
When the target changes, position resets to 0 but the first frame is
submitted for processing so the display refreshes immediately."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls


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
    ctrl._current_source_path = Path("/dummy/source.jpg")  # noqa: SLF001
    ctrl._current_target_path = Path("/dummy/target.mp4")  # noqa: SLF001
    fake_executor = MagicMock()
    fake_executor.is_playing.get.return_value = playing
    fake_executor.current_frame.get.return_value = current_frame
    ctrl._executor = fake_executor  # noqa: SLF001
    return fake_executor


class TestChangeSourceNoOp:
    def test_no_op_without_session(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        called: list[object] = []
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: called.append(a))
        ctrl.change_source(Path("/new/source.png"))
        assert called == []
        ctrl.shutdown()

    def test_no_op_without_target_loaded(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        # Executor present but no target path stored — unusual but
        # belt-and-suspenders against bad state.
        ctrl._executor = MagicMock()  # noqa: SLF001
        ctrl._current_target_path = None  # noqa: SLF001
        called: list[object] = []
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: called.append(a))
        ctrl.change_source(Path("/new/source.png"))
        assert called == []
        ctrl.shutdown()


class TestChangeSourceWithSession:
    def test_triggers_rebuild_with_new_source_and_cached_target(
        self, widgets, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, current_frame=42)
        rebuild_calls: list[tuple[Path | None, Path | None]] = []
        monkeypatch.setattr(
            ctrl,
            "set_source_and_target",
            lambda s, t: rebuild_calls.append((s, t)),
        )
        ctrl.change_source(Path("/new/source.png"))
        # Rebuild with the new source but the same target.
        assert rebuild_calls == [(Path("/new/source.png"), Path("/dummy/target.mp4"))]
        ctrl.shutdown()

    def test_preserves_play_state_when_was_playing(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=True, current_frame=42)
        # Stub the rebuild so _executor reference survives — play/seek
        # calls land on our fake instead of a replacement instance.
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.change_source(Path("/new/source.png"))
        fake.seek.assert_called_once_with(42)
        fake.play.assert_called_once_with()
        ctrl.shutdown()

    def test_paused_session_seeks_but_does_not_play(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=False, current_frame=10)
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.change_source(Path("/new/source.png"))
        # Seek triggers processing so display updates without resuming.
        fake.seek.assert_called_once_with(10)
        fake.play.assert_not_called()
        ctrl.shutdown()

    def test_seeks_to_zero_when_current_frame_is_zero(self, widgets, monkeypatch):
        # Even at frame 0 we seek explicitly — the seek handler submits
        # the frame for processing so the display refreshes after the
        # source change. Without the seek, the user would see the OLD
        # frame from the cache (or blank).
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=False, current_frame=0)
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.change_source(Path("/new/source.png"))
        fake.seek.assert_called_once_with(0)
        ctrl.shutdown()


class TestChangeTargetNoOp:
    def test_no_op_without_session(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        called: list[object] = []
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: called.append(a))
        ctrl.change_target(Path("/new/target.mp4"))
        assert called == []
        ctrl.shutdown()

    def test_no_op_without_source_loaded(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        ctrl._executor = MagicMock()  # noqa: SLF001
        ctrl._current_source_path = None  # noqa: SLF001
        called: list[object] = []
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: called.append(a))
        ctrl.change_target(Path("/new/target.mp4"))
        assert called == []
        ctrl.shutdown()


class TestChangeTargetWithSession:
    def test_triggers_rebuild_with_cached_source_and_new_target(
        self, widgets, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, current_frame=42)
        rebuild_calls: list[tuple[Path | None, Path | None]] = []
        monkeypatch.setattr(
            ctrl,
            "set_source_and_target",
            lambda s, t: rebuild_calls.append((s, t)),
        )
        ctrl.change_target(Path("/new/target.mp4"))
        assert rebuild_calls == [(Path("/dummy/source.jpg"), Path("/new/target.mp4"))]
        ctrl.shutdown()

    def test_always_seeks_to_zero(self, widgets, monkeypatch):
        # New target = new timeline. Frame 0 of the new target is the
        # right thing to display, regardless of where the old timeline was.
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=False, current_frame=999)
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.change_target(Path("/new/target.mp4"))
        fake.seek.assert_called_once_with(0)
        ctrl.shutdown()

    def test_preserves_play_state(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(ctrl, playing=True, current_frame=999)
        monkeypatch.setattr(ctrl, "set_source_and_target", lambda *a: None)
        ctrl.change_target(Path("/new/target.mp4"))
        fake.seek.assert_called_once_with(0)
        fake.play.assert_called_once_with()
        ctrl.shutdown()
