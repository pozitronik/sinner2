"""Tests for PlayerController.set_video_backend rebuild behaviour.

A backend change rebuilds the session IN PLACE (the live executor adopts a
freshly built world via reconfigure_from) rather than tearing the executor
down — recreating the worker threads leaks GPU memory. We don't exercise a full
session (would require real models + a real video file); instead we patch the
internal session state to simulate "a session is active" and verify the change
routes through the async reconfigure path preserving frame + play state."""
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
    ctrl = PlayerController(
        frame_display=display,
        transport=transport,
        audio_backend_factory=lambda name: MagicMock(),
    )
    # Run the swap job inline so the async reconfigure completes synchronously.
    ctrl._spawn_swap = lambda job: (job(), None)[1]  # noqa: SLF001
    return ctrl


def _attach_fake_session(
    ctrl: PlayerController, *, playing: bool = False, current_frame: int = 0
) -> MagicMock:
    """Make the controller look like a session is active. The rebuild path keys
    off _executor + the two stored paths; reconfigure_from returns a displaced
    (reader_pool, buffer) tuple so the swap job's off-thread shutdown runs."""
    ctrl._current_source_path = Path("/dummy/source.jpg")  # noqa: SLF001
    ctrl._current_target_path = Path("/dummy/target.mp4")  # noqa: SLF001
    fake_executor = MagicMock()
    fake_executor.is_playing.get.return_value = playing
    fake_executor.current_frame.get.return_value = current_frame
    fake_executor.reconfigure_from.return_value = (MagicMock(), MagicMock())
    ctrl._executor = fake_executor  # noqa: SLF001
    return fake_executor


def _stub_build(ctrl: PlayerController, monkeypatch) -> tuple[list, MagicMock]:
    """Stub _build_session to record (source, target) and return a bundle around
    a fresh UNSTARTED executor."""
    builds: list[tuple[Path, Path]] = []
    unstarted = MagicMock(name="unstarted_executor")

    def fake_build(source_path, target_path):
        builds.append((source_path, target_path))
        bundle = MagicMock()
        bundle.executor = unstarted
        bundle.source_path = Path("/dummy/source.jpg")
        bundle.target_path = Path("/dummy/target.mp4")
        bundle.target_fps = 30.0
        bundle.frame_count = 100
        bundle.native_size = (1920, 1080)
        bundle.cache_dir = Path("/tmp/cache")
        bundle.write_executor = None
        bundle.session_store = None
        return bundle

    monkeypatch.setattr(ctrl, "_build_session", fake_build)
    return builds, unstarted


class TestSetVideoBackendNoOp:
    def test_same_backend_is_no_op(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        builds, _ = _stub_build(ctrl, monkeypatch)
        # Default backend is FFMPEG; setting it again should do nothing.
        ctrl.set_video_backend(VideoBackend.FFMPEG)
        assert builds == []
        ctrl.shutdown()

    def test_change_without_session_stores_value_only(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.set_video_backend(VideoBackend.CV2)
        # No session → no rebuild; value is stored for the next session.
        assert builds == []
        assert ctrl.video_backend() is VideoBackend.CV2
        ctrl.shutdown()


class TestSetVideoBackendRebuild:
    def test_change_with_session_rebuilds_in_place(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, current_frame=42)
        builds, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.set_video_backend(VideoBackend.CV2)
        # Rebuilds the SAME source+target with the new backend, in place.
        assert builds == [(Path("/dummy/source.jpg"), Path("/dummy/target.mp4"))]
        live.reconfigure_from.assert_called_once()
        live.stop.assert_not_called()  # executor is NOT torn down
        assert ctrl.video_backend() is VideoBackend.CV2
        ctrl.shutdown()

    def test_play_state_preserved_after_rebuild(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=True, current_frame=42)
        _, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.set_video_backend(VideoBackend.CV2)
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=42, play=True
        )
        ctrl.shutdown()

    def test_paused_state_preserved_after_rebuild(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=False, current_frame=10)
        _, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.set_video_backend(VideoBackend.CV2)
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=10, play=False
        )
        ctrl.shutdown()


class TestStructuralSettingsRebuildInPlace:
    """Reader-pool size and processing scale are structural (fresh reader pool /
    cache dir) but must rebuild via the in-place reconfigure path too, NOT
    teardown — so they don't churn worker threads and leak GPU memory."""

    def test_reader_pool_size_rebuilds_in_place(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=True, current_frame=7)
        builds, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.set_reader_pool_size(4)
        assert builds == [(Path("/dummy/source.jpg"), Path("/dummy/target.mp4"))]
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=7, play=True
        )
        live.stop.assert_not_called()
        assert ctrl.reader_pool_size() == 4
        ctrl.shutdown()

    def test_reader_pool_size_unchanged_is_no_op(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl)
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.set_reader_pool_size(ctrl.reader_pool_size())
        assert builds == []
        live.reconfigure_from.assert_not_called()
        ctrl.shutdown()

    def test_processing_scale_rebuilds_in_place(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=False, current_frame=3)
        builds, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.set_processing_scale(0.5)
        assert builds == [(Path("/dummy/source.jpg"), Path("/dummy/target.mp4"))]
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=3, play=False
        )
        live.stop.assert_not_called()
        assert ctrl.processing_scale() == 0.5
        ctrl.shutdown()

    def test_no_session_stores_value_only(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.set_processing_scale(0.25)
        assert builds == []  # no executor → just stored for next session
        assert ctrl.processing_scale() == 0.25
        ctrl.shutdown()
