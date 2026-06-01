"""Tests for PlayerController.change_source / change_target.

The QOL flow: when only the source path changes, the session rebuilds under the
hood but the current frame and play state are preserved; a target change resets
to frame 0 but submits it for processing immediately. The rebuild is now
ASYNCHRONOUS (the teardown can block on uninterruptible in-flight inference, so
it runs off the GUI thread) — these tests drive the swap inline via the
`_spawn_swap` injection point and stub the build/install so they exercise the
path-routing + state-restore logic without real sessions."""
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


def _make_controller(widgets) -> PlayerController:
    display, transport = widgets
    ctrl = PlayerController(frame_display=display, transport=transport)
    # Run the swap job inline so the async flow completes synchronously.
    ctrl._spawn_swap = lambda job: (job(), None)[1]  # noqa: SLF001
    return ctrl


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


def _stub_build(ctrl: PlayerController, monkeypatch) -> tuple[list, MagicMock]:
    """Stub _build_session to record (source, target) and return a bundle around
    a fresh fake executor; stub _install_session to just install it. Returns the
    build-call log and the new fake executor (restore targets this one)."""
    builds: list[tuple[Path, Path]] = []
    new_executor = MagicMock()

    def fake_build(source_path, target_path):
        builds.append((source_path, target_path))
        bundle = MagicMock()
        bundle.executor = new_executor
        return bundle

    monkeypatch.setattr(ctrl, "_build_session", fake_build)
    monkeypatch.setattr(
        ctrl, "_install_session",
        lambda bundle: setattr(ctrl, "_executor", bundle.executor),
    )
    return builds, new_executor


class TestChangeSourceNoOp:
    def test_no_op_without_session(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        assert builds == []
        ctrl.shutdown()

    def test_no_op_without_target_loaded(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        ctrl._executor = MagicMock()  # noqa: SLF001
        ctrl._current_target_path = None  # noqa: SLF001
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        assert builds == []
        ctrl.shutdown()


class TestChangeSourceWithSession:
    def test_triggers_rebuild_with_new_source_and_cached_target(
        self, widgets, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, current_frame=42)
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        assert builds == [(Path("/new/source.png"), Path("/dummy/target.mp4"))]
        ctrl.shutdown()

    def test_preserves_play_state_when_was_playing(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=True, current_frame=42)
        _, new_ex = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        new_ex.seek.assert_called_once_with(42)
        new_ex.play.assert_called_once_with()
        ctrl.shutdown()

    def test_paused_session_seeks_but_does_not_play(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=False, current_frame=10)
        _, new_ex = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        new_ex.seek.assert_called_once_with(10)
        new_ex.play.assert_not_called()
        ctrl.shutdown()

    def test_seeks_to_zero_when_current_frame_is_zero(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=False, current_frame=0)
        _, new_ex = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        new_ex.seek.assert_called_once_with(0)
        ctrl.shutdown()


class TestChangeTargetNoOp:
    def test_no_op_without_session(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.change_target(Path("/new/target.mp4"))
        assert builds == []
        ctrl.shutdown()

    def test_no_op_without_source_loaded(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        ctrl._executor = MagicMock()  # noqa: SLF001
        ctrl._current_source_path = None  # noqa: SLF001
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.change_target(Path("/new/target.mp4"))
        assert builds == []
        ctrl.shutdown()


class TestChangeTargetWithSession:
    def test_triggers_rebuild_with_cached_source_and_new_target(
        self, widgets, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, current_frame=42)
        builds, _ = _stub_build(ctrl, monkeypatch)
        ctrl.change_target(Path("/new/target.mp4"))
        assert builds == [(Path("/dummy/source.jpg"), Path("/new/target.mp4"))]
        ctrl.shutdown()

    def test_always_seeks_to_zero(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=False, current_frame=999)
        _, new_ex = _stub_build(ctrl, monkeypatch)
        ctrl.change_target(Path("/new/target.mp4"))
        new_ex.seek.assert_called_once_with(0)
        ctrl.shutdown()

    def test_preserves_play_state(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=True, current_frame=999)
        _, new_ex = _stub_build(ctrl, monkeypatch)
        ctrl.change_target(Path("/new/target.mp4"))
        new_ex.seek.assert_called_once_with(0)
        new_ex.play.assert_called_once_with()
        ctrl.shutdown()


class TestAudioRestoredOnSwap:
    """A source/target change must re-drive the audio backend to match the
    restored position + play state. The restore block resumes the EXECUTOR
    only; without _restore_audio_state the audio (paused in _detach_for_swap,
    reloaded in _install_session) stays silent until the user manually toggles
    play — the reported "sound disappears on source change" bug."""

    def _attach_audio(self, ctrl: PlayerController) -> MagicMock:
        audio = MagicMock()
        audio.is_loaded.return_value = True
        ctrl._audio_backend = audio  # noqa: SLF001
        ctrl._target_fps = 30.0  # noqa: SLF001
        return audio

    def test_resumes_audio_with_seek_when_playing(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=True, current_frame=60)
        _stub_build(ctrl, monkeypatch)
        audio = self._attach_audio(ctrl)
        ctrl.change_source(Path("/new/source.png"))
        audio.seek_seconds.assert_called_once_with(60 / 30.0)
        audio.play.assert_called_once_with()
        ctrl.shutdown()

    def test_keeps_audio_paused_when_not_playing(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=False, current_frame=10)
        _stub_build(ctrl, monkeypatch)
        audio = self._attach_audio(ctrl)
        ctrl.change_source(Path("/new/source.png"))
        audio.seek_seconds.assert_called_once_with(10 / 30.0)
        audio.play.assert_not_called()
        ctrl.shutdown()

    def test_target_change_seeks_audio_to_zero(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=True, current_frame=999)
        _stub_build(ctrl, monkeypatch)
        audio = self._attach_audio(ctrl)
        ctrl.change_target(Path("/new/target.mp4"))
        audio.seek_seconds.assert_called_once_with(0.0)
        audio.play.assert_called_once_with()
        ctrl.shutdown()


class TestAsyncSwapBehavior:
    def test_emits_session_switching_around_swap(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl)
        _stub_build(ctrl, monkeypatch)
        states: list[bool] = []
        ctrl.sessionSwitching.connect(states.append)
        ctrl.change_source(Path("/new/source.png"))
        assert states == [True, False]  # switching on, then off when ready
        assert ctrl._swapping is False  # noqa: SLF001
        ctrl.shutdown()

    def test_does_not_stop_executor_on_gui_thread(self, widgets, monkeypatch):
        # The whole point: the slow stop() must run in the swap job, not on the
        # calling (GUI) thread before the job is spawned.
        ctrl = _make_controller(widgets)
        old = _attach_fake_session(ctrl)
        _stub_build(ctrl, monkeypatch)
        spawned: list = []
        # Defer the job instead of running inline so we can observe ordering.
        ctrl._spawn_swap = lambda job: spawned.append(job) or None  # noqa: SLF001
        ctrl.change_source(Path("/new/source.png"))
        old.stop.assert_not_called()  # not stopped before the job runs
        assert len(spawned) == 1
        spawned[0]()  # run the deferred job → now the old executor is stopped
        old.stop.assert_called_once_with()
        ctrl.shutdown()

    def test_coalesces_rapid_swaps_latest_wins(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl)
        builds, _ = _stub_build(ctrl, monkeypatch)
        jobs: list = []
        ctrl._spawn_swap = lambda job: jobs.append(job) or None  # noqa: SLF001
        ctrl.change_source(Path("/a.png"))   # swap 1 starts (job deferred)
        assert ctrl._swapping is True  # noqa: SLF001
        ctrl.change_source(Path("/b.png"))   # arrives mid-swap → coalesced
        assert ctrl._swap_pending is not None  # noqa: SLF001
        assert len(jobs) == 1                 # second didn't spawn its own job
        jobs[0]()                             # finish swap 1 → dispatches pending
        assert len(jobs) == 2                 # pending (b) now running
        jobs[1]()
        assert builds[0][0] == Path("/a.png")
        assert builds[1][0] == Path("/b.png")
        ctrl.shutdown()
