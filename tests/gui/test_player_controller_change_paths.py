"""Tests for PlayerController.change_source / change_target.

The QOL flow: when only the source path changes, the running session is
re-pointed at the new source under the hood but the current frame and play
state are preserved; a target change resets to frame 0 but submits it for
processing immediately.

The rebuild is now done WITHOUT tearing down the executor — the live executor
ADOPTS the freshly built (unstarted) executor's world via reconfigure_from, so
its worker threads (and their ORT per-thread CUDA state) survive the swap. That
churn was what leaked GPU memory cycle after cycle. The teardown still runs off
the GUI thread (the reader probe is slow), so these tests drive the swap inline
via the `_spawn_swap` injection point and stub the build to exercise the
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
    ctrl = PlayerController(
        frame_display=display,
        transport=transport,
        # Keep real QtMultimedia out of unit tests — adopt() touches audio.
        audio_backend_factory=lambda name: MagicMock(),
    )
    # Run the swap job inline so the async flow completes synchronously.
    ctrl._swap.spawn = lambda job: (job(), None)[1]  # noqa: SLF001
    return ctrl


def _attach_fake_session(
    ctrl: PlayerController, *, playing: bool = False, current_frame: int = 0
) -> MagicMock:
    """Attach a fake LIVE executor. reconfigure_from returns a displaced
    (reader_pool, buffer) tuple so the swap job's off-thread shutdown runs."""
    ctrl._current_source_path = Path("/dummy/source.jpg")  # noqa: SLF001
    ctrl._current_target_path = Path("/dummy/target.mp4")  # noqa: SLF001
    fake_executor = MagicMock()
    fake_executor.is_playing.get.return_value = playing
    fake_executor.current_frame.get.return_value = current_frame
    fake_executor.reconfigure_from.return_value = (MagicMock(), MagicMock())
    ctrl._executor = fake_executor  # noqa: SLF001
    return fake_executor


def _make_bundle(unstarted: MagicMock) -> MagicMock:
    """A session bundle around the UNSTARTED executor handed to reconfigure_from,
    with realistic numeric fields so _adopt_swapped_bundle's audio/transport math
    works."""
    bundle = MagicMock()
    bundle.executor = unstarted
    bundle.source = MagicMock()
    bundle.source_path = Path("/dummy/source.jpg")
    bundle.target_path = Path("/dummy/target.mp4")
    bundle.target_fps = 30.0
    bundle.frame_count = 100
    bundle.native_size = (1920, 1080)
    bundle.cache_dir = Path("/tmp/cache")
    bundle.write_executor = None
    bundle.session_store = None
    return bundle


def _stub_build(ctrl: PlayerController, monkeypatch) -> tuple[list, MagicMock]:
    """Stub _build_session to record (source, target) and return a bundle around
    a fresh UNSTARTED executor. Returns the build-call log and that unstarted
    executor (reconfigure_from must receive it)."""
    builds: list[tuple[Path, Path]] = []
    unstarted = MagicMock(name="unstarted_executor")

    def fake_build(source_path, target_path):
        builds.append((source_path, target_path))
        return _make_bundle(unstarted)

    monkeypatch.setattr(ctrl, "_build_session", fake_build)
    return builds, unstarted


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

    def test_adopts_new_world_preserving_play_state(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=True, current_frame=42)
        _, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=42, play=True
        )
        live.stop.assert_not_called()  # the whole point: executor is NOT torn down
        ctrl.shutdown()

    def test_paused_session_restores_frame_without_playing(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=False, current_frame=10)
        _, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=10, play=False
        )
        ctrl.shutdown()

    def test_restores_frame_zero_when_current_frame_is_zero(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=False, current_frame=0)
        _, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.change_source(Path("/new/source.png"))
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=0, play=False
        )
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

    def test_always_restores_frame_zero(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=False, current_frame=999)
        _, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.change_target(Path("/new/target.mp4"))
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=0, play=False
        )
        ctrl.shutdown()

    def test_preserves_play_state(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl, playing=True, current_frame=999)
        _, unstarted = _stub_build(ctrl, monkeypatch)
        ctrl.change_target(Path("/new/target.mp4"))
        live.reconfigure_from.assert_called_once_with(
            unstarted, restore_frame=0, play=True
        )
        ctrl.shutdown()


class TestAudioRestoredOnSwap:
    """A source/target change must re-drive the audio backend to match the
    restored position + play state. reconfigure resumes only the EXECUTOR;
    without _restore_audio_state the audio (paused at swap start, reloaded in
    _adopt_swapped_bundle) stays silent until the user manually toggles play —
    the reported "sound disappears on source change" bug."""

    def _attach_audio(self, ctrl: PlayerController) -> MagicMock:
        audio = MagicMock()
        audio.is_loaded.return_value = True
        ctrl._audio._backend = audio  # noqa: SLF001  (inject a loaded backend)
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

    def test_source_change_reloads_audio_backend_to_rearm(self, widgets, monkeypatch):
        # A source-only swap leaves the target unchanged, so load() no-ops; the
        # adopt path must force reload() to re-arm the deferred play/seek,
        # otherwise the resume is a bare play() on a just-paused player that MF
        # silently drops (audio dies on source change until manual restart).
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl, playing=True, current_frame=60)
        _stub_build(ctrl, monkeypatch)
        audio = self._attach_audio(ctrl)
        ctrl.change_source(Path("/new/source.png"))
        audio.reload.assert_called_once_with()
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
        assert ctrl._swap.swapping is False  # noqa: SLF001
        ctrl.shutdown()

    def test_does_not_touch_executor_until_job_runs(self, widgets, monkeypatch):
        # The whole point: the live executor must not be reconfigured (or
        # stopped) on the calling (GUI) thread before the deferred job runs.
        ctrl = _make_controller(widgets)
        live = _attach_fake_session(ctrl)
        _stub_build(ctrl, monkeypatch)
        spawned: list = []
        ctrl._swap.spawn = lambda job: spawned.append(job) or None  # noqa: SLF001
        ctrl.change_source(Path("/new/source.png"))
        live.reconfigure_from.assert_not_called()  # nothing before the job runs
        live.stop.assert_not_called()
        assert len(spawned) == 1
        spawned[0]()  # run the deferred job → now the live executor adopts
        live.reconfigure_from.assert_called_once()
        live.stop.assert_not_called()  # never torn down
        ctrl.shutdown()

    def test_coalesces_rapid_swaps_latest_wins(self, widgets, monkeypatch):
        ctrl = _make_controller(widgets)
        _attach_fake_session(ctrl)
        builds, _ = _stub_build(ctrl, monkeypatch)
        jobs: list = []
        ctrl._swap.spawn = lambda job: jobs.append(job) or None  # noqa: SLF001
        ctrl.change_source(Path("/a.png"))   # swap 1 starts (job deferred)
        assert ctrl._swap.swapping is True  # noqa: SLF001
        ctrl.change_source(Path("/b.png"))   # arrives mid-swap → coalesced
        assert ctrl._swap._pending is not None  # noqa: SLF001
        assert len(jobs) == 1                 # second didn't spawn its own job
        jobs[0]()                             # finish swap 1 → dispatches pending
        assert len(jobs) == 2                 # pending (b) now running
        jobs[1]()
        assert builds[0][0] == Path("/a.png")
        assert builds[1][0] == Path("/b.png")
        ctrl.shutdown()
