"""Tests for the preprocessing orchestration controller."""
from __future__ import annotations

from types import SimpleNamespace

from sinner2.gui.preprocess_controller import PreprocessController
from sinner2.pipeline.cache_mode import CacheMode


class _FakeExecutor:
    """Records the orchestration calls; exposes the pollable state."""

    def __init__(self, frame_count: int, fps: float = 0.0, completed: int = -1):
        self._frame_count = frame_count
        self._fps = fps
        self.completed = completed
        self.calls: list = []
        self.processing_fps = SimpleNamespace(get=lambda: self._fps)

    def set_fps(self, fps: float) -> None:
        self._fps = fps

    def frame_count(self) -> int:
        return self._frame_count

    def last_completed_frame(self) -> int:
        return self.completed

    def set_skip_strategy(self, strategy) -> None:
        self.calls.append(("strategy", type(strategy).__name__))

    def set_cache_mode(self, mode) -> None:
        self.calls.append(("cache", mode))

    def seek(self, frame) -> None:
        self.calls.append(("seek", frame))

    def start_buffering(self) -> None:
        self.calls.append(("start_buffering",))

    def release_buffering(self, play: bool = True) -> None:
        self.calls.append(("release", play))


def _controller(executor, qtbot):
    ctrl = PreprocessController(get_executor=lambda: executor)
    qtbot.addWidget  # ensure the qtbot fixture (QApplication) is live
    ctrl._timer.stop()  # noqa: SLF001 — drive _tick manually in tests
    return ctrl


class TestStart:
    def test_forces_sequential_disk_fill_and_buffers(self, qtbot):
        ex = _FakeExecutor(frame_count=300)
        ctrl = _controller(ex, qtbot)
        with qtbot.waitSignal(ctrl.started, timeout=1000):
            ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        assert ("strategy", "BestEffortStrategy") in ex.calls
        assert ("cache", CacheMode.WRITE_READ) in ex.calls
        assert ("seek", 0) in ex.calls
        assert ("start_buffering",) in ex.calls
        assert ctrl.is_active()

    def test_no_executor_fails(self, qtbot):
        ctrl = PreprocessController(get_executor=lambda: None)
        with qtbot.waitSignal(ctrl.failed, timeout=1000):
            ctrl.start(target_fps=30.0)
        assert not ctrl.is_active()

    def test_empty_target_fails(self, qtbot):
        ex = _FakeExecutor(frame_count=0)
        ctrl = _controller(ex, qtbot)
        with qtbot.waitSignal(ctrl.failed, timeout=1000):
            ctrl.start(target_fps=30.0)
        assert not ctrl.is_active()

    def test_start_while_active_is_ignored(self, qtbot):
        ex = _FakeExecutor(frame_count=300)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        before = len(ex.calls)
        ctrl.start(target_fps=30.0)  # second call no-ops
        assert len(ex.calls) == before


class TestHeadStartRelease:
    def test_releases_when_head_start_reached(self, qtbot):
        # R=10, F=30, N=300 → B=200. Release once 200 frames are done.
        ex = _FakeExecutor(frame_count=300, fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001

        ex.completed = 100  # done=101 < 200 → keep buffering
        with qtbot.waitSignal(ctrl.progressChanged, timeout=1000) as blocker:
            ctrl._tick()  # noqa: SLF001
        assert blocker.args == [101, 200]
        assert ctrl.is_active()
        assert ("release", True) not in ex.calls

        ex.completed = 199  # done=200 >= 200 → release to play
        with qtbot.waitSignal(ctrl.finished, timeout=1000) as blocker:
            ctrl._tick()  # noqa: SLF001
        assert blocker.args == [True]
        assert ("release", True) in ex.calls
        assert not ctrl.is_active()

    def test_fast_pipeline_releases_early(self, qtbot):
        # R>=F → B=0 → release right after the warm-up frames.
        ex = _FakeExecutor(frame_count=300, fps=60.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 4  # done=5, past warm-up; B computes to 0
        with qtbot.waitSignal(ctrl.finished, timeout=1000):
            ctrl._tick()  # noqa: SLF001
        assert ("release", True) in ex.calls

    def test_releases_when_all_frames_done(self, qtbot):
        # Even with no throughput estimate, a fully-rendered clip releases.
        ex = _FakeExecutor(frame_count=5, fps=0.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 4  # done=5 == N
        with qtbot.waitSignal(ctrl.finished, timeout=1000):
            ctrl._tick()  # noqa: SLF001
        assert ("release", True) in ex.calls


class TestUserActions:
    def test_play_now_releases_to_play(self, qtbot):
        ex = _FakeExecutor(frame_count=300, fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        with qtbot.waitSignal(ctrl.finished, timeout=1000) as blocker:
            ctrl.play_now()
        assert blocker.args == [True]
        assert ("release", True) in ex.calls

    def test_cancel_releases_without_play(self, qtbot):
        ex = _FakeExecutor(frame_count=300, fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        with qtbot.waitSignal(ctrl.finished, timeout=1000) as blocker:
            ctrl.cancel()
        assert blocker.args == [False]
        assert ("release", False) in ex.calls
        assert not ctrl.is_active()

    def test_play_now_when_idle_is_noop(self, qtbot):
        ex = _FakeExecutor(frame_count=300)
        ctrl = _controller(ex, qtbot)
        ctrl.play_now()  # not active → nothing happens
        assert ex.calls == []
