"""Tests for the preprocessing orchestration controller."""
from __future__ import annotations

from types import SimpleNamespace

from sinner2.gui.preprocess_controller import PreprocessController
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    PredictiveStrategy,
    SyncedStrategy,
)


class _FakeExecutor:
    """Records the orchestration calls; exposes the pollable state."""

    def __init__(
        self, frame_count: int, fps: float = 0.0, face_fps: float = 0.0,
        completed: int = -1, current: int = 0,
    ):
        self._frame_count = frame_count
        self._fps = fps
        self._face_fps = face_fps
        self.completed = completed
        self.current = current  # the live playhead (moves on a section jump)
        self.calls: list = []
        self.processing_fps = SimpleNamespace(get=lambda: self._fps)
        self.current_frame = SimpleNamespace(get=lambda: self.current)

    def set_fps(self, fps: float) -> None:
        self._fps = fps

    def set_face_fps(self, fps: float) -> None:
        self._face_fps = fps

    def face_processing_fps(self) -> float:
        return self._face_fps

    def frame_count(self) -> int:
        return self._frame_count

    def last_completed_frame(self) -> int:
        return self.completed

    def preprocess_progress(self) -> tuple[int, int]:
        ahead = max(0, self.completed - self.current + 1)
        remaining = max(1, self._frame_count - self.current)
        return ahead, remaining

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
        assert ("start_buffering",) in ex.calls
        # Buffers from the current playhead — no seek to 0.
        assert not any(c[0] == "seek" for c in ex.calls)
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
        # Face R=10, F=30, M=300 → B=300-floor(300*10/30)=200.
        ex = _FakeExecutor(frame_count=300, face_fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001

        ex.completed = 100  # done_ahead=101 < 200 → keep buffering
        with qtbot.waitSignal(ctrl.progressChanged, timeout=1000) as blocker:
            ctrl._tick()  # noqa: SLF001
        assert blocker.args == [101, 200]
        assert ctrl.is_active()
        assert ("release", True) not in ex.calls

        ex.completed = 199  # done_ahead=200 >= 200 → release to play
        with qtbot.waitSignal(ctrl.finished, timeout=1000) as blocker:
            ctrl._tick()  # noqa: SLF001
        assert blocker.args == [True]
        assert ("release", True) in ex.calls
        assert not ctrl.is_active()

    def test_buffers_from_current_playhead(self, qtbot):
        # Start at frame 100 → M=200, face R=10, F=30 → B=200-floor(2000/30)=134.
        ex = _FakeExecutor(frame_count=300, face_fps=10.0, current=100)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 150  # done_ahead = 150-100+1 = 51 < 134 → keep
        with qtbot.waitSignal(ctrl.progressChanged, timeout=1000) as blocker:
            ctrl._tick()  # noqa: SLF001
        assert blocker.args == [51, 134]
        ex.completed = 233  # done_ahead = 134 >= 134 → release
        with qtbot.waitSignal(ctrl.finished, timeout=1000):
            ctrl._tick()  # noqa: SLF001
        assert ("release", True) in ex.calls

    def test_section_jump_not_misread_as_buffered(self, qtbot):
        # Regression: Play at frame 0, but the section fast-forward seeks to
        # frame 24240 → last_completed=24239 (seek sentinel) AND playhead=24240.
        # 'ahead' must read 0 (nothing actually rendered), NOT 24240 → must NOT
        # release immediately (which left 1-FPS playback with no real buffer).
        ex = _FakeExecutor(frame_count=59587, fps=21.74)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)  # captured at frame 0
        ctrl._timer.stop()  # noqa: SLF001
        ex.current = 24240   # playhead jumped onto the first section
        ex.completed = 24239  # seek high-water mark (section start - 1)
        ctrl._tick()  # noqa: SLF001
        assert ctrl.is_active()  # still buffering — not fooled into releasing
        assert ("release", True) not in ex.calls

    def test_sizes_off_face_rate_not_overall_rate(self, qtbot):
        # Overall pipeline looks fast (100 fps) but FACE frames are slow (10) →
        # must size off 10 (B=200), NOT release early on the inflated overall rate.
        ex = _FakeExecutor(frame_count=300, fps=100.0, face_fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 20  # done_ahead=21 << 200
        ctrl._tick()  # noqa: SLF001
        assert ctrl.is_active()  # still buffering — not fooled by the 100 fps
        assert ("release", True) not in ex.calls

    def test_fast_pipeline_releases_early(self, qtbot):
        # Face R>=F → B=0 → release right after the warm-up frames.
        ex = _FakeExecutor(frame_count=300, face_fps=60.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 4  # done_ahead=5, past warm-up; B computes to 0
        with qtbot.waitSignal(ctrl.finished, timeout=1000):
            ctrl._tick()  # noqa: SLF001
        assert ("release", True) in ex.calls

    def test_releases_when_all_frames_done(self, qtbot):
        # Even with no throughput estimate, a fully-rendered clip releases.
        ex = _FakeExecutor(frame_count=5)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 4  # completed >= N-1 → all done
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


class TestSparseWarmStart:
    """With a skip strategy active, buffering lays the SPARSE ladder (small fixed
    cushion) and restores that strategy on release — no longer locking the session
    on BestEffort, no minutes-long dense prebuffer."""

    def test_installs_buffering_strategy_for_skip_strategy(self, qtbot):
        ex = _FakeExecutor(frame_count=3000)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0, runtime_strategy=PredictiveStrategy())
        ctrl._timer.stop()  # noqa: SLF001
        assert ("strategy", "BufferingStrategy") in ex.calls
        assert ("strategy", "BestEffortStrategy") not in ex.calls
        assert ("cache", CacheMode.WRITE_READ) in ex.calls
        assert ("start_buffering",) in ex.calls

    def test_cushion_is_small_and_independent_of_clip_length(self, qtbot):
        # 2.5 s @ 30 fps = 75-frame cushion, regardless of the 3000-frame clip
        # (dense would demand B = 3000·(1−10/30) = 2000 frames up front).
        ex = _FakeExecutor(frame_count=3000, face_fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0, runtime_strategy=PredictiveStrategy())
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 40  # done_ahead=41 < 75 → keep buffering
        with qtbot.waitSignal(ctrl.progressChanged, timeout=1000) as blocker:
            ctrl._tick()  # noqa: SLF001
        assert blocker.args == [41, 75]
        assert ctrl.is_active()

    def test_restores_skip_strategy_before_release(self, qtbot):
        ex = _FakeExecutor(frame_count=3000, face_fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0, runtime_strategy=PredictiveStrategy())
        ctrl._timer.stop()  # noqa: SLF001
        ex.completed = 74  # done_ahead=75 >= 75 → release
        with qtbot.waitSignal(ctrl.finished, timeout=1000):
            ctrl._tick()  # noqa: SLF001
        assert ("strategy", "PredictiveStrategy") in ex.calls
        # Restore is ordered BEFORE release so playback resumes the ladder.
        strat_idx = ex.calls.index(("strategy", "PredictiveStrategy"))
        rel_idx = ex.calls.index(("release", True))
        assert strat_idx < rel_idx

    def test_cancel_also_restores_the_skip_strategy(self, qtbot):
        ex = _FakeExecutor(frame_count=3000, fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0, runtime_strategy=SyncedStrategy())
        ctrl._timer.stop()  # noqa: SLF001
        with qtbot.waitSignal(ctrl.finished, timeout=1000):
            ctrl.cancel()
        assert ("strategy", "SyncedStrategy") in ex.calls
        assert ("release", False) in ex.calls

    def test_best_effort_keeps_the_dense_path(self, qtbot):
        # Explicit BestEffort → dense full-rate render-ahead, no sparse ladder.
        ex = _FakeExecutor(frame_count=300, face_fps=10.0)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0, runtime_strategy=BestEffortStrategy())
        ctrl._timer.stop()  # noqa: SLF001
        assert ("strategy", "BestEffortStrategy") in ex.calls
        assert ("strategy", "BufferingStrategy") not in ex.calls
        ex.completed = 100  # dense B = 300−floor(300·10/30) = 200
        with qtbot.waitSignal(ctrl.progressChanged, timeout=1000) as blocker:
            ctrl._tick()  # noqa: SLF001
        assert blocker.args == [101, 200]

    def test_no_strategy_keeps_the_dense_path(self, qtbot):
        # Legacy call without a strategy → unchanged dense behavior.
        ex = _FakeExecutor(frame_count=300)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=30.0)
        ctrl._timer.stop()  # noqa: SLF001
        assert ("strategy", "BestEffortStrategy") in ex.calls
        assert ("strategy", "BufferingStrategy") not in ex.calls

    def test_unknown_fps_falls_back_to_dense(self, qtbot):
        # target_fps<=0 → no cushion can be sized → dense even with a skip strat.
        ex = _FakeExecutor(frame_count=300)
        ctrl = _controller(ex, qtbot)
        ctrl.start(target_fps=0.0, runtime_strategy=PredictiveStrategy())
        ctrl._timer.stop()  # noqa: SLF001
        assert ("strategy", "BestEffortStrategy") in ex.calls
        assert ("strategy", "BufferingStrategy") not in ex.calls
