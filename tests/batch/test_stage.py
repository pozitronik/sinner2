"""Tests for the processor-major stage runner."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from sinner2.batch.stage import (
    FramesDirInput,
    StageStatus,
    frame_ok,
    run_stage,
)
from sinner2.pipeline.image_writer import ImageFormat, build_image_writer
from sinner2.types import Frame


class _Pass:
    name = "Pass"

    def __init__(self) -> None:
        self.setup_calls = 0
        self.process_calls = 0
        self.release_calls = 0
        self._lock = threading.Lock()

    def setup(self) -> None:
        self.setup_calls += 1

    def process(self, frame: Frame) -> Frame:
        with self._lock:
            self.process_calls += 1
        return frame

    def release(self) -> None:
        self.release_calls += 1


class _FailAt(_Pass):
    """Raises on the frame whose pixel (0,0) equals `value`."""

    def __init__(self, value: int) -> None:
        super().__init__()
        self._value = value

    def process(self, frame: Frame) -> Frame:
        if int(frame[0, 0, 0]) == self._value:
            raise RuntimeError("boom")
        return super().process(frame)


class _SetupFails(_Pass):
    """setup() raises — simulates a worker instance that fails to load (e.g.
    the Nth GFPGAN OOMs while building a per-worker pool)."""

    def setup(self) -> None:
        super().setup()
        raise RuntimeError("setup boom")


class _ConcurrencyTracked(_Pass):
    """Records the max number of threads simultaneously inside process() for
    THIS instance, so a test can prove leasing keeps a non-thread-safe
    instance single-user while a shared instance is genuinely concurrent."""

    def __init__(self) -> None:
        super().__init__()
        self._active = 0
        self._guard = threading.Lock()
        self.max_active = 0

    def process(self, frame: Frame) -> Frame:
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.005)  # widen the window so real overlap is observable
        with self._guard:
            self._active -= 1
        return super().process(frame)


class _StubInput:
    """Frames encode their index in pixel (0,0). read() returns None for
    indices in `none_at`, simulating a decode gap."""

    def __init__(self, frame_count: int, none_at=()) -> None:
        self._n = frame_count
        self._none_at = set(none_at)

    @property
    def frame_count(self) -> int:
        return self._n

    def read(self, index):
        if index in self._none_at:
            return None
        return np.full((8, 8, 3), index % 256, dtype=np.uint8)

    def close(self) -> None:
        pass


class _ShortInput:
    """Claims `claimed` frames but read() returns None at/after `real` —
    simulates a video shorter than its nb_frames metadata."""

    def __init__(self, claimed: int, real: int) -> None:
        self._claimed = claimed
        self._real = real

    @property
    def frame_count(self) -> int:
        return self._claimed

    def read(self, index):
        if index >= self._real:
            return None
        return np.full((8, 8, 3), index % 256, dtype=np.uint8)

    def close(self) -> None:
        pass


@pytest.fixture
def writer():
    return build_image_writer(ImageFormat.JPEG, 80)


def _run(out, inp, proc, writer, **kw):
    return run_stage(
        stage_input=inp,
        processor_factory=lambda: proc,
        thread_safe=kw.get("thread_safe", True),
        output_dir=out,
        ext=writer.extension,
        writer=writer,
        workers=kw.get("workers", 2),
        pause_event=kw.get("pause", threading.Event()),
        cancel_event=kw.get("cancel", threading.Event()),
        on_progress=kw.get("on_progress"),
    )


class TestFrameOk:
    def test_nonempty_file_is_ok(self, tmp_path):
        p = tmp_path / "f.jpg"
        p.write_bytes(b"x")
        assert frame_ok(p)

    def test_missing_file_not_ok(self, tmp_path):
        assert not frame_ok(tmp_path / "nope.jpg")

    def test_zero_byte_file_not_ok(self, tmp_path):
        p = tmp_path / "z.jpg"
        p.write_bytes(b"")
        assert not frame_ok(p)


class TestRunStage:
    def test_runs_all_frames(self, tmp_path, writer):
        out = tmp_path / "stage0"
        proc = _Pass()
        res = _run(out, _StubInput(5), proc, writer)
        assert res.status is StageStatus.COMPLETED
        assert res.completed_frames == 5
        assert proc.process_calls == 5
        assert proc.setup_calls == 1
        assert proc.release_calls == 1
        assert len(list(out.glob(f"*.{writer.extension}"))) == 5

    def test_resume_skips_valid_outputs(self, tmp_path, writer):
        out = tmp_path / "stage0"
        out.mkdir()
        for i in (0, 1):  # pre-mark as done (non-empty)
            (out / f"{i:08d}.{writer.extension}").write_bytes(b"x")
        proc = _Pass()
        res = _run(out, _StubInput(5), proc, writer)
        assert res.status is StageStatus.COMPLETED
        assert proc.process_calls == 3  # only 2, 3, 4

    def test_zero_byte_output_reprocessed(self, tmp_path, writer):
        out = tmp_path / "stage0"
        out.mkdir()
        (out / f"{1:08d}.{writer.extension}").write_bytes(b"")  # corrupt
        proc = _Pass()
        res = _run(out, _StubInput(3), proc, writer)
        assert res.status is StageStatus.COMPLETED
        assert proc.process_calls == 3  # 1 was zero-byte → redone

    def test_persistent_input_gap_fails(self, tmp_path, writer):
        out = tmp_path / "stage0"
        res = _run(out, _StubInput(4, none_at=(2,)), _Pass(), writer)
        assert res.status is StageStatus.FAILED
        assert res.missing == [2]

    def test_processor_error_fails(self, tmp_path, writer):
        out = tmp_path / "stage0"
        res = _run(out, _StubInput(4), _FailAt(2), writer, workers=1)
        assert res.status is StageStatus.FAILED
        assert 2 in res.missing

    def test_pause_returns_paused_and_releases(self, tmp_path, writer):
        out = tmp_path / "stage0"
        pause = threading.Event()
        pause.set()  # paused before the first frame
        proc = _Pass()
        res = _run(out, _StubInput(5), proc, writer, pause=pause)
        assert res.status is StageStatus.PAUSED
        assert proc.process_calls == 0
        assert proc.release_calls == 1  # released even when interrupted

    def test_cancel_returns_cancelled(self, tmp_path, writer):
        out = tmp_path / "stage0"
        cancel = threading.Event()
        cancel.set()
        res = _run(out, _StubInput(5), _Pass(), writer, cancel=cancel)
        assert res.status is StageStatus.CANCELLED

    def test_progress_is_monotonic_and_reaches_total(self, tmp_path, writer):
        out = tmp_path / "stage0"
        seen: list[int] = []
        _run(out, _StubInput(4), _Pass(), writer, on_progress=seen.append)
        assert seen
        assert seen[-1] == 4
        assert seen == sorted(seen)

    def test_on_preview_receives_a_frame(self, tmp_path, writer):
        previews: list = []
        run_stage(
            stage_input=_StubInput(5),
            processor_factory=_Pass,
            thread_safe=True,
            output_dir=tmp_path / "stage0",
            ext=writer.extension,
            writer=writer,
            workers=2,
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
            on_preview=previews.append,
        )
        assert previews  # at least the first frame is previewed
        assert previews[0].shape == (8, 8, 3)


class TestEofTolerance:
    def test_eof_on_none_truncates_to_real_count(self, tmp_path, writer):
        out = tmp_path / "stage0"
        res = run_stage(
            stage_input=_ShortInput(claimed=10, real=7),
            processor_factory=_Pass,
            thread_safe=True,
            output_dir=out,
            ext=writer.extension,
            writer=writer,
            workers=2,
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
            eof_on_none=True,
        )
        assert res.status is StageStatus.COMPLETED
        assert res.total == 7  # discovered real count, not the claimed 10
        assert res.completed_frames == 7
        assert len(list(out.glob(f"*.{writer.extension}"))) == 7

    def test_without_eof_flag_trailing_none_fails(self, tmp_path, writer):
        out = tmp_path / "stage0"
        res = run_stage(
            stage_input=_ShortInput(claimed=10, real=7),
            processor_factory=_Pass,
            thread_safe=True,
            output_dir=out,
            ext=writer.extension,
            writer=writer,
            workers=2,
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
            eof_on_none=False,
        )
        assert res.status is StageStatus.FAILED
        assert res.missing == [7, 8, 9]


class TestFramesDirInput:
    def test_reads_written_frames_and_none_for_missing(self, tmp_path, writer):
        d = tmp_path / "frames"
        d.mkdir()
        writer.write(
            d / f"{0:08d}.{writer.extension}",
            np.full((8, 8, 3), 7, dtype=np.uint8),
        )
        inp = FramesDirInput(d, writer.extension, frame_count=2)
        assert inp.frame_count == 2
        assert inp.read(0) is not None
        assert inp.read(1) is None  # not written


class TestProcessorPool:
    """run_stage's instance strategy: one shared instance for thread-safe
    processors, one-per-worker (leased) for non-thread-safe ones."""

    def _run(self, tmp_path, writer, *, factory, thread_safe, workers, frames):
        return run_stage(
            stage_input=_StubInput(frames),
            processor_factory=factory,
            thread_safe=thread_safe,
            output_dir=tmp_path / "stage",
            ext=writer.extension,
            writer=writer,
            workers=workers,
            pause_event=threading.Event(),
            cancel_event=threading.Event(),
        )

    def test_thread_safe_builds_single_shared_instance(self, tmp_path, writer):
        built: list[_Pass] = []

        def factory() -> _Pass:
            p = _Pass()
            built.append(p)
            return p

        self._run(tmp_path, writer, factory=factory, thread_safe=True,
                  workers=4, frames=6)
        assert len(built) == 1  # shared, not one per worker
        assert built[0].process_calls == 6
        assert built[0].setup_calls == 1
        assert built[0].release_calls == 1

    def test_non_thread_safe_builds_one_instance_per_worker(self, tmp_path, writer):
        built: list[_Pass] = []

        def factory() -> _Pass:
            p = _Pass()
            built.append(p)
            return p

        self._run(tmp_path, writer, factory=factory, thread_safe=False,
                  workers=3, frames=6)
        assert len(built) == 3  # one per worker
        assert all(p.setup_calls == 1 for p in built)
        assert all(p.release_calls == 1 for p in built)
        assert sum(p.process_calls for p in built) == 6  # all frames covered

    def test_leased_instance_never_used_concurrently(self, tmp_path, writer):
        built: list[_ConcurrencyTracked] = []

        def factory() -> _ConcurrencyTracked:
            p = _ConcurrencyTracked()
            built.append(p)
            return p

        self._run(tmp_path, writer, factory=factory, thread_safe=False,
                  workers=3, frames=8)
        assert len(built) == 3
        # The point of one-instance-per-worker: each lease is exclusive.
        assert all(p.max_active == 1 for p in built)

    def test_shared_instance_is_used_concurrently(self, tmp_path, writer):
        # Counterpart to the leased test — proves the tracker isn't vacuous:
        # a thread-safe (shared) instance really does run on multiple workers
        # at once.
        built: list[_ConcurrencyTracked] = []

        def factory() -> _ConcurrencyTracked:
            p = _ConcurrencyTracked()
            built.append(p)
            return p

        self._run(tmp_path, writer, factory=factory, thread_safe=True,
                  workers=3, frames=8)
        assert len(built) == 1
        assert built[0].max_active >= 2

    def test_setup_failure_releases_already_built_instances(self, tmp_path, writer):
        built: list[_Pass] = []

        def factory() -> _Pass:
            # 2nd instance fails to set up; the 1st (already built) must be
            # released rather than leaked.
            p: _Pass = _SetupFails() if len(built) == 1 else _Pass()
            built.append(p)
            return p

        with pytest.raises(RuntimeError, match="setup boom"):
            self._run(tmp_path, writer, factory=factory, thread_safe=False,
                      workers=3, frames=4)
        assert built[0].release_calls == 1
