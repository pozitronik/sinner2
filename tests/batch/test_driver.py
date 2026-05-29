"""Tests for BatchDriver — processor-major staged execution with stubs.

We stub _build_stages so tests don't need real face-swap models (those take
seconds to load and need real images). The driver's orchestration is what's
exercised here: staged execution, per-stage resume/skip, pause, cancel,
stage-failure handling, and the encode/package path. The stage runner's own
behavior (integrity, single-thread reads) is covered in test_stage.py.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sinner2.batch.driver import BatchDriver
from sinner2.batch.task import (
    BatchCleanupMode,
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.io.video_encoder import FfmpegMissingError
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.types import Frame


class _PassthroughProcessor:
    """Returns the input frame unchanged; counts process() calls."""

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


class _SleepProcessor(_PassthroughProcessor):
    """Slow processor so pause/cancel windows are observable."""

    name = "Sleep"

    def __init__(self, seconds: float = 0.05) -> None:
        super().__init__()
        self._seconds = seconds

    def process(self, frame: Frame) -> Frame:
        time.sleep(self._seconds)
        return super().process(frame)


class _FailFrameProcessor(_PassthroughProcessor):
    """Raises on the frame whose pixel (0,0) equals fail_value."""

    name = "FailOne"

    def __init__(self, fail_value: int) -> None:
        super().__init__()
        self._fail_value = fail_value

    def process(self, frame: Frame) -> Frame:
        if int(frame[0, 0, 0]) == self._fail_value:
            raise RuntimeError("boom on target frame")
        return super().process(frame)


class _IndexEncodingReader:
    """Stub TargetReader whose frames encode their index in pixel (0,0)."""

    def __init__(self, frame_count: int) -> None:
        self._frame_count = frame_count

    @property
    def fps(self) -> float:
        return 5.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read(self, index: int):
        return np.full((8, 8, 3), index, dtype=np.uint8)

    def release(self) -> None:
        pass


class _ConcurrencyCheckingReader:
    """Flags overlapping read() calls — proves reads stay single-threaded."""

    def __init__(self, frame_count: int, on_overlap) -> None:
        self._frame_count = frame_count
        self._inside = 0
        self._guard = threading.Lock()
        self._on_overlap = on_overlap

    @property
    def fps(self) -> float:
        return 5.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read(self, index: int):
        with self._guard:
            self._inside += 1
            overlap = self._inside > 1
        if overlap:
            self._on_overlap()
        try:
            time.sleep(0.005)
            return np.full((8, 8, 3), index % 256, dtype=np.uint8)
        finally:
            with self._guard:
                self._inside -= 1

    def release(self) -> None:
        pass


def _make_image(path: Path, w: int = 16, h: int = 16) -> Path:
    Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8)).save(path)
    return path


def _make_video(path: Path, frames: int) -> Path:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 5.0, (16, 16))
    if not w.isOpened():
        pytest.skip("cv2 VideoWriter mp4v unavailable")
    try:
        for i in range(frames):
            w.write(np.full((16, 16, 3), i * 20, dtype=np.uint8))
    finally:
        w.release()
    return path


def _make_task(
    tmp_path: Path,
    *,
    image_target: bool = True,
    enhancer_enabled: bool = False,
    output_format: BatchOutputFormat = BatchOutputFormat.FRAMES,
) -> BatchTask:
    source = _make_image(tmp_path / "src.png")
    target = (
        _make_image(tmp_path / "tgt.png")
        if image_target
        else _make_video(tmp_path / "tgt.mp4", 3)
    )
    return BatchTask(
        source_path=source,
        target_path=target,
        output_path=tmp_path / "out",
        output_format=output_format,
        enhancer_enabled=enhancer_enabled,
        image_format=ImageFormat.JPEG,
        image_quality=80,
        worker_count=2,
    )


def _stage_dir(driver: BatchDriver, task: BatchTask, index: int, name: str):
    return driver._cache_root / task.id / f"stage{index}-{name}"  # noqa: SLF001


@pytest.fixture
def stub_stages(monkeypatch):
    """Patch _build_stages → passthrough stubs (one per real stage). Records
    every stub built across runs."""
    built: list[_PassthroughProcessor] = []

    def fake_build(_source, task):
        stages = [("faceswapper", _PassthroughProcessor())]
        if task.enhancer_enabled:
            stages.append(("faceenhancer", _PassthroughProcessor()))
        for _, p in stages:
            built.append(p)
        return stages

    monkeypatch.setattr(BatchDriver, "_build_stages", staticmethod(fake_build))
    return built


@pytest.fixture
def driver(tmp_path: Path) -> BatchDriver:
    return BatchDriver(cache_root=tmp_path / "cache")


class TestImageTarget:
    def test_single_image_runs_to_completion(
        self, driver, stub_stages, tmp_path
    ):
        task = _make_task(tmp_path, image_target=True)
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert task.last_completed_frame == 0
        assert task.total_frames == 1
        assert task.completed_stages == 1
        assert task.output_path.is_dir()
        assert len(list(task.output_path.glob("*.jpg"))) == 1

    def test_setup_release_process_called_once_per_stage(
        self, driver, stub_stages, tmp_path
    ):
        task = _make_task(tmp_path)
        driver.run(task)
        assert len(stub_stages) == 1
        p = stub_stages[0]
        assert p.setup_calls == 1
        assert p.release_calls == 1
        assert p.process_calls == 1


class TestVideoTarget:
    def test_three_frame_video_runs(self, driver, stub_stages, tmp_path):
        task = _make_task(tmp_path, image_target=False)
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert task.total_frames == 3
        assert task.last_completed_frame == 2
        assert len(list(task.output_path.glob("*.jpg"))) == 3


class TestProcessorMajor:
    def test_each_stage_processes_all_frames(
        self, driver, stub_stages, tmp_path
    ):
        # Two stages (swapper + enhancer). Each must process all 3 frames,
        # and (default Keep) both stage dirs are retained.
        task = _make_task(tmp_path, image_target=False, enhancer_enabled=True)
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert len(stub_stages) == 2
        assert stub_stages[0].process_calls == 3  # swapper over all frames
        assert stub_stages[1].process_calls == 3  # enhancer over all frames
        s0 = _stage_dir(driver, task, 0, "faceswapper")
        s1 = _stage_dir(driver, task, 1, "faceenhancer")
        assert len(list(s0.glob("*.jpg"))) == 3
        assert len(list(s1.glob("*.jpg"))) == 3
        # Output packaged from the LAST stage.
        assert len(list(task.output_path.glob("*.jpg"))) == 3


class TestResumeFromCache:
    def test_completed_stage_is_skipped_on_rerun(
        self, driver, stub_stages, tmp_path
    ):
        task = _make_task(tmp_path, image_target=False)
        driver.run(task)
        assert stub_stages[0].process_calls == 3
        # Delete output (not the cache); re-run must rebuild output WITHOUT
        # reprocessing — the stage is complete on disk.
        shutil.rmtree(task.output_path)
        stub_stages.clear()
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert stub_stages[0].process_calls == 0  # skipped entirely
        assert len(list(task.output_path.glob("*.jpg"))) == 3


class TestPauseCancel:
    def _video_task(self, tmp_path, frames):
        return BatchTask(
            source_path=_make_image(tmp_path / "src.png"),
            target_path=_make_video(tmp_path / "long.mp4", frames),
            output_path=tmp_path / "out",
            output_format=BatchOutputFormat.FRAMES,
            worker_count=1,
            image_format=ImageFormat.JPEG,
            image_quality=80,
        )

    def test_pause_stops_mid_task(self, tmp_path, monkeypatch):
        procs: list[_SleepProcessor] = []

        def fake_build(_source, task):
            p = _SleepProcessor(0.05)
            procs.append(p)
            return [("faceswapper", p)]

        monkeypatch.setattr(
            BatchDriver, "_build_stages", staticmethod(fake_build)
        )
        task = self._video_task(tmp_path, 10)
        driver = BatchDriver(cache_root=tmp_path / "cache")
        threading.Thread(
            target=lambda: (time.sleep(0.1), driver.pause()), daemon=True
        ).start()
        status = driver.run(task)
        assert status is BatchTaskStatus.PAUSED
        assert 0 < procs[0].process_calls < 10
        assert not task.output_path.exists()  # encode skipped

    def test_resume_after_pause_completes(self, tmp_path, monkeypatch):
        procs: list[_SleepProcessor] = []

        def fake_build(_source, task):
            p = _SleepProcessor(0.02)
            procs.append(p)
            return [("faceswapper", p)]

        monkeypatch.setattr(
            BatchDriver, "_build_stages", staticmethod(fake_build)
        )
        task = self._video_task(tmp_path, 8)
        driver = BatchDriver(cache_root=tmp_path / "cache")
        threading.Thread(
            target=lambda: (time.sleep(0.05), driver.pause()), daemon=True
        ).start()
        assert driver.run(task) is BatchTaskStatus.PAUSED
        first = procs[0].process_calls
        procs.clear()
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert procs[0].process_calls == 8 - first  # only the remainder
        assert len(list(task.output_path.glob("*.jpg"))) == 8

    def test_cancel_wipes_cache(self, tmp_path, monkeypatch):
        def fake_build(_source, task):
            return [("faceswapper", _SleepProcessor(0.03))]

        monkeypatch.setattr(
            BatchDriver, "_build_stages", staticmethod(fake_build)
        )
        task = self._video_task(tmp_path, 10)
        driver = BatchDriver(cache_root=tmp_path / "cache")
        threading.Thread(
            target=lambda: (time.sleep(0.05), driver.cancel()), daemon=True
        ).start()
        status = driver.run(task)
        assert status is BatchTaskStatus.CANCELLED
        task_cache = driver._cache_root / task.id  # noqa: SLF001
        assert list(task_cache.rglob("*.jpg")) == []  # whole cache wiped
        assert task.last_completed_frame == -1


class TestStageFailure:
    def test_persistent_failure_fails_task_keeps_frames(
        self, tmp_path, monkeypatch
    ):
        reader = _IndexEncodingReader(4)
        monkeypatch.setattr(
            BatchDriver,
            "_build_reader",
            staticmethod(lambda target, backend: reader),
        )
        monkeypatch.setattr(
            BatchDriver,
            "_build_stages",
            staticmethod(
                lambda _source, task: [("faceswapper", _FailFrameProcessor(2))]
            ),
        )
        task = _make_task(tmp_path, image_target=True)
        task.worker_count = 1
        driver = BatchDriver(cache_root=tmp_path / "cache")
        status = driver.run(task)
        assert status is BatchTaskStatus.FAILED
        assert "missing or empty" in (task.error_message or "")
        assert not task.output_path.exists()  # no truncated output
        s0 = _stage_dir(driver, task, 0, "faceswapper")
        assert len(list(s0.glob("*.jpg"))) == 3  # good frames kept


class TestFfmpegFallback:
    def test_missing_ffmpeg_falls_back_to_frames(
        self, tmp_path, stub_stages, monkeypatch
    ):
        import sinner2.batch.driver as driver_module

        monkeypatch.setattr(
            driver_module,
            "encode_frames_to_mp4",
            lambda *a, **k: (_ for _ in ()).throw(FfmpegMissingError("test")),
        )
        driver = BatchDriver(cache_root=tmp_path / "cache")
        task = _make_task(
            tmp_path, image_target=True, output_format=BatchOutputFormat.VIDEO
        )
        task.output_path = tmp_path / "out.mp4"
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert "fell back to frames mode" in (task.error_message or "")
        frames_dir = tmp_path / "out"  # .mp4 suffix stripped
        assert frames_dir.is_dir()
        assert len(list(frames_dir.glob("*.jpg"))) == 1


class TestReaderThreadSafety:
    def test_reads_stay_single_threaded(
        self, tmp_path, monkeypatch, stub_stages
    ):
        overlaps: list[int] = []
        reader = _ConcurrencyCheckingReader(6, lambda: overlaps.append(1))
        monkeypatch.setattr(
            BatchDriver,
            "_build_reader",
            staticmethod(lambda target, backend: reader),
        )
        task = _make_task(tmp_path, image_target=True)
        task.worker_count = 4  # would race if reads happened in workers
        driver = BatchDriver(cache_root=tmp_path / "cache")
        assert driver.run(task) is BatchTaskStatus.COMPLETED
        assert overlaps == [], "stage-0 reads overlapped"


class TestProgressCallback:
    def test_progress_reaches_total(self, driver, stub_stages, tmp_path):
        task = _make_task(tmp_path, image_target=False)
        events: list[tuple[int, int]] = []
        driver.run(task, progress_callback=lambda c, t: events.append((c, t)))
        assert events
        assert events[-1] == (3, 3)


class TestCleanupModes:
    def _two_stage(self, tmp_path, mode: BatchCleanupMode) -> BatchTask:
        task = _make_task(
            tmp_path, image_target=False, enhancer_enabled=True
        )
        task.cleanup_mode = mode
        return task

    def test_keep_retains_all_stage_dirs(self, driver, stub_stages, tmp_path):
        task = self._two_stage(tmp_path, BatchCleanupMode.KEEP)
        assert driver.run(task) is BatchTaskStatus.COMPLETED
        assert _stage_dir(driver, task, 0, "faceswapper").is_dir()
        assert _stage_dir(driver, task, 1, "faceenhancer").is_dir()
        assert len(list(task.output_path.glob("*.jpg"))) == 3

    def test_auto_removes_all_stage_dirs_keeps_output(
        self, driver, stub_stages, tmp_path
    ):
        task = self._two_stage(tmp_path, BatchCleanupMode.AUTO)
        assert driver.run(task) is BatchTaskStatus.COMPLETED
        assert not _stage_dir(driver, task, 0, "faceswapper").exists()
        assert not _stage_dir(driver, task, 1, "faceenhancer").exists()
        assert len(list(task.output_path.glob("*.jpg"))) == 3

    def test_drop_at_end_removes_all_stage_dirs_keeps_output(
        self, driver, stub_stages, tmp_path
    ):
        task = self._two_stage(tmp_path, BatchCleanupMode.DROP_AT_END)
        assert driver.run(task) is BatchTaskStatus.COMPLETED
        assert not _stage_dir(driver, task, 0, "faceswapper").exists()
        assert not _stage_dir(driver, task, 1, "faceenhancer").exists()
        assert len(list(task.output_path.glob("*.jpg"))) == 3
