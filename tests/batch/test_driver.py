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

from sinner2.batch.driver import BatchDriver, StageSpec
from sinner2.batch.task import (
    BatchCleanupMode,
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.config.execution import OnnxExecution
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

    @property
    def width(self) -> int:
        return 8

    @property
    def height(self) -> int:
        return 8

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

    @property
    def width(self) -> int:
        return 8

    @property
    def height(self) -> int:
        return 8

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


class _ShortReader:
    """Reports `claimed` frames but read() returns None at/after `real` —
    the ffprobe-over-counts-nb_frames case."""

    def __init__(self, claimed: int, real: int) -> None:
        self._claimed = claimed
        self._real = real

    @property
    def fps(self) -> float:
        return 25.0

    @property
    def frame_count(self) -> int:
        return self._claimed

    @property
    def width(self) -> int:
        return 8

    @property
    def height(self) -> int:
        return 8

    def read(self, index: int):
        if index >= self._real:
            return None
        return np.full((8, 8, 3), index % 256, dtype=np.uint8)

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
    )


def _stage_dir(driver: BatchDriver, task: BatchTask, index: int, name: str):
    # Stage dirs now carry a scaled-size token ("stage0-faceswapper@WxH"); the
    # WxH depends on the stub reader, so resolve by prefix glob. Falls back to
    # the token-less path (which won't exist) so "not exists" assertions hold.
    parent = driver._cache_root / task.id  # noqa: SLF001
    matches = list(parent.glob(f"stage{index}-{name}@*"))
    return matches[0] if matches else parent / f"stage{index}-{name}"


@pytest.fixture
def stub_stages(monkeypatch):
    """Patch _build_stages → passthrough stubs (one per real stage). Records
    every stub built across runs."""
    built: list[_PassthroughProcessor] = []

    def fake_build(_source, task):
        # Build instances eagerly and record them so tests can inspect
        # setup/process/release counts after the run; the factory just hands
        # back the pre-built instance (thread_safe=True → built once).
        names = ["faceswapper"]
        if task.enhancer_enabled:
            names.append("faceenhancer")
        stages = []
        for name in names:
            p = _PassthroughProcessor()
            built.append(p)
            workers = (
                task.enhancer_execution.workers
                if name == "faceenhancer"
                else task.swapper_execution.workers
            )
            stages.append(
                StageSpec(name=name, factory=lambda _p=p: _p,
                          thread_safe=True, workers=workers)
            )
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
            swapper_execution=OnnxExecution(workers=1),
            image_format=ImageFormat.JPEG,
            image_quality=80,
        )

    def test_pause_stops_mid_task(self, tmp_path, monkeypatch):
        procs: list[_SleepProcessor] = []

        def fake_build(_source, task):
            p = _SleepProcessor(0.05)
            procs.append(p)
            return [
                StageSpec("faceswapper", lambda _p=p: _p, True,
                          task.swapper_execution.workers)
            ]

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
            return [
                StageSpec("faceswapper", lambda _p=p: _p, True,
                          task.swapper_execution.workers)
            ]

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
            return [
                StageSpec("faceswapper", lambda: _SleepProcessor(0.03), True,
                          task.swapper_execution.workers)
            ]

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
            staticmethod(lambda target, backend, scale=1.0: reader),
        )
        monkeypatch.setattr(
            BatchDriver,
            "_build_stages",
            staticmethod(
                lambda _source, task: [
                    StageSpec("faceswapper", lambda: _FailFrameProcessor(2),
                              True, task.swapper_execution.workers)
                ]
            ),
        )
        task = _make_task(tmp_path, image_target=True)
        task.swapper_execution.workers = 1
        driver = BatchDriver(cache_root=tmp_path / "cache")
        status = driver.run(task)
        assert status is BatchTaskStatus.FAILED
        assert "missing or empty" in (task.error_message or "")
        # The underlying cause is surfaced, not just the symptom — so the user
        # sees "RuntimeError: boom ..." rather than a bare frame-count failure.
        assert "boom on target frame" in (task.error_message or "")
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
            staticmethod(lambda target, backend, scale=1.0: reader),
        )
        task = _make_task(tmp_path, image_target=True)
        task.swapper_execution.workers = 4  # would race if reads happened in workers
        driver = BatchDriver(cache_root=tmp_path / "cache")
        assert driver.run(task) is BatchTaskStatus.COMPLETED
        assert overlaps == [], "stage-0 reads overlapped"


class TestPreview:
    def test_preview_callback_receives_frames(
        self, driver, stub_stages, tmp_path
    ):
        previews: list = []
        task = _make_task(tmp_path, image_target=False)  # 3-frame video
        driver.run(task, preview_callback=previews.append)
        assert previews  # at least one processed frame surfaced


class TestProgressCallback:
    def test_progress_reports_overall_frame_units(
        self, driver, stub_stages, tmp_path
    ):
        # Two processor stages + the combine step × 3 frames = 9 overall units;
        # final event hits 9/9 and the overall count is monotonic and bounded.
        task = _make_task(tmp_path, image_target=False, enhancer_enabled=True)
        events = []
        driver.run(task, progress_callback=events.append)
        assert events
        last = events[-1]
        assert last.stage_count == 3  # swapper + enhancer + combine
        assert last.overall_total == 9
        assert last.overall_completed == 9
        overall = [e.overall_completed for e in events]
        assert overall == sorted(overall)
        assert all(0 <= c <= 9 for c in overall)

    def test_combine_step_is_reported_as_final_stage(
        self, driver, stub_stages, tmp_path
    ):
        # The combine/encode step surfaces as a distinct trailing stage so the
        # bar doesn't freeze at the last processor stage while packaging runs.
        task = _make_task(tmp_path, image_target=False)  # 1 processor stage
        events = []
        driver.run(task, progress_callback=events.append)
        names = {e.stage_name for e in events}
        assert "copy" in names  # FRAMES output → combine step is a copy
        last = events[-1]
        assert last.stage_index == 1  # combine is the stage after the processor
        assert last.stage_name == "copy"

    def test_single_stage_overall(self, driver, stub_stages, tmp_path):
        task = _make_task(tmp_path, image_target=False)  # 1 stage, 3 frames
        events = []
        driver.run(task, progress_callback=events.append)
        last = events[-1]
        # 1 processor stage + the combine step = 2 stages; the last event is
        # the combine step at 3/3 → overall 6/6.
        assert last.stage_index == 1
        assert last.overall_total == 6
        assert last.overall_completed == 6


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


class TestOverReportedFrameCount:
    def test_small_shortfall_completes_at_real_count(
        self, driver, stub_stages, tmp_path, monkeypatch
    ):
        # Metadata claims 10 frames; the stream only decodes 8. Must complete
        # with the real count, not fail on the 2 phantom frames.
        reader = _ShortReader(claimed=10, real=8)
        monkeypatch.setattr(
            BatchDriver,
            "_build_reader",
            staticmethod(lambda target, backend, scale=1.0: reader),
        )
        task = _make_task(tmp_path, image_target=True)
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert task.total_frames == 8  # corrected down from 10
        assert len(list(task.output_path.glob("*.jpg"))) == 8

    def test_large_shortfall_fails_loudly(
        self, driver, stub_stages, tmp_path, monkeypatch
    ):
        # Decoding only 10 of a claimed 100 is not a metadata glitch — fail.
        reader = _ShortReader(claimed=100, real=10)
        monkeypatch.setattr(
            BatchDriver,
            "_build_reader",
            staticmethod(lambda target, backend, scale=1.0: reader),
        )
        task = _make_task(tmp_path, image_target=True)
        status = driver.run(task)
        assert status is BatchTaskStatus.FAILED
        assert "truncated" in (task.error_message or "")

    def test_resume_uses_persisted_real_length_not_container(
        self, driver, stub_stages, tmp_path, monkeypatch
    ):
        # A 2-stage AUTO task whose stage 0 already ran: the real decoded length
        # (8) was persisted + stage 0 marked done. On resume the container still
        # over-reports 10. Stage 0 is skipped (so its EOF correction does NOT
        # re-run), so the driver MUST use the persisted real length — else stage
        # 1 looks for 10 frames in an 8-frame dir and fails "frames missing".
        import cv2

        reader = _ShortReader(claimed=10, real=8)
        monkeypatch.setattr(
            BatchDriver,
            "_build_reader",
            staticmethod(lambda target, backend, scale=1.0: reader),
        )
        task = _make_task(tmp_path, image_target=False, enhancer_enabled=True)
        task.cleanup_mode = BatchCleanupMode.AUTO
        task.total_frames = 8     # persisted real length from the prior run
        task.completed_stages = 1  # stage 0 already done
        task_cache = driver._cache_root / task.id  # noqa: SLF001
        stages = driver._build_stages(None, task)  # stubbed passthrough stubs  # noqa: SLF001
        size_token = f"{reader.width}x{reader.height}"
        stage0 = BatchDriver._stage_cache_dirs(  # noqa: SLF001
            task_cache, size_token, task, stages
        )[0]
        stage0.mkdir(parents=True)
        for i in range(8):
            cv2.imwrite(str(stage0 / f"{i:08d}.jpg"), np.full((8, 8, 3), i, np.uint8))
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED, task.error_message
        assert task.total_frames == 8


class TestBuildStages:
    @staticmethod
    def _stub_procs(monkeypatch):
        import sinner2.batch.driver as drv

        class _S:
            name = "faceswapper"
            thread_safe = True

            def __init__(self, source, params):
                ...

        class _E:
            name = "faceenhancer"
            thread_safe = False

            def __init__(self, params):
                ...

        monkeypatch.setattr(drv, "FaceSwapper", _S)
        monkeypatch.setattr(drv, "FaceEnhancer", _E)

    def test_both_enabled(self, tmp_path, monkeypatch):
        from sinner2.config.source import Source

        self._stub_procs(monkeypatch)
        task = _make_task(tmp_path, enhancer_enabled=True)
        task.swapper_enabled = True
        stages = BatchDriver._build_stages(  # noqa: SLF001
            Source(path=task.source_path), task
        )
        assert [s.name for s in stages] == ["faceswapper", "faceenhancer"]

    def test_swapper_disabled_enhancer_only(self, tmp_path, monkeypatch):
        from sinner2.config.source import Source

        self._stub_procs(monkeypatch)
        task = _make_task(tmp_path, enhancer_enabled=True)
        task.swapper_enabled = False
        stages = BatchDriver._build_stages(  # noqa: SLF001
            Source(path=task.source_path), task
        )
        assert [s.name for s in stages] == ["faceenhancer"]

    def test_both_disabled_is_passthrough(self, tmp_path, monkeypatch):
        from sinner2.config.source import Source

        self._stub_procs(monkeypatch)
        task = _make_task(tmp_path, enhancer_enabled=False)
        task.swapper_enabled = False
        stages = BatchDriver._build_stages(  # noqa: SLF001
            Source(path=task.source_path), task
        )
        assert [s.name for s in stages] == ["passthrough"]

    def test_both_disabled_runs_to_passthrough_output(self, driver, tmp_path):
        # End-to-end (no stub): the identity passthrough re-encodes the
        # source frames unchanged — no face models loaded.
        task = _make_task(tmp_path, image_target=False)  # 3-frame video
        task.swapper_enabled = False
        task.enhancer_enabled = False
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert len(list(task.output_path.glob("*.jpg"))) == 3


class TestStageCacheKeying:
    """Stage dirs are keyed by stage position/name + size + task IDENTITY
    (source/target) — deliberately NOT by stage params. A settings edit must
    resume IN PLACE (frames already rendered are kept; only remaining frames
    use the new config — the user chose continuation over output purity).
    Identity and size changes still re-render."""

    def _stages(self, names=("stagea", "stageb")):
        return [StageSpec(n, lambda: None, True, 1) for n in names]

    def test_settings_edit_keeps_stage_dirs(self, tmp_path):
        # Two configs of the SAME task (e.g. enhancer model switched mid-run)
        # must map to the SAME dirs — that's what lets the stage continue
        # instead of restarting from frame 0 in a fresh dir.
        cache = tmp_path / "c"
        t1 = BatchTask(
            source_path=tmp_path / "s.png", target_path=tmp_path / "t.mp4",
            enhancer_model="gfpgan",
        )
        t2 = BatchTask(
            source_path=tmp_path / "s.png", target_path=tmp_path / "t.mp4",
            enhancer_model="gfpgan_onnx",
        )
        stages = self._stages()
        d1 = BatchDriver._stage_cache_dirs(cache, "640x480", t1, stages)  # noqa: SLF001
        d2 = BatchDriver._stage_cache_dirs(cache, "640x480", t2, stages)  # noqa: SLF001
        assert d1 == d2

    def test_source_change_invalidates_all_stages(self, tmp_path):
        cache = tmp_path / "c"
        t1 = BatchTask(source_path=tmp_path / "a.png", target_path=tmp_path / "t.mp4")
        t2 = BatchTask(source_path=tmp_path / "b.png", target_path=tmp_path / "t.mp4")
        stages = self._stages()
        d1 = BatchDriver._stage_cache_dirs(cache, "640x480", t1, stages)  # noqa: SLF001
        d2 = BatchDriver._stage_cache_dirs(cache, "640x480", t2, stages)  # noqa: SLF001
        assert d1[0] != d2[0]
        assert d1[1] != d2[1]

    def test_target_change_invalidates_all_stages(self, tmp_path):
        cache = tmp_path / "c"
        t1 = BatchTask(source_path=tmp_path / "s.png", target_path=tmp_path / "a.mp4")
        t2 = BatchTask(source_path=tmp_path / "s.png", target_path=tmp_path / "b.mp4")
        stages = self._stages()
        d1 = BatchDriver._stage_cache_dirs(cache, "640x480", t1, stages)  # noqa: SLF001
        d2 = BatchDriver._stage_cache_dirs(cache, "640x480", t2, stages)  # noqa: SLF001
        assert d1[0] != d2[0]
        assert d1[1] != d2[1]

    def test_stage_position_and_name_distinguish_dirs(self, tmp_path):
        # Enabling/disabling processors shifts stage layout; position+name in
        # the dir name keeps a stage from ever reading another stage's frames.
        task = BatchTask(
            source_path=tmp_path / "s.png", target_path=tmp_path / "t.mp4"
        )
        dirs = BatchDriver._stage_cache_dirs(  # noqa: SLF001
            tmp_path / "c", "640x480", task, self._stages(("faceswapper", "upscaler"))
        )
        assert "stage0-faceswapper" in dirs[0].name
        assert "stage1-upscaler" in dirs[1].name

    def test_chain_fingerprint_tracks_identity_not_config(self, tmp_path):
        t1 = BatchTask(
            source_path=tmp_path / "s.png", target_path=tmp_path / "t.mp4",
            enhancer_model="gfpgan",
        )
        t_cfg = BatchTask(
            source_path=tmp_path / "s.png", target_path=tmp_path / "t.mp4",
            enhancer_model="gfpgan_onnx",
        )
        t_src = BatchTask(
            source_path=tmp_path / "other.png", target_path=tmp_path / "t.mp4",
        )
        fp1 = BatchDriver._chain_fingerprint(t1, "640x480")  # noqa: SLF001
        # A settings edit keeps the fingerprint → resume markers stay valid.
        assert BatchDriver._chain_fingerprint(t_cfg, "640x480") == fp1  # noqa: SLF001
        # Identity / size changes reset it → markers re-derived from scratch.
        assert BatchDriver._chain_fingerprint(t_src, "640x480") != fp1  # noqa: SLF001
        assert BatchDriver._chain_fingerprint(t1, "320x240") != fp1  # noqa: SLF001


class TestAutoResumeStaleFingerprint:
    def test_auto_resume_with_changed_identity_re_renders(
        self, driver, stub_stages, tmp_path
    ):
        # AUTO task whose persisted markers (completed_stages=1) belong to a
        # DIFFERENT identity (stale fingerprint — e.g. the source was swapped).
        # On resume the token changed, so the trusted stage-0 marker must be
        # invalidated and the task re-rendered — not trusted into reading an
        # empty new-token dir and failing "frames missing".
        task = _make_task(tmp_path, image_target=False, enhancer_enabled=True)
        task.cleanup_mode = BatchCleanupMode.AUTO
        task.completed_stages = 1
        task.cache_fingerprint = "stale-from-old-identity"
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED, task.error_message
        assert len(list(task.output_path.glob("*.jpg"))) == 3


class TestSectionSelection:
    """A section selection processes ONLY the selected frames and renumbers them
    contiguous in the output (a multi-range trim)."""

    def _read_means(self, out_dir: Path) -> list[float]:
        import cv2

        means = []
        for p in sorted(out_dir.glob("*")):
            img = cv2.imread(str(p))
            means.append(float(np.mean(img)))
        return means

    def test_processes_only_selected_frames(self, driver, stub_stages, tmp_path):
        # 6-frame video; frame i is a uniform i*20 image. Select [1,2] and [4,5].
        video = _make_video(tmp_path / "clip.mp4", 6)
        out = tmp_path / "trimmed"
        task = BatchTask(
            source_path=_make_image(tmp_path / "src.png"),
            target_path=video,
            output_path=out,
            output_format=BatchOutputFormat.FRAMES,
            image_format=ImageFormat.JPEG,
            image_quality=95,
            sections=[[1, 2], [4, 5]],
        )
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        # Plan = [1,2,4,5] → 4 output frames, renumbered 0..3.
        assert task.total_frames == 4
        means = self._read_means(out)
        assert len(means) == 4
        # Output frame 0 is source frame 1 (~20), not the EXCLUDED frame 0 (~0).
        assert means[0] > 8
        # Ascending across the selection (frame 5 ~100 is the brightest).
        assert means[-1] > means[0]

    def test_no_sections_processes_all_frames(self, driver, stub_stages, tmp_path):
        video = _make_video(tmp_path / "clip.mp4", 6)
        out = tmp_path / "full"
        task = BatchTask(
            source_path=_make_image(tmp_path / "src.png"),
            target_path=video,
            output_path=out,
            output_format=BatchOutputFormat.FRAMES,
            image_format=ImageFormat.JPEG,
            image_quality=95,
        )
        driver.run(task)
        assert task.total_frames == 6
        assert len(list(out.glob("*"))) == 6

    def test_changing_sections_changes_fingerprint(self, tmp_path):
        base = dict(
            source_path=tmp_path / "s.png",
            target_path=tmp_path / "t.mp4",
        )
        a = BatchTask(**base, sections=[[1, 2]])
        b = BatchTask(**base, sections=[[3, 4]])
        c = BatchTask(**base)  # no sections
        fp = BatchDriver._chain_fingerprint  # noqa: SLF001
        assert fp(a, "16x16") != fp(b, "16x16")
        assert fp(a, "16x16") != fp(c, "16x16")


class TestResolveFaceMap:
    """The face map is loaded LIVE at render time from the per-target sidecars
    (the GUI stamps the store dir), so a re-scan/edit is reflected in queued
    renders. Routing-off / no-map / no-store degrade to the global source."""

    @staticmethod
    def _task(tmp_path, store=None, face_map=None):
        return BatchTask(
            source_path=tmp_path / "s.png",
            target_path=tmp_path / "t.mp4",
            face_map_store_dir=str(store) if store else None,
            face_map=face_map,
        )

    @staticmethod
    def _save_catalog(tmp_path, store, *, use):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.face_map_store import (
            face_map_path, save_face_map, save_use_map, use_map_path,
        )
        fm = FaceMap(identities=(
            Identity("a", normalize([1.0, 0.0, 0.0]), source_path="/s.png"),
        ))
        save_face_map(face_map_path(tmp_path / "t.mp4", store), fm)
        save_use_map(use_map_path(tmp_path / "t.mp4", store), use)
        return fm

    def test_live_load_when_routing_on(self, tmp_path):
        from sinner2.batch.driver import _resolve_face_map  # noqa: SLF001
        store = tmp_path / "face_maps"
        self._save_catalog(tmp_path, store, use=True)
        fm, geom = _resolve_face_map(self._task(tmp_path, store=store))
        assert fm is not None and len(fm.identities) == 1
        assert geom is None  # no geometry NPZ saved

    def test_loads_geometry_when_present(self, tmp_path):
        from sinner2.batch.driver import _resolve_face_map  # noqa: SLF001
        from sinner2.pipeline.face_map_geometry import (
            FrameGeometry, GeomFace, geometry_path, save_geometry,
        )
        store = tmp_path / "face_maps"
        self._save_catalog(tmp_path, store, use=True)
        kps = tuple((float(i), 0.0) for i in range(5))
        save_geometry(
            geometry_path(tmp_path / "t.mp4", store),
            FrameGeometry(faces={0: (GeomFace("a", (0., 0., 4., 4.), kps),)},
                          frame_count=1),
        )
        fm, geom = _resolve_face_map(self._task(tmp_path, store=store))
        assert fm is not None and geom is not None and not geom.is_empty()

    def test_routing_off_uses_global_source(self, tmp_path):
        from sinner2.batch.driver import _resolve_face_map  # noqa: SLF001
        store = tmp_path / "face_maps"
        self._save_catalog(tmp_path, store, use=False)  # pref off
        assert _resolve_face_map(self._task(tmp_path, store=store)) == (None, None)

    def test_no_catalog_uses_global_source(self, tmp_path):
        from sinner2.batch.driver import _resolve_face_map  # noqa: SLF001
        from sinner2.pipeline.face_map_store import save_use_map, use_map_path
        store = tmp_path / "face_maps"
        save_use_map(use_map_path(tmp_path / "t.mp4", store), True)  # pref on, no map
        assert _resolve_face_map(self._task(tmp_path, store=store)) == (None, None)

    def test_legacy_by_value_fallback_without_store(self, tmp_path):
        from sinner2.batch.driver import _resolve_face_map  # noqa: SLF001
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        fm = FaceMap(identities=(Identity("a", normalize([1.0, 0.0, 0.0])),))
        loaded, geom = _resolve_face_map(
            self._task(tmp_path, store=None, face_map=fm.to_dict())
        )
        assert loaded is not None and len(loaded.identities) == 1 and geom is None
