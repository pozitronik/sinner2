"""Tests for BatchDriver — end-to-end with stub processors.

We stub the chain builder so we don't need real face-swap models in
tests (those take seconds to load and need real images). The driver
itself is exercised: frame loop, cache-skip on resume, pause, cancel,
encode path.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from sinner2.batch.driver import BatchDriver
from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.io.video_encoder import FfmpegMissingError
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.types import Frame


class _PassthroughProcessor:
    """Returns the input frame unchanged. Tracks process() call count
    so tests can assert work was done (or not, on cache hit)."""

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
    """Slow processor so pause/cancel windows are observable from
    the test thread."""

    name = "Sleep"

    def __init__(self, seconds: float = 0.05) -> None:
        super().__init__()
        self._seconds = seconds

    def process(self, frame: Frame) -> Frame:
        time.sleep(self._seconds)
        return super().process(frame)


class _ConcurrencyCheckingReader:
    """Stub TargetReader that flags overlapping read() calls — used to
    prove the driver serializes decode across workers."""

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
            time.sleep(0.005)  # widen the race window
            return np.full((8, 8, 3), index % 256, dtype=np.uint8)
        finally:
            with self._guard:
                self._inside -= 1

    def release(self) -> None:
        pass


class _IndexEncodingReader:
    """Stub TargetReader whose frames encode their index in pixel (0,0)
    so a processor can deterministically key on a specific frame."""

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


class _FailFrameProcessor(_PassthroughProcessor):
    """Raises on the frame whose pixel (0,0) equals fail_value; passes
    every other frame through."""

    name = "FailOne"

    def __init__(self, fail_value: int) -> None:
        super().__init__()
        self._fail_value = fail_value

    def process(self, frame: Frame) -> Frame:
        if int(frame[0, 0, 0]) == self._fail_value:
            raise RuntimeError("boom on target frame")
        return super().process(frame)


def _make_image(path: Path, w: int = 16, h: int = 16) -> Path:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _make_task(
    tmp_path: Path,
    *,
    image_target: bool = True,
    enhancer_enabled: bool = False,
    output_format: BatchOutputFormat = BatchOutputFormat.FRAMES,
) -> BatchTask:
    source = _make_image(tmp_path / "src.png")
    if image_target:
        target = _make_image(tmp_path / "tgt.png")
    else:
        # Build a tiny mp4 via cv2 for the multi-frame case.
        import cv2

        target = tmp_path / "tgt.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(target), fourcc, 5.0, (16, 16))
        if not writer.isOpened():
            pytest.skip("cv2 VideoWriter mp4v unavailable")
        try:
            for i in range(3):
                writer.write(np.full((16, 16, 3), i * 60, dtype=np.uint8))
        finally:
            writer.release()
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


@pytest.fixture
def stub_chain(monkeypatch):
    """Replace BatchDriver._build_chain with a no-arg stub processor
    so tests don't need real face models / source images with faces."""
    procs: list[_PassthroughProcessor] = []

    def fake_build(_source, task):
        p = _PassthroughProcessor()
        procs.append(p)
        return [p]

    monkeypatch.setattr(BatchDriver, "_build_chain", staticmethod(fake_build))
    return procs


@pytest.fixture
def driver(tmp_path: Path) -> BatchDriver:
    return BatchDriver(cache_root=tmp_path / "cache")


class TestImageTarget:
    def test_single_image_runs_to_completion(
        self, driver, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path, image_target=True)
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert task.last_completed_frame == 0
        assert task.total_frames == 1
        # FRAMES output: a directory at task.output_path with one file.
        assert task.output_path.is_dir()
        assert len(list(task.output_path.glob("*.jpg"))) == 1

    def test_setup_release_called_on_chain(
        self, driver, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path)
        driver.run(task)
        assert len(stub_chain) == 1
        p = stub_chain[0]
        assert p.setup_calls == 1
        assert p.release_calls == 1
        assert p.process_calls == 1


class TestVideoTarget:
    def test_three_frame_video_runs(
        self, driver, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path, image_target=False)
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert task.total_frames == 3
        assert task.last_completed_frame == 2
        # FRAMES output: dir with 3 jpgs.
        assert len(list(task.output_path.glob("*.jpg"))) == 3


class TestResumeFromCache:
    def test_second_run_skips_already_cached_frames(
        self, driver, stub_chain, tmp_path
    ):
        task = _make_task(tmp_path, image_target=False)
        # First run: 3 frames processed.
        driver.run(task)
        assert stub_chain[0].process_calls == 3
        # Wipe the output (NOT the cache) to simulate "re-run after
        # the encoded video was deleted". The processor's process()
        # MUST NOT be called again — cache covers everything.
        shutil.rmtree(task.output_path)
        # Fresh chain instance for the second run.
        stub_chain.clear()
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        # Stub for second run was set up + released but processed
        # ZERO frames because all 3 were already in cache.
        assert len(stub_chain) == 1
        assert stub_chain[0].process_calls == 0
        # Output rebuilt from cache.
        assert len(list(task.output_path.glob("*.jpg"))) == 3


class TestPauseCancel:
    def test_pause_stops_mid_task(self, tmp_path, monkeypatch):
        # Use the slow processor so we have time to fire pause
        # between frame submissions.
        slow: list[_SleepProcessor] = []

        def fake_build(_source, task):
            p = _SleepProcessor(seconds=0.05)
            slow.append(p)
            return [p]

        monkeypatch.setattr(
            BatchDriver, "_build_chain", staticmethod(fake_build)
        )
        # Build a 10-frame video so pause has time to land.
        import cv2

        target = tmp_path / "long.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(target), fourcc, 5.0, (16, 16))
        if not w.isOpened():
            pytest.skip("cv2 VideoWriter mp4v unavailable")
        try:
            for i in range(10):
                w.write(np.full((16, 16, 3), i * 20, dtype=np.uint8))
        finally:
            w.release()
        task = BatchTask(
            source_path=_make_image(tmp_path / "src.png"),
            target_path=target,
            output_path=tmp_path / "out",
            output_format=BatchOutputFormat.FRAMES,
            worker_count=1,
            image_format=ImageFormat.JPEG,
            image_quality=80,
        )
        driver = BatchDriver(cache_root=tmp_path / "cache")

        # Fire pause from a side thread shortly after run starts.
        def trigger_pause():
            time.sleep(0.1)
            driver.pause()

        threading.Thread(target=trigger_pause, daemon=True).start()
        status = driver.run(task)
        assert status is BatchTaskStatus.PAUSED
        # We must have processed FEWER than the full 10 frames before
        # pause took effect, but at least one.
        assert 0 < slow[0].process_calls < 10
        # No output produced on pause (encode skipped).
        assert not task.output_path.exists()

    def test_resume_after_pause_completes_task(
        self, tmp_path, monkeypatch
    ):
        # Same scenario as above, then resume; cache covers the
        # partial work, fresh chain processes the rest.
        all_procs: list[_SleepProcessor] = []

        def fake_build(_source, task):
            p = _SleepProcessor(seconds=0.02)
            all_procs.append(p)
            return [p]

        monkeypatch.setattr(
            BatchDriver, "_build_chain", staticmethod(fake_build)
        )
        import cv2

        target = tmp_path / "long.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(target), fourcc, 5.0, (16, 16))
        if not w.isOpened():
            pytest.skip("cv2 VideoWriter mp4v unavailable")
        try:
            for i in range(8):
                w.write(np.full((16, 16, 3), i * 20, dtype=np.uint8))
        finally:
            w.release()
        task = BatchTask(
            source_path=_make_image(tmp_path / "src.png"),
            target_path=target,
            output_path=tmp_path / "out",
            output_format=BatchOutputFormat.FRAMES,
            worker_count=1,
            image_format=ImageFormat.JPEG,
            image_quality=80,
        )
        driver = BatchDriver(cache_root=tmp_path / "cache")

        # First run: pause early.
        def trigger_pause():
            time.sleep(0.05)
            driver.pause()

        threading.Thread(target=trigger_pause, daemon=True).start()
        assert driver.run(task) is BatchTaskStatus.PAUSED
        first_run_count = all_procs[0].process_calls
        # Resume.
        all_procs.clear()
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        # Combined processing equals total frames (8); the resumed
        # run processes 8 - first_run_count fresh frames.
        assert all_procs[0].process_calls == 8 - first_run_count
        assert len(list(task.output_path.glob("*.jpg"))) == 8

    def test_cancel_wipes_cache(self, tmp_path, monkeypatch):
        slow: list[_SleepProcessor] = []

        def fake_build(_source, task):
            p = _SleepProcessor(seconds=0.03)
            slow.append(p)
            return [p]

        monkeypatch.setattr(
            BatchDriver, "_build_chain", staticmethod(fake_build)
        )
        import cv2

        target = tmp_path / "long.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(target), fourcc, 5.0, (16, 16))
        if not w.isOpened():
            pytest.skip("cv2 VideoWriter mp4v unavailable")
        try:
            for i in range(10):
                w.write(np.full((16, 16, 3), i * 20, dtype=np.uint8))
        finally:
            w.release()
        task = BatchTask(
            source_path=_make_image(tmp_path / "src.png"),
            target_path=target,
            output_path=tmp_path / "out",
            output_format=BatchOutputFormat.FRAMES,
            worker_count=1,
            image_format=ImageFormat.JPEG,
            image_quality=80,
        )
        driver = BatchDriver(cache_root=tmp_path / "cache")

        def trigger_cancel():
            time.sleep(0.05)
            driver.cancel()

        threading.Thread(target=trigger_cancel, daemon=True).start()
        status = driver.run(task)
        assert status is BatchTaskStatus.CANCELLED
        # Cache wiped — task cache dir empty.
        task_cache = driver._cache_root / task.id  # noqa: SLF001
        assert len(list(task_cache.glob("*.jpg"))) == 0
        # last_completed_frame reset so a fresh run starts from 0.
        assert task.last_completed_frame == -1


class TestFfmpegFallback:
    def test_missing_ffmpeg_falls_back_to_frames(
        self, tmp_path, stub_chain, monkeypatch
    ):
        # Patch the encoder to raise FfmpegMissingError so the driver
        # exercises its fallback path even on a machine with ffmpeg.
        import sinner2.batch.driver as driver_module

        def fake_encode(*_a, **_k):
            raise FfmpegMissingError("test")

        monkeypatch.setattr(
            driver_module, "encode_frames_to_mp4", fake_encode
        )
        driver = BatchDriver(cache_root=tmp_path / "cache")
        task = _make_task(
            tmp_path,
            image_target=True,
            output_format=BatchOutputFormat.VIDEO,
        )
        # Override output_path so VIDEO suffix points to <out.mp4>;
        # fallback strips the suffix and creates a directory.
        task.output_path = tmp_path / "out.mp4"
        status = driver.run(task)
        # Status is COMPLETED with a note in error_message — the user
        # still gets something usable, just not encoded.
        assert status is BatchTaskStatus.COMPLETED
        assert "fell back to frames mode" in (task.error_message or "")
        # Frames at tmp_path/out (without .mp4) — driver strips suffix.
        frames_dir = tmp_path / "out"
        assert frames_dir.is_dir()
        assert len(list(frames_dir.glob("*.jpg"))) == 1


class TestProgressCallback:
    def test_progress_fires_for_each_completed_frame(
        self, driver, stub_chain, tmp_path
    ):
        # Stub chain → fast processing; just verify callback received
        # the final (total, total) tuple at minimum.
        task = _make_task(tmp_path, image_target=False)
        events: list[tuple[int, int]] = []
        driver.run(task, progress_callback=lambda c, t: events.append((c, t)))
        assert events, "progress callback never fired"
        # Final event must report total/total.
        assert events[-1] == (3, 3)


class TestReaderThreadSafety:
    def test_reads_are_serialized_across_workers(
        self, tmp_path, monkeypatch, stub_chain
    ):
        # cv2 / ffmpeg readers aren't thread-safe; the driver must
        # serialize reader.read() even with worker_count > 1. The stub
        # flags any overlapping read() call.
        overlaps: list[int] = []
        reader = _ConcurrencyCheckingReader(6, lambda: overlaps.append(1))
        monkeypatch.setattr(
            BatchDriver,
            "_build_reader",
            staticmethod(lambda target, backend: reader),
        )
        task = _make_task(tmp_path, image_target=True)
        task.worker_count = 4  # force concurrency
        driver = BatchDriver(cache_root=tmp_path / "cache")
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        assert overlaps == [], "reader.read() was called concurrently"


class TestCompletenessGate:
    def test_missing_frame_fails_instead_of_truncating(
        self, tmp_path, monkeypatch
    ):
        # One frame's worker raises → that frame never lands in cache.
        # The driver must FAIL (not encode a truncated video) and keep
        # the good frames for a retry.
        reader = _IndexEncodingReader(4)
        monkeypatch.setattr(
            BatchDriver,
            "_build_reader",
            staticmethod(lambda target, backend: reader),
        )
        monkeypatch.setattr(
            BatchDriver,
            "_build_chain",
            staticmethod(lambda _source, task: [_FailFrameProcessor(2)]),
        )
        task = _make_task(tmp_path, image_target=True)
        task.worker_count = 1  # deterministic
        driver = BatchDriver(cache_root=tmp_path / "cache")
        status = driver.run(task)
        assert status is BatchTaskStatus.FAILED
        assert "missing or empty" in (task.error_message or "")
        # No truncated output was written.
        assert not task.output_path.exists()
        # The 3 good frames are kept for a retry.
        task_cache = driver._cache_root / task.id  # noqa: SLF001
        assert len(list(task_cache.glob("*.jpg"))) == 3

    def test_zero_byte_cache_frame_is_reprocessed_on_resume(
        self, driver, stub_chain, tmp_path
    ):
        # A zero-byte cache file (disk full mid-write) must be treated as
        # missing and reprocessed on the next run — not skipped and fed
        # to the encoder.
        task = _make_task(tmp_path, image_target=False)  # 3-frame video
        driver.run(task)
        assert stub_chain[0].process_calls == 3
        task_cache = driver._cache_root / task.id  # noqa: SLF001
        frames = sorted(task_cache.glob("*.jpg"))
        frames[1].write_bytes(b"")  # truncate one frame to 0 bytes
        shutil.rmtree(task.output_path)
        stub_chain.clear()
        status = driver.run(task)
        assert status is BatchTaskStatus.COMPLETED
        # Exactly the corrupted frame was reprocessed.
        assert stub_chain[0].process_calls == 1
        assert len(list(task.output_path.glob("*.jpg"))) == 3
