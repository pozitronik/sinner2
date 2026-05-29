"""Headless per-task processing loop.

One BatchDriver = one task end-to-end:
  1. Build reader + chain + writer from the task's config.
  2. Sequentially walk every frame index.
     - If the corresponding cache file exists, skip (resume-from-cache).
     - Else submit through the chain → write cache file.
  3. Honor pause / cancel between submissions.
  4. When all frames are present, encode:
     - VIDEO: ffmpeg from cache_dir to output_path.mp4, audio re-muxed.
     - FRAMES: copy cache_dir to output_path directory.
  5. Update task status / last_completed_frame / error_message on the
     instance the caller passed in. Caller persists via the store.

Threading: BatchDriver.run() itself runs on whatever thread the caller
chose (the BatchQueue spins it on a QThread). Inside run(), it manages
its own ThreadPoolExecutor for the per-frame chain.process() calls.

Pause/resume contract: pause_event.set() makes the driver stop
submitting NEW frames; in-flight frames complete and land in the cache.
Resume = re-call run(task) on the same instance — the cache covers
what was done.
"""
from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
    resolve_output_path,
)
from sinner2.config.source import Source
from sinner2.config.target import Target, TargetKind
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.io.video_backend import VideoBackend, build_video_target_reader
from sinner2.io.video_encoder import (
    FfmpegMissingError,
    encode_frames_to_mp4,
)
from sinner2.pipeline.image_writer import (
    ImageFormat,
    ImageWriter,
    build_image_writer,
)
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_enhancer import (
    FaceEnhancer,
    FaceEnhancerParams,
)
from sinner2.pipeline.processors.face_swapper import (
    FaceSwapper,
    FaceSwapperParams,
    TargetSex,
)
from sinner2.types import Frame, FrameIndex


# Progress callback: (completed_frames, total_frames). Driver calls
# this from worker threads — caller is responsible for marshalling
# back to the GUI thread (BatchQueue does this via Qt signals).
ProgressCallback = Callable[[int, int], None]


class BatchDriver:
    """Run one BatchTask to completion (or to pause/cancel)."""

    def __init__(
        self,
        cache_root: Path,
        *,
        global_output_dir: Path | None = None,
    ) -> None:
        # cache_root is the parent of per-task cache subdirs. Each task
        # gets its own <cache_root>/<task_id>/ folder. Distinct from
        # the realtime preview cache so a pause+cancel+rerun doesn't
        # pick up stale preview frames.
        self._cache_root = cache_root
        self._global_output_dir = global_output_dir
        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        # Serializes reader.read() across the worker pool — TargetReaders
        # are not thread-safe (see _process_one).
        self._read_lock = threading.Lock()

    # ---- External controls (call from any thread) ----

    def pause(self) -> None:
        self._pause_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()

    def reset_signals(self) -> None:
        """Clear pause + cancel before a new run() call."""
        self._pause_event.clear()
        self._cancel_event.clear()

    # ---- Main entry point ----

    def run(
        self,
        task: BatchTask,
        progress_callback: ProgressCallback | None = None,
    ) -> BatchTaskStatus:
        """Drive the task to a terminal status. Returns the final
        status; also mutates task.status / last_completed_frame /
        error_message / started_at / finished_at on the input model
        so the caller can persist via the store."""
        self.reset_signals()
        task.status = BatchTaskStatus.RUNNING
        task.started_at = time.time()
        task.error_message = None

        try:
            return self._run_inner(task, progress_callback)
        except Exception as exc:  # any unexpected failure → failed
            task.status = BatchTaskStatus.FAILED
            task.error_message = f"unexpected: {exc}"
            return task.status
        finally:
            task.finished_at = time.time()

    def _run_inner(
        self,
        task: BatchTask,
        progress_callback: ProgressCallback | None,
    ) -> BatchTaskStatus:
        # Build reader + chain + writer per-task. None of these can be
        # shared across tasks (different source, different target).
        target = Target(path=task.target_path)
        source = Source(path=task.source_path)
        reader = self._build_reader(target, task.video_backend)
        chain = self._build_chain(source, task)
        writer = build_image_writer(task.image_format, task.image_quality)

        task_cache = self._cache_root / task.id
        task_cache.mkdir(parents=True, exist_ok=True)

        try:
            total = reader.frame_count
            task.total_frames = total
            ext = writer.extension

            # Set up chain processors. Setup can take seconds (GFPGAN
            # weight load), so this is the visible "task started"
            # latency. We don't surface it specially — the queue's
            # taskStarted signal fires before run().
            for p in chain:
                p.setup()

            try:
                terminal = self._process_frames(
                    reader, chain, writer, task_cache, total, ext,
                    progress_callback, task,
                )
                if terminal is not None:
                    return terminal

                # Completeness gate: never feed a gappy / zero-byte
                # sequence to the encoder — ffmpeg `-i %08d.ext` stops at
                # the first missing index, silently truncating the video.
                # Fail loudly and keep the cache so a re-run retries the
                # bad frames. (Proper auto-reprocess arrives with the
                # processor-major integrity model.)
                missing = self._missing_frames(task_cache, total, ext)
                if missing:
                    task.status = BatchTaskStatus.FAILED
                    task.error_message = self._missing_message(missing)
                    task.last_completed_frame = (
                        self._count_existing_cached_frames(
                            task_cache, total, ext
                        )
                        - 1
                    )
                    return task.status

                # Every frame present & non-empty → encode / package.
                self._package_output(task, task_cache, reader.fps, ext)
                task.status = BatchTaskStatus.COMPLETED
                return task.status
            finally:
                for p in chain:
                    p.release()
        finally:
            reader.release()

    # ---- Frame processing ----

    def _process_frames(
        self,
        reader: TargetReader,
        chain: list[Processor],
        writer: ImageWriter,
        cache_dir: Path,
        total: int,
        ext: str,
        progress_callback: ProgressCallback | None,
        task: BatchTask,
    ) -> BatchTaskStatus | None:
        """Submit every uncached frame through the chain. Returns a
        terminal status (PAUSED / CANCELLED) when interrupted, or
        None when all frames landed in cache."""
        max_workers = max(1, task.worker_count)
        completed = self._count_existing_cached_frames(cache_dir, total, ext)
        task.last_completed_frame = completed - 1
        if progress_callback is not None:
            progress_callback(completed, total)
        # Backpressure: at most 2× workers worth of in-flight futures
        # so the submit loop doesn't outpace the workers by orders of
        # magnitude (would balloon memory with frame buffers).
        in_flight_cap = max_workers * 2
        active: list[Future] = []
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sinner2-batch-worker",
        ) as pool:
            for idx in range(total):
                if self._cancel_event.is_set():
                    return self._after_interrupt(
                        BatchTaskStatus.CANCELLED, active, cache_dir, task
                    )
                if self._pause_event.is_set():
                    return self._after_interrupt(
                        BatchTaskStatus.PAUSED, active, cache_dir, task
                    )
                cache_path = cache_dir / f"{idx:08d}.{ext}"
                if self._frame_ok(cache_path):
                    continue  # resume: skip only valid (non-empty) frames
                fut = pool.submit(
                    self._process_one, reader, idx, chain, writer, cache_path
                )
                active.append(fut)
                # Drain to keep in-flight under cap.
                if len(active) >= in_flight_cap:
                    self._drain_one(active, cache_dir, total, ext,
                                    progress_callback, task)
            # All submitted; wait for the tail.
            while active:
                if self._cancel_event.is_set():
                    return self._after_interrupt(
                        BatchTaskStatus.CANCELLED, active, cache_dir, task
                    )
                self._drain_one(active, cache_dir, total, ext,
                                progress_callback, task)
        # Final progress update.
        if progress_callback is not None:
            progress_callback(total, total)
        return None

    def _drain_one(
        self,
        active: list[Future],
        cache_dir: Path,
        total: int,
        ext: str,
        progress_callback: ProgressCallback | None,
        task: BatchTask,
    ) -> None:
        done, _pending = wait(active, return_when="FIRST_COMPLETED")
        for fut in done:
            active.remove(fut)
            exc = fut.exception()
            if exc is not None:
                # One frame failed — log and continue; the encoder
                # will skip missing frames or fail loudly if too many
                # are missing. Could be more aggressive (fail task);
                # for v1 stay resilient to single-frame issues.
                task.error_message = (
                    f"frame error (one or more): {exc}"
                )
        completed = self._count_existing_cached_frames(cache_dir, total, ext)
        task.last_completed_frame = completed - 1
        if progress_callback is not None:
            progress_callback(completed, total)

    def _after_interrupt(
        self,
        status: BatchTaskStatus,
        active: list[Future],
        cache_dir: Path,
        task: BatchTask,
    ) -> BatchTaskStatus:
        """Pause / cancel branch. Let in-flight workers finish so cache
        stays consistent; on CANCELLED, clear the cache so a re-run
        starts fresh."""
        # Wait for in-flight frames to finish so their cache writes
        # land. They check the cancel flag too but only between frames;
        # the cooperative model is "don't start new, finish current."
        for fut in active:
            try:
                fut.result(timeout=30)
            except Exception:
                pass
        if status is BatchTaskStatus.CANCELLED:
            # Clear cache so a fresh run starts from 0. Pause keeps the
            # cache (that's the whole point of resume-from-cache).
            self._wipe_cache(cache_dir)
            task.last_completed_frame = -1
        task.status = status
        return status

    def _process_one(
        self,
        reader: TargetReader,
        idx: FrameIndex,
        chain: list[Processor],
        writer: ImageWriter,
        cache_path: Path,
    ) -> None:
        # TargetReaders (cv2.VideoCapture / ffmpeg pipe) are NOT
        # thread-safe: each holds a single capture/subprocess plus a
        # mutable _next_index, and documents single-threaded access as a
        # caller invariant. With worker_count > 1 the batch pool violated
        # that, racing concurrent seeks/reads. Serialize the decode; the
        # expensive chain.process() runs OUTSIDE the lock, so GPU work
        # still parallelizes across workers.
        with self._read_lock:
            frame = reader.read(idx)
        if frame is None:
            raise OSError(f"reader returned None at index {idx}")
        result: Frame = frame
        for p in chain:
            result = p.process(result)
        writer.write(cache_path, result)

    # ---- Builders ----

    @staticmethod
    def _build_reader(
        target: Target, video_backend: VideoBackend
    ) -> TargetReader:
        # Mirrors PlayerController._make_reader. Kept duplicated for
        # now to avoid pulling player_controller into batch (one less
        # GUI dep for the headless driver).
        if target.kind == TargetKind.IMAGE:
            return ImageTargetReader(target)
        if target.kind == TargetKind.VIDEO:
            return build_video_target_reader(target, video_backend)
        raise ValueError(f"unsupported target kind: {target.kind}")

    @staticmethod
    def _build_chain(source: Source, task: BatchTask) -> list[Processor]:
        swapper_params = FaceSwapperParams(
            detection_interval=task.swapper_detection_interval,
            many_faces=task.swapper_many_faces,
            target_sex=TargetSex(task.swapper_target_sex),
        )
        chain: list[Processor] = [FaceSwapper(source=source, params=swapper_params)]
        if task.enhancer_enabled:
            chain.append(
                FaceEnhancer(
                    params=FaceEnhancerParams(
                        upscale=task.enhancer_upscale,
                        only_center_face=task.enhancer_only_center_face,
                    )
                )
            )
        return chain

    # ---- Cache helpers ----

    @staticmethod
    def _frame_ok(path: Path) -> bool:
        """A cached frame counts as done only if it exists AND is
        non-empty. Zero-byte files (disk full mid-write) must be
        reprocessed, never handed to the encoder."""
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @classmethod
    def _missing_frames(
        cls, cache_dir: Path, total: int, ext: str
    ) -> list[int]:
        """Indices in [0, total) whose cache frame is absent or empty."""
        return [
            idx
            for idx in range(total)
            if not cls._frame_ok(cache_dir / f"{idx:08d}.{ext}")
        ]

    @staticmethod
    def _missing_message(missing: list[int]) -> str:
        preview = ", ".join(str(i) for i in missing[:8])
        more = "" if len(missing) <= 8 else f", +{len(missing) - 8} more"
        return (
            f"{len(missing)} frame(s) missing or empty ({preview}{more}); "
            "refusing to encode a truncated video. Re-run to retry — "
            "cached frames are kept."
        )

    @classmethod
    def _count_existing_cached_frames(
        cls, cache_dir: Path, total: int, ext: str
    ) -> int:
        """Count contiguous VALID (present + non-empty) cached frames
        from 0. Stops at the first missing/empty index so a gap mid-cache
        isn't reported as completed."""
        for idx in range(total):
            if not cls._frame_ok(cache_dir / f"{idx:08d}.{ext}"):
                return idx
        return total

    @staticmethod
    def _wipe_cache(cache_dir: Path) -> None:
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            pass
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Output packaging ----

    def _package_output(
        self,
        task: BatchTask,
        cache_dir: Path,
        fps: float,
        ext: str,
    ) -> None:
        output = resolve_output_path(task, self._global_output_dir)
        if task.output_format is BatchOutputFormat.VIDEO:
            try:
                encode_frames_to_mp4(
                    cache_dir,
                    output,
                    fps=fps,
                    frame_ext=ext,
                    audio_source=task.target_path,
                )
            except FfmpegMissingError as exc:
                # Fallback: ship the frames as a directory next to where
                # the mp4 would have lived. User keeps something usable.
                task.error_message = (
                    f"ffmpeg missing — fell back to frames mode: {exc}"
                )
                self._copy_frames(cache_dir, output.with_suffix(""))
        else:
            self._copy_frames(cache_dir, output)

    @staticmethod
    def _copy_frames(cache_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for src in cache_dir.glob("*"):
            if not src.is_file():
                continue
            dst = output_dir / src.name
            shutil.copy2(src, dst)
