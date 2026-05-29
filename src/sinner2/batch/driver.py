"""Headless per-task batch processing — processor-major staged execution.

One BatchDriver = one task end-to-end. Unlike the realtime executor
(frame-major, latency-optimized), the batch driver runs each processor over
ALL frames before the next — "stage-major" — so only one model is resident
at a time and the device does one kind of work (throughput-optimized):

    stage 0  FaceSwapper  : source video  → <cache>/<task>/stage0-faceswapper/
    stage 1  FaceEnhancer : stage0 frames → <cache>/<task>/stage1-faceenhancer/  [if enabled]
    encode                : last stage dir → output.mp4 (audio re-muxed) | frames copy

Each stage is resume-aware and integrity-checked by run_stage()
(batch/stage.py): the frame read happens on a single thread (sequential
decode for stage 0; no concurrent reader access), workers only process +
write, and a stage fails loudly rather than handing a gappy sequence to the
encoder.

Pause/resume: pause makes the running stage stop submitting new frames;
in-flight frames land on disk. Resume = re-run the task — completed stages
are skipped and the interrupted stage continues from its on-disk frames.
Cancel wipes the whole task cache so a re-run starts fresh.

Threading: run() executes on whatever thread the caller chose (BatchQueue
spins it on a QThread); each stage manages its own worker pool internally.
"""
from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from sinner2.batch.stage import (
    FramesDirInput,
    ReaderStageInput,
    StageInput,
    StageStatus,
    frame_ok,
    run_stage,
)
from sinner2.batch.task import (
    BatchCleanupMode,
    BatchExtractionMode,
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
    resolve_output_path,
)
from sinner2.config.source import Source
from sinner2.config.target import Target, TargetKind
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.io.video_backend import VideoBackend, build_video_target_reader
from sinner2.io.video_encoder import FfmpegMissingError, encode_frames_to_mp4
from sinner2.pipeline.image_writer import build_image_writer
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


# Progress callback: (completed_frames, total_frames) for the CURRENT stage.
# Step 5 enriches this into a structured per-stage + overall update.
ProgressCallback = Callable[[int, int], None]


class BatchDriver:
    """Run one BatchTask to completion (or to pause/cancel)."""

    def __init__(
        self,
        cache_root: Path,
        *,
        global_output_dir: Path | None = None,
    ) -> None:
        # cache_root is the parent of per-task cache subdirs. Each task gets
        # <cache_root>/<task_id>/stage{N}-{name}/ folders. Distinct from the
        # realtime preview cache so a pause+cancel+rerun doesn't pick up
        # stale preview frames.
        self._cache_root = cache_root
        self._global_output_dir = global_output_dir
        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()

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
        """Drive the task to a terminal status. Returns the final status and
        also mutates task.status / last_completed_frame / completed_stages /
        error_message / timing on the input model so the caller can persist."""
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
        if task.extraction_mode is BatchExtractionMode.PREEXTRACT:
            # Seam exists (task field + FramesDirInput); the bulk-extract
            # pass isn't built yet. Fail clearly rather than silently
            # falling back, so the choice is honoured or surfaced.
            task.status = BatchTaskStatus.FAILED
            task.error_message = "pre-extraction mode is not implemented yet"
            return task.status

        target = Target(path=task.target_path)
        source = Source(path=task.source_path)
        writer = build_image_writer(task.image_format, task.image_quality)
        ext = writer.extension
        stages = self._build_stages(source, task)
        task_cache = self._cache_root / task.id
        stage_dirs = [
            task_cache / f"stage{i}-{name}"
            for i, (name, _) in enumerate(stages)
        ]

        # The first-stage reader also probes total frame count + fps.
        reader = self._build_reader(target, task.video_backend)
        try:
            total = reader.frame_count
            fps = reader.fps
            task.total_frames = total
            stage_cb: Callable[[int], None] | None = (
                (lambda done: progress_callback(done, total))
                if progress_callback is not None
                else None
            )

            for i, (name, processor) in enumerate(stages):
                trusted = (
                    task.cleanup_mode is BatchCleanupMode.AUTO
                    and i < task.completed_stages
                )
                if trusted or self._stage_complete(stage_dirs[i], ext, total):
                    # Done already — verified on disk, or (under Auto, whose
                    # intermediate dirs get deleted) trusted via the marker.
                    if stage_cb is not None:
                        stage_cb(total)
                else:
                    stage_input: StageInput = (
                        ReaderStageInput(reader)
                        if i == 0
                        else FramesDirInput(stage_dirs[i - 1], ext, total)
                    )
                    result = run_stage(
                        stage_input=stage_input,
                        processor=processor,
                        output_dir=stage_dirs[i],
                        ext=ext,
                        writer=writer,
                        workers=task.worker_count,
                        pause_event=self._pause_event,
                        cancel_event=self._cancel_event,
                        on_progress=stage_cb,
                    )
                    if result.status is StageStatus.PAUSED:
                        task.status = BatchTaskStatus.PAUSED
                        task.last_completed_frame = result.completed_frames - 1
                        return task.status
                    if result.status is StageStatus.CANCELLED:
                        self._wipe_cache(task_cache)
                        task.status = BatchTaskStatus.CANCELLED
                        task.last_completed_frame = -1
                        task.completed_stages = 0
                        return task.status
                    if result.status is StageStatus.FAILED:
                        task.status = BatchTaskStatus.FAILED
                        task.error_message = self._stage_failed_message(
                            name, result.missing
                        )
                        task.last_completed_frame = result.completed_frames - 1
                        return task.status
                # max() so resuming a paused task can't regress the marker.
                task.completed_stages = max(task.completed_stages, i + 1)
                # Auto: drop the now-consumed previous stage to cap peak disk.
                if task.cleanup_mode is BatchCleanupMode.AUTO and i > 0:
                    shutil.rmtree(stage_dirs[i - 1], ignore_errors=True)

            # Guard: the last stage's frames must be present to package.
            # Normally they are (we package before any cleanup); this only
            # trips if a completed Auto task is re-run without a refresh (its
            # intermediate dirs are gone) — fail loudly, never write empty.
            if not self._stage_complete(stage_dirs[-1], ext, total):
                task.status = BatchTaskStatus.FAILED
                task.error_message = (
                    "intermediate frames were cleaned up; refresh the task "
                    "to re-run it from scratch"
                )
                return task.status

            # All stages complete → package the last stage's frames, then
            # clean up per the cleanup mode (Keep leaves everything).
            self._package_output(task, stage_dirs[-1], fps, ext)
            task.status = BatchTaskStatus.COMPLETED
            task.last_completed_frame = total - 1
            self._cleanup_stage_dirs(task, stage_dirs)
            return task.status
        finally:
            reader.release()

    # ---- Builders ----

    @staticmethod
    def _build_reader(
        target: Target, video_backend: VideoBackend
    ) -> TargetReader:
        if target.kind == TargetKind.IMAGE:
            return ImageTargetReader(target)
        if target.kind == TargetKind.VIDEO:
            return build_video_target_reader(target, video_backend)
        raise ValueError(f"unsupported target kind: {target.kind}")

    @staticmethod
    def _build_stages(
        source: Source, task: BatchTask
    ) -> list[tuple[str, Processor]]:
        """Ordered (name, processor) stages. One processor per stage — they
        run in turns, not chained per-frame."""
        swapper = FaceSwapper(
            source=source,
            params=FaceSwapperParams(
                detection_interval=task.swapper_detection_interval,
                many_faces=task.swapper_many_faces,
                target_sex=TargetSex(task.swapper_target_sex),
            ),
        )
        stages: list[tuple[str, Processor]] = [("faceswapper", swapper)]
        if task.enhancer_enabled:
            stages.append(
                (
                    "faceenhancer",
                    FaceEnhancer(
                        params=FaceEnhancerParams(
                            upscale=task.enhancer_upscale,
                            only_center_face=task.enhancer_only_center_face,
                        )
                    ),
                )
            )
        return stages

    # ---- Helpers ----

    @staticmethod
    def _stage_complete(stage_dir: Path, ext: str, total: int) -> bool:
        """True iff every frame for this stage is already valid on disk."""
        return all(
            frame_ok(stage_dir / f"{i:08d}.{ext}") for i in range(total)
        )

    @staticmethod
    def _stage_failed_message(stage_name: str, missing: list[int]) -> str:
        preview = ", ".join(str(i) for i in missing[:8])
        more = "" if len(missing) <= 8 else f", +{len(missing) - 8} more"
        return (
            f"stage '{stage_name}': {len(missing)} frame(s) missing or empty "
            f"({preview}{more}); refusing to encode a truncated video. "
            "Re-run to retry — cached frames are kept."
        )

    @staticmethod
    def _wipe_cache(cache_dir: Path) -> None:
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            pass

    @staticmethod
    def _cleanup_stage_dirs(task: BatchTask, stage_dirs: list[Path]) -> None:
        """Post-run cleanup. Auto and Drop-at-end remove all stage dirs once
        the final output exists; Keep leaves them for inspection / re-export."""
        if task.cleanup_mode is BatchCleanupMode.KEEP:
            return
        for stage_dir in stage_dirs:
            shutil.rmtree(stage_dir, ignore_errors=True)

    # ---- Output packaging ----

    def _package_output(
        self,
        task: BatchTask,
        frames_dir: Path,
        fps: float,
        ext: str,
    ) -> None:
        output = resolve_output_path(task, self._global_output_dir)
        if task.output_format is BatchOutputFormat.VIDEO:
            try:
                encode_frames_to_mp4(
                    frames_dir,
                    output,
                    fps=fps,
                    frame_ext=ext,
                    audio_source=task.target_path,
                )
            except FfmpegMissingError as exc:
                # Fallback: ship the frames as a directory next to where the
                # mp4 would have lived. The user keeps something usable.
                task.error_message = (
                    f"ffmpeg missing — fell back to frames mode: {exc}"
                )
                self._copy_frames(frames_dir, output.with_suffix(""))
        else:
            self._copy_frames(frames_dir, output)

    @staticmethod
    def _copy_frames(frames_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for src in frames_dir.glob("*"):
            if src.is_file():
                shutil.copy2(src, output_dir / src.name)
