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

import hashlib
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
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
    BatchProgress,
    BatchTask,
    BatchTaskStatus,
    resolve_output_path,
)
from sinner2.config.source import Source
from sinner2.config.target import Target, TargetKind
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.io.video_backend import VideoBackend, build_video_target_reader
from sinner2.io.video_encoder import FfmpegMissingError, encode_frames_to_mp4
from sinner2.pipeline.detectors import DetectorModel
from sinner2.pipeline.image_writer import build_image_writer
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    FaceEnhancer,
    FaceEnhancerParams,
)
from sinner2.pipeline.processors.face_swapper import (
    FaceSwapper,
    FaceSwapperParams,
    RotationAngleSource,
    SwapperModel,
    TargetSex,
)
from sinner2.pipeline.processors.occlusion import (
    FaceParser,
    OccluderModel,
    OcclusionMaskMode,
)
from sinner2.pipeline.processors.upscaler import (
    Upscaler,
    UpscalerModel,
    UpscalerParams,
)
from sinner2.types import Frame


# Progress callback: a structured per-stage + overall update. The driver
# calls this from a worker context — the caller marshals to the GUI thread
# (BatchQueue does this via a queued Qt signal).
ProgressCallback = Callable[[BatchProgress], None]

# Preview callback: a recently-processed frame (throttled by the stage) so
# the GUI can show what the batch is producing. Same marshalling caveat.
PreviewCallback = Callable[[Frame], None]


@dataclass(frozen=True)
class StageSpec:
    """One processor-major stage: how to build its processor, whether the
    processor can be shared across workers, and how many workers to run it
    with. The factory (not a pre-built instance) lets the stage runner build
    the right NUMBER of instances — one shared for thread-safe processors,
    one per worker otherwise."""

    name: str
    factory: Callable[[], Processor]
    thread_safe: bool
    workers: int


class _IdentityProcessor:
    """No-op stage used when BOTH processors are disabled — re-encodes the
    source frames unchanged (the user-requested raw passthrough)."""

    name = "passthrough"
    thread_safe = True  # stateless no-op — safe to share across workers

    def setup(self) -> None:
        pass

    def process(self, frame: Frame) -> Frame:
        return frame

    def release(self) -> None:
        pass


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
        preview_callback: PreviewCallback | None = None,
    ) -> BatchTaskStatus:
        """Drive the task to a terminal status. Returns the final status and
        also mutates task.status / last_completed_frame / completed_stages /
        error_message / timing on the input model so the caller can persist."""
        self.reset_signals()
        task.status = BatchTaskStatus.RUNNING
        task.started_at = time.time()
        task.error_message = None

        try:
            return self._run_inner(task, progress_callback, preview_callback)
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
        preview_callback: PreviewCallback | None,
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

        # The first-stage reader probes total frame count + fps, and its
        # (scaled) dimensions seed the cache-dir token: a processing-scale
        # change yields different dir names, so frames of a different size
        # never get reused as if they matched. Built before stage_dirs so the
        # token reflects the actual output size.
        reader = self._build_reader(
            target, task.video_backend, task.processing_scale
        )
        size_token = f"{reader.width}x{reader.height}"
        task_cache = self._cache_root / task.id
        stage_dirs = self._stage_cache_dirs(task_cache, size_token, task, stages)
        current_fp = self._chain_fingerprint(task, size_token)
        if task.cache_fingerprint and task.cache_fingerprint != current_fp:
            # Source / target / scale changed since the cached run: the persisted
            # resume markers point at frames rendered for a DIFFERENT render under
            # a now-stale token. Reset so the task re-renders from scratch instead
            # of trusting them — esp. the AUTO trusted-skip path, which would
            # otherwise read an empty new-token dir and hard-fail "frames missing".
            # (Mere settings edits keep the fingerprint and resume in place.)
            task.completed_stages = 0
            task.last_completed_frame = -1
            task.total_frames = -1
        task.cache_fingerprint = current_fp
        try:
            total = reader.frame_count
            fps = reader.fps
            # Prefer the real decoded length persisted from a prior run. The
            # container's nb_frames over-counts for VFR sources; stage 0 corrects
            # it to the true length at EOF (below), but on RESUME stage 0 is
            # skipped, so without this stages 1+ look for frames that never
            # existed and fail "frames missing". 0 < persisted <= container
            # guards a stale or absent value (total_frames defaults to -1).
            if 0 < task.total_frames <= total:
                total = task.total_frames
            task.total_frames = total
            stage_count = len(stages)

            for i, spec in enumerate(stages):
                name = spec.name
                stage_cb = self._stage_progress(
                    progress_callback, i, stage_count, name, total
                )
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
                        processor_factory=spec.factory,
                        thread_safe=spec.thread_safe,
                        output_dir=stage_dirs[i],
                        ext=ext,
                        writer=writer,
                        workers=spec.workers,
                        pause_event=self._pause_event,
                        cancel_event=self._cancel_event,
                        on_progress=stage_cb,
                        eof_on_none=(i == 0),  # only the video source streams
                        on_preview=preview_callback,
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
                            name, result.missing, result.error
                        )
                        task.last_completed_frame = result.completed_frames - 1
                        return task.status
                    # EOF: a streaming source can decode fewer frames than its
                    # container metadata (nb_frames) claims. Accept a small
                    # trailing shortfall as the true length; reject a large one
                    # as a truncated / corrupt source.
                    if i == 0 and result.total < total:
                        shortfall = total - result.total
                        if shortfall > max(round(2 * fps), 10):
                            task.status = BatchTaskStatus.FAILED
                            task.error_message = (
                                f"decoded only {result.total} of {total} "
                                "expected frames — source may be truncated "
                                "or corrupt"
                            )
                            task.last_completed_frame = result.total - 1
                            return task.status
                        total = result.total
                        task.total_frames = total
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
        target: Target, video_backend: VideoBackend, scale: float = 1.0
    ) -> TargetReader:
        if target.kind == TargetKind.IMAGE:
            return ImageTargetReader(target, scale)
        if target.kind == TargetKind.VIDEO:
            return build_video_target_reader(target, video_backend, scale)
        raise ValueError(f"unsupported target kind: {target.kind}")

    @staticmethod
    def _build_stages(source: Source, task: BatchTask) -> list[StageSpec]:
        """Ordered stages, one processor each — they run in turns, not chained
        per-frame. Each stage carries its own execution profile: the swapper is
        ONNX (providers + workers), the enhancer is PyTorch (device + workers).
        Either processor can be disabled; with both off, a single identity
        stage re-encodes the source unprocessed (the user-requested
        passthrough)."""
        stages: list[StageSpec] = []
        if task.swapper_enabled:
            swapper_params = FaceSwapperParams(
                model=SwapperModel(task.swapper_model),
                detection_interval=task.swapper_detection_interval,
                detection_size=task.swapper_detection_size,
                detector=DetectorModel(task.swapper_detector),
                many_faces=task.swapper_many_faces,
                fast_paste=task.swapper_fast_paste,
                target_sex=TargetSex(task.swapper_target_sex),
                rotation_compensation=task.swapper_rotation_compensation,
                rotation_threshold_deg=task.swapper_rotation_threshold_deg,
                rotation_redetect=task.swapper_rotation_redetect,
                rotation_angle_source=RotationAngleSource(
                    task.swapper_rotation_angle_source
                ),
                landmark_refine=task.swapper_landmark_refine,
                occlusion_mask=task.swapper_occlusion_mask,
                occlusion_mode=OcclusionMaskMode(task.swapper_occlusion_mode),
                occlusion_parser=FaceParser(task.swapper_occlusion_parser),
                occluder_model=OccluderModel(task.swapper_occluder_model),
            )
            providers = list(task.swapper_execution.providers)
            stages.append(StageSpec(
                name="faceswapper",
                factory=lambda p=swapper_params, eps=providers: FaceSwapper(
                    source=source, params=p, providers=eps
                ),
                thread_safe=FaceSwapper.thread_safe,
                workers=task.swapper_execution.workers,
            ))
        if task.enhancer_enabled:
            enhancer_params = FaceEnhancerParams(
                model=EnhancerModel(task.enhancer_model),
                upscale=task.enhancer_upscale,
                only_center_face=task.enhancer_only_center_face,
                codeformer_fidelity=task.enhancer_codeformer_fidelity,
                fp16=task.enhancer_fp16,
                # Rotation compensation is shared config (same task fields as
                # the swapper) — the enhancer needs it too, GFPGAN has none.
                rotation_compensation=task.swapper_rotation_compensation,
                rotation_threshold_deg=task.swapper_rotation_threshold_deg,
                rotation_redetect=task.swapper_rotation_redetect,
                rotation_angle_source=RotationAngleSource(
                    task.swapper_rotation_angle_source
                ),
            )
            device = task.enhancer_execution.device
            stages.append(StageSpec(
                name="faceenhancer",
                factory=lambda p=enhancer_params, d=device: FaceEnhancer(
                    params=p, device=d
                ),
                thread_safe=FaceEnhancer.thread_safe,
                workers=task.enhancer_execution.workers,
            ))
        if task.upscaler_enabled:
            upscaler_params = UpscalerParams(
                model=UpscalerModel(task.upscaler_model),
                tile=task.upscaler_tile,
                fp16=task.upscaler_fp16,
            )
            up_device = task.upscaler_execution.device
            stages.append(StageSpec(
                name="upscaler",
                factory=lambda p=upscaler_params, d=up_device: Upscaler(
                    params=p, device=d
                ),
                thread_safe=Upscaler.thread_safe,
                workers=task.upscaler_execution.workers,
            ))
        if not stages:
            stages.append(StageSpec(
                name="passthrough",
                factory=_IdentityProcessor,
                thread_safe=_IdentityProcessor.thread_safe,
                workers=1,
            ))
        return stages

    @staticmethod
    def _stage_cache_dirs(
        task_cache: Path,
        size_token: str,
        task: BatchTask,
        stages: list[StageSpec],
    ) -> list[Path]:
        """Per-stage cache dirs, keyed by stage position/name, output size, and
        the task's IDENTITY (source + target) — deliberately NOT by stage
        params. Editing a task's settings mid-run therefore resumes IN PLACE:
        frames already on disk are kept and only the remaining frames render
        with the new config (the user chose continuation over output purity —
        a settings tweak must not throw away hours of finished frames). The
        explicit refresh action remains the way to re-render everything under
        new settings. Identity and size still re-render: a different source or
        target is a different render (not a tweak), and mixed frame sizes
        would break the encode (the size token covers processing-scale)."""
        digest = hashlib.sha1(
            f"{task.source_path}|{task.target_path}".encode()
        ).hexdigest()[:10]
        return [
            task_cache / f"stage{i}-{spec.name}@{size_token}-{digest}"
            for i, spec in enumerate(stages)
        ]

    @staticmethod
    def _chain_fingerprint(task: BatchTask, size_token: str) -> str:
        """Stable hash of everything the stage-dir token depends on (source /
        target / output size — NOT stage params, which resume in place; see
        _stage_cache_dirs). Persisted on the task so a resume whose identity
        changed since the cached run resets the stale resume markers instead
        of trusting them."""
        parts = [str(task.source_path), str(task.target_path), size_token]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

    # ---- Helpers ----

    @staticmethod
    def _stage_complete(stage_dir: Path, ext: str, total: int) -> bool:
        """True iff every frame for this stage is already valid on disk."""
        return all(
            frame_ok(stage_dir / f"{i:08d}.{ext}") for i in range(total)
        )

    @staticmethod
    def _stage_progress(
        progress_callback: ProgressCallback | None,
        stage_index: int,
        stage_count: int,
        stage_name: str,
        total: int,
    ) -> Callable[[int], None] | None:
        """Adapt run_stage's int (frames done in THIS stage) into a
        BatchProgress carrying stage position + overall frame-units."""
        if progress_callback is None:
            return None

        def emit(stage_completed: int) -> None:
            progress_callback(
                BatchProgress(
                    stage_index=stage_index,
                    stage_count=stage_count,
                    stage_name=stage_name,
                    stage_completed=stage_completed,
                    stage_total=total,
                    overall_completed=stage_index * total + stage_completed,
                    overall_total=stage_count * total,
                )
            )

        return emit

    @staticmethod
    def _stage_failed_message(
        stage_name: str, missing: list[int], error: str | None = None
    ) -> str:
        preview = ", ".join(str(i) for i in missing[:8])
        more = "" if len(missing) <= 8 else f", +{len(missing) - 8} more"
        cause = f" First error: {error}." if error else ""
        return (
            f"stage '{stage_name}': {len(missing)} frame(s) missing or empty "
            f"({preview}{more}); refusing to encode a truncated video.{cause} "
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
