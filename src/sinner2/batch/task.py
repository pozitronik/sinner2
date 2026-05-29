"""Batch-task schema + output-path resolution.

One BatchTask = one queued job: source + target + full session config snapshot
+ runtime state. Persisted as a single JSON file per task by BatchTaskStore,
so the on-disk format is the authoritative source of truth (the GUI just
reads/writes through the store).

Design notes:
  - Config fields mirror the Settings subset that the realtime preview
    exposes, so "Add current state to batch" is just a field-by-field
    copy. The chain knobs (swapper / enhancer params + enhancer_enabled)
    affect the OUTPUT pixels; execution knobs (worker_count, providers,
    video_backend, reader_pool_size) affect throughput, not pixels.
  - Output path: Path | None. None = auto-derive via resolve_output_path()
    using the global default (next to target) or the user's configured
    batch_global_output_path.
  - Runtime state lives on the same model so a single save() round-trips
    everything. Cheaper than tracking it separately in memory.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.image_writer import ImageFormat


class BatchTaskStatus(str, Enum):
    """Per-task lifecycle. Stored as the string token so settings files
    round-trip cleanly across sinner1/2 versions (matches Settings str-Enum
    convention)."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BatchOutputFormat(str, Enum):
    """Output kind. VIDEO requires ffmpeg; the driver falls back to FRAMES
    when ffmpeg is missing and records a note in error_message."""

    VIDEO = "video"
    FRAMES = "frames"


class BatchCleanupMode(str, Enum):
    """When to delete per-stage intermediate frame directories.

    KEEP         — never delete (default); every stage dir + the final
                   output remain. Pure disk-truth resume, highest disk use.
    AUTO         — delete stage N-1 once stage N completes (peak ≈ 2 stages).
                   Needs the completed_stages marker for cross-restart resume
                   since the intermediate dirs are gone.
    DROP_AT_END  — keep all stage dirs during the run, delete them once the
                   final output validates. Pure disk-truth, drops at the end.

    The final output file is never deleted by any mode.
    """

    KEEP = "keep"
    AUTO = "auto"
    DROP_AT_END = "drop_at_end"


class BatchExtractionMode(str, Enum):
    """How the first stage gets frames from a video source.

    STREAM      — decode the source sequentially on demand (default).
    PREEXTRACT  — bulk-extract all frames to disk first, then run the first
                  stage off those files. Not implemented in v1; the field is
                  the seam (the driver rejects PREEXTRACT for now).
    """

    STREAM = "stream"
    PREEXTRACT = "preextract"


def _new_id() -> str:
    # Short uuid4 — full uuid is 32 hex chars; the leading 12 are plenty
    # to disambiguate across thousands of tasks and shorter for filenames.
    return uuid.uuid4().hex[:12]


class BatchTask(SinnerBaseModel):
    """One queued processing job. Persisted as <store_root>/<id>.json."""

    # ---- Identity ----
    id: str = Field(default_factory=_new_id)
    source_path: Path
    target_path: Path
    output_path: Path | None = None
    output_format: BatchOutputFormat = BatchOutputFormat.VIDEO

    # ---- Chain config (output-affecting) ----
    swapper_detection_interval: int = 1
    swapper_many_faces: bool = True
    swapper_target_sex: str = "B"  # M/F/B/I (matches FaceSwapperParams)
    enhancer_enabled: bool = True
    enhancer_upscale: int = 1
    enhancer_only_center_face: bool = False

    # ---- Execution config (throughput-affecting) ----
    worker_count: int = 1
    video_backend: VideoBackend = VideoBackend.FFMPEG
    reader_pool_size: int = 1
    onnx_providers: list[str] | None = None

    # ---- Output / cache config (used by frames mode + ffmpeg input glob) ----
    image_format: ImageFormat = ImageFormat.JPEG
    image_quality: int = 95

    # ---- Stage execution config (processor-major batch) ----
    cleanup_mode: BatchCleanupMode = BatchCleanupMode.KEEP
    extraction_mode: BatchExtractionMode = BatchExtractionMode.STREAM

    # ---- Runtime state ----
    status: BatchTaskStatus = BatchTaskStatus.PENDING
    last_completed_frame: int = -1
    total_frames: int = -1
    completed_stages: int = 0  # fully-validated stages (AUTO-cleanup resume)
    error_message: str | None = None
    started_at: float | None = None  # epoch seconds
    finished_at: float | None = None


def resolve_output_path(
    task: BatchTask, global_output_dir: Path | None = None
) -> Path:
    """Compute the final output path for a task.

    Precedence (most-specific wins):
      1. task.output_path — explicit per-task override.
      2. global_output_dir — folder for all batch outputs, with auto name.
      3. fallback — next to the target, with auto name.

    Auto name: "{source_stem}+{target_stem}.{ext}", where ext is "mp4"
    for VIDEO and the source's existing extension preserved for FRAMES
    (frames-mode output is a directory, so the .mp4 suffix would be
    misleading; we just use the source stem as the folder name).
    """
    if task.output_path is not None:
        return task.output_path
    base_name = f"{task.source_path.stem}+{task.target_path.stem}"
    if task.output_format is BatchOutputFormat.VIDEO:
        filename = f"{base_name}.mp4"
    else:
        # Frames mode: output is a directory of image files.
        filename = base_name
    if global_output_dir is not None:
        return global_output_dir / filename
    return task.target_path.parent / filename


@dataclass(frozen=True)
class BatchProgress:
    """Live progress for a running task: position within the current stage
    plus overall frame-units across all stages.

    overall_completed / overall_total advances monotonically; it is NOT
    linear in wall-clock time (stages differ in per-frame cost), so it is a
    work-units gauge, not an ETA.
    """

    stage_index: int  # 0-based; also the count of fully-done prior stages
    stage_count: int
    stage_name: str
    stage_completed: int
    stage_total: int
    overall_completed: int
    overall_total: int

    @property
    def overall_fraction(self) -> float:
        return (
            self.overall_completed / self.overall_total
            if self.overall_total
            else 0.0
        )
