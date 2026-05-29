"""Processor-major stage execution for batch processing.

A *stage* runs ONE processor over ALL frames of an input, resume-aware, and
writes validated output frames to a directory. This is the throughput-
optimized counterpart to the realtime (frame-major) executor: one model is
resident at a time, so the device does one kind of work and peak VRAM stays
low.

Two design choices matter:

  * The frame READ happens on the single submit-loop thread, not in workers.
    For the first stage (video source) that means decode streams in index
    order — no random-seek thrash — and the non-thread-safe readers are never
    touched concurrently. Workers only run processor.process() + write.
  * Resume and integrity are disk-truth. A frame counts as done iff its
    output file exists AND is non-empty. After the main pass, any missing or
    zero-byte frame gets one reprocess pass; if any remain, the stage fails
    loudly rather than handing a gappy sequence to the encoder (ffmpeg
    ``-i %08d.ext`` would silently truncate at the first gap).

The processor is set up before the run and released after, so only this
stage's model is resident while it runs.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from sinner2.io.cv2_unicode import imread_unicode
from sinner2.io.target_reader import TargetReader
from sinner2.pipeline.image_writer import ImageWriter
from sinner2.pipeline.processor import Processor
from sinner2.types import Frame, FrameIndex


def frame_ok(path: Path) -> bool:
    """A frame counts as done iff it exists AND is non-empty. Zero-byte
    files (e.g. disk full mid-write) must be reprocessed, never encoded."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


class StageStatus(str, Enum):
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class StageResult:
    status: StageStatus
    completed_frames: int  # contiguous valid frames from index 0
    missing: list[int] = field(default_factory=list)  # populated on FAILED


class StageInput(Protocol):
    """Source of frames for one stage. read() is called only from the
    stage's single submit-loop thread, in ascending index order."""

    @property
    def frame_count(self) -> int: ...
    def read(self, index: FrameIndex) -> Frame | None: ...
    def close(self) -> None: ...


class ReaderStageInput:
    """First-stage input: wraps any TargetReader (image or video). Because
    the submit loop reads ascending indices, a video reader stays sequential
    (no per-frame seeks)."""

    def __init__(self, reader: TargetReader) -> None:
        self._reader = reader

    @property
    def frame_count(self) -> int:
        return self._reader.frame_count

    def read(self, index: FrameIndex) -> Frame | None:
        return self._reader.read(index)

    def close(self) -> None:
        self._reader.release()


class FramesDirInput:
    """Later-stage input: reads the previous stage's output frames from a
    directory of zero-padded image files."""

    def __init__(self, directory: Path, ext: str, frame_count: int) -> None:
        self._dir = directory
        self._ext = ext
        self._frame_count = frame_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read(self, index: FrameIndex) -> Frame | None:
        return imread_unicode(self._dir / f"{index:08d}.{self._ext}")

    def close(self) -> None:
        pass


def run_stage(
    *,
    stage_input: StageInput,
    processor: Processor,
    output_dir: Path,
    ext: str,
    writer: ImageWriter,
    workers: int,
    pause_event: threading.Event,
    cancel_event: threading.Event,
    on_progress: Callable[[int], None] | None = None,
) -> StageResult:
    """Run ``processor`` over every frame of ``stage_input``, writing
    ``output_dir/{idx:08d}.{ext}``. Resumes by skipping frames already valid
    on disk. ``on_progress(done_count)`` fires as frames complete."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = stage_input.frame_count
    workers = max(1, workers)

    def contiguous() -> int:
        for i in range(total):
            if not frame_ok(output_dir / f"{i:08d}.{ext}"):
                return i
        return total

    def missing() -> list[int]:
        return [
            i for i in range(total)
            if not frame_ok(output_dir / f"{i:08d}.{ext}")
        ]

    pending = missing()
    done = total - len(pending)
    if on_progress is not None:
        on_progress(done)

    def bump() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done)

    processor.setup()
    try:
        status = _feed(
            pending, stage_input, processor, output_dir, ext, writer,
            workers, pause_event, cancel_event, bump,
        )
        if status in (StageStatus.PAUSED, StageStatus.CANCELLED):
            return StageResult(status, contiguous())

        # Integrity: one reprocess pass over anything still missing/zero-byte.
        remaining = missing()
        if remaining:
            status = _feed(
                remaining, stage_input, processor, output_dir, ext, writer,
                workers, pause_event, cancel_event, bump,
            )
            if status is StageStatus.CANCELLED:
                return StageResult(status, contiguous())
            remaining = missing()
        if remaining:
            return StageResult(StageStatus.FAILED, contiguous(), remaining)
        return StageResult(StageStatus.COMPLETED, total)
    finally:
        processor.release()


def _feed(
    indices: list[int],
    stage_input: StageInput,
    processor: Processor,
    output_dir: Path,
    ext: str,
    writer: ImageWriter,
    workers: int,
    pause_event: threading.Event,
    cancel_event: threading.Event,
    bump: Callable[[], None],
) -> StageStatus:
    """Submit ``indices`` through the processor pool. Reads happen here on a
    single thread (sequential decode); workers only process + write. In-flight
    work is allowed to finish so the on-disk cache stays consistent."""
    inflight: list[Future] = []
    cap = max(2, workers * 2)
    interrupted: StageStatus | None = None
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="sinner2-batch-stage"
    ) as pool:
        for idx in indices:
            if cancel_event.is_set():
                interrupted = StageStatus.CANCELLED
                break
            if pause_event.is_set():
                interrupted = StageStatus.PAUSED
                break
            frame = stage_input.read(idx)  # single-thread sequential read
            if frame is None:
                continue  # gap; the integrity pass will catch it
            out_path = output_dir / f"{idx:08d}.{ext}"
            inflight.append(
                pool.submit(_process_write, processor, frame, out_path, writer)
            )
            if len(inflight) >= cap:
                _drain_one(inflight, bump)
        while inflight:
            _drain_one(inflight, bump)
    return interrupted or StageStatus.COMPLETED


def _drain_one(inflight: list[Future], bump: Callable[[], None]) -> None:
    done, _pending = wait(inflight, return_when="FIRST_COMPLETED")
    for fut in done:
        inflight.remove(fut)
        # Per-frame errors are swallowed: the missing output is caught by the
        # integrity pass, which reprocesses then fails loudly if persistent.
        if fut.exception() is None:
            bump()


def _process_write(
    processor: Processor, frame: Frame, out_path: Path, writer: ImageWriter
) -> None:
    writer.write(out_path, processor.process(frame))
