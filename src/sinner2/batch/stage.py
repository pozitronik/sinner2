"""Processor-major stage execution for batch processing.

A *stage* runs ONE processor over ALL frames of an input, resume-aware, and
writes validated output frames to a directory. This is the throughput-
optimized counterpart to the realtime (frame-major) executor: one model is
resident at a time, so the device does one kind of work and peak VRAM stays
low.

Three design choices matter:

  * The frame READ happens on the single submit-loop thread, not in workers.
    For the first stage (video source) that means decode streams in index
    order — no random-seek thrash — and the non-thread-safe readers are never
    touched concurrently. Workers only run processor.process() + write.
  * Resume and integrity are disk-truth. A frame counts as done iff its
    output file exists AND is non-empty. After the main pass, any missing or
    zero-byte frame gets one reprocess pass; if any remain, the stage fails
    loudly rather than handing a gappy sequence to the encoder.
  * End-of-stream tolerance (`eof_on_none`): a streaming video source can
    decode FEWER frames than its container metadata (ffprobe nb_frames)
    claims. With eof_on_none the first None read is taken as the true end of
    the media; the stage's effective total shrinks to that point and the
    phantom trailing indices are not treated as "missing". The driver decides
    whether the shortfall is benign (metadata glitch) or alarming (truncated
    source).

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
    total: int = 0  # effective frame count (may shrink from EOF tolerance)


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
    eof_on_none: bool = False,
) -> StageResult:
    """Run ``processor`` over every frame of ``stage_input``, writing
    ``output_dir/{idx:08d}.{ext}``. Resumes by skipping frames already valid
    on disk. ``on_progress(done_count)`` fires as frames complete. With
    ``eof_on_none`` a None read ends the stage early (streaming source shorter
    than its claimed frame_count); StageResult.total reports the real count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = stage_input.frame_count
    workers = max(1, workers)

    def missing(t: int) -> list[int]:
        return [
            i for i in range(t)
            if not frame_ok(output_dir / f"{i:08d}.{ext}")
        ]

    def contiguous(t: int) -> int:
        for i in range(t):
            if not frame_ok(output_dir / f"{i:08d}.{ext}"):
                return i
        return t

    pending = missing(total)
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
        status, eof_at = _feed(
            pending, stage_input, processor, output_dir, ext, writer,
            workers, pause_event, cancel_event, bump, eof_on_none,
        )
        if status in (StageStatus.PAUSED, StageStatus.CANCELLED):
            return StageResult(status, contiguous(total), total=total)

        if eof_at is not None:
            # Streaming source ended early — the real total is the EOF index.
            total = eof_at

        # Integrity: one reprocess pass over anything still missing/zero-byte
        # within the (possibly shrunk) real range.
        remaining = missing(total)
        if remaining:
            status, _ = _feed(
                remaining, stage_input, processor, output_dir, ext, writer,
                workers, pause_event, cancel_event, bump, eof_on_none=False,
            )
            if status is StageStatus.CANCELLED:
                return StageResult(status, contiguous(total), total=total)
            remaining = missing(total)
        if remaining:
            return StageResult(
                StageStatus.FAILED, contiguous(total), remaining, total=total
            )
        return StageResult(StageStatus.COMPLETED, total, total=total)
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
    eof_on_none: bool,
) -> tuple[StageStatus, int | None]:
    """Submit ``indices`` through the processor pool. Reads happen here on a
    single thread (sequential decode); workers only process + write. Returns
    the interrupt status and, if eof_on_none hit a None read, the EOF index."""
    inflight: list[Future] = []
    cap = max(2, workers * 2)
    interrupted: StageStatus | None = None
    eof_at: int | None = None
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
                if eof_on_none:
                    # Streaming source exhausted — everything from idx on is
                    # past the real end of the media. Stop here.
                    eof_at = idx
                    break
                continue  # gap within the real range; integrity catches it
            out_path = output_dir / f"{idx:08d}.{ext}"
            inflight.append(
                pool.submit(_process_write, processor, frame, out_path, writer)
            )
            if len(inflight) >= cap:
                _drain_one(inflight, bump)
        while inflight:
            _drain_one(inflight, bump)
    return (interrupted or StageStatus.COMPLETED), eof_at


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
