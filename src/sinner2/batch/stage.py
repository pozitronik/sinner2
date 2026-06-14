"""Processor-major stage execution for batch processing.

A *stage* runs ONE processor over ALL frames of an input, resume-aware, and
writes validated output frames to a directory. This is the throughput-
optimized counterpart to the realtime (frame-major) executor: one model is
resident at a time, so the device does one kind of work and peak VRAM stays
low.

Design choices that matter:

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
    the media; the stage's effective total shrinks to that point.
  * Live preview (`on_preview`): a throttled callback fires with a recently
    processed frame so the GUI can show what the batch is producing without
    flooding it (one frame every `preview_interval` seconds).

The processor is set up before the run and released after, so only this
stage's model is resident while it runs.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from sinner2.io.cv2_unicode import imread_unicode
from sinner2.io.target_reader import TargetReader
from sinner2.pipeline.image_writer import ImageWriter
from sinner2.pipeline.processor import Processor
from sinner2.types import Frame, FrameIndex

_log = logging.getLogger(__name__)


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
    # First per-frame exception (type + message), carried so the driver can
    # tell the user WHY the stage failed instead of just "frames missing".
    error: str | None = None


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


class PlanReaderStageInput:
    """First-stage input over a SELECTION: maps an output position ``p`` to the
    original source frame ``plan[p]`` (an ascending list of included indices).
    The output is renumbered contiguous (0..len(plan)-1) so a multi-range trim
    encodes into one continuous clip. With no selection ``plan`` is the full
    range and this behaves exactly like ReaderStageInput.

    Reads stay ascending in the source (plan is sorted), so a video reader
    decodes sequentially — gaps just skip forward, no random-seek thrash."""

    def __init__(self, reader: TargetReader, plan: list[int]) -> None:
        self._reader = reader
        self._plan = plan

    @property
    def frame_count(self) -> int:
        return len(self._plan)

    def read(self, index: FrameIndex) -> Frame | None:
        return self._reader.read(self._plan[index])

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


class _ContiguousCounter:
    """Tracks the contiguous-from-0 count of completed frames as they finish OUT
    OF ORDER. ``value`` is the number of frames done with no gap from index 0 —
    the durable resume high-water mark — as opposed to a raw completion count,
    which can momentarily run ahead of that prefix under multiple workers."""

    def __init__(self, pending: set[int], total: int) -> None:
        self._pending = pending
        self._total = total
        self._cursor = 0
        self._advance()

    def _advance(self) -> None:
        while self._cursor < self._total and self._cursor not in self._pending:
            self._cursor += 1

    @property
    def value(self) -> int:
        return self._cursor

    def complete(self, idx: int) -> int:
        self._pending.discard(idx)
        if idx == self._cursor:
            self._advance()
        return self._cursor


def run_stage(
    *,
    stage_input: StageInput,
    processor_factory: Callable[[], Processor],
    thread_safe: bool,
    output_dir: Path,
    ext: str,
    writer: ImageWriter,
    workers: int,
    pause_event: threading.Event,
    cancel_event: threading.Event,
    on_progress: Callable[[int], None] | None = None,
    eof_on_none: bool = False,
    on_preview: Callable[[Frame], None] | None = None,
    preview_interval: float = 0.5,
) -> StageResult:
    """Run a processor over every frame of ``stage_input``, writing
    ``output_dir/{idx:08d}.{ext}``. Resumes by skipping frames already valid
    on disk. ``on_progress(done_count)`` fires as frames complete;
    ``on_preview(frame)`` fires at most once per ``preview_interval`` seconds
    with a just-processed frame. With ``eof_on_none`` a None read ends the
    stage early; StageResult.total reports the real count.

    The processor is created via ``processor_factory`` and its lifecycle
    (setup → process* → release) is owned here. ``thread_safe`` decides how
    many instances back the worker pool: a thread-safe processor (e.g.
    FaceSwapper's shared ORT session) is built ONCE and used concurrently by
    all workers; a non-thread-safe one (e.g. FaceEnhancer/GFPGAN, which mutates
    torch state) gets ONE instance per worker, leased so no instance is ever
    used by two workers at once — that's what lets it actually run in parallel
    instead of serializing on an internal lock."""
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
    # Report the contiguous-from-0 prefix (the durable resume point), not a raw
    # completion count: with multiple workers frames finish out of order, so a
    # plain count can momentarily exceed the gap-free prefix and overstate the
    # persisted last_completed_frame on a mid-task restart.
    counter = _ContiguousCounter(set(pending), total)
    if on_progress is not None:
        on_progress(counter.value)

    def bump(idx: int) -> None:
        v = counter.complete(idx)
        if on_progress is not None:
            on_progress(v)

    errors: list[str] = []

    def record_error(exc: BaseException) -> None:
        # The output gap is caught by the integrity pass, but the *cause* would
        # otherwise vanish. Log it (with the traceback) and keep the first one
        # to surface on the result.
        _log.warning("stage frame processing error: %s", exc, exc_info=exc)
        if not errors:
            errors.append(f"{type(exc).__name__}: {exc}")

    preview_last = 0.0

    def maybe_preview(frame: Frame | None) -> None:
        # Throttled live preview. The submit loop drains on a single thread,
        # so this needs no lock.
        nonlocal preview_last
        if on_preview is None or frame is None:
            return
        now = time.monotonic()
        if now - preview_last >= preview_interval:
            preview_last = now
            on_preview(frame)

    pool = _ProcessorPool(processor_factory, thread_safe, workers)
    try:
        status, eof_at = _feed(
            pending, stage_input, pool, output_dir, ext, writer,
            workers, pause_event, cancel_event, bump, eof_on_none,
            maybe_preview, record_error,
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
                remaining, stage_input, pool, output_dir, ext, writer,
                workers, pause_event, cancel_event, bump, False,
                maybe_preview, record_error,
            )
            if status is StageStatus.CANCELLED:
                return StageResult(status, contiguous(total), total=total)
            remaining = missing(total)
        if remaining:
            return StageResult(
                StageStatus.FAILED, contiguous(total), remaining, total=total,
                error=errors[0] if errors else None,
            )
        return StageResult(StageStatus.COMPLETED, total, total=total)
    finally:
        pool.release()


class _ProcessorPool:
    """Owns the processor instance(s) for one stage run and leases them to
    worker tasks. Built and torn down by run_stage; not reused across stages.

    thread_safe=True  → ONE instance, shared by every worker concurrently
                        (e.g. FaceSwapper's shared ORT session).
    thread_safe=False → ONE instance per worker, each leased to at most one
                        task at a time (e.g. FaceEnhancer/GFPGAN). Because the
                        worker pool has exactly ``workers`` threads and there
                        are ``workers`` instances, a lease never blocks.
    """

    def __init__(
        self,
        factory: Callable[[], Processor],
        thread_safe: bool,
        workers: int,
    ) -> None:
        self._thread_safe = thread_safe
        self._free: queue.SimpleQueue[Processor] | None = None
        self._instances: list[Processor] = []
        count = 1 if thread_safe else max(1, workers)
        try:
            for _ in range(count):
                proc = factory()
                proc.setup()
                self._instances.append(proc)
        except Exception:
            # A partial build (e.g. 2nd GFPGAN OOMs) must not leak the
            # instances that did load — release them before propagating.
            self.release()
            raise
        if not thread_safe:
            self._free = queue.SimpleQueue()
            for proc in self._instances:
                self._free.put(proc)

    @contextmanager
    def lease(self) -> Iterator[Processor]:
        if self._thread_safe or self._free is None:
            yield self._instances[0]
            return
        proc = self._free.get()
        try:
            yield proc
        finally:
            self._free.put(proc)

    def release(self) -> None:
        for proc in self._instances:
            try:
                proc.release()
            except Exception:
                pass
        self._instances = []


def _feed(
    indices: list[int],
    stage_input: StageInput,
    pool: _ProcessorPool,
    output_dir: Path,
    ext: str,
    writer: ImageWriter,
    workers: int,
    pause_event: threading.Event,
    cancel_event: threading.Event,
    bump: Callable[[int], None],
    eof_on_none: bool,
    maybe_preview: Callable[[Frame | None], None],
    on_error: Callable[[BaseException], None],
) -> tuple[StageStatus, int | None]:
    """Submit ``indices`` through the processor pool. Reads happen here on a
    single thread (sequential decode); workers only process + write. Returns
    the interrupt status and, if eof_on_none hit a None read, the EOF index."""
    inflight: list[tuple[int, Future]] = []
    cap = max(2, workers * 2)
    interrupted: StageStatus | None = None
    eof_at: int | None = None
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="sinner2-batch-stage"
    ) as executor:
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
                (idx, executor.submit(_process_write, pool, frame, out_path, writer))
            )
            if len(inflight) >= cap:
                _drain_one(inflight, bump, maybe_preview, on_error)
        while inflight:
            _drain_one(inflight, bump, maybe_preview, on_error)
    return (interrupted or StageStatus.COMPLETED), eof_at


def _drain_one(
    inflight: list[tuple[int, Future]],
    bump: Callable[[int], None],
    maybe_preview: Callable[[Frame | None], None],
    on_error: Callable[[BaseException], None],
) -> None:
    done, _pending = wait([f for _, f in inflight], return_when="FIRST_COMPLETED")
    for fut in done:
        entry = next(e for e in inflight if e[1] is fut)
        inflight.remove(entry)
        idx = entry[0]
        # Per-frame errors don't stop the stage: the missing output is caught by
        # the integrity pass, which reprocesses then fails loudly if persistent.
        # But the cause must not vanish — on_error logs it and retains the first
        # one so a persistent failure surfaces WHY, not just "frames missing".
        exc = fut.exception()
        if exc is None:
            bump(idx)
            maybe_preview(fut.result())
        else:
            on_error(exc)


def _process_write(
    pool: _ProcessorPool, frame: Frame, out_path: Path, writer: ImageWriter
) -> Frame:
    # Hold the lease only across process(); the disk write doesn't touch the
    # processor, so release the instance back to the pool before writing to
    # keep non-thread-safe instances available to other workers sooner.
    with pool.lease() as processor:
        result = processor.process(frame)
    writer.write(out_path, result)
    return result
