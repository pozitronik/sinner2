"""Build a target's identity catalog by a strided scan.

Samples the target on a configurable stride, detects faces (buffalo_l → ArcFace
embeddings), and online-clusters them into a FaceMap. Each identity's clearest
occurrence (highest det_score) is recorded as ``ref_frame``/``ref_bbox`` so the
UI can extract a representative thumbnail later. Headless + cancellable; the GUI
runs it on a worker thread with a progress callback.

The detect function is injected (a closure over the shared analyser in the app,
a stub in tests) so this module needs no models and the analyser's lifecycle
stays with its owner.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from sinner2.io.target_reader import TargetReader
from sinner2.pipeline.face_map import FaceMap
from sinner2.pipeline.sections import SectionSet
from sinner2.types import Frame

# (frames_scanned, frames_to_scan)
ProgressFn = Callable[[int, int], None]
# frame -> faces, each with normed_embedding / det_score / bbox (buffalo_l)
DetectFn = Callable[[Any], list]
PreviewFn = Callable[[Frame], None]
PositionFn = Callable[[int], None]  # the frame index currently being scanned

# (best_score, frame, bbox, sex, age) of an identity's clearest occurrence.
_Rep = tuple[float, int, tuple[float, float, float, float], "str | None", "int | None"]


class _Counter:
    """Sampled-position counter; starts at the resume offset so progress + the
    returned ``done`` reflect total work, not just this run's slice."""

    def __init__(self, start: int) -> None:
        self.done = start

    def tick(self) -> int:
        self.done += 1
        return self.done


class _ClusterState:
    """Accumulates the catalog as detected faces arrive. ``ingest`` MUST be
    called from a single thread (it mutates the FaceMap); the detection that
    feeds it can run in parallel."""

    def __init__(self, face_map: FaceMap) -> None:
        self.face_map = face_map
        # identity id -> (best_score, frame, bbox, sex, age) of its clearest
        # occurrence — drives the thumbnail + the displayed demographics.
        self.reps: dict[str, _Rep] = {}
        # identity id -> earliest frame it was seen at (drives navigation).
        self.firsts: dict[str, int] = {}
        # Seed from an existing catalog so a RESUMED scan keeps its identities,
        # occurrence counts, and references instead of starting blank. The
        # seeded rep uses score 0 so the persisted thumbnail is kept unless the
        # resumed portion finds the person again (a real det_score beats 0).
        for ident in face_map.identities:
            if ident.ref_frame is not None and ident.ref_bbox is not None:
                self.reps[ident.id] = (
                    0.0, ident.ref_frame, ident.ref_bbox, ident.sex, ident.age,
                )
            if ident.first_frame is not None:
                self.firsts[ident.id] = ident.first_frame

    def ingest(self, frame_idx: int, faces: list) -> None:
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            self.face_map, joined_id = self.face_map.observe_with_id(emb)
            self.firsts.setdefault(joined_id, int(frame_idx))
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            prev = self.reps.get(joined_id)
            if prev is None or score > prev[0]:
                age_raw = getattr(face, "age", None)
                self.reps[joined_id] = (
                    score,
                    int(frame_idx),
                    (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    getattr(face, "sex", None),
                    int(age_raw) if age_raw is not None else None,
                )

    def finish(self) -> FaceMap:
        face_map = self.face_map
        for ident_id, (_score, frame_idx, bbox, sex, age) in self.reps.items():
            face_map = face_map.with_reference(
                ident_id, frame_idx, bbox, sex=sex, age=age,
                first_frame=self.firsts.get(ident_id),
            )
        return face_map


def analyze_target(
    reader: TargetReader,
    detect: DetectFn,
    *,
    stride: int = 15,
    threshold: float = 0.5,
    sections: SectionSet | None = None,
    workers: int = 1,
    start_index: int = 0,
    initial: FaceMap | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: ProgressFn | None = None,
    on_preview: PreviewFn | None = None,
    on_position: PositionFn | None = None,
    preview_interval: float = 0.1,
) -> tuple[FaceMap, int, int]:
    """Scan ``reader`` on ``stride`` and cluster detected faces into a FaceMap.

    A face with no embedding (a detection-only detector handed in by mistake) is
    skipped — embeddings are what make the clustering identity-stable. With
    ``sections`` non-empty the scan is confined to the selected frames (stridden
    within them), so the catalog only covers the parts you kept. ``on_preview``
    (throttled to ``preview_interval``) surfaces the frame being scanned so the
    GUI can show progress on the display. ``workers`` > 1 detects frames in
    parallel (reads stay single-threaded; clustering stays serial) so the GPU
    saturates instead of idling between sequential detections. Returns the
    catalog with every identity's representative occurrence attached; an early
    cancel returns the partial catalog built so far.

    ``start_index`` + ``initial`` RESUME a prior scan: skip the first
    ``start_index`` sampled positions and seed the clustering with ``initial``
    (its identities, counts, refs) so an interrupted analysis continues instead
    of restarting. ``on_position`` reports the frame index being scanned (for a
    live playhead). Returns ``(catalog, scanned, total)`` — ``scanned`` is how
    many sampled positions completed (for resume bookkeeping), ``total`` the
    full count.
    """
    total_frames = max(0, reader.frame_count)
    stride = max(1, stride)
    if sections is not None and not sections.is_empty():
        # Sample only the SELECTED frames (stridden) — respects the parts trim.
        indices = sections.frame_plan(total_frames)[::stride]
    else:
        indices = list(range(0, total_frames, stride)) or ([0] if total_frames else [])
    total = len(indices)
    start_index = max(0, min(start_index, total))
    base = (initial if initial is not None else FaceMap.empty()).with_threshold(threshold)
    state = _ClusterState(base)
    counter = _Counter(start_index)

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    to_scan = indices[start_index:]
    if workers <= 1:
        _scan_serial(
            reader, detect, to_scan, state, counter, total, cancelled,
            on_progress, on_preview, on_position, preview_interval,
        )
    else:
        _scan_parallel(
            reader, detect, to_scan, state, workers, counter, total, cancelled,
            on_progress, on_preview, on_position, preview_interval,
        )
    return state.finish(), counter.done, total


def _maybe_preview(
    on_preview: PreviewFn | None, frame: Frame, last: float, interval: float
) -> float:
    if on_preview is None:
        return last
    now = time.monotonic()
    if now - last >= interval:
        on_preview(frame)
        return now
    return last


def _scan_serial(
    reader: TargetReader,
    detect: DetectFn,
    to_scan: list[int],
    state: _ClusterState,
    counter: _Counter,
    total: int,
    cancelled: Callable[[], bool],
    on_progress: ProgressFn | None,
    on_preview: PreviewFn | None,
    on_position: PositionFn | None,
    preview_interval: float,
) -> None:
    preview_last = 0.0
    for frame_idx in to_scan:
        if cancelled():
            break
        if on_position is not None:
            on_position(frame_idx)
        frame = reader.read(frame_idx)
        if frame is not None:
            preview_last = _maybe_preview(
                on_preview, frame, preview_last, preview_interval
            )
            state.ingest(frame_idx, detect(frame))
        done = counter.tick()
        if on_progress is not None:
            on_progress(done, total)


def _scan_parallel(
    reader: TargetReader,
    detect: DetectFn,
    to_scan: list[int],
    state: _ClusterState,
    workers: int,
    counter: _Counter,
    total: int,
    cancelled: Callable[[], bool],
    on_progress: ProgressFn | None,
    on_preview: PreviewFn | None,
    on_position: PositionFn | None,
    preview_interval: float,
) -> None:
    """Reads on THIS thread (sequential decode — the cv2 reader isn't thread
    safe), runs detection in a worker pool (the shared buffalo_l ORT session is
    thread-safe for concurrent inference, so N detections overlap and keep the
    GPU busy — same idea as the live/batch worker pools), and clusters each
    result here, serially, as it lands. Bounded in-flight so memory stays flat.

    Reads (hence ``on_position``) advance in frame order on this one thread, so
    the reported position is naturally the largest index in flight."""
    cap = max(2, workers * 2)
    preview_last = 0.0
    inflight: deque[tuple[int, Future]] = deque()

    def drain_one() -> None:
        frame_idx, fut = inflight.popleft()
        state.ingest(frame_idx, fut.result())
        done = counter.tick()
        if on_progress is not None:
            on_progress(done, total)

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="sinner2-facemap"
    ) as pool:
        for frame_idx in to_scan:
            if cancelled():
                break
            if on_position is not None:
                on_position(frame_idx)
            frame = reader.read(frame_idx)
            if frame is None:
                done = counter.tick()
                if on_progress is not None:
                    on_progress(done, total)
                continue
            preview_last = _maybe_preview(
                on_preview, frame, preview_last, preview_interval
            )
            inflight.append((frame_idx, pool.submit(detect, frame)))
            while len(inflight) >= cap:
                drain_one()
        # Drain the tail (unless cancelling — then abandon; the pool's exit waits
        # for the few still running and their results are simply discarded).
        while inflight and not cancelled():
            drain_one()
