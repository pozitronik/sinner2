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


class _ClusterState:
    """Accumulates the catalog as detected faces arrive. ``ingest`` MUST be
    called from a single thread (it mutates the FaceMap); the detection that
    feeds it can run in parallel."""

    def __init__(self, face_map: FaceMap) -> None:
        self.face_map = face_map
        # identity id -> (best_score, frame, bbox) for the thumbnail reference.
        self.reps: dict[str, tuple[float, int, tuple[float, float, float, float]]] = {}

    def ingest(self, frame_idx: int, faces: list) -> None:
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            self.face_map, joined_id = self.face_map.observe_with_id(emb)
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            prev = self.reps.get(joined_id)
            if prev is None or score > prev[0]:
                self.reps[joined_id] = (
                    score,
                    int(frame_idx),
                    (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )

    def finish(self) -> FaceMap:
        face_map = self.face_map
        for ident_id, (_score, frame_idx, bbox) in self.reps.items():
            face_map = face_map.with_reference(ident_id, frame_idx, bbox)
        return face_map


def analyze_target(
    reader: TargetReader,
    detect: DetectFn,
    *,
    stride: int = 15,
    threshold: float = 0.5,
    sections: SectionSet | None = None,
    workers: int = 1,
    cancel_event: threading.Event | None = None,
    on_progress: ProgressFn | None = None,
    on_preview: PreviewFn | None = None,
    preview_interval: float = 0.3,
) -> FaceMap:
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
    """
    total = max(0, reader.frame_count)
    stride = max(1, stride)
    if sections is not None and not sections.is_empty():
        # Sample only the SELECTED frames (stridden) — respects the parts trim.
        indices = sections.frame_plan(total)[::stride]
    else:
        indices = list(range(0, total, stride)) or ([0] if total else [])
    state = _ClusterState(FaceMap.empty().with_threshold(threshold))

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if workers <= 1:
        _scan_serial(
            reader, detect, indices, state, cancelled,
            on_progress, on_preview, preview_interval,
        )
    else:
        _scan_parallel(
            reader, detect, indices, state, workers, cancelled,
            on_progress, on_preview, preview_interval,
        )
    return state.finish()


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
    indices: list[int],
    state: _ClusterState,
    cancelled: Callable[[], bool],
    on_progress: ProgressFn | None,
    on_preview: PreviewFn | None,
    preview_interval: float,
) -> None:
    preview_last = 0.0
    total = len(indices)
    for done, frame_idx in enumerate(indices, start=1):
        if cancelled():
            break
        frame = reader.read(frame_idx)
        if frame is not None:
            preview_last = _maybe_preview(
                on_preview, frame, preview_last, preview_interval
            )
            state.ingest(frame_idx, detect(frame))
        if on_progress is not None:
            on_progress(done, total)


def _scan_parallel(
    reader: TargetReader,
    detect: DetectFn,
    indices: list[int],
    state: _ClusterState,
    workers: int,
    cancelled: Callable[[], bool],
    on_progress: ProgressFn | None,
    on_preview: PreviewFn | None,
    preview_interval: float,
) -> None:
    """Reads on THIS thread (sequential decode — the cv2 reader isn't thread
    safe), runs detection in a worker pool (the shared buffalo_l ORT session is
    thread-safe for concurrent inference, so N detections overlap and keep the
    GPU busy — same idea as the live/batch worker pools), and clusters each
    result here, serially, as it lands. Bounded in-flight so memory stays flat."""
    total = len(indices)
    cap = max(2, workers * 2)
    preview_last = 0.0
    done = 0
    inflight: deque[tuple[int, Future]] = deque()

    def drain_one() -> None:
        nonlocal done
        frame_idx, fut = inflight.popleft()
        state.ingest(frame_idx, fut.result())
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="sinner2-facemap"
    ) as pool:
        for frame_idx in indices:
            if cancelled():
                break
            frame = reader.read(frame_idx)
            if frame is None:
                done += 1
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
