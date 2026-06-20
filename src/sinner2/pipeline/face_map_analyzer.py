"""Build a target's identity catalog by a strided scan — and, for face-mapping's
detection-free runtime, a full-frame per-frame geometry table.

``analyze_target`` samples the target on a configurable stride, detects faces
(buffalo_l → ArcFace embeddings), and online-clusters them into a FaceMap. Each
identity's clearest occurrence (highest det_score) is recorded as
``ref_frame``/``ref_bbox`` so the UI can extract a representative thumbnail
later. ``precompute_geometry`` then does a full-frame (stride-1) pass, matching
each detected face to that catalog and recording its bbox + keypoints per frame
so the runtime can swap without detecting. Both are headless + cancellable; the
GUI runs them on a worker thread with a progress callback.

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
from typing import Any, Protocol

from sinner2.io.target_reader import TargetReader
from sinner2.pipeline.face_map import FaceMap
from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace
from sinner2.pipeline.sections import SectionSet
from sinner2.types import Frame

_KPS = 5  # insightface 5-point keypoints; geometry needs exactly these

# Bake geometry against a PERMISSIVE floor (not the catalog's current threshold):
# any face within this cosine of an identity is recorded with its real embedding.
# The runtime re-routes each baked face by embedding against the LIVE catalog, so
# LOWERING the threshold later recovers these borderline faces in detection-free
# mode too — without the floor, geometry (baked at the old threshold) and live
# detection would disagree after a threshold drop. Below this, matches are noise
# (ArcFace impostor pairs rarely exceed it), so it doesn't bloat the table.
_GEOMETRY_BAKE_FLOOR = 0.3

# (frames_scanned, frames_to_scan)
ProgressFn = Callable[[int, int], None]
# frame -> faces, each with normed_embedding / det_score / bbox (buffalo_l)
DetectFn = Callable[[Any], list]
PreviewFn = Callable[[Frame], None]
PositionFn = Callable[[int], None]  # the frame index currently being scanned

# (best_score, frame, bbox, sex, age, pitch, yaw, roll) of an identity's clearest
# occurrence. pitch/yaw/roll (insightface face.pose, degrees) are None unless the
# full buffalo_l pack ran (fast det+rec mode yields no pose).
_Rep = tuple[
    float, int, tuple[float, float, float, float],
    "str | None", "int | None", "float | None", "float | None", "float | None",
]


def _pose(face: Any) -> tuple["float | None", "float | None", "float | None"]:
    """(pitch, yaw, roll) from insightface's face.pose, or all-None when absent."""
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) < 3:
        return None, None, None
    return float(pose[0]), float(pose[1]), float(pose[2])


class _ScanSink(Protocol):
    """What the scanners feed each detected-faces batch to — the catalog
    clusterer or the geometry collector both satisfy it structurally."""

    def ingest(self, frame_idx: int, faces: list) -> None: ...


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
        # identity id -> its box on that earliest frame, so navigating to a face
        # can draw it without a live re-detect.
        self.first_bboxes: dict[str, tuple[float, float, float, float]] = {}
        # Seed from an existing catalog so a RESUMED scan keeps its identities,
        # occurrence counts, and references instead of starting blank. The seed
        # carries the stored det_score so the best-ever occurrence (its thumbnail,
        # score, and pose) survives the resume — a re-found occurrence replaces it
        # only when it scores higher.
        for ident in face_map.identities:
            if ident.ref_frame is not None and ident.ref_bbox is not None:
                self.reps[ident.id] = (
                    float(ident.det_score or 0.0),
                    ident.ref_frame, ident.ref_bbox, ident.sex, ident.age,
                    ident.pitch, ident.yaw, ident.roll,
                )
            if ident.first_frame is not None:
                self.firsts[ident.id] = ident.first_frame
            if ident.first_bbox is not None:
                self.first_bboxes[ident.id] = ident.first_bbox

    def ingest(self, frame_idx: int, faces: list) -> None:
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            self.face_map, joined_id = self.face_map.observe_with_id(emb)
            is_first = joined_id not in self.firsts
            self.firsts.setdefault(joined_id, int(frame_idx))
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            box4 = (
                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            )
            if is_first:
                # The box on this person's EARLIEST frame (same frame as
                # first_frame), so navigating to them can draw it.
                self.first_bboxes[joined_id] = box4
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            prev = self.reps.get(joined_id)
            if prev is None or score > prev[0]:
                age_raw = getattr(face, "age", None)
                pitch, yaw, roll = _pose(face)
                self.reps[joined_id] = (
                    score,
                    int(frame_idx),
                    box4,
                    getattr(face, "sex", None),
                    int(age_raw) if age_raw is not None else None,
                    pitch, yaw, roll,
                )

    def finish(self) -> FaceMap:
        face_map = self.face_map
        for ident_id, rep in self.reps.items():
            score, frame_idx, bbox, sex, age, pitch, yaw, roll = rep
            face_map = face_map.with_reference(
                ident_id, frame_idx, bbox, sex=sex, age=age,
                first_frame=self.firsts.get(ident_id),
                first_bbox=self.first_bboxes.get(ident_id),
                det_score=score, pitch=pitch, yaw=yaw, roll=roll,
            )
        return face_map


class _GeometryCollector:
    """Scan sink (duck-types ``_ClusterState.ingest`` so the same scanners feed
    it) that, instead of clustering, MATCHES each detected face to an existing
    catalog and records its geometry (bbox + 5 kps) tagged with the matched
    identity. Faces with no embedding, no match, or the wrong keypoint count are
    dropped — the runtime simply skips anything not in this table."""

    def __init__(self, catalog: FaceMap) -> None:
        # Match permissively (the lower of the catalog threshold and the floor) so
        # borderline faces are baked WITH their embedding; the runtime then routes
        # by embedding at the live threshold. See _GEOMETRY_BAKE_FLOOR.
        self._catalog = catalog.with_threshold(
            min(catalog.threshold, _GEOMETRY_BAKE_FLOOR)
        )
        self.faces: dict[int, list[GeomFace]] = {}

    def ingest(self, frame_idx: int, faces: list) -> None:
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            match = self._catalog.best_match(emb)
            if match is None:
                continue  # unmapped → no swap at runtime, so don't record it
            bbox = getattr(face, "bbox", None)
            kps = getattr(face, "kps", None)
            if bbox is None or kps is None or len(kps) != _KPS:
                continue
            roll = getattr(face, "baked_roll", None)
            self.faces.setdefault(int(frame_idx), []).append(
                GeomFace(
                    match.id,
                    (
                        float(bbox[0]), float(bbox[1]),
                        float(bbox[2]), float(bbox[3]),
                    ),
                    tuple((float(p[0]), float(p[1])) for p in kps),
                    # Bake the REAL embedding so the runtime routes against the
                    # live catalog (merges/reassignments need no re-precompute).
                    tuple(float(x) for x in emb),
                    # Baked in-plane roll for detection-free rotation comp (or None
                    # when the detect closure didn't measure it).
                    float(roll) if roll is not None else None,
                )
            )

    def finish(
        self, frame_count: int, refined: bool,
        bake_size: tuple[int, int] | None,
    ) -> FrameGeometry:
        return FrameGeometry(
            {k: tuple(v) for k, v in self.faces.items()},
            frame_count, refined, bake_size,
        )


def precompute_geometry(
    reader: TargetReader,
    detect: DetectFn,
    catalog: FaceMap,
    *,
    sections: SectionSet | None = None,
    workers: int = 1,
    refined: bool = False,
    cancel_event: threading.Event | None = None,
    on_progress: ProgressFn | None = None,
    on_position: PositionFn | None = None,
) -> tuple[FrameGeometry, int, int]:
    """Full-frame pass (stride 1) recording per-frame geometry for every face
    within the permissive bake floor of ``catalog`` (see _GEOMETRY_BAKE_FLOOR, so
    lowering the threshold later still routes them) — the artifact that lets the
    runtime skip detection.

    With ``sections`` non-empty only the selected frames are covered (others get
    no geometry, hence no swap there). Reads stay single-threaded; ``workers`` >
    1 detects in parallel (same pool pattern as the catalog scan). ``refined``
    is metadata stamped onto the result: pass True when ``detect`` already
    2dfan4-refines the keypoints it returns (landmark-refine baking), so the
    runtime knows it can use them as-is. Returns ``(geometry, scanned, total)``;
    an early cancel returns the partial table.
    """
    total_frames = max(0, reader.frame_count)
    if sections is not None and not sections.is_empty():
        indices = sections.frame_plan(total_frames)
    else:
        indices = list(range(total_frames))
    total = len(indices)
    collector = _GeometryCollector(catalog)
    counter = _Counter(0)

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if workers <= 1:
        _scan_serial(
            reader, detect, indices, collector, counter, total, cancelled,
            on_progress, None, on_position, 0.0,
        )
    else:
        _scan_parallel(
            reader, detect, indices, collector, workers, counter, total,
            cancelled, on_progress, None, on_position, 0.0,
        )
    # Bake resolution = what the reader produces (the scan reads at native,
    # processing_scale=1.0); the runtime rescales geometry to its own frame size.
    bake_size = (int(reader.width), int(reader.height))
    return collector.finish(total_frames, refined, bake_size), counter.done, total


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
    state: _ScanSink,
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
    state: _ScanSink,
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
