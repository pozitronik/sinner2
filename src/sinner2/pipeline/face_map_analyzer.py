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
from collections.abc import Callable
from typing import Any

from sinner2.io.target_reader import TargetReader
from sinner2.pipeline.face_map import FaceMap

# (frames_scanned, frames_to_scan)
ProgressFn = Callable[[int, int], None]
# frame -> faces, each with normed_embedding / det_score / bbox (buffalo_l)
DetectFn = Callable[[Any], list]


def analyze_target(
    reader: TargetReader,
    detect: DetectFn,
    *,
    stride: int = 15,
    threshold: float = 0.5,
    cancel_event: threading.Event | None = None,
    on_progress: ProgressFn | None = None,
) -> FaceMap:
    """Scan ``reader`` on ``stride`` and cluster detected faces into a FaceMap.

    A face with no embedding (a detection-only detector handed in by mistake) is
    skipped — embeddings are what make the clustering identity-stable. Returns
    the catalog with every identity's representative occurrence attached; an
    early cancel returns the partial catalog built so far.
    """
    total = max(0, reader.frame_count)
    stride = max(1, stride)
    indices = list(range(0, total, stride)) or ([0] if total else [])
    face_map = FaceMap.empty().with_threshold(threshold)
    # identity id -> (best_score, frame, bbox) for the thumbnail reference.
    reps: dict[str, tuple[float, int, tuple[float, float, float, float]]] = {}

    scanned = 0
    for frame_idx in indices:
        if cancel_event is not None and cancel_event.is_set():
            break
        frame = reader.read(frame_idx)
        scanned += 1
        if frame is not None:
            for face in detect(frame):
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    continue
                face_map, joined_id = face_map.observe_with_id(emb)
                bbox = getattr(face, "bbox", None)
                if bbox is None:
                    continue
                score = float(getattr(face, "det_score", 0.0) or 0.0)
                prev = reps.get(joined_id)
                if prev is None or score > prev[0]:
                    reps[joined_id] = (
                        score,
                        int(frame_idx),
                        (
                            float(bbox[0]), float(bbox[1]),
                            float(bbox[2]), float(bbox[3]),
                        ),
                    )
        if on_progress is not None:
            on_progress(scanned, len(indices))

    for ident_id, (_score, frame_idx, bbox) in reps.items():
        face_map = face_map.with_reference(ident_id, frame_idx, bbox)
    return face_map
