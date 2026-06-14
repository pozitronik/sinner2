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
from collections.abc import Callable
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


def analyze_target(
    reader: TargetReader,
    detect: DetectFn,
    *,
    stride: int = 15,
    threshold: float = 0.5,
    sections: SectionSet | None = None,
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
    GUI can show progress on the display. Returns the catalog with every
    identity's representative occurrence attached; an early cancel returns the
    partial catalog built so far.
    """
    total = max(0, reader.frame_count)
    stride = max(1, stride)
    if sections is not None and not sections.is_empty():
        # Sample only the SELECTED frames (stridden) — respects the parts trim.
        indices = sections.frame_plan(total)[::stride]
    else:
        indices = list(range(0, total, stride)) or ([0] if total else [])
    face_map = FaceMap.empty().with_threshold(threshold)
    # identity id -> (best_score, frame, bbox) for the thumbnail reference.
    reps: dict[str, tuple[float, int, tuple[float, float, float, float]]] = {}

    scanned = 0
    preview_last = 0.0
    for frame_idx in indices:
        if cancel_event is not None and cancel_event.is_set():
            break
        frame = reader.read(frame_idx)
        scanned += 1
        if frame is not None:
            if on_preview is not None:
                now = time.monotonic()
                if now - preview_last >= preview_interval:
                    preview_last = now
                    on_preview(frame)
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
