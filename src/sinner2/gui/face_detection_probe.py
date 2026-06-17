"""Background face-detection probe for the debug overlay.

Runs detection off the GUI thread (lives on its own QThread) so enabling the
overlay never stalls the live preview. The caller throttles submissions; the
probe just detects whatever frame it's handed and reports the drawable result.

Detection reuses the process-wide insightface model (same singleton the
swapper uses), so enabling the overlay doesn't load a second copy. The detect
function is injectable for tests so they need no real models.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from sinner2.gui.widgets.face_detection_overlay import (
    FaceDetection,
    face_from_insightface,
)
from sinner2.types import Frame

DetectFn = Callable[[Frame], list]


class FaceDetectionSink:
    """Thread-safe holder for the face swapper's most recent PRE-swap
    detections.

    The swapper publishes here (from worker threads) right after it detects,
    before it swaps; the GUI polls `latest_detections()` so the overlay can
    show the exact boxes/keypoints that drove the swap — the real diagnostic,
    versus re-detecting the already-swapped output. Raw insightface faces are
    stored as-is and converted on read (cheap, and only ~6 Hz when polled),
    so the swapper never imports the GUI types.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # faces, width, height, frame_index (the frame the faces were detected
        # on; None when the producer didn't tag it). The index lets a consumer
        # reject a click against a STALE snapshot — the boxes on screen are from
        # frame N, but the sink may have advanced before the click landed.
        self._latest: tuple[list, int, int, int | None] | None = None
        # Original/swapped crop pairs for the comparison overlay, and whether
        # the swapper should bother extracting them (false → zero cost).
        self._wants_crops = False
        self._latest_crops: tuple[list, int, int] | None = None

    def publish(
        self, faces: list, width: int, height: int, frame_index: int | None = None
    ) -> None:
        with self._lock:
            self._latest = (list(faces), width, height, frame_index)

    def latest_detections(self) -> tuple[list[FaceDetection], int, int] | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        faces, w, h, _idx = latest
        return [face_from_insightface(f) for f in faces], w, h

    def latest_raw(self) -> tuple[list, int, int, int | None] | None:
        """The RAW insightface faces (with embeddings) + dims + the frame index
        they were detected on — for the highlight / click-to-pick path (which
        needs the embedding and the index to check freshness). Public so
        consumers stop reaching into ``_latest``."""
        with self._lock:
            return self._latest

    # ---- Comparison crops ----

    def set_wants_crops(self, on: bool) -> None:
        with self._lock:
            self._wants_crops = bool(on)

    def wants_crops(self) -> bool:
        with self._lock:
            return self._wants_crops

    def publish_crops(self, pairs: list, width: int, height: int) -> None:
        """`pairs`: list of (bbox, original_bgr, swapped_bgr) for swapped faces."""
        with self._lock:
            self._latest_crops = (list(pairs), width, height)

    def latest_crops(self) -> tuple[list, int, int] | None:
        with self._lock:
            return self._latest_crops

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._latest_crops = None


class FaceDetectionProbe(QObject):
    """Detect faces on submitted frames and emit drawable detections."""

    detectionsReady = Signal(object, int, int)  # list[FaceDetection], w, h

    def __init__(
        self,
        detect_fn: DetectFn | None = None,
        providers: list[str] | None = None,
        parent: QObject | None = None,
        detection_size: int = 640,
        sink: FaceDetectionSink | None = None,
    ) -> None:
        super().__init__(parent)
        self._detect_fn = detect_fn
        self._providers = list(providers) if providers else None
        self._detection_size = detection_size
        # The same sink the swapper publishes to. The probe runs ONLY when the
        # swapper is off (the two never feed it at once), so publishing the raw
        # faces here keeps the selection-highlight + face-pick working in the
        # swapper-off mapping workflow — otherwise the sink would be empty and
        # clicking a face couldn't capture it.
        self._sink = sink
        self._analyser: Any = None
        # configure() is called from the GUI thread while _detect runs on the
        # probe's own thread — guard the providers/size/analyser trio.
        self._config_lock = threading.Lock()

    def configure(
        self, providers: list[str] | None, detection_size: int
    ) -> None:
        """Re-point detection at a new EP list / detection size (a live
        settings change). Drops the cached analyser so the next probe rebuilds
        on the new config — otherwise a providers change that resets the
        SHARED face analysis could see this probe rebuild it on the STALE
        construction-time providers. No-op when nothing changed (the analyser
        is expensive to rebuild)."""
        new = list(providers) if providers else None
        with self._config_lock:
            if new == self._providers and detection_size == self._detection_size:
                return
            self._providers = new
            self._detection_size = detection_size
            self._analyser = None

    @Slot(object, int, int)
    def analyze(self, frame: Frame, width: int, height: int) -> None:
        """Detect on `frame` (runs on the probe's thread) and emit the result.
        Swallows detection errors — a debug overlay must never crash the app."""
        try:
            faces = self._detect(frame)
        except Exception:
            return
        # Publish the RAW faces (with embeddings) for the highlight/pick path
        # BEFORE converting to the drawable subset, which drops the embedding.
        if self._sink is not None:
            self._sink.publish(faces, width, height)
        detections: list[FaceDetection] = [
            face_from_insightface(f) for f in faces
        ]
        self.detectionsReady.emit(detections, width, height)

    def _detect(self, frame: Frame) -> list:
        if self._detect_fn is not None:
            return self._detect_fn(frame)
        with self._config_lock:
            analyser = self._analyser
            providers = list(self._providers) if self._providers else None
            size = self._detection_size
        if analyser is None:
            from sinner2.pipeline.face_analyser import FaceAnalyser

            analyser = FaceAnalyser(providers=providers, detection_size=size)
            with self._config_lock:
                # Cache only if the config didn't change mid-build; a racing
                # configure() wins and the next detect rebuilds on its config.
                if (
                    (list(self._providers) if self._providers else None)
                    == providers
                    and self._detection_size == size
                ):
                    self._analyser = analyser
        return analyser.analyse_uncached(frame)
