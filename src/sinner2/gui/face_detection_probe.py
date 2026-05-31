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
        self._latest: tuple[list, int, int] | None = None  # faces, width, height

    def publish(self, faces: list, width: int, height: int) -> None:
        with self._lock:
            self._latest = (list(faces), width, height)

    def latest_detections(self) -> tuple[list[FaceDetection], int, int] | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        faces, w, h = latest
        return [face_from_insightface(f) for f in faces], w, h

    def clear(self) -> None:
        with self._lock:
            self._latest = None


class FaceDetectionProbe(QObject):
    """Detect faces on submitted frames and emit drawable detections."""

    detectionsReady = Signal(object, int, int)  # list[FaceDetection], w, h

    def __init__(
        self,
        detect_fn: DetectFn | None = None,
        providers: list[str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._detect_fn = detect_fn
        self._providers = list(providers) if providers else None
        self._analyser: Any = None

    @Slot(object, int, int)
    def analyze(self, frame: Frame, width: int, height: int) -> None:
        """Detect on `frame` (runs on the probe's thread) and emit the result.
        Swallows detection errors — a debug overlay must never crash the app."""
        try:
            faces = self._detect(frame)
        except Exception:
            return
        detections: list[FaceDetection] = [
            face_from_insightface(f) for f in faces
        ]
        self.detectionsReady.emit(detections, width, height)

    def _detect(self, frame: Frame) -> list:
        if self._detect_fn is not None:
            return self._detect_fn(frame)
        if self._analyser is None:
            from sinner2.pipeline.face_analyser import FaceAnalyser

            self._analyser = FaceAnalyser(providers=self._providers)
        return self._analyser.analyse_uncached(frame)
