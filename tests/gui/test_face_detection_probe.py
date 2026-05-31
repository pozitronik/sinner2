"""Tests for FaceDetectionProbe: maps detector output to FaceDetection and
emits it; never raises out of detection."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sinner2.gui.face_detection_probe import FaceDetectionProbe


def test_emits_detections_from_detect_fn(qtbot):
    face = SimpleNamespace(
        bbox=np.array([0.0, 0.0, 10.0, 10.0]),
        det_score=0.5,
        sex="F",
        age=20,
    )
    probe = FaceDetectionProbe(detect_fn=lambda frame: [face])
    received: list = []
    probe.detectionsReady.connect(lambda dets, w, h: received.append((dets, w, h)))

    probe.analyze(np.zeros((10, 10, 3), dtype=np.uint8), 10, 10)

    assert len(received) == 1
    dets, w, h = received[0]
    assert (w, h) == (10, 10)
    assert dets[0].sex == "F"
    assert dets[0].age == 20


def test_swallows_detect_errors(qtbot):
    def boom(_frame):
        raise RuntimeError("detector exploded")

    probe = FaceDetectionProbe(detect_fn=boom)
    received: list = []
    probe.detectionsReady.connect(lambda *a: received.append(a))

    probe.analyze(np.zeros((4, 4, 3), dtype=np.uint8), 4, 4)

    assert received == []  # error swallowed, nothing emitted
