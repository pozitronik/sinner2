"""Tests for the face-detection overlay: Face extraction, coordinate
mapping via the display, and crash-free painting."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.gui.widgets.face_detection_overlay import (
    FaceDetection,
    QFaceDetectionOverlay,
    face_from_insightface,
)
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget


class TestFaceExtraction:
    def test_extracts_all_fields(self):
        face = SimpleNamespace(
            bbox=np.array([10.0, 20.0, 110.0, 220.0]),
            kps=np.array([[1.0, 2.0], [3.0, 4.0]]),
            det_score=0.97,
            sex="M",
            age=31,
            pose=np.array([5.0, -10.0, 35.0]),
        )
        d = face_from_insightface(face)
        assert d.bbox == (10.0, 20.0, 110.0, 220.0)
        assert d.kps == [(1.0, 2.0), (3.0, 4.0)]
        assert d.score == pytest.approx(0.97)
        assert d.sex == "M"
        assert d.age == 31
        assert d.pose == (5.0, -10.0, 35.0)

    def test_handles_missing_optional_fields(self):
        face = SimpleNamespace(bbox=np.array([0.0, 0.0, 10.0, 10.0]))
        d = face_from_insightface(face)
        assert d.kps == []
        assert d.score is None
        assert d.sex is None
        assert d.age is None
        assert d.pose is None


class TestCoordinateMapping:
    def test_map_centers_and_scales(self, qtbot):
        d = QFrameDisplayWidget()
        qtbot.addWidget(d)
        d.resize(400, 400)
        # 200x100 frame in a 400x400 widget → scale 2, letterboxed top/bottom.
        d._on_frame_ready(np.zeros((100, 200, 3), dtype=np.uint8), 0)  # noqa: SLF001
        assert d.current_frame_size() == (200, 100)
        tl = d.map_from_frame(0, 0)
        br = d.map_from_frame(200, 100)
        assert (round(tl.x()), round(tl.y())) == (0, 100)
        assert (round(br.x()), round(br.y())) == (400, 300)

    def test_map_returns_none_without_frame(self, qtbot):
        d = QFrameDisplayWidget()
        qtbot.addWidget(d)
        assert d.map_from_frame(0, 0) is None
        assert d.current_frame_size() is None


class TestPainting:
    def test_paints_without_crashing(self, qtbot):
        d = QFrameDisplayWidget()
        qtbot.addWidget(d)
        d.resize(400, 400)
        d._on_frame_ready(np.zeros((100, 200, 3), dtype=np.uint8), 0)  # noqa: SLF001
        overlay = QFaceDetectionOverlay(parent=d)
        overlay.setGeometry(d.rect())
        overlay.set_detections(
            [
                FaceDetection(
                    bbox=(10, 10, 60, 80),
                    kps=[(20, 20), (50, 20)],
                    score=0.9,
                    sex="M",
                    age=30,
                    pose=(1.0, 2.0, 40.0),
                )
            ],
            200,
            100,
        )
        overlay.show()
        overlay.grab()  # forces paintEvent; must not raise

    def test_skips_draw_on_frame_size_mismatch(self, qtbot):
        # Detections computed on a 999x999 frame must not draw over a 200x100
        # one — the size guard prevents stale boxes on the wrong frame.
        d = QFrameDisplayWidget()
        qtbot.addWidget(d)
        d.resize(400, 400)
        d._on_frame_ready(np.zeros((100, 200, 3), dtype=np.uint8), 0)  # noqa: SLF001
        overlay = QFaceDetectionOverlay(parent=d)
        overlay.setGeometry(d.rect())
        overlay.set_detections([FaceDetection(bbox=(0, 0, 10, 10))], 999, 999)
        overlay.show()
        overlay.grab()  # no crash; guard simply skips drawing
