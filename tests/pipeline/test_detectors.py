"""Tests for alternative target detectors (yoloface / scrfd).

The yoloface ONNX parse (letterbox → /255 BGR → split[4,5] → scale-by-ratio →
NMS) was verified numerically against buffalo_l on real frames (box IoU 0.93+,
kps err <7px). Here we test the parsing math with a stub session carrying a
planted detection, plus the SCRFD output→FaceLite mapping and the factory.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sinner2.pipeline.detectors import (
    DetectorModel,
    ScrfdDetector,
    YoloFaceDetector,
    _yolo_detect,
    build_detector,
)


class _PlantedYoloSession:
    """Returns a (1,20,8400) tensor with one planted anchor (col 0): a face at
    box (cx,cy,w,h)=(320,320,200,240), score 0.9, 5 keypoints; everything else
    is zero (score 0 → filtered)."""

    def __init__(self, score=0.9):
        out = np.zeros((1, 20, 8400), np.float32)
        out[0, 0:4, 0] = [320, 320, 200, 240]   # cx,cy,w,h
        out[0, 4, 0] = score                      # score
        # 5 kps as (x,y,conf) triplets at rows 5..19
        kps = [(280, 290), (360, 290), (320, 330), (290, 380), (350, 380)]
        for j, (kx, ky) in enumerate(kps):
            out[0, 5 + j * 3 + 0, 0] = kx
            out[0, 5 + j * 3 + 1, 0] = ky
            out[0, 5 + j * 3 + 2, 0] = 0.9
        self._out = out
        self.planted_kps = np.array([k[:2] for k in kps], np.float32)

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[1, 3, 640, 640])]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, _names, _feeds):
        return [self._out]


def _frame(w=640, h=640):
    return np.zeros((h, w, 3), np.uint8)


class TestYoloParse:
    def test_planted_detection_box_and_kps(self):
        s = _PlantedYoloSession()
        faces = _yolo_detect(s, _frame(640, 640), "input", "output", 0.5, 0.4, 640)
        assert len(faces) == 1
        f = faces[0]
        # 640x640 frame → scale 1, ratio 1: box = cx±w/2, cy±h/2.
        np.testing.assert_allclose(f.bbox, [220, 200, 420, 440], atol=1.0)
        np.testing.assert_allclose(f.kps, s.planted_kps, atol=1.0)
        assert abs(f.det_score - 0.9) < 1e-5

    def test_scales_boxes_to_original_resolution(self):
        # A 1280x1280 frame is fit to 640 (ratio 2): the planted 640-space box
        # maps back ×2.
        s = _PlantedYoloSession()
        faces = _yolo_detect(s, _frame(1280, 1280), "input", "output", 0.5, 0.4, 640)
        assert len(faces) == 1
        np.testing.assert_allclose(faces[0].bbox, [440, 400, 840, 880], atol=2.0)

    def test_below_threshold_is_dropped(self):
        s = _PlantedYoloSession(score=0.3)
        faces = _yolo_detect(s, _frame(640, 640), "input", "output", 0.5, 0.4, 640)
        assert faces == []

    def test_face_lite_has_no_sex_or_pose(self):
        # The swapper's gender filter + pose-rotation use getattr(...,None);
        # FaceLite must simply not carry those (→ graceful degradation).
        s = _PlantedYoloSession()
        f = _yolo_detect(s, _frame(), "input", "output", 0.5, 0.4, 640)[0]
        assert getattr(f, "sex", None) is None
        assert getattr(f, "pose", None) is None


class TestScrfdMapping:
    def test_maps_insightface_output_to_face_lite(self):
        det = ScrfdDetector("scrfd_2.5g.onnx")
        bboxes = np.array([[10, 20, 110, 220, 0.88]], np.float32)
        kpss = np.array([[[30, 40], [80, 40], [55, 70], [35, 90], [75, 90]]], np.float32)
        det._det = SimpleNamespace(  # noqa: SLF001
            detect=lambda _img, max_num=0, metric="default": (bboxes, kpss)
        )
        faces = det.detect(_frame())
        assert len(faces) == 1
        np.testing.assert_allclose(faces[0].bbox, [10, 20, 110, 220])
        np.testing.assert_allclose(faces[0].kps, kpss[0])
        assert abs(faces[0].det_score - 0.88) < 1e-5


class TestFactory:
    def test_build_yoloface(self):
        d = build_detector(DetectorModel.YOLOFACE)
        assert isinstance(d, YoloFaceDetector)

    def test_build_scrfd(self):
        d = build_detector(DetectorModel.SCRFD_2_5G)
        assert isinstance(d, ScrfdDetector)

    def test_buffalo_l_has_no_standalone_detector(self):
        # buffalo_l is the full pack handled by FaceAnalyser itself, not a
        # standalone target detector.
        assert build_detector(DetectorModel.BUFFALO_L) is None
