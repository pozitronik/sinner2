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

    def test_degenerate_frame_returns_empty_not_div_by_zero(self):
        # A 0-height (or 0-width) frame must not crash on `size / h` — it has no
        # faces, so return [] instead of raising ZeroDivisionError.
        s = _PlantedYoloSession()
        assert _yolo_detect(s, _frame(640, 0), "input", "output", 0.5, 0.4, 640) == []
        assert _yolo_detect(s, _frame(0, 640), "input", "output", 0.5, 0.4, 640) == []

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


class TestScrfdSetup:
    def test_passes_str_path_not_pathlib(self, monkeypatch):
        # Regression: insightface.get_model does name.endswith('.onnx'), so the
        # model path must be a str — get_model_path returns a Path, which raised
        # AttributeError('WindowsPath' object has no attribute 'endswith').
        import sys
        import types
        from pathlib import Path

        from sinner2.pipeline import detectors

        captured = {}

        def fake_get_model(name, **_kw):
            captured["name"] = name
            return SimpleNamespace(prepare=lambda **_k: None)

        mz = types.ModuleType("insightface.model_zoo")
        mz.get_model = fake_get_model  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "insightface.model_zoo", mz)
        monkeypatch.setattr(
            detectors, "get_model_path", lambda f: Path("/models") / f
        )
        ScrfdDetector("scrfd_2.5g.onnx").setup()
        assert isinstance(captured["name"], str)
        assert captured["name"].endswith(".onnx")

    def test_setup_records_memory_footprint(self, monkeypatch):
        # The Models-tab Memory column is fed by measure_model_load; the SCRFD
        # load must register a footprint under its filename like the ONNX
        # detectors do — otherwise its cell stays blank forever.
        import sys
        import types
        from pathlib import Path

        from sinner2.pipeline import detectors
        from sinner2.pipeline import memory_probe as mp

        mp.reset_footprints()
        mp._measuring = False  # noqa: SLF001

        mz = types.ModuleType("insightface.model_zoo")
        mz.get_model = lambda name, **_kw: SimpleNamespace(  # type: ignore[attr-defined]
            prepare=lambda **_k: None
        )
        monkeypatch.setitem(sys.modules, "insightface.model_zoo", mz)
        monkeypatch.setattr(
            detectors, "get_model_path", lambda f: Path("/models") / f
        )

        ScrfdDetector("scrfd_2.5g.onnx").setup()

        assert "scrfd_2.5g.onnx" in mp.model_footprints()


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

    def test_build_forwards_size_to_detectors(self):
        # det_size must reach the standalone detectors. It was silently dropped:
        # build_detector took no size, so the knob did nothing for yolo/scrfd.
        assert build_detector(DetectorModel.YOLOFACE, size=320)._size == 320  # noqa: SLF001
        assert build_detector(DetectorModel.SCRFD_2_5G, size=512)._size == 512  # noqa: SLF001

    def test_strips_tensorrt_from_detector_by_default(self, monkeypatch):
        # The factory routes providers through detector_providers(): a TRT entry
        # is dropped (detector runs on CUDA) unless the toggle is on.
        from sinner2.pipeline import model_cache

        monkeypatch.setattr(model_cache, "_tensorrt_detector", False)
        d = build_detector(
            DetectorModel.SCRFD_2_5G,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider",
                       "CPUExecutionProvider"],
        )
        assert d._providers == [  # noqa: SLF001
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]

    def test_keeps_tensorrt_when_detector_trt_enabled(self, monkeypatch):
        from sinner2.pipeline import model_cache

        monkeypatch.setattr(model_cache, "_tensorrt_detector", True)
        d = build_detector(
            DetectorModel.YOLOFACE,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        )
        assert "TensorrtExecutionProvider" in d._providers  # noqa: SLF001


class TestYoloStaticInput:
    def test_setup_pins_size_to_static_model_input(self, monkeypatch):
        # yoloface_8n exports a FIXED [1,3,640,640] input — a requested det_size
        # must be pinned to the model's size, else inference shape-mismatches.
        from sinner2.pipeline import detectors

        monkeypatch.setattr(
            detectors,
            "get_onnx_session_io",
            lambda *a, **k: (_PlantedYoloSession(), "input", "output"),
        )
        d = YoloFaceDetector(size=320)
        d.setup()
        assert d._size == 640  # noqa: SLF001


class TestDetectorRelease:
    """The scan builds a detector per run; release() must free its ONNX session
    or the CUDA session leaks across repeated scans."""

    def test_yoloface_release_drops_session_refcount(self, monkeypatch):
        from sinner2.pipeline import detectors as det_mod
        from sinner2.pipeline.detectors import YoloFaceDetector

        released = []
        monkeypatch.setattr(
            det_mod, "release_onnx_session",
            lambda name, providers=None: released.append((name, providers)),
        )
        d = YoloFaceDetector(
            model_file="yoloface_8n.onnx", providers=["CPUExecutionProvider"]
        )
        d._session = object()  # noqa: SLF001 — pretend setup() ran
        d.release()
        assert released == [("yoloface_8n.onnx", ["CPUExecutionProvider"])]
        assert d._session is None  # noqa: SLF001
        d.release()  # idempotent — no double-release
        assert len(released) == 1

    def test_scrfd_release_drops_model(self):
        from sinner2.pipeline.detectors import ScrfdDetector

        d = ScrfdDetector()
        d._det = object()  # noqa: SLF001
        d.release()
        assert d._det is None  # noqa: SLF001
