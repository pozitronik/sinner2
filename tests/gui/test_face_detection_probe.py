"""Tests for FaceDetectionProbe + FaceDetectionSink: mapping detector output
to FaceDetection, emitting/holding it; never raising out of detection."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.gui.face_detection_probe import FaceDetectionProbe, FaceDetectionSink


class TestFaceDetectionSink:
    def test_empty_returns_none(self):
        assert FaceDetectionSink().latest_detections() is None

    def test_publish_then_latest_converts_faces(self):
        sink = FaceDetectionSink()
        face = SimpleNamespace(
            bbox=np.array([0.0, 0.0, 10.0, 10.0]),
            det_score=0.8,
            sex="M",
            age=25,
        )
        sink.publish([face], 100, 50)
        result = sink.latest_detections()
        assert result is not None
        dets, w, h = result
        assert (w, h) == (100, 50)
        assert dets[0].sex == "M"
        assert dets[0].score == pytest.approx(0.8)

    def test_latest_reflects_most_recent_publish(self):
        sink = FaceDetectionSink()
        sink.publish([SimpleNamespace(bbox=np.array([0.0, 0.0, 1.0, 1.0]))], 10, 10)
        sink.publish([SimpleNamespace(bbox=np.array([0.0, 0.0, 2.0, 2.0]))], 20, 20)
        result = sink.latest_detections()
        assert result is not None
        assert result[1:] == (20, 20)

    def test_clear_drops_latest(self):
        sink = FaceDetectionSink()
        sink.publish([SimpleNamespace(bbox=np.array([0.0, 0.0, 1.0, 1.0]))], 10, 10)
        sink.clear()
        assert sink.latest_detections() is None

    def test_wants_crops_toggle(self):
        sink = FaceDetectionSink()
        assert sink.wants_crops() is False
        sink.set_wants_crops(True)
        assert sink.wants_crops() is True

    def test_publish_and_latest_crops(self):
        sink = FaceDetectionSink()
        assert sink.latest_crops() is None
        pair = ((0, 0, 4, 4), np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8))
        sink.publish_crops([pair], 20, 10)
        crops = sink.latest_crops()
        assert crops is not None
        pairs, w, h = crops
        assert (w, h) == (20, 10) and len(pairs) == 1

    def test_clear_drops_crops(self):
        sink = FaceDetectionSink()
        sink.publish_crops([((0, 0, 1, 1), np.zeros((2, 2, 3), np.uint8),
                             np.zeros((2, 2, 3), np.uint8))], 10, 10)
        sink.clear()
        assert sink.latest_crops() is None


class TestConfigure:
    """A live providers / detection-size change must re-point the probe —
    its cached analyser was built on construction-time providers, and after a
    providers change resets the SHARED face analysis, a stale-list probe
    could rebuild the shared detector on the old EPs (audit rank 25)."""

    def test_provider_change_drops_cached_analyser(self):
        probe = FaceDetectionProbe(providers=["CUDAExecutionProvider"])
        probe._analyser = object()  # noqa: SLF001 — pretend it was built
        probe.configure(["CPUExecutionProvider"], 640)
        assert probe._analyser is None  # noqa: SLF001 — rebuilt on next detect
        assert probe._providers == ["CPUExecutionProvider"]  # noqa: SLF001

    def test_detection_size_change_drops_cached_analyser(self):
        probe = FaceDetectionProbe(
            providers=["CUDAExecutionProvider"], detection_size=640
        )
        probe._analyser = object()  # noqa: SLF001
        probe.configure(["CUDAExecutionProvider"], 728)
        assert probe._analyser is None  # noqa: SLF001
        assert probe._detection_size == 728  # noqa: SLF001

    def test_unchanged_config_keeps_analyser(self):
        probe = FaceDetectionProbe(
            providers=["CUDAExecutionProvider"], detection_size=640
        )
        marker = object()
        probe._analyser = marker  # noqa: SLF001
        probe.configure(["CUDAExecutionProvider"], 640)
        assert probe._analyser is marker  # noqa: SLF001 — no needless rebuild


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
