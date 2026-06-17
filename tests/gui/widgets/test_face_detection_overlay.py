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


class TestScaledMapper:
    def test_scales_coords_when_display_larger(self):
        # Detections in a 100x50 frame, display showing a 400x200 (upscaled)
        # frame → coords scale x4 before mapping.
        seen: list = []
        mapper = lambda fx, fy: seen.append((fx, fy))  # noqa: E731
        m = QFaceDetectionOverlay._scaled_mapper(mapper, (400, 200), (100, 50))  # noqa: SLF001
        assert m is not None
        m(10, 20)
        assert seen[-1] == (40.0, 80.0)

    def test_identity_when_same_size(self):
        seen: list = []
        mapper = lambda fx, fy: seen.append((fx, fy))  # noqa: E731
        m = QFaceDetectionOverlay._scaled_mapper(mapper, (200, 100), (200, 100))  # noqa: SLF001
        m(5, 5)
        assert seen[-1] == (5.0, 5.0)

    def test_skips_on_aspect_mismatch(self):
        # 2:1 display vs 1:1 detection frame → stale/different content → skip.
        m = QFaceDetectionOverlay._scaled_mapper(  # noqa: SLF001
            lambda fx, fy: None, (200, 100), (100, 100)
        )
        assert m is None

    def test_none_without_detection_frame(self):
        m = QFaceDetectionOverlay._scaled_mapper(  # noqa: SLF001
            lambda fx, fy: None, (200, 100), None
        )
        assert m is None


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

    def test_comparison_pairs_paint_without_crashing(self, qtbot):
        d = QFrameDisplayWidget()
        qtbot.addWidget(d)
        d.resize(400, 400)
        d._on_frame_ready(np.zeros((100, 200, 3), dtype=np.uint8), 0)  # noqa: SLF001
        overlay = QFaceDetectionOverlay(parent=d)
        overlay.setGeometry(d.rect())
        overlay.set_comparison(True)
        overlay.set_crop_pairs(
            [
                (
                    (10, 10, 60, 80),
                    np.full((20, 20, 3), 100, np.uint8),
                    np.full((20, 20, 3), 200, np.uint8),
                )
            ],
            200,
            100,
        )
        overlay.show()
        overlay.grab()  # forces paintEvent over the crop pairs; must not raise

    def test_comparison_off_clears_pairs(self, qtbot):
        overlay = QFaceDetectionOverlay()
        qtbot.addWidget(overlay)
        overlay.set_crop_pairs(
            [((0, 0, 4, 4), np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8))],
            200,
            100,
        )
        overlay.set_comparison(False)
        assert overlay._crop_pairs == []  # noqa: SLF001

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


class TestPickMode:
    """Face-mapping pick mode: the overlay receives clicks and hit-tests them
    against the boxes, emitting the clicked face's frame-space bbox."""

    def _overlay(self, qtbot):
        from PySide6.QtCore import QPointF
        from PySide6.QtWidgets import QWidget

        class _StubDisplay(QWidget):
            def map_from_frame(self, fx, fy):
                return QPointF(fx, fy)  # identity mapping

            def current_frame_size(self):
                return (200, 100)

        parent = _StubDisplay()
        qtbot.addWidget(parent)
        overlay = QFaceDetectionOverlay(parent=parent)
        # Keep the parent alive with the returned overlay (the C++ parent owns
        # the child, so a GC'd parent would take the overlay with it).
        overlay._test_parent = parent  # noqa: SLF001
        overlay.set_detections([FaceDetection(bbox=(10, 10, 60, 80))], 200, 100)
        return overlay

    def test_set_pick_enabled_toggles_mouse_transparency(self, qtbot):
        from PySide6.QtCore import Qt

        overlay = self._overlay(qtbot)
        overlay.set_pick_enabled(True)
        assert not overlay.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        overlay.set_pick_enabled(False)
        assert overlay.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

    def test_hit_test_inside_box(self, qtbot):
        from PySide6.QtCore import QPointF

        overlay = self._overlay(qtbot)
        assert overlay._hit_test(QPointF(30, 40)) == (10, 10, 60, 80)  # noqa: SLF001

    def test_hit_test_outside_is_none(self, qtbot):
        from PySide6.QtCore import QPointF

        overlay = self._overlay(qtbot)
        assert overlay._hit_test(QPointF(150, 95)) is None  # noqa: SLF001

    def test_mouse_press_emits_bbox_when_picking(self, qtbot):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        overlay = self._overlay(qtbot)
        overlay.set_pick_enabled(True)
        ev = QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(30, 40),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        with qtbot.waitSignal(overlay.faceClicked) as blocker:
            overlay.mousePressEvent(ev)
        assert blocker.args[0] == (10, 10, 60, 80)

    def test_no_emit_when_pick_disabled(self, qtbot):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        overlay = self._overlay(qtbot)  # pick off by default
        fired = []
        overlay.faceClicked.connect(lambda b: fired.append(b))
        ev = QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(30, 40),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        overlay.mousePressEvent(ev)
        assert fired == []


class TestHighlight:
    """Selecting an identity highlights just its box (the nearest detection)."""

    def test_highlight_index_picks_nearest_box(self, qtbot):
        overlay = QFaceDetectionOverlay()
        qtbot.addWidget(overlay)
        overlay.set_detections(
            [FaceDetection(bbox=(0, 0, 10, 10)),
             FaceDetection(bbox=(100, 100, 110, 110))],
            200, 200,
        )
        assert overlay._highlight_index() == -1  # noqa: SLF001 — no highlight set
        overlay.set_highlight((98, 98, 112, 112))  # near the 2nd box
        assert overlay._highlight_index() == 1  # noqa: SLF001
        overlay.set_highlight(None)
        assert overlay._highlight_index() == -1  # noqa: SLF001

    def test_clear_drops_highlight(self, qtbot):
        overlay = QFaceDetectionOverlay()
        qtbot.addWidget(overlay)
        overlay.set_highlight((0, 0, 10, 10))
        overlay.clear()
        assert overlay._highlight is None  # noqa: SLF001
