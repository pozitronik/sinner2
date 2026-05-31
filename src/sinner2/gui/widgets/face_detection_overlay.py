"""Non-destructive face-detection debug overlay over the frame display.

Draws each detected face's bounding box, keypoints, and a text callout
(sex / age / score / pose) on top of the displayed frame — without touching
the pixels, so nothing is baked into the cache or the output. A diagnostic
aid: turn the swapper off, enable this, and scrub to see exactly what the
detector reports for a given face (e.g. a rotated one — does it find it, are
the keypoints sane, what's the roll/yaw?).

The overlay is a transparent child that covers the whole display. It maps
detection coordinates (frame space) to widget space via the display's
`map_from_frame`, so boxes track the frame through fit-scaling and the
display's rotation. It only draws when the detections match the frame
currently shown (size check), so stale boxes never land on the wrong frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

_BOX_COLOR = QColor(90, 220, 120)     # green box
_KPS_COLOR = QColor(240, 220, 90)     # yellow keypoints
_TEXT_COLOR = QColor(240, 240, 240)
_TEXT_BG = QColor(0, 0, 0, 170)


@dataclass(frozen=True)
class FaceDetection:
    """The drawable subset of an insightface Face, in frame-pixel space."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    kps: list[tuple[float, float]] = field(default_factory=list)
    score: float | None = None
    sex: str | None = None
    age: int | None = None
    pose: tuple[float, float, float] | None = None  # pitch, yaw, roll (deg)


def face_from_insightface(face: Any) -> FaceDetection:
    """Pull the drawable fields off an insightface Face, defensively — any
    field can be absent depending on which models are in the pack."""
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        coords = (0.0, 0.0, 0.0, 0.0)
    else:
        coords = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    kps_raw = getattr(face, "kps", None)
    kps = (
        [(float(p[0]), float(p[1])) for p in kps_raw]
        if kps_raw is not None
        else []
    )
    score = getattr(face, "det_score", None)
    pose_raw = getattr(face, "pose", None)
    pose = (
        (float(pose_raw[0]), float(pose_raw[1]), float(pose_raw[2]))
        if pose_raw is not None and len(pose_raw) >= 3
        else None
    )
    age = getattr(face, "age", None)
    return FaceDetection(
        bbox=coords,
        kps=kps,
        score=float(score) if score is not None else None,
        sex=getattr(face, "sex", None),
        age=int(age) if age is not None else None,
        pose=pose,
    )


def _bgr_to_pixmap(crop: np.ndarray) -> QPixmap:
    """A BGR ndarray crop → QPixmap (copy detaches from the buffer)."""
    crop = np.ascontiguousarray(crop)
    h, w = crop.shape[:2]
    image = QImage(crop.data, w, h, w * 3, QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(image)


class QFaceDetectionOverlay(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Transparent HUD child: click-through, and unpainted areas show the
        # frame (and the metrics overlay) beneath.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._detections: list[FaceDetection] = []
        self._frame_size: tuple[int, int] | None = None
        # Comparison mode: [orig | swapped] thumbnail pairs next to each box.
        self._comparison_on = False
        self._crop_pairs: list[tuple[tuple, QPixmap, QPixmap]] = []
        self._crop_frame_size: tuple[int, int] | None = None
        self.hide()

    def set_detections(
        self, detections: list[FaceDetection], frame_w: int, frame_h: int
    ) -> None:
        self._detections = list(detections)
        self._frame_size = (frame_w, frame_h)
        self.update()

    def set_comparison(self, on: bool) -> None:
        self._comparison_on = on
        if not on:
            self._crop_pairs = []
            self._crop_frame_size = None
        self.update()

    def set_crop_pairs(self, pairs: list, frame_w: int, frame_h: int) -> None:
        """`pairs`: list of (bbox, original_bgr, swapped_bgr). Converts the BGR
        crops to pixmaps once, here, rather than on every paint."""
        self._crop_pairs = [
            (bbox, _bgr_to_pixmap(orig), _bgr_to_pixmap(swap))
            for bbox, orig, swap in pairs
        ]
        self._crop_frame_size = (frame_w, frame_h)
        self.update()

    def clear(self) -> None:
        self._detections = []
        self._frame_size = None
        self._crop_pairs = []
        self._crop_frame_size = None
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        parent = self.parent()
        mapper = getattr(parent, "map_from_frame", None)
        cur_size_fn = getattr(parent, "current_frame_size", None)
        cur = cur_size_fn() if cur_size_fn is not None else None
        if mapper is None or cur is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QFont("Sans", 8))
        # Detections/crops are in the space of the frame the SWAPPER saw, which
        # may be smaller than the displayed frame (a later upscaler stage scales
        # it up). Scale the coords to the displayed frame — but skip if the
        # aspect ratios disagree (a genuinely stale frame), so boxes never land
        # distorted on the wrong content.
        det_mapper = self._scaled_mapper(mapper, cur, self._frame_size)
        if self._detections and det_mapper is not None:
            for det in self._detections:
                self._draw_face(painter, det_mapper, det)
        crop_mapper = self._scaled_mapper(mapper, cur, self._crop_frame_size)
        if self._comparison_on and self._crop_pairs and crop_mapper is not None:
            for bbox, pm_orig, pm_swap in self._crop_pairs:
                self._draw_crop_pair(painter, crop_mapper, bbox, pm_orig, pm_swap)

    @staticmethod
    def _scaled_mapper(mapper, displayed: tuple[int, int], det: tuple | None):
        """A mapper that first scales a point from the detection's frame space
        to the displayed frame, then maps to widget coords. None when there's
        no detection frame or the aspect ratios disagree (stale frame)."""
        if det is None or det[0] <= 0 or det[1] <= 0:
            return None
        sx = displayed[0] / det[0]
        sy = displayed[1] / det[1]
        if max(sx, sy) > 0 and abs(sx - sy) > 0.02 * max(sx, sy):
            return None  # aspect mismatch → different content, don't draw
        return lambda fx, fy: mapper(fx * sx, fy * sy)

    def _draw_face(self, painter: QPainter, mapper, det: FaceDetection) -> None:
        x1, y1, x2, y2 = det.bbox
        p1 = mapper(x1, y1)
        p2 = mapper(x2, y2)
        if p1 is None or p2 is None:
            return
        rect = QRectF(p1, p2).normalized()
        pen = QPen(_BOX_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.setPen(QPen(_KPS_COLOR))
        for kx, ky in det.kps:
            kp = mapper(kx, ky)
            if kp is not None:
                painter.drawEllipse(kp, 2.0, 2.0)
        self._draw_callout(painter, rect, det)

    def _draw_callout(
        self, painter: QPainter, rect: QRectF, det: FaceDetection
    ) -> None:
        lines: list[str] = []
        if det.sex is not None:
            lines.append(f"Sex: {det.sex}")
        if det.age is not None:
            lines.append(f"Age: {det.age}")
        if det.score is not None:
            lines.append(f"Score: {det.score:.2f}")
        if det.pose is not None:
            pitch, yaw, roll = det.pose
            lines.append(f"Roll {roll:+.0f}  Yaw {yaw:+.0f}  Pitch {pitch:+.0f}")
        if not lines:
            return
        fm = painter.fontMetrics()
        line_h = fm.height()
        text_w = max(fm.horizontalAdvance(s) for s in lines) + 8
        text_h = line_h * len(lines) + 4
        # Sit the callout just above the box; flip inside if there's no room.
        bx = rect.left()
        by = rect.top() - text_h - 2
        if by < 0:
            by = rect.top() + 2
        painter.fillRect(QRectF(bx, by, text_w, text_h), _TEXT_BG)
        painter.setPen(QPen(_TEXT_COLOR))
        ty = by + line_h - 2
        for s in lines:
            painter.drawText(QPointF(bx + 4, ty), s)
            ty += line_h

    def _draw_crop_pair(
        self,
        painter: QPainter,
        mapper,
        bbox: tuple,
        pm_orig: QPixmap,
        pm_swap: QPixmap,
    ) -> None:
        x1, y1, x2, y2 = bbox
        anchor = mapper(x2, y1)  # top-right corner of the face box
        if anchor is None:
            return
        label_h = painter.fontMetrics().height()
        x = anchor.x() + 6
        y = anchor.y()
        for label, pm in (("orig", pm_orig), ("swap", pm_swap)):
            painter.fillRect(QRectF(x, y, pm.width(), label_h), _TEXT_BG)
            painter.setPen(QPen(_TEXT_COLOR))
            painter.drawText(QPointF(x + 2, y + label_h - 3), label)
            painter.drawPixmap(int(x), int(y + label_h), pm)
            painter.setPen(QPen(_BOX_COLOR))
            painter.drawRect(QRectF(x, y + label_h, pm.width(), pm.height()))
            x += pm.width() + 3
