import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPaintEvent, QPixmap, QTransform
from PySide6.QtWidgets import QWidget

from sinner2.types import Frame, FrameIndex


_VALID_ROTATIONS = (0, 90, 180, 270)


class QFrameDisplayWidget(QWidget):
    """Displays BGR ndarray frames, proportionally scaled and centered.

    show_frame() is the public seam used by RealtimeExecutor.on_frame_ready;
    it is safe to call from any thread. Internally it emits a queued signal
    that re-enters the GUI thread to update the cached pixmap and request
    a repaint. The widget never holds a reference to the caller's ndarray —
    QImage.copy() detaches the pixel buffer at hand-off time.

    Display-only rotation (0/90/180/270): applied at paint time via a
    QTransform on the cached pixmap, so it doesn't touch the pixel
    buffer the executor handed us. Save-current-frame consults the
    rotation so what's written to disk matches what's on screen.
    """

    _frameReady = Signal(object, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._rotation: int = 0
        self._frameReady.connect(self._on_frame_ready, type=Qt.ConnectionType.QueuedConnection)
        self.setMinimumSize(160, 90)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def rotation(self) -> int:
        return self._rotation

    def set_rotation(self, degrees: int) -> None:
        # Clamp to the four valid quarter turns; anything else is a
        # caller bug — silently snap to 0 rather than raise so a bad
        # persisted setting doesn't break the GUI.
        if degrees not in _VALID_ROTATIONS:
            degrees = 0
        if degrees == self._rotation:
            return
        self._rotation = degrees
        self.update()

    def cycle_rotation(self) -> int:
        """Advance 0→90→180→270→0. Returns the new value so callers
        can persist it without a separate getter call."""
        idx = (_VALID_ROTATIONS.index(self._rotation) + 1) % len(_VALID_ROTATIONS)
        self.set_rotation(_VALID_ROTATIONS[idx])
        return self._rotation

    def current_pixmap(self) -> QPixmap | None:
        """The rotated pixmap as currently rendered, or None when no
        frame has been shown yet. Used by save-current-frame to get
        exactly what the user sees."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        if self._rotation == 0:
            return self._pixmap
        transform = QTransform().rotate(self._rotation)
        return self._pixmap.transformed(
            transform, Qt.TransformationMode.SmoothTransformation
        )

    def show_frame(self, frame: Frame, index: FrameIndex = 0) -> None:
        """Schedule the frame for display. Safe to call from any thread."""
        self._frameReady.emit(frame, index)

    @Slot(object, int)
    def _on_frame_ready(self, frame: Frame, _index: FrameIndex) -> None:
        h, w, _ = frame.shape
        contiguous = np.ascontiguousarray(frame)
        image = QImage(
            contiguous.data,
            w,
            h,
            w * 3,
            QImage.Format.Format_BGR888,
        ).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pixmap is None:
            return
        # Rotation swaps the apparent width/height for 90/270, so we
        # measure AFTER applying it. transformed() returns a new
        # pixmap; the source remains intact for save-current-frame.
        if self._rotation == 0:
            source = self._pixmap
        else:
            source = self._pixmap.transformed(
                QTransform().rotate(self._rotation),
                Qt.TransformationMode.SmoothTransformation,
            )
        widget_w = self.width()
        widget_h = self.height()
        pix_w = source.width()
        pix_h = source.height()
        scale = min(widget_w / pix_w, widget_h / pix_h)
        new_w = max(1, int(pix_w * scale))
        new_h = max(1, int(pix_h * scale))
        x = (widget_w - new_w) // 2
        y = (widget_h - new_h) // 2
        scaled = source.scaled(
            new_w,
            new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(x, y, scaled)
