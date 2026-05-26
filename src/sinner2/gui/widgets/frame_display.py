import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget

from sinner2.types import Frame, FrameIndex


class QFrameDisplayWidget(QWidget):
    """Displays BGR ndarray frames, proportionally scaled and centered.

    show_frame() is the public seam used by RealtimeExecutor.on_frame_ready;
    it is safe to call from any thread. Internally it emits a queued signal
    that re-enters the GUI thread to update the cached pixmap and request
    a repaint. The widget never holds a reference to the caller's ndarray —
    QImage.copy() detaches the pixel buffer at hand-off time.
    """

    _frameReady = Signal(object, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._frameReady.connect(self._on_frame_ready, type=Qt.ConnectionType.QueuedConnection)
        self.setMinimumSize(160, 90)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

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
        widget_w = self.width()
        widget_h = self.height()
        pix_w = self._pixmap.width()
        pix_h = self._pixmap.height()
        scale = min(widget_w / pix_w, widget_h / pix_h)
        new_w = max(1, int(pix_w * scale))
        new_h = max(1, int(pix_h * scale))
        x = (widget_w - new_w) // 2
        y = (widget_h - new_h) // 2
        scaled = self._pixmap.scaled(
            new_w,
            new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(x, y, scaled)
