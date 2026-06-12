import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
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
    # Emitted on the GUI thread each time a frame lands, carrying the frame.
    # The face-detection overlay feeder taps this to run detection.
    frameDisplayed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._rotation: int = 0
        # Cache of the rotated render source so paintEvent (once per displayed
        # frame, 30-60 fps) doesn't re-allocate a full-res SmoothTransformation
        # rotation every paint. Invalidated by key = (source cacheKey, rotation).
        self._rotated_cache: QPixmap | None = None
        self._rotated_cache_key: tuple[int, int] | None = None
        # A full-cover child overlay (face-detection debug) kept stretched to
        # the display's rect; None when no overlay is attached.
        self._face_overlay: QWidget | None = None
        # A second full-cover overlay for the busy caption, kept stretched the
        # same way (it self-positions its pill at the bottom-centre).
        self._busy_caption: QWidget | None = None
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

    def _rotated_source(self) -> QPixmap | None:
        """The display pixmap with the active rotation baked in, cached so the
        full-resolution rotation is computed once per (frame, rotation) rather
        than on every paint. Returns the un-rotated pixmap directly at 0°."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        if self._rotation == 0:
            return self._pixmap
        key = (self._pixmap.cacheKey(), self._rotation)
        if self._rotated_cache is None or self._rotated_cache_key != key:
            self._rotated_cache = self._pixmap.transformed(
                QTransform().rotate(self._rotation),
                Qt.TransformationMode.SmoothTransformation,
            )
            self._rotated_cache_key = key
        return self._rotated_cache

    def current_pixmap(self) -> QPixmap | None:
        """The rotated pixmap as currently rendered, or None when no
        frame has been shown yet. Used by save-current-frame to get
        exactly what the user sees."""
        return self._rotated_source()

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
        self.frameDisplayed.emit(frame)

    # ---- Overlay support (coordinate mapping + full-cover child) ----

    def current_frame_size(self) -> tuple[int, int] | None:
        """Native (w, h) of the frame currently shown, or None if none yet."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        return self._pixmap.width(), self._pixmap.height()

    def map_from_frame(self, fx: float, fy: float) -> QPointF | None:
        """Map a point in the displayed frame's pixel space to widget
        coordinates, accounting for the display rotation and the
        fit-scale-and-center used by paintEvent. None if no frame is shown."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if self._rotation == 0:
            rx, ry, rw, rh = fx, fy, float(pw), float(ph)
        else:
            # Rotate the point exactly as paintEvent rotates the pixmap, then
            # shift into the rotated pixmap's (0,0)-origin coordinate space.
            transform = QTransform().rotate(self._rotation)
            bound = transform.mapRect(QRectF(0, 0, pw, ph))
            p = transform.map(QPointF(fx, fy))
            rx, ry = p.x() - bound.left(), p.y() - bound.top()
            rw, rh = bound.width(), bound.height()
        if rw <= 0 or rh <= 0:
            return None
        scale = min(self.width() / rw, self.height() / rh)
        ox = (self.width() - rw * scale) / 2
        oy = (self.height() - rh * scale) / 2
        return QPointF(ox + rx * scale, oy + ry * scale)

    def set_face_overlay(self, widget: QWidget | None) -> None:
        """Attach a child overlay kept stretched to cover the display."""
        self._face_overlay = widget
        if widget is not None:
            widget.setGeometry(self.rect())

    def set_busy_caption(self, widget: QWidget | None) -> None:
        """Attach the busy-caption overlay, kept stretched to cover the
        display (it paints only its centred pill)."""
        self._busy_caption = widget
        if widget is not None:
            widget.setGeometry(self.rect())

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self._face_overlay is not None:
            self._face_overlay.setGeometry(self.rect())
        if self._busy_caption is not None:
            self._busy_caption.setGeometry(self.rect())

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        # Rotation swaps the apparent width/height for 90/270, so we measure
        # AFTER applying it. _rotated_source() caches the rotated pixmap; the
        # source stays intact for save-current-frame.
        source = self._rotated_source()
        if source is None:
            return
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
