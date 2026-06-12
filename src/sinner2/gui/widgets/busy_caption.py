"""A subtitle-style busy caption overlaid on the preview.

Long operations (model loads, the first-run buffalo_l download, chain/settings
applies, source/target swaps) run OFF the GUI thread, so the app isn't frozen —
but the preview sits on the last frame with no obvious signal, which reads as a
freeze on slower hardware. This caption sits at the bottom-centre of the frame
display and shows what's happening, clearing itself when the operation ends.

It's a transparent child of QFrameDisplayWidget (same pattern as the face
overlay), stretched to cover it; only the centred pill is painted, so it never
obscures the frame except where the text sits.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget


class QBusyCaption(QWidget):
    """Bottom-centred "something's happening" caption over the preview."""

    _MARGIN = 18  # px from the bottom edge
    _PAD_X = 14
    _PAD_Y = 7

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = ""
        # Click-through: the caption must never eat transport / overlay clicks.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

    def show_message(self, text: str) -> None:
        """Show `text` (empty hides the caption)."""
        text = text.strip()
        if text == self._text:
            return
        self._text = text
        if text:
            self.show()
            self.raise_()
            self.update()
        else:
            self.hide()

    def clear(self) -> None:
        self.show_message("")

    def paintEvent(self, event: QPaintEvent) -> None:
        if not self._text:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(self.font())
        font.setPointSizeF(max(10.0, font.pointSizeF() + 1))
        painter.setFont(font)

        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(self._text)
        th = metrics.height()
        pill_w = tw + self._PAD_X * 2
        pill_h = th + self._PAD_Y * 2
        x = (self.width() - pill_w) / 2
        y = self.height() - pill_h - self._MARGIN
        pill = QRectF(x, y, pill_w, pill_h)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 175))  # translucent dark pill
        painter.drawRoundedRect(pill, pill_h / 2, pill_h / 2)
        painter.setPen(QColor(255, 255, 255, 235))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, self._text)
