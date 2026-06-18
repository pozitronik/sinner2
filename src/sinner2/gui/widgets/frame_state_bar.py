"""Processing-visualiser heatmap bar.

A thin horizontal strip that paints each target frame's pipeline state (from the
executor's FrameStateMap snapshot) as a colour, so the user can SEE how the
frame-skip strategy behaves and how the buffer fills/evicts — and tune from it.

Binning is PROPORTIONAL: when there are more frames than pixels, each pixel
column shows a vertical stack proportional to its frames' state mix (e.g. 60%
green + 20% orange + 20% gray); NOT_REACHED frames are the unpainted remainder,
so a partly-processed column is partly filled. When there are fewer frames than
pixels, each column shows its single covering frame solid. A left-click maps the
column back to a frame index and emits seekRequested.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter
from PySide6.QtWidgets import QWidget

from sinner2.pipeline.realtime.frame_state import FrameState

_NSTATES = len(FrameState)

# Per-state colour. NOT_REACHED is the background (unpainted remainder).
_BACKGROUND = QColor(38, 40, 44)
_PLAYHEAD = QColor(240, 240, 240)
_COLORS: dict[int, QColor] = {
    int(FrameState.SKIPPED): QColor(120, 122, 128),     # gray — strategy drop
    int(FrameState.QUEUED): QColor(60, 120, 220),       # blue — awaiting a worker
    int(FrameState.PROCESSING): QColor(232, 150, 40),   # orange — in flight
    int(FrameState.READY_MEM): QColor(60, 190, 95),     # green — ready (in memory)
    int(FrameState.READY_DISK): QColor(40, 165, 165),   # teal — ready (on disk)
    int(FrameState.INVALID): QColor(212, 72, 72),       # red — stale
}
# Bottom-to-top stack order (most-progressed at the bottom). NOT_REACHED omitted
# so it reads as the empty top remainder.
_STACK_ORDER: tuple[int, ...] = (
    int(FrameState.READY_MEM),
    int(FrameState.READY_DISK),
    int(FrameState.PROCESSING),
    int(FrameState.QUEUED),
    int(FrameState.SKIPPED),
    int(FrameState.INVALID),
)


class QFrameStateBar(QWidget):
    """Per-frame pipeline-state heatmap. Feed it snapshots via set_data(); it
    emits seekRequested(frame) on a left-click."""

    seekRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_count = 0
        self._states = b""
        self._playhead = -1
        self.setMinimumHeight(16)
        self.setToolTip(
            "Processing visualiser — per-frame pipeline state.\n"
            "gray=skipped  blue=queued  orange=processing  "
            "green=ready (memory)  teal=ready (disk)  red=stale.\n"
            "Click to seek."
        )

    def set_data(self, states: bytes, frame_count: int) -> None:
        self._states = states
        self._frame_count = max(0, frame_count)
        self.update()

    def set_playhead(self, frame: int) -> None:
        if frame != self._playhead:
            self._playhead = frame
            self.update()

    def clear(self) -> None:
        self._states = b""
        self._frame_count = 0
        self._playhead = -1
        self.update()

    def _frame_at(self, x: int) -> int:
        """Map a local x pixel to a frame index (inverse of the paint mapping)."""
        w = max(1, self.width())
        n = self._frame_count
        if n <= 0:
            return 0
        return max(0, min(n - 1, x * n // w))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._frame_count > 0:
            self.seekRequested.emit(self._frame_at(int(event.position().x())))
        else:
            super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, _BACKGROUND)
        n = self._frame_count
        w = rect.width()
        h = rect.height()
        if n <= 0 or w <= 0 or not self._states:
            return
        arr = np.frombuffer(self._states, dtype=np.uint8)
        count = min(n, arr.shape[0])
        if count <= 0:
            return
        arr = arr[:count]
        for cx in range(w):
            lo = cx * n // w
            hi = max(lo + 1, (cx + 1) * n // w)
            seg = arr[lo:hi]
            seg = seg[seg < _NSTATES]  # guard against any out-of-range byte
            if seg.size == 0:
                continue
            counts = np.bincount(seg, minlength=_NSTATES)
            total = int(seg.size)
            y = float(h)
            for state in _STACK_ORDER:
                c = int(counts[state])
                if c == 0:
                    continue
                seg_h = c / total * h
                painter.fillRect(
                    QRectF(cx, y - seg_h, 1.0, seg_h), _COLORS[state]
                )
                y -= seg_h
        if 0 <= self._playhead < n:
            px = self._playhead * w // n
            painter.fillRect(QRectF(px, 0.0, 1.0, float(h)), _PLAYHEAD)
