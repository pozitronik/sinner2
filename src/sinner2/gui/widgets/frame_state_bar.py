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
from PySide6.QtGui import (
    QColor,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import QWidget

from sinner2.pipeline.realtime.frame_state import FaceMark, FrameState

_NSTATES = len(FrameState)
# Problem-frame marker (detection ran, found no face): a bright magenta tick
# along the TOP of the column, over the state stack, so it can't be missed.
_NO_FACE = QColor(235, 64, 170)
_NO_FACE_MARK = int(FaceMark.ABSENT)

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
        self._faces = b""
        self._playhead = -1
        # Cached render of the state stack + markers (everything EXCEPT the
        # playhead). Rebuilt only when the data / size change — NOT on the
        # per-frame playhead moves, which were re-running the whole per-column
        # bin on the GUI thread (py-spy: the heaviest GUI-thread cost).
        self._cache: QPixmap | None = None
        self.setMinimumHeight(16)
        self.setToolTip(
            "Processing visualiser — per-frame pipeline state.\n"
            "gray=skipped  blue=queued  orange=processing  "
            "green=ready (memory)  teal=ready (disk)  red=stale.\n"
            "magenta tick (top) = no face detected (P / Shift+P to jump).\n"
            "Click to seek."
        )

    def set_data(
        self, states: bytes, frame_count: int, faces: bytes = b""
    ) -> None:
        fc = max(0, frame_count)
        if (
            states == self._states
            and faces == self._faces
            and fc == self._frame_count
        ):
            return  # unchanged snapshot — the 20 Hz visualiser polls at a fixed
            # rate, so a slow chain leaves the bytes identical across many ticks;
            # skip re-binning every column (the heaviest GUI-thread cost).
        self._states = states
        self._faces = faces
        self._frame_count = fc
        self._cache = None  # data changed → rebuild the stack on the next paint
        self.update()

    def set_playhead(self, frame: int) -> None:
        if frame != self._playhead:
            self._playhead = frame
            self.update()

    def clear(self) -> None:
        self._states = b""
        self._faces = b""
        self._frame_count = 0
        self._playhead = -1
        self._cache = None
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._cache = None  # re-render the stack at the new width
        super().resizeEvent(event)

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
        w = self.rect().width()
        h = self.rect().height()
        if self._cache is None:
            self._cache = self._render_stack(w, h)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        # The playhead is the only per-frame-changing element — draw it fresh
        # over the cached stack (cheap) so a playhead move doesn't re-bin.
        n = self._frame_count
        if n > 0 and w > 0 and 0 <= self._playhead < n:
            px = self._playhead * w // n
            painter.fillRect(QRectF(px, 0.0, 1.0, float(h)), _PLAYHEAD)

    def _render_stack(self, w: int, h: int) -> QPixmap:
        """Render the per-frame state stack + no-face markers into a pixmap (the
        playhead is added per-paint, not here). Cached by paintEvent and rebuilt
        only when the data / size change."""
        pm = QPixmap(max(1, w), max(1, h))
        pm.fill(_BACKGROUND)
        n = self._frame_count
        if n <= 0 or w <= 0 or not self._states:
            return pm
        arr = np.frombuffer(self._states, dtype=np.uint8)
        count = min(n, arr.shape[0])
        if count <= 0:
            return pm
        arr = arr[:count]
        painter = QPainter(pm)
        try:
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
            # Problem-frame markers: a magenta tick at the TOP of any column that
            # covers a no-face frame, painted over the state stack.
            if self._faces:
                farr = np.frombuffer(self._faces, dtype=np.uint8)
                fcount = min(n, farr.shape[0])
                if fcount > 0:
                    farr = farr[:fcount]
                    mark_h = min(3.0, float(h))
                    for cx in range(w):
                        lo = cx * n // w
                        hi = max(lo + 1, (cx + 1) * n // w)
                        seg = farr[lo:hi]
                        if seg.size and bool(np.any(seg == _NO_FACE_MARK)):
                            painter.fillRect(
                                QRectF(cx, 0.0, 1.0, mark_h), _NO_FACE
                            )
        finally:
            painter.end()
        return pm
