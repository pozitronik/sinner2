"""Floating real-time metrics overlay over the frame display.

A diagnostic surface for the user. Shows rolling-window throughput
for the four pipeline stages (read → process → write, plus drops),
plus a compact numeric readout for cache hit ratio, memory used, and
queue depths. Designed for tuning: when the user changes worker count
or strategy, the graph responds visibly within a second or two.

Render path: custom QPainter, no extra deps. ~10 Hz QTimer ticks the
sampler, appends a fresh MetricsSample to the rolling deque, requests
a repaint. Position is managed by the parent (main_window) — the
widget itself just draws inside its assigned geometry.

Toggled with a keyboard shortcut from main_window; persisted via
Settings.metrics_overlay_visible. Hidden by default.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

_WINDOW_SECONDS = 30.0
_UPDATE_HZ = 10
_SAMPLE_CAP = int(_WINDOW_SECONDS * _UPDATE_HZ * 2)  # generous: 2x for safety
_PANEL_W = 320
_PANEL_H = 196  # +16 over original to fit the per-processor timing row
_PANEL_MARGIN = 8

_BG_COLOR = QColor(0, 0, 0, 180)  # near-opaque black; legible over any frame
_TEXT_COLOR = QColor(220, 220, 220)
_GRID_COLOR = QColor(80, 80, 80, 120)
_READ_COLOR = QColor(90, 160, 240)    # blue
_PROC_COLOR = QColor(120, 220, 120)   # green
_WRITE_COLOR = QColor(240, 220, 90)   # yellow
_DROP_COLOR = QColor(240, 90, 90)     # red


@dataclass(frozen=True)
class MetricsSample:
    """One snapshot at a point in time. Floats are per-second rates;
    counters are absolute. Used both for the graph (per-tick) and the
    numeric readout (latest sample wins)."""

    timestamp: float
    read_fps: float
    process_fps: float
    write_fps: float
    drop_fps: float
    cache_hit_ratio: float
    memory_used_mb: float
    work_outstanding: int
    work_capacity: int
    write_outstanding: int
    write_capacity: int
    total_drops: int
    last_completed: int  # -1 = no frame produced yet (loading state)
    # Per-processor average ms over the last few seconds. Empty when no
    # processing has happened in the window (paused / idle / pre-first-
    # frame). Order is whatever the executor returned — typically chain
    # order — and the overlay renders verbatim.
    processor_timings: dict[str, float] = field(default_factory=dict)


# Caller supplies a snapshot function that produces a sample on demand.
# `None` is a valid return when no session exists — the overlay then
# shows the "no session" placeholder text.
SnapshotFn = Callable[[], MetricsSample | None]


class QMetricsOverlay(QWidget):
    def __init__(self, snapshot_fn: SnapshotFn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Don't block mouse events on the frame display behind us —
        # the user must still be able to click through to anything
        # interactive (currently nothing, but future-proof).
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._snapshot_fn = snapshot_fn
        self._samples: deque[MetricsSample] = deque(maxlen=_SAMPLE_CAP)
        # Track previous cumulative counters so we can compute write_fps
        # / drop_fps as deltas per tick. None on the first tick.
        self._prev_write_completed: int | None = None
        self._prev_total_drops: int | None = None
        self._prev_timestamp: float | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(int(1000 / _UPDATE_HZ))
        # Only run the timer when visible — saves CPU when the user
        # has the overlay toggled off.
        self.hide()

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)
        if visible:
            self._reset_rate_state()
            self._timer.start()
        else:
            self._timer.stop()

    def sizeHint(self):  # type: ignore[override]
        from PySide6.QtCore import QSize

        return QSize(_PANEL_W, _PANEL_H)

    def _reset_rate_state(self) -> None:
        # Force the next sample to use the next-tick delta as the
        # baseline rather than reusing stale counters from a previous
        # session.
        self._prev_write_completed = None
        self._prev_total_drops = None
        self._prev_timestamp = None
        self._samples.clear()

    def _tick(self) -> None:
        snap = self._snapshot_fn()
        if snap is None:
            # No session — show placeholder. Clear samples so the graph
            # doesn't persist a frozen tail when a session ends.
            self._samples.clear()
            self.update()
            return
        self._samples.append(snap)
        # Drop samples older than the window. With maxlen=cap this
        # happens automatically on overflow, but a slow tick rate
        # leaves stale entries within the deque limit; trim explicitly.
        cutoff = snap.timestamp - _WINDOW_SECONDS
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
        self.update()

    # ---- Paint ----

    def paintEvent(self, event):  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        rect = self.rect()
        # Background panel
        painter.fillRect(rect, _BG_COLOR)
        painter.setPen(QPen(_GRID_COLOR))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        if not self._samples:
            self._draw_placeholder(painter, rect, "no session")
            return

        latest = self._samples[-1]
        if latest.last_completed < 0:
            # Session is running but no frame has completed yet. Show
            # "awaiting" text rather than an empty graph — the graph
            # area is reserved at the bottom; numeric readouts at top
            # are blank-ish but still readable, so just render normal.
            self._draw_header(painter, rect, latest)
            self._draw_placeholder(
                painter,
                rect.adjusted(0, 30, 0, -8),
                "awaiting first frame...",
            )
            return

        self._draw_header(painter, rect, latest)
        self._draw_graph(painter, rect.adjusted(8, 76, -8, -8))

    def _draw_placeholder(self, painter: QPainter, rect, text: str) -> None:
        painter.setPen(QPen(_TEXT_COLOR))
        painter.setFont(QFont("Sans", 10))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_header(
        self, painter: QPainter, rect, sample: MetricsSample
    ) -> None:
        painter.setPen(QPen(_TEXT_COLOR))
        font = QFont("Sans", 8)
        painter.setFont(font)
        # Row 1: cache hit + memory + queues
        line1 = (
            f"cache {sample.cache_hit_ratio * 100:.0f}%  "
            f"mem {sample.memory_used_mb:.0f} MB  "
            f"work {sample.work_outstanding}/{sample.work_capacity}  "
            f"write {sample.write_outstanding}/{sample.write_capacity}"
        )
        painter.drawText(rect.adjusted(8, 4, -8, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, line1)
        # Row 2: rates with color swatches via colored text
        # Use simple inline label/value pairs; coloring per value is
        # done by drawing each segment separately.
        y = 22
        x = 8
        segments = [
            ("R", f"{sample.read_fps:.0f}", _READ_COLOR),
            ("P", f"{sample.process_fps:.0f}", _PROC_COLOR),
            ("W", f"{sample.write_fps:.0f}", _WRITE_COLOR),
        ]
        if sample.total_drops > 0:
            segments.append(("D", f"{sample.total_drops}", _DROP_COLOR))
        for label, value, color in segments:
            painter.setPen(QPen(_TEXT_COLOR))
            painter.drawText(x, y + 10, f"{label} ")
            x += painter.fontMetrics().horizontalAdvance(f"{label} ")
            painter.setPen(QPen(color))
            painter.drawText(x, y + 10, value)
            x += painter.fontMetrics().horizontalAdvance(value) + 10
        # Row 3: per-processor average ms — surfaces which step in the
        # chain owns the wall-clock. Truncated to fit the panel width;
        # full info also rendered to a tooltip via paint logic upstream
        # if needed. Skipped silently when the window contains no
        # samples (paused / idle).
        painter.setPen(QPen(_TEXT_COLOR))
        if sample.processor_timings:
            parts: list[str] = []
            for name, avg_ms in sample.processor_timings.items():
                # Short name: take the first 4 chars (FaceSwapper →
                # "Face"… not very informative). Better: keep the full
                # name but cap the segment count.
                parts.append(f"{name} {avg_ms:.0f}ms")
            timings_line = "  ".join(parts)
        else:
            timings_line = "—"
        painter.drawText(
            rect.adjusted(8, 38, -8, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            timings_line,
        )
        # Row 4: explicit fps label so the user knows the rate units.
        painter.setPen(QPen(_GRID_COLOR))
        painter.drawText(
            rect.adjusted(8, 56, -8, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            f"fps · last {_WINDOW_SECONDS:.0f}s",
        )

    def _draw_graph(self, painter: QPainter, rect) -> None:
        if not self._samples:
            return
        # Axes: x = time (older → newer left→right), y = fps (0 → max).
        # Auto-scale Y so the highest series fills the panel; floor at
        # ~10 fps so a steady low rate still has visible vertical room.
        max_fps = max(
            (
                max(s.read_fps, s.process_fps, s.write_fps, s.drop_fps)
                for s in self._samples
            ),
            default=0.0,
        )
        max_fps = max(10.0, max_fps * 1.15)

        # Grid: horizontal lines at quartile heights
        painter.setPen(QPen(_GRID_COLOR))
        for i in range(1, 4):
            y = rect.top() + rect.height() * i / 4
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        # Time axis: latest sample is rightmost. Map each sample by its
        # time offset from `latest.timestamp - _WINDOW_SECONDS`.
        latest_ts = self._samples[-1].timestamp
        start_ts = latest_ts - _WINDOW_SECONDS
        rect_w = rect.width()
        rect_h = rect.height()

        def map_point(ts: float, fps: float) -> QPointF:
            x_frac = (ts - start_ts) / _WINDOW_SECONDS
            x_frac = max(0.0, min(1.0, x_frac))
            y_frac = min(1.0, fps / max_fps)
            return QPointF(
                rect.left() + x_frac * rect_w,
                rect.bottom() - y_frac * rect_h,
            )

        series = [
            (_READ_COLOR, [map_point(s.timestamp, s.read_fps) for s in self._samples]),
            (_PROC_COLOR, [map_point(s.timestamp, s.process_fps) for s in self._samples]),
            (_WRITE_COLOR, [map_point(s.timestamp, s.write_fps) for s in self._samples]),
        ]
        # Drop series only when there's anything to show — keeps the
        # graph quiet under normal operation.
        if any(s.drop_fps > 0 for s in self._samples):
            series.append(
                (_DROP_COLOR, [map_point(s.timestamp, s.drop_fps) for s in self._samples])
            )

        for color, points in series:
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            for i in range(1, len(points)):
                painter.drawLine(points[i - 1], points[i])

        # Outline the chart area so the user can see where rates land
        # against the bounding box.
        painter.setPen(QPen(_GRID_COLOR))
        painter.drawRect(QRectF(rect))


# Helper to compute write_fps / drop_fps from cumulative counters.
# Lives in this module so callers building a SnapshotFn can use the
# same delta calculation the overlay's tick logic relies on.
class CumulativeRateTracker:
    """Convert (cumulative_count, timestamp) samples into per-second
    rates by diffing consecutive readings. Returns 0.0 on the first
    reading (no baseline) and on out-of-order timestamps."""

    def __init__(self) -> None:
        self._prev_count: int | None = None
        self._prev_ts: float | None = None

    def update(self, count: int, now: float | None = None) -> float:
        if now is None:
            now = time.monotonic()
        if self._prev_count is None or self._prev_ts is None:
            self._prev_count = count
            self._prev_ts = now
            return 0.0
        dt = now - self._prev_ts
        delta = count - self._prev_count
        self._prev_count = count
        self._prev_ts = now
        if dt <= 0 or delta < 0:
            return 0.0
        return delta / dt

    def reset(self) -> None:
        self._prev_count = None
        self._prev_ts = None
