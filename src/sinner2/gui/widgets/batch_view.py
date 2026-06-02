"""The Batch-tab content widget.

Composes:
  - QTableView populated from BatchTaskStore.
  - Toolbar above the table: Start / Pause / Stop queue, Open output folder.
  - Right-click context menu per row: Edit, Run now, Pause, Cancel,
    Refresh, Delete.
  - Live updates: subscribes to BatchQueue signals to flip status /
    update progress per row.

The model is a plain QStandardItemModel with one row per task; the
task's id is stored in the Source column's UserRole so handlers can
resolve the underlying BatchTask from any row.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QMessageBox,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sinner2.batch.queue import BatchQueue
from sinner2.batch.task import BatchTask, BatchTaskStatus, resolve_output_path
from sinner2.batch.task_store import BatchTaskStore


_ROLE_TASK_ID = Qt.ItemDataRole.UserRole + 1


def _fmt_eta(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class _StepTracker:
    """Per-task frames/sec + elapsed/expected time, scoped to the CURRENT step.

    Everything resets when the stage index advances, so the rate, elapsed,
    and expected duration all describe the step that's running now — the
    enhancer step is slower than the swap step, and a resume burns through
    cached frames instantly; an average-since-start would smear all of that
    together. fps is measured over a short trailing window (not since the
    step began) so it tracks the current speed rather than lagging it.
    """

    _WINDOW_S = 3.0

    def __init__(self) -> None:
        self._stage_index: int | None = None
        self._step_start = 0.0
        self._samples: deque[tuple[float, int]] = deque()

    def update(
        self, stage_index: int, completed: int, total: int
    ) -> tuple[float, float, float | None]:
        """Return (fps, elapsed_s, expected_total_s) for the current step.

        expected_total is elapsed + projected-remaining; None until there's a
        rate to project from (or the step is already complete)."""
        now = time.monotonic()
        if stage_index != self._stage_index:
            # New step: drop the prior step's samples and restart the clock.
            self._stage_index = stage_index
            self._step_start = now
            self._samples.clear()
        self._samples.append((now, completed))
        cutoff = now - self._WINDOW_S
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()
        elapsed = now - self._step_start
        fps = 0.0
        if len(self._samples) >= 2:
            t0, c0 = self._samples[0]
            span = now - t0
            if span > 0:
                fps = (completed - c0) / span
        remaining = (
            (total - completed) / fps if fps > 0 and total > completed else None
        )
        expected = elapsed + remaining if remaining is not None else None
        return fps, elapsed, expected


def _format_progress(
    stage_index: int,
    stage_count: int,
    stage_name: str,
    completed: int,
    total: int,
) -> str:
    """Step-scoped progress line, e.g. `[1/2] 15% (1500/10000, faceswapper)`.

    The percentage is of the CURRENT step (completed/total), not the overall
    job — it lines up with the (completed/total) pair shown alongside it."""
    if total <= 0:
        return ""
    pct = round(completed / total * 100)
    return f"[{stage_index + 1}/{stage_count}] {pct}% ({completed}/{total}, {stage_name})"


def _stage_names(task: BatchTask) -> list[str]:
    """Ordered stage names for a task, mirroring BatchDriver._build_stages so
    a reloaded (non-running) task shows the same stage labels the live signal
    would. Both processors off → a single passthrough re-encode stage."""
    names: list[str] = []
    if task.swapper_enabled:
        names.append("faceswapper")
    if task.enhancer_enabled:
        names.append("faceenhancer")
    if not names:
        names.append("passthrough")
    return names

_COL_PROGRESS = 0
_COL_SOURCE = 1
_COL_TARGET = 2
_COL_OUTPUT = 3
_COL_FORMAT = 4
_COL_STATUS = 5
_COLUMN_HEADERS = ("Progress", "Source", "Target", "Output", "Format", "Status")


class QBatchView(QWidget):
    """The Batch tab.

    Exposes editRequested(task_id) so main_window can pop the edit
    dialog without this widget knowing about the dialog class.
    """

    editRequested = Signal(str)  # task_id

    def __init__(
        self,
        store: BatchTaskStore,
        queue: BatchQueue,
        *,
        global_output_dir_resolver: Callable[[], Path | None] = lambda: None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._queue = queue
        self._resolve_global_output_dir = global_output_dir_resolver

        # Toolbar: queue-level actions.
        self._start_btn = QToolButton()
        self._start_btn.setText("Start")
        self._start_btn.setToolTip("Start processing the queue.")
        self._start_btn.clicked.connect(self._queue.start)
        self._pause_btn = QToolButton()
        self._pause_btn.setText("Pause")
        self._pause_btn.setToolTip(
            "Stop scheduling new tasks. The running task continues.\n"
            "Use right-click → Pause on a row to interrupt the running task."
        )
        self._pause_btn.clicked.connect(self._queue.pause)
        self._stop_btn = QToolButton()
        self._stop_btn.setText("Stop")
        self._stop_btn.setToolTip(
            "Cancel the running task and stop the queue."
        )
        self._stop_btn.clicked.connect(self._queue.stop)
        self._refresh_btn = QToolButton()
        self._refresh_btn.setText("Reload")
        self._refresh_btn.setToolTip(
            "Re-read the store from disk (handles external edits)."
        )
        self._refresh_btn.clicked.connect(self.reload_from_store)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._start_btn)
        toolbar.addWidget(self._pause_btn)
        toolbar.addWidget(self._stop_btn)
        toolbar.addWidget(self._refresh_btn)
        toolbar.addStretch(1)

        # Table.
        self._model = QStandardItemModel(0, len(_COLUMN_HEADERS), self)
        self._model.setHorizontalHeaderLabels(_COLUMN_HEADERS)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.doubleClicked.connect(
            lambda idx: self._emit_edit_for_row(idx.row())
        )
        # Stretch the path columns so they reveal as much filename as
        # the user has space for.
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        for col in (_COL_SOURCE, _COL_TARGET, _COL_OUTPUT):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        # Progress carries the longest text ("[1/2] 15% (1500/10000, …) · …"),
        # so give it a roomy default; it stays user-resizable (Interactive).
        header.resizeSection(_COL_PROGRESS, 320)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self._table, stretch=1)

        # Per-running-task step throughput/time trackers, keyed by task id.
        self._throughput: dict[str, _StepTracker] = {}

        # Wire queue signals → row updates.
        self._queue.taskStarted.connect(self._on_task_started)
        self._queue.taskProgress.connect(self._on_task_progress)
        self._queue.taskCompleted.connect(self._on_task_completed)
        self._queue.taskFailed.connect(self._on_task_failed)

        self.reload_from_store()

    # ---- Public API ----

    def reload_from_store(self) -> None:
        """Rebuild the table from disk. Used on startup, when an
        external edit happens, or after a new task is added."""
        self._model.removeRows(0, self._model.rowCount())
        for task in self._store.list():
            self._append_row(task)

    def append_task(self, task: BatchTask) -> None:
        """Add a row for a task that's already been persisted via
        the store."""
        self._append_row(task)

    # ---- Row plumbing ----

    def _append_row(self, task: BatchTask) -> None:
        items = self._task_to_row(task)
        self._model.appendRow(items)

    def _task_to_row(self, task: BatchTask) -> list[QStandardItem]:
        items = [QStandardItem() for _ in _COLUMN_HEADERS]
        # Source / Target / Output use just the name for the table,
        # full path lives in the tooltip so the user can hover.
        items[_COL_SOURCE].setText(task.source_path.name)
        items[_COL_SOURCE].setToolTip(str(task.source_path))
        items[_COL_TARGET].setText(task.target_path.name)
        items[_COL_TARGET].setToolTip(str(task.target_path))
        out_resolved = resolve_output_path(task, self._resolve_global_output_dir())
        items[_COL_OUTPUT].setText(out_resolved.name)
        items[_COL_OUTPUT].setToolTip(str(out_resolved))
        items[_COL_FORMAT].setText(task.output_format.value)
        items[_COL_STATUS].setText(task.status.value)
        if task.error_message:
            items[_COL_STATUS].setToolTip(task.error_message)
        items[_COL_PROGRESS].setText(self._progress_text(task))
        # Stash the id on the Source column so we can resolve a row → task.
        items[_COL_SOURCE].setData(task.id, _ROLE_TASK_ID)
        for it in items:
            it.setEditable(False)
        return items

    @staticmethod
    def _progress_text(task: BatchTask) -> str:
        """Step-scoped progress for a task loaded from the store (no live
        signal). Mirrors the live format using the persisted stage marker +
        current-stage frame, so a paused/completed row reads the same as a
        running one (minus the throughput/time, which need the live rate)."""
        total = task.total_frames
        if total <= 0:
            return ""
        names = _stage_names(task)
        stage_count = len(names)
        # completed_stages = fully-done prior stages = the current stage index;
        # clamp into range (a completed task may carry completed_stages ==
        # stage_count). A completed task shows the final stage at 100%.
        stage_index = min(max(0, task.completed_stages), stage_count - 1)
        if task.status is BatchTaskStatus.COMPLETED:
            stage_index = stage_count - 1
            completed = total
        else:
            completed = min(total, max(0, task.last_completed_frame + 1))
        return _format_progress(
            stage_index, stage_count, names[stage_index], completed, total
        )

    def _row_for_task_id(self, task_id: str) -> int | None:
        for row in range(self._model.rowCount()):
            if self._model.item(row, _COL_SOURCE).data(_ROLE_TASK_ID) == task_id:
                return row
        return None

    def _task_id_at_row(self, row: int) -> str | None:
        item = self._model.item(row, _COL_SOURCE)
        return item.data(_ROLE_TASK_ID) if item is not None else None

    def _refresh_row(self, task_id: str) -> None:
        row = self._row_for_task_id(task_id)
        if row is None or not self._store.exists(task_id):
            return
        task = self._store.load(task_id)
        status_item = self._model.item(row, _COL_STATUS)
        status_item.setText(task.status.value)
        # Surface the failure reason on hover (Status cell tooltip).
        status_item.setToolTip(task.error_message or "")
        self._model.item(row, _COL_PROGRESS).setText(self._progress_text(task))
        out_resolved = resolve_output_path(task, self._resolve_global_output_dir())
        self._model.item(row, _COL_OUTPUT).setText(out_resolved.name)
        self._model.item(row, _COL_OUTPUT).setToolTip(str(out_resolved))
        self._model.item(row, _COL_FORMAT).setText(task.output_format.value)

    # ---- Queue signal handlers ----

    def _on_task_started(self, task_id: str) -> None:
        self._throughput[task_id] = _StepTracker()
        self._refresh_row(task_id)

    def _on_task_progress(self, task_id: str, progress) -> None:
        row = self._row_for_task_id(task_id)
        if row is None:
            return
        # Update the cell directly (cheaper than _refresh_row, which re-loads
        # the task from disk on every tick). Shows the step index + step % +
        # frame counts + stage name, then a recent-window throughput and the
        # step's elapsed/expected time.
        tracker = self._throughput.setdefault(task_id, _StepTracker())
        fps, elapsed, expected = tracker.update(
            progress.stage_index, progress.stage_completed, progress.stage_total
        )
        parts = [
            _format_progress(
                progress.stage_index,
                progress.stage_count,
                progress.stage_name,
                progress.stage_completed,
                progress.stage_total,
            )
        ]
        if fps > 0:
            parts.append(f"{fps:.0f} fps")
        # Rough elapsed / expected-total for the current step; the "~" flags
        # that the expected value is a moving estimate from the current rate.
        time_part = _fmt_eta(elapsed)
        if expected is not None:
            time_part += f" / ~{_fmt_eta(expected)}"
        parts.append(time_part)
        self._model.item(row, _COL_PROGRESS).setText(" · ".join(parts))

    def _on_task_completed(self, task_id: str) -> None:
        self._throughput.pop(task_id, None)
        self._refresh_row(task_id)

    def _on_task_failed(self, task_id: str, message: str) -> None:
        self._throughput.pop(task_id, None)
        self._refresh_row(task_id)
        # No modal — failures are common (missing model, codec) and
        # spamming a popup per failure would be annoying. The Status
        # cell shows "failed" and the tooltip on the row could hold
        # the message; for v1 we keep the message in the task file
        # itself (visible via Edit).
        _ = message

    # ---- Context menu ----

    def _on_context_menu(self, pos: QPoint) -> None:
        idx = self._table.indexAt(pos)
        if not idx.isValid():
            return
        task_id = self._task_id_at_row(idx.row())
        if task_id is None or not self._store.exists(task_id):
            return
        task = self._store.load(task_id)
        menu = QMenu(self._table)
        # Per-task actions depend on status + whether it's the running one.
        is_running = task.id == self._queue.current_task_id
        if not is_running:
            self._add_action(
                menu, "Edit…", lambda: self.editRequested.emit(task_id)
            )
        if is_running:
            self._add_action(
                menu, "Pause this task",
                lambda: self._queue.pause_task(task_id),
            )
            self._add_action(
                menu, "Cancel this task (discard cache)",
                lambda: self._queue.cancel_task(task_id),
            )
        elif task.status is BatchTaskStatus.PENDING:
            self._add_action(menu, "Run", self._queue.start)
        elif task.status in (
            BatchTaskStatus.PAUSED,
            BatchTaskStatus.FAILED,
        ):
            self._add_action(menu, "Resume", lambda: self._resume_task(task_id))
            self._add_action(
                menu, "Reset to Pending (discard cache)",
                lambda: self._reset_task_to_pending(task_id),
            )
        elif task.status in (
            BatchTaskStatus.COMPLETED,
            BatchTaskStatus.CANCELLED,
        ):
            self._add_action(
                menu, "Reset to Pending (discard cache)",
                lambda: self._reset_task_to_pending(task_id),
            )
        menu.addSeparator()
        delete_action = QAction("Delete", menu)
        delete_action.triggered.connect(
            lambda _checked=False, tid=task_id: self._delete_task(tid)
        )
        menu.addAction(delete_action)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    @staticmethod
    def _add_action(menu: QMenu, label: str, slot) -> None:
        action = QAction(label, menu)
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)

    def _resume_task(self, task_id: str) -> None:
        """Re-queue a paused/failed task keeping its cache, then refresh."""
        self._queue.resume_task(task_id)
        self._refresh_row(task_id)

    def _reset_task_to_pending(self, task_id: str) -> None:
        """Confirm, then reset the task to Pending and discard its
        processed-frame cache so it re-runs from scratch."""
        reply = QMessageBox.question(
            self,
            "Reset to Pending",
            "Reset this task to Pending and re-run it from scratch?\n\n"
            "Frames already processed for this task will be discarded.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._queue.refresh_task(task_id)
        self._refresh_row(task_id)

    def _delete_task(self, task_id: str) -> None:
        if task_id == self._queue.current_task_id:
            QMessageBox.warning(
                self,
                "Delete task",
                "Cannot delete a running task. Cancel it first.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete task",
            "Remove this task from the batch?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._store.delete(task_id)
            row = self._row_for_task_id(task_id)
            if row is not None:
                self._model.removeRow(row)

    def _emit_edit_for_row(self, row: int) -> None:
        task_id = self._task_id_at_row(row)
        if task_id is None:
            return
        # Editing the RUNNING task races the queue's store writer and re-opens
        # its resume state mid-render. The context menu hides Edit for the
        # running task; the double-click path must honour the same guard.
        if task_id == self._queue.current_task_id:
            return
        self.editRequested.emit(task_id)
