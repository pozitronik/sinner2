"""The Batch-tab content widget.

Composes:
  - QTableView populated from BatchTaskStore.
  - Toolbar above the table: Start / Pause / Stop queue + a live queue-state
    label, Reload, and Settings… (defaults). The run controls enable only when
    they apply (see _on_queue_state_changed).
  - Right-click context menu per row, scoped to the task's status: Move
    up/down, Edit, Run this task next, Pause/Cancel (running), Resume + Re-run
    from scratch (paused/failed), Re-run from scratch (done/cancelled), Delete
    cache, Delete.
  - Live updates: subscribes to BatchQueue signals to flip status /
    update progress per row.

The model is a plain QStandardItemModel with one row per task; the
task's id is stored in the Source column's UserRole so handlers can
resolve the underlying BatchTask from any row.
"""
from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sinner2.batch.queue import BatchQueue, QueueState
from sinner2.gui.confirm import confirm
from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
    resolve_output_path,
)
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
    """Ordered stage names for a task, mirroring BatchDriver's progress stages
    so a reloaded (non-running) task shows the same stage labels the live
    signal would. Both processors off → a single passthrough re-encode stage.
    The final combine/encode step (package_output) is always the last stage —
    "encode" for video output, "copy" for a frames directory."""
    names: list[str] = []
    if task.swapper_enabled:
        names.append("faceswapper")
    if task.enhancer_enabled:
        names.append("faceenhancer")
    if task.upscaler_enabled:
        names.append("upscaler")
    if not names:
        names.append("passthrough")
    names.append(
        "encode" if task.output_format is BatchOutputFormat.VIDEO else "copy"
    )
    return names

_COL_PROGRESS = 0
_COL_SOURCE = 1
_COL_TARGET = 2
_COL_OUTPUT = 3
_COL_FORMAT = 4
_COL_STATUS = 5
_COL_TEMP = 6
_COLUMN_HEADERS = (
    "Progress", "Source", "Target", "Output", "Format", "Status", "Temp",
)


# While a task runs, its cache grows over minutes/hours; re-walk it this often
# (off-thread) so the Temp cell tracks the growing cache instead of freezing on
# an early snapshot. Throttled because stat-walking a tens-of-GB dir isn't free.
_SIZE_REFRESH_SEC = 5.0


def _dir_size(path: Path) -> int:
    """Total bytes under ``path`` (recursive), tolerant of races / missing
    dirs — a task with no cache yet just sizes to 0."""
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += _dir_size(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _human_size(n: int) -> str:
    """Compact byte count: '—' for empty, else e.g. '512 KB' / '3.4 GB'."""
    if n <= 0:
        return "—"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class _SizeSignals(QObject):
    """Carrier so an off-thread size walk can hand results back on the GUI
    thread (queued delivery to the connected slot)."""

    sized = Signal(str, object)  # task_id, int bytes (object to avoid C int overflow at 2 GB+)


class _SizeJob(QRunnable):
    """Walks each task's cache dir off the GUI thread (stat-heavy) and emits
    its size — the Temp column must not stall the UI on a big cache."""

    def __init__(
        self, cache_root: Path, task_ids: list[str], signals: _SizeSignals
    ) -> None:
        super().__init__()
        self._cache_root = cache_root
        self._task_ids = task_ids
        self._signals = signals

    def run(self) -> None:
        for task_id in self._task_ids:
            self._signals.sized.emit(task_id, _dir_size(self._cache_root / task_id))


class QBatchView(QWidget):
    """The Batch tab.

    Exposes editRequested(task_id) so main_window can pop the edit
    dialog without this widget knowing about the dialog class.
    """

    editRequested = Signal(str)  # task_id
    settingsRequested = Signal()  # open the Batch settings (defaults) dialog

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
        self._start_btn.setToolTip(
            "Start processing the queue — or resume it after a Pause/Stop.\n"
            "Runs pending tasks in order. To resume a single paused/failed task,\n"
            "right-click it → Resume."
        )
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
            "Stop the queue, keeping each task's progress. The running task is\n"
            "PAUSED (its rendered frames are kept and it resumes next run), not\n"
            "discarded — use right-click → Cancel to discard a task's work."
        )
        self._stop_btn.clicked.connect(self._queue.stop)
        self._refresh_btn = QToolButton()
        self._refresh_btn.setText("Reload")
        self._refresh_btn.setToolTip(
            "Re-read the store from disk (handles external edits)."
        )
        self._refresh_btn.clicked.connect(self.reload_from_store)
        # Queue-wide defaults + paths. Lives at the far right, set apart from
        # the run controls — it configures NEW tasks, not the running queue.
        self._settings_btn = QToolButton()
        self._settings_btn.setText("Settings…")
        self._settings_btn.setToolTip(
            "Edit the defaults every new task is created with, plus the "
            "task-store and global-output folders."
        )
        self._settings_btn.clicked.connect(self.settingsRequested.emit)

        # Live queue state ("Idle / Running / Paused") so the run controls aren't
        # the only (silent) indication of what the queue is doing.
        self._status_label = QLabel()
        self._status_label.setToolTip("Current queue state.")

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._start_btn)
        toolbar.addWidget(self._pause_btn)
        toolbar.addWidget(self._stop_btn)
        toolbar.addWidget(self._refresh_btn)
        toolbar.addSpacing(12)
        toolbar.addWidget(self._status_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self._settings_btn)

        # Table.
        self._model = QStandardItemModel(0, len(_COLUMN_HEADERS), self)
        self._model.setHorizontalHeaderLabels(_COLUMN_HEADERS)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        # Extended (Ctrl/Shift) multi-select so bulk actions can target several
        # tasks at once; the context menu switches to bulk items when >1 is set.
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
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
        header.resizeSection(_COL_TEMP, 80)

        # Per-task cache size is computed off the GUI thread (stat-walk) and
        # handed back via this signal carrier → the Temp cell.
        self._size_signals = _SizeSignals(self)
        self._size_signals.sized.connect(self._on_size_computed)
        # Last monotonic time each task's cache size was (re)walked, to throttle
        # the live refresh during a run (see _on_task_progress).
        self._last_size_walk: dict[str, float] = {}

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
        # Run controls reflect the queue state (idle/running/paused) instead of
        # always being clickable.
        self._queue.queueStateChanged.connect(self._on_queue_state_changed)
        self._on_queue_state_changed(self._queue.state.value)  # initial paint

        self.reload_from_store()

    # ---- Public API ----

    def reload_from_store(self) -> None:
        """Rebuild the table from disk. Used on startup, when an
        external edit happens, or after a new task is added."""
        self._model.removeRows(0, self._model.rowCount())
        for task in self._store.list():
            self._append_row(task)
        self._recompute_sizes()

    def append_task(self, task: BatchTask) -> None:
        """Add a row for a task that's already been persisted via
        the store."""
        self._append_row(task)
        self._recompute_sizes([task.id])

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
        items[_COL_TEMP].setText("…")  # filled async by the size walk
        items[_COL_TEMP].setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        # Stash the id on the Source column so we can resolve a row → task.
        items[_COL_SOURCE].setData(task.id, _ROLE_TASK_ID)
        for it in items:
            it.setEditable(False)
        return items

    def _recompute_sizes(self, task_ids: list[str] | None = None) -> None:
        """Kick an off-thread walk of the given tasks' cache dirs (all rows when
        None) → updates the Temp cells as results arrive."""
        if task_ids is None:
            task_ids = [
                tid
                for r in range(self._model.rowCount())
                if (tid := self._task_id_at_row(r))
            ]
        if task_ids:
            QThreadPool.globalInstance().start(
                _SizeJob(self._queue.cache_root, task_ids, self._size_signals)
            )

    def _on_size_computed(self, task_id: str, nbytes: int) -> None:
        row = self._row_for_task_id(task_id)
        if row is None:
            return
        item = self._model.item(row, _COL_TEMP)
        if item is not None:
            item.setText(_human_size(nbytes))

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

    def _on_queue_state_changed(self, state_value: str) -> None:
        """Enable only the run controls that apply, and show the state.

        Start: available unless a task is actively running (idle/paused → it
        (re)starts scheduling). Pause: only while running. Stop: whenever a task
        is actually in flight to tear down — queried live, since a PAUSED queue
        may be either draining a task (stoppable) or already stopped (not)."""
        state = QueueState(state_value)
        running = state is QueueState.RUNNING
        self._start_btn.setEnabled(not running)
        self._pause_btn.setEnabled(running)
        self._stop_btn.setEnabled(self._queue.is_running)
        labels = {
            QueueState.IDLE: "Idle",
            QueueState.RUNNING: "Running",
            QueueState.PAUSED: "Paused",
        }
        text = f"Queue: {labels[state]}"
        # Name the task in flight so "Running" isn't anonymous in a long queue.
        if self._queue.is_running:
            tid = self._queue.current_task_id
            task = (
                self._store.load(tid)
                if tid is not None and self._store.exists(tid)
                else None
            )
            if task is not None:
                text += f" — {task.source_path.name} → {task.target_path.name}"
        self._status_label.setText(text)

    def _on_task_started(self, task_id: str) -> None:
        self._throughput[task_id] = _StepTracker()
        self._refresh_row(task_id)
        # The state stays RUNNING from one task to the next, so queueStateChanged
        # dedups and won't re-fire — refresh the status label's task name here.
        self._on_queue_state_changed(self._queue.state.value)

    def _on_task_progress(self, task_id: str, progress) -> None:
        row = self._row_for_task_id(task_id)
        if row is None:
            return
        # Update the cell directly (cheaper than _refresh_row, which re-loads
        # the task from disk on every tick). Shows the step index + step % +
        # frame counts + stage name, then a recent-window throughput and the
        # step's elapsed/expected time.
        # get-then-set, not setdefault: setdefault eagerly builds (and discards)
        # a _StepTracker + its deque on every tick once the key exists.
        tracker = self._throughput.get(task_id)
        if tracker is None:
            tracker = self._throughput[task_id] = _StepTracker()
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
        # Track the growing cache live (throttled) — the Temp cell otherwise
        # froze on the snapshot taken when the row was added (cache ~empty).
        now = time.monotonic()
        if now - self._last_size_walk.get(task_id, 0.0) >= _SIZE_REFRESH_SEC:
            self._last_size_walk[task_id] = now
            self._recompute_sizes([task_id])

    def _on_task_completed(self, task_id: str) -> None:
        self._throughput.pop(task_id, None)
        self._last_size_walk.pop(task_id, None)
        self._refresh_row(task_id)
        self._recompute_sizes([task_id])  # final accurate size after the run

    def _on_task_failed(self, task_id: str, _message: str) -> None:
        self._throughput.pop(task_id, None)
        self._last_size_walk.pop(task_id, None)
        self._recompute_sizes([task_id])
        # Flip the row to its failed state — Status reads "failed" with the
        # reason on hover (set in _refresh_row from the task's error_message).
        # The prominent, can't-miss surfacing of the message lives in
        # main_window (a consolidated error dialog when the queue goes idle),
        # which owns the modal surface; this widget just updates the row.
        self._refresh_row(task_id)

    # ---- Context menu ----

    def _on_context_menu(self, pos: QPoint) -> None:
        idx = self._table.indexAt(pos)
        if not idx.isValid():
            return
        task_id = self._task_id_at_row(idx.row())
        if task_id is None or not self._store.exists(task_id):
            return
        # Right-clicking a row OUTSIDE the current multi-selection acts on that
        # one row (and selects it) — matches every file manager. Clicking inside
        # a multi-selection keeps it and shows the bulk menu.
        selected = self._selected_task_ids()
        if task_id not in selected:
            self._table.selectRow(idx.row())
            selected = [task_id]
        if len(selected) > 1:
            self._show_bulk_menu(selected, pos)
        else:
            self._show_single_menu(task_id, idx.row(), pos)

    def _selected_task_ids(self) -> list[str]:
        """Task ids of the currently-selected rows, in row order."""
        rows = sorted(
            idx.row() for idx in self._table.selectionModel().selectedRows()
        )
        return [tid for r in rows if (tid := self._task_id_at_row(r))]

    def _show_single_menu(self, task_id: str, row: int, pos: QPoint) -> None:
        task = self._store.load(task_id)
        menu = QMenu(self._table)
        # Queue position: shift this task earlier/later (only when it can move).
        if row > 0:
            self._add_action(
                menu, "Move up", lambda: self._move_task(task_id, -1)
            )
        if row < self._model.rowCount() - 1:
            self._add_action(
                menu, "Move down", lambda: self._move_task(task_id, +1)
            )
        if menu.actions():
            menu.addSeparator()
        # Per-task actions depend on status + whether it's the running one.
        is_running = task.id == self._queue.current_task_id
        if not is_running:
            self._add_action(
                menu, "Edit…", lambda: self.editRequested.emit(task_id)
            )
        if is_running:
            # Pause keeps this task's rendered frames (resumable); Cancel throws
            # them away. Parallel wording so the keep-vs-discard choice is clear.
            self._add_action(
                menu, "Pause (keep progress)",
                lambda: self._queue.pause_task(task_id),
            )
            self._add_action(
                menu, "Cancel (discard progress)",
                lambda: self._queue.cancel_task(task_id),
            )
        elif task.status is BatchTaskStatus.PENDING:
            # Run THIS task next (jump it to the front of the queue, then start)
            # — the old "Run" just called queue.start(), which runs whatever
            # task is first, not necessarily this one.
            self._add_action(
                menu, "Run this task next",
                lambda: self._run_task_next(task_id),
            )
        elif task.status in (
            BatchTaskStatus.PAUSED,
            BatchTaskStatus.FAILED,
        ):
            # Resume continues from the cache; Re-run discards it and starts over.
            self._add_action(
                menu, "Resume (keep progress)",
                lambda: self._resume_task(task_id),
            )
            self._add_action(
                menu, "Re-run from scratch (discard progress)",
                lambda: self._reset_task_to_pending(task_id),
            )
        elif task.status in (
            BatchTaskStatus.COMPLETED,
            BatchTaskStatus.CANCELLED,
        ):
            self._add_action(
                menu, "Re-run from scratch (discard progress)",
                lambda: self._reset_task_to_pending(task_id),
            )
        if not is_running:
            # Frees the intermediate frames but keeps the task + its output; a
            # later re-run just re-renders. Distinct from Cancel/Re-run, which
            # also change the task's status.
            self._add_action(
                menu, "Delete cache (free disk, keep task)",
                lambda: self._delete_task_cache(task_id),
            )
        menu.addSeparator()
        delete_action = QAction("Delete", menu)
        delete_action.triggered.connect(
            lambda _checked=False, tid=task_id: self._delete_task(tid)
        )
        menu.addAction(delete_action)
        self._append_remove_completed(menu)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _show_bulk_menu(self, task_ids: list[str], pos: QPoint) -> None:
        """Menu for a multi-row selection: act on all selected tasks at once.
        The running task is skipped by each handler (it can't be mutated mid-
        run), so a selection that includes it still works for the rest."""
        menu = QMenu(self._table)
        n = len(task_ids)
        self._add_action(
            menu, f"Re-run {n} tasks from scratch (discard progress)",
            lambda: self._bulk_reset(task_ids),
        )
        self._add_action(
            menu, f"Delete cache for {n} tasks (free disk, keep tasks)",
            lambda: self._bulk_delete_cache(task_ids),
        )
        menu.addSeparator()
        self._add_action(
            menu, f"Delete {n} tasks",
            lambda: self._bulk_delete(task_ids),
        )
        self._append_remove_completed(menu)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _append_remove_completed(self, menu: QMenu) -> None:
        """Add a queue-wide 'Remove all completed tasks' item when any exist —
        a one-click cleanup independent of the current selection."""
        completed = [
            tid
            for r in range(self._model.rowCount())
            if (tid := self._task_id_at_row(r))
            and self._store.exists(tid)
            and self._store.load(tid).status is BatchTaskStatus.COMPLETED
        ]
        if not completed:
            return
        menu.addSeparator()
        self._add_action(
            menu, f"Remove all completed tasks ({len(completed)})",
            lambda: self._remove_completed(completed),
        )

    @staticmethod
    def _add_action(menu: QMenu, label: str, slot) -> None:
        action = QAction(label, menu)
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)

    def _run_task_next(self, task_id: str) -> None:
        """Jump a pending task to the front of the queue, then start scheduling.

        'Run this task next' must run THIS task, not whatever happens to be first
        — so reorder it ahead of the others, then start(). If a task is already
        running this just reorders (start() is a no-op mid-run); it then runs as
        soon as the current task finishes — i.e. genuinely 'next'."""
        ids = [
            tid
            for r in range(self._model.rowCount())
            if (tid := self._task_id_at_row(r))
        ]
        if task_id in ids:
            ids.remove(task_id)
            ids.insert(0, task_id)
            self._store.set_order(ids)
            self.reload_from_store()
            moved = self._row_for_task_id(task_id)
            if moved is not None:
                self._table.selectRow(moved)
        self._queue.start()

    def _resume_task(self, task_id: str) -> None:
        """Re-queue a paused/failed task keeping its cache, then refresh."""
        self._queue.resume_task(task_id)
        self._refresh_row(task_id)

    def _reset_task_to_pending(self, task_id: str) -> None:
        """Confirm, then reset the task to Pending and discard its
        processed-frame cache so it re-runs from scratch."""
        if not confirm(
            self,
            "reset_task",
            "Re-run from scratch",
            "Re-run this task from scratch?\n\n"
            "Frames already processed for this task will be discarded.",
        ):
            return
        self._queue.refresh_task(task_id)
        self._refresh_row(task_id)

    def _move_task(self, task_id: str, delta: int) -> None:
        """Shift a task's queue position by ``delta`` (-1 up / +1 down) and
        persist the new order, then rebuild + reselect the moved row."""
        ids = [
            tid
            for r in range(self._model.rowCount())
            if (tid := self._task_id_at_row(r))
        ]
        if task_id not in ids:
            return
        i = ids.index(task_id)
        j = i + delta
        if not 0 <= j < len(ids):
            return
        ids[i], ids[j] = ids[j], ids[i]
        self._store.set_order(ids)
        self.reload_from_store()
        moved = self._row_for_task_id(task_id)
        if moved is not None:
            self._table.selectRow(moved)

    def _delete_task_cache(self, task_id: str) -> None:
        """Free a task's cached intermediate frames (keeps the task + its
        output); a re-run re-renders from scratch."""
        if task_id == self._queue.current_task_id:
            QMessageBox.warning(
                self,
                "Delete cache",
                "Cannot clear a running task's cache. Cancel it first.",
            )
            return
        if not confirm(
            self,
            "delete_task_cache",
            "Delete cache",
            "Delete this task's cached intermediate frames to free disk space?\n\n"
            "The final output (if any) is kept. If the task is re-run it will "
            "re-render from scratch.",
        ):
            return
        self._queue.delete_task_cache(task_id)
        self._refresh_row(task_id)
        self._recompute_sizes([task_id])

    def _delete_task(self, task_id: str) -> None:
        if task_id == self._queue.current_task_id:
            QMessageBox.warning(
                self,
                "Delete task",
                "Cannot delete a running task. Cancel it first.",
            )
            return
        if confirm(
            self, "delete_task", "Delete task", "Remove this task from the batch?"
        ):
            self._store.delete(task_id)
            row = self._row_for_task_id(task_id)
            if row is not None:
                self._model.removeRow(row)

    # ---- Bulk actions (multi-selection) ----

    def _runnable_targets(self, task_ids: list[str], verb: str) -> list[str]:
        """Drop the running task from a bulk target list (it can't be mutated
        mid-run); warn if that leaves nothing to do. Returns the actionable ids."""
        targets = [t for t in task_ids if t != self._queue.current_task_id]
        if not targets:
            QMessageBox.warning(
                self, verb, f"Cannot {verb.lower()} the running task. Cancel it first."
            )
        return targets

    def _bulk_reset(self, task_ids: list[str]) -> None:
        targets = self._runnable_targets(task_ids, "Re-run")
        if not targets or not confirm(
            self, "reset_task", "Re-run from scratch",
            f"Re-run {len(targets)} task(s) from scratch?\n\n"
            "Frames already processed for these tasks will be discarded.",
        ):
            return
        for tid in targets:
            self._queue.refresh_task(tid)
        self.reload_from_store()

    def _bulk_delete_cache(self, task_ids: list[str]) -> None:
        targets = self._runnable_targets(task_ids, "Delete cache")
        if not targets or not confirm(
            self, "delete_task_cache", "Delete cache",
            f"Delete cached intermediate frames for {len(targets)} task(s)?\n\n"
            "Final outputs (if any) are kept; a re-run re-renders from scratch.",
        ):
            return
        for tid in targets:
            self._queue.delete_task_cache(tid)
        self.reload_from_store()
        self._recompute_sizes(targets)

    def _bulk_delete(self, task_ids: list[str]) -> None:
        targets = self._runnable_targets(task_ids, "Delete")
        if not targets or not confirm(
            self, "delete_task", "Delete tasks",
            f"Remove {len(targets)} task(s) from the batch?",
        ):
            return
        for tid in targets:
            self._store.delete(tid)
        self.reload_from_store()

    def _remove_completed(self, task_ids: list[str]) -> None:
        """Delete every completed task (queue-wide cleanup). ``task_ids`` is the
        completed set captured when the menu was built."""
        if not task_ids or not confirm(
            self, "delete_task", "Remove completed",
            f"Remove {len(task_ids)} completed task(s) from the batch?",
        ):
            return
        for tid in task_ids:
            self._store.delete(tid)
        self.reload_from_store()

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
