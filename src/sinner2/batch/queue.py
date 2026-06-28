"""Sequential scheduler for batch tasks.

One running task at a time. Tasks live in BatchTaskStore (file-backed).
The queue picks Pending tasks in file-mtime order, spawns a QThread to
drive each, and emits Qt signals for the GUI. Per-task pause/cancel
forwards to the active driver; refresh resets a task's status to
Pending so the queue picks it up next.

State machine:
  - idle         — no task running, queue not actively scheduling.
  - running      — a task is being driven; queue picks next on completion.
  - paused       — queue is suspended; the current task (if any) is also
                   paused via the driver.

Threading: BatchQueue is a QObject; methods are called from the GUI
thread. The runner QThread executes BatchDriver.run() off the GUI
thread. Driver progress callbacks marshal back via queued signals.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal

from sinner2.batch.driver import BatchDriver
from sinner2.batch.task import (
    BatchProgress,
    BatchTask,
    BatchTaskStatus,
)
from sinner2.batch.task_store import BatchTaskStore


class QueueState(str, Enum):
    """The queue's scheduling state, broadcast via ``queueStateChanged`` so the
    toolbar can enable only the controls that apply right now.

    - IDLE: nothing running, not paused — Start is the only meaningful action.
    - RUNNING: a task is being driven — Pause / Stop apply, Start does not.
    - PAUSED: scheduling suspended (a task may still be draining) — Start
      resumes, Stop tears down; Pause is already in effect.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"


class _DriverWorker(QObject):
    """Lives on a QThread; runs one task via driver.run() and emits
    completion + progress signals back to the queue."""

    progress = Signal(str, object)      # task_id, BatchProgress
    preview = Signal(str, object)       # task_id, Frame
    completed = Signal(str, str)        # task_id, terminal_status_value

    def __init__(
        self,
        driver: BatchDriver,
        task: BatchTask,
    ) -> None:
        super().__init__()
        self._driver = driver
        self._task = task

    def run(self) -> None:
        def on_progress(progress: BatchProgress) -> None:
            self.progress.emit(self._task.id, progress)

        def on_preview(frame: object) -> None:
            self.preview.emit(self._task.id, frame)

        status = self._driver.run(
            self._task,
            progress_callback=on_progress,
            preview_callback=on_preview,
        )
        self.completed.emit(self._task.id, status.value)


def _fast_delete_dir(path: Path) -> threading.Thread | None:
    """Free a directory FAST: rename it aside (atomic + instant on the same
    filesystem) then rmtree the renamed copy on a daemon thread.

    ``shutil.rmtree`` is not an atomic drop — it unlinks every file one by one,
    so on a stage dir of tens of thousands of frames it takes seconds and must
    not block the caller (often the GUI thread). Returns the deletion thread
    (None when there was nothing to delete) so callers/tests can join it."""
    if not path.exists():
        return None
    aside = path.with_name(f".deleting-{path.name}-{uuid.uuid4().hex[:8]}")
    try:
        os.replace(path, aside)  # instant same-filesystem rename
    except OSError:
        # Cross-device / locked — can't rename; rmtree the original off-thread.
        aside = path
    thread = threading.Thread(
        target=lambda: shutil.rmtree(aside, ignore_errors=True),
        name="batch-cache-delete",
        daemon=True,
    )
    thread.start()
    return thread


class BatchQueue(QObject):
    """The queue. Sequence: idle → start() → drive Pending tasks in
    order → idle when queue empties."""

    taskStarted = Signal(str)              # task_id
    taskProgress = Signal(str, object)     # task_id, BatchProgress
    taskPreview = Signal(str, object)      # task_id, Frame (throttled)
    taskCompleted = Signal(str)            # task_id (terminal: completed/cancelled/failed/paused)
    taskFailed = Signal(str, str)          # task_id, error_message
    queueIdle = Signal()
    queueStateChanged = Signal(str)        # QueueState value (idle/running/paused)

    def __init__(
        self,
        store: BatchTaskStore,
        cache_root: Path,
        *,
        global_output_dir: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._cache_root = cache_root
        self._global_output_dir = global_output_dir
        self._driver: BatchDriver | None = None
        self._thread: QThread | None = None
        self._worker: _DriverWorker | None = None
        self._current_task_id: str | None = None
        # The BatchTask the runner is driving — captured here (not read off the
        # worker) so _on_completed can always persist + report the terminal
        # state even if a teardown raced the queued completion signal.
        self._current_task: BatchTask | None = None
        self._paused = False  # queue-level pause
        # Last state broadcast via queueStateChanged, so we emit only on a real
        # transition (toolbar slots are cheap, but dedup keeps the signal honest).
        self._last_state: QueueState | None = None
        # Throttle progress persistence: _on_progress fires per-frame, but the
        # full task JSON only needs to hit disk occasionally (resume reads the
        # frame cache, not this counter). Saving every tick churns the file and
        # widens the Windows AV/indexer collision window on os.replace.
        self._last_progress_save = 0.0  # time.monotonic() of last persisted tick
        self._last_saved_stage = -1     # force a save when the stage advances

    # ---- Queue state ----

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def state(self) -> QueueState:
        """Current scheduling state. PAUSED takes precedence over a still-
        draining task: once the user pauses, the queue won't schedule more even
        if the current task is finishing, so the controls should read 'paused'."""
        if self._paused:
            return QueueState.PAUSED
        if self.is_running:
            return QueueState.RUNNING
        return QueueState.IDLE

    def _emit_state(self) -> None:
        """Broadcast the current state if it changed since the last emission."""
        state = self.state
        if state != self._last_state:
            self._last_state = state
            self.queueStateChanged.emit(state.value)

    @property
    def current_task_id(self) -> str | None:
        return self._current_task_id

    def set_global_output_dir(self, path: Path | None) -> None:
        """Update the folder all task outputs land in (None = next to each
        target). A fresh BatchDriver is built per run from this value, so the
        change takes effect on the next task started — it can't disturb a run
        already in flight."""
        self._global_output_dir = path

    # ---- Queue controls ----

    def start(self) -> None:
        """Begin — or resume — processing the queue. Runs Pending tasks AND
        resumes Paused ones (from their kept cache) in queue order, so a
        Stop→Start, or a per-task Pause then Start, continues from where it left
        off. FAILED tasks are left alone (Resume/Re-run them explicitly)."""
        self._paused = False
        self._schedule_next()
        self._emit_state()

    def pause(self) -> None:
        """Stop scheduling new tasks. Does NOT interrupt the running
        task — use pause_task(id) for that."""
        self._paused = True
        self._emit_state()

    def stop(self) -> None:
        """Stop the queue at app shutdown WITHOUT losing work. PAUSES the running
        task (if any) rather than cancelling it: pause returns a PAUSED status
        that leaves the task's already-rendered frames on disk so it resumes next
        run, whereas cancel() rmtree's the whole per-task cache and resets the
        resume markers (data loss). Pause still lets the runner thread finish, so
        we don't leak it. Explicit user cancellation goes through cancel_task()."""
        self._paused = True
        if self._driver is not None:
            self._driver.pause()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(30_000)  # generous: in-flight encode can be slow
        self._driver = None
        self._thread = None
        self._worker = None
        self._current_task_id = None
        self._current_task = None
        self._emit_state()

    # ---- Per-task controls ----

    def pause_task(self, task_id: str) -> None:
        """Pause the currently-running task. No-op if a different
        task is running."""
        if self._current_task_id == task_id and self._driver is not None:
            self._driver.pause()

    def cancel_task(self, task_id: str) -> None:
        """Cancel the currently-running task. Discards its partial
        cache. No-op if a different task is running."""
        if self._current_task_id == task_id and self._driver is not None:
            self._driver.cancel()

    def resume_task(self, task_id: str) -> None:
        """Re-queue a paused or failed task WITHOUT discarding its cache, so
        the driver continues from where it stopped, then start scheduling.
        No-op if the task is running or not in a resumable state."""
        if self._current_task_id == task_id:
            return
        if not self._store.exists(task_id):
            return
        task = self._store.load(task_id)
        if task.status not in (
            BatchTaskStatus.PAUSED,
            BatchTaskStatus.FAILED,
        ):
            return
        task.status = BatchTaskStatus.PENDING
        task.error_message = None
        self._store.save(task)
        self.start()

    def refresh_task(self, task_id: str) -> None:
        """Reset a task to Pending and DISCARD its processed-frame cache
        so the queue re-runs it from scratch. No-op if currently running."""
        if self._current_task_id == task_id:
            return  # can't refresh a task mid-run
        if not self._store.exists(task_id):
            return
        # Wipe the per-task cache dir. Otherwise the driver's
        # resume-from-cache would skip every already-rendered frame, so a
        # "refresh" (e.g. after a param edit) would only re-encode, never
        # re-process. Refresh == full re-run.
        shutil.rmtree(self._cache_root / task_id, ignore_errors=True)
        task = self._store.load(task_id)
        task.status = BatchTaskStatus.PENDING
        task.last_completed_frame = -1
        task.total_frames = -1
        task.error_message = None
        task.started_at = None
        task.finished_at = None
        self._store.save(task)
        # Don't auto-start the queue — user clicks Start when ready.

    @property
    def cache_root(self) -> Path:
        """Root holding every task's processed-frame cache (one dir per id)."""
        return self._cache_root

    def task_cache_dir(self, task_id: str) -> Path:
        """The per-task cache dir (intermediate stage frames)."""
        return self._cache_root / task_id

    def delete_task_cache(self, task_id: str) -> bool:
        """Free a task's processed-frame cache to reclaim disk WITHOUT changing
        its status — a COMPLETED task keeps its final output; the intermediate
        frames just go. Resets the resume markers so a later re-run re-renders
        cleanly (the frames are gone). No-op (returns False) on the running task
        — its cache is in active use. Returns True when the cache was cleared."""
        if self._current_task_id == task_id:
            return False
        # Rename-aside + background rmtree: the dir is gone for the caller at
        # once (so the UI frees instantly), the slow per-file unlink runs off
        # the GUI thread.
        _fast_delete_dir(self._cache_root / task_id)
        if self._store.exists(task_id):
            task = self._store.load(task_id)
            task.completed_stages = 0
            task.last_completed_frame = -1
            task.cache_fingerprint = ""
            self._store.save(task)
        return True

    # ---- Scheduling ----

    def _schedule_next(self) -> None:
        if self._paused or self.is_running:
            return
        next_task = self._pop_pending()
        if next_task is None:
            self.queueIdle.emit()
            self._emit_state()
            return
        self._spawn_runner(next_task)

    def _pop_pending(self) -> BatchTask | None:
        """Return the next runnable task in queue order, or None.

        Runnable = Pending OR Paused. A Paused task is resumed from its kept
        cache (its frames weren't discarded), so Start — and auto-advance —
        continue past a Stop or a per-task Pause instead of SKIPPING the paused
        task to the next never-run one (which read as 'Start runs the second
        task'). FAILED tasks are deliberately NOT picked up: they halt for
        inspection — the user resumes/re-runs them explicitly."""
        for task in self._store.list():
            if task.status in (
                BatchTaskStatus.PENDING,
                BatchTaskStatus.PAUSED,
            ):
                return task
        return None

    def _spawn_runner(self, task: BatchTask) -> None:
        self._current_task_id = task.id
        self._current_task = task
        self._driver = BatchDriver(
            cache_root=self._cache_root,
            global_output_dir=self._global_output_dir,
        )
        thread = QThread(self)
        worker = _DriverWorker(self._driver, task)
        worker.moveToThread(thread)
        # Queued connections route slots to the GUI thread (BatchQueue
        # was constructed there). Without explicit queue type, the
        # bound-method receiver detection would still pick QueuedConnection
        # since the worker is on a different thread — but being explicit
        # makes the threading contract unambiguous.
        worker.progress.connect(
            self._on_progress, type=Qt.ConnectionType.QueuedConnection
        )
        worker.preview.connect(
            self._on_preview, type=Qt.ConnectionType.QueuedConnection
        )
        worker.completed.connect(
            self._on_completed, type=Qt.ConnectionType.QueuedConnection
        )
        thread.started.connect(worker.run)
        self._thread = thread
        self._worker = worker
        self.taskStarted.emit(task.id)
        thread.start()
        self._emit_state()

    # ---- Worker callbacks (GUI thread) ----

    def _on_progress(self, task_id: str, progress: BatchProgress) -> None:
        # Persist a consistent snapshot so a reload (or app restart) mid-task
        # shows sensible progress. completed_stages = stage_index (fully-done
        # prior stages); last_completed_frame tracks the current stage. The
        # final resume marker is written by _on_completed at terminal state.
        #
        # Throttled: persist at most ~once/second, but always on a stage
        # advance (a meaningful resume boundary). The per-tick UI update is
        # emitted unconditionally below.
        now = time.monotonic()
        stage_advanced = progress.stage_index != self._last_saved_stage
        if (stage_advanced or now - self._last_progress_save >= 1.0) and self._store.exists(task_id):
            task = self._store.load(task_id)
            task.completed_stages = progress.stage_index
            task.last_completed_frame = progress.stage_completed - 1
            task.total_frames = progress.stage_total
            self._store.save(task)
            self._last_progress_save = now
            self._last_saved_stage = progress.stage_index
        self.taskProgress.emit(task_id, progress)

    def _on_preview(self, task_id: str, frame: object) -> None:
        # Transient — not persisted; forward straight to the GUI.
        self.taskPreview.emit(task_id, frame)

    def _on_completed(self, task_id: str, status_value: str) -> None:
        terminal = BatchTaskStatus(status_value)
        # The worker mutated this BatchTask in place; it's the same object we
        # captured in _spawn_runner, safe to read here on the GUI thread (the
        # worker thread has exited — `completed` is its last emission). Captured
        # on the queue, not read off _worker, so a teardown race can't drop the
        # terminal signals.
        task = self._current_task
        if task is not None:
            self._store.save(task)
            if terminal is BatchTaskStatus.FAILED:
                self.taskFailed.emit(task_id, task.error_message or "")
        self.taskCompleted.emit(task_id)
        self._teardown_runner()
        # A FAILED task HALTS the queue (so the user sees the error and decides)
        # unless it opted into continue-on-error — then auto-skip it and roll on.
        # Pause the queue on a halt so a stray _schedule_next won't advance; the
        # user clears it via start() / resume_task().
        if terminal is BatchTaskStatus.FAILED and not (
            task is not None and task.continue_on_error
        ):
            self._paused = True
        # Auto-pick the next pending task unless we're paused or the task reached
        # a user-intent stop: Cancel OR Pause (pause_task pauses just the current
        # task without setting the queue-level _paused flag, so PAUSED must also
        # halt scheduling — else pausing one task silently starts the next).
        if not self._paused and terminal not in (
            BatchTaskStatus.CANCELLED,
            BatchTaskStatus.PAUSED,
        ):
            self._schedule_next()
        else:
            self.queueIdle.emit()
            self._emit_state()

    def _teardown_runner(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self._driver = None
        self._current_task_id = None
        self._current_task = None
