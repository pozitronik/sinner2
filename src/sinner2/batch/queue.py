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

import shutil
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal

from sinner2.batch.driver import BatchDriver
from sinner2.batch.task import BatchTask, BatchTaskStatus
from sinner2.batch.task_store import BatchTaskStore


class _DriverWorker(QObject):
    """Lives on a QThread; runs one task via driver.run() and emits
    completion + progress signals back to the queue."""

    progress = Signal(str, int, int)   # task_id, completed, total
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
        def on_progress(completed: int, total: int) -> None:
            self.progress.emit(self._task.id, completed, total)

        status = self._driver.run(self._task, progress_callback=on_progress)
        self.completed.emit(self._task.id, status.value)


class BatchQueue(QObject):
    """The queue. Sequence: idle → start() → drive Pending tasks in
    order → idle when queue empties."""

    taskStarted = Signal(str)              # task_id
    taskProgress = Signal(str, int, int)   # task_id, completed, total
    taskCompleted = Signal(str)            # task_id (terminal: completed/cancelled/failed/paused)
    taskFailed = Signal(str, str)          # task_id, error_message
    queueIdle = Signal()

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
        self._paused = False  # queue-level pause

    # ---- Queue state ----

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def current_task_id(self) -> str | None:
        return self._current_task_id

    # ---- Queue controls ----

    def start(self) -> None:
        """Begin (or resume) scheduling. If a task is currently
        paused, this is a no-op — the user should resume that task
        explicitly via refresh_task() / pause_task()."""
        self._paused = False
        self._schedule_next()

    def pause(self) -> None:
        """Stop scheduling new tasks. Does NOT interrupt the running
        task — use pause_task(id) for that."""
        self._paused = True

    def stop(self) -> None:
        """Stop the queue and cancel the running task (if any). Used
        at app shutdown so we don't leak the runner thread."""
        self._paused = True
        if self._driver is not None:
            self._driver.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(30_000)  # generous: in-flight encode can be slow
        self._driver = None
        self._thread = None
        self._worker = None
        self._current_task_id = None

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

    # ---- Scheduling ----

    def _schedule_next(self) -> None:
        if self._paused or self.is_running:
            return
        next_task = self._pop_pending()
        if next_task is None:
            self.queueIdle.emit()
            return
        self._spawn_runner(next_task)

    def _pop_pending(self) -> BatchTask | None:
        """Return the next Pending task in the store, or None."""
        for task in self._store.list():
            if task.status is BatchTaskStatus.PENDING:
                return task
        return None

    def _spawn_runner(self, task: BatchTask) -> None:
        self._current_task_id = task.id
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
        worker.completed.connect(
            self._on_completed, type=Qt.ConnectionType.QueuedConnection
        )
        thread.started.connect(worker.run)
        self._thread = thread
        self._worker = worker
        self.taskStarted.emit(task.id)
        thread.start()

    # ---- Worker callbacks (GUI thread) ----

    def _on_progress(self, task_id: str, completed: int, total: int) -> None:
        # Persist progress so the GUI reload sees an up-to-date number
        # even if the app restarts mid-task. Cheap (JSON write) — only
        # fires on chunked frame completions, not per-frame.
        if self._store.exists(task_id):
            task = self._store.load(task_id)
            task.last_completed_frame = completed - 1
            task.total_frames = total
            self._store.save(task)
        self.taskProgress.emit(task_id, completed, total)

    def _on_completed(self, task_id: str, status_value: str) -> None:
        # The driver mutated the task in place inside the worker
        # thread, but the in-memory instance we have is on THAT
        # thread's side of memory — re-load fresh from disk (the
        # driver's run() persists nothing; the queue saves below).
        # Actually the worker's BatchTask object IS shared — we passed
        # it by reference. So reading current state is just the
        # instance we kept in _spawn_runner... but we didn't. Re-load
        # from disk after writing on this side.
        terminal = BatchTaskStatus(status_value)
        # Persist the final state by reading the in-memory task off
        # the worker. The worker's `_task` reference IS the same
        # object the driver mutated; safe to read here on GUI thread
        # because the worker thread has exited (completed signal is
        # its last emission before run() returns).
        if self._worker is not None:
            task = self._worker._task  # noqa: SLF001
            self._store.save(task)
            if terminal is BatchTaskStatus.FAILED:
                self.taskFailed.emit(task_id, task.error_message or "")
        self.taskCompleted.emit(task_id)
        self._teardown_runner()
        # Auto-pick next pending task unless we're paused or the
        # terminal was a Cancel (user intent: stop scheduling).
        if not self._paused and terminal is not BatchTaskStatus.CANCELLED:
            self._schedule_next()
        else:
            self.queueIdle.emit()

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
