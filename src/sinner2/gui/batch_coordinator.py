"""Batch run-state machine for the main window.

Owns the queue-signal-driven lifecycle of a batch render: lock the live editing
surface while a task runs, repurpose the position bar to track the render's
progress (in original-timeline coordinates), restore the live session on idle,
and surface failures in one consolidated dialog at the end.

The ``_batch_active`` flag stays a window attribute (it's read across the window
and seeded by tests) — the coordinator flips it through the injected
``set_active`` / ``is_active`` callbacks so it remains a single source of truth.
Dialog-opening actions (add / settings / edit task) stay on the window.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sinner2.batch.task import BatchProgress
    from sinner2.batch.task_store import BatchTaskStore
    from sinner2.types import Frame


class BatchCoordinator:
    def __init__(
        self,
        *,
        controller: Any,
        transport: Any,
        status_bar: Any,
        display: Any,
        batch_store: "BatchTaskStore",
        is_active: Callable[[], bool],
        set_active: Callable[[bool], None],
        set_editing_locked: Callable[[bool], None],
        show_error: Callable[[str], None],
    ) -> None:
        self._controller = controller
        self._transport = transport
        self._status_bar = status_bar
        self._display = display
        self._batch_store = batch_store
        self._is_active = is_active
        self._set_active = set_active
        self._set_editing_locked = set_editing_locked
        self._show_error = show_error
        self._failures: list[tuple[str, str]] = []
        self._slider_total = -1  # re-armed per task; -1 forces a re-range

    def on_task_started(self, _task_id: str) -> None:
        # DaVinci-style: while a batch renders, pause the live executor and lock
        # the ENTIRE editing surface. Two simultaneous ORT sessions contend for
        # the GPU (OOM risk), and the display must act purely as a render preview.
        if not self._is_active():
            self._failures = []  # first task of a fresh run
        self._set_active(True)
        self._slider_total = -1  # re-arm the position bar for this task
        if self._controller.executor() is not None:
            self._controller.executor().pause()
        self._set_editing_locked(True)
        self._status_bar.show_message("Batch running — editing locked", 5000)

    def on_progress(self, _task_id: str, progress: "BatchProgress") -> None:
        # The editing surface is locked, so repurpose the position bar to track
        # the batch in ORIGINAL-timeline coordinates (full source length + the
        # source frame mapped through the section plan), so on a trimmed task the
        # knob sits INSIDE the section band for every stage. Older callers without
        # the source fields (source_total == 0) fall back to stage-relative.
        if progress.source_total > 0:
            total, frame = progress.source_total, progress.source_frame
        else:
            total, frame = progress.stage_total, progress.stage_completed - 1
        # set_frame_count snaps the value to 0, so only re-range when it changes.
        if total != self._slider_total:
            self._slider_total = total
            self._transport.set_frame_count(total)
        self._transport.set_current_frame(max(0, frame))

    def on_queue_idle(self) -> None:
        self._set_active(False)
        self._slider_total = -1
        # Restore the position bar to the live session we hijacked it from.
        self._controller.resync_transport()
        self._set_editing_locked(False)
        self._status_bar.show_message("Batch queue idle — editing unlocked", 3000)
        # Surface the real failure reason(s) prominently now the run is done —
        # one dialog for the whole run (a continue-on-error run can fail many).
        self.report_failures()

    def on_task_failed(self, task_id: str, message: str) -> None:
        # Collect for the consolidated dialog at queue-idle (avoids modal spam
        # mid-run); a short status note flags it immediately.
        label = self.task_label(task_id)
        self._failures.append((label, message or "unknown error"))
        self._status_bar.show_message(f"Batch task failed: {label}", 8000)

    def task_label(self, task_id: str) -> str:
        """A readable 'source → target' label for a task id, or the id if the
        task can't be loaded."""
        try:
            if self._batch_store.exists(task_id):
                task = self._batch_store.load(task_id)
                return f"{task.source_path.name} → {task.target_path.name}"
        except Exception:
            pass
        return task_id

    def report_failures(self) -> None:
        """Show one error dialog summarising every task that failed this run,
        then clear the list. No-op when nothing failed."""
        failures = self._failures
        self._failures = []
        if not failures:
            return
        if len(failures) == 1:
            label, msg = failures[0]
            self._show_error(f"Batch task failed — {label}:\n\n{msg}")
            return
        lines = "\n\n".join(f"• {label}:\n  {msg}" for label, msg in failures)
        self._show_error(f"{len(failures)} batch tasks failed:\n\n{lines}")

    def on_preview(self, _task_id: str, frame: "Frame") -> None:
        # Show what the batch is producing on the (idle) preview surface.
        self._display.show_frame(frame)
