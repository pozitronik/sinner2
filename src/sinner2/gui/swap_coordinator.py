"""Async source/target swap orchestration for the realtime player.

Owns the *when/how* of a live session swap: coalescing (latest-wins) of requests
arriving mid-swap, spawning the worker thread, the queued worker→GUI completion
hop (a Qt Signal — hence this is a QObject), and the swapping/pending state.

The actual *work* — building the new session, adopting it via the live
executor's `reconfigure_from`, and re-pointing GUI refs — is all controller-
coupled and stays on PlayerController, injected here as callbacks. Keeping this
split means the bug-sensitive triad (stash-before-emit / clear-after-adopt /
shutdown-drain) still lives where the resources do; this class only sequences it.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from sinner2.gui.session_builder import _SessionBundle


def _spawn_daemon(fn: Callable[[], None]) -> threading.Thread:
    """Default swap runner: fire the job on a daemon thread. Swapped out in
    tests for an inline runner so the async flow executes deterministically."""
    thread = threading.Thread(target=fn, name="sinner2-session-swap", daemon=True)
    thread.start()
    return thread


@dataclass
class _SwapOutcome:
    """Result of a background session swap, marshaled back to the GUI thread.
    Exactly one of bundle/error is set."""

    bundle: _SessionBundle | None = None
    error: str | None = None


class SwapCoordinator(QObject):
    """Sequences the async swap; the work is injected as callbacks.

    - ``run_job(src, tgt) -> _SwapOutcome`` runs on the WORKER thread (build +
      reconfigure + shut down displaced resources + stash).
    - ``on_complete(outcome)`` runs on the GUI thread (adopt / error + audio).
    - ``on_begin()`` runs on the GUI thread at swap start (pause audio).
    - ``on_switching(bool)`` mirrors the sessionSwitching signal.
    """

    _ready = Signal(object)  # _SwapOutcome — marshals worker → GUI thread

    def __init__(
        self,
        *,
        run_job: Callable[[Path, Path], _SwapOutcome],
        on_complete: Callable[[_SwapOutcome], None],
        on_begin: Callable[[], None],
        on_switching: Callable[[bool], None],
        spawn: Callable[[Callable[[], None]], threading.Thread | None] = _spawn_daemon,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_job = run_job
        self._on_complete = on_complete
        self._on_begin = on_begin
        self._on_switching = on_switching
        self._spawn = spawn
        self._swapping = False
        self._pending: tuple[Path, Path] | None = None
        self._thread: threading.Thread | None = None
        self._ready.connect(self._on_ready)

    @property
    def swapping(self) -> bool:
        return self._swapping

    @property
    def spawn(self) -> Callable[[Callable[[], None]], threading.Thread | None]:
        """The thread-spawn indirection — tests swap in an inline runner so the
        async flow executes deterministically."""
        return self._spawn

    @spawn.setter
    def spawn(
        self, fn: Callable[[Callable[[], None]], threading.Thread | None]
    ) -> None:
        self._spawn = fn

    def request(self, source_path: Path, target_path: Path) -> None:
        """Begin a swap, or coalesce (latest wins) when one is already running —
        callers update their restore/path state before calling, so the coalesced
        run targets the latest selection."""
        if self._swapping:
            self._pending = (source_path, target_path)
            return
        self._begin(source_path, target_path)

    def _begin(self, source_path: Path, target_path: Path) -> None:
        self._swapping = True
        self._on_switching(True)
        # Pause audio for the swap window; on_complete restores it once the new
        # world is live. The executor is NOT torn down — it keeps running the old
        # frames until the in-place swap lands, so the display never blanks.
        self._on_begin()
        self._thread = self._spawn(
            lambda: self._ready.emit(self._run_job(source_path, target_path))
        )

    def _on_ready(self, outcome: _SwapOutcome) -> None:
        """GUI thread: a background swap finished. Reset state, run the
        controller's completion (adopt / error), then any coalesced request."""
        self._swapping = False
        self._thread = None
        self._on_complete(outcome)
        self._on_switching(False)
        if self._pending is not None:
            source_path, target_path = self._pending
            self._pending = None
            self._begin(source_path, target_path)

    def cancel_pending_and_join(self, timeout: float) -> None:
        """Shutdown helper: drop any coalesced swap + wait for an in-flight
        worker so teardown doesn't race a session still being built/stopped."""
        self._pending = None
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
