"""Cache-management actions for the main window: the cache-storage controls
(browse / reset root, invalidate session, rerender, clear-all, size cap) plus the
startup restore of the persisted cache root + size cap.

Action handlers wired to QProcessorControls' cache signals; each delegates to the
PlayerController's cache API, with dialogs parented to the window. The threaded
size/count *stats* walk stays on the window — it is bound to the close lifecycle
and a queued GUI-thread callback.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from sinner2.gui.confirm import confirm

if TYPE_CHECKING:
    from sinner2.config.settings import Settings
    from sinner2.gui.player_controller import PlayerController
    from sinner2.gui.widgets.processor_controls import QProcessorControls


class CacheManagementController:
    def __init__(
        self,
        *,
        window: QWidget,
        controller: "PlayerController",
        processors: "QProcessorControls",
        update_settings: Callable[..., None],
        settings_getter: "Callable[[], Settings]",
    ) -> None:
        self._window = window
        self._controller = controller
        self._processors = processors
        self._update_settings = update_settings
        self._settings_getter = settings_getter

    def restore_state(self) -> None:
        settings = self._settings_getter()
        # Cache root: settings → controller → widget display.
        if settings.cache_root_path:
            self._controller.set_cache_root(Path(settings.cache_root_path))
        self._processors.set_cache_root_text(self._controller.cache_root())
        # Size cap: settings → controller (state) + widget (display).
        cap_mb = settings.cache_size_cap_mb or 0
        cap_bytes = cap_mb * 1024 * 1024 if cap_mb > 0 else 0
        self._controller.set_cache_size_cap_bytes(cap_bytes)
        self._processors.set_cache_size_cap_bytes(cap_bytes)

    def on_browse_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self._window,
            "Choose cache root directory",
            str(self._controller.cache_root()),
        )
        if not chosen:
            return
        self._controller.set_cache_root(Path(chosen))
        self._processors.set_cache_root_text(self._controller.cache_root())
        self._update_settings(cache_root_path=str(self._controller.cache_root()))

    def on_reset_root(self) -> None:
        self._controller.set_cache_root(None)
        self._processors.set_cache_root_text(self._controller.cache_root())
        self._update_settings(cache_root_path=None)

    def on_invalidate_session(self) -> None:
        if self._controller.executor() is None:
            return
        if not confirm(
            self._window,
            "invalidate_session",
            "Invalidate current session",
            "Drop all cached frames for this session and reprocess from scratch?",
        ):
            return
        self._controller.invalidate_current_session()

    def on_rerender_from_current(self) -> None:
        # No confirmation: it only reprocesses from the playhead forward and is
        # the natural "apply my param change retroactively" gesture.
        self._controller.rerender_from_current()

    def on_clear_all(self) -> None:
        manager = self._controller.cache_manager()
        protected = self._controller.session_cache_dir()
        # Count via entry_paths() (no per-file size walk) so the dialog opens
        # instantly even on a huge cache — the size walk is what hung the app.
        deletable = [p for p in manager.entry_paths() if p != protected]
        if not deletable:
            QMessageBox.information(
                self._window,
                "Clear all caches",
                "Nothing to delete — only the current session's cache is present.",
            )
            return
        if not confirm(
            self._window,
            "clear_all_caches",
            "Clear all caches",
            f"Delete {len(deletable)} cache entries?\n"
            "The currently active session will be spared.",
        ):
            return
        self._controller.clear_all_caches()

    def on_size_cap_changed(self, bytes_cap: int) -> None:
        self._controller.set_cache_size_cap_bytes(bytes_cap)
        cap_mb = bytes_cap // (1024 * 1024) if bytes_cap > 0 else 0
        self._update_settings(cache_size_cap_mb=cap_mb or None)
