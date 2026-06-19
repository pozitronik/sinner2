"""Owns the persisted Settings object + the assign-before-save write funnel.

Centralizes the one bug-sensitive rule: assign the in-memory copy BEFORE the disk
write, so a transient OSError (full / read-only disk) can't leave the in-memory
settings stale while the UI shows the new value — that stale base would silently
corrupt every later model_copy. The window delegates `settings`/`update()` here
and keeps a `_settings` property alias backed by this binder.
"""
from __future__ import annotations

import logging

from sinner2.config import settings as user_settings
from sinner2.config.settings import Settings

_log = logging.getLogger(__name__)


class SettingsBinder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        return self._settings

    def set_settings(self, settings: Settings) -> None:
        """Swap the whole settings object WITHOUT writing to disk — e.g. a
        project-file restore that saves separately, or a test seeding state."""
        self._settings = settings

    def update(self, **fields: object) -> None:
        """Apply field changes in-memory FIRST, then persist. Assign before save
        so a failed write can't leave the in-memory copy stale (a stale base
        would corrupt every later model_copy). Never crash the GUI on a write."""
        updated = self._settings.model_copy(update=fields)
        self._settings = updated
        try:
            user_settings.save(updated)
        except Exception as exc:  # noqa: BLE001 - never crash the GUI on a settings write
            _log.warning("failed to persist settings: %s", exc)
