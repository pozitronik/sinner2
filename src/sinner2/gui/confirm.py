"""Shared yes/no confirmation with an optional "Don't ask me again".

Every confirmation dialog routes through `confirm()`, so a user can permanently
suppress an individual prompt (keyed by a stable `dialog_id`) and have the
remembered answer reused. The suppression map is persisted in Settings; the
helper reaches it through a `SuppressionStore` (a load/save pair) injected by
the app owner so this module stays free of Settings/persistence wiring.

The owner installs one process-wide store via `set_default_suppression_store()`;
call sites then just call `confirm(parent, dialog_id, title, text)`. Pass
`suppressible=False` for prompts that must never be auto-confirmed (e.g.
deleting a REQUIRED model) — those show no checkbox and ignore any stored answer.
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QCheckBox, QMessageBox, QWidget


class SuppressionStore:
    """Loads/saves the per-dialog suppression map (dialog_id -> remembered
    answer). Backed by Settings through the injected load/save callables, so
    child widgets never need a Settings reference."""

    def __init__(
        self,
        load: Callable[[], dict[str, bool]],
        save: Callable[[dict[str, bool]], None],
    ) -> None:
        self._load = load
        self._save = save

    def remembered(self, dialog_id: str) -> bool | None:
        """The stored answer for this dialog, or None if not suppressed."""
        return self._load().get(dialog_id)

    def remember(self, dialog_id: str, answer: bool) -> None:
        updated = dict(self._load())
        updated[dialog_id] = answer
        self._save(updated)


_default_store: SuppressionStore | None = None


def set_default_suppression_store(store: SuppressionStore | None) -> None:
    """Install (or clear) the process-wide store used when `confirm()` is
    called without an explicit one. The app owner calls this once at startup."""
    global _default_store
    _default_store = store


def confirm(
    parent: QWidget | None,
    dialog_id: str,
    title: str,
    text: str,
    *,
    store: SuppressionStore | None = None,
    suppressible: bool = True,
    default_yes: bool = False,
) -> bool:
    """Ask a yes/no question; return True for Yes.

    When `suppressible` and a store has a remembered answer for `dialog_id`,
    return it without showing any UI. Otherwise show a Yes/No box carrying a
    "Don't ask me again" checkbox; if ticked, persist the chosen answer."""
    effective = store if store is not None else _default_store
    if suppressible and effective is not None:
        remembered = effective.remembered(dialog_id)
        if remembered is not None:
            return remembered

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    box.setDefaultButton(
        QMessageBox.StandardButton.Yes
        if default_yes
        else QMessageBox.StandardButton.No
    )
    checkbox: QCheckBox | None = None
    if suppressible and effective is not None:
        checkbox = QCheckBox("Don't ask me again")
        box.setCheckBox(checkbox)

    answer = box.exec() == QMessageBox.StandardButton.Yes
    if checkbox is not None and checkbox.isChecked() and effective is not None:
        effective.remember(dialog_id, answer)
    return answer
