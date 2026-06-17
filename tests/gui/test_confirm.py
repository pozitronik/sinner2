"""Tests for the shared 'Don't ask again' confirm helper + SuppressionStore.

The dialog-showing paths patch QMessageBox.exec / QCheckBox.isChecked so no
real modal blocks the headless run (the same class-attr patching the existing
dialog tests rely on for QMessageBox.question)."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QCheckBox, QMessageBox

from sinner2.gui.confirm import (
    SuppressionStore,
    confirm,
    set_default_suppression_store,
)


@pytest.fixture(autouse=True)
def _clear_default_store():
    # Keep the process-wide default out of these tests (and shield them from a
    # main_window built elsewhere in the suite that installs its own store).
    set_default_suppression_store(None)
    yield
    set_default_suppression_store(None)


def _store(initial: dict | None = None):
    data = dict(initial or {})

    def save(m):
        data.clear()
        data.update(m)

    return data, SuppressionStore(load=lambda: dict(data), save=save)


class TestSuppressionStore:
    def test_remembered_is_none_when_absent(self):
        _, store = _store()
        assert store.remembered("x") is None

    def test_remember_then_recall(self):
        data, store = _store()
        store.remember("x", True)
        assert store.remembered("x") is True
        assert data == {"x": True}

    def test_remember_false_is_distinct_from_absent(self):
        _, store = _store()
        store.remember("x", False)
        assert store.remembered("x") is False  # present-but-False, not None


class TestConfirmShortCircuit:
    def test_remembered_answer_skips_dialog(self, qtbot, monkeypatch):
        _, store = _store({"x": True})
        monkeypatch.setattr(
            QMessageBox, "exec",
            lambda self: pytest.fail("dialog shown despite remembered answer"),
        )
        assert confirm(None, "x", "T", "msg", store=store) is True

    def test_remembered_false_is_ignored_and_shows(self, qtbot, monkeypatch):
        # A remembered (or legacy) NO must NOT suppress — the dialog shows again,
        # so "Don't ask again" + Cancel can't lock an action out for good.
        _, store = _store({"x": False})
        monkeypatch.setattr(
            QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes
        )
        assert confirm(None, "x", "T", "msg", store=store) is True

    def test_non_suppressible_ignores_remembered_and_shows(self, qtbot, monkeypatch):
        _, store = _store({"x": True})
        monkeypatch.setattr(
            QMessageBox, "exec", lambda self: QMessageBox.StandardButton.No
        )
        # A remembered Yes must NOT auto-confirm a non-suppressible prompt; it
        # shows and returns the live answer (No).
        assert confirm(None, "x", "T", "msg", store=store, suppressible=False) is False


class TestConfirmPersistsOnTick:
    def test_ticking_checkbox_persists_answer(self, qtbot, monkeypatch):
        data, store = _store()
        monkeypatch.setattr(
            QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes
        )
        monkeypatch.setattr(QCheckBox, "isChecked", lambda self: True)
        assert confirm(None, "x", "T", "msg", store=store) is True
        assert data == {"x": True}  # remembered for next time

    def test_unticked_does_not_persist(self, qtbot, monkeypatch):
        data, store = _store()
        monkeypatch.setattr(
            QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes
        )
        monkeypatch.setattr(QCheckBox, "isChecked", lambda self: False)
        assert confirm(None, "x", "T", "msg", store=store) is True
        assert data == {}  # not remembered

    def test_ticked_no_does_not_persist(self, qtbot, monkeypatch):
        # The reported trap: ticking "Don't ask again" then Cancelling must NOT
        # remember anything (otherwise the action is locked out permanently).
        data, store = _store()
        monkeypatch.setattr(
            QMessageBox, "exec", lambda self: QMessageBox.StandardButton.No
        )
        monkeypatch.setattr(QCheckBox, "isChecked", lambda self: True)
        assert confirm(None, "x", "T", "msg", store=store) is False
        assert data == {}  # ticked No is NOT persisted

    def test_uses_default_store_when_none_passed(self, qtbot, monkeypatch):
        _, store = _store({"x": True})
        set_default_suppression_store(store)
        monkeypatch.setattr(
            QMessageBox, "exec", lambda self: pytest.fail("dialog shown")
        )
        assert confirm(None, "x", "T", "msg") is True  # default store consulted

    def test_no_store_shows_plain_dialog(self, qtbot, monkeypatch):
        # With no store at all, confirm still works as a plain yes/no (no
        # checkbox, no persistence) — it must not crash.
        monkeypatch.setattr(
            QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes
        )
        assert confirm(None, "x", "T", "msg") is True
