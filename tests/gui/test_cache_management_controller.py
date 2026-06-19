"""Characterization + unit tests for CacheManagementController.

The cache-storage actions (browse/reset root, invalidate, rerender, clear-all,
size cap) had ZERO coverage before this extraction — these pin their behaviour so
the move out of SinnerMainWindow is provably faithful.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from sinner2.config.settings import Settings
from sinner2.gui import cache_management_controller as cmc
from sinner2.gui.cache_management_controller import CacheManagementController


def _make(settings=None, executor=object(), entry_paths=None, session_dir=None):
    controller = MagicMock()
    controller.cache_root.return_value = Path("/cache/root")
    controller.executor.return_value = executor
    controller.session_cache_dir.return_value = session_dir or Path("/cache/session")
    controller.cache_manager.return_value.entry_paths.return_value = (
        entry_paths if entry_paths is not None else []
    )
    processors = MagicMock()
    updates: list[dict] = []
    ctl = CacheManagementController(
        window=MagicMock(),
        controller=controller,
        processors=processors,
        update_settings=lambda **k: updates.append(k),
        settings_getter=lambda: settings or Settings(),
    )
    return ctl, controller, processors, updates


class TestRestoreState:
    def test_applies_root_and_size_cap_from_settings(self):
        s = Settings(cache_root_path="/my/cache", cache_size_cap_mb=10)
        ctl, controller, processors, _ = _make(settings=s)
        ctl.restore_state()
        controller.set_cache_root.assert_called_once_with(Path("/my/cache"))
        controller.set_cache_size_cap_bytes.assert_called_once_with(10 * 1024 * 1024)
        processors.set_cache_size_cap_bytes.assert_called_once_with(10 * 1024 * 1024)

    def test_no_root_path_skips_set_root_and_zero_cap(self):
        ctl, controller, _p, _ = _make(settings=Settings())
        ctl.restore_state()
        controller.set_cache_root.assert_not_called()
        controller.set_cache_size_cap_bytes.assert_called_once_with(0)


class TestBrowseReset:
    def test_browse_sets_root_and_persists(self, monkeypatch):
        monkeypatch.setattr(cmc.QFileDialog, "getExistingDirectory", lambda *a: "/chosen")
        ctl, controller, _p, updates = _make()
        ctl.on_browse_root()
        controller.set_cache_root.assert_called_once_with(Path("/chosen"))
        assert updates == [{"cache_root_path": str(Path("/cache/root"))}]

    def test_browse_cancelled_is_noop(self, monkeypatch):
        monkeypatch.setattr(cmc.QFileDialog, "getExistingDirectory", lambda *a: "")
        ctl, controller, _p, updates = _make()
        ctl.on_browse_root()
        controller.set_cache_root.assert_not_called()
        assert updates == []

    def test_reset_clears_root_and_persists_none(self):
        ctl, controller, _p, updates = _make()
        ctl.on_reset_root()
        controller.set_cache_root.assert_called_once_with(None)
        assert updates == [{"cache_root_path": None}]


class TestInvalidate:
    def test_noop_when_no_executor(self):
        ctl, controller, _p, _ = _make(executor=None)
        ctl.on_invalidate_session()
        controller.invalidate_current_session.assert_not_called()

    def test_invalidates_when_confirmed(self, monkeypatch):
        monkeypatch.setattr(cmc, "confirm", lambda *a: True)
        ctl, controller, _p, _ = _make(executor=object())
        ctl.on_invalidate_session()
        controller.invalidate_current_session.assert_called_once()

    def test_skips_when_declined(self, monkeypatch):
        monkeypatch.setattr(cmc, "confirm", lambda *a: False)
        ctl, controller, _p, _ = _make(executor=object())
        ctl.on_invalidate_session()
        controller.invalidate_current_session.assert_not_called()


class TestRerender:
    def test_rerenders_without_confirmation(self):
        ctl, controller, _p, _ = _make()
        ctl.on_rerender_from_current()
        controller.rerender_from_current.assert_called_once()


class TestClearAll:
    def test_clears_when_confirmed_sparing_session(self, monkeypatch):
        monkeypatch.setattr(cmc, "confirm", lambda *a: True)
        session = Path("/cache/session")
        ctl, controller, _p, _ = _make(
            entry_paths=[session, Path("/cache/other")], session_dir=session
        )
        ctl.on_clear_all()
        controller.clear_all_caches.assert_called_once()

    def test_info_and_no_clear_when_only_session_present(self, monkeypatch):
        info = MagicMock()
        monkeypatch.setattr(cmc.QMessageBox, "information", info)
        session = Path("/cache/session")
        ctl, controller, _p, _ = _make(entry_paths=[session], session_dir=session)
        ctl.on_clear_all()
        info.assert_called_once()
        controller.clear_all_caches.assert_not_called()

    def test_skips_when_declined(self, monkeypatch):
        monkeypatch.setattr(cmc, "confirm", lambda *a: False)
        session = Path("/cache/session")
        ctl, controller, _p, _ = _make(
            entry_paths=[session, Path("/cache/x")], session_dir=session
        )
        ctl.on_clear_all()
        controller.clear_all_caches.assert_not_called()


class TestSizeCap:
    def test_sets_bytes_and_persists_mb(self):
        ctl, controller, _p, updates = _make()
        ctl.on_size_cap_changed(20 * 1024 * 1024)
        controller.set_cache_size_cap_bytes.assert_called_once_with(20 * 1024 * 1024)
        assert updates == [{"cache_size_cap_mb": 20}]

    def test_zero_cap_persists_none(self):
        ctl, _controller, _p, updates = _make()
        ctl.on_size_cap_changed(0)
        assert updates == [{"cache_size_cap_mb": None}]
