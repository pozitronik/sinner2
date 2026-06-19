"""Unit tests for SettingsBinder — Settings ownership + the assign-before-save
write funnel (the bug-sensitive rule extracted out of SinnerMainWindow)."""
from __future__ import annotations

from sinner2.config.settings import Settings
from sinner2.gui import settings_binder
from sinner2.gui.settings_binder import SettingsBinder


class TestUpdate:
    def test_applies_field_and_persists(self, monkeypatch):
        saved: list[Settings] = []
        monkeypatch.setattr(settings_binder.user_settings, "save", saved.append)
        b = SettingsBinder(Settings())
        b.update(source_path="/a.png")
        assert b.settings.source_path == "/a.png"
        assert len(saved) == 1 and saved[0].source_path == "/a.png"

    def test_assigns_before_save_so_failed_write_keeps_memory(self, monkeypatch):
        # The invariant: a transient OSError must NOT leave the in-memory copy
        # stale (a stale base would corrupt every later model_copy).
        def boom(_s: Settings) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(settings_binder.user_settings, "save", boom)
        b = SettingsBinder(Settings())
        b.update(source_path="/new.png")  # must not raise
        assert b.settings.source_path == "/new.png"

    def test_update_is_an_immutable_copy_not_an_in_place_mutation(self, monkeypatch):
        monkeypatch.setattr(settings_binder.user_settings, "save", lambda _s: None)
        b = SettingsBinder(Settings())
        first = b.settings
        b.update(audio_volume=50)
        assert b.settings is not first
        assert first.audio_volume is None       # original untouched
        assert b.settings.audio_volume == 50


class TestSetSettings:
    def test_swaps_object_without_persisting(self, monkeypatch):
        saved: list[Settings] = []
        monkeypatch.setattr(settings_binder.user_settings, "save", saved.append)
        b = SettingsBinder(Settings())
        replacement = Settings(source_path="/replaced.png")
        b.set_settings(replacement)
        assert b.settings is replacement
        assert saved == []  # set_settings does NOT write to disk
