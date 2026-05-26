from pathlib import Path

import pytest

from sinner2.config import settings


class TestSettingsPath:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        custom = tmp_path / "custom-settings.json"
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(custom))
        assert settings.settings_path() == custom

    def test_default_is_install_relative(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SINNER2_SETTINGS_PATH", raising=False)
        path = settings.settings_path()
        assert path.name == "settings.json"


class TestLoadAndSave:
    def test_load_returns_defaults_when_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(tmp_path / "absent.json"))
        s = settings.load()
        assert s.window_geometry_hex is None

    def test_roundtrip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(tmp_path / "settings.json"))
        original = settings.Settings(window_geometry_hex="deadbeef")
        settings.save(original)
        loaded = settings.load()
        assert loaded.window_geometry_hex == "deadbeef"

    def test_load_returns_defaults_on_corrupt_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        path = tmp_path / "settings.json"
        path.write_text("not valid json {", encoding="utf-8")
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(path))
        s = settings.load()
        assert s.window_geometry_hex is None

    def test_save_creates_parent_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        nested = tmp_path / "a" / "b" / "settings.json"
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(nested))
        settings.save(settings.Settings(window_geometry_hex="aa"))
        assert nested.is_file()

    def test_unknown_fields_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        path = tmp_path / "settings.json"
        path.write_text(
            '{"window_geometry_hex": "ff", "future_field": "x"}',
            encoding="utf-8",
        )
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(path))
        s = settings.load()
        assert s.window_geometry_hex == "ff"
        assert not hasattr(s, "future_field")
