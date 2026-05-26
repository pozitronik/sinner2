import json
import os
from pathlib import Path

from sinner2.config.base import SinnerBaseModel


def _install_dir() -> Path:
    """Project install root.

    For editable installs this resolves to the repo's top level — same logic
    as `model_cache._project_models_dir`. For wheel installs it lands inside
    site-packages, which is wrong for user data; set SINNER2_SETTINGS_PATH
    in that case.
    """
    return Path(__file__).resolve().parents[3]


class Settings(SinnerBaseModel):
    """User preferences persisted next to the install dir.

    Schema is intentionally small for v1 — extends as features need state to
    persist across launches. Unknown fields are silently ignored (SinnerBaseModel
    default), which gives forward-compat for older settings.json files.
    """

    window_geometry_hex: str | None = None


def settings_path() -> Path:
    """Resolve the settings.json location.

    SINNER2_SETTINGS_PATH env var takes precedence; otherwise defaults to
    `<install>/settings.json`.
    """
    env = os.environ.get("SINNER2_SETTINGS_PATH")
    if env:
        return Path(env)
    return _install_dir() / "settings.json"


def load() -> Settings:
    path = settings_path()
    if not path.is_file():
        return Settings()
    try:
        return Settings.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return Settings()


def save(settings: Settings) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
