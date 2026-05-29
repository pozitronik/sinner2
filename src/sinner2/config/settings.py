import json
import os
from pathlib import Path

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.config.base import SinnerBaseModel
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode


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
    default), which gives forward-compat for older settings.json files. Fields
    are Optional so absent fields fall back to widget defaults rather than
    forcing every user's first run to ship with stale stored values.
    """

    window_geometry_hex: str | None = None
    source_path: str | None = None
    target_path: str | None = None
    worker_count: int | None = None
    strategy_name: str | None = None
    enhancer_enabled: bool | None = None
    swapper_enabled: bool | None = None
    swapper_detection_interval: int | None = None
    swapper_many_faces: bool | None = None
    swapper_target_sex: str | None = None  # "M"/"F"/"B"/"I"
    enhancer_upscale: int | None = None
    enhancer_only_center_face: bool | None = None
    playback_mode: PlaybackMode | None = None
    cache_mode: CacheMode | None = None
    image_format: ImageFormat | None = None
    image_quality: int | None = None
    memory_cache_mb: int | None = None
    write_workers: int | None = None
    write_queue_size: int | None = None
    cache_root_path: str | None = None
    cache_size_cap_mb: int | None = None
    audio_backend: AudioBackendName | None = None
    audio_volume: int | None = None  # 0-100
    audio_muted: bool | None = None
    video_backend: VideoBackend | None = None
    reader_pool_size: int | None = None
    synced_max_lag_frames: int | None = None
    side_panel_visible: bool | None = None
    metrics_overlay_visible: bool | None = None
    onnx_providers: list[str] | None = None
    recent_sources: list[str] | None = None
    recent_targets: list[str] | None = None
    library_sources: list[str] | None = None
    library_targets: list[str] | None = None
    top_splitter_state_hex: str | None = None
    library_display_dim: int | None = None
    window_stays_on_top: bool | None = None
    display_rotation: int | None = None  # 0 / 90 / 180 / 270
    batch_store_path: str | None = None  # default <install>/batch
    batch_global_output_path: str | None = None  # default: next to target
    batch_default_format: str | None = None  # "video" / "frames"
    batch_default_cleanup: str | None = None  # "keep" / "auto" / "drop_at_end"


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
