import json
import logging
import os
from pathlib import Path

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.config.base import SinnerBaseModel
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode

_log = logging.getLogger(__name__)


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
    realtime_workers: int | None = None
    strategy_name: str | None = None
    enhancer_enabled: bool | None = None
    swapper_enabled: bool | None = None
    swapper_model: str | None = None  # inswapper_128 | reswapper_128 | ghost_* | simswap_256 | uniface_256
    swapper_detection_interval: int | None = None
    swapper_many_faces: bool | None = None
    swapper_target_sex: str | None = None  # "M"/"F"/"B"/"I"
    enhancer_model: str | None = None  # gfpgan | codeformer
    enhancer_upscale: int | None = None
    enhancer_only_center_face: bool | None = None
    enhancer_codeformer_fidelity: float | None = None  # CodeFormer w knob
    enhancer_fp16: bool | None = None  # GFPGAN half precision (CUDA only)
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
    video_backend: VideoBackend | None = None
    reader_pool_size: int | None = None
    processing_scale: float | None = None  # 0 < s <= 1; downscale frames before processing
    synced_max_lag_frames: int | None = None
    side_panel_visible: bool | None = None
    metrics_overlay_visible: bool | None = None
    face_overlay_visible: bool | None = None  # face-detection debug overlay
    face_comparison_visible: bool | None = None  # orig/swapped comparison thumbs
    # Rotation compensation (swapper, experimental)
    swapper_rotation_compensation: bool | None = None
    swapper_rotation_threshold_deg: int | None = None
    swapper_rotation_redetect: bool | None = None
    swapper_rotation_angle_source: str | None = None
    swapper_occlusion_mask: bool | None = None
    swapper_occlusion_parser: str | None = None  # bisenet | parsenet
    swapper_providers: list[str] | None = None  # realtime ONNX EPs (swapper + analyser)
    enhancer_device: str | None = None  # realtime torch device for GFPGAN
    # Upscaler (Real-ESRGAN) — whole-frame super-resolution
    upscaler_enabled: bool | None = None
    upscaler_model: str | None = None  # general-x4v3 | x4plus | x2plus
    upscaler_tile: int | None = None
    upscaler_fp16: bool | None = None
    upscaler_device: str | None = None
    recent_sources: list[str] | None = None
    recent_targets: list[str] | None = None
    library_sources: list[str] | None = None
    library_targets: list[str] | None = None
    top_splitter_state_hex: str | None = None
    library_display_dim: int | None = None  # legacy shared zoom (fallback)
    # Configurable accepted file extensions (no UI — edit the file). None = the
    # comprehensive defaults in config.media_extensions.
    library_image_extensions: list[str] | None = None
    library_video_extensions: list[str] | None = None
    # Per-panel zoom + sort (source vs target kept independent).
    library_sources_display_dim: int | None = None
    library_targets_display_dim: int | None = None
    library_sources_sort_field: str | None = None
    library_sources_sort_order: str | None = None  # asc | desc
    library_targets_sort_field: str | None = None
    library_targets_sort_order: str | None = None
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
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        # The file exists but is unreadable/corrupt (e.g. truncated by a crash
        # or power loss mid-write). Preserve it as .bak instead of letting the
        # next save() silently overwrite it with defaults — that would
        # permanently destroy every persisted preference. Then start fresh.
        _log.warning(
            "settings file unreadable (%s); backing up to %s.bak", exc, path.name
        )
        try:
            os.replace(path, path.with_name(path.name + ".bak"))
        except OSError:
            pass  # best-effort; never block startup on the backup
        return Settings()


def save(settings: Settings) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a crash/power-loss mid-write must never truncate the real
    # settings file. Write a sibling temp then os.replace() onto the target —
    # an atomic rename on the same filesystem on both POSIX and Windows.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
