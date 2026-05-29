from pathlib import Path

import pytest

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.config import settings
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode


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


class TestExtendedFieldsRoundtrip:
    """Each persisted processor/execution field must survive save → load.

    These are the values main_window persists on every configChanged and
    restores via QProcessorControls.apply_restored_settings on startup;
    if any one breaks silently, the user sees their settings reset
    randomly on next launch.
    """

    @pytest.fixture(autouse=True)
    def _isolate_settings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(tmp_path / "settings.json"))

    def test_worker_count(self):
        settings.save(settings.Settings(worker_count=7))
        assert settings.load().worker_count == 7

    def test_strategy_name(self):
        settings.save(settings.Settings(strategy_name="SyncedStrategy"))
        assert settings.load().strategy_name == "SyncedStrategy"

    @pytest.mark.parametrize("value", [True, False])
    def test_enhancer_enabled(self, value: bool):
        settings.save(settings.Settings(enhancer_enabled=value))
        assert settings.load().enhancer_enabled is value

    def test_swapper_detection_interval(self):
        settings.save(settings.Settings(swapper_detection_interval=7))
        assert settings.load().swapper_detection_interval == 7

    @pytest.mark.parametrize("value", [True, False])
    def test_swapper_many_faces(self, value: bool):
        settings.save(settings.Settings(swapper_many_faces=value))
        assert settings.load().swapper_many_faces is value

    @pytest.mark.parametrize("token", ["M", "F", "B", "I"])
    def test_swapper_target_sex(self, token: str):
        settings.save(settings.Settings(swapper_target_sex=token))
        assert settings.load().swapper_target_sex == token

    def test_enhancer_upscale(self):
        settings.save(settings.Settings(enhancer_upscale=4))
        assert settings.load().enhancer_upscale == 4

    @pytest.mark.parametrize("value", [True, False])
    def test_enhancer_only_center_face(self, value: bool):
        settings.save(settings.Settings(enhancer_only_center_face=value))
        assert settings.load().enhancer_only_center_face is value

    @pytest.mark.parametrize("mode", list(PlaybackMode))
    def test_playback_mode(self, mode: PlaybackMode):
        settings.save(settings.Settings(playback_mode=mode))
        assert settings.load().playback_mode is mode

    @pytest.mark.parametrize("mode", list(CacheMode))
    def test_cache_mode(self, mode: CacheMode):
        settings.save(settings.Settings(cache_mode=mode))
        assert settings.load().cache_mode is mode

    @pytest.mark.parametrize("fmt", list(ImageFormat))
    def test_image_format(self, fmt: ImageFormat):
        settings.save(settings.Settings(image_format=fmt))
        assert settings.load().image_format is fmt

    def test_image_quality(self):
        settings.save(settings.Settings(image_quality=72))
        assert settings.load().image_quality == 72

    def test_memory_cache_mb(self):
        settings.save(settings.Settings(memory_cache_mb=512))
        assert settings.load().memory_cache_mb == 512

    def test_write_workers(self):
        settings.save(settings.Settings(write_workers=6))
        assert settings.load().write_workers == 6

    def test_write_queue_size(self):
        settings.save(settings.Settings(write_queue_size=24))
        assert settings.load().write_queue_size == 24

    def test_cache_root_path(self):
        settings.save(settings.Settings(cache_root_path="/somewhere/else"))
        assert settings.load().cache_root_path == "/somewhere/else"

    def test_cache_size_cap_mb(self):
        settings.save(settings.Settings(cache_size_cap_mb=4096))
        assert settings.load().cache_size_cap_mb == 4096

    @pytest.mark.parametrize("name", list(AudioBackendName))
    def test_audio_backend(self, name: AudioBackendName):
        settings.save(settings.Settings(audio_backend=name))
        assert settings.load().audio_backend is name

    def test_audio_volume(self):
        settings.save(settings.Settings(audio_volume=42))
        assert settings.load().audio_volume == 42

    @pytest.mark.parametrize("value", [True, False])
    def test_audio_muted(self, value: bool):
        settings.save(settings.Settings(audio_muted=value))
        assert settings.load().audio_muted is value

    @pytest.mark.parametrize("backend", list(VideoBackend))
    def test_video_backend(self, backend: VideoBackend):
        settings.save(settings.Settings(video_backend=backend))
        assert settings.load().video_backend is backend

    def test_reader_pool_size(self):
        settings.save(settings.Settings(reader_pool_size=8))
        assert settings.load().reader_pool_size == 8

    def test_synced_max_lag_frames(self):
        settings.save(settings.Settings(synced_max_lag_frames=120))
        assert settings.load().synced_max_lag_frames == 120

    @pytest.mark.parametrize("value", [True, False])
    def test_side_panel_visible(self, value: bool):
        settings.save(settings.Settings(side_panel_visible=value))
        assert settings.load().side_panel_visible is value

    @pytest.mark.parametrize("value", [True, False])
    def test_metrics_overlay_visible(self, value: bool):
        settings.save(settings.Settings(metrics_overlay_visible=value))
        assert settings.load().metrics_overlay_visible is value

    def test_onnx_providers(self):
        settings.save(settings.Settings(onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]))
        assert settings.load().onnx_providers == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_onnx_providers_empty_list(self):
        # Empty list is distinct from None (None = "never set", empty
        # list = "user explicitly unchecked everything"). Pydantic keeps
        # the distinction.
        settings.save(settings.Settings(onnx_providers=[]))
        assert settings.load().onnx_providers == []

    def test_recent_sources(self):
        settings.save(settings.Settings(recent_sources=["/a.jpg", "/b.png"]))
        assert settings.load().recent_sources == ["/a.jpg", "/b.png"]

    def test_recent_targets(self):
        settings.save(settings.Settings(recent_targets=["/x.mp4", "/y.mov"]))
        assert settings.load().recent_targets == ["/x.mp4", "/y.mov"]

    def test_library_sources(self):
        settings.save(
            settings.Settings(library_sources=["/lib/a.jpg", "/lib/b.png"])
        )
        assert settings.load().library_sources == ["/lib/a.jpg", "/lib/b.png"]

    def test_library_targets(self):
        settings.save(
            settings.Settings(library_targets=["/lib/x.mp4", "/lib/y.mov"])
        )
        assert settings.load().library_targets == ["/lib/x.mp4", "/lib/y.mov"]

    def test_top_splitter_state_hex(self):
        settings.save(settings.Settings(top_splitter_state_hex="deadbeef"))
        assert settings.load().top_splitter_state_hex == "deadbeef"

    def test_library_display_dim(self):
        settings.save(settings.Settings(library_display_dim=192))
        assert settings.load().library_display_dim == 192

    @pytest.mark.parametrize("value", [True, False])
    def test_window_stays_on_top(self, value: bool):
        settings.save(settings.Settings(window_stays_on_top=value))
        assert settings.load().window_stays_on_top is value

    @pytest.mark.parametrize("rot", [0, 90, 180, 270])
    def test_display_rotation(self, rot: int):
        settings.save(settings.Settings(display_rotation=rot))
        assert settings.load().display_rotation == rot

    def test_batch_store_path(self):
        settings.save(settings.Settings(batch_store_path="/my/batches"))
        assert settings.load().batch_store_path == "/my/batches"

    def test_batch_global_output_path(self):
        settings.save(settings.Settings(batch_global_output_path="/out"))
        assert settings.load().batch_global_output_path == "/out"

    @pytest.mark.parametrize("fmt", ["video", "frames"])
    def test_batch_default_format(self, fmt: str):
        settings.save(settings.Settings(batch_default_format=fmt))
        assert settings.load().batch_default_format == fmt

    @pytest.mark.parametrize("mode", ["keep", "auto", "drop_at_end"])
    def test_batch_default_cleanup(self, mode: str):
        settings.save(settings.Settings(batch_default_cleanup=mode))
        assert settings.load().batch_default_cleanup == mode

    @pytest.mark.parametrize("enabled", [True, False])
    def test_swapper_enabled_roundtrip(self, enabled: bool):
        settings.save(settings.Settings(swapper_enabled=enabled))
        assert settings.load().swapper_enabled is enabled

    def test_full_payload_roundtrip(self):
        original = settings.Settings(
            window_geometry_hex="abcd1234",
            source_path="/path/source.jpg",
            target_path="/path/target.mp4",
            worker_count=8,
            strategy_name="SyncedStrategy",
            enhancer_enabled=False,
            swapper_detection_interval=3,
            swapper_many_faces=False,
            swapper_target_sex="F",
            enhancer_upscale=4,
            enhancer_only_center_face=True,
            playback_mode=PlaybackMode.SOURCE,
            cache_mode=CacheMode.READ_ONLY,
            image_format=ImageFormat.PNG,
            image_quality=80,
            memory_cache_mb=512,
            write_workers=6,
            write_queue_size=24,
            cache_root_path="/custom/root",
            cache_size_cap_mb=4096,
            audio_backend=AudioBackendName.QT,
            audio_volume=80,
            audio_muted=True,
            video_backend=VideoBackend.CV2,
            reader_pool_size=8,
            synced_max_lag_frames=120,
            side_panel_visible=False,
            metrics_overlay_visible=True,
            onnx_providers=["CUDAExecutionProvider"],
            recent_sources=["/a.jpg"],
            recent_targets=["/x.mp4"],
            library_sources=["/lib/a.jpg"],
            library_targets=["/lib/x.mp4"],
            top_splitter_state_hex="cafebabe",
            library_display_dim=160,
            window_stays_on_top=True,
            display_rotation=90,
            batch_store_path="/b",
            batch_global_output_path="/o",
            batch_default_format="frames",
            batch_default_cleanup="auto",
        )
        settings.save(original)
        assert settings.load() == original

    def test_unspecified_fields_load_as_none(self):
        # Persisting one field must NOT inject defaults for the others; the
        # widget defaults take over on restore when a field is None.
        settings.save(settings.Settings(worker_count=2))
        loaded = settings.load()
        assert loaded.worker_count == 2
        assert loaded.strategy_name is None
        assert loaded.enhancer_enabled is None
        assert loaded.swapper_detection_interval is None
        assert loaded.swapper_many_faces is None
        assert loaded.swapper_target_sex is None
        assert loaded.enhancer_upscale is None
        assert loaded.enhancer_only_center_face is None
        assert loaded.playback_mode is None
        assert loaded.cache_mode is None
        assert loaded.image_format is None
        assert loaded.image_quality is None
        assert loaded.memory_cache_mb is None
        assert loaded.write_workers is None
        assert loaded.write_queue_size is None
        assert loaded.cache_root_path is None
        assert loaded.cache_size_cap_mb is None
        assert loaded.audio_backend is None
        assert loaded.audio_volume is None
        assert loaded.audio_muted is None
        assert loaded.video_backend is None
        assert loaded.reader_pool_size is None
        assert loaded.synced_max_lag_frames is None
        assert loaded.side_panel_visible is None
        assert loaded.metrics_overlay_visible is None
        assert loaded.onnx_providers is None
        assert loaded.recent_sources is None
        assert loaded.recent_targets is None
        assert loaded.library_sources is None
        assert loaded.library_targets is None
        assert loaded.top_splitter_state_hex is None
        assert loaded.library_display_dim is None
        assert loaded.window_stays_on_top is None
        assert loaded.display_rotation is None
        assert loaded.batch_store_path is None
        assert loaded.batch_global_output_path is None
        assert loaded.batch_default_format is None
        assert loaded.batch_default_cleanup is None
        assert loaded.swapper_enabled is None
