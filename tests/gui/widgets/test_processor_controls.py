from pathlib import Path

import pytest

from sinner2.config import settings
from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.image_writer import ImageFormat
from sinner2.pipeline.playback_mode import PlaybackMode


@pytest.fixture
def widget(qtbot):
    w = QProcessorControls()
    qtbot.addWidget(w)
    return w


class TestProviderFloor:
    """You can't run on no provider — unchecking everything forces CPU back on
    (the floor), so the swapper provider selection is never empty."""

    def test_unchecking_all_forces_cpu(self, widget):
        for cb in list(widget._provider_checkboxes.values()):  # noqa: SLF001
            cb.setChecked(False)
        cpu = widget._provider_checkboxes["CPUExecutionProvider"]  # noqa: SLF001
        assert cpu.isChecked()
        assert widget.swapper_providers() == ["CPUExecutionProvider"]

    def test_cuda_only_selection_is_left_alone(self, widget):
        # Only when EVERYTHING is unchecked does CPU get forced — a non-empty
        # selection (e.g. CUDA only) is respected.
        for name, cb in widget._provider_checkboxes.items():  # noqa: SLF001
            cb.setChecked(name == "CUDAExecutionProvider")
        if "CUDAExecutionProvider" in widget._provider_checkboxes:  # noqa: SLF001
            assert widget.swapper_providers() == ["CUDAExecutionProvider"]

    def test_restore_empty_selection_forces_cpu(self, widget):
        # _NONE_RESTORE_KWARGS is a module global (defined below) — resolved at
        # call time, so referencing it here is fine.
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "swapper_providers": []}
        )
        assert widget.swapper_providers() == ["CPUExecutionProvider"]


class TestQProcessorControls:
    def test_default_swapper_params(self, widget):
        params = widget.swapper_params()
        assert params.detection_interval == 1
        assert params.many_faces is True

    def test_default_enhancer_params(self, widget):
        params = widget.enhancer_params()
        assert params.upscale == 1
        assert params.only_center_face is False

    def test_default_enhancer_enabled(self, widget):
        assert widget.enhancer_enabled() is True

    def test_default_swapper_enabled(self, widget):
        assert widget.swapper_enabled() is True

    def test_toggling_swapper_box_reflects_and_emits(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._swapper_box.setChecked(False)  # noqa: SLF001
        assert widget.swapper_enabled() is False

    def test_changing_detection_interval_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._detection_interval.setValue(5)  # noqa: SLF001
        assert widget.swapper_params().detection_interval == 5

    def test_default_target_sex_is_both(self, widget):
        from sinner2.pipeline.processors.face_swapper import TargetSex

        assert widget.swapper_params().target_sex is TargetSex.BOTH

    def test_target_sex_combo_emits_config_changed(self, widget, qtbot):
        # Change the combo via setCurrentIndex (the path a click takes
        # via the dropdown) and verify configChanged fires AND the new
        # value reaches the params accessor.
        from sinner2.pipeline.processors.face_swapper import TargetSex

        female_index = next(
            i
            for i in range(widget._target_sex.count())  # noqa: SLF001
            if widget._target_sex.itemData(i) == TargetSex.FEMALE.value  # noqa: SLF001
        )
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._target_sex.setCurrentIndex(female_index)  # noqa: SLF001
        assert widget.swapper_params().target_sex is TargetSex.FEMALE

    def test_toggling_many_faces_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._many_faces.setChecked(False)  # noqa: SLF001
        assert widget.swapper_params().many_faces is False

    def test_changing_upscale_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._upscale.setValue(2)  # noqa: SLF001
        assert widget.enhancer_params().upscale == 2

    def test_toggling_center_face_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._only_center_face.setChecked(True)  # noqa: SLF001
        assert widget.enhancer_params().only_center_face is True

    def test_toggling_enhancer_group_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._enhancer_box.setChecked(False)  # noqa: SLF001
        assert widget.enhancer_enabled() is False

    def test_enhancer_model_flows_to_params(self, widget):
        from sinner2.pipeline.processors.face_enhancer import EnhancerModel

        widget.set_enhancer_model(EnhancerModel.CODEFORMER.value)
        widget._enhancer_fidelity.setValue(0.4)  # noqa: SLF001
        params = widget.enhancer_params()
        assert params.model is EnhancerModel.CODEFORMER
        assert params.codeformer_fidelity == 0.4

    def test_codeformer_disables_upscale_enables_fidelity(self, widget):
        from sinner2.pipeline.processors.face_enhancer import EnhancerModel

        widget.set_enhancer_model(EnhancerModel.CODEFORMER.value)
        assert widget._upscale.isEnabled() is False  # noqa: SLF001
        assert widget._enhancer_fidelity.isEnabled() is True  # noqa: SLF001
        widget.set_enhancer_model(EnhancerModel.GFPGAN.value)
        assert widget._upscale.isEnabled() is True  # noqa: SLF001
        assert widget._enhancer_fidelity.isEnabled() is False  # noqa: SLF001

    def test_plain_bfr_models_disable_both_knobs(self, widget):
        from sinner2.pipeline.processors.face_enhancer import EnhancerModel

        for model in (EnhancerModel.GPEN_512, EnhancerModel.RESTOREFORMER_PP):
            widget.set_enhancer_model(model.value)
            assert widget._upscale.isEnabled() is False  # noqa: SLF001
            assert widget._enhancer_fidelity.isEnabled() is False  # noqa: SLF001
            assert widget.enhancer_params().model is model

    def test_enhancer_combo_lists_all_models(self, widget):
        from sinner2.pipeline.processors.face_enhancer import EnhancerModel

        values = {
            widget._enhancer_model.itemData(i)  # noqa: SLF001
            for i in range(widget._enhancer_model.count())  # noqa: SLF001
        }
        assert {m.value for m in EnhancerModel} <= values

    def test_set_enhancer_model_does_not_emit_config_changed(self, widget, qtbot):
        # Revert path (declined download) must not re-trigger a chain rebuild.
        from sinner2.pipeline.processors.face_enhancer import EnhancerModel

        spy = []
        widget.configChanged.connect(lambda: spy.append(1))
        widget.set_enhancer_model(EnhancerModel.CODEFORMER.value)
        assert spy == []

    def test_default_strategy_is_best_effort(self, widget):
        from sinner2.pipeline.skip_strategy import BestEffortStrategy

        assert isinstance(widget.skip_strategy(), BestEffortStrategy)

    def test_changing_strategy_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._strategy_combo.setCurrentIndex(1)  # noqa: SLF001
        from sinner2.pipeline.skip_strategy import SyncedStrategy

        assert isinstance(widget.skip_strategy(), SyncedStrategy)

    def test_default_worker_count(self, widget):
        assert widget.realtime_workers() == 1

    def test_changing_worker_count_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._worker_count.setValue(4)  # noqa: SLF001
        assert widget.realtime_workers() == 4


_FULL_RESTORE_KWARGS = dict(
    realtime_workers=8,
    strategy_name="SyncedStrategy",
    enhancer_enabled=False,
    swapper_model="ghost_2_256",
    swapper_detection_interval=3,
    swapper_detection_size=320,
    swapper_detector="yoloface",
    swapper_many_faces=False,
    swapper_target_sex="F",
    swapper_occlusion_mask=True,
    swapper_occlusion_parser="parsenet",
    swapper_rotation_compensation=True,
    swapper_rotation_threshold_deg=25,
    swapper_rotation_redetect=False,
    swapper_rotation_angle_source="pose",
    enhancer_model="codeformer",
    enhancer_upscale=4,
    enhancer_only_center_face=True,
    enhancer_codeformer_fidelity=0.3,
    playback_mode=PlaybackMode.UNLIMITED,
    cache_mode=CacheMode.READ_ONLY,
    image_format=ImageFormat.PNG,
    image_quality=80,
    memory_cache_mb=256,
    write_workers=8,
    write_queue_size=16,
    video_backend=VideoBackend.CV2,
    reader_pool_size=4,
    processing_scale=0.5,
    synced_max_lag_frames=120,
    swapper_providers=["CPUExecutionProvider"],
    enhancer_device="cpu",
)

_NONE_RESTORE_KWARGS = dict(
    realtime_workers=None,
    strategy_name=None,
    enhancer_enabled=None,
    swapper_model=None,
    swapper_detection_interval=None,
    swapper_detection_size=None,
    swapper_detector=None,
    swapper_many_faces=None,
    swapper_target_sex=None,
    swapper_occlusion_mask=None,
    swapper_occlusion_parser=None,
    swapper_rotation_compensation=None,
    swapper_rotation_threshold_deg=None,
    swapper_rotation_redetect=None,
    swapper_rotation_angle_source=None,
    enhancer_model=None,
    enhancer_upscale=None,
    enhancer_only_center_face=None,
    enhancer_codeformer_fidelity=None,
    playback_mode=None,
    cache_mode=None,
    image_format=None,
    image_quality=None,
    memory_cache_mb=None,
    write_workers=None,
    write_queue_size=None,
    video_backend=None,
    reader_pool_size=None,
    processing_scale=None,
    synced_max_lag_frames=None,
    swapper_providers=None,
    enhancer_device=None,
)


class TestApplyRestoredSettings:
    """apply_restored_settings is the seam main_window uses on startup to
    push persisted values back into the widget. Each field must land where
    the corresponding getter reads from, and None values must leave widget
    defaults alone (so a first-run user with no settings.json sees the
    spinbox defaults rather than zeros)."""

    def test_applies_worker_count(self, widget):
        widget.apply_restored_settings(**{**_NONE_RESTORE_KWARGS, "realtime_workers": 7})
        assert widget.realtime_workers() == 7

    def test_applies_enhancer_device(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "enhancer_device": "cpu"}
        )
        assert widget.enhancer_device() == "cpu"

    def test_default_enhancer_device_is_auto(self, widget):
        assert widget.enhancer_device() == "auto"

    def test_applies_swapper_enabled(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "swapper_enabled": False}
        )
        assert widget.swapper_enabled() is False

    def test_applies_strategy_name(self, widget):
        from sinner2.pipeline.skip_strategy import SyncedStrategy

        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "strategy_name": "SyncedStrategy"}
        )
        assert widget.strategy_name() == "SyncedStrategy"
        assert isinstance(widget.skip_strategy(), SyncedStrategy)

    def test_applies_enhancer_enabled(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "enhancer_enabled": False}
        )
        assert widget.enhancer_enabled() is False

    def test_applies_swapper_params(self, widget):
        from sinner2.pipeline.processors.face_swapper import TargetSex

        widget.apply_restored_settings(
            **{
                **_NONE_RESTORE_KWARGS,
                "swapper_detection_interval": 5,
                "swapper_many_faces": False,
                "swapper_target_sex": "F",
            }
        )
        params = widget.swapper_params()
        assert params.detection_interval == 5
        assert params.many_faces is False
        assert params.target_sex is TargetSex.FEMALE

    def test_applies_enhancer_params(self, widget):
        widget.apply_restored_settings(
            **{
                **_NONE_RESTORE_KWARGS,
                "enhancer_upscale": 3,
                "enhancer_only_center_face": True,
                "enhancer_fp16": False,
            }
        )
        params = widget.enhancer_params()
        assert params.upscale == 3
        assert params.only_center_face is True
        assert params.fp16 is False

    def test_enhancer_fp16_defaults_on_and_flows_to_params(self, widget):
        # Default checked (matches FaceEnhancerParams default), and the checkbox
        # state flows into the params the controller reads.
        assert widget.enhancer_params().fp16 is True
        widget._enhancer_fp16.setChecked(False)  # noqa: SLF001
        assert widget.enhancer_params().fp16 is False


    def test_none_values_preserve_widget_defaults(self, widget):
        default_worker = widget.realtime_workers()
        default_strategy = widget.strategy_name()
        default_enhancer_enabled = widget.enhancer_enabled()
        default_swapper = widget.swapper_params()
        default_enhancer = widget.enhancer_params()
        widget.apply_restored_settings(**_NONE_RESTORE_KWARGS)
        assert widget.realtime_workers() == default_worker
        assert widget.strategy_name() == default_strategy
        assert widget.enhancer_enabled() is default_enhancer_enabled
        assert widget.swapper_params() == default_swapper
        assert widget.enhancer_params() == default_enhancer

    def test_unknown_strategy_name_falls_back_to_current(self, widget):
        original = widget.strategy_name()
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "strategy_name": "DoesNotExistStrategy"}
        )
        assert widget.strategy_name() == original

    def test_emits_config_changed_exactly_once(self, widget):
        # The whole point of the bulk apply: bypass per-field signal storms
        # so the controller doesn't get spammed during startup restore.
        count = [0]
        widget.configChanged.connect(lambda: count.__setitem__(0, count[0] + 1))
        widget.apply_restored_settings(**_FULL_RESTORE_KWARGS)
        assert count[0] == 1

    def test_applies_all_fields_together(self, widget):
        widget.apply_restored_settings(**_FULL_RESTORE_KWARGS)
        assert widget.realtime_workers() == 8
        assert widget.strategy_name() == "SyncedStrategy"
        assert widget.enhancer_enabled() is False
        sp = widget.swapper_params()
        assert sp.detection_interval == 3
        assert sp.many_faces is False
        ep = widget.enhancer_params()
        assert ep.upscale == 4
        assert ep.only_center_face is True
        assert widget.enhancer_device() == "cpu"
        assert widget.playback_mode() is PlaybackMode.UNLIMITED

    @pytest.mark.parametrize("mode", list(PlaybackMode))
    def test_applies_each_playback_mode(self, widget, mode):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "playback_mode": mode}
        )
        assert widget.playback_mode() is mode

    def test_default_playback_mode_is_fixed_30(self, widget):
        assert widget.playback_mode() is PlaybackMode.FIXED_30

    @pytest.mark.parametrize("mode", list(CacheMode))
    def test_applies_each_cache_mode(self, widget, mode):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "cache_mode": mode}
        )
        assert widget.cache_mode() is mode

    @pytest.mark.parametrize("fmt", list(ImageFormat))
    def test_applies_each_image_format(self, widget, fmt):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "image_format": fmt}
        )
        assert widget.image_format() is fmt

    def test_image_quality_disabled_for_png(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "image_format": ImageFormat.PNG}
        )
        assert not widget._image_quality.isEnabled()  # noqa: SLF001
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "image_format": ImageFormat.JPEG}
        )
        assert widget._image_quality.isEnabled()  # noqa: SLF001

    def test_applies_memory_cache_mb(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "memory_cache_mb": 256}
        )
        assert widget.memory_cache_mb() == 256

    def test_applies_write_workers(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "write_workers": 6}
        )
        assert widget.write_workers() == 6

    def test_applies_write_queue_size(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "write_queue_size": 32}
        )
        assert widget.write_queue_size() == 32

    @pytest.mark.parametrize("backend", list(VideoBackend))
    def test_applies_each_video_backend(self, widget, backend):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "video_backend": backend}
        )
        assert widget.video_backend() is backend

    def test_default_video_backend_is_ffmpeg(self, widget):
        assert widget.video_backend() is VideoBackend.FFMPEG


class TestCacheManagementControls:
    """Verify the cache-storage panel surface — signal emissions and the
    setter methods main_window uses to push state into the widget."""

    def test_size_cap_emits_zero_when_disabled(self, widget, qtbot):
        widget._size_cap_enabled.setChecked(True)  # noqa: SLF001 — UI state
        with qtbot.waitSignal(widget.sizeCapChanged, timeout=1000) as blocker:
            widget._size_cap_enabled.setChecked(False)  # noqa: SLF001
        assert blocker.args == [0]

    def test_size_cap_emits_bytes_when_enabled(self, widget, qtbot):
        widget._size_cap_mb.setValue(500)  # noqa: SLF001 — UI state
        with qtbot.waitSignal(widget.sizeCapChanged, timeout=1000) as blocker:
            widget._size_cap_enabled.setChecked(True)  # noqa: SLF001
        assert blocker.args == [500 * 1024 * 1024]

    def test_size_cap_widget_setter_does_not_emit(self, widget):
        # main_window restores persisted values via set_cache_size_cap_bytes;
        # we don't want that to feed back into the change handler.
        received: list[int] = []
        widget.sizeCapChanged.connect(received.append)
        widget.set_cache_size_cap_bytes(256 * 1024 * 1024)
        assert received == []
        assert widget.cache_size_cap_bytes() == 256 * 1024 * 1024

    def test_zero_disables_size_cap_widget(self, widget):
        widget.set_cache_size_cap_bytes(128 * 1024 * 1024)
        assert widget._size_cap_enabled.isChecked()  # noqa: SLF001
        widget.set_cache_size_cap_bytes(0)
        assert not widget._size_cap_enabled.isChecked()  # noqa: SLF001
        assert widget.cache_size_cap_bytes() == 0

    def test_browse_button_emits_signal(self, widget, qtbot):
        with qtbot.waitSignal(widget.browseRootRequested, timeout=1000):
            # Find the Browse button by walking children.
            from PySide6.QtWidgets import QPushButton

            for btn in widget.findChildren(QPushButton):
                if btn.text() == "Browse...":
                    btn.click()
                    break

    def test_clear_all_button_emits_signal(self, widget, qtbot):
        from PySide6.QtWidgets import QPushButton

        with qtbot.waitSignal(widget.clearAllRequested, timeout=1000):
            for btn in widget.findChildren(QPushButton):
                if btn.text() == "Clear all caches":
                    btn.click()
                    break

    def test_invalidate_button_disabled_initially(self, widget):
        assert not widget._invalidate_btn.isEnabled()  # noqa: SLF001

    def test_set_invalidate_enabled(self, widget):
        widget.set_invalidate_enabled(True)
        assert widget._invalidate_btn.isEnabled()  # noqa: SLF001
        widget.set_invalidate_enabled(False)
        assert not widget._invalidate_btn.isEnabled()  # noqa: SLF001

    def test_cache_root_text_setter(self, widget):
        widget.set_cache_root_text(Path("/some/custom/path"))
        assert widget._cache_root_edit.text() == str(Path("/some/custom/path"))  # noqa: SLF001

    def test_cache_stats_text_setter(self, widget):
        widget.set_cache_stats_text("5 entries · 1.0 GB · 50.0 GB free")
        assert "5 entries" in widget._cache_stats_label.text()  # noqa: SLF001


class TestSettingsEndToEnd:
    """Mimics the full path main_window takes: live widget values → Settings
    dataclass → JSON on disk → Settings → fresh widget. Catches any field
    that's wired into apply_restored_settings but missing from the persist
    path, or vice versa — the kind of drift that would silently lose
    user preferences on the next launch."""

    def test_widget_state_persists_through_save_load_apply(
        self, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(tmp_path / "s.json"))

        first = QProcessorControls()
        qtbot.addWidget(first)
        first.apply_restored_settings(**_FULL_RESTORE_KWARGS)

        # Build Settings from live widget state — same composition as
        # main_window._persist_processor_settings.
        swapper = first.swapper_params()
        enhancer = first.enhancer_params()
        settings.save(
            settings.Settings(
                realtime_workers=first.realtime_workers(),
                strategy_name=first.strategy_name(),
                enhancer_enabled=first.enhancer_enabled(),
                swapper_model=swapper.model.value,
                swapper_detection_interval=swapper.detection_interval,
                swapper_detection_size=swapper.detection_size,
                swapper_detector=swapper.detector.value,
                swapper_many_faces=swapper.many_faces,
                swapper_target_sex=swapper.target_sex.value,
                swapper_occlusion_mask=swapper.occlusion_mask,
                swapper_occlusion_parser=swapper.occlusion_parser.value,
                swapper_rotation_compensation=swapper.rotation_compensation,
                swapper_rotation_threshold_deg=swapper.rotation_threshold_deg,
                swapper_rotation_redetect=swapper.rotation_redetect,
                swapper_rotation_angle_source=swapper.rotation_angle_source.value,
                enhancer_model=enhancer.model.value,
                enhancer_upscale=enhancer.upscale,
                enhancer_only_center_face=enhancer.only_center_face,
                enhancer_codeformer_fidelity=enhancer.codeformer_fidelity,
                playback_mode=first.playback_mode(),
                cache_mode=first.cache_mode(),
                image_format=first.image_format(),
                image_quality=first.image_quality(),
                memory_cache_mb=first.memory_cache_mb(),
                write_workers=first.write_workers(),
                write_queue_size=first.write_queue_size(),
                video_backend=first.video_backend(),
                reader_pool_size=first.reader_pool_size(),
                processing_scale=first.processing_scale(),
                synced_max_lag_frames=first.synced_max_lag_frames(),
                swapper_providers=first.swapper_providers(),
                enhancer_device=first.enhancer_device(),
            )
        )

        # Fresh widget + load — mimics next launch's startup.
        reloaded = settings.load()
        second = QProcessorControls()
        qtbot.addWidget(second)
        second.apply_restored_settings(
            realtime_workers=reloaded.realtime_workers,
            strategy_name=reloaded.strategy_name,
            enhancer_enabled=reloaded.enhancer_enabled,
            swapper_model=reloaded.swapper_model,
            swapper_detection_interval=reloaded.swapper_detection_interval,
            swapper_detection_size=reloaded.swapper_detection_size,
            swapper_detector=reloaded.swapper_detector,
            swapper_many_faces=reloaded.swapper_many_faces,
            swapper_target_sex=reloaded.swapper_target_sex,
            swapper_occlusion_mask=reloaded.swapper_occlusion_mask,
            swapper_occlusion_parser=reloaded.swapper_occlusion_parser,
            swapper_rotation_compensation=reloaded.swapper_rotation_compensation,
            swapper_rotation_threshold_deg=reloaded.swapper_rotation_threshold_deg,
            swapper_rotation_redetect=reloaded.swapper_rotation_redetect,
            swapper_rotation_angle_source=reloaded.swapper_rotation_angle_source,
            enhancer_model=reloaded.enhancer_model,
            enhancer_upscale=reloaded.enhancer_upscale,
            enhancer_only_center_face=reloaded.enhancer_only_center_face,
            enhancer_codeformer_fidelity=reloaded.enhancer_codeformer_fidelity,
            playback_mode=reloaded.playback_mode,
            cache_mode=reloaded.cache_mode,
            image_format=reloaded.image_format,
            image_quality=reloaded.image_quality,
            memory_cache_mb=reloaded.memory_cache_mb,
            write_workers=reloaded.write_workers,
            write_queue_size=reloaded.write_queue_size,
            video_backend=reloaded.video_backend,
            reader_pool_size=reloaded.reader_pool_size,
            processing_scale=reloaded.processing_scale,
            synced_max_lag_frames=reloaded.synced_max_lag_frames,
            swapper_providers=reloaded.swapper_providers,
            enhancer_device=reloaded.enhancer_device,
        )

        assert second.realtime_workers() == first.realtime_workers()
        assert second.strategy_name() == first.strategy_name()
        assert second.enhancer_enabled() == first.enhancer_enabled()
        assert second.swapper_params() == first.swapper_params()
        assert second.enhancer_params() == first.enhancer_params()
        assert second.playback_mode() is first.playback_mode()
        assert second.cache_mode() is first.cache_mode()
        assert second.image_format() is first.image_format()
        assert second.image_quality() == first.image_quality()
        assert second.memory_cache_mb() == first.memory_cache_mb()
        assert second.write_workers() == first.write_workers()
        assert second.write_queue_size() == first.write_queue_size()
        assert second.video_backend() is first.video_backend()
        assert second.reader_pool_size() == first.reader_pool_size()
        assert second.processing_scale() == first.processing_scale()
        assert second.synced_max_lag_frames() == first.synced_max_lag_frames()
        assert second.swapper_providers() == first.swapper_providers()
        assert second.enhancer_device() == first.enhancer_device()


class TestGroupOrganization:
    """Detection and swapping options are split into separate group boxes, with
    the Face detector group first (detection precedes the swap)."""

    def test_group_titles_are_consistent(self, widget):
        assert widget._face_box.title() == "Face detector"  # noqa: SLF001
        assert widget._swapper_box.title() == "Face swapper"  # noqa: SLF001
        assert widget._enhancer_box.title() == "Face enhancer"  # noqa: SLF001

    def test_detection_knobs_live_in_detector_group(self, widget):
        for w in (
            widget._detector,  # noqa: SLF001
            widget._detection_size,  # noqa: SLF001
            widget._detection_interval,  # noqa: SLF001
        ):
            assert widget._face_box.isAncestorOf(w)  # noqa: SLF001
            assert not widget._swapper_box.isAncestorOf(w)  # noqa: SLF001

    def test_swap_knobs_stay_in_swapper_group(self, widget):
        for w in (
            widget._swapper_model,  # noqa: SLF001
            widget._many_faces,  # noqa: SLF001
            widget._target_sex,  # noqa: SLF001
            widget._occlusion_mask,  # noqa: SLF001
        ):
            assert widget._swapper_box.isAncestorOf(w)  # noqa: SLF001

    def test_detector_group_precedes_swapper(self, widget):
        inner = widget._face_box.parent()  # noqa: SLF001
        layout = inner.layout()
        assert layout.indexOf(widget._face_box) < layout.indexOf(  # noqa: SLF001
            widget._swapper_box  # noqa: SLF001
        )


class TestDetectorControl:
    def test_detector_flows_to_swapper_params(self, widget):
        from sinner2.pipeline.detectors import DetectorModel

        widget.set_swapper_detector(DetectorModel.YOLOFACE.value)
        assert widget.swapper_params().detector is DetectorModel.YOLOFACE

    def test_default_detector_is_buffalo_l(self, widget):
        from sinner2.pipeline.detectors import DetectorModel

        assert widget.swapper_params().detector is DetectorModel.BUFFALO_L

    def test_alt_detector_greys_gender_filter(self, widget):
        from sinner2.pipeline.detectors import DetectorModel

        widget.set_swapper_detector(DetectorModel.YOLOFACE.value)
        assert widget._target_sex.isEnabled() is False  # noqa: SLF001
        widget.set_swapper_detector(DetectorModel.BUFFALO_L.value)
        assert widget._target_sex.isEnabled() is True  # noqa: SLF001

    def test_restore_sets_detector(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "swapper_detector": "scrfd_2.5g"}
        )
        assert widget._detector.currentData() == "scrfd_2.5g"  # noqa: SLF001


class TestDetectionSizeControl:
    def test_detection_size_flows_to_swapper_params(self, widget):
        widget._detection_size.setValue(320)  # noqa: SLF001
        assert widget.swapper_params().detection_size == 320

    def test_default_detection_size_is_640(self, widget):
        assert widget.swapper_params().detection_size == 640

    def test_restore_sets_detection_size(self, widget):
        widget.apply_restored_settings(
            **{**_NONE_RESTORE_KWARGS, "swapper_detection_size": 256}
        )
        assert widget._detection_size.value() == 256  # noqa: SLF001


class TestFaceDetectorGroup:
    def test_rotation_fields_flow_to_swapper_params(self, widget):
        widget._rotation_enabled.setChecked(True)  # noqa: SLF001
        widget._rotation_threshold.setValue(30)  # noqa: SLF001
        widget._rotation_redetect.setChecked(False)  # noqa: SLF001
        p = widget.swapper_params()
        assert p.rotation_compensation is True
        assert p.rotation_threshold_deg == 30
        assert p.rotation_redetect is False

    def test_rotation_subcontrols_disabled_when_off(self, widget):
        widget._rotation_enabled.setChecked(True)  # noqa: SLF001
        assert widget._rotation_threshold.isEnabled()  # noqa: SLF001
        assert widget._rotation_redetect.isEnabled()  # noqa: SLF001
        assert widget._rotation_source.isEnabled()  # noqa: SLF001
        widget._rotation_enabled.setChecked(False)  # noqa: SLF001
        assert not widget._rotation_threshold.isEnabled()  # noqa: SLF001
        assert not widget._rotation_redetect.isEnabled()  # noqa: SLF001
        assert not widget._rotation_source.isEnabled()  # noqa: SLF001

    def test_enabling_comparison_checks_overlay(self, widget):
        # The widget-level link: checking comparison checks the overlay box too.
        assert widget._overlay_enabled.isChecked() is False  # noqa: SLF001
        widget._comparison_enabled.setChecked(True)  # noqa: SLF001
        assert widget._overlay_enabled.isChecked() is True  # noqa: SLF001

    def test_unchecking_overlay_unchecks_comparison(self, widget):
        widget._comparison_enabled.setChecked(True)  # noqa: SLF001  # both on (linked)
        assert widget._comparison_enabled.isChecked() is True  # noqa: SLF001
        widget._overlay_enabled.setChecked(False)  # noqa: SLF001
        assert widget._comparison_enabled.isChecked() is False  # noqa: SLF001


class TestProcessingScaleReadout:
    def test_percent_only_without_target(self, widget):
        widget._scale_slider.setValue(50)  # noqa: SLF001
        assert widget._scale_label.text() == "50%"  # noqa: SLF001
        assert widget.processing_scale() == 0.5

    def test_shows_resulting_dims_with_target(self, widget):
        widget.set_target_native_size((1920, 1080))
        widget._scale_slider.setValue(50)  # noqa: SLF001
        assert widget._scale_label.text() == "50% [960x540]"  # noqa: SLF001

    def test_clearing_target_drops_dims(self, widget):
        widget.set_target_native_size((1920, 1080))
        widget.set_target_native_size(None)
        widget._scale_slider.setValue(75)  # noqa: SLF001
        assert widget._scale_label.text() == "75%"  # noqa: SLF001


class TestResponsiveFormDensity:
    """The settings groups share ONE caption-column width and flip together
    between side-by-side (wide) and stacked (narrow)."""

    def _labels(self, widget):
        from PySide6.QtWidgets import QFormLayout

        out = []
        for form in widget._forms:  # noqa: SLF001
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item is not None and item.widget() is not None:
                    out.append(item.widget())
        return out

    def test_wide_is_side_by_side_with_uniform_caption_column(self, widget):
        from PySide6.QtWidgets import QFormLayout

        widget.resize(700, 900)
        widget._apply_form_density()  # noqa: SLF001
        for form in widget._forms:  # noqa: SLF001
            assert (
                form.rowWrapPolicy() == QFormLayout.RowWrapPolicy.DontWrapRows
            )
        # Every caption across every group shares one non-zero column width.
        widths = {lbl.minimumWidth() for lbl in self._labels(widget)}
        assert len(widths) == 1
        assert widths.pop() > 0

    def test_narrow_stacks_caption_above_control(self, widget):
        from PySide6.QtWidgets import QFormLayout

        widget.resize(220, 900)
        widget._apply_form_density()  # noqa: SLF001
        for form in widget._forms:  # noqa: SLF001
            assert (
                form.rowWrapPolicy() == QFormLayout.RowWrapPolicy.WrapAllRows
            )
        # Stacked → no forced caption width (the control gets the full row).
        assert all(lbl.minimumWidth() == 0 for lbl in self._labels(widget))


class TestRerenderButton:
    def test_disabled_until_session_active(self, widget):
        assert widget._rerender_btn.isEnabled() is False  # noqa: SLF001
        widget.set_invalidate_enabled(True)
        assert widget._rerender_btn.isEnabled() is True  # noqa: SLF001
        assert widget._invalidate_btn.isEnabled() is True  # noqa: SLF001
        widget.set_invalidate_enabled(False)
        assert widget._rerender_btn.isEnabled() is False  # noqa: SLF001

    def test_click_emits_signal(self, widget, qtbot):
        widget.set_invalidate_enabled(True)
        with qtbot.waitSignal(widget.rerenderRequested, timeout=1000):
            widget._rerender_btn.click()  # noqa: SLF001
