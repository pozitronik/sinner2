"""Tests for QProcessorControls.snapshot() / apply_snapshot() and the
ProcessorParamsSnapshot value object.

These pin the keystone of the Phase-3 GUI refactor: a single immutable snapshot
that captures (snapshot) and restores (apply_snapshot) the entire processor +
session parameter surface. Two gates:
  * READ fidelity — every snapshot field equals its individual getter (nothing
    dropped or mismapped on capture);
  * WRITE round-trip — perturb a widget, capture, apply to a fresh widget, and
    the re-captured snapshot must be identical (nothing dropped or mismapped on
    the flatten-into-restore path).
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
)

from sinner2.gui.processor_snapshot import ProcessorParamsSnapshot
from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    PredictiveStrategy,
    SyncedStrategy,
)


@pytest.fixture
def widget(qtbot):
    w = QProcessorControls()
    qtbot.addWidget(w)
    return w


# Every snapshot-backed widget. Providers (a checkbox group) and the scale
# slider are perturbed separately / not at all: emptying providers trips the
# CPU-floor, and the slider's int→float scale mapping isn't bit-exact — both are
# covered by the READ-fidelity test instead.
_PERTURB_ATTRS = (
    "_swapper_box", "_swapper_model", "_detection_interval", "_detection_size",
    "_detector", "_many_faces", "_target_sex", "_occlusion_mask",
    "_occlusion_parser", "_occlusion_cache", "_rotation_enabled",
    "_rotation_threshold",
    "_rotation_redetect", "_rotation_source", "_enhancer_box", "_upscaler_box",
    "_upscaler_model", "_upscaler_tile", "_upscaler_fp16", "_upscaler_device",
    "_enhancer_model", "_upscale", "_enhancer_fidelity", "_only_center_face",
    "_only_swapped",
    "_enhancer_fp16", "_enhancer_device", "_strategy_combo", "_worker_count",
    "_playback_combo", "_cache_mode_combo", "_image_format_combo",
    "_image_quality", "_memory_cache_mb", "_write_workers", "_write_queue_size",
    "_video_backend_combo", "_reader_pool_size", "_synced_max_lag_frames",
    "_predictive_max_lead_seconds", "_preprocess_before_play",
)


def _perturb(w: QProcessorControls) -> None:
    """Nudge every snapshot-backed widget to a different VALID value, using each
    widget's own range/options so the result is always in-bounds."""
    for attr in _PERTURB_ATTRS:
        ctl = getattr(w, attr)
        if isinstance(ctl, QCheckBox):
            ctl.setChecked(not ctl.isChecked())
        elif isinstance(ctl, QComboBox):
            if ctl.count() > 1:
                ctl.setCurrentIndex((ctl.currentIndex() + 1) % ctl.count())
        elif isinstance(ctl, (QSpinBox, QDoubleSpinBox, QSlider)):
            hi = ctl.maximum()
            ctl.setValue(hi if ctl.value() != hi else ctl.minimum())


def test_snapshot_captures_each_getter(widget):
    """READ fidelity: every snapshot field mirrors its dedicated getter."""
    s = widget.snapshot()
    assert isinstance(s, ProcessorParamsSnapshot)
    assert s.swapper_enabled == widget.swapper_enabled()
    assert s.swapper_params == widget.swapper_params()
    assert s.swapper_providers == tuple(widget.swapper_providers())
    assert s.enhancer_enabled == widget.enhancer_enabled()
    assert s.enhancer_params == widget.enhancer_params()
    assert s.enhancer_device == widget.enhancer_device()
    assert s.upscaler_enabled == widget.upscaler_enabled()
    assert s.upscaler_params == widget.upscaler_params()
    assert s.upscaler_device == widget.upscaler_device()
    assert s.strategy_name == widget.strategy_name()
    assert s.realtime_workers == widget.realtime_workers()
    assert s.playback_mode == widget.playback_mode()
    assert s.reader_pool_size == widget.reader_pool_size()
    assert s.processing_scale == widget.processing_scale()
    assert s.synced_max_lag_frames == widget.synced_max_lag_frames()
    assert s.predictive_max_lead_seconds == widget.predictive_max_lead_seconds()
    assert s.preprocess_before_play == widget.preprocess_before_play()
    assert s.cache_mode == widget.cache_mode()
    assert s.image_format == widget.image_format()
    assert s.image_quality == widget.image_quality()
    assert s.memory_cache_mb == widget.memory_cache_mb()
    assert s.write_workers == widget.write_workers()
    assert s.write_queue_size == widget.write_queue_size()
    assert s.video_backend == widget.video_backend()


def test_apply_snapshot_is_idempotent_on_default_state(widget):
    """Applying a freshly-captured snapshot leaves the controls unchanged."""
    s0 = widget.snapshot()
    widget.apply_snapshot(s0)
    assert widget.snapshot() == s0


def test_perturbed_snapshot_round_trips_to_a_fresh_widget(qtbot):
    """WRITE round-trip: a non-default snapshot, applied to a fresh widget,
    re-captures identically — so no field is dropped or mismapped through the
    flatten-into-apply_restored_settings path."""
    src = QProcessorControls()
    qtbot.addWidget(src)
    _perturb(src)
    s = src.snapshot()

    dst = QProcessorControls()
    qtbot.addWidget(dst)
    # Sanity: the perturbation actually moved the state off the fresh default,
    # so the round-trip is a real test and not a no-op.
    assert dst.snapshot() != s

    dst.apply_snapshot(s)
    assert dst.snapshot() == s


def test_to_session_config_synced_strategy_and_cache_bundle(widget):
    """to_session_config rebuilds a SyncedStrategy from the name + lag, bundles
    the cache knobs (MB→bytes), and passes the param models through unchanged."""
    s = dataclasses.replace(
        widget.snapshot(),
        strategy_name="SyncedStrategy",
        synced_max_lag_frames=42,
        memory_cache_mb=7,
    )
    cfg = s.to_session_config()
    assert isinstance(cfg["strategy"], SyncedStrategy)
    assert cfg["strategy"].max_lag_frames == 42
    assert cfg["cache_settings"].memory_max_bytes == 7 * 1024 * 1024
    assert cfg["cache_settings"].mode == s.cache_mode
    assert cfg["cache_settings"].image_format == s.image_format
    assert cfg["cache_settings"].image_quality == s.image_quality
    assert cfg["cache_settings"].write_workers == s.write_workers
    assert cfg["cache_settings"].write_queue_size == s.write_queue_size
    # Param models + the rest pass straight through.
    assert cfg["swapper_params"] is s.swapper_params
    assert cfg["enhancer_params"] is s.enhancer_params
    assert cfg["upscaler_params"] is s.upscaler_params
    assert cfg["swapper_enabled"] == s.swapper_enabled
    assert cfg["enhancer_enabled"] == s.enhancer_enabled
    assert cfg["upscaler_enabled"] == s.upscaler_enabled
    assert cfg["swapper_providers"] == s.swapper_providers
    assert cfg["enhancer_device"] == s.enhancer_device
    assert cfg["upscaler_device"] == s.upscaler_device
    assert cfg["worker_count"] == s.realtime_workers
    assert cfg["playback_mode"] == s.playback_mode


def test_to_session_config_best_effort_strategy(widget):
    s = dataclasses.replace(widget.snapshot(), strategy_name="BestEffortStrategy")
    assert isinstance(s.to_session_config()["strategy"], BestEffortStrategy)


def test_to_session_config_predictive_strategy_with_lead(widget):
    # Predictive rebuilds from the name + the max-lead cap.
    s = dataclasses.replace(
        widget.snapshot(),
        strategy_name="PredictiveStrategy",
        predictive_max_lead_seconds=2.5,
    )
    strat = s.to_session_config()["strategy"]
    assert isinstance(strat, PredictiveStrategy)
    assert strat.max_lead_seconds == 2.5


def test_to_session_config_defaults_to_predictive(widget):
    # The widget default (Predictive first in the combo) round-trips to a
    # PredictiveStrategy — the default viewing strategy.
    s = widget.snapshot()
    assert s.strategy_name == "PredictiveStrategy"
    assert isinstance(s.to_session_config()["strategy"], PredictiveStrategy)


def test_to_settings_kwargs_flattens_with_value_tokens(widget):
    """Persist surface: str-Enum model fields become their stable .value tokens,
    providers serialize as a plain list, primitives + session enums pass through."""
    s = dataclasses.replace(
        widget.snapshot(), swapper_providers=("CPUExecutionProvider",)
    )
    kw = s.to_settings_kwargs()
    assert kw["swapper_model"] == s.swapper_params.model.value
    assert kw["swapper_detector"] == s.swapper_params.detector.value
    assert kw["swapper_target_sex"] == s.swapper_params.target_sex.value
    assert kw["swapper_occlusion_parser"] == s.swapper_params.occlusion_parser.value
    assert (
        kw["swapper_rotation_angle_source"]
        == s.swapper_params.rotation_angle_source.value
    )
    assert kw["enhancer_model"] == s.enhancer_params.model.value
    assert kw["upscaler_model"] == s.upscaler_params.model.value
    assert kw["swapper_providers"] == ["CPUExecutionProvider"]
    assert isinstance(kw["swapper_providers"], list)
    assert kw["realtime_workers"] == s.realtime_workers
    assert kw["playback_mode"] == s.playback_mode
    assert kw["synced_max_lag_frames"] == s.synced_max_lag_frames
    assert kw["predictive_max_lead_seconds"] == s.predictive_max_lead_seconds
    assert kw["preprocess_before_play"] == s.preprocess_before_play
    assert kw["memory_cache_mb"] == s.memory_cache_mb


def test_to_settings_kwargs_keys_match_apply_restored_settings(widget):
    """The flat map IS the apply_restored_settings keyword surface — so the
    shared persist/restore path provides exactly those keys, no more, no less.
    This guards the dedup: a field added to one side must appear on the other."""
    kw = widget.snapshot().to_settings_kwargs()
    params = inspect.signature(QProcessorControls.apply_restored_settings).parameters
    expected = {
        name for name, p in params.items() if p.kind is p.KEYWORD_ONLY
    }
    assert set(kw) == expected


def test_restore_keys_all_exist_on_settings():
    """main_window._restore_processor_settings spreads
    {name: getattr(self._settings, name)} over apply_restored_settings's
    keyword-only params, so every one of those names MUST be a Settings field —
    otherwise the spread raises AttributeError at startup. This pins that parity
    so a newly-added restored param can't silently break restore."""
    from sinner2.config.settings import Settings

    params = inspect.signature(QProcessorControls.apply_restored_settings).parameters
    restore_keys = {
        name for name, p in params.items() if p.kind is p.KEYWORD_ONLY
    }
    missing = restore_keys - set(Settings.model_fields)
    assert not missing, (
        f"apply_restored_settings params missing from Settings: {missing}"
    )
