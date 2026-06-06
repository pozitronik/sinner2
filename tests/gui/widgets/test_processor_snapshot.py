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
    "_occlusion_parser", "_rotation_enabled", "_rotation_threshold",
    "_rotation_redetect", "_rotation_source", "_enhancer_box", "_upscaler_box",
    "_upscaler_model", "_upscaler_tile", "_upscaler_fp16", "_upscaler_device",
    "_enhancer_model", "_upscale", "_enhancer_fidelity", "_only_center_face",
    "_enhancer_fp16", "_enhancer_device", "_strategy_combo", "_worker_count",
    "_playback_combo", "_cache_mode_combo", "_image_format_combo",
    "_image_quality", "_memory_cache_mb", "_write_workers", "_write_queue_size",
    "_video_backend_combo", "_reader_pool_size", "_synced_max_lag_frames",
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
