"""Shared enable/disable gating for the processor-chain controls.

The live Settings panel (``QProcessorControls``) and the batch task form
(``QBatchTaskDialog``) expose the same chain knobs and must gray them out by the
*exact* same rules — otherwise the two forms drift (a knob live in one, dead in
the other). These free functions are the single authority for those rules; each
panel passes its own widgets, so neither can diverge. Each panel keeps a thin
``_update_*_rows`` method (so existing signal connections still bind) that just
forwards its widgets here.

Detector-row gating is deliberately NOT shared: the live panel additionally
grays the gender filter while face-map routing supersedes it, which the batch
form has no notion of — so it stays panel-local.
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QWidget

from sinner2.pipeline.processors.face_enhancer import EnhancerModel
from sinner2.pipeline.processors.face_swapper import SwapperModel
from sinner2.pipeline.processors.occlusion import OcclusionMaskMode
from sinner2.pipeline.processors.swapper_models import is_insightface_model
from sinner2.pipeline.processors.upscaler import (
    UpscalerModel,
    model_runtime,
    model_supports_fp16,
)


def update_enhancer_rows(
    model_combo: QComboBox,
    upscale: QWidget,
    fidelity: QWidget,
    fp16: QWidget,
    device: QWidget,
    providers_row: QWidget,
) -> None:
    """Enable only the knobs the selected enhancer model uses — Upscale for
    GFPGAN, Fidelity for CodeFormer; GPEN / RestoreFormer++ have neither. fp16 +
    torch device are GFPGAN-only; the ONNX restorers use the ONNX providers row
    instead, so the two are mutually exclusive by model."""
    model = model_combo.currentData()
    is_gfpgan = model == EnhancerModel.GFPGAN.value
    upscale.setEnabled(is_gfpgan)
    fidelity.setEnabled(model == EnhancerModel.CODEFORMER.value)
    fp16.setEnabled(is_gfpgan)
    device.setEnabled(is_gfpgan)
    providers_row.setEnabled(not is_gfpgan)


def update_rotation_rows(
    enabled: QCheckBox, threshold: QWidget, redetect: QWidget, source: QWidget
) -> None:
    """Gray the rotation knobs (threshold / re-detect / angle source) when
    rotation compensation is off — they have no effect then."""
    on = enabled.isChecked()
    threshold.setEnabled(on)
    redetect.setEnabled(on)
    source.setEnabled(on)


def update_temporal_rows(
    enabled: QCheckBox, window: QWidget, strength: QWidget
) -> None:
    """Gray the temporal-stabilization knobs (window / strength) when the
    feature is off — they have no effect then."""
    on = enabled.isChecked()
    window.setEnabled(on)
    strength.setEnabled(on)


def update_occlusion_rows(
    mask: QCheckBox,
    mode_combo: QComboBox,
    parser: QWidget,
    occluder_model: QWidget,
    cache: QWidget,
) -> None:
    """Link the occlusion sub-controls to the master checkbox and to each other:
    everything grays out when the mask is off; the parser applies only to
    region/both, the occluder model only to occluder/both."""
    on = mask.isChecked()
    mode = mode_combo.currentData()
    mode_combo.setEnabled(on)
    parser.setEnabled(on and mode != OcclusionMaskMode.OCCLUDER.value)
    occluder_model.setEnabled(on and mode != OcclusionMaskMode.REGION.value)
    cache.setEnabled(on)


def update_upscaler_rows(
    model_combo: QComboBox,
    fp16: QWidget,
    device: QWidget,
    providers_row: QWidget,
) -> None:
    """Gray the fp16 knob for models it doesn't apply to — the ONNX upscalers
    (no effect) and SwinIR (its attention can't run in half) — and split the
    torch device vs ONNX providers row by the selected model's runtime."""
    model = UpscalerModel(model_combo.currentData())
    is_onnx = model_runtime(model) == "onnx"
    fp16.setEnabled(model_supports_fp16(model))
    device.setEnabled(not is_onnx)
    providers_row.setEnabled(is_onnx)


def update_swapper_model_rows(model_combo: QComboBox, fast_paste: QWidget) -> None:
    """Gray the fast-paste knob for the 256px swappers — they ALWAYS blend
    through the fast ROI paste; the toggle only applies to the insightface
    models (inswapper / reswapper), whose original blend it replaces."""
    model = SwapperModel(model_combo.currentData())
    fast_paste.setEnabled(is_insightface_model(model))
