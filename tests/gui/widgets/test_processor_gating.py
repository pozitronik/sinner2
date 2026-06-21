"""Direct tests for the shared processor-chain gating rules.

These pin the single authority that both the live Settings panel and the batch
task form delegate to (so the two can't drift). Pure logic — the helpers only
call currentData()/isChecked()/setEnabled(), so duck-typed fakes suffice (no Qt).
"""
from __future__ import annotations

from sinner2.gui.widgets.processor_gating import (
    update_enhancer_rows,
    update_occlusion_rows,
    update_rotation_rows,
    update_swapper_model_rows,
    update_upscaler_rows,
)


class _Combo:
    def __init__(self, data: object) -> None:
        self._data = data
        self.enabled: bool | None = None

    def currentData(self) -> object:
        return self._data

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class _Check:
    def __init__(self, checked: bool) -> None:
        self._checked = checked
        self.enabled: bool | None = None

    def isChecked(self) -> bool:
        return self._checked

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class _W:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class TestEnhancerRows:
    def _run(self, model: str):
        upscale, fidelity, fp16, device, providers = _W(), _W(), _W(), _W(), _W()
        update_enhancer_rows(
            _Combo(model), upscale, fidelity, fp16, device, providers
        )
        return upscale, fidelity, fp16, device, providers

    def test_gfpgan_enables_upscale_fp16_device_only(self):
        upscale, fidelity, fp16, device, providers = self._run("gfpgan")
        assert (upscale.enabled, fp16.enabled, device.enabled) == (True, True, True)
        assert fidelity.enabled is False        # fidelity is CodeFormer's knob
        assert providers.enabled is False       # GFPGAN is torch, not ONNX

    def test_codeformer_enables_fidelity_and_providers_only(self):
        upscale, fidelity, fp16, device, providers = self._run("codeformer")
        assert fidelity.enabled is True
        assert providers.enabled is True        # ONNX restorer → providers row
        assert (upscale.enabled, fp16.enabled, device.enabled) == (False, False, False)

    def test_other_onnx_model_enables_neither_scalar_knob(self):
        # GPEN / RestoreFormer++ have neither Upscale nor Fidelity; still ONNX.
        upscale, fidelity, fp16, device, providers = self._run("restoreformer_pp")
        assert (upscale.enabled, fidelity.enabled) == (False, False)
        assert providers.enabled is True


class TestRotationRows:
    def test_on_enables_all(self):
        thr, redet, src = _W(), _W(), _W()
        update_rotation_rows(_Check(True), thr, redet, src)
        assert (thr.enabled, redet.enabled, src.enabled) == (True, True, True)

    def test_off_disables_all(self):
        thr, redet, src = _W(), _W(), _W()
        update_rotation_rows(_Check(False), thr, redet, src)
        assert (thr.enabled, redet.enabled, src.enabled) == (False, False, False)


class TestOcclusionRows:
    def _run(self, on: bool, mode: str):
        mode_combo = _Combo(mode)
        parser, occluder, cache = _W(), _W(), _W()
        update_occlusion_rows(_Check(on), mode_combo, parser, occluder, cache)
        return mode_combo, parser, occluder, cache

    def test_mask_off_disables_everything(self):
        mode_combo, parser, occluder, cache = self._run(False, "region")
        assert mode_combo.enabled is False
        assert (parser.enabled, occluder.enabled, cache.enabled) == (
            False, False, False
        )

    def test_region_mode_enables_parser_not_occluder(self):
        _, parser, occluder, cache = self._run(True, "region")
        assert (parser.enabled, occluder.enabled, cache.enabled) == (True, False, True)

    def test_occluder_mode_enables_occluder_not_parser(self):
        _, parser, occluder, _c = self._run(True, "occluder")
        assert (parser.enabled, occluder.enabled) == (False, True)

    def test_both_mode_enables_parser_and_occluder(self):
        _, parser, occluder, _c = self._run(True, "both")
        assert (parser.enabled, occluder.enabled) == (True, True)


class TestUpscalerRows:
    def _run(self, model: str):
        fp16, device, providers = _W(), _W(), _W()
        update_upscaler_rows(_Combo(model), fp16, device, providers)
        return fp16, device, providers

    def test_torch_model_enables_device_and_fp16(self):
        fp16, device, providers = self._run("general-x4v3")  # torch, fp16_ok
        assert (fp16.enabled, device.enabled, providers.enabled) == (
            True, True, False
        )

    def test_swinir_torch_but_fp16_unsupported(self):
        fp16, device, providers = self._run("swinir-m")  # torch, fp16_ok=False
        assert fp16.enabled is False
        assert device.enabled is True   # still a torch device
        assert providers.enabled is False

    def test_onnx_model_enables_providers_only(self):
        fp16, device, providers = self._run("hat-x4")  # onnx
        assert (fp16.enabled, device.enabled, providers.enabled) == (
            False, False, True
        )


class TestSwapperModelRows:
    def test_insightface_model_enables_fast_paste(self):
        fast = _W()
        update_swapper_model_rows(_Combo("inswapper_128"), fast)
        assert fast.enabled is True

    def test_256px_model_disables_fast_paste(self):
        fast = _W()
        update_swapper_model_rows(_Combo("ghost_1_256"), fast)
        assert fast.enabled is False
