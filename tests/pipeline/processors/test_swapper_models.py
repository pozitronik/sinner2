"""Tests for the alternative-swapper registry + generic ONNX backend.

No real models are loaded — sessions/converters are stubbed. These verify the
pipeline MECHANICS (registry lookups, warp/mask/paste shapes, source-embedding
dispatch, normalization round-trip), not model quality."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.pipeline.processors.swapper_models import (
    GenericOnnxSwapper,
    SwapperModel,
    _box_mask,
    _paste_back,
    _warp_face,
    get_spec,
    is_insightface_model,
    model_filename,
    model_files,
)

_KPS = np.array(
    [[20, 25], [44, 25], [32, 38], [22, 50], [42, 50]], np.float32
)
_TEMPLATE = np.array(
    [[0.36, 0.4], [0.63, 0.4], [0.5, 0.56], [0.38, 0.72], [0.61, 0.72]],
    np.float32,
)


class _StubConverter:
    def __init__(self, out: np.ndarray) -> None:
        self._out = out

    def run(self, _names, feeds):  # noqa: ANN001
        return [self._out]


class _IdentitySession:
    """Echoes the 'target' tensor back as the model output (NCHW)."""

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _names, feeds):  # noqa: ANN001
        return [feeds["target"]]


class TestRegistry:
    def test_model_filename(self):
        assert model_filename(SwapperModel.GHOST_2_256) == "ghost_2_256.onnx"

    def test_ghost_needs_converter_companion(self):
        assert model_files(SwapperModel.GHOST_1_256) == [
            "ghost_1_256.onnx", "crossface_ghost.onnx",
        ]

    def test_simswap_needs_converter_companion(self):
        assert model_files(SwapperModel.SIMSWAP_256) == [
            "simswap_256.onnx", "crossface_simswap.onnx",
        ]

    def test_uniface_has_no_companion(self):
        assert model_files(SwapperModel.UNIFACE_256) == ["uniface_256.onnx"]

    def test_inswapper_and_reswapper_are_insightface(self):
        assert is_insightface_model(SwapperModel.INSWAPPER_128) is True
        assert is_insightface_model(SwapperModel.RESWAPPER_128) is True
        assert is_insightface_model(SwapperModel.GHOST_2_256) is False


class TestGeometryHelpers:
    def test_warp_face_shape(self):
        frame = np.full((64, 64, 3), 100, np.uint8)
        crop, matrix = _warp_face(frame, _KPS, _TEMPLATE, 256)
        assert crop.shape == (256, 256, 3)
        assert matrix.shape == (2, 3)

    def test_box_mask_center_one_corners_zero(self):
        mask = _box_mask(256)
        assert mask.shape == (256, 256)
        assert mask[128, 128] == pytest.approx(1.0, abs=1e-3)
        assert mask[0, 0] < 0.05  # corner ~0 (slight Gaussian-feather bleed)

    def test_paste_back_blends_center_keeps_corner(self):
        frame = np.full((64, 64, 3), 50, np.uint8)
        crop = np.full((256, 256, 3), 200, np.uint8)
        _, matrix = _warp_face(frame, _KPS, _TEMPLATE, 256)
        out = _paste_back(frame, crop, _box_mask(256), matrix)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8
        assert out.max() > 50  # the bright crop landed somewhere


def _make_backend(model: SwapperModel) -> GenericOnnxSwapper:
    return GenericOnnxSwapper(get_spec(model), providers=["CPUExecutionProvider"])


class TestSourceEmbeddingDispatch:
    def test_ghost_uses_raw_converted_embedding(self):
        backend = _make_backend(SwapperModel.GHOST_2_256)
        converted = np.arange(512, dtype=np.float32).reshape(1, 512)
        backend._converter = _StubConverter(converted)  # noqa: SLF001
        face = SimpleNamespace(embedding=np.ones(512, np.float32), kps=_KPS)
        backend.prepare_source(np.zeros((64, 64, 3), np.uint8), face)
        # ghost: raw ravel, NOT normalized
        np.testing.assert_allclose(
            backend._source_input.ravel(), converted.ravel()  # noqa: SLF001
        )

    def test_simswap_normalizes_converted_embedding(self):
        backend = _make_backend(SwapperModel.SIMSWAP_256)
        converted = np.arange(1, 513, dtype=np.float32).reshape(1, 512)
        backend._converter = _StubConverter(converted)  # noqa: SLF001
        face = SimpleNamespace(embedding=np.ones(512, np.float32), kps=_KPS)
        backend.prepare_source(np.zeros((64, 64, 3), np.uint8), face)
        got = backend._source_input.ravel()  # noqa: SLF001
        assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-5)

    def test_uniface_uses_aligned_source_frame(self):
        backend = _make_backend(SwapperModel.UNIFACE_256)
        face = SimpleNamespace(embedding=None, kps=_KPS)  # no embedding needed
        backend.prepare_source(np.full((64, 64, 3), 128, np.uint8), face)
        assert backend._source_input.shape == (1, 3, 256, 256)  # noqa: SLF001


class TestGenericGet:
    def test_get_roundtrips_to_frame_shape(self):
        backend = _make_backend(SwapperModel.GHOST_2_256)
        backend._session = _IdentitySession()  # noqa: SLF001
        backend._source_input = np.zeros((1, 512), np.float32)  # noqa: SLF001
        frame = np.full((64, 64, 3), 90, np.uint8)
        face = SimpleNamespace(kps=_KPS)
        out = backend.get(frame, face)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_get_before_setup_raises(self):
        backend = _make_backend(SwapperModel.GHOST_2_256)
        with pytest.raises(RuntimeError, match="before setup"):
            backend.get(np.zeros((8, 8, 3), np.uint8), SimpleNamespace(kps=_KPS))
