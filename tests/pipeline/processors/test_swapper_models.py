"""Tests for the alternative-swapper registry + generic ONNX backend.

No real models are loaded — sessions/converters are stubbed. These verify the
pipeline MECHANICS (registry lookups, warp/mask/paste shapes, source-embedding
dispatch, normalization round-trip), not model quality."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.pipeline.face_geometry import paste_back
from sinner2.pipeline.processors.swapper_models import (
    FastPasteSwapper,
    GenericOnnxSwapper,
    SwapperModel,
    _box_mask,
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
        out = paste_back(
            frame, crop, matrix, _box_mask(256),
            border_replicate=True, clip_mask=True,
        )
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

    def test_release_evicts_model_and_converter(self, monkeypatch):
        import sinner2.pipeline.processors.swapper_models as sm

        evicted: list[str] = []
        monkeypatch.setattr(
            sm, "release_onnx_session", lambda name, providers=None: evicted.append(name)
        )
        backend = _make_backend(SwapperModel.GHOST_2_256)
        backend._session = _IdentitySession()  # noqa: SLF001
        backend._converter = object()  # noqa: SLF001
        backend.release()
        assert backend._session is None  # noqa: SLF001
        assert backend._converter is None  # noqa: SLF001
        # ghost evicts both its weights AND the crossface converter
        assert set(evicted) == {"ghost_2_256.onnx", "crossface_ghost.onnx"}

    def test_release_uniface_has_no_converter(self, monkeypatch):
        import sinner2.pipeline.processors.swapper_models as sm

        evicted: list[str] = []
        monkeypatch.setattr(
            sm, "release_onnx_session", lambda name, providers=None: evicted.append(name)
        )
        backend = _make_backend(SwapperModel.UNIFACE_256)
        backend._session = _IdentitySession()  # noqa: SLF001
        backend.release()
        assert evicted == ["uniface_256.onnx"]  # no converter companion


class _StubInswapper:
    """Real INSwapper surface: paste_back=False → (aligned crop, matrix);
    paste_back=True → the (legacy) internally-pasted frame."""

    def __init__(self, crop_size: int = 8) -> None:
        self.calls: list[bool] = []
        self._crop = np.full((crop_size, crop_size, 3), 200, np.uint8)
        # Identity warp translated to (1, 2): the crop lands at that offset.
        self._matrix = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -2.0]], np.float32)

    def get(self, img, target_face, source_face=None, paste_back=True):
        self.calls.append(paste_back)
        if paste_back:
            return np.full_like(img, 7)  # marker: the legacy internal paste
        return self._crop, self._matrix


class TestFastPasteSwapper:
    def test_paste_true_blends_crop_via_roi_paste(self):
        inner = _StubInswapper()
        fp = FastPasteSwapper(inner)
        frame = np.zeros((20, 20, 3), np.uint8)
        out = fp.get(frame, target_face=object(), source_face=object())
        # The inner model is asked for the crop ONLY — its internal paste
        # (the 113ms/frame full-frame blend) must never run.
        assert inner.calls == [False]
        assert out.shape == frame.shape
        # Crop center blended in at the matrix offset; far corner untouched.
        assert out[6, 5, 0] > 150
        assert out[19, 19].tolist() == [0, 0, 0]

    def test_paste_false_passes_through_crop_and_matrix(self):
        inner = _StubInswapper()
        fp = FastPasteSwapper(inner)
        crop, matrix = fp.get(
            np.zeros((20, 20, 3), np.uint8), object(), object(), paste_back=False
        )
        assert inner.calls == [False]
        assert crop.shape == (8, 8, 3)
        assert matrix.shape == (2, 3)

    def test_mask_built_once_per_crop_size(self):
        inner = _StubInswapper()
        fp = FastPasteSwapper(inner)
        frame = np.zeros((20, 20, 3), np.uint8)
        fp.get(frame, object(), object())
        first = fp._mask  # noqa: SLF001
        fp.get(frame, object(), object())
        assert fp._mask is first  # noqa: SLF001 — cached, not rebuilt

    def test_blend_matches_shared_roi_paste(self):
        # The adapter's output must be exactly paste_back(...) with the box
        # mask — same blend the 256px swappers use (one consistent look).
        inner = _StubInswapper()
        fp = FastPasteSwapper(inner)
        frame = np.random.default_rng(7).integers(
            0, 255, (20, 20, 3), dtype=np.uint8
        )
        out = fp.get(frame.copy(), object(), object())
        crop, matrix = inner._crop, inner._matrix  # noqa: SLF001
        expected = paste_back(
            frame.copy(), crop, matrix, _box_mask(8),
            border_replicate=True, clip_mask=True,
        )
        assert np.array_equal(out, expected)
