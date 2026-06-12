"""Tests for the occlusion-mask composite (model-agnostic, stub masker)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.pipeline.processors.occlusion import apply_occlusion


def _face():
    return SimpleNamespace(
        kps=np.array(
            [[40, 45], [60, 45], [50, 55], [42, 62], [58, 62]], np.float32
        )
    )


class _StubMasker:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def face_mask(self, _aligned) -> np.ndarray:
        return self._mask


class TestApplyOcclusion:
    def test_composites_swapped_in_face_region_only(self):
        before = np.full((100, 100, 3), 50, np.uint8)
        swapped = np.full((100, 100, 3), 200, np.uint8)
        mask = np.zeros((512, 512), np.float32)
        mask[180:330, 180:330] = 1.0  # central (face) region of the aligned crop
        out = apply_occlusion(before, swapped, _face(), _StubMasker(mask))
        assert out.shape == before.shape
        assert out.max() > 50   # swapped present where the mask mapped back
        assert out.min() == 50  # original kept in the corners (mask 0)

    def test_roi_blend_matches_legacy_full_frame_blend(self):
        # The ROI restriction must reproduce the original full-frame composite
        # (same warp + blend; outside the warped square alpha is 0 → `before`).
        # Tolerance contract as for paste_back: translated-matrix warps can
        # shift isolated pixels one interpolation tap (cv2 fixed-point math).
        import cv2

        from sinner2.pipeline.processors.occlusion import (
            _ALIGN_SIZE,
            _FEATHER_SIGMA,
            _align_matrix,
        )

        rng = np.random.default_rng(8)
        before = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)
        swapped = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)
        mask512 = np.zeros((512, 512), np.float32)
        mask512[100:400, 120:380] = 1.0
        face = SimpleNamespace(
            kps=np.array(
                [[200, 100], [240, 100], [220, 122], [205, 140], [236, 140]],
                np.float32,
            )
        )

        def legacy() -> np.ndarray:
            m = _align_matrix(face.kps)
            mask = cv2.GaussianBlur(mask512, (0, 0), sigmaX=_FEATHER_SIGMA)
            m_inv = cv2.invertAffineTransform(m)
            h, w = before.shape[:2]
            alpha = cv2.warpAffine(mask, m_inv, (w, h))[..., None]
            return (
                swapped.astype(np.float32) * alpha
                + before.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)

        out = apply_occlusion(before, swapped, face, _StubMasker(mask512))
        expected = legacy()
        diff = np.abs(out.astype(np.int16) - expected.astype(np.int16))
        assert (diff > 0).mean() < 0.005
        assert diff.max() <= 8
        # _ALIGN_SIZE pinned so the legacy reproduction can't silently drift.
        assert _ALIGN_SIZE == 512

    def test_inputs_not_mutated(self):
        before = np.full((100, 100, 3), 50, np.uint8)
        swapped = np.full((100, 100, 3), 200, np.uint8)
        b0, s0 = before.copy(), swapped.copy()
        mask = np.ones((512, 512), np.float32)
        apply_occlusion(before, swapped, _face(), _StubMasker(mask))
        assert np.array_equal(before, b0)
        assert np.array_equal(swapped, s0)

    def test_falls_back_to_swapped_on_error(self):
        before = np.full((10, 10, 3), 50, np.uint8)
        swapped = np.full((10, 10, 3), 200, np.uint8)

        class _Boom:
            def face_mask(self, _a):
                raise RuntimeError("parser down")

        out = apply_occlusion(before, swapped, _face(), _Boom())
        assert np.array_equal(out, swapped)


class TestOnnxParserMasker:
    """facefusion's BiSeNet ONNX parsers: contract pinned (512 RGB, /255 then
    ImageNet mean/std, NCHW in; 19-class logits out, argmax → _FACE_CLASSES),
    thread-safe (shared session → no swapper lock)."""

    class _SpySession:
        """Returns logits that put class 1 (skin) everywhere except a corner
        block of class 0 (background); records the input blob."""

        def __init__(self) -> None:
            self.blobs: list[np.ndarray] = []

        def get_inputs(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name="in")]

        def get_outputs(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name="out")]

        def run(self, _names, feeds):
            blob = feeds["in"]
            self.blobs.append(blob)
            logits = np.zeros((1, 19, 512, 512), np.float32)
            logits[0, 1] = 1.0       # skin wins everywhere...
            logits[0, 0, :64, :64] = 2.0  # ...except a background corner
            return [logits]

    def _masker(self, monkeypatch):
        from sinner2.pipeline import model_cache
        from sinner2.pipeline.processors.occlusion import OnnxParserMasker

        session = self._SpySession()
        monkeypatch.setattr(
            model_cache, "get_onnx_session", lambda *a, **k: session
        )
        # occlusion.py imports get_onnx_session lazily inside setup() from
        # model_cache, so patching the source module is enough.
        m = OnnxParserMasker("bisenet_resnet_34.onnx")
        m.setup()
        return m, session

    def test_thread_safe_flag(self):
        from sinner2.pipeline.processors.occlusion import (
            OcclusionMasker,
            OnnxParserMasker,
        )

        assert OnnxParserMasker.thread_safe is True
        assert OcclusionMasker.thread_safe is False

    def test_preprocessing_contract(self, monkeypatch):
        m, session = self._masker(monkeypatch)
        aligned = np.zeros((512, 512, 3), np.uint8)
        aligned[:, :, 2] = 255  # pure red in BGR
        m.face_mask(aligned)
        blob = session.blobs[0]
        assert blob.shape == (1, 3, 512, 512)
        assert blob.dtype == np.float32
        # BGR→RGB flip: the red channel must land in channel 0, normalized
        # (1.0 - 0.485) / 0.229; blue/green carry (0 - mean) / std.
        assert np.isclose(blob[0, 0, 0, 0], (1.0 - 0.485) / 0.229, atol=1e-5)
        assert np.isclose(blob[0, 2, 0, 0], (0.0 - 0.406) / 0.225, atol=1e-5)

    def test_mask_from_argmax_classes(self, monkeypatch):
        m, _ = self._masker(monkeypatch)
        mask = m.face_mask(np.zeros((512, 512, 3), np.uint8))
        assert mask.shape == (512, 512)
        assert mask[300, 300] == 1.0  # skin (class 1) → facial region
        assert mask[10, 10] == 0.0    # background corner → reverted

    def test_face_mask_before_setup_raises(self):
        from sinner2.pipeline.processors.occlusion import OnnxParserMasker

        with pytest.raises(RuntimeError, match="before setup"):
            OnnxParserMasker("x.onnx").face_mask(np.zeros((512, 512, 3), np.uint8))

    def test_release_evicts_session(self, monkeypatch):
        from sinner2.pipeline import model_cache

        m, _ = self._masker(monkeypatch)
        evicted: list[str] = []
        monkeypatch.setattr(
            model_cache, "release_onnx_session",
            lambda name, providers=None: evicted.append(name),
        )
        m.release()
        assert evicted == ["bisenet_resnet_34.onnx"]
        assert m._session is None  # noqa: SLF001


class TestBuildParserMasker:
    def test_torch_parsers_build_occlusion_masker(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            OcclusionMasker,
            build_parser_masker,
        )

        for p in (FaceParser.BISENET, FaceParser.PARSENET):
            assert isinstance(build_parser_masker(p), OcclusionMasker)

    def test_onnx_parsers_build_onnx_masker(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            OnnxParserMasker,
            build_parser_masker,
        )

        m = build_parser_masker(FaceParser.BISENET_ONNX_34)
        assert isinstance(m, OnnxParserMasker)
        assert m._model_file == "bisenet_resnet_34.onnx"  # noqa: SLF001
        m18 = build_parser_masker(FaceParser.BISENET_ONNX_18)
        assert m18._model_file == "bisenet_resnet_18.onnx"  # noqa: SLF001

    def test_parser_model_file_covers_all_parsers(self):
        from sinner2.pipeline.processors.occlusion import (
            FaceParser,
            parser_model_file,
        )

        files = {parser_model_file(p) for p in FaceParser}
        assert files == {
            "parsing_bisenet.pth", "parsing_parsenet.pth",
            "bisenet_resnet_34.onnx", "bisenet_resnet_18.onnx",
        }


class TestRelease:
    def test_release_frees_cuda_and_nulls_model(self, monkeypatch):
        from unittest.mock import MagicMock

        import torch

        from sinner2.pipeline.processors.occlusion import OcclusionMasker

        m = OcclusionMasker()
        m._model = MagicMock()  # noqa: SLF001
        m._device_is_cuda = True  # noqa: SLF001  (set in setup() after the fix)
        empties: list[int] = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empties.append(1))
        m.release()
        assert m._model is None  # noqa: SLF001
        assert empties == [1]  # VRAM handed back
