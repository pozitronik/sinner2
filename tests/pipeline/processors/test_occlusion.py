"""Tests for the occlusion-mask composite (model-agnostic, stub masker)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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
