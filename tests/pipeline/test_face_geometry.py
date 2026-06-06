"""Tests for the shared face-geometry primitives (feather_mask + paste_back).

These pin the two pieces extracted from the CodeFormer / plain-BFR backends.
The feather_mask + paste_back tests reproduce the legacy formulae inline and
assert byte-identity, so the extraction is provably behavior-preserving (the
backends previously carried their own copies of exactly this code).
"""
from __future__ import annotations

import cv2
import numpy as np

from sinner2.pipeline.face_geometry import feather_mask, paste_back


def _legacy_feather(size: int, pad_frac: float = 0.08) -> np.ndarray:
    """The formula the backends carried verbatim before the extraction."""
    m = np.zeros((size, size), np.float32)
    pad = int(size * pad_frac)
    m[pad:size - pad, pad:size - pad] = 1.0
    return cv2.GaussianBlur(m, (0, 0), sigmaX=size * 0.02)


def _legacy_paste(
    frame: np.ndarray, patch: np.ndarray, m: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """The paste-back the backends carried verbatim before the extraction."""
    h, w = frame.shape[:2]
    m_inv = cv2.invertAffineTransform(m)
    back = cv2.warpAffine(patch, m_inv, (w, h)).astype(np.float32)
    alpha = cv2.warpAffine(mask, m_inv, (w, h))[..., None]
    return (frame.astype(np.float32) * (1.0 - alpha) + back * alpha).astype(np.uint8)


def test_feather_mask_shape_matches_requested_size():
    assert feather_mask(512).shape == (512, 512)
    assert feather_mask(1024).shape == (1024, 1024)


def test_feather_mask_is_byte_identical_to_legacy_formula():
    # Pixel-identity gate: the shared mask must equal the formula the backends
    # carried (at both restorer resolutions in use).
    for size in (512, 1024):
        assert np.array_equal(feather_mask(size), _legacy_feather(size))


def test_paste_back_blends_center_keeps_corners():
    frame = np.full((100, 100, 3), 50, np.uint8)
    patch = np.full((512, 512, 3), 200, np.uint8)
    m = np.array([[5.12, 0, 0], [0, 5.12, 0]], np.float32)  # frame→512 scale
    out = paste_back(frame, patch, m, feather_mask(512))
    assert out.shape == (100, 100, 3)
    assert out.max() > 50   # patch blended into the center
    assert out.min() == 50  # corners untouched (feather mask is 0 there)


def test_paste_back_is_byte_identical_to_legacy_formula():
    # Pixel-identity gate over a non-trivial frame so warp + blend round-trips
    # are exercised, not just flat fills.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    patch = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    m = np.array([[5.12, 0, 1.3], [0, 5.12, -2.7]], np.float32)
    mask = feather_mask(512)
    assert np.array_equal(
        paste_back(frame, patch, m, mask), _legacy_paste(frame, patch, m, mask)
    )
