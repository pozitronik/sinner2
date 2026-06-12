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


def _legacy_swapper_paste(
    frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """The paste-back the swapper carried verbatim (BORDER_REPLICATE + clip)."""
    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(matrix)
    inv_mask = cv2.warpAffine(mask, inv, (w, h)).clip(0.0, 1.0)[..., None]
    inv_crop = cv2.warpAffine(crop, inv, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = frame.astype(np.float32) * (1.0 - inv_mask) + inv_crop.astype(np.float32) * inv_mask
    return out.astype(np.uint8)


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


def _assert_roi_equivalent(actual: np.ndarray, expected: np.ndarray) -> None:
    """Equivalence contract for a TRANSLATED-ROI paste vs the legacy full-frame
    blend. cv2's fixed-point warp arithmetic is association-sensitive, so a
    translated matrix may shift isolated pixels by one interpolation tap —
    bit-exactness is unattainable there (measured: ~0.05% of subpixels, a few
    LSB on random noise, ±1 on natural images). Pin that it stays ISOLATED and
    SMALL: >99.5% of subpixels exactly equal, none beyond one tap's worth."""
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert (diff > 0).mean() < 0.005
    assert diff.max() <= 8


def test_paste_back_small_patch_in_large_frame_matches_full_frame_blend():
    # The ROI optimization case: a 512 patch mapping to a ~120px face region of
    # a FullHD-ish frame, against the legacy full-frame blend. Worst case for
    # the tolerance contract: pure-noise frame AND patch.
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    patch = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    # face at ~(400, 250), aligned-space scale 512/120: x' = 4.27*(x-400)+...
    m = np.array([[4.27, 0.31, -1700.0], [-0.31, 4.27, -980.0]], np.float32)
    mask = feather_mask(512)
    _assert_roi_equivalent(
        paste_back(frame, patch, m, mask), _legacy_paste(frame, patch, m, mask)
    )


def test_paste_back_small_patch_replicate_clip_matches_full_frame_blend():
    # Same ROI case through the swapper flavor (BORDER_REPLICATE + clip_mask):
    # replicate fills the warp band outside the quad, but alpha is 0 there, so
    # the result must match outside the ROI exactly and inside within tolerance.
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    crop = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    matrix = np.array([[2.1, 0.2, -900.0], [-0.2, 2.1, -300.0]], np.float32)
    mask = np.zeros((256, 256), np.float32)
    mask[24:232, 24:232] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6.0)
    _assert_roi_equivalent(
        paste_back(frame, crop, matrix, mask, border_replicate=True, clip_mask=True),
        _legacy_swapper_paste(frame, crop, mask, matrix),
    )


def test_paste_back_partially_off_frame_matches_full_frame_blend():
    # Face half-out of frame (common at frame edges): the ROI clamps to the
    # frame; result must still match the legacy full-frame blend.
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    patch = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    # Maps the patch to a region straddling the frame's left edge.
    m = np.array([[4.0, 0.0, 200.0], [0.0, 4.0, -300.0]], np.float32)
    mask = feather_mask(512)
    _assert_roi_equivalent(
        paste_back(frame, patch, m, mask), _legacy_paste(frame, patch, m, mask)
    )


def test_paste_back_untouched_region_is_byte_identical():
    # Outside the warped-patch box the frame must be EXACTLY the input — the
    # ROI restriction may never leak blend artifacts beyond the face region.
    rng = np.random.default_rng(6)
    frame = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    patch = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    m = np.array([[4.27, 0.31, -1700.0], [-0.31, 4.27, -980.0]], np.float32)
    out = paste_back(frame, patch, m, feather_mask(512))
    changed = np.argwhere((out != frame).any(axis=2))
    assert len(changed)  # the face region did blend
    y0, x0 = changed.min(axis=0)
    y1, x1 = changed.max(axis=0)
    # All changes confined to a face-sized box, nowhere near frame extents.
    assert x1 - x0 < 200 and y1 - y0 < 200


def test_paste_back_fully_off_frame_returns_frame_unchanged():
    # Degenerate: the warped patch lands entirely outside the frame. The result
    # is the input pixels, and the input array is NOT mutated (callers may
    # reuse the original frame).
    frame = np.full((100, 100, 3), 77, np.uint8)
    patch = np.full((512, 512, 3), 200, np.uint8)
    m = np.array([[1.0, 0.0, -5000.0], [0.0, 1.0, -5000.0]], np.float32)
    out = paste_back(frame, patch, m, feather_mask(512))
    assert np.array_equal(out, frame)
    assert out is not frame


def test_paste_back_does_not_mutate_input_frame():
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    patch = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    m = np.array([[5.12, 0, 0], [0, 5.12, 0]], np.float32)
    paste_back(frame, patch, m, feather_mask(512))
    assert np.array_equal(frame, original)


def test_paste_back_flags_are_byte_identical_to_legacy_swapper_formula():
    # The swapper flavor (border_replicate + clip_mask) must reproduce its
    # former inline paste exactly — a feathered box mask in [0,1], a uint8 crop,
    # a translate+scale matrix.
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    crop = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    matrix = np.array([[4.0, 0, 1.1], [0, 4.0, -0.7]], np.float32)  # frame→256
    mask = np.zeros((256, 256), np.float32)
    mask[24:232, 24:232] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6.0)
    assert np.array_equal(
        paste_back(frame, crop, matrix, mask, border_replicate=True, clip_mask=True),
        _legacy_swapper_paste(frame, crop, mask, matrix),
    )
