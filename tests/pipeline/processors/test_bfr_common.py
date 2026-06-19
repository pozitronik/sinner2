"""Tests for the shared BFR tensor I/O (normalize / denormalize).

These pin the [-1,1] RGB-NCHW convention shared by the CodeFormer and plain-BFR
ONNX backends. A stray scale/clip change that broke the round-trip would corrupt
every restored face, so the pixel-identity guard is the point — not coverage for
its own sake.
"""
from __future__ import annotations

import numpy as np

from sinner2.pipeline.processors.bfr_common import (
    denormalize_restored_face,
    normalize_aligned_face,
)


def _face(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


class TestNormalize:
    def test_shape_dtype_range_contiguous(self):
        chw = normalize_aligned_face(_face())
        assert chw.shape == (1, 3, 64, 64)
        assert chw.dtype == np.float32
        assert chw.min() >= -1.0 and chw.max() <= 1.0
        assert chw.flags["C_CONTIGUOUS"]  # ONNX runtime needs a contiguous buffer

    def test_black_and_white_map_to_range_extremes(self):
        assert np.allclose(normalize_aligned_face(np.zeros((4, 4, 3), np.uint8)), -1.0)
        assert np.allclose(normalize_aligned_face(np.full((4, 4, 3), 255, np.uint8)), 1.0)


class TestDenormalize:
    def test_shape_and_dtype(self):
        bgr = denormalize_restored_face(np.zeros((1, 3, 8, 8), np.float32))
        assert bgr.shape == (8, 8, 3)
        assert bgr.dtype == np.uint8

    def test_out_of_range_values_are_clipped(self):
        hi = np.full((1, 3, 2, 2), 2.0, np.float32)
        lo = np.full((1, 3, 2, 2), -2.0, np.float32)
        assert np.array_equal(denormalize_restored_face(hi), np.full((2, 2, 3), 255, np.uint8))
        assert np.array_equal(denormalize_restored_face(lo), np.zeros((2, 2, 3), np.uint8))


def test_round_trip_is_pixel_identical():
    """normalize → denormalize returns the original face exactly: each uint8
    value lands on an exact step of the [-1,1] grid, so there's no rounding loss
    and the BGR↔RGB swap is its own inverse."""
    face = _face(7)
    assert np.array_equal(denormalize_restored_face(normalize_aligned_face(face)), face)
