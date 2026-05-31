"""Tests for the shared frame-downscale helpers."""
from __future__ import annotations

import numpy as np

from sinner2.io.frame_resize import resize_frame, scaled_dims


class TestScaledDims:
    def test_half_scale(self):
        assert scaled_dims(1920, 1080, 0.5) == (960, 540)

    def test_unity_is_noop(self):
        assert scaled_dims(1921, 1081, 1.0) == (1921, 1081)

    def test_above_unity_is_noop_never_upscales(self):
        assert scaled_dims(1920, 1080, 1.5) == (1920, 1080)

    def test_result_is_forced_even(self):
        # 102 * 0.5 = 51 (odd) → floored to 50 for yuv420p compatibility.
        assert scaled_dims(102, 102, 0.5) == (50, 50)

    def test_tiny_scale_clamps_to_two(self):
        assert scaled_dims(3, 3, 0.1) == (2, 2)


class TestResizeFrame:
    def test_downscales_to_target(self):
        f = np.full((20, 20, 3), 128, dtype=np.uint8)
        out = resize_frame(f, 10, 10)
        assert out.shape == (10, 10, 3)

    def test_same_size_is_passthrough_same_object(self):
        f = np.full((20, 20, 3), 128, dtype=np.uint8)
        assert resize_frame(f, 20, 20) is f
