"""Tests for the 2dfan4 face landmarker — crop/normalize contract, output
decode, 68->5 mapping, roll. Stub session; no model/weights."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sinner2.pipeline.processors.landmarker import (
    FaceLandmarker,
    landmark_68_to_5,
    roll_from_landmark_68,
)


def _identity_lm68() -> np.ndarray:
    # 68 points on a regular grid — distinct, deterministic.
    pts = np.zeros((68, 2), np.float32)
    pts[:, 0] = np.arange(68)
    pts[:, 1] = np.arange(68) * 2
    return pts


class TestLandmark68To5:
    def test_maps_to_arcface_five_points(self):
        lm = np.zeros((68, 2), np.float32)
        lm[36:42] = [[10, 20]] * 6   # left eye block → centre (10,20)
        lm[42:48] = [[30, 20]] * 6   # right eye block → centre (30,20)
        lm[30] = [20, 25]            # nose tip
        lm[48] = [12, 40]            # left mouth corner
        lm[54] = [28, 40]            # right mouth corner
        five = landmark_68_to_5(lm)
        assert five.shape == (5, 2)
        np.testing.assert_allclose(five[0], [10, 20])
        np.testing.assert_allclose(five[1], [30, 20])
        np.testing.assert_allclose(five[2], [20, 25])
        np.testing.assert_allclose(five[3], [12, 40])
        np.testing.assert_allclose(five[4], [28, 40])


class TestRollFromLandmark68:
    def test_level_eyes_zero_roll(self):
        lm = np.zeros((68, 2), np.float32)
        lm[36:42] = [[10, 50]] * 6
        lm[42:48] = [[40, 50]] * 6   # same y → 0°
        assert roll_from_landmark_68(lm) == pytest.approx(0.0, abs=1e-4)

    def test_tilted_eyes_positive_roll(self):
        lm = np.zeros((68, 2), np.float32)
        lm[36:42] = [[10, 50]] * 6
        lm[42:48] = [[40, 80]] * 6   # right eye lower → +45°
        assert roll_from_landmark_68(lm) == pytest.approx(45.0, abs=1e-3)


class _StubSession:
    """Returns a fixed (landmark, heatmap) pair and records the input blob."""

    def __init__(self, lm64: np.ndarray, heatmap: np.ndarray) -> None:
        self._lm = lm64
        self._heatmap = heatmap
        self.blobs: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="landmark"), SimpleNamespace(name="heatmap")]

    def run(self, _names, feeds):
        self.blobs.append(feeds["input"])
        return [self._lm, self._heatmap]


def _make(monkeypatch, lm64, heatmap):
    from sinner2.pipeline import model_cache

    session = _StubSession(lm64, heatmap)
    monkeypatch.setattr(model_cache, "get_onnx_session", lambda *a, **k: session)
    lmk = FaceLandmarker()
    lmk.setup()
    return lmk, session


class TestDetect68:
    def test_thread_safe(self):
        assert FaceLandmarker.thread_safe is True

    def test_input_contract(self, monkeypatch):
        # 256x256, NCHW, /255, BGR passthrough (no RGB swap).
        lm64 = np.zeros((1, 68, 3), np.float32)
        heatmap = np.zeros((1, 68, 64, 64), np.float32)
        lmk, session = _make(monkeypatch, lm64, heatmap)
        frame = np.full((400, 400, 3), 100, np.uint8)
        lmk.detect_68(frame, [100, 100, 200, 200])
        blob = session.blobs[0]
        assert blob.shape == (1, 3, 256, 256)
        assert blob.dtype == np.float32
        assert blob.max() <= 1.0  # /255 normalized

    def test_decode_scales_and_inverts_to_frame_coords(self, monkeypatch):
        # A landmark at heatmap (32,32) is the crop centre (32/64*256 = 128);
        # with an identity-ish box mapping it inverts back near the box centre.
        lm64 = np.zeros((1, 68, 3), np.float32)
        lm64[0, :, 0] = 32.0  # x in 64-space → 128 in 256-space (crop centre)
        lm64[0, :, 1] = 32.0
        heatmap = np.full((1, 68, 64, 64), 0.5, np.float32)
        lmk, _ = _make(monkeypatch, lm64, heatmap)
        # A box centred at (250,250) — all points should land near there.
        lm, score = lmk.detect_68(np.zeros((500, 500, 3), np.uint8),
                                  [200, 200, 300, 300])
        assert lm.shape == (68, 2)
        np.testing.assert_allclose(lm[0], [250, 250], atol=1.0)

    def test_score_interpolated_from_heatmap_peak(self, monkeypatch):
        lm64 = np.zeros((1, 68, 3), np.float32)
        # Peak 0.9 → interp([0,0.9]→[0,1]) = 1.0; peak 0.45 → 0.5.
        heatmap = np.full((1, 68, 64, 64), 0.45, np.float32)
        lmk, _ = _make(monkeypatch, lm64, heatmap)
        _, score = lmk.detect_68(np.zeros((500, 500, 3), np.uint8),
                                 [200, 200, 300, 300])
        assert score == pytest.approx(0.5, abs=1e-3)

    def test_refine_kps_returns_five_points(self, monkeypatch):
        lm64 = np.zeros((1, 68, 3), np.float32)
        lm64[0, :, 0] = np.arange(68)
        lm64[0, :, 1] = np.arange(68)
        heatmap = np.full((1, 68, 64, 64), 0.9, np.float32)
        lmk, _ = _make(monkeypatch, lm64, heatmap)
        kps, score = lmk.refine_kps(np.zeros((500, 500, 3), np.uint8),
                                    [200, 200, 300, 300])
        assert kps.shape == (5, 2)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_detect_before_setup_raises(self):
        with pytest.raises(RuntimeError, match="before setup"):
            FaceLandmarker().detect_68(np.zeros((10, 10, 3), np.uint8),
                                       [0, 0, 5, 5])

    def test_release_evicts_session(self, monkeypatch):
        from sinner2.pipeline import model_cache

        lm64 = np.zeros((1, 68, 3), np.float32)
        heatmap = np.zeros((1, 68, 64, 64), np.float32)
        lmk, _ = _make(monkeypatch, lm64, heatmap)
        evicted: list[str] = []
        monkeypatch.setattr(
            model_cache, "release_onnx_session",
            lambda name, providers=None: evicted.append(name),
        )
        lmk.release()
        assert evicted == ["2dfan4.onnx"]
        assert lmk._session is None  # noqa: SLF001
