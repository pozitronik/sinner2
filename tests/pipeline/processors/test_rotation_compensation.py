"""Tests for the rotation-compensation helper: roll measurement, the upright
transform, and the swap-and-composite path (with stubs)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sinner2.pipeline.processors import rotation_compensation as rc
from sinner2.pipeline.processors.face_swapper_types import RotationAngleSource


class TestComputeRoll:
    def test_keypoints_level_eyes_is_zero(self):
        face = SimpleNamespace(
            kps=np.array([[40, 50], [60, 50], [50, 60], [42, 65], [58, 65]], float)
        )
        assert rc.compute_roll(face, RotationAngleSource.KEYPOINTS) == pytest.approx(0.0)

    def test_keypoints_diagonal_is_45(self):
        face = SimpleNamespace(kps=np.array([[40, 50], [60, 70]], float))  # dx=dy=20
        assert rc.compute_roll(face, RotationAngleSource.KEYPOINTS) == pytest.approx(45.0)

    def test_pose_reads_roll_component(self):
        face = SimpleNamespace(pose=np.array([5.0, -3.0, 30.0]))
        assert rc.compute_roll(face, RotationAngleSource.POSE) == pytest.approx(30.0)

    def test_pose_falls_back_to_keypoints_when_absent(self):
        face = SimpleNamespace(kps=np.array([[0, 0], [10, 10]], float))
        assert rc.compute_roll(face, RotationAngleSource.POSE) == pytest.approx(45.0)

    def test_no_data_returns_zero(self):
        assert rc.compute_roll(SimpleNamespace(), RotationAngleSource.KEYPOINTS) == 0.0

    def test_baked_roll_wins_over_every_source(self):
        # D5: a detection-free geometry face carries baked_roll → used as-is,
        # ignoring pose/kps (a rebuilt face has no live pose estimate to read).
        face = SimpleNamespace(
            baked_roll=22.0,
            pose=np.array([0.0, 0.0, 99.0]),  # would say 99 if pose were read
            kps=np.array([[0, 0], [10, 10]], float),  # would say 45 from kps
        )
        assert rc.compute_roll(face, RotationAngleSource.POSE) == pytest.approx(22.0)
        assert rc.compute_roll(
            face, RotationAngleSource.KEYPOINTS
        ) == pytest.approx(22.0)

    def test_landmark_68_uses_eye_centre_line(self):
        lm = np.zeros((68, 2), np.float32)
        lm[36:42] = [40, 50]   # left eye centre
        lm[42:48] = [60, 70]   # right eye centre → dx=dy=20 → 45°
        face = SimpleNamespace(kps=np.array([[0, 0], [1, 0]], float))  # would be 0°
        assert rc.compute_roll(
            face, RotationAngleSource.LANDMARK_68, lm
        ) == pytest.approx(45.0)

    def test_landmark_68_falls_back_to_keypoints_when_absent(self):
        # Source is LANDMARK_68 but no 68 supplied → detector eye-line.
        face = SimpleNamespace(kps=np.array([[0, 0], [10, 10]], float))
        assert rc.compute_roll(
            face, RotationAngleSource.LANDMARK_68, None
        ) == pytest.approx(45.0)


class TestUprightMatrix:
    def test_levels_a_tilted_eye_line(self):
        roll, cx, cy, half = 30.0, 50.0, 50.0, 20.0
        rad = math.radians(roll)
        eye_l = (cx - half * math.cos(rad), cy - half * math.sin(rad))
        eye_r = (cx + half * math.cos(rad), cy + half * math.sin(rad))
        m = rc._upright_matrix(cx, cy, roll, 128)
        out = cv2.transform(np.array([[eye_l, eye_r]], np.float32), m)[0]
        # After uprighting, the two eyes share a y-coordinate (horizontal).
        assert out[0][1] == pytest.approx(out[1][1], abs=1e-3)


class _MarkSwapper:
    """Stub inswapper: paints a marker in the centre of whatever it's given."""

    def get(self, img, target, source, paste_back=True):
        out = img.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (w // 2 - 4, h // 2 - 4), (w // 2 + 4, h // 2 + 4), (255, 255, 255), -1)
        return out


class TestSwapWithUprighting:
    def _face(self):
        return SimpleNamespace(
            bbox=np.array([30, 30, 70, 70], float),
            kps=np.array([[40, 45], [60, 45], [50, 55], [42, 62], [58, 62]], float),
        )

    def test_composites_change_near_face_centre(self):
        frame = np.full((100, 100, 3), 50, np.uint8)
        analyser = SimpleNamespace(detect_only=lambda img: [])  # → fallback kps
        out = rc.swap_with_uprighting(
            frame, self._face(), object(), _MarkSwapper(), analyser,
            angle_deg=20.0, redetect=False,
        )
        assert out.shape == frame.shape
        # The marker (crop centre) maps back to the face centre (50, 50).
        assert out[50, 50, 0] != frame[50, 50, 0]
        # A far corner is untouched.
        assert out[5, 5, 0] == frame[5, 5, 0]

    def test_no_black_halo_near_frame_edge(self):
        # A face near the corner makes the upright crop extend off-frame, so the
        # rotate-back warp samples black there. The validity clamp must keep
        # those border zeros out of the composite — no pixel should go black.
        frame = np.full((100, 100, 3), 60, np.uint8)
        face = SimpleNamespace(
            bbox=np.array([4, 4, 44, 44], float),
            kps=np.array([[14, 18], [34, 18], [24, 28], [16, 36], [32, 36]], float),
        )
        analyser = SimpleNamespace(detect_only=lambda img: [])
        out = rc.swap_with_uprighting(
            frame, face, object(), _MarkSwapper(), analyser,
            angle_deg=25.0, redetect=False,
        )
        assert out.min() > 0  # no black halo bled in from the warp border

    def test_falls_back_to_direct_swap_on_error(self):
        frame = np.full((100, 100, 3), 50, np.uint8)
        seen_shapes: list = []

        class _Recorder:
            def get(self, img, target, source, paste_back=True):
                seen_shapes.append(img.shape[0])
                return img

        class _BoomAnalyser:
            def detect_only(self, img):
                raise RuntimeError("detector down")

        out = rc.swap_with_uprighting(
            frame, self._face(), object(), _Recorder(), _BoomAnalyser(),
            angle_deg=30.0, redetect=True,
        )
        assert out.shape == frame.shape
        assert seen_shapes == [100]  # only the full-frame fallback swap ran

    def test_redetect_uses_detection_only_not_full_pack(self):
        # The redetect must use the cheap det-only call (box+kps), NOT the full
        # buffalo_l pack (aux landmark/genderage/recognition nets) — that's the
        # whole point of the rotation perf fix.
        frame = np.full((100, 100, 3), 50, np.uint8)
        calls = {"detect_only": 0, "analyse_uncached": 0}

        class _Analyser:
            def detect_only(self, img):
                calls["detect_only"] += 1
                return []  # empty → fallback kps; we only assert which ran

            def analyse_uncached(self, img):
                calls["analyse_uncached"] += 1
                return []

        rc.swap_with_uprighting(
            frame, self._face(), object(), _MarkSwapper(), _Analyser(),
            angle_deg=20.0, redetect=True,
        )
        assert calls["detect_only"] == 1
        assert calls["analyse_uncached"] == 0
