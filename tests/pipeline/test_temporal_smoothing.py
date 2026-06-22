"""Tests for the geometry temporal-smoothing primitive (``smooth_geometry``)."""
from __future__ import annotations

import numpy as np
import pytest

from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace
from sinner2.pipeline.temporal_smoothing import smooth_geometry

# 5 keypoints around a centre; keypoint[2] (the nose) sits exactly on the centre,
# so a test can read the smoothed centre straight off ``kps[2]``.
_OFFSETS = np.array([[-10, -8], [10, -8], [0, 0], [-8, 10], [8, 10]], dtype=float)


def _kps(cx: float, cy: float) -> tuple[tuple[float, float], ...]:
    pts = _OFFSETS + np.array([cx, cy])
    return tuple((float(x), float(y)) for x, y in pts)


def _face(cx: float, cy: float, identity: str = "id0") -> GeomFace:
    return GeomFace(identity, (cx - 1, cy - 1, cx + 1, cy + 1), _kps(cx, cy))


def _geom_from_centres(
    centres: list[tuple[float, float]], identity: str = "id0"
) -> FrameGeometry:
    faces = {f: (_face(cx, cy, identity),) for f, (cx, cy) in enumerate(centres)}
    return FrameGeometry(faces, frame_count=len(centres))


class TestSmoothGeometry:
    def test_reduces_jitter_around_clean_motion(self) -> None:
        rng = np.random.default_rng(0)
        n = 120
        clean = np.stack([np.linspace(0.0, 50.0, n), np.zeros(n)], axis=1)
        noisy = clean + rng.normal(0.0, 2.0, size=(n, 2))
        geom = _geom_from_centres([(float(x), float(y)) for x, y in noisy])

        out = smooth_geometry(geom, window=9, strength=1.0)

        sm = np.array([out.faces_at(f)[0].kps[2] for f in range(n)])
        raw = np.array([geom.faces_at(f)[0].kps[2] for f in range(n)])
        assert np.var(sm - clean) < np.var(raw - clean)

    def test_strength_zero_and_window_one_are_identity(self) -> None:
        geom = _geom_from_centres([(float(i), 0.5 * i) for i in range(10)])
        assert smooth_geometry(geom, window=9, strength=0.0) is geom
        assert smooth_geometry(geom, window=1, strength=1.0) is geom

    def test_empty_geometry_is_returned_unchanged(self) -> None:
        empty = FrameGeometry.empty()
        assert smooth_geometry(empty, window=7, strength=1.0) is empty

    def test_gap_is_not_smoothed_across(self) -> None:
        # Identity present 0..4 (centre 0) then 20..24 (centre 100); a run-spanning
        # smoother would pull frame 4 toward 100. A gap-aware one must not.
        faces: dict[int, tuple[GeomFace, ...]] = {}
        for f in range(5):
            faces[f] = (_face(0.0, 0.0),)
        for f in range(20, 25):
            faces[f] = (_face(100.0, 0.0),)
        geom = FrameGeometry(faces, frame_count=25)

        out = smooth_geometry(geom, window=9, strength=1.0)

        assert out.faces_at(4)[0].kps[2][0] == 0.0
        assert out.faces_at(20)[0].kps[2][0] == pytest.approx(100.0)

    def test_identities_smoothed_independently(self) -> None:
        # A moves, B is static; A must not bleed into B's track.
        faces: dict[int, tuple[GeomFace, ...]] = {}
        for f in range(10):
            faces[f] = (_face(float(f), 0.0, "A"), _face(0.0, 100.0, "B"))
        geom = FrameGeometry(faces, frame_count=10)

        out = smooth_geometry(geom, window=5, strength=1.0)

        for f in range(10):
            assert out.faces_at(f)[1].kps[2] == pytest.approx((0.0, 100.0))

    def test_strength_blends_between_raw_and_smoothed(self) -> None:
        centres = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (10.0, 0.0), (0.0, 0.0)]
        geom = _geom_from_centres(centres)
        full = smooth_geometry(geom, window=3, strength=1.0).faces_at(2)[0].kps[2][0]
        half = smooth_geometry(geom, window=3, strength=0.5).faces_at(2)[0].kps[2][0]
        raw = geom.faces_at(2)[0].kps[2][0]
        # Half-strength sits between the raw value and the fully-smoothed value.
        assert min(raw, full) <= half <= max(raw, full)
        assert half != raw

    def test_metadata_is_preserved(self) -> None:
        base = _geom_from_centres([(float(i), 0.0) for i in range(8)])
        geom = FrameGeometry(base.faces, frame_count=8, refined=True, bake_size=(1920, 1080))

        out = smooth_geometry(geom, window=5, strength=1.0)

        assert out.frame_count == 8
        assert out.refined is True
        assert out.bake_size == (1920, 1080)
        # Per-frame face structure is unchanged (same count each frame).
        for f in range(8):
            assert len(out.faces_at(f)) == 1

    def test_static_face_is_unchanged(self) -> None:
        geom = _geom_from_centres([(5.0, 5.0)] * 12)
        out = smooth_geometry(geom, window=7, strength=1.0)
        for f in range(12):
            assert np.allclose(out.faces_at(f)[0].kps, geom.faces_at(f)[0].kps)
