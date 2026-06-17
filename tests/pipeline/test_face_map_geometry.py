"""Tests for the per-frame face-geometry table + its NPZ sidecar."""
from __future__ import annotations

from pathlib import Path

from sinner2.pipeline.face_map_geometry import (
    FrameGeometry,
    GeomFace,
    delete_geometry,
    geometry_path,
    load_geometry,
    save_geometry,
)


def _kps(fn):
    return tuple((float(fn(i)[0]), float(fn(i)[1])) for i in range(5))


def _geom() -> FrameGeometry:
    return FrameGeometry(
        faces={
            0: (GeomFace("a", (0.0, 0.0, 4.0, 4.0), _kps(lambda i: (i, i))),),
            2: (
                GeomFace("a", (1.0, 1.0, 5.0, 5.0), _kps(lambda i: (i, 0))),
                GeomFace("b", (10.0, 10.0, 14.0, 14.0), _kps(lambda i: (0, i))),
            ),
        },
        frame_count=3,
    )


class TestFrameGeometry:
    def test_faces_at_and_counts(self):
        g = _geom()
        assert len(g.faces_at(0)) == 1
        assert len(g.faces_at(2)) == 2
        assert g.faces_at(1) == ()  # gap frame → no faces
        assert g.face_count() == 3
        assert g.is_empty() is False
        assert FrameGeometry.empty().is_empty() is True


class TestPath:
    def test_distinct_per_target_and_kind(self, tmp_path):
        from sinner2.pipeline.face_map_store import face_map_path, progress_path

        t = Path("/v/clip.mp4")
        gp = geometry_path(t, tmp_path)
        assert gp.suffix == ".npz"
        assert gp != face_map_path(t, tmp_path)
        assert gp != progress_path(t, tmp_path)
        assert geometry_path(Path("/v/a.mp4"), tmp_path) != geometry_path(
            Path("/v/b.mp4"), tmp_path
        )

    def test_same_file_via_redundant_path_shares_one_sidecar(self, tmp_path):
        import os

        (tmp_path / "sub").mkdir()
        real = tmp_path / "clip.mp4"
        real.write_bytes(b"x")
        detour = tmp_path / "sub" / os.pardir / "clip.mp4"
        assert geometry_path(detour, tmp_path) == geometry_path(real, tmp_path)


class TestSaveLoad:
    def test_round_trips(self, tmp_path):
        p = geometry_path(Path("/v/clip.mp4"), tmp_path)
        save_geometry(p, _geom())
        loaded = load_geometry(p)
        assert loaded is not None
        assert loaded.frame_count == 3
        assert loaded.face_count() == 3
        assert loaded.refined is False  # default (raw detector kps)
        f0 = loaded.faces_at(0)[0]
        assert f0.identity_id == "a"
        assert f0.bbox == (0.0, 0.0, 4.0, 4.0)
        assert len(f0.kps) == 5 and f0.kps[1] == (1.0, 1.0)
        assert sorted(gf.identity_id for gf in loaded.faces_at(2)) == ["a", "b"]

    def test_embeddings_round_trip(self, tmp_path):
        # A0: baked embeddings survive the NPZ round-trip (float16 tolerance).
        p = geometry_path(Path("/v/emb.mp4"), tmp_path)
        emb_a = tuple(float(x) for x in (0.5, 0.25, -0.5, 0.125, 0.75))
        g = FrameGeometry(
            faces={0: (GeomFace("a", (0.0, 0.0, 4.0, 4.0),
                                _kps(lambda i: (i, i)), emb_a),)},
            frame_count=1,
        )
        save_geometry(p, g)
        loaded = load_geometry(p)
        assert loaded is not None
        f0 = loaded.faces_at(0)[0]
        assert len(f0.embedding) == 5
        for got, want in zip(f0.embedding, emb_a):
            assert abs(got - want) < 1e-3

    def test_bake_size_round_trips(self, tmp_path):
        # The bake resolution survives the round-trip so the runtime can rescale;
        # absent (old sidecar) → None.
        p = geometry_path(Path("/v/bake.mp4"), tmp_path)
        g = FrameGeometry(
            faces={0: (GeomFace("a", (0.0, 0.0, 4.0, 4.0),
                                _kps(lambda i: (i, i))),)},
            frame_count=1,
            bake_size=(1920, 1080),
        )
        save_geometry(p, g)
        loaded = load_geometry(p)
        assert loaded is not None and loaded.bake_size == (1920, 1080)
        # A geometry saved without bake_size loads as None (back-compat).
        p2 = geometry_path(Path("/v/nobake.mp4"), tmp_path)
        save_geometry(p2, FrameGeometry(faces=g.faces, frame_count=1))
        loaded2 = load_geometry(p2)
        assert loaded2 is not None and loaded2.bake_size is None

    def test_roll_round_trips(self, tmp_path):
        # D5: a baked per-face roll survives the round-trip; absent → None.
        p = geometry_path(Path("/v/roll.mp4"), tmp_path)
        g = FrameGeometry(
            faces={0: (GeomFace("a", (0.0, 0.0, 4.0, 4.0),
                                _kps(lambda i: (i, i)), (), 12.5),)},
            frame_count=1,
        )
        save_geometry(p, g)
        loaded = load_geometry(p)
        assert loaded is not None
        assert abs(loaded.faces_at(0)[0].roll - 12.5) < 1e-3

    def test_legacy_no_roll_loads_none(self, tmp_path):
        # The _geom() fixture has no roll → faces load with roll is None.
        p = geometry_path(Path("/v/noroll.mp4"), tmp_path)
        save_geometry(p, _geom())
        loaded = load_geometry(p)
        assert loaded is not None
        assert loaded.faces_at(0)[0].roll is None

    def test_legacy_no_embeddings_loads_empty(self, tmp_path):
        # Back-compat: a sidecar with no baked embeddings (the _geom() fixture
        # builds id-only faces) loads with embedding == () → routes by id.
        p = geometry_path(Path("/v/legacy.mp4"), tmp_path)
        save_geometry(p, _geom())
        loaded = load_geometry(p)
        assert loaded is not None
        assert loaded.faces_at(0)[0].embedding == ()

    def test_partial_embeddings_are_dropped(self, tmp_path):
        # If ANY face lacks an embedding the array would be ragged → skip it
        # entirely (all faces fall back to id routing), never a half-baked mix.
        p = geometry_path(Path("/v/mixed.mp4"), tmp_path)
        g = FrameGeometry(
            faces={0: (
                GeomFace("a", (0.0, 0.0, 4.0, 4.0), _kps(lambda i: (i, i)),
                         (0.5, 0.25)),
                GeomFace("b", (1.0, 1.0, 5.0, 5.0), _kps(lambda i: (0, i))),  # none
            )},
            frame_count=1,
        )
        save_geometry(p, g)
        loaded = load_geometry(p)
        assert loaded is not None
        assert all(gf.embedding == () for gf in loaded.faces_at(0))

    def test_refined_flag_round_trips(self, tmp_path):
        p = geometry_path(Path("/v/r.mp4"), tmp_path)
        g = _geom()
        save_geometry(p, FrameGeometry(faces=g.faces, frame_count=3, refined=True))
        loaded = load_geometry(p)
        assert loaded is not None and loaded.refined is True

    def test_atomic_no_tmp_left(self, tmp_path):
        p = geometry_path(Path("/v/clip.mp4"), tmp_path)
        save_geometry(p, _geom())
        assert p.is_file()
        assert not p.with_name(p.name + ".tmp").exists()

    def test_empty_round_trips(self, tmp_path):
        p = geometry_path(Path("/v/e.mp4"), tmp_path)
        save_geometry(p, FrameGeometry(faces={}, frame_count=10))
        loaded = load_geometry(p)
        assert loaded is not None
        assert loaded.is_empty() and loaded.frame_count == 10

    def test_load_missing_is_none(self, tmp_path):
        assert load_geometry(tmp_path / "nope.geometry.npz") is None

    def test_load_corrupt_is_none(self, tmp_path):
        p = tmp_path / "bad.geometry.npz"
        p.write_bytes(b"not an npz at all")
        assert load_geometry(p) is None


class TestDelete:
    def test_delete_existing_then_missing(self, tmp_path):
        p = geometry_path(Path("/v/clip.mp4"), tmp_path)
        save_geometry(p, _geom())
        assert delete_geometry(p) is True
        assert not p.exists()
        assert delete_geometry(p) is False
