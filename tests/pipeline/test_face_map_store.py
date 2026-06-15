"""Tests for the per-target FaceMap sidecar store."""
from __future__ import annotations

from pathlib import Path

from sinner2.pipeline.face_map import FaceMap, Identity, normalize
from sinner2.pipeline.face_map_store import (
    delete_face_map,
    face_map_path,
    load_face_map,
    save_face_map,
)


def _map() -> FaceMap:
    return FaceMap(
        identities=(
            Identity("a", normalize([1, 0, 0]), source_path="/s/alice.png", occurrences=9),
        ),
        threshold=0.55,
    )


class TestPath:
    def test_keyed_by_target(self, tmp_path):
        a = face_map_path(Path("/videos/a.mp4"), tmp_path)
        b = face_map_path(Path("/videos/b.mp4"), tmp_path)
        assert a != b
        assert a.parent == tmp_path

    def test_stable_for_same_target(self, tmp_path):
        assert face_map_path(Path("/v/x.mp4"), tmp_path) == face_map_path(
            Path("/v/x.mp4"), tmp_path
        )


class TestSaveLoad:
    def test_round_trips(self, tmp_path):
        path = face_map_path(Path("/v/clip.mp4"), tmp_path)
        save_face_map(path, _map())
        loaded = load_face_map(path)
        assert loaded is not None
        assert loaded.threshold == 0.55
        assert loaded.identities[0].source_path == "/s/alice.png"
        assert loaded.identities[0].occurrences == 9

    def test_atomic_no_tmp_left(self, tmp_path):
        path = face_map_path(Path("/v/clip.mp4"), tmp_path)
        save_face_map(path, _map())
        assert path.is_file()
        assert not path.with_name(path.name + ".tmp").exists()

    def test_load_missing_is_none(self, tmp_path):
        assert load_face_map(tmp_path / "nope.json") is None

    def test_load_corrupt_is_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ not json", encoding="utf-8")
        assert load_face_map(path) is None


class TestDelete:
    def test_delete_existing(self, tmp_path):
        path = face_map_path(Path("/v/clip.mp4"), tmp_path)
        save_face_map(path, _map())
        assert delete_face_map(path) is True
        assert not path.exists()

    def test_delete_missing_is_false(self, tmp_path):
        assert delete_face_map(tmp_path / "nope.json") is False


class TestProgress:
    def test_round_trips(self, tmp_path):
        from sinner2.pipeline.face_map_store import (
            load_progress,
            progress_path,
            save_progress,
        )

        p = progress_path(Path("/v/clip.mp4"), tmp_path)
        save_progress(p, "15|[]", 30, 100)
        prog = load_progress(p)
        assert prog == {"signature": "15|[]", "scanned": 30, "total": 100}

    def test_progress_path_distinct_from_catalog(self, tmp_path):
        from sinner2.pipeline.face_map_store import face_map_path, progress_path

        t = Path("/v/clip.mp4")
        assert progress_path(t, tmp_path) != face_map_path(t, tmp_path)

    def test_load_missing_is_none(self, tmp_path):
        from sinner2.pipeline.face_map_store import load_progress

        assert load_progress(tmp_path / "nope.progress.json") is None
