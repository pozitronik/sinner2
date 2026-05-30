import json
import time
from pathlib import Path


from sinner2.pipeline.cache_manager import (
    CacheManager,
    CacheMeta,
    make_meta,
)


def _make_entry(root: Path, name: str, frame_bytes: list[int], with_meta: bool = True) -> Path:
    """Create a fake cache entry directory with the given per-frame byte sizes.

    Each frame is a file named NNNN.jpg padded with zero bytes to the
    requested size. Optionally writes a meta.json sidecar so we can test
    both the metadata path and legacy hash-only directories.
    """
    entry = root / name
    entry.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(frame_bytes):
        (entry / f"{i:08d}.jpg").write_bytes(b"\0" * n)
    if with_meta:
        meta = make_meta(
            source_path="/source.jpg",
            target_path=f"/{name}.mp4",
            target_frame_count=len(frame_bytes),
            image_format="jpg",
            image_quality=95,
            chain_summary="FaceSwapper",
        )
        (entry / "meta.json").write_text(meta.to_json(), encoding="utf-8")
    return entry


class TestCacheMetaJson:
    def test_round_trip(self):
        m = make_meta(
            source_path="/s.jpg",
            target_path="/t.mp4",
            target_frame_count=120,
            image_format="jpg",
            image_quality=95,
            chain_summary="FaceSwapper, FaceEnhancer",
        )
        back = CacheMeta.from_json(m.to_json())
        assert back == m

    def test_from_json_returns_none_for_garbage(self):
        assert CacheMeta.from_json("not json{") is None
        assert CacheMeta.from_json("[1, 2, 3]") is None

    def test_unknown_fields_are_tolerated(self):
        payload = json.dumps(
            {
                "schema": 99,
                "source_path": "/s",
                "target_path": "/t",
                "target_frame_count": 1,
                "image_format": "jpg",
                "image_quality": 95,
                "chain_summary": "x",
                "created_at": "2026-01-01T00:00:00+00:00",
                "last_used_at": "2026-01-01T00:00:00+00:00",
                "future_field_we_dont_know_about": True,
            }
        )
        m = CacheMeta.from_json(payload)
        assert m is not None
        assert m.image_format == "jpg"


class TestIsAvailable:
    def test_returns_true_for_writable_root(self, tmp_path: Path):
        m = CacheManager(tmp_path / "cache")
        assert m.is_available()
        # Probe file must be cleaned up.
        assert not (tmp_path / "cache" / ".sinner2_write_probe").exists()

    def test_returns_false_for_unwritable_path(self, tmp_path: Path):
        # Create a file where we want a directory — mkdir will fail.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir", encoding="utf-8")
        m = CacheManager(blocker / "subdir")
        assert m.is_available() is False


class TestListEntries:
    def test_empty_root(self, tmp_path: Path):
        m = CacheManager(tmp_path)
        assert m.list_entries() == []

    def test_returns_entries_with_sizes_and_frame_counts(self, tmp_path: Path):
        _make_entry(tmp_path, "aaaa1111", [100, 200, 300])
        _make_entry(tmp_path, "bbbb2222", [50])
        entries = CacheManager(tmp_path).list_entries()
        assert len(entries) == 2
        by_name = {e.path.name: e for e in entries}
        assert by_name["aaaa1111"].size_bytes >= 600  # frames + meta
        assert by_name["aaaa1111"].frame_count == 3
        assert by_name["bbbb2222"].frame_count == 1

    def test_includes_legacy_entries_without_meta(self, tmp_path: Path):
        _make_entry(tmp_path, "no_meta_dir", [100], with_meta=False)
        entries = CacheManager(tmp_path).list_entries()
        assert len(entries) == 1
        assert entries[0].meta is None
        assert entries[0].frame_count == 1

    def test_skips_non_directories(self, tmp_path: Path):
        _make_entry(tmp_path, "dir_entry", [100])
        (tmp_path / "stray.txt").write_text("ignore me", encoding="utf-8")
        entries = CacheManager(tmp_path).list_entries()
        assert len(entries) == 1


class TestTotalSizeAndFreeBytes:
    def test_total_size(self, tmp_path: Path):
        _make_entry(tmp_path, "a", [1024])
        _make_entry(tmp_path, "b", [2048])
        total = CacheManager(tmp_path).total_size_bytes()
        # Plus a bit of meta.json overhead per entry.
        assert total >= 3072

    def test_free_disk_bytes_returns_positive_on_real_path(self, tmp_path: Path):
        # Hard to make an exact assertion; just confirm it's a sane number.
        free = CacheManager(tmp_path).free_disk_bytes()
        assert free > 0


class TestDelete:
    def test_delete_entry(self, tmp_path: Path):
        entry = _make_entry(tmp_path, "to_delete", [100])
        mgr = CacheManager(tmp_path)
        assert mgr.delete_entry(entry) is True
        assert not entry.exists()

    def test_delete_refuses_paths_outside_root(self, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "file.txt").write_text("don't touch")
        mgr = CacheManager(tmp_path / "cache_root")
        assert mgr.delete_entry(outside) is False
        assert outside.exists()  # untouched


class TestClearAll:
    def test_clears_every_entry(self, tmp_path: Path):
        _make_entry(tmp_path, "a", [100])
        _make_entry(tmp_path, "b", [200])
        _make_entry(tmp_path, "c", [300])
        mgr = CacheManager(tmp_path)
        deleted, freed = mgr.clear_all()
        assert deleted == 3
        assert freed >= 600
        assert mgr.list_entries() == []

    def test_protect_spares_listed_paths(self, tmp_path: Path):
        a = _make_entry(tmp_path, "a", [100])
        b = _make_entry(tmp_path, "b", [200])
        mgr = CacheManager(tmp_path)
        deleted, _ = mgr.clear_all(protect=[a])
        assert deleted == 1
        assert a.exists()
        assert not b.exists()


class TestEnforceSizeCap:
    def test_no_eviction_when_under_cap(self, tmp_path: Path):
        _make_entry(tmp_path, "a", [100])
        mgr = CacheManager(tmp_path)
        deleted, freed = mgr.enforce_size_cap(max_bytes=10 * 1024 * 1024)
        assert deleted == 0
        assert freed == 0

    def test_evicts_oldest_first_under_pressure(self, tmp_path: Path):
        # Each entry is ~10 KB of frame + ~250 B of meta.json. Cap of
        # 15 KB fits exactly one — old should be evicted, new survives.
        # Sleep > 1s because meta.json's last_used_at has 1-second
        # resolution and ties on that field fall back to mtime, which
        # also has limited resolution on some filesystems.
        old = _make_entry(tmp_path, "old", [10_000])
        time.sleep(1.1)
        new = _make_entry(tmp_path, "new", [10_000])
        mgr = CacheManager(tmp_path)
        deleted, _ = mgr.enforce_size_cap(max_bytes=15_000)
        assert deleted == 1
        assert not old.exists()
        assert new.exists()

    def test_respects_protect(self, tmp_path: Path):
        old = _make_entry(tmp_path, "old", [1000])
        time.sleep(0.05)
        new = _make_entry(tmp_path, "new", [1000])
        # Protect the only candidate — nothing should be deleted even
        # though we're over the cap.
        mgr = CacheManager(tmp_path)
        deleted, _ = mgr.enforce_size_cap(max_bytes=500, protect=[old, new])
        assert deleted == 0
        assert old.exists()
        assert new.exists()

    def test_max_bytes_zero_is_uncapped(self, tmp_path: Path):
        _make_entry(tmp_path, "a", [10000])
        mgr = CacheManager(tmp_path)
        deleted, _ = mgr.enforce_size_cap(max_bytes=0)
        assert deleted == 0


class TestMetaIO:
    def test_write_then_list_round_trips(self, tmp_path: Path):
        entry = tmp_path / "abc123"
        entry.mkdir()
        meta = make_meta(
            source_path="/s.jpg",
            target_path="/t.mp4",
            target_frame_count=42,
            image_format="jpg",
            image_quality=80,
            chain_summary="FaceSwapper",
        )
        mgr = CacheManager(tmp_path)
        mgr.write_meta(entry, meta)
        entries = mgr.list_entries()
        assert len(entries) == 1
        loaded = entries[0].meta
        assert loaded is not None
        assert loaded.source_path == "/s.jpg"
        assert loaded.target_frame_count == 42

    def test_touch_last_used_updates_timestamp(self, tmp_path: Path):
        entry = _make_entry(tmp_path, "e", [100])
        mgr = CacheManager(tmp_path)
        meta_path = entry / "meta.json"
        first_text = meta_path.read_text(encoding="utf-8")
        time.sleep(1.1)  # iso seconds resolution
        mgr.touch_last_used(entry)
        second_text = meta_path.read_text(encoding="utf-8")
        first = json.loads(first_text)
        second = json.loads(second_text)
        assert first["created_at"] == second["created_at"]
        assert first["last_used_at"] != second["last_used_at"]

    def test_touch_is_noop_for_legacy_entries(self, tmp_path: Path):
        entry = _make_entry(tmp_path, "e", [100], with_meta=False)
        mgr = CacheManager(tmp_path)
        mgr.touch_last_used(entry)  # must not raise
        assert not (entry / "meta.json").exists()
