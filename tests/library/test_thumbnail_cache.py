"""Tests for the disk thumbnail cache.

The cache is a v1 file-count-bounded LRU keyed by (resolved path, mtime,
size, thumb_dimension). The mtime + size component is what makes the
cache safe across source-file edits — a re-saved face image must NOT
return its old thumbnail.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from sinner2.library.thumbnail_cache import ThumbnailCache, ThumbnailMeta


@pytest.fixture
def cache(tmp_path: Path) -> ThumbnailCache:
    return ThumbnailCache(tmp_path / "cache")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    p = tmp_path / "src.jpg"
    p.write_bytes(b"original-pixel-data-stand-in")
    return p


class TestCacheKey:
    def test_changes_with_mtime(self, cache, source):
        k1 = cache.cache_key(source, 200)
        # Backdate the file so mtime changes; cache must invalidate.
        os.utime(source, (1000, 1000))
        k2 = cache.cache_key(source, 200)
        assert k1 != k2

    def test_changes_with_size(self, cache, source):
        k1 = cache.cache_key(source, 200)
        source.write_bytes(b"new-data-different-size-than-before")
        # Even at the same mtime resolution, the size component differs.
        k2 = cache.cache_key(source, 200)
        assert k1 != k2

    def test_changes_with_thumb_dim(self, cache, source):
        k1 = cache.cache_key(source, 200)
        k2 = cache.cache_key(source, 400)
        assert k1 != k2

    def test_same_inputs_same_key(self, cache, source):
        # Stable hash — calling twice gives the same result.
        assert cache.cache_key(source, 200) == cache.cache_key(source, 200)


class TestGetPut:
    def test_get_missing_returns_none(self, cache, source):
        assert cache.get(source, 200) is None

    def test_put_then_get_roundtrip(self, cache, source):
        jpeg = b"\xff\xd8\xff\xe0jpeg-bytes-stand-in"
        meta = ThumbnailMeta(caption="src.jpg [10x10]", pixel_count=100)
        cache.put(source, 200, jpeg, meta)
        got = cache.get(source, 200)
        assert got is not None
        jpeg_path, got_meta = got
        assert jpeg_path.read_bytes() == jpeg
        assert got_meta == meta

    def test_mtime_change_invalidates(self, cache, source):
        cache.put(source, 200, b"x", ThumbnailMeta(caption="x", pixel_count=1))
        # Wait a tick + edit the source — re-keying must miss.
        time.sleep(0.01)
        source.write_bytes(b"new content")
        os.utime(source, None)
        assert cache.get(source, 200) is None

    def test_thumb_dim_change_invalidates(self, cache, source):
        cache.put(source, 200, b"x", ThumbnailMeta(caption="x", pixel_count=1))
        assert cache.get(source, 400) is None

    def test_half_written_entry_is_miss(self, cache, source):
        # Sidecar missing simulates a partial write / external deletion.
        # We mustn't return a JPEG without its meta.
        cache.put(source, 200, b"x", ThumbnailMeta(caption="x", pixel_count=1))
        key = cache.cache_key(source, 200)
        meta_path = cache.root / f"{key}.json"
        meta_path.unlink()
        assert cache.get(source, 200) is None


class TestPrune:
    def test_prune_drops_oldest_when_well_over_cap(self, tmp_path):
        # Prune is throttled — only runs every Nth put AND when the
        # cache crosses the 10%-over watermark. Force both conditions
        # by writing enough entries and calling _maybe_prune directly.
        cache = ThumbnailCache(tmp_path / "cache", max_entries=3)
        meta = ThumbnailMeta(caption="x", pixel_count=1)
        sources: list[Path] = []
        for i in range(10):
            p = tmp_path / f"s{i}.jpg"
            p.write_bytes(b"x")
            sources.append(p)
            cache.put(p, 200, b"x", meta)
            # Force monotonic mtime so prune ordering is deterministic.
            time.sleep(0.01)
        # Throttle: 10 puts is below the 100-put threshold. Trigger a
        # prune directly — same call put() would make at the boundary.
        cache._puts_since_prune = 1000  # noqa: SLF001
        cache._maybe_prune()  # noqa: SLF001
        jpegs = list(cache.root.glob("*.jpg"))
        assert len(jpegs) == 3
        # The seven oldest sources must have been pruned.
        for i in range(7):
            assert cache.get(sources[i], 200) is None
        # Newest survives.
        assert cache.get(sources[9], 200) is not None

    def test_prune_no_op_when_under_watermark(self, tmp_path):
        # max=10, watermark=int(10*1.1)=11 → 11 entries are still
        # AT the watermark (not over), so prune is a no-op.
        cache = ThumbnailCache(tmp_path / "cache", max_entries=10)
        meta = ThumbnailMeta(caption="x", pixel_count=1)
        for i in range(11):
            p = tmp_path / f"s{i:02d}.jpg"
            p.write_bytes(b"x")
            cache.put(p, 200, b"x", meta)
        cache._puts_since_prune = 1000  # noqa: SLF001 — bypass throttle
        cache._maybe_prune()  # noqa: SLF001
        assert len(list(cache.root.glob("*.jpg"))) == 11

    def test_throttle_blocks_per_put_pruning(self, tmp_path):
        # Even with the cache way over cap, prune doesn't run on every
        # put — that was the shutdown-hang root cause (N workers each
        # walking the cache dir on every put). After 99 puts the next
        # put should NOT have pruned.
        cache = ThumbnailCache(tmp_path / "cache", max_entries=3)
        meta = ThumbnailMeta(caption="x", pixel_count=1)
        for i in range(99):
            p = tmp_path / f"s{i:02d}.jpg"
            p.write_bytes(b"x")
            cache.put(p, 200, b"x", meta)
        # 99 < 100 → no prune triggered yet. All 99 still on disk.
        assert len(list(cache.root.glob("*.jpg"))) == 99


class TestShutdown:
    """Shutdown signal short-circuits prune so a put in flight during
    app exit doesn't kick off a multi-second directory walk."""

    def test_shutdown_short_circuits_prune(self, tmp_path):
        cache = ThumbnailCache(tmp_path / "cache", max_entries=3)
        meta = ThumbnailMeta(caption="x", pixel_count=1)
        # Stage 1000+ entries so prune would be expensive.
        for i in range(20):
            p = tmp_path / f"s{i:02d}.jpg"
            p.write_bytes(b"x")
            cache.put(p, 200, b"x", meta)
        cache.shutdown()
        # Force the throttle counter past the threshold + try prune:
        # must be a no-op because shutdown flag is set.
        cache._puts_since_prune = 1000  # noqa: SLF001
        before = len(list(cache.root.glob("*.jpg")))
        cache._maybe_prune()  # noqa: SLF001
        after = len(list(cache.root.glob("*.jpg")))
        assert before == after  # no pruning happened
