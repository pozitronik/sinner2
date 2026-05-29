"""On-disk thumbnail cache.

Each thumbnail is a JPEG keyed by a hash of (resolved-path, mtime, size,
thumb_dimension). mtime + size invalidate when the source file is edited
or replaced; thumb_dimension prevents stale caches when the requested
size changes between runs. Sidecar JSON holds caption and pixel_count
for the source so the model can render them without re-opening the
original file.

JPEG over PNG: thumbnails are throwaway previews — 80% quality cuts
filesize ~4x for visually-identical results in a 200px tile. The cache
is bounded by file count, not bytes; old entries are pruned LRU-by-mtime
when the cap is exceeded.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

_CACHE_VERSION = 1
_MAX_ENTRIES_DEFAULT = 5000
# Don't run the full prune on every put — for bulk ingestion of a big
# folder this turns into N enumerations of the (growing) cache dir,
# each O(N) over the whole tree. Throttle to once every N puts so the
# amortised cost stays low. Triggered eagerly when the cache crosses
# the throttle threshold AND the upper-bound watermark below.
_PRUNE_EVERY_N_PUTS = 100
# Only run the prune when we're at least 10% over cap — gives the
# throttle slack without letting the cache balloon. Most ingestions
# will fall back to the throttle, not this watermark.
_PRUNE_WATERMARK_FACTOR = 1.1


@dataclass(frozen=True)
class ThumbnailMeta:
    """Captioning + dimensions stored alongside the cached pixel data."""

    caption: str
    pixel_count: int


class ThumbnailCache:
    """Maps source-file path → cached thumbnail JPEG + sidecar meta."""

    def __init__(
        self,
        root: Path,
        *,
        max_entries: int = _MAX_ENTRIES_DEFAULT,
    ) -> None:
        self._root = root
        self._max_entries = max_entries
        self._root.mkdir(parents=True, exist_ok=True)
        # Throttle state for _maybe_prune. Protected by the lock so
        # concurrent puts from N worker threads don't all decide to
        # prune at the same instant (the prune itself does its own
        # work without the lock — only the bookkeeping counter is
        # critical).
        self._prune_lock = threading.Lock()
        self._puts_since_prune = 0
        # Set by shutdown() so a put landing during app exit returns
        # without scheduling a slow prune that would block Python's
        # atexit thread-join. The previous launch's hang traced to
        # _maybe_prune iterating thousands of files via pathlib.glob
        # AFTER the generator had been shut down.
        self._shutdown = False

    @property
    def root(self) -> Path:
        return self._root

    def cache_key(self, source: Path, thumb_dim: int) -> str:
        """Stable hash including version + mtime + size so source-file
        edits or thumb-dimension changes invalidate naturally."""
        try:
            st = source.stat()
            mtime_ns = st.st_mtime_ns
            size = st.st_size
        except OSError:
            mtime_ns = 0
            size = 0
        material = "|".join(
            [
                str(_CACHE_VERSION),
                str(source.resolve()),
                str(mtime_ns),
                str(size),
                str(thumb_dim),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def _entry_paths(self, key: str) -> tuple[Path, Path]:
        """JPEG and sidecar-JSON paths for a given cache key."""
        return (self._root / f"{key}.jpg", self._root / f"{key}.json")

    def get(self, source: Path, thumb_dim: int) -> tuple[Path, ThumbnailMeta] | None:
        """Return (jpeg_path, meta) if cached, else None. Both files must
        exist; a half-written entry counts as miss."""
        key = self.cache_key(source, thumb_dim)
        jpeg_path, meta_path = self._entry_paths(key)
        if not (jpeg_path.is_file() and meta_path.is_file()):
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = ThumbnailMeta(
                caption=str(payload["caption"]),
                pixel_count=int(payload["pixel_count"]),
            )
        except (OSError, ValueError, KeyError):
            return None
        # Touch the JPEG so future LRU prunes treat this as recently used.
        # mtime-touch is cheap; we use mtime as the LRU clock instead of
        # tracking access times separately.
        try:
            now = os.path.getmtime(jpeg_path)
            os.utime(jpeg_path, (now, now))
        except OSError:
            pass
        return jpeg_path, meta

    def put(
        self,
        source: Path,
        thumb_dim: int,
        jpeg_bytes: bytes,
        meta: ThumbnailMeta,
    ) -> Path:
        """Store the thumbnail. Returns the on-disk JPEG path."""
        key = self.cache_key(source, thumb_dim)
        jpeg_path, meta_path = self._entry_paths(key)
        jpeg_path.write_bytes(jpeg_bytes)
        meta_path.write_text(
            json.dumps(
                {"caption": meta.caption, "pixel_count": meta.pixel_count},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._maybe_prune()
        return jpeg_path

    def shutdown(self) -> None:
        """Mark the cache as shutting down — _maybe_prune will return
        immediately on subsequent calls so an in-flight put() doesn't
        kick off a slow directory walk that blocks Python's atexit
        thread-join."""
        self._shutdown = True

    def _maybe_prune(self) -> None:
        """Trim oldest entries when over cap, throttled.

        Two layers of throttling so the bulk-ingest path (N workers
        all calling put() in tight succession against a cache that's
        just crossed the cap) doesn't degenerate into N concurrent
        full-directory walks:

          1) Per-put counter — only consider pruning every Nth put.
          2) Watermark — only actually prune when the cache is at
             least 10% over the cap, so we drop a chunk per visit
             instead of one entry at a time.

        Both checks are cheap (counter increment + one scandir count
        pass). The expensive sort + unlink only happens when we
        actually exceed the watermark.

        scandir is used over pathlib.glob because the latter builds a
        Path object per entry and triggers extra OS calls; scandir
        gives us mtime in the DirEntry cache for free.
        """
        if self._shutdown:
            return
        with self._prune_lock:
            self._puts_since_prune += 1
            if self._puts_since_prune < _PRUNE_EVERY_N_PUTS:
                return
            self._puts_since_prune = 0
        watermark = int(self._max_entries * _PRUNE_WATERMARK_FACTOR)
        # Single fast pass collecting (mtime, name) for sorting. Skip
        # non-jpg entries up-front (sidecar JSONs get deleted as a
        # side-effect when their JPEG is dropped, not on their own).
        entries: list[tuple[float, str]] = []
        try:
            with os.scandir(self._root) as it:
                for de in it:
                    if not de.name.endswith(".jpg"):
                        continue
                    try:
                        entries.append((de.stat().st_mtime, de.name))
                    except OSError:
                        continue
        except OSError:
            return
        if len(entries) <= watermark:
            return
        entries.sort()
        to_drop_count = len(entries) - self._max_entries
        for _mtime, name in entries[:to_drop_count]:
            if self._shutdown:
                # Belt-and-suspenders: bail mid-prune if shutdown was
                # signalled while we were sorting + deleting. Each
                # unlink is fast on local disk but we may have
                # thousands of them.
                return
            jpeg_path = self._root / name
            meta_path = jpeg_path.with_suffix(".json")
            for p in (jpeg_path, meta_path):
                try:
                    p.unlink()
                except OSError:
                    pass

    def clear(self) -> None:
        for p in self._root.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
