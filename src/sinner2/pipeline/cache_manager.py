"""On-disk cache enumeration, stats, and cleanup.

Each cache entry is a directory under the cache root, named with a
sha256-hash of (source path, target path, chain config, image writer
settings). Each directory contains processed frame files plus an
optional `meta.json` written at session start with human-readable
fields the management UI displays.

Entries without `meta.json` (legacy hash-only dirs from earlier
builds, or partially-written ones whose meta was lost) are still
enumerated; the management UI shows them as "unknown" with mtime + size.

This module deliberately does no caching — it stats the filesystem on
each call. Cache management is an infrequent action triggered by user
intent (open the panel, click clear, change cap); the cost of a stat
walk is not a hot-path concern.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_META_FILENAME = "meta.json"
# Schema-version on the metadata so older shapes can be migrated by the
# reader without breaking forward-compat. Bump when adding required fields.
_META_VERSION = 1


@dataclass(frozen=True)
class CacheMeta:
    """Sidecar metadata written into each cache directory at session start.

    Friendly fields the UI surfaces to the user — none of these affect
    the cache directory's primary key (which is the dir name itself, a
    sha256 hash of all variant-affecting inputs).
    """

    source_path: str
    target_path: str
    target_frame_count: int
    image_format: str
    image_quality: int
    chain_summary: str
    created_at_iso: str
    last_used_at_iso: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": _META_VERSION,
                "source_path": self.source_path,
                "target_path": self.target_path,
                "target_frame_count": self.target_frame_count,
                "image_format": self.image_format,
                "image_quality": self.image_quality,
                "chain_summary": self.chain_summary,
                "created_at": self.created_at_iso,
                "last_used_at": self.last_used_at_iso,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "CacheMeta | None":
        try:
            d = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        # Unknown future fields ignored; missing required ones → None.
        try:
            return cls(
                source_path=str(d["source_path"]),
                target_path=str(d["target_path"]),
                target_frame_count=int(d.get("target_frame_count", 0)),
                image_format=str(d.get("image_format", "")),
                image_quality=int(d.get("image_quality", 0)),
                chain_summary=str(d.get("chain_summary", "")),
                created_at_iso=str(d.get("created_at", "")),
                last_used_at_iso=str(d.get("last_used_at", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class CacheEntry:
    """One on-disk cache directory plus stats. `meta` is None for legacy
    entries that predate the meta.json sidecar — those still display
    (with dir name + size + mtime) so the user can manage them."""

    path: Path
    size_bytes: int
    frame_count: int
    mtime_epoch: float
    meta: CacheMeta | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dir_size_and_count(path: Path) -> tuple[int, int]:
    """Sum file sizes and count frame files (anything that's not meta.json)."""
    total = 0
    frames = 0
    try:
        for f in path.iterdir():
            if not f.is_file():
                continue
            try:
                size = f.stat().st_size
            except OSError:
                continue
            total += size
            if f.name != _META_FILENAME:
                frames += 1
    except OSError:
        pass
    return total, frames


class CacheManager:
    def __init__(self, cache_root: Path) -> None:
        self._root = Path(cache_root)

    @property
    def root(self) -> Path:
        return self._root

    def is_available(self) -> bool:
        """True iff the cache root exists (or can be created) and writes work.

        Used at session start to decide whether to fall back to memory-only
        when the cache root is on a removable / unavailable volume.
        """
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        probe = self._root / ".sinner2_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    def list_entries(self) -> list[CacheEntry]:
        if not self._root.is_dir():
            return []
        out: list[CacheEntry] = []
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            size, frames = _dir_size_and_count(child)
            try:
                mtime = child.stat().st_mtime
            except OSError:
                mtime = 0.0
            meta_path = child / _META_FILENAME
            meta: CacheMeta | None = None
            if meta_path.is_file():
                try:
                    meta = CacheMeta.from_json(
                        meta_path.read_text(encoding="utf-8")
                    )
                except OSError:
                    meta = None
            out.append(
                CacheEntry(
                    path=child,
                    size_bytes=size,
                    frame_count=frames,
                    mtime_epoch=mtime,
                    meta=meta,
                )
            )
        return out

    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self.list_entries())

    def free_disk_bytes(self) -> int:
        """Free bytes on the volume hosting the cache root.

        Returns 0 if the root doesn't exist yet or the call fails (e.g.
        removable drive offline). Caller treats 0 as "unknown / unavailable".
        """
        target = self._root if self._root.is_dir() else self._root.parent
        try:
            return shutil.disk_usage(target).free
        except (OSError, ValueError):
            return 0

    def delete_entry(self, path: Path) -> bool:
        """Remove one cache directory. Returns False if it isn't a child
        of our root (sanity check) or the rmtree fails. We never delete
        outside the configured cache root."""
        try:
            resolved = path.resolve()
            resolved.relative_to(self._root.resolve())
        except (ValueError, OSError):
            return False
        try:
            shutil.rmtree(resolved)
            return True
        except OSError:
            return False

    def entry_paths(self) -> list[Path]:
        """The cache-entry directories under the root, WITHOUT walking them for
        sizes — fast, for a bulk wipe or a count."""
        if not self._root.is_dir():
            return []
        return [c for c in sorted(self._root.iterdir()) if c.is_dir()]

    def clear_all(self, protect: Iterable[Path] = ()) -> int:
        """Remove every cache-entry directory (rmtree), sparing any in `protect`;
        return how many were deleted. Deliberately does NOT measure sizes first —
        a bulk wipe shouldn't pay to walk every file it's about to delete (that
        size walk hung the UI on large caches). It just drops the directories.
        """
        protected = {Path(p).resolve() for p in protect}
        deleted = 0
        for entry_dir in self.entry_paths():
            try:
                if entry_dir.resolve() in protected:
                    continue
            except OSError:
                continue
            if self.delete_entry(entry_dir):
                deleted += 1
        return deleted

    def enforce_size_cap(
        self,
        max_bytes: int,
        protect: Iterable[Path] = (),
    ) -> tuple[int, int]:
        """Delete oldest entries (by last_used_at if meta present, otherwise
        mtime) until total cache size is under `max_bytes`. Returns
        (entries_deleted, bytes_freed). Entries in `protect` are skipped
        — typically just the currently-active session."""
        if max_bytes <= 0:
            return 0, 0
        protected = {Path(p).resolve() for p in protect}
        entries = self.list_entries()
        # Base the budget ONLY on deletable (non-protected) bytes. Counting the
        # protected entry's bytes in `total` while subtracting only deleted
        # candidates' sizes over-evicts (and, when the protected dir alone
        # exceeds the cap, wipes every evictable entry without ever satisfying
        # it). The protected (active) session dir is excluded from the budget.
        candidates = [e for e in entries if e.path.resolve() not in protected]
        total = sum(e.size_bytes for e in candidates)
        if total <= max_bytes:
            return 0, 0
        # Oldest first: prefer last_used_at from meta, else fall back to
        # filesystem mtime.
        candidates.sort(key=_entry_age_key)
        deleted = 0
        freed = 0
        for entry in candidates:
            if total <= max_bytes:
                break
            size = entry.size_bytes
            if self.delete_entry(entry.path):
                deleted += 1
                freed += size
                total -= size
        return deleted, freed

    def write_meta(self, entry_path: Path, meta: CacheMeta) -> None:
        """Write meta.json into the entry directory. Creates the dir if
        needed; tolerates concurrent writers (last-writer-wins is fine
        because all fields except last_used_at are stable for a given
        cache key)."""
        entry_path.mkdir(parents=True, exist_ok=True)
        target = entry_path / _META_FILENAME
        try:
            target.write_text(meta.to_json(), encoding="utf-8")
        except OSError:
            pass

    def touch_last_used(self, entry_path: Path) -> None:
        """Update last_used_at on an existing meta.json without changing
        other fields. No-op if the file doesn't exist (legacy entries)."""
        target = entry_path / _META_FILENAME
        if not target.is_file():
            return
        try:
            existing = CacheMeta.from_json(target.read_text(encoding="utf-8"))
        except OSError:
            return
        if existing is None:
            return
        new = CacheMeta(
            source_path=existing.source_path,
            target_path=existing.target_path,
            target_frame_count=existing.target_frame_count,
            image_format=existing.image_format,
            image_quality=existing.image_quality,
            chain_summary=existing.chain_summary,
            created_at_iso=existing.created_at_iso,
            last_used_at_iso=_now_iso(),
        )
        try:
            target.write_text(new.to_json(), encoding="utf-8")
        except OSError:
            pass


def make_meta(
    *,
    source_path: str,
    target_path: str,
    target_frame_count: int,
    image_format: str,
    image_quality: int,
    chain_summary: str,
) -> CacheMeta:
    """Construct a fresh CacheMeta with created_at and last_used_at = now."""
    now = _now_iso()
    return CacheMeta(
        source_path=source_path,
        target_path=target_path,
        target_frame_count=target_frame_count,
        image_format=image_format,
        image_quality=image_quality,
        chain_summary=chain_summary,
        created_at_iso=now,
        last_used_at_iso=now,
    )


def _entry_age_key(entry: CacheEntry) -> float:
    """Sort key for eviction: oldest first.

    Prefer last_used_at from meta (parsed back to epoch) and fall back to
    the filesystem mtime. Entries without a parseable last_used_at sort
    to the front (treated as 'oldest') so legacy / corrupt entries get
    evicted first under pressure.
    """
    if entry.meta is None or not entry.meta.last_used_at_iso:
        return entry.mtime_epoch
    try:
        return datetime.fromisoformat(entry.meta.last_used_at_iso).timestamp()
    except (ValueError, TypeError):
        return entry.mtime_epoch
