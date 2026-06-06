"""Cache-storage concern for the realtime player.

Owns *where* processed frames live (the cache root + each session's subdir key),
the total-size cap + its eviction, and bulk clears. Extracted from
PlayerController (Phase 3.1) so the controller no longer carries the cache
filesystem policy inline.

The controller keeps the executor-coupled frame operations (invalidate /
rerender — they pause + resume the live session) and owns the
`cacheStorageStatsChanged` Qt signal; this helper takes a plain `on_changed`
callback the controller wires to that signal, so the helper stays Qt-free.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from sinner2.pipeline.cache_manager import CacheManager

if TYPE_CHECKING:
    from sinner2.config.source import Source
    from sinner2.config.target import Target
    from sinner2.pipeline.image_writer import ImageWriter
    from sinner2.pipeline.processor import Processor


def default_cache_root() -> Path:
    """Persistent processed-frame cache root used when the user has not
    set a custom path.

    `SINNER2_CACHE_DIR` env var overrides; defaults to `<install>/temp/`.
    Exposed (not `_` prefixed) so the GUI can show the default in tooltips
    and as the file-dialog start path.
    """
    env = os.environ.get("SINNER2_CACHE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "temp"


def _cache_key(
    source: Source,
    target: Target,
    chain: list[Processor],
    writer: ImageWriter,
    scale: float = 1.0,
) -> str:
    """Stable hash of (source path, target path, chain config, writer settings,
    processing scale).

    Two sessions with identical inputs land in the same cache subdirectory
    so processed frames carry over between runs. Different chain params,
    different image format, different quality, or a different processing
    scale go to a different subdirectory — keeps stale frames from a
    different configuration out of view and lets the user toggle formats or
    downscale without colliding with the full-resolution cache.
    """
    parts: list[str] = [
        str(source.path.resolve()),
        str(target.path.resolve()),
        writer.cache_key,
        f"scale={scale:.4f}",
    ]
    for p in chain:
        parts.append(p.name)
        # cache_identity() is the public contract for "what params affect my
        # output pixels" — replaces reaching into a private _params attribute.
        # Append only when non-empty so the hash is unchanged for processors
        # that carry no params (cache-continuity with the old reflection form).
        identity = getattr(p, "cache_identity", None)
        ident = identity() if callable(identity) else ""
        if ident:
            parts.append(ident)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class CacheController:
    """The realtime cache-storage policy: root, per-session subdir, size cap,
    eviction, and bulk clear. Qt-free — it reports storage changes (root switch,
    clear) through the `on_changed` callback the owning controller supplies."""

    def __init__(self, on_changed: Callable[[], None]) -> None:
        self._on_changed = on_changed
        # User-overridable cache root path; default when unset.
        self._cache_root: Path = default_cache_root()
        # Hard cap on total cache size in bytes. 0 → uncapped.
        self._size_cap_bytes: int = 0

    def cache_root(self) -> Path:
        return self._cache_root

    def cache_manager(self) -> CacheManager:
        """Fresh CacheManager for the current root. Cheap to construct; we
        rebuild rather than cache so a root change is immediately visible."""
        return CacheManager(self._cache_root)

    def set_cache_root(self, path: Path | None) -> None:
        """Switch the cache root. None reverts to the default. Does NOT migrate
        existing caches — only future sessions land in the new location. The
        current session keeps its existing path until teardown so we don't yank
        the rug out mid-write."""
        new_root = Path(path) if path is not None else default_cache_root()
        if new_root == self._cache_root:
            return
        self._cache_root = new_root
        self._on_changed()

    def cache_size_cap_bytes(self) -> int:
        return self._size_cap_bytes

    def set_cache_size_cap_bytes(self, max_bytes: int) -> None:
        """Hard cap on total cache size. 0 = uncapped. Enforced at the start of
        each session; not enforced live (would require periodic size walks)."""
        self._size_cap_bytes = max(0, max_bytes)

    def cache_dir_for(
        self,
        source: Source,
        target: Target,
        chain: list[Processor],
        writer: ImageWriter,
        scale: float = 1.0,
    ) -> Path:
        """The cache subdirectory this exact session config maps to under the
        current root (stable per source/target/chain/writer/scale)."""
        return self._cache_root / _cache_key(source, target, chain, writer, scale)

    def enforce_cap(self, manager: CacheManager, cache_dir: Path) -> None:
        """Evict old cache dirs down to the size cap, sparing the active
        session's dir. Without protect=, a cache-HIT reuse (cache_dir already
        the LRU-oldest under pressure) could be evicted moments before this
        session reattaches to it, forcing a needless full re-render (rank 29).
        touch_last_used refreshes its recency first (no-op for a brand-new dir)."""
        if self._size_cap_bytes <= 0:
            return
        manager.touch_last_used(cache_dir)
        manager.enforce_size_cap(self._size_cap_bytes, protect=[cache_dir])

    def clear_all(self, protect: list[Path]) -> tuple[int, int]:
        """Wipe every cache entry under the current root, sparing `protect`.
        Returns (entries_deleted, bytes_freed) for the UI to display."""
        result = self.cache_manager().clear_all(protect=protect)
        self._on_changed()
        return result
