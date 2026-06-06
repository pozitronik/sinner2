"""Unit tests for CacheController — the cache-storage helper extracted from
PlayerController (Phase 3.1): root + per-session subdir, size cap + eviction,
bulk clear. Qt-free; storage changes are reported via the on_changed callback.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sinner2.gui.cache_controller import (
    CacheController,
    _cache_key,
    default_cache_root,
)


class _SpyManager:
    def __init__(self) -> None:
        self.touched = None
        self.cap = None
        self.protect = None
        self.enforce_called = False
        self.cleared_protect = None

    def touch_last_used(self, p) -> None:
        self.touched = p

    def enforce_size_cap(self, cap, protect=()) -> tuple[int, int]:
        self.enforce_called = True
        self.cap = cap
        self.protect = list(protect)
        return 0, 0

    def clear_all(self, protect=()) -> tuple[int, int]:
        self.cleared_protect = list(protect)
        return 3, 1234


def _cache(on_changed=lambda: None) -> CacheController:
    return CacheController(on_changed=on_changed)


# ---- size cap + eviction (rank 29 characterization) ----

def test_enforce_cap_protects_active_dir(tmp_path):
    # The active dir must be passed to protect= (so it can't be evicted) and its
    # recency refreshed first.
    cache = _cache()
    cache.set_cache_size_cap_bytes(1000)
    mgr = _SpyManager()
    cache_dir = tmp_path / "deadbeef"
    cache.enforce_cap(mgr, cache_dir)
    assert mgr.protect == [cache_dir]
    assert mgr.touched == cache_dir
    assert mgr.cap == 1000


def test_enforce_cap_noop_when_uncapped(tmp_path):
    cache = _cache()  # default cap 0 = uncapped
    mgr = _SpyManager()
    cache.enforce_cap(mgr, tmp_path / "x")
    assert mgr.enforce_called is False


def test_set_cache_size_cap_clamps_negative_to_zero():
    cache = _cache()
    cache.set_cache_size_cap_bytes(-5)
    assert cache.cache_size_cap_bytes() == 0
    cache.set_cache_size_cap_bytes(2048)
    assert cache.cache_size_cap_bytes() == 2048


# ---- root switching + the on_changed callback ----

def test_set_cache_root_updates_and_signals(tmp_path):
    fired = []
    cache = _cache(on_changed=lambda: fired.append(True))
    new_root = tmp_path / "custom"
    cache.set_cache_root(new_root)
    assert cache.cache_root() == new_root
    assert fired == [True]


def test_set_cache_root_noop_when_unchanged():
    fired = []
    cache = _cache(on_changed=lambda: fired.append(True))
    cache.set_cache_root(cache.cache_root())  # same root → no signal
    assert fired == []


def test_set_cache_root_none_reverts_to_default(tmp_path):
    cache = _cache()
    cache.set_cache_root(tmp_path / "custom")
    cache.set_cache_root(None)
    assert cache.cache_root() == default_cache_root()


# ---- bulk clear ----

def test_clear_all_passes_protect_and_signals(tmp_path, monkeypatch):
    fired = []
    cache = _cache(on_changed=lambda: fired.append(True))
    mgr = _SpyManager()
    monkeypatch.setattr(cache, "cache_manager", lambda: mgr)
    protect = [tmp_path / "active"]
    result = cache.clear_all(protect)
    assert result == (3, 1234)
    assert mgr.cleared_protect == protect
    assert fired == [True]


# ---- per-session cache dir ----

def test_cache_dir_for_is_root_plus_key(tmp_path):
    cache = _cache()
    cache.set_cache_root(tmp_path / "root")
    src = SimpleNamespace(path=Path("/s.png"))
    tgt = SimpleNamespace(path=Path("/t.mp4"))
    writer = SimpleNamespace(cache_key="jpg-q95")
    got = cache.cache_dir_for(src, tgt, [], writer, 1.0)
    assert got == cache.cache_root() / _cache_key(src, tgt, [], writer, 1.0)
