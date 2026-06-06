"""Session-start cache eviction must spare the active session's dir (rank 29).

_build_session opens real files + loads models, so it isn't unit-testable; the
eviction is factored into _enforce_cache_cap, tested here in isolation via
object.__new__ (no Qt needed — the helper only reads _cache_size_cap_bytes).
"""
from __future__ import annotations

from sinner2.gui.player_controller import PlayerController


class _SpyManager:
    def __init__(self) -> None:
        self.touched = None
        self.cap = None
        self.protect = None
        self.enforce_called = False

    def touch_last_used(self, p) -> None:
        self.touched = p

    def enforce_size_cap(self, cap, protect=()) -> tuple[int, int]:
        self.enforce_called = True
        self.cap = cap
        self.protect = list(protect)
        return 0, 0


def test_enforce_cache_cap_protects_active_dir(tmp_path):
    # The active dir must be passed to protect= (so it can't be evicted) and its
    # recency refreshed first.
    ctrl = PlayerController.__new__(PlayerController)
    ctrl._cache_size_cap_bytes = 1000  # noqa: SLF001
    mgr = _SpyManager()
    cache_dir = tmp_path / "deadbeef"
    ctrl._enforce_cache_cap(mgr, cache_dir)  # noqa: SLF001
    assert mgr.protect == [cache_dir]
    assert mgr.touched == cache_dir
    assert mgr.cap == 1000


def test_enforce_cache_cap_noop_when_uncapped(tmp_path):
    ctrl = PlayerController.__new__(PlayerController)
    ctrl._cache_size_cap_bytes = 0  # noqa: SLF001
    mgr = _SpyManager()
    ctrl._enforce_cache_cap(mgr, tmp_path / "x")  # noqa: SLF001
    assert mgr.enforce_called is False
