"""The processing scale must partition the realtime frame cache: two scales
of the same source/target/chain land in DIFFERENT cache subdirs so a downscaled
run never reads full-resolution frames (or vice versa)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sinner2.gui.player_controller import _cache_key


def _fake(path: str):
    return SimpleNamespace(path=Path(path))


def test_cache_key_differs_by_scale():
    src, tgt = _fake("/s.png"), _fake("/t.mp4")
    writer = SimpleNamespace(cache_key="jpg-q95")
    full = _cache_key(src, tgt, [], writer, 1.0)
    half = _cache_key(src, tgt, [], writer, 0.5)
    assert full != half


def test_cache_key_stable_for_same_scale():
    src, tgt = _fake("/s.png"), _fake("/t.mp4")
    writer = SimpleNamespace(cache_key="jpg-q95")
    assert _cache_key(src, tgt, [], writer, 0.5) == _cache_key(
        src, tgt, [], writer, 0.5
    )
