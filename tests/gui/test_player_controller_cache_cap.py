"""PlayerController cache accessor(s).

The session-start size-cap eviction now lives in CacheController (see
tests/gui/test_cache_controller.py). What remains controller-level here is the
public session-cache-dir accessor (so main_window doesn't reach the private attr).
"""
from __future__ import annotations

from sinner2.gui.player_controller import PlayerController


def test_session_cache_dir_accessor(tmp_path):
    # Public accessor so main_window doesn't reach _controller._session_cache_dir.
    ctrl = PlayerController.__new__(PlayerController)
    ctrl._session_cache_dir = tmp_path / "abc"  # noqa: SLF001
    assert ctrl.session_cache_dir() == tmp_path / "abc"
