"""Unit tests for the pure playback-timing decisions (playback.py) — extracted
from RealtimeExecutor's playback tick so the sleep policy and the catch-up
fallback / backward-stutter guard are tested without spinning the executor."""
from __future__ import annotations

from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.realtime.playback import (
    compute_playback_sleep,
    select_fallback_index,
)


class TestComputePlaybackSleep:
    def test_not_playing_blocks_until_woken(self):
        assert compute_playback_sleep(False, PlaybackMode.FIXED_30, 30.0) is None

    def test_unlimited_uses_the_floor(self):
        assert compute_playback_sleep(True, PlaybackMode.UNLIMITED, 30.0) == 0.001

    def test_source_follows_timeline_fps(self):
        assert compute_playback_sleep(True, PlaybackMode.SOURCE, 25.0) == 1.0 / 25.0

    def test_fixed_is_30hz_regardless_of_source_fps(self):
        assert compute_playback_sleep(True, PlaybackMode.FIXED_30, 99.0) == 1.0 / 30


class TestSelectFallbackIndex:
    def test_none_when_no_fallback_available(self):
        assert select_fallback_index(None, 5) is None

    def test_uses_fallback_when_nothing_shown_yet(self):
        assert select_fallback_index(7, None) == 7

    def test_uses_fallback_when_not_older_than_shown(self):
        assert select_fallback_index(10, 8) == 10
        assert select_fallback_index(8, 8) == 8  # equal to shown is fine

    def test_holds_when_fallback_older_than_shown(self):
        # Backward-stutter guard: 5 < 8 already shown → hold the current frame.
        assert select_fallback_index(5, 8) is None
