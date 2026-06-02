from unittest.mock import MagicMock

import pytest

from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.skip_strategy import (
    BestEffortStrategy,
    FrameSkipStrategy,
    SkipDecision,
    SyncedStrategy,
)


def _zero_metrics() -> BufferMetrics:
    return BufferMetrics(
        frame_lag=0,
        time_lag_s=0.0,
        display_frame_lag=0,
        display_time_lag_s=0.0,
        current_frame_miss=0,
        memory_used_bytes=0,
        cache_hit_ratio=0.0,
    )


class TestSkipDecision:
    def test_can_signal_idle(self):
        d = SkipDecision(next_frame=None)
        assert d.next_frame is None

    def test_carries_frame_index(self):
        d = SkipDecision(next_frame=42)
        assert d.next_frame == 42

    def test_is_frozen(self):
        d = SkipDecision(next_frame=0)
        with pytest.raises(Exception):
            d.next_frame = 1  # type: ignore[misc]


class TestBestEffortStrategy:
    def test_compliant_with_protocol(self):
        assert isinstance(BestEffortStrategy(), FrameSkipStrategy)

    def test_first_call_returns_zero_when_initialized_negative(self):
        s = BestEffortStrategy()
        d = s.decide(
            last_submitted=-1,
            last_completed=-1,
            timeline=MagicMock(),
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 0

    def test_always_advances_by_one(self):
        s = BestEffortStrategy()
        for last in [0, 5, 100, 10000]:
            d = s.decide(
                last_submitted=last,
                last_completed=last - 1,
                timeline=MagicMock(),
                metrics=_zero_metrics(),
            )
            assert d.next_frame == last + 1

    def test_ignores_lag(self):
        s = BestEffortStrategy()
        high_lag = BufferMetrics(
            frame_lag=1000,
            time_lag_s=33.3,
            display_frame_lag=500,
            display_time_lag_s=16.6,
            current_frame_miss=200,
            memory_used_bytes=0,
            cache_hit_ratio=0.0,
        )
        d = s.decide(
            last_submitted=10,
            last_completed=5,
            timeline=MagicMock(),
            metrics=high_lag,
        )
        assert d.next_frame == 11


class TestSyncedStrategy:
    def test_compliant_with_protocol(self):
        assert isinstance(SyncedStrategy(), FrameSkipStrategy)

    def test_advances_by_one_when_in_sync(self):
        s = SyncedStrategy()
        timeline = MagicMock()
        timeline.current_frame.return_value = 10
        d = s.decide(
            last_submitted=10,
            last_completed=9,
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 11

    def test_jumps_to_target_when_modestly_behind(self):
        # "Modestly behind" = within the adaptive fallback threshold,
        # so the strategy should still try to keep up by jumping ahead.
        # Gap here (target - last_completed = 50) is below the default
        # threshold of 60 frames.
        s = SyncedStrategy()
        timeline = MagicMock()
        timeline.current_frame.return_value = 55
        d = s.decide(
            last_submitted=10,
            last_completed=5,
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 55

    def test_never_goes_backward(self):
        s = SyncedStrategy()
        timeline = MagicMock()
        timeline.current_frame.return_value = 5
        d = s.decide(
            last_submitted=10,
            last_completed=8,
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 11

    def test_first_call_returns_zero(self):
        s = SyncedStrategy()
        timeline = MagicMock()
        timeline.current_frame.return_value = 0
        d = s.decide(
            last_submitted=-1,
            last_completed=-1,
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 0


class TestSyncedStrategyAdaptiveFallback:
    """The adaptive fallback: when processing is catastrophically behind
    the timeline, the strategy stops asking for jump-aheads and submits
    sequentially. Prevents the death spiral on slow sources where every
    jump-ahead is a costly random seek the reader can't service."""

    def test_falls_back_to_sequential_when_far_behind(self):
        # target - last_completed = 200, well above default threshold (60), AND
        # reads are slow (I/O-bound) so the sequential fallback is warranted.
        s = SyncedStrategy()
        timeline = MagicMock()
        timeline.current_frame.return_value = 200
        d = s.decide(
            last_submitted=10,
            last_completed=0,
            timeline=timeline,
            metrics=_zero_metrics(),
            read_latency_ms=120.0,
        )
        # Sequential submission instead of jumping to 200.
        assert d.next_frame == 11

    def test_at_threshold_still_jumps(self):
        # Exactly threshold = NOT over, so jump is allowed. Tests the
        # boundary condition (`>` not `>=`).
        s = SyncedStrategy(max_lag_frames=10)
        timeline = MagicMock()
        timeline.current_frame.return_value = 15
        d = s.decide(
            last_submitted=4,
            last_completed=5,  # target - last_completed = 10 == threshold
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 15

    def test_just_over_threshold_falls_back(self):
        s = SyncedStrategy(max_lag_frames=10)
        timeline = MagicMock()
        timeline.current_frame.return_value = 16
        d = s.decide(
            last_submitted=4,
            last_completed=5,  # target - last_completed = 11 > threshold
            timeline=timeline,
            metrics=_zero_metrics(),
            read_latency_ms=120.0,  # I/O-bound → fallback allowed
        )
        assert d.next_frame == 5

    def test_warmup_submits_sequentially_not_jump_to_target(self):
        # last_completed = -1 means nothing has completed yet (cold start while
        # the first frame loads). The wall-clock target has run far ahead (999),
        # but chasing it would flood the queue with sparse high indices and skip
        # the opening — so warm-up submits sequentially from where we are.
        s = SyncedStrategy(max_lag_frames=0)
        timeline = MagicMock()
        timeline.current_frame.return_value = 999
        d = s.decide(
            last_submitted=-1,
            last_completed=-1,
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 0  # sequential from the start, NOT a jump to 999

    def test_warmup_does_not_enter_fallback_mode(self):
        # Warm-up isn't "lagging" — it's just cold. Mode stays "synced".
        s = SyncedStrategy(max_lag_frames=0)
        timeline = MagicMock()
        timeline.current_frame.return_value = 999
        s.decide(last_submitted=-1, last_completed=-1, timeline=timeline,
                 metrics=_zero_metrics())
        assert s.current_mode() == "synced"

    def test_seek_to_zero_after_completion_is_not_warmup(self):
        # Warm-up is a COLD-START concept (first frame loading models). A seek to
        # frame 0 mid-session also drives last_completed back to -1, but it must
        # NOT be treated as warm-up (which would flood the opening sequentially)
        # — normal skip-to-wall-clock applies once any frame has ever completed.
        s = SyncedStrategy()
        tl = MagicMock()
        tl.current_frame.return_value = 50
        s.decide(last_submitted=50, last_completed=50, timeline=tl,
                 metrics=_zero_metrics())  # a frame completed this session
        # Seek-to-0: last_completed back to -1, but wall-clock playhead is at 10.
        tl.current_frame.return_value = 10
        d = s.decide(last_submitted=-1, last_completed=-1, timeline=tl,
                     metrics=_zero_metrics())
        assert d.next_frame == 10  # normal skip to wall-clock, NOT warm-up's 0

    def test_recovers_when_caught_up(self):
        # Once last_completed catches up to within threshold, behaviour
        # returns to "jump to target." This is the recovery path —
        # makes sense for transient slow periods.
        s = SyncedStrategy(max_lag_frames=10)
        timeline = MagicMock()
        timeline.current_frame.return_value = 20
        d = s.decide(
            last_submitted=15,
            last_completed=15,  # caught up — gap = 5, below threshold
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 20

    def test_custom_max_lag_frames(self):
        s = SyncedStrategy(max_lag_frames=200)
        timeline = MagicMock()
        timeline.current_frame.return_value = 150
        d = s.decide(
            last_submitted=10,
            last_completed=5,  # gap = 145, below the custom threshold
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert d.next_frame == 150

    def test_max_lag_frames_property(self):
        # Public property — controller relies on this for change detection.
        s = SyncedStrategy(max_lag_frames=120)
        assert s.max_lag_frames == 120
        assert SyncedStrategy().max_lag_frames == 60  # default


class TestSyncedStrategyHysteresis:
    """Fallback enters above max_lag but only LEAVES below recover_lag, so a lag
    parked near the boundary doesn't flap the mode/read-pattern every frame."""

    def _decide(self, s, target, last_completed, last_submitted=0):
        # Hold reads I/O-bound throughout so these tests exercise the LAG
        # thresholds (the bottleneck-awareness gate is tested separately).
        timeline = MagicMock()
        timeline.current_frame.return_value = target
        return s.decide(
            last_submitted=last_submitted,
            last_completed=last_completed,
            timeline=timeline,
            metrics=_zero_metrics(),
            read_latency_ms=120.0,
        )

    def test_recover_lag_defaults_to_half(self):
        assert SyncedStrategy().recover_lag_frames == 30
        assert SyncedStrategy(max_lag_frames=40).recover_lag_frames == 20

    def test_recover_clamped_below_enter(self):
        s = SyncedStrategy(max_lag_frames=10, recover_lag_frames=50)
        assert s.recover_lag_frames == 10

    def test_stays_in_fallback_in_the_hysteresis_band(self):
        s = SyncedStrategy(max_lag_frames=60, recover_lag_frames=30)
        # Enter fallback (lag 100 > 60).
        assert self._decide(s, target=200, last_completed=100).next_frame == 1
        assert s.current_mode() == "synced (lagging)"
        # Lag now 45 — between recover(30) and enter(60). Must STAY in fallback
        # (sequential), not flip back to skip-to-target.
        d = self._decide(s, target=200, last_completed=155, last_submitted=0)
        assert d.next_frame == 1  # still sequential
        assert s.current_mode() == "synced (lagging)"

    def test_exits_fallback_below_recover(self):
        s = SyncedStrategy(max_lag_frames=60, recover_lag_frames=30)
        self._decide(s, target=200, last_completed=100)  # enter fallback
        # Lag now 20 (< recover 30) → leave fallback → skip to target.
        d = self._decide(s, target=200, last_completed=180, last_submitted=0)
        assert d.next_frame == 200
        assert s.current_mode() == "synced"

    def test_does_not_reenter_in_the_band(self):
        s = SyncedStrategy(max_lag_frames=60, recover_lag_frames=30)
        # Not in fallback; lag 45 is in the band but below the ENTER threshold,
        # so we keep skipping (don't enter on the lower bound).
        d = self._decide(s, target=200, last_completed=155)
        assert d.next_frame == 200
        assert s.current_mode() == "synced"


class TestSyncedStrategyBottleneckAware:
    """Fallback to sequential is only right when READS are the bottleneck
    (slow source → skipping thrashes random I/O to ~0 fps). When COMPUTE is the
    bottleneck (fast disk, slow GPU), skipping is free for the reader and keeps
    playback synced; sequential there is just slow-motion."""

    def _decide(self, s, target, last_completed, last_submitted=0, read_latency_ms=None):
        timeline = MagicMock()
        timeline.current_frame.return_value = target
        return s.decide(
            last_submitted=last_submitted,
            last_completed=last_completed,
            timeline=timeline,
            metrics=_zero_metrics(),
            read_latency_ms=read_latency_ms,
        )

    def test_compute_bound_keeps_skipping_when_far_behind(self):
        # Lag 200 (far behind) but reads are FAST (2 ms) → compute-bound → keep
        # skipping to stay synced, NOT slow-motion sequential.
        s = SyncedStrategy(io_bound_read_ms=50)
        d = self._decide(s, target=200, last_completed=0, last_submitted=10,
                         read_latency_ms=2.0)
        assert d.next_frame == 200
        assert s.current_mode() == "synced"

    def test_io_bound_falls_back_when_far_behind(self):
        # Lag 200 AND reads SLOW (150 ms, e.g. ffmpeg random seek) → I/O-bound →
        # sequential fallback.
        s = SyncedStrategy(io_bound_read_ms=50)
        d = self._decide(s, target=200, last_completed=0, last_submitted=10,
                         read_latency_ms=150.0)
        assert d.next_frame == 11
        assert s.current_mode() == "synced (lagging)"

    def test_no_latency_signal_keeps_skipping(self):
        # No read-cost signal → default to staying synced, not assume I/O-bound.
        s = SyncedStrategy(io_bound_read_ms=50)
        d = self._decide(s, target=200, last_completed=0, last_submitted=10,
                         read_latency_ms=None)
        assert d.next_frame == 200


class TestSyncedStrategyLookaheadCap:
    """A faster-than-target pipeline must not pre-render the whole rest of the
    clip ahead of the playhead — render-ahead is bounded to lookahead_frames."""

    def _decide(self, s, target, last_submitted, last_completed=None):
        timeline = MagicMock()
        timeline.current_frame.return_value = target
        return s.decide(
            last_submitted=last_submitted,
            last_completed=last_completed if last_completed is not None else target,
            timeline=timeline,
            metrics=_zero_metrics(),
        )

    def test_idles_when_rendered_full_lookahead_ahead(self):
        s = SyncedStrategy(lookahead_frames=120)
        # Submitted 250, playhead at 10 → 240 ahead, past the 120 cushion → idle.
        d = self._decide(s, target=10, last_submitted=250, last_completed=10)
        assert d.next_frame is None

    def test_submits_within_the_lookahead_window(self):
        s = SyncedStrategy(lookahead_frames=120)
        # Submitted 100, playhead 10 → 90 ahead, within the cushion → keep going.
        d = self._decide(s, target=10, last_submitted=100, last_completed=10)
        assert d.next_frame == 101

    def test_default_lookahead_does_not_block_normal_sync(self):
        # In-sync submission (target just ahead of last_submitted) is never capped.
        s = SyncedStrategy()
        d = self._decide(s, target=500, last_submitted=499, last_completed=499)
        assert d.next_frame == 500


class TestCurrentMode:
    """current_mode() surfaces in the status bar so the user can tell
    when an adaptive strategy has shifted behaviour."""

    def test_best_effort_mode(self):
        assert BestEffortStrategy().current_mode() == "best effort"

    def test_synced_initial_mode(self):
        # Before any decide() call, Synced reports the not-lagging mode.
        assert SyncedStrategy().current_mode() == "synced"

    def test_synced_mode_after_normal_decide(self):
        s = SyncedStrategy(max_lag_frames=10)
        timeline = MagicMock()
        timeline.current_frame.return_value = 5
        s.decide(
            last_submitted=4,
            last_completed=4,
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert s.current_mode() == "synced"

    def test_synced_mode_after_fallback_decide(self):
        s = SyncedStrategy(max_lag_frames=10)
        timeline = MagicMock()
        timeline.current_frame.return_value = 100
        s.decide(
            last_submitted=4,
            last_completed=5,  # gap = 95, over threshold
            timeline=timeline,
            metrics=_zero_metrics(),
            read_latency_ms=120.0,  # I/O-bound → fallback engages
        )
        assert s.current_mode() == "synced (lagging)"

    def test_synced_mode_recovers(self):
        # Mode should track decide() outcomes — falling into fallback
        # then catching up should reset the mode label.
        s = SyncedStrategy(max_lag_frames=10)
        timeline = MagicMock()
        timeline.current_frame.return_value = 100
        s.decide(
            last_submitted=4,
            last_completed=5,
            timeline=timeline,
            metrics=_zero_metrics(),
            read_latency_ms=120.0,  # I/O-bound → fallback engages
        )
        assert s.current_mode() == "synced (lagging)"
        # Now caught up.
        timeline.current_frame.return_value = 15
        s.decide(
            last_submitted=14,
            last_completed=14,
            timeline=timeline,
            metrics=_zero_metrics(),
        )
        assert s.current_mode() == "synced"
