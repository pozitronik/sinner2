"""Tests for the per-frame pipeline state map (processing-visualiser data model)."""
from __future__ import annotations

from sinner2.pipeline.realtime.frame_state import FrameState, FrameStateMap


class TestFrameStateMap:
    def test_starts_all_not_reached(self):
        m = FrameStateMap(5)
        assert m.frame_count == 5
        assert m.snapshot() == bytes([FrameState.NOT_REACHED]) * 5
        assert all(m.get(i) is FrameState.NOT_REACHED for i in range(5))

    def test_set_and_get(self):
        m = FrameStateMap(4)
        m.set(2, FrameState.PROCESSING)
        assert m.get(2) is FrameState.PROCESSING
        assert m.get(0) is FrameState.NOT_REACHED

    def test_set_out_of_range_is_noop(self):
        m = FrameStateMap(3)
        m.set(-1, FrameState.READY_MEM)  # no crash
        m.set(99, FrameState.READY_MEM)
        assert m.snapshot() == bytes(3)

    def test_get_out_of_range_is_not_reached(self):
        m = FrameStateMap(3)
        assert m.get(-1) is FrameState.NOT_REACHED
        assert m.get(99) is FrameState.NOT_REACHED

    def test_set_range_sets_half_open_interval(self):
        m = FrameStateMap(6)
        m.set_range(1, 4, FrameState.SKIPPED)
        assert m.get(0) is FrameState.NOT_REACHED
        assert m.get(1) is FrameState.SKIPPED
        assert m.get(3) is FrameState.SKIPPED
        assert m.get(4) is FrameState.NOT_REACHED  # hi is exclusive

    def test_set_range_clamps_to_bounds(self):
        m = FrameStateMap(4)
        m.set_range(-5, 100, FrameState.QUEUED)
        assert m.snapshot() == bytes([FrameState.QUEUED]) * 4

    def test_set_range_empty_when_hi_le_lo(self):
        m = FrameStateMap(4)
        m.set_range(3, 3, FrameState.INVALID)
        m.set_range(3, 1, FrameState.INVALID)
        assert m.snapshot() == bytes(4)

    def test_snapshot_is_a_copy(self):
        m = FrameStateMap(3)
        snap = m.snapshot()
        m.set(0, FrameState.READY_MEM)
        assert snap == bytes(3)  # old snapshot unaffected
        assert m.snapshot()[0] == FrameState.READY_MEM

    def test_reset_clears_to_not_reached(self):
        m = FrameStateMap(4)
        m.set_range(0, 4, FrameState.READY_DISK)
        m.reset()
        assert m.snapshot() == bytes(4)

    def test_zero_length_map(self):
        m = FrameStateMap(0)
        assert m.frame_count == 0
        assert m.snapshot() == b""
        m.set(0, FrameState.QUEUED)  # no crash
        m.set_range(0, 5, FrameState.QUEUED)
        assert m.snapshot() == b""

    def test_negative_count_clamped_to_zero(self):
        m = FrameStateMap(-3)
        assert m.frame_count == 0

    def test_state_byte_values_are_stable(self):
        # The map stores the enum's int value directly; the GUI relies on these.
        assert FrameState.NOT_REACHED == 0
        assert FrameState.SKIPPED == 1
        assert FrameState.QUEUED == 2
        assert FrameState.PROCESSING == 3
        assert FrameState.READY_MEM == 4
        assert FrameState.READY_DISK == 5
        assert FrameState.INVALID == 6
