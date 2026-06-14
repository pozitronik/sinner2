"""Tests for the SectionSet value object (timeline range selection)."""
from __future__ import annotations

from sinner2.pipeline.sections import SectionSet


class TestNormalization:
    def test_empty_by_default(self):
        assert SectionSet.empty().is_empty()
        assert SectionSet.of([]).ranges == ()

    def test_reorders_inverted_pair(self):
        assert SectionSet.of([(120, 50)]).ranges == ((50, 120),)

    def test_sorts_ranges(self):
        s = SectionSet.of([(180, 240), (50, 120)])
        assert s.ranges == ((50, 120), (180, 240))

    def test_merges_overlapping(self):
        assert SectionSet.of([(50, 120), (100, 200)]).ranges == ((50, 200),)

    def test_merges_touching_zero_gap(self):
        # [50,120] and [121,180] have no excluded frame between → one region.
        assert SectionSet.of([(50, 120), (121, 180)]).ranges == ((50, 180),)

    def test_keeps_distinct_when_real_gap(self):
        # A gap of >= 1 excluded frame keeps them separate.
        s = SectionSet.of([(50, 120), (122, 180)])
        assert s.ranges == ((50, 120), (122, 180))

    def test_floors_negative_start(self):
        assert SectionSet.of([(-10, 30)]).ranges == ((0, 30),)

    def test_drops_fully_negative(self):
        assert SectionSet.of([(-10, -5), (10, 20)]).ranges == ((10, 20),)


class TestQueries:
    def setup_method(self):
        self.s = SectionSet.of([(50, 120), (180, 240)])

    def test_contains(self):
        assert self.s.contains(50)
        assert self.s.contains(120)
        assert not self.s.contains(49)
        assert not self.s.contains(150)  # gap
        assert self.s.contains(200)

    def test_index_at(self):
        assert self.s.index_at(60) == 0
        assert self.s.index_at(200) == 1
        assert self.s.index_at(150) is None

    def test_next_included_frame(self):
        assert self.s.next_included_frame(0) == 50  # before first → first start
        assert self.s.next_included_frame(60) == 60  # inside → itself
        assert self.s.next_included_frame(150) == 180  # in gap → next start
        assert self.s.next_included_frame(240) == 240
        assert self.s.next_included_frame(241) is None  # past last

    def test_first_last_total(self):
        assert self.s.first_frame() == 50
        assert self.s.last_frame() == 240
        assert self.s.total_frames() == (120 - 50 + 1) + (240 - 180 + 1)

    def test_empty_queries(self):
        e = SectionSet.empty()
        assert e.first_frame() is None
        assert e.last_frame() is None
        assert e.next_included_frame(10) is None
        assert e.total_frames() == 0

    def test_count_included_between(self):
        # Strictly between 40 and 130: frames 50..120 (gap part excluded).
        assert self.s.count_included_between(40, 130) == 71
        # Between 120 and 180 (the gap): nothing included strictly inside.
        assert self.s.count_included_between(120, 180) == 0
        # Spanning the gap, 100 to 200: 101..120 (20) + 180..199 (20) = 40.
        assert self.s.count_included_between(100, 200) == 40


class TestEdits:
    def test_with_added(self):
        s = SectionSet.empty().with_added(50, 120)
        assert s.ranges == ((50, 120),)
        s2 = s.with_added(180, 240)
        assert s2.ranges == ((50, 120), (180, 240))

    def test_with_added_reorders(self):
        s = SectionSet.empty().with_added(240, 180)
        assert s.ranges == ((180, 240),)

    def test_with_range_replaced_nudges_start(self):
        s = SectionSet.of([(50, 120), (180, 240)])
        # Nudge section 1's start 180 → 175.
        s2 = s.with_range_replaced(1, 175, 240)
        assert s2.ranges == ((50, 120), (175, 240))

    def test_with_range_replaced_can_merge(self):
        s = SectionSet.of([(50, 120), (180, 240)])
        # Pull section 1's start back into section 0 → they fuse.
        s2 = s.with_range_replaced(1, 100, 240)
        assert s2.ranges == ((50, 240),)

    def test_without_index(self):
        s = SectionSet.of([(50, 120), (180, 240)])
        assert s.without_index(0).ranges == ((180, 240),)
        assert s.without_index(1).ranges == ((50, 120),)

    def test_without_index_out_of_range_is_noop(self):
        s = SectionSet.of([(50, 120)])
        assert s.without_index(5).ranges == ((50, 120),)

    def test_clamp(self):
        s = SectionSet.of([(50, 120), (180, 240)])
        # Target only has 150 frames (last index 149): trim section 1 away,
        # second range fully past the end is dropped.
        assert s.clamp(149).ranges == ((50, 120),)
        # Trim a range that straddles the end.
        assert SectionSet.of([(50, 200)]).clamp(149).ranges == ((50, 149),)


class TestFramePlan:
    def test_empty_plan_is_full_range(self):
        assert SectionSet.empty().frame_plan(5) == [0, 1, 2, 3, 4]

    def test_plan_is_union_in_order(self):
        s = SectionSet.of([(1, 2), (5, 6)])
        assert s.frame_plan(10) == [1, 2, 5, 6]

    def test_plan_clamps_to_total(self):
        s = SectionSet.of([(3, 99)])
        assert s.frame_plan(6) == [3, 4, 5]

    def test_plan_no_duplicates_when_merged(self):
        s = SectionSet.of([(1, 4), (3, 6)])  # merges to (1,6)
        assert s.frame_plan(10) == [1, 2, 3, 4, 5, 6]


class TestSerialization:
    def test_to_pairs_round_trips(self):
        s = SectionSet.of([(180, 240), (50, 120)])
        pairs = s.to_pairs()
        assert pairs == [[50, 120], [180, 240]]
        assert SectionSet.of(pairs) == s

    def test_empty_to_pairs(self):
        assert SectionSet.empty().to_pairs() == []

    def test_value_equality(self):
        assert SectionSet.of([(50, 120)]) == SectionSet.of([(50, 120)])
        assert SectionSet.of([(50, 120)]) != SectionSet.of([(50, 121)])
        # Hashable (frozen) → usable in sets / as dict keys.
        assert len({SectionSet.of([(1, 2)]), SectionSet.of([(1, 2)])}) == 1
