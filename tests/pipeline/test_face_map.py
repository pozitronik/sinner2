"""Tests for the FaceMap identity catalog (face-mapping domain)."""
from __future__ import annotations

import math

from sinner2.pipeline.face_map import (
    FaceMap,
    Identity,
    IdentityMode,
    UnmatchedPolicy,
    cosine,
    normalize,
)


def _unit(*vals: float) -> tuple[float, ...]:
    return normalize(vals)


class TestHelpers:
    def test_normalize_unit_length(self):
        v = normalize([3.0, 4.0])
        assert math.isclose(math.hypot(*v), 1.0)
        assert math.isclose(v[0], 0.6) and math.isclose(v[1], 0.8)

    def test_normalize_zero_vector_passes_through(self):
        assert normalize([0.0, 0.0]) == (0.0, 0.0)

    def test_cosine_identical_is_one(self):
        a = _unit(1.0, 2.0, 3.0)
        assert math.isclose(cosine(a, a), 1.0)

    def test_cosine_orthogonal_is_zero(self):
        assert math.isclose(cosine(_unit(1, 0), _unit(0, 1)), 0.0, abs_tol=1e-9)

    def test_cosine_mismatched_length(self):
        assert cosine([1.0], [1.0, 2.0]) == -1.0
        assert cosine([], []) == -1.0


class TestIdentity:
    def test_new_normalizes_centroid(self):
        ident = Identity.new([3.0, 4.0])
        assert math.isclose(math.hypot(*ident.centroid), 1.0)
        assert ident.source_path is None
        assert ident.occurrences == 1

    def test_observed_moves_centroid_and_counts(self):
        ident = Identity.new([1.0, 0.0])
        moved = ident.observed([0.0, 1.0])
        assert moved.occurrences == 2
        # Halfway between the two unit axes → ~45°, still unit length.
        assert math.isclose(math.hypot(*moved.centroid), 1.0)
        assert moved.centroid[0] > 0 and moved.centroid[1] > 0

    def test_observed_pulls_toward_repeated_observation(self):
        ident = Identity.new([1.0, 0.0])
        # Many observations of the SAME vector keep the centroid put.
        for _ in range(5):
            ident = ident.observed([1.0, 0.0])
        assert math.isclose(ident.centroid[0], 1.0, abs_tol=1e-6)


class TestMatching:
    def _map(self) -> FaceMap:
        # Two well-separated identities (orthogonal in 3-space).
        a = Identity("a", _unit(1, 0, 0), source_path="/src/alice.png")
        b = Identity("b", _unit(0, 1, 0), source_path="/src/bob.png")
        return FaceMap(identities=(a, b), threshold=0.5)

    def test_best_match_picks_nearest(self):
        m = self._map()
        assert m.best_match(_unit(0.9, 0.1, 0.0)).id == "a"
        assert m.best_match(_unit(0.1, 0.9, 0.0)).id == "b"

    def test_best_match_below_threshold_is_none(self):
        m = self._map()
        # Equidistant-ish from both, similarity ~0.7/sqrt2 ≈ 0.5 each but the
        # orthogonal third axis pushes it below — a clear stranger.
        assert m.best_match(_unit(0, 0, 1)) is None

    def test_best_match_exact_tie_keeps_first(self):
        # Identical centroids → an exact tie; the FIRST (catalog order) wins, not
        # the last (the >= last-wins quirk the docstring contradicted).
        m = FaceMap(
            identities=(Identity("x", _unit(1, 0)), Identity("y", _unit(1, 0))),
            threshold=0.5,
        )
        assert m.best_match(_unit(1, 0)).id == "x"

    def test_best_match_threshold_is_inclusive(self):
        # A similarity EXACTLY at the threshold still matches (the tie-break fix
        # must not turn the threshold gate strict). sim == 1.0 == threshold.
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),), threshold=1.0)
        assert m.best_match(_unit(1, 0)).id == "a"

    def test_source_for_matched_identity(self):
        m = self._map()
        assert m.source_for(_unit(0.95, 0.05, 0)) == "/src/alice.png"

    def test_source_for_matched_but_unassigned_is_none(self):
        a = Identity("a", _unit(1, 0, 0), source_path=None)  # tracked, no source
        m = FaceMap(identities=(a,), threshold=0.5)
        assert m.source_for(_unit(1, 0, 0)) is None

    def test_unmatched_skip(self):
        m = self._map()  # default policy SKIP
        assert m.source_for(_unit(0, 0, 1)) is None

    def test_unmatched_default(self):
        m = self._map().with_unmatched(
            UnmatchedPolicy.DEFAULT, default_source="/src/extra.png"
        )
        assert m.source_for(_unit(0, 0, 1)) == "/src/extra.png"

    def test_unmatched_first(self):
        m = FaceMap(
            identities=self._map().identities,
            threshold=0.5,
            unmatched=UnmatchedPolicy.FIRST,
        )
        assert m.source_for(_unit(0, 0, 1)) == "/src/alice.png"


class TestActivation:
    def test_empty_is_inactive(self):
        assert FaceMap.empty().is_empty()
        assert not FaceMap.empty().is_active()

    def test_identities_without_sources_is_inactive(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),))
        assert not m.is_empty()
        assert not m.is_active()  # tracked but nothing assigned → no routing change

    def test_assigned_source_activates(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0), source_path="/s.png"),))
        assert m.is_active()

    def test_default_policy_with_source_activates(self):
        m = FaceMap(unmatched=UnmatchedPolicy.DEFAULT, default_source="/d.png")
        assert m.is_active()

    def test_armed_activates_without_assignments(self):
        # Faces mode on but nothing mapped yet → routing still engages (every
        # face is unmapped → shows original, not the global source).
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),)).with_armed(True)
        assert m.is_active()
        assert m.with_armed(False).is_active() is False

    def test_armed_is_not_serialized(self):
        # Transient UI state — a reloaded map is unarmed until the live mode
        # re-arms it.
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),)).with_armed(True)
        assert "armed" not in m.to_dict()
        assert FaceMap.from_dict(m.to_dict()).armed is False


class TestEdits:
    def test_assign_source(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),))
        m2 = m.assign_source("a", "/src/x.png")
        assert m2.identities[0].source_path == "/src/x.png"
        assert m.identities[0].source_path is None  # original untouched

    def test_without_identity(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)), Identity("b", _unit(0, 1))))
        assert [i.id for i in m.without_identity("a").identities] == ["b"]

    def test_assigned_sources_distinct_in_order(self):
        m = FaceMap(
            identities=(
                Identity("a", _unit(1, 0), source_path="/x.png"),
                Identity("b", _unit(0, 1), source_path="/x.png"),  # dup
                Identity("c", _unit(0, 0, 1), source_path="/y.png"),
            ),
            unmatched=UnmatchedPolicy.DEFAULT,
            default_source="/z.png",
        )
        assert m.assigned_sources() == ["/x.png", "/y.png", "/z.png"]

    def test_index_of(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),))
        assert m.index_of("a") == 0
        assert m.index_of("nope") is None


class TestMerge:
    def test_merges_into_first_and_sums_occurrences(self):
        m = FaceMap(identities=(
            Identity("a", _unit(1, 0, 0), occurrences=10, source_path="/alice.png"),
            Identity("b", _unit(0.9, 0.1, 0), occurrences=2),
            Identity("c", _unit(0, 0, 1), occurrences=5),  # untouched
        ))
        merged = m.merge(["a", "b"])
        assert [i.id for i in merged.identities] == ["a", "c"]  # b absorbed, order kept
        assert merged.identities[0].occurrences == 12
        assert merged.identities[0].source_path == "/alice.png"  # keeper's source

    def test_keeper_source_falls_back_to_a_member(self):
        m = FaceMap(identities=(
            Identity("a", _unit(1, 0), occurrences=1),  # no source
            Identity("b", _unit(0.9, 0.1), occurrences=1, source_path="/bob.png"),
        ))
        assert m.merge(["a", "b"]).identities[0].source_path == "/bob.png"

    def test_centroid_is_occurrence_weighted(self):
        import math

        m = FaceMap(identities=(
            Identity("a", _unit(1, 0), occurrences=3),
            Identity("b", _unit(0, 1), occurrences=1),
        ))
        got = m.merge(["a", "b"]).identities[0].centroid
        for g, e in zip(got, normalize((3.0, 1.0))):
            assert math.isclose(g, e, abs_tol=1e-6)

    def test_keeps_clearest_rep_and_earliest_first_frame(self):
        m = FaceMap(identities=(
            Identity("a", _unit(1, 0), occurrences=1, first_frame=10,
                     det_score=0.6, sex="M", age=30),
            Identity("b", _unit(0.9, 0.1), occurrences=1, first_frame=3,
                     det_score=0.95, sex="M", age=31),
        ))
        merged = m.merge(["a", "b"]).identities[0]
        assert merged.first_frame == 3        # earliest
        assert merged.det_score == 0.95       # clearest occurrence
        assert merged.age == 31               # from the clearest

    def test_needs_two_valid_ids(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),))
        assert m.merge(["a"]) == m
        assert m.merge([]) == m
        assert m.merge(["a", "ghost"]) == m  # only one valid id


class TestClustering:
    def test_observe_starts_first_identity(self):
        m = FaceMap.empty().observe(_unit(1, 0, 0))
        assert len(m.identities) == 1
        assert m.identities[0].occurrences == 1

    def test_observe_merges_similar(self):
        m = FaceMap.empty().observe(_unit(1, 0, 0)).observe(_unit(0.95, 0.05, 0))
        assert len(m.identities) == 1
        assert m.identities[0].occurrences == 2

    def test_observe_splits_dissimilar(self):
        m = (
            FaceMap.empty()
            .observe(_unit(1, 0, 0))
            .observe(_unit(0, 1, 0))
            .observe(_unit(0, 0, 1))
        )
        assert len(m.identities) == 3

    def test_clusters_a_noisy_stream_into_two_people(self):
        m = FaceMap.empty()
        # Alternating two people with small jitter on each axis.
        for k in range(6):
            jitter = 0.03 * (k % 2)
            m = m.observe(_unit(1.0, jitter, 0.0))
            m = m.observe(_unit(0.0, 1.0, jitter))
        assert len(m.identities) == 2
        counts = sorted(i.occurrences for i in m.identities)
        assert counts == [6, 6]

    def test_observe_with_id_returns_joined_identity(self):
        m, id1 = FaceMap.empty().observe_with_id(_unit(1, 0, 0))
        m, id2 = m.observe_with_id(_unit(0.95, 0.05, 0))  # same person
        assert id1 == id2  # joined the existing identity
        m, id3 = m.observe_with_id(_unit(0, 1, 0))  # new person
        assert id3 != id1


class TestReference:
    def test_with_reference_sets_occurrence(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),))
        m2 = m.with_reference("a", 42, (1.0, 2.0, 3.0, 4.0))
        assert m2.identities[0].ref_frame == 42
        assert m2.identities[0].ref_bbox == (1.0, 2.0, 3.0, 4.0)

    def test_reference_round_trips(self):
        m = FaceMap(
            identities=(
                Identity("a", _unit(1, 0), ref_frame=7, ref_bbox=(0.0, 1.0, 2.0, 3.0)),
            )
        )
        restored = FaceMap.from_dict(m.to_dict())
        assert restored.identities[0].ref_frame == 7
        assert restored.identities[0].ref_bbox == (0.0, 1.0, 2.0, 3.0)

    def test_with_reference_sets_demographics(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),))
        m2 = m.with_reference("a", 5, (0.0, 1.0, 2.0, 3.0), sex="M", age=34)
        assert m2.identities[0].sex == "M"
        assert m2.identities[0].age == 34

    def test_demographics_round_trip(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0), sex="F", age=28),))
        restored = FaceMap.from_dict(m.to_dict())
        assert restored.identities[0].sex == "F"
        assert restored.identities[0].age == 28

    def test_first_frame_set_and_round_trip(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),))
        m2 = m.with_reference("a", 5, (0.0, 1.0, 2.0, 3.0), first_frame=2)
        assert m2.identities[0].first_frame == 2
        assert FaceMap.from_dict(m2.to_dict()).identities[0].first_frame == 2


class TestSerialization:
    def test_round_trips(self):
        m = FaceMap(
            identities=(
                Identity("a", _unit(1, 0, 0), source_path="/src/alice.png", occurrences=12, label="Alice"),
                Identity("b", _unit(0, 1, 0)),
            ),
            threshold=0.55,
            unmatched=UnmatchedPolicy.DEFAULT,
            default_source="/src/extra.png",
            mode=IdentityMode.EMBEDDING,
        )
        restored = FaceMap.from_dict(m.to_dict())
        assert restored.threshold == 0.55
        assert restored.unmatched is UnmatchedPolicy.DEFAULT
        assert restored.default_source == "/src/extra.png"
        assert len(restored.identities) == 2
        assert restored.identities[0].label == "Alice"
        assert restored.identities[0].occurrences == 12
        # Centroid survives (within float round-trip).
        for got, exp in zip(restored.identities[0].centroid, m.identities[0].centroid):
            assert math.isclose(got, exp)

    def test_score_and_pose_round_trip(self):
        m = FaceMap(identities=(
            Identity("a", _unit(1, 0), det_score=0.91,
                     pitch=2.0, yaw=-4.0, roll=11.0),
        ))
        i = FaceMap.from_dict(m.to_dict()).identities[0]
        assert i.det_score == 0.91
        assert (i.pitch, i.yaw, i.roll) == (2.0, -4.0, 11.0)

    def test_legacy_dict_without_score_pose(self):
        # Older sidecars predate these fields → None, never a KeyError.
        i = FaceMap.from_dict({"identities": [{"id": "a", "centroid": [1.0, 0.0]}]})
        assert i.identities[0].det_score is None
        assert i.identities[0].roll is None

    def test_with_reference_stores_score_and_pose(self):
        m = FaceMap(identities=(Identity("a", _unit(1, 0)),)).with_reference(
            "a", 5, (0.0, 0.0, 4.0, 4.0),
            det_score=0.8, pitch=1.0, yaw=2.0, roll=3.0,
        )
        i = m.identities[0]
        assert i.det_score == 0.8 and (i.pitch, i.yaw, i.roll) == (1.0, 2.0, 3.0)

    def test_from_empty_dict_is_safe(self):
        assert FaceMap.from_dict({}).is_empty()
