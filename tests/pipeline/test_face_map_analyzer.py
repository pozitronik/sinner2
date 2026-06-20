"""Tests for the strided face-map analysis pass."""
from __future__ import annotations

import threading

from sinner2.pipeline.face_map import normalize
from sinner2.pipeline.face_map_analyzer import analyze_target


def _catalog(*args, **kwargs):
    """analyze_target now returns (catalog, scanned, total); most tests want
    just the catalog."""
    return analyze_target(*args, **kwargs)[0]


class _Face:
    def __init__(self, embedding, score=0.9, bbox=(0.0, 0.0, 4.0, 4.0),
                 sex=None, age=None):
        self.normed_embedding = embedding
        self.det_score = score
        self.bbox = bbox
        self.sex = sex
        self.age = age


class _StubReader:
    """A reader whose frame i IS a list of faces; detect() just returns it."""

    def __init__(self, frames):
        self._frames = frames

    @property
    def frame_count(self):
        return len(self._frames)

    @property
    def width(self):
        return 64

    @property
    def height(self):
        return 48

    def read(self, idx):
        return self._frames[idx] if 0 <= idx < len(self._frames) else None

    def release(self):
        pass


def _emb(*vals):
    return normalize(vals)


def _identity(*vals):
    return [_Face(_emb(*vals))]


class TestClustering:
    def test_clusters_two_people(self):
        frames = [
            _identity(1, 0, 0),
            _identity(0, 1, 0),
            [_Face(_emb(0.95, 0.05, 0))],  # person A again
        ]
        fm = _catalog(_StubReader(frames), lambda f: f, stride=1, threshold=0.5)
        assert len(fm.identities) == 2
        counts = sorted(i.occurrences for i in fm.identities)
        assert counts == [1, 2]

    def test_records_clearest_occurrence(self):
        frames = [
            [_Face(_emb(1, 0, 0), score=0.5, bbox=(0.0, 0.0, 4.0, 4.0))],
            [_Face(_emb(1, 0, 0), score=0.9, bbox=(1.0, 1.0, 5.0, 5.0))],  # clearer
        ]
        fm = _catalog(_StubReader(frames), lambda f: f, stride=1)
        ident = fm.identities[0]
        assert ident.occurrences == 2
        assert ident.ref_frame == 1  # the higher-score detection
        assert ident.ref_bbox == (1.0, 1.0, 5.0, 5.0)

    def test_face_without_embedding_skipped(self):
        fm = _catalog(
            _StubReader([[_Face(None)]]), lambda f: f, stride=1
        )
        assert fm.is_empty()

    def test_captures_demographics_from_clearest(self):
        frames = [
            [_Face(_emb(1, 0, 0), score=0.6, sex="M", age=20)],
            [_Face(_emb(1, 0, 0), score=0.95, sex="M", age=31)],  # clearer
        ]
        fm = _catalog(_StubReader(frames), lambda f: f, stride=1)
        assert fm.identities[0].sex == "M"
        assert fm.identities[0].age == 31  # from the higher-score occurrence

    def test_empty_target_is_empty_catalog(self):
        fm = _catalog(_StubReader([]), lambda f: f, stride=1)
        assert fm.is_empty()


class TestStrideAndProgress:
    def test_stride_samples_every_nth(self):
        # 10 frames, all the same person; stride 5 → scan frames 0 and 5.
        frames = [_identity(1, 0, 0) for _ in range(10)]
        seen_frames = []
        analyze_target(
            _StubReader(frames),
            lambda f: (seen_frames.append(1) or f),
            stride=5,
        )
        assert len(seen_frames) == 2  # indices 0, 5

    def test_progress_reaches_total(self):
        frames = [_identity(1, 0, 0) for _ in range(6)]
        events = []
        analyze_target(
            _StubReader(frames), lambda f: f, stride=2,
            on_progress=lambda done, total: events.append((done, total)),
        )
        # indices 0,2,4 → 3 scans; final event hits 3/3.
        assert events[-1] == (3, 3)

    def test_threshold_propagates_to_catalog(self):
        fm = _catalog(
            _StubReader([_identity(1, 0)]), lambda f: f, stride=1, threshold=0.62
        )
        assert fm.threshold == 0.62


class TestSectionsAndPreview:
    def test_scan_confined_to_sections(self):
        from sinner2.pipeline.sections import SectionSet

        # 10 frames; only frames 2..4 selected → scan exactly those.
        frames = [_identity(1, 0, 0) for _ in range(10)]
        seen = []
        analyze_target(
            _StubReader(frames),
            lambda f: (seen.append(1) or f),
            stride=1, sections=SectionSet.of([(2, 4)]),
        )
        assert len(seen) == 3  # frames 2, 3, 4 only

    def test_sections_with_stride(self):
        from sinner2.pipeline.sections import SectionSet

        frames = [_identity(1, 0, 0) for _ in range(20)]
        seen = []
        analyze_target(
            _StubReader(frames),
            lambda f: (seen.append(1) or f),
            stride=2, sections=SectionSet.of([(4, 11)]),
        )
        # Selected 4..11 (8 frames), every 2nd → 4 scans.
        assert len(seen) == 4

    def test_empty_sections_scans_whole(self):
        from sinner2.pipeline.sections import SectionSet

        frames = [_identity(1, 0, 0) for _ in range(6)]
        seen = []
        analyze_target(
            _StubReader(frames), lambda f: (seen.append(1) or f),
            stride=1, sections=SectionSet.empty(),
        )
        assert len(seen) == 6

    def test_preview_callback_fires(self):
        frames = [_identity(1, 0, 0) for _ in range(4)]
        previews = []
        analyze_target(
            _StubReader(frames), lambda f: f, stride=1,
            on_preview=previews.append, preview_interval=0.0,
        )
        assert len(previews) == 4  # interval 0 → every scanned frame


class TestParallel:
    def test_parallel_matches_serial(self):
        # Two people alternating; workers=4 must cluster identically to workers=1.
        frames = []
        for k in range(12):
            frames.append([_Face(_emb(1, 0.02 * (k % 2), 0))])
            frames.append([_Face(_emb(0, 1, 0.02 * (k % 2)))])
        serial = _catalog(
            _StubReader(frames), lambda f: f, stride=1, threshold=0.5, workers=1
        )
        parallel = _catalog(
            _StubReader(frames), lambda f: f, stride=1, threshold=0.5, workers=4
        )
        assert len(serial.identities) == len(parallel.identities) == 2
        assert sorted(i.occurrences for i in parallel.identities) == [12, 12]

    def test_parallel_progress_reaches_total(self):
        frames = [_identity(1, 0, 0) for _ in range(10)]
        events = []
        analyze_target(
            _StubReader(frames), lambda f: f, stride=2, workers=3,
            on_progress=lambda d, t: events.append((d, t)),
        )
        assert events[-1] == (5, 5)  # indices 0,2,4,6,8

    def test_parallel_respects_sections(self):
        from sinner2.pipeline.sections import SectionSet

        frames = [_identity(1, 0, 0) for _ in range(10)]
        seen = []
        analyze_target(
            _StubReader(frames), lambda f: (seen.append(1) or f),
            stride=1, workers=4, sections=SectionSet.of([(3, 6)]),
        )
        assert len(seen) == 4  # frames 3,4,5,6


class TestCancellation:
    def test_cancel_returns_partial_catalog(self):
        frames = [_identity(1, 0, 0), _identity(0, 1, 0)]
        ev = threading.Event()

        def detect(frame):
            ev.set()  # cancel after the first frame is detected
            return frame

        fm = _catalog(
            _StubReader(frames), detect, stride=1, cancel_event=ev
        )
        # Loop checks cancel at the top → second frame never scanned.
        assert len(fm.identities) == 1


class TestPositionAndFirstFrame:
    def test_emits_scan_position(self):
        frames = [_identity(1, 0, 0) for _ in range(6)]
        positions = []
        analyze_target(
            _StubReader(frames), lambda f: f, stride=2,
            on_position=positions.append,
        )
        assert positions == [0, 2, 4]

    def test_first_frame_is_earliest_occurrence(self):
        # Person A at frames 0 and 4; first_frame must be 0 (not the clearest).
        frames = [
            [_Face(_emb(1, 0, 0), score=0.5)],   # frame 0
            [_Face(_emb(0, 1, 0), score=0.9)],   # frame 1 (person B)
            [_Face(_emb(1, 0, 0), score=0.99)],  # frame 2 (A again, clearer)
        ]
        fm = _catalog(_StubReader(frames), lambda f: f, stride=1)
        a = next(i for i in fm.identities if i.ref_frame == 2)  # clearest = frame 2
        assert a.first_frame == 0  # but earliest = frame 0

    def test_first_bbox_is_the_box_on_the_earliest_frame(self):
        # first_bbox is the box on the EARLIEST frame (0), NOT the clearest (2),
        # so navigating to first_frame draws the box that's actually there.
        frames = [
            [_Face(_emb(1, 0, 0), score=0.5, bbox=(0.0, 0.0, 4.0, 4.0))],   # f0 A
            [_Face(_emb(0, 1, 0), score=0.9, bbox=(8.0, 8.0, 9.0, 9.0))],   # f1 B
            [_Face(_emb(1, 0, 0), score=0.99, bbox=(5.0, 5.0, 9.0, 9.0))],  # f2 A
        ]
        fm = _catalog(_StubReader(frames), lambda f: f, stride=1)
        a = next(i for i in fm.identities if i.ref_frame == 2)
        assert a.first_frame == 0
        assert a.first_bbox == (0.0, 0.0, 4.0, 4.0)  # frame 0's box
        assert a.ref_bbox == (5.0, 5.0, 9.0, 9.0)    # frame 2's box (clearest)


class TestRepMetadata:
    def test_captures_det_score_and_pose(self):
        # The clearest occurrence's det_score + pose (pitch, yaw, roll) land on
        # the identity for the Faces table.
        import numpy as np

        f = _Face(_emb(1, 0, 0), score=0.88)
        f.pose = np.array([3.0, -7.0, 15.0])  # insightface: pitch, yaw, roll
        fm = _catalog(_StubReader([[f]]), lambda fr: fr, stride=1)
        ident = fm.identities[0]
        assert ident.det_score == 0.88
        assert ident.pitch == 3.0 and ident.yaw == -7.0 and ident.roll == 15.0

    def test_pose_none_without_full_pack(self):
        # Fast det+rec faces have no .pose → pitch/yaw/roll stay None.
        fm = _catalog(_StubReader([[_Face(_emb(1, 0, 0))]]), lambda fr: fr, stride=1)
        ident = fm.identities[0]
        assert ident.pitch is None and ident.yaw is None and ident.roll is None
        assert ident.det_score == 0.9  # det_score still captured


class TestResume:
    def test_returns_scanned_and_total(self):
        frames = [_identity(1, 0, 0) for _ in range(10)]
        _fm, scanned, total = analyze_target(
            _StubReader(frames), lambda f: f, stride=2
        )
        assert scanned == 5 and total == 5  # indices 0,2,4,6,8

    def test_start_index_skips_already_scanned(self):
        frames = [_identity(1, 0, 0) for _ in range(10)]
        seen = []
        _fm, scanned, total = analyze_target(
            _StubReader(frames), lambda f: (seen.append(1) or f),
            stride=2, start_index=3,
        )
        assert len(seen) == 2  # indices[3:] of [0,2,4,6,8] = [6,8]
        assert scanned == 5 and total == 5  # progress reflects the whole job

    def test_initial_seeds_clustering(self):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize as _n

        seed = FaceMap(
            identities=(Identity("a", _n([1, 0, 0]), source_path="/s.png", occurrences=4),)
        )
        frames = [[_Face(_emb(1, 0, 0))], [_Face(_emb(1, 0, 0))]]  # same person
        fm, _s, _t = analyze_target(
            _StubReader(frames), lambda f: f, stride=1, initial=seed, threshold=0.5
        )
        assert len(fm.identities) == 1
        assert fm.identities[0].id == "a"            # joined the seeded identity
        assert fm.identities[0].occurrences == 6     # 4 seeded + 2 new
        assert fm.identities[0].source_path == "/s.png"  # assignment preserved


class TestPrecomputeGeometry:
    """The full-frame geometry pass: matches each detected face to the catalog
    and records bbox + kps per frame (drops unmapped / embeddingless faces)."""

    def _face(self, emb, bbox=(0.0, 0.0, 4.0, 4.0), kps=None):
        f = _Face(_emb(*emb), bbox=bbox)
        f.kps = kps if kps is not None else [(float(i), float(i)) for i in range(5)]
        return f

    def _catalog2(self):
        from sinner2.pipeline.face_map import FaceMap, Identity

        return FaceMap(
            identities=(Identity("a", _emb(1, 0, 0)), Identity("b", _emb(0, 1, 0))),
            threshold=0.5,
        )

    def test_records_matched_faces_with_bbox_kps(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        frames = [
            [self._face((0.95, 0.05, 0), bbox=(0.0, 0.0, 4.0, 4.0))],   # ~a
            [self._face((0.05, 0.95, 0), bbox=(1.0, 1.0, 5.0, 5.0))],   # ~b
        ]
        geom, scanned, total = precompute_geometry(
            _StubReader(frames), lambda f: f, self._catalog2(), workers=1
        )
        assert total == 2 and scanned == 2 and geom.frame_count == 2
        assert geom.refined is False  # default: detect closure didn't refine
        f0 = geom.faces_at(0)[0]
        assert f0.identity_id == "a"
        assert f0.bbox == (0.0, 0.0, 4.0, 4.0)
        assert len(f0.kps) == 5
        assert geom.faces_at(1)[0].identity_id == "b"

    def test_bakes_real_embedding(self):
        # A0: the matched face's real (normalized) embedding is baked into the
        # geometry so the runtime can route against the live catalog.
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        emb = (0.95, 0.05, 0)
        geom, *_ = precompute_geometry(
            _StubReader([[self._face(emb)]]), lambda f: f, self._catalog2()
        )
        f0 = geom.faces_at(0)[0]
        assert f0.embedding == tuple(float(x) for x in _emb(*emb))

    def test_bakes_roll_from_face(self):
        # D5: the detect closure's `baked_roll` is recorded onto the GeomFace.
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        f = self._face((0.95, 0.05, 0))
        f.baked_roll = 17.0

        geom, *_ = precompute_geometry(
            _StubReader([[f]]), lambda fr: fr, self._catalog2()
        )
        assert geom.faces_at(0)[0].roll == 17.0

    def test_roll_none_when_face_has_no_baked_roll(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        geom, *_ = precompute_geometry(
            _StubReader([[self._face((1, 0, 0))]]), lambda fr: fr, self._catalog2()
        )
        assert geom.faces_at(0)[0].roll is None

    def test_refined_metadata_is_stamped(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        frames = [[self._face((1, 0, 0))]]
        geom, *_ = precompute_geometry(
            _StubReader(frames), lambda f: f, self._catalog2(), refined=True
        )
        assert geom.refined is True  # caller's detect closure pre-refined kps

    def test_skips_unmatched_and_embeddingless(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        stranger = self._face((0, 0, 1))      # orthogonal → below the bake floor
        noemb = _Face(None)
        noemb.kps = [(0.0, 0.0)] * 5
        geom, *_ = precompute_geometry(
            _StubReader([[stranger, noemb]]), lambda f: f, self._catalog2()
        )
        assert geom.is_empty()

    def test_bakes_below_threshold_but_above_floor(self):
        # #13: a face matching BELOW the catalog threshold but within the
        # permissive bake floor is still recorded (with its embedding) — so
        # LOWERING the threshold later recovers it in detection-free mode, instead
        # of geometry and live detection disagreeing across the change.
        from sinner2.pipeline.face_map import FaceMap, Identity
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        cat = FaceMap(identities=(Identity("a", _emb(1, 0, 0)),), threshold=0.5)
        borderline = self._face((0.4, 0.917, 0))  # cosine ~0.4 to "a": <0.5, >floor
        geom, *_ = precompute_geometry(
            _StubReader([[borderline]]), lambda f: f, cat
        )
        assert not geom.is_empty()  # baked despite being below the 0.5 threshold
        assert geom.faces_at(0)[0].identity_id == "a"

    def test_below_floor_is_still_dropped(self):
        # The floor isn't a free-for-all: a near-orthogonal face (below the floor)
        # is NOT baked, so the table doesn't bloat with impostor detections.
        from sinner2.pipeline.face_map import FaceMap, Identity
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        cat = FaceMap(identities=(Identity("a", _emb(1, 0, 0)),), threshold=0.5)
        impostor = self._face((0.2, 0.98, 0))  # cosine ~0.2 to "a": below the floor
        geom, *_ = precompute_geometry(
            _StubReader([[impostor]]), lambda f: f, cat
        )
        assert geom.is_empty()

    def test_skips_wrong_keypoint_count(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        bad = self._face((1, 0, 0), kps=[(0.0, 0.0)] * 3)  # not 5 kps
        geom, *_ = precompute_geometry(
            _StubReader([[bad]]), lambda f: f, self._catalog2()
        )
        assert geom.is_empty()

    def test_sections_confine_coverage(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry
        from sinner2.pipeline.sections import SectionSet

        frames = [[self._face((1, 0, 0))] for _ in range(5)]
        secs = SectionSet.of([(1, 2)])
        geom, scanned, total = precompute_geometry(
            _StubReader(frames), lambda f: f, self._catalog2(), sections=secs
        )
        plan = secs.frame_plan(5)
        assert total == len(plan)
        assert set(geom.faces.keys()) <= set(plan)  # nothing outside the section

    def test_cancel_returns_partial(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        ev = threading.Event()
        ev.set()  # cancelled before the first ingest
        frames = [[self._face((1, 0, 0))] for _ in range(3)]
        geom, scanned, _t = precompute_geometry(
            _StubReader(frames), lambda f: f, self._catalog2(), cancel_event=ev
        )
        assert geom.is_empty() and scanned == 0

    def test_parallel_path_records_all(self):
        from sinner2.pipeline.face_map_analyzer import precompute_geometry

        frames = [[self._face((1, 0, 0))] for _ in range(6)]
        geom, scanned, total = precompute_geometry(
            _StubReader(frames), lambda f: f, self._catalog2(), workers=3
        )
        assert geom.face_count() == 6 and scanned == 6 and total == 6
