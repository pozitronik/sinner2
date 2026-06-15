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
