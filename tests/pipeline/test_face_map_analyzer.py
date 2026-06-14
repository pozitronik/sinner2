"""Tests for the strided face-map analysis pass."""
from __future__ import annotations

import threading

from sinner2.pipeline.face_map import normalize
from sinner2.pipeline.face_map_analyzer import analyze_target


class _Face:
    def __init__(self, embedding, score=0.9, bbox=(0.0, 0.0, 4.0, 4.0)):
        self.normed_embedding = embedding
        self.det_score = score
        self.bbox = bbox


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
        fm = analyze_target(_StubReader(frames), lambda f: f, stride=1, threshold=0.5)
        assert len(fm.identities) == 2
        counts = sorted(i.occurrences for i in fm.identities)
        assert counts == [1, 2]

    def test_records_clearest_occurrence(self):
        frames = [
            [_Face(_emb(1, 0, 0), score=0.5, bbox=(0.0, 0.0, 4.0, 4.0))],
            [_Face(_emb(1, 0, 0), score=0.9, bbox=(1.0, 1.0, 5.0, 5.0))],  # clearer
        ]
        fm = analyze_target(_StubReader(frames), lambda f: f, stride=1)
        ident = fm.identities[0]
        assert ident.occurrences == 2
        assert ident.ref_frame == 1  # the higher-score detection
        assert ident.ref_bbox == (1.0, 1.0, 5.0, 5.0)

    def test_face_without_embedding_skipped(self):
        fm = analyze_target(
            _StubReader([[_Face(None)]]), lambda f: f, stride=1
        )
        assert fm.is_empty()

    def test_empty_target_is_empty_catalog(self):
        fm = analyze_target(_StubReader([]), lambda f: f, stride=1)
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
        fm = analyze_target(
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


class TestCancellation:
    def test_cancel_returns_partial_catalog(self):
        frames = [_identity(1, 0, 0), _identity(0, 1, 0)]
        ev = threading.Event()

        def detect(frame):
            ev.set()  # cancel after the first frame is detected
            return frame

        fm = analyze_target(
            _StubReader(frames), detect, stride=1, cancel_event=ev
        )
        # Loop checks cancel at the top → second frame never scanned.
        assert len(fm.identities) == 1
