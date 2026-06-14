"""Tests for the background face-map analysis job."""
from __future__ import annotations

import threading

from sinner2.gui.face_map_job import FaceMapAnalysisJob
from sinner2.pipeline.face_map import FaceMap, normalize


class _Face:
    def __init__(self, embedding, score=0.9, bbox=(0.0, 0.0, 4.0, 4.0)):
        self.normed_embedding = embedding
        self.det_score = score
        self.bbox = bbox


class _StubReader:
    def __init__(self, frames):
        self._frames = frames
        self.released = False

    @property
    def frame_count(self):
        return len(self._frames)

    def read(self, idx):
        return self._frames[idx] if 0 <= idx < len(self._frames) else None

    def release(self):
        self.released = True


def _job(frames, *, detect=None):
    reader = _StubReader(frames)
    job = FaceMapAnalysisJob(
        reader_factory=lambda _path: reader,
        detect_factory=lambda _prov, _size: (detect or (lambda f: f)),
    )
    return job, reader


def _emb(*v):
    return normalize(v)


class TestRun:
    def test_finished_emits_catalog(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))], [_Face(_emb(0, 1, 0))]]
        job, reader = _job(frames)
        with qtbot.waitSignal(job.finished, timeout=2000) as blocker:
            job.run("clip.mp4", 1, 0.5, ["CPUExecutionProvider"], 640)
        face_map = blocker.args[0]
        assert isinstance(face_map, FaceMap)
        assert len(face_map.identities) == 2
        assert reader.released  # reader cleaned up

    def test_progress_emitted(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))] for _ in range(4)]
        job, _ = _job(frames)
        events = []
        job.progress.connect(lambda d, t: events.append((d, t)))
        with qtbot.waitSignal(job.finished, timeout=2000):
            job.run("clip.mp4", 2, 0.5, None, 640)
        assert events[-1] == (2, 2)  # indices 0, 2

    def test_reader_failure_emits_failed(self, qtbot):
        def boom(_path):
            raise OSError("no such file")

        job = FaceMapAnalysisJob(
            reader_factory=boom, detect_factory=lambda _p, _s: (lambda f: f)
        )
        with qtbot.waitSignal(job.failed, timeout=2000) as blocker:
            job.run("missing.mp4", 1, 0.5, None, 640)
        assert "cannot open target" in blocker.args[0]

    def test_detect_failure_emits_failed_and_releases(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))]]
        reader = _StubReader(frames)

        def bad_detect(_frame):
            raise RuntimeError("detector exploded")

        job = FaceMapAnalysisJob(
            reader_factory=lambda _p: reader,
            detect_factory=lambda _prov, _size: bad_detect,
        )
        with qtbot.waitSignal(job.failed, timeout=2000) as blocker:
            job.run("clip.mp4", 1, 0.5, None, 640)
        assert "detector exploded" in blocker.args[0]
        assert reader.released

    def test_cancel_stops_the_scan(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))], [_Face(_emb(0, 1, 0))]]
        ev_seen = threading.Event()
        job_ref: list = []

        def detect(frame):
            job_ref[0].cancel()  # cancel after the first frame
            return frame

        job, _ = _job(frames, detect=detect)
        job_ref.append(job)
        with qtbot.waitSignal(job.finished, timeout=2000) as blocker:
            job.run("clip.mp4", 1, 0.5, None, 640)
        assert len(blocker.args[0].identities) == 1  # second frame not scanned
