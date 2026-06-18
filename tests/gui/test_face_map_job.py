"""Tests for the background face-map analysis job."""
from __future__ import annotations

from sinner2.gui.face_map_job import AnalysisRequest, FaceMapAnalysisJob
from sinner2.pipeline.face_map import FaceMap, normalize


class _Face:
    def __init__(self, embedding, score=0.9, bbox=(0.0, 0.0, 4.0, 4.0), kps=None):
        self.normed_embedding = embedding
        self.det_score = score
        self.bbox = bbox
        self.kps = kps if kps is not None else [(float(i), float(i)) for i in range(5)]


class _StubReader:
    def __init__(self, frames):
        self._frames = frames
        self.released = False

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
        self.released = True


class _StubLandmarker:
    """A 2dfan4 stand-in so geometry-phase tests stay hermetic — the real
    landmarker's setup() loads 2dfan4.onnx, which isn't present on CI. Returns a
    confident, all-zero 68-point set (callers gate on the score, not the coords)."""

    def detect_68(self, frame, bbox):
        import numpy as np
        return np.zeros((68, 2), np.float32), 0.99

    def release(self):
        pass


def _job(frames, *, detect=None):
    reader = _StubReader(frames)
    # bake_angle defaults on, so the geometry phase builds a landmarker — stub it
    # (the default factory would load the real 2dfan4 ONNX model).
    job = FaceMapAnalysisJob(
        reader_factory=lambda _path: reader,
        detect_factory=lambda _prov, _size, _fast, _det: (detect or (lambda f: f)),
        landmarker_factory=lambda _prov: _StubLandmarker(),
    )
    return job, reader


def _emb(*v):
    return normalize(v)


class TestRun:
    def test_finished_emits_catalog(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))], [_Face(_emb(0, 1, 0))]]
        job, reader = _job(frames)
        with qtbot.waitSignal(job.finished, timeout=2000) as blocker:
            job.run(AnalysisRequest(
                "clip.mp4", stride=1, providers=["CPUExecutionProvider"],
                compute_geometry=False,
            ))
        face_map = blocker.args[0]
        assert isinstance(face_map, FaceMap)
        assert len(face_map.identities) == 2
        assert blocker.args[1] is None  # geometry skipped
        assert reader.released  # reader cleaned up

    def test_progress_emitted(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))] for _ in range(4)]
        job, _ = _job(frames)
        events = []
        job.progress.connect(lambda d, t: events.append((d, t)))
        with qtbot.waitSignal(job.finished, timeout=2000):
            job.run(AnalysisRequest("clip.mp4", stride=2, compute_geometry=False))
        assert events[-1] == (2, 2)  # indices 0, 2

    def test_geometry_phase_builds_table(self, qtbot):
        # Phase 2: a full-frame geometry pass matches each face to the catalog.
        frames = [[_Face(_emb(1, 0, 0))], [_Face(_emb(1, 0, 0))]]
        job, _ = _job(frames)
        started = []
        job.geometryStarted.connect(lambda: started.append(1))
        with qtbot.waitSignal(job.finished, timeout=2000) as blocker:
            job.run(AnalysisRequest("clip.mp4", stride=1))  # compute_geometry default
        geometry = blocker.args[1]
        assert started == [1]
        assert geometry is not None
        assert geometry.face_count() == 2  # one mapped face per frame
        assert geometry.refined is False   # landmark_refine off → raw kps

    def test_geometry_phase_uses_fast_detector(self, qtbot):
        # The catalog scan may use the slow age/sex pack, but the geometry pass
        # must build the fast det+rec detector (fast=True) — no genderage/frame.
        fasts = []
        reader = _StubReader([[_Face(_emb(1, 0, 0))]])
        job = FaceMapAnalysisJob(
            reader_factory=lambda _p: reader,
            detect_factory=lambda _prov, _size, fast, _det: (
                fasts.append(fast) or (lambda f: f)
            ),
            landmarker_factory=lambda _prov: _StubLandmarker(),  # bake_angle on
        )
        with qtbot.waitSignal(job.finished, timeout=2000):
            job.run(AnalysisRequest("clip.mp4", stride=1, fast=False))  # age/sex ON
        assert fasts == [False, True]  # phase 1 full-pack, phase 2 det+rec only

    def test_landmark_refine_bakes_and_flags_refined(self, qtbot):
        # With landmark_refine on, the job builds a landmarker and stamps refined.
        frames = [[_Face(_emb(1, 0, 0))]]
        reader = _StubReader(frames)
        job = FaceMapAnalysisJob(
            reader_factory=lambda _p: reader,
            detect_factory=lambda _prov, _size, _fast, _det: (lambda f: f),
            landmarker_factory=lambda _prov: _StubLandmarker(),
        )
        with qtbot.waitSignal(job.finished, timeout=2000) as blocker:
            job.run(AnalysisRequest("clip.mp4", stride=1, landmark_refine=True))
        geometry = blocker.args[1]
        assert geometry is not None and geometry.refined is True

    def test_geometry_detect_bakes_roll_from_landmarks(self):
        # D5: with bake_angle on, the geometry detector attaches a per-face roll
        # measured from the 2dfan4 eye-line (45° here) — even with refine off.
        import numpy as np

        class _LM:
            def detect_68(self, frame, bbox):
                lm = np.zeros((68, 2), np.float32)
                lm[36:42] = [0, 0]      # left eye centre
                lm[42:48] = [10, 10]    # right eye centre → 45°
                return lm, 0.99

        base = [_Face(_emb(1, 0, 0))]
        detect = FaceMapAnalysisJob._geometry_detect(  # noqa: SLF001
            lambda _fr: base, _LM(), 0.5, refine=False, bake_angle=True
        )
        out = detect(object())
        assert abs(out[0].baked_roll - 45.0) < 1e-3
        # refine off → keypoints untouched
        assert base[0].kps == [(float(i), float(i)) for i in range(5)]

    def test_geometry_detect_no_landmarker_is_passthrough(self):
        base = [_Face(_emb(1, 0, 0))]
        detect = FaceMapAnalysisJob._geometry_detect(  # noqa: SLF001
            lambda _fr: base, None, 0.5, refine=False, bake_angle=False
        )
        out = detect(object())
        assert out is base and not hasattr(out[0], "baked_roll")

    def test_reader_failure_emits_failed(self, qtbot):
        def boom(_path):
            raise OSError("no such file")

        job = FaceMapAnalysisJob(
            reader_factory=boom,
            detect_factory=lambda _p, _s, _fast, _det: (lambda f: f),
        )
        with qtbot.waitSignal(job.failed, timeout=2000) as blocker:
            job.run(AnalysisRequest("missing.mp4", stride=1))
        assert "cannot open target" in blocker.args[0]

    def test_detect_failure_emits_failed_and_releases(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))]]
        reader = _StubReader(frames)

        def bad_detect(_frame):
            raise RuntimeError("detector exploded")

        job = FaceMapAnalysisJob(
            reader_factory=lambda _p: reader,
            detect_factory=lambda _prov, _size, _fast, _det: bad_detect,
        )
        with qtbot.waitSignal(job.failed, timeout=2000) as blocker:
            job.run(AnalysisRequest("clip.mp4", stride=1))
        assert "detector exploded" in blocker.args[0]
        assert reader.released

    def test_cancel_stops_the_scan(self, qtbot):
        frames = [[_Face(_emb(1, 0, 0))], [_Face(_emb(0, 1, 0))]]
        job_ref: list = []

        def detect(frame):
            job_ref[0].cancel()  # cancel after the first frame
            return frame

        job, _ = _job(frames, detect=detect)
        job_ref.append(job)
        with qtbot.waitSignal(job.finished, timeout=2000) as blocker:
            job.run(AnalysisRequest("clip.mp4", stride=1))
        assert len(blocker.args[0].identities) == 1  # second frame not scanned


class TestDetectorRelease:
    def test_scan_releases_the_detector(self, qtbot):
        # The scan's detect fn owns a FaceAnalyser whose standalone detector
        # holds an ONNX session — the job must release it (no per-scan leak).
        reader = _StubReader([[_Face(_emb(1, 0, 0))]])
        released = []

        class _Det:
            def __call__(self, f):
                return f
            def release(self):
                released.append(1)

        job = FaceMapAnalysisJob(
            reader_factory=lambda _p: reader,
            detect_factory=lambda *_a: _Det(),
        )
        with qtbot.waitSignal(job.finished, timeout=2000):
            job.run(AnalysisRequest("clip.mp4", stride=1, compute_geometry=False))
        assert released == [1]  # the catalog detector was released


class TestSharedPackPin:
    def test_scan_pins_the_shared_pack_during_inference(self, qtbot):
        # While the scan infers, the shared buffalo_l pack is pinned so a
        # concurrent providers/det-size change can't tear it down under the scan
        # workers (the teardown defers); the pin releases once the scan ends.
        from sinner2.pipeline import face_analyser

        seen: list[int] = []

        def detect(frame):
            seen.append(face_analyser._shared_pins)  # noqa: SLF001
            return frame

        frames = [[_Face(_emb(1, 0, 0))], [_Face(_emb(0, 1, 0))]]
        job, _ = _job(frames, detect=detect)
        with qtbot.waitSignal(job.finished, timeout=2000):
            job.run(AnalysisRequest("clip.mp4", stride=1, compute_geometry=False))
        assert seen and all(p >= 1 for p in seen)  # pinned during every detect
        assert face_analyser._shared_pins == 0  # noqa: SLF001  # released after
