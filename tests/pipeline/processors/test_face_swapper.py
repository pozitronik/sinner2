from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from sinner2.config.source import Source
from sinner2.pipeline import face_analyser
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors import face_swapper
from sinner2.pipeline.processors.face_swapper import (
    FaceSwapper,
    FaceSwapperParams,
    TargetSex,
    _face_matches,
)
from sinner2.types import Frame


@pytest.fixture
def stub_insightface_app(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    stub.get = MagicMock(return_value=[MagicMock(name="Face")])
    monkeypatch.setattr(face_analyser, "_get_shared_face_analysis", lambda *a, **k: stub)
    face_analyser.reset_shared_face_analysis()
    return stub


@pytest.fixture
def stub_inswapper(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    swapper = MagicMock()
    swapper.get = MagicMock(side_effect=lambda f, *a, **k: f)
    monkeypatch.setattr(face_swapper, "_load_inswapper", lambda *a, **k: swapper)
    return swapper


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    p = tmp_path / "src.png"
    cv2.imwrite(str(p), np.full((32, 32, 3), 128, dtype=np.uint8))
    return p


@pytest.fixture
def models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    md = tmp_path / "models"
    md.mkdir()
    (md / "inswapper_128.onnx").write_bytes(b"x")
    monkeypatch.setenv("SINNER2_MODELS_DIR", str(md))
    return md


def _blank() -> Frame:
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestThreadSafety:
    def test_swapper_is_thread_safe(self):
        assert FaceSwapper.thread_safe is True


class TestFaceSwapperParams:
    def test_defaults(self):
        p = FaceSwapperParams()
        assert p.detection_interval == 1
        assert p.many_faces is True
        assert p.target_sex is TargetSex.BOTH

    def test_rejects_zero_interval(self):
        with pytest.raises(Exception):
            FaceSwapperParams(detection_interval=0)

    def test_target_sex_accepts_string_literals(self):
        # str-Enum lets settings round-trip as the single-letter token
        # ("M"/"F"/"B"/"I"). Verify pydantic coerces correctly.
        assert FaceSwapperParams(target_sex="M").target_sex is TargetSex.MALE
        assert FaceSwapperParams(target_sex="F").target_sex is TargetSex.FEMALE
        assert FaceSwapperParams(target_sex="B").target_sex is TargetSex.BOTH
        assert FaceSwapperParams(target_sex="I").target_sex is TargetSex.AS_SOURCE


class TestFaceSwapper:
    def test_name(self):
        assert FaceSwapper.name == "FaceSwapper"

    def test_compliant_with_processor_protocol(self, source_image: Path):
        assert isinstance(FaceSwapper(source=Source(path=source_image)), Processor)

    def test_process_before_setup_raises(self, source_image: Path):
        fs = FaceSwapper(source=Source(path=source_image))
        with pytest.raises(RuntimeError, match="before setup"):
            fs.process(_blank())

    def test_setup_raises_when_source_has_no_face(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        stub_insightface_app.get.return_value = []
        fs = FaceSwapper(source=Source(path=source_image))
        with pytest.raises(ValueError, match="no face detected"):
            fs.setup()

    def test_process_after_setup_returns_frame(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        fs = FaceSwapper(source=Source(path=source_image))
        fs.setup()
        out = fs.process(_blank())
        assert out.shape == (10, 10, 3)

    def test_process_calls_swapper_per_target_face(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        stub_insightface_app.get.side_effect = [
            [MagicMock(name="src")],
            [MagicMock(name="t1"), MagicMock(name="t2")],
        ]
        fs = FaceSwapper(source=Source(path=source_image))
        fs.setup()
        fs.process(_blank())
        assert stub_inswapper.get.call_count == 2


class TestModelDispatch:
    def test_generic_model_builds_generic_backend(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # A non-insightface model (ghost) must route through GenericOnnxSwapper,
        # NOT _load_inswapper — and prepare_source must run after source detect.
        from sinner2.pipeline.processors.face_swapper import (
            FaceSwapperParams,
            SwapperModel,
        )

        events: list[str] = []

        class _StubBackend:
            def __init__(self, *a, **k):
                pass

            def setup(self):
                events.append("setup")

            def prepare_source(self, img, face):
                events.append("prepare_source")

            def get(self, frame, *a, **k):
                return frame

        def _boom(*a, **k):
            raise AssertionError("_load_inswapper must not run for a ghost model")

        monkeypatch.setattr(face_swapper, "GenericOnnxSwapper", _StubBackend)
        monkeypatch.setattr(face_swapper, "_load_inswapper", _boom)

        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(
                model=SwapperModel.GHOST_2_256, rotation_compensation=False
            ),
        )
        fs.setup()
        assert events == ["setup", "prepare_source"]
        assert fs.process(_blank()).shape == (10, 10, 3)

    def test_publishes_detections_to_sink(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        t1, t2 = MagicMock(name="t1"), MagicMock(name="t2")
        stub_insightface_app.get.side_effect = [
            [MagicMock(name="src")],  # setup: source face
            [t1, t2],                 # process: target faces (all published)
        ]
        sink = MagicMock()
        fs = FaceSwapper(source=Source(path=source_image), detection_sink=sink)
        fs.setup()
        fs.process(_blank())  # 10x10 blank
        sink.publish.assert_called_once()
        args = sink.publish.call_args.args
        assert args[0] == [t1, t2]
        assert args[1:] == (10, 10)  # width, height

    def test_publishes_comparison_crops_when_wanted(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        face = MagicMock(name="t1")
        face.bbox = np.array([2, 2, 8, 8], float)
        face.sex = "M"
        stub_insightface_app.get.side_effect = [[MagicMock(name="src")], [face]]
        sink = MagicMock()
        sink.wants_crops.return_value = True
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(rotation_compensation=False),
            detection_sink=sink,
        )
        fs.setup()
        fs.process(np.full((10, 10, 3), 50, np.uint8))
        sink.publish_crops.assert_called_once()
        pairs = sink.publish_crops.call_args.args[0]
        assert len(pairs) == 1  # one swapped face → one (bbox, orig, swap) pair

    def test_no_comparison_crops_when_not_wanted(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        face = MagicMock(name="t1")
        face.bbox = np.array([2, 2, 8, 8], float)
        face.sex = "M"
        stub_insightface_app.get.side_effect = [[MagicMock(name="src")], [face]]
        sink = MagicMock()
        sink.wants_crops.return_value = False
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(rotation_compensation=False),
            detection_sink=sink,
        )
        fs.setup()
        fs.process(np.full((10, 10, 3), 50, np.uint8))
        sink.publish_crops.assert_not_called()

    def test_occlusion_mask_applied_when_enabled(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class _StubMasker:
            def __init__(self, *a, **k):
                pass

            def setup(self):
                pass

            def face_mask(self, _a):
                return None

        monkeypatch.setattr(face_swapper, "OcclusionMasker", _StubMasker)
        called: list = []
        monkeypatch.setattr(
            face_swapper, "apply_occlusion",
            lambda before, swapped, face, masker: called.append(True) or swapped,
        )
        face = MagicMock(name="t1")
        face.bbox = np.array([10, 10, 30, 30], float)
        face.sex = "M"
        stub_insightface_app.get.side_effect = [[MagicMock(name="src")], [face]]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(occlusion_mask=True, rotation_compensation=False),
        )
        fs.setup()
        fs.process(_blank())
        assert called == [True]

    def test_occlusion_mask_not_built_when_disabled(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(occlusion_mask=False),
        )
        fs.setup()
        assert fs._masker is None  # noqa: SLF001 — not built when off

    def test_rotation_compensation_uprights_tilted_face(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        called: list = []
        monkeypatch.setattr(
            face_swapper,
            "swap_with_uprighting",
            lambda result, *a, **k: called.append(True) or result,
        )
        tilted = MagicMock(name="tilted")
        tilted.bbox = np.array([10, 10, 30, 30], float)
        tilted.kps = np.array([[12, 12], [28, 28]], float)  # 45° roll
        tilted.sex = "M"
        stub_insightface_app.get.side_effect = [[MagicMock(name="src")], [tilted]]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(
                rotation_compensation=True, rotation_threshold_deg=15
            ),
        )
        fs.setup()
        fs.process(_blank())
        assert called == [True]  # uprighting path used
        assert stub_inswapper.get.call_count == 0  # not the direct path

    def test_rotation_compensation_skips_near_upright_face(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        called: list = []
        monkeypatch.setattr(
            face_swapper,
            "swap_with_uprighting",
            lambda result, *a, **k: called.append(True) or result,
        )
        upright = MagicMock(name="upright")
        upright.bbox = np.array([10, 10, 30, 30], float)
        upright.kps = np.array([[12, 20], [28, 20]], float)  # level eyes → 0°
        upright.sex = "M"
        stub_insightface_app.get.side_effect = [[MagicMock(name="src")], [upright]]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(
                rotation_compensation=True, rotation_threshold_deg=15
            ),
        )
        fs.setup()
        fs.process(_blank())
        assert called == []  # below threshold → no uprighting
        assert stub_inswapper.get.call_count == 1  # plain direct swap

    def test_sink_failure_does_not_break_swap(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        stub_insightface_app.get.side_effect = [
            [MagicMock(name="src")],
            [MagicMock(name="t1")],
        ]
        sink = MagicMock()
        sink.publish.side_effect = RuntimeError("overlay exploded")
        fs = FaceSwapper(source=Source(path=source_image), detection_sink=sink)
        fs.setup()
        fs.process(_blank())  # must not raise despite the sink failing
        assert stub_inswapper.get.call_count == 1

    def test_single_face_mode_swaps_only_first(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        stub_insightface_app.get.side_effect = [
            [MagicMock(name="src")],
            [MagicMock(name="t1"), MagicMock(name="t2")],
        ]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(many_faces=False),
        )
        fs.setup()
        fs.process(_blank())
        assert stub_inswapper.get.call_count == 1

    def test_target_sex_filters_processed_faces(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        # Source face is male; targets include one male and one female.
        # With target_sex=FEMALE, only the female target should be
        # swapped — verifies the filter actually narrows the pass-through
        # from analyser to swapper.
        src = MagicMock(sex="M")
        male = MagicMock(sex="M")
        female = MagicMock(sex="F")
        stub_insightface_app.get.side_effect = [
            [src],         # setup-time source detection
            [male, female],  # process-time target frame
        ]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(target_sex=TargetSex.FEMALE),
        )
        fs.setup()
        fs.process(_blank())
        # Only one swap should have happened (for the female face).
        assert stub_inswapper.get.call_count == 1
        # And the face passed to the swapper was the female one.
        called_face = stub_inswapper.get.call_args.args[1]
        assert called_face is female

    def test_target_sex_both_swaps_every_face_regardless_of_sex(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        stub_insightface_app.get.side_effect = [
            [MagicMock(sex="M")],  # source
            [MagicMock(sex="M"), MagicMock(sex="F"), MagicMock(sex=None)],
        ]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(target_sex=TargetSex.BOTH),
        )
        fs.setup()
        fs.process(_blank())
        # All three target faces processed (BOTH is the no-filter path).
        assert stub_inswapper.get.call_count == 3

    def test_target_sex_as_source_resolves_to_source_face_sex(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        # Source is female; AS_SOURCE must restrict targets to female.
        female_src = MagicMock(sex="F")
        stub_insightface_app.get.side_effect = [
            [female_src],
            [MagicMock(sex="M"), MagicMock(sex="F"), MagicMock(sex="M")],
        ]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(target_sex=TargetSex.AS_SOURCE),
        )
        fs.setup()
        fs.process(_blank())
        assert stub_inswapper.get.call_count == 1

    def test_target_sex_with_unknown_face_sex_skips(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        # Faces with no sex attribute (older insightface model, or
        # the gender head was wrong on a degenerate crop) get skipped
        # under M/F filters. Better to miss one than swap the wrong
        # gender — matches sinner1's behaviour.
        stub_insightface_app.get.side_effect = [
            [MagicMock(sex="M")],
            [MagicMock(sex=None), MagicMock(sex="M")],
        ]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(target_sex=TargetSex.MALE),
        )
        fs.setup()
        fs.process(_blank())
        # Only the explicitly-M face was swapped.
        assert stub_inswapper.get.call_count == 1


class TestFaceMatchesHelper:
    def test_both_matches_any(self):
        assert _face_matches(MagicMock(sex="M"), TargetSex.BOTH)
        assert _face_matches(MagicMock(sex="F"), TargetSex.BOTH)
        assert _face_matches(MagicMock(sex=None), TargetSex.BOTH)

    def test_male_filter(self):
        assert _face_matches(MagicMock(sex="M"), TargetSex.MALE)
        assert not _face_matches(MagicMock(sex="F"), TargetSex.MALE)
        assert not _face_matches(MagicMock(sex=None), TargetSex.MALE)

    def test_female_filter(self):
        assert _face_matches(MagicMock(sex="F"), TargetSex.FEMALE)
        assert not _face_matches(MagicMock(sex="M"), TargetSex.FEMALE)

    def test_as_source_unresolved_falls_back_to_both(self):
        # Defensive: AS_SOURCE should normally be resolved upstream
        # via _resolved_target_sex, but if it leaks through (e.g.
        # bypass path) we don't want to silently skip every face.
        assert _face_matches(MagicMock(sex="M"), TargetSex.AS_SOURCE)
        assert _face_matches(MagicMock(sex="F"), TargetSex.AS_SOURCE)


    def test_release_makes_process_raise(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        fs = FaceSwapper(source=Source(path=source_image))
        fs.setup()
        fs.release()
        with pytest.raises(RuntimeError):
            fs.process(_blank())


class TestProviderResolution:
    """None (unspecified) → platform default; an explicit empty list (user
    unchecked all providers) stays empty so ORT falls back to CPU — no hidden
    GPU default is substituted."""

    def _capture_load(self, monkeypatch):
        captured: dict = {}

        def fake_load(_path, providers):
            captured["providers"] = providers
            m = MagicMock()
            m.get = MagicMock(side_effect=lambda f, *a, **k: f)
            return m

        monkeypatch.setattr(face_swapper, "_load_inswapper", fake_load)
        return captured

    def test_empty_providers_stay_empty(
        self, models_dir, source_image, stub_insightface_app, monkeypatch
    ):
        captured = self._capture_load(monkeypatch)
        fs = FaceSwapper(source=Source(path=source_image), providers=[])
        fs.setup()
        assert captured["providers"] == []

    def test_none_providers_use_default(
        self, models_dir, source_image, stub_insightface_app, monkeypatch
    ):
        from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS

        captured = self._capture_load(monkeypatch)
        fs = FaceSwapper(source=Source(path=source_image), providers=None)
        fs.setup()
        assert captured["providers"] == list(DEFAULT_ONNX_PROVIDERS)


class TestReleaseRaceContract:
    """process() must snapshot its backend handles so a concurrent release()
    (from a live chain swap, bounded inflight wait) can't null them mid-call and
    turn the worker's frame into a fatal None.get() AttributeError."""

    def test_process_survives_release_during_swap(
        self, models_dir, source_image, stub_insightface_app, stub_inswapper
    ):
        src = MagicMock(name="src")
        src.sex = "M"
        f1 = MagicMock(name="f1")
        f1.sex = "M"
        f1.bbox = np.array([0, 0, 4, 4], float)
        f2 = MagicMock(name="f2")
        f2.sex = "M"
        f2.bbox = np.array([4, 4, 8, 8], float)
        # setup() detects the source (1 face); process() detects 2 target faces.
        stub_insightface_app.get.side_effect = [[src], [f1, f2]]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=FaceSwapperParams(
                rotation_compensation=False, occlusion_mask=False, many_faces=True
            ),
        )
        fs.setup()

        calls = {"n": 0}

        def swap(img, target, source, paste_back=True):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate set_chain/reconfigure releasing this processor while a
                # worker is mid-process() (the bounded _wait_for_inflight expired).
                fs.release()
            return img

        stub_inswapper.get.side_effect = swap
        out = fs.process(np.zeros((10, 10, 3), dtype=np.uint8))  # must NOT raise
        assert calls["n"] == 2  # both faces swapped via the local snapshots
        assert out is not None


class TestReleaseFreesMasker:
    def test_release_releases_occlusion_masker(
        self, models_dir, source_image, stub_insightface_app, stub_inswapper
    ):
        # The torch occlusion masker holds CUDA memory; FaceSwapper.release()
        # must release it (not just drop the ref) so its VRAM is freed on a
        # chain rebuild — same as it does for the swap backend.
        fs = FaceSwapper(source=Source(path=source_image))
        fs.setup()
        masker = MagicMock()
        fs._masker = masker  # noqa: SLF001  simulate occlusion enabled
        fs.release()
        masker.release.assert_called_once()
        assert fs._masker is None  # noqa: SLF001


class TestOnSeekResetsDetectionCache:
    """A seek is a discontinuity — the interval-based detection cache from the
    old position must be dropped so the new position re-detects (else the swap
    lands at the old face box and the new face shows unswapped)."""

    def test_on_seek_resets_analyser_cache(self):
        fs = object.__new__(FaceSwapper)
        fs._analyser = MagicMock()  # noqa: SLF001
        fs.on_seek()
        fs._analyser.reset_cache.assert_called_once()  # noqa: SLF001

    def test_on_seek_without_analyser_is_noop(self):
        fs = object.__new__(FaceSwapper)
        fs._analyser = None  # noqa: SLF001
        fs.on_seek()  # must not raise
