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


def _params(**overrides) -> FaceSwapperParams:
    """Legacy-paste params for the orchestration tests below: their backend
    stubs echo the frame from get(paste_back=True), which is insightface's
    ORIGINAL paste surface. The FastPaste adapter (fast_paste=True — the
    default) consumes get(paste_back=False)'s (crop, matrix) contract instead
    and has its own dedicated tests; wrapping it here would change what these
    tests assert (orchestration: filters, sinks, crops — not the blend)."""
    overrides.setdefault("fast_paste", False)
    return FaceSwapperParams(**overrides)


def _blank() -> Frame:
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestThreadSafety:
    def test_swapper_is_thread_safe(self):
        assert FaceSwapper.thread_safe is True


class TestLandmarkRefine:
    """landmark_refine builds a 2dfan4 landmarker at setup() and replaces each
    face's kps with the 68->5 derived points before swapping + publishing."""

    def _stub_landmarker(self, monkeypatch, kps5, score=0.9):
        from sinner2.pipeline.processors import face_swapper as fs_mod

        class _StubLM:
            def __init__(self, *a, **k):
                pass

            def setup(self):
                pass

            def detect_68(self, frame, bbox):
                # 68 points whose 36:42 / 42:48 / 30 / 48 / 54 give kps5.
                lm = np.zeros((68, 2), np.float32)
                lm[36:42] = kps5[0]
                lm[42:48] = kps5[1]
                lm[30] = kps5[2]
                lm[48] = kps5[3]
                lm[54] = kps5[4]
                return lm, score

            def release(self):
                pass

        monkeypatch.setattr(fs_mod, "FaceLandmarker", _StubLM)

    def test_no_landmarker_built_by_default(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper
    ):
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
        fs.setup()
        assert fs._landmarker is None  # noqa: SLF001

    def test_refine_replaces_kps_and_publishes_refined(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper,
        monkeypatch,
    ):
        from sinner2.pipeline.processor import ChainContext

        refined = np.array(
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], np.float32
        )
        self._stub_landmarker(monkeypatch, refined)
        face = MagicMock(name="t")
        face.bbox = np.array([10, 10, 40, 40], float)
        face.sex = "M"
        face.kps = np.zeros((5, 2), np.float32)  # the detector's (to be replaced)
        stub_insightface_app.get.side_effect = [[MagicMock(name="src")], [face]]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(landmark_refine=True, rotation_compensation=False),
        )
        fs.setup()
        assert fs._landmarker is not None  # noqa: SLF001
        ctx = ChainContext()
        fs.process(_blank(), ctx)
        np.testing.assert_allclose(face.kps, refined)  # kps refined in place
        assert ctx.faces[0] is face  # refined face published downstream

    def test_low_score_leaves_detector_kps(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper,
        monkeypatch,
    ):
        refined = np.array(
            [[9, 9], [9, 9], [9, 9], [9, 9], [9, 9]], np.float32
        )
        self._stub_landmarker(monkeypatch, refined, score=0.2)  # below 0.5
        original = np.full((5, 2), 7.0, np.float32)
        face = MagicMock(name="t")
        face.bbox = np.array([10, 10, 40, 40], float)
        face.sex = "M"
        face.kps = original.copy()
        stub_insightface_app.get.side_effect = [[MagicMock(name="src")], [face]]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(landmark_refine=True, rotation_compensation=False),
        )
        fs.setup()
        fs.process(_blank())
        np.testing.assert_allclose(face.kps, original)  # untouched

    def test_landmark_68_angle_source_builds_landmarker(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper,
        monkeypatch,
    ):
        from sinner2.pipeline.processors.face_swapper import RotationAngleSource

        self._stub_landmarker(monkeypatch, np.zeros((5, 2), np.float32))
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_angle_source=RotationAngleSource.LANDMARK_68),
        )
        fs.setup()
        # Built even though landmark_refine is False — the angle source needs it.
        assert fs._landmarker is not None  # noqa: SLF001


class TestChainContextPublish:
    def test_publishes_prefilter_faces_to_context(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper
    ):
        # The swapper publishes its PRE-filter detections (the enhancer
        # restores every detected face, including ones the sex filter skips
        # for swapping), so downstream consumers see exactly what their own
        # re-detection would have produced.
        from unittest.mock import MagicMock

        from sinner2.pipeline.processor import ChainContext

        f_male = MagicMock(name="male")
        f_male.sex = "M"
        f_male.bbox = np.array([0, 0, 4, 4], float)
        f_female = MagicMock(name="female")
        f_female.sex = "F"
        f_female.bbox = np.array([5, 5, 9, 9], float)
        stub_insightface_app.get.side_effect = [
            [MagicMock(name="src")], [f_male, f_female],
        ]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_compensation=False, target_sex="F"),
        )
        fs.setup()
        ctx = ChainContext()
        fs.process(_blank(), ctx)
        assert ctx.faces == [f_male, f_female]  # pre-filter, both published

    def test_no_context_is_fine(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper
    ):
        from unittest.mock import MagicMock

        stub_insightface_app.get.side_effect = [
            [MagicMock(name="src")], [],
        ]
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
        fs.setup()
        out = fs.process(_blank())  # plain one-arg call still works
        assert out.shape == _blank().shape


class TestFastPasteWiring:
    """fast_paste (default ON) wraps the insightface backend in the FastPaste
    adapter at setup(); OFF keeps insightface's raw backend (original blend)."""

    def test_default_wraps_inswapper(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper
    ):
        fs = FaceSwapper(source=Source(path=source_image))  # defaults
        fs.setup()
        wrapped = fs._swapper  # noqa: SLF001
        assert wrapped.__class__.__name__ == "FastPasteSwapper"
        assert wrapped._inner is stub_inswapper  # noqa: SLF001

    def test_fast_paste_off_keeps_raw_backend(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper
    ):
        fs = FaceSwapper(
            source=Source(path=source_image), params=_params()
        )
        fs.setup()
        assert fs._swapper is stub_inswapper  # noqa: SLF001


class TestFaceSwapperParams:
    def test_defaults(self):
        p = FaceSwapperParams()
        assert p.detection_interval == 1
        assert p.many_faces is True
        assert p.fast_paste is True  # ROI feather blend is the default
        assert p.landmark_refine is False  # experimental, opt-in
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
        assert isinstance(FaceSwapper(source=Source(path=source_image), params=_params()), Processor)

    def test_process_before_setup_raises(self, source_image: Path):
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
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
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
        with pytest.raises(ValueError, match="no face detected"):
            fs.setup()

    def test_process_after_setup_returns_frame(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
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
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
        fs.setup()
        fs.process(_blank())
        assert stub_inswapper.get.call_count == 2


class TestSetSource:
    """set_source() re-points the swap at a new face WITHOUT reloading models —
    the fast path for live source-face changes."""

    def test_reanalyses_without_reloading_models(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper,
        tmp_path,
    ):
        face1, face2 = MagicMock(name="src1"), MagicMock(name="src2")
        stub_insightface_app.get.side_effect = [[face1], [face2]]
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
        fs.setup()
        assert fs._source_face is face1  # noqa: SLF001
        analyser, swapper = fs._analyser, fs._swapper  # noqa: SLF001
        src2 = tmp_path / "src2.png"
        cv2.imwrite(str(src2), np.full((32, 32, 3), 64, dtype=np.uint8))
        fs.set_source(Source(path=src2))
        assert fs._source_face is face2  # noqa: SLF001  re-analysed
        assert fs._analyser is analyser  # noqa: SLF001  analyser NOT rebuilt
        assert fs._swapper is swapper    # noqa: SLF001  model NOT reloaded
        assert fs._source.path == src2   # noqa: SLF001

    def test_before_setup_just_records_source(self, source_image, tmp_path):
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
        src2 = tmp_path / "s2.png"
        src2.write_bytes(b"x")
        fs.set_source(Source(path=src2))  # no models loaded → record only
        assert fs._source.path == src2   # noqa: SLF001
        assert fs._source_face is None   # noqa: SLF001

    def test_raises_when_new_source_has_no_face(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper,
        tmp_path,
    ):
        stub_insightface_app.get.side_effect = [[MagicMock()], []]
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
        fs.setup()
        src2 = tmp_path / "s2.png"
        cv2.imwrite(str(src2), np.zeros((8, 8, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="no face"):
            fs.set_source(Source(path=src2))


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
        from sinner2.pipeline.processors.face_swapper import SwapperModel

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
            params=_params(
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
        fs = FaceSwapper(source=Source(path=source_image), params=_params(), detection_sink=sink)
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
            params=_params(rotation_compensation=False),
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
            params=_params(rotation_compensation=False),
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
            def setup(self):
                pass

            def face_mask(self, _a):
                return None

        monkeypatch.setattr(
            face_swapper, "build_occlusion_masker", lambda *a, **k: _StubMasker()
        )
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
            params=_params(occlusion_mask=True, rotation_compensation=False),
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
            params=_params(occlusion_mask=False),
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
            params=_params(
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
            params=_params(
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
        fs = FaceSwapper(source=Source(path=source_image), params=_params(), detection_sink=sink)
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
            params=_params(many_faces=False),
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
            params=_params(target_sex=TargetSex.FEMALE),
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
            params=_params(target_sex=TargetSex.BOTH),
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
            params=_params(target_sex=TargetSex.AS_SOURCE),
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
            params=_params(target_sex=TargetSex.MALE),
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
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
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
            params=_params(
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
        fs = FaceSwapper(source=Source(path=source_image), params=_params())
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
