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
    _CatalogMatcher,
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
        # ...but swapped_faces is the POST-filter subset: target_sex="F" swaps
        # only the female, so the enhancer's "only swapped" sees just her.
        assert ctx.swapped_faces == [f_female]

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
        assert args[1:3] == (10, 10)  # width, height
        assert args[3] is None  # no ctx → no frame index tag

    def test_publishes_detections_with_frame_index(
        self,
        source_image: Path,
        models_dir: Path,
        stub_insightface_app: MagicMock,
        stub_inswapper: MagicMock,
    ):
        from types import SimpleNamespace

        stub_insightface_app.get.side_effect = [
            [MagicMock(name="src")], [MagicMock(name="t")],
        ]
        sink = MagicMock()
        fs = FaceSwapper(
            source=Source(path=source_image), params=_params(), detection_sink=sink
        )
        fs.setup()
        # A ctx carries the frame index → the sink is tagged with it (so the GUI
        # can reject a stale click).
        fs.process(_blank(), SimpleNamespace(frame_index=42))
        assert sink.publish.call_args.args[3] == 42

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


class TestFaceMapping:
    """A FaceMap routes each detected face to a per-identity source (insightface
    backends only). Unmatched / unassigned / no-embedding faces are skipped."""

    def _sources(self, tmp_path):
        a = tmp_path / "alice.png"
        b = tmp_path / "bob.png"
        cv2.imwrite(str(a), np.full((32, 32, 3), 10, np.uint8))
        cv2.imwrite(str(b), np.full((32, 32, 3), 20, np.uint8))
        return a, b

    def _target_face(self, embedding):
        from types import SimpleNamespace

        return SimpleNamespace(
            normed_embedding=np.asarray(embedding, np.float32),
            bbox=np.array([0, 0, 4, 4], float),
            kps=np.zeros((5, 2), np.float32),
        )

    def test_routes_each_face_to_its_mapped_source(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper, tmp_path
    ):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        a_path, b_path = self._sources(tmp_path)
        src_a = object()  # sentinel "source face" objects the analyser returns
        src_b = object()
        t_a = self._target_face([1, 0, 0])     # → Alice
        t_b = self._target_face([0, 1, 0])     # → Bob
        t_stranger = self._target_face([0, 0, 1])  # matches neither
        # setup: main source, then the two mapped sources; then process targets.
        stub_insightface_app.get.side_effect = [
            [object()], [src_a], [src_b], [t_a, t_b, t_stranger],
        ]
        fm = FaceMap(
            identities=(
                Identity("a", normalize([1, 0, 0]), source_path=str(a_path)),
                Identity("b", normalize([0, 1, 0]), source_path=str(b_path)),
            ),
            threshold=0.5,
        )
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_compensation=False),
            face_map=fm,
        )
        fs.setup()
        assert fs._mapped_sources == {str(a_path): src_a, str(b_path): src_b}  # noqa: SLF001
        fs.process(_blank())
        # Two faces swapped; the stranger skipped. Each got ITS source.
        calls = stub_inswapper.get.call_args_list
        assert len(calls) == 2
        assert calls[0].args[2] is src_a  # t_a → Alice's source
        assert calls[1].args[2] is src_b  # t_b → Bob's source

    def test_face_without_embedding_is_skipped(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper, tmp_path
    ):
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        a_path, _ = self._sources(tmp_path)
        no_emb = SimpleNamespace(normed_embedding=None, bbox=np.zeros(4), kps=np.zeros((5, 2)))
        stub_insightface_app.get.side_effect = [
            [object()], [object()], [no_emb],
        ]
        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path=str(a_path)),),
            threshold=0.5,
        )
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_compensation=False),
            face_map=fm,
        )
        fs.setup()
        fs.process(_blank())
        stub_inswapper.get.assert_not_called()  # no embedding → can't route → skip

    def test_inactive_map_uses_global_source(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper, tmp_path
    ):
        from sinner2.pipeline.face_map import FaceMap

        # A map with identities but NO assigned sources is inactive → today's
        # single-source path (every face swapped with the global source).
        t = self._target_face([1, 0, 0])
        stub_insightface_app.get.side_effect = [[object()], [t]]
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_compensation=False),
            face_map=FaceMap.empty(),
        )
        fs.setup()
        fs.process(_blank())
        stub_inswapper.get.assert_called_once()  # global-source swap, not skipped

    def test_generic_backend_not_routable(self):
        # Bypass-init: a generic (non-insightface) backend doesn't support
        # per-call sources, so even an active map isn't routed (deferred).
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        fs = object.__new__(FaceSwapper)
        fs._supports_multi_source = False  # noqa: SLF001
        fs._face_map = FaceMap(  # noqa: SLF001
            identities=(Identity("a", normalize([1, 0]), source_path="/s.png"),)
        )
        assert fs._face_map_is_routable() is False  # noqa: SLF001
        fs._supports_multi_source = True  # noqa: SLF001
        assert fs._face_map_is_routable() is True  # noqa: SLF001


class TestCatalogMatcher:
    """The numpy hot-path matcher must route IDENTICALLY to FaceMap.source_for
    (the pure reference) — just via one GEMV. Verified across match / no-match /
    unassigned / every unmatched policy, on normalized queries (the runtime
    contract), and on the >= tie-break."""

    def _fm(self, **kw):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        idents = (
            Identity("a", normalize([1, 0, 0]), source_path="/a.png"),
            Identity("b", normalize([0, 1, 0]), source_path="/b.png"),
            Identity("c", normalize([0, 0, 1])),  # tracked but unassigned
        )
        return FaceMap(identities=idents, threshold=0.5, **kw)

    def _queries(self):
        from sinner2.pipeline.face_map import normalize

        # Clear winners + an unassigned hit + a stranger; kept off the exact
        # threshold boundary so float32-vs-float64 can't flip a decision.
        return [
            normalize([1, 0, 0]), normalize([0, 1, 0]), normalize([0, 0, 1]),
            normalize([1, 1, 0]), normalize([3, 1, 0]), normalize([1, 0, 5]),
            normalize([-1, -1, -1]),
        ]

    def _assert_parity(self, fm):
        m = _CatalogMatcher(fm)
        for q in self._queries():
            assert m.source_for(q) == fm.source_for(q)

    def test_parity_skip_policy(self):
        self._assert_parity(self._fm())  # SKIP is the default

    def test_parity_default_policy(self):
        from sinner2.pipeline.face_map import UnmatchedPolicy

        self._assert_parity(
            self._fm(unmatched=UnmatchedPolicy.DEFAULT, default_source="/def.png")
        )

    def test_parity_first_policy(self):
        from sinner2.pipeline.face_map import UnmatchedPolicy

        self._assert_parity(self._fm(unmatched=UnmatchedPolicy.FIRST))

    def test_by_id_indexes_every_identity(self):
        m = _CatalogMatcher(self._fm())
        assert set(m.by_id) == {"a", "b", "c"}
        assert m.by_id["a"].source_path == "/a.png"

    def test_empty_map_routes_via_policy(self):
        from sinner2.pipeline.face_map import (
            FaceMap,
            UnmatchedPolicy,
            normalize,
        )

        fm = FaceMap.empty().with_unmatched(UnmatchedPolicy.DEFAULT, "/d.png")
        m = _CatalogMatcher(fm)
        q = normalize([1, 2, 3])
        assert m.source_for(q) == fm.source_for(q) == "/d.png"

    def test_tie_goes_to_first_identity_like_pure(self):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        # Identical centroids, different sources: best_match keeps the FIRST on an
        # exact tie (deterministic by catalog order); the matcher must agree.
        fm = FaceMap(
            identities=(
                Identity("x", normalize([1, 0, 0]), source_path="/x.png"),
                Identity("y", normalize([1, 0, 0]), source_path="/y.png"),
            ),
            threshold=0.5,
        )
        m = _CatalogMatcher(fm)
        q = normalize([1, 0, 0])
        assert m.source_for(q) == fm.source_for(q) == "/x.png"


class TestGeometryMappingPath:
    """Detection-free runtime: with precomputed geometry loaded, process()
    rebuilds the frame's faces from it (no detection) and routes each by its
    identity's centroid through the existing multi-source path."""

    def _fs(self, face_map):
        fs = object.__new__(FaceSwapper)
        fs._supports_multi_source = True  # noqa: SLF001
        fs._face_map = face_map  # noqa: SLF001
        fs._matcher = (  # noqa: SLF001  # matcher rides with the map (cached by_id)
            _CatalogMatcher(face_map) if face_map is not None else None
        )
        fs._geometry = None  # noqa: SLF001
        return fs

    def _geom(self, frame, ident_id="a", refined=False):
        from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

        kps = tuple((float(i), 0.0) for i in range(5))
        return FrameGeometry(
            faces={frame: (GeomFace(ident_id, (0.0, 0.0, 4.0, 4.0), kps),)},
            frame_count=10,
            refined=refined,
        )

    def test_reconstructs_face_with_centroid_embedding(self):
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path="/s.png"),),
            threshold=0.5,
        )
        faces = self._fs(fm)._geometry_faces(  # noqa: SLF001
            self._geom(5), SimpleNamespace(frame_index=5)
        )
        assert faces is not None and len(faces) == 1
        f = faces[0]
        assert f.kps.shape == (5, 2)
        assert tuple(float(v) for v in f.bbox) == (0.0, 0.0, 4.0, 4.0)
        # centroid-as-embedding routes back to its own identity's source
        assert fm.source_for(f.normed_embedding) == "/s.png"

    def test_geometry_rescales_to_processing_scaled_frame(self):
        # Geometry baked at native res must rescale to the (downscaled) frame the
        # swapper actually processes, or mapped swaps land at the wrong coords.
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path="/s"),),
            threshold=0.5,
        )
        kps = tuple((float(i * 2), 0.0) for i in range(5))  # x = 0,2,4,6,8
        geom = FrameGeometry(
            faces={5: (GeomFace("a", (10.0, 20.0, 30.0, 40.0), kps),)},
            frame_count=10,
            bake_size=(640, 480),  # baked at 640x480
        )
        frame = np.zeros((240, 320, 3), np.uint8)  # live at half (320x240)
        f = self._fs(fm)._geometry_faces(  # noqa: SLF001
            geom, SimpleNamespace(frame_index=5), frame
        )[0]
        assert tuple(float(v) for v in f.bbox) == (5.0, 10.0, 15.0, 20.0)  # halved
        assert float(f.kps[2][0]) == 2.0  # kps x halved (4 -> 2)

    def test_geometry_no_rescale_when_bake_size_unset(self):
        # Old sidecars (no bake_size) assume the live frame → never rescale.
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path="/s"),),
            threshold=0.5,
        )
        frame = np.zeros((240, 320, 3), np.uint8)
        f = self._fs(fm)._geometry_faces(  # noqa: SLF001
            self._geom(5), SimpleNamespace(frame_index=5), frame  # bake_size=None
        )[0]
        assert tuple(float(v) for v in f.bbox) == (0.0, 0.0, 4.0, 4.0)  # unchanged

    def test_baked_embedding_routes_against_live_catalog(self):
        # A0: a face whose baked identity_id is GONE from the catalog still routes
        # — by its baked embedding — to the current best match. This is what lets
        # merge / reassignment work with no re-precompute.
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path="/s.png"),),
            threshold=0.5,
        )
        kps = tuple((float(i), 0.0) for i in range(5))
        geom = FrameGeometry(
            faces={5: (GeomFace(
                "GONE", (0.0, 0.0, 4.0, 4.0), kps,
                tuple(normalize([0.98, 0.02, 0])),  # baked emb ~ identity "a"
            ),)},
            frame_count=10,
        )
        faces = self._fs(fm)._geometry_faces(  # noqa: SLF001
            geom, SimpleNamespace(frame_index=5)
        )
        assert faces is not None and len(faces) == 1
        # Routed via the baked embedding to "a" despite the dead id (not dropped).
        assert fm.source_for(faces[0].normed_embedding) == "/s.png"

    def test_mapped_face_carries_identity_metadata(self):
        # Item 5: the overlay shows REAL score/sex/age/pose (not hardcoded 1.00).
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

        fm = FaceMap(identities=(
            Identity("a", normalize([1, 0, 0]), source_path="/s",
                     det_score=0.87, sex="M", age=34,
                     pitch=2.0, yaw=-4.0, roll=10.0),
        ))
        kps = tuple((float(i), 0.0) for i in range(5))
        geom = FrameGeometry(
            faces={3: (GeomFace("a", (0.0, 0.0, 4.0, 4.0), kps, (), 18.0),)},
            frame_count=10,
        )
        f = self._fs(fm)._geometry_faces(  # noqa: SLF001
            geom, SimpleNamespace(frame_index=3)
        )[0]
        assert f.det_score == 0.87 and f.sex == "M" and f.age == 34
        # pose = rep pitch/yaw + the PER-FRAME baked roll (18, not the rep 10).
        assert f.pose == (2.0, -4.0, 18.0)

    def test_mapped_face_no_pose_without_full_pack(self):
        # Fast-mode identity (no pitch/yaw) → pose None, but score still real.
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

        fm = FaceMap(identities=(
            Identity("a", normalize([1, 0, 0]), source_path="/s", det_score=0.7),
        ))
        kps = tuple((float(i), 0.0) for i in range(5))
        geom = FrameGeometry(
            faces={3: (GeomFace("a", (0.0, 0.0, 4.0, 4.0), kps),)}, frame_count=10
        )
        f = self._fs(fm)._geometry_faces(  # noqa: SLF001
            geom, SimpleNamespace(frame_index=3)
        )[0]
        assert f.det_score == 0.7 and f.pose is None

    def test_baked_roll_flows_to_mapped_face(self):
        # D5: a geometry face's baked roll reaches the rebuilt face so rotation
        # compensation uses it instead of a (missing) pose estimate.
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path="/s"),)
        )
        kps = tuple((float(i), 0.0) for i in range(5))
        geom = FrameGeometry(
            faces={3: (GeomFace("a", (0.0, 0.0, 4.0, 4.0), kps, (), 18.5),)},
            frame_count=10,
        )
        faces = self._fs(fm)._geometry_faces(  # noqa: SLF001
            geom, SimpleNamespace(frame_index=3)
        )
        assert faces is not None and faces[0].baked_roll == 18.5

    def test_stale_only_frame_falls_back_to_detection(self):
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        fm = FaceMap(identities=(Identity("a", normalize([1, 0, 0]), source_path="/s"),))
        # The frame's only geometry face is a since-deleted identity → no usable
        # mapped faces → None, so process() re-detects rather than show nothing.
        faces = self._fs(fm)._geometry_faces(  # noqa: SLF001
            self._geom(0, ident_id="GONE"), SimpleNamespace(frame_index=0)
        )
        assert faces is None

    def test_none_when_not_usable(self):
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        active = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path="/s"),)
        )
        geom = self._geom(0)
        at0 = SimpleNamespace(frame_index=0)
        assert self._fs(active)._geometry_faces(None, at0) is None       # no geometry  # noqa: SLF001
        assert self._fs(active)._geometry_faces(geom, None) is None      # no ctx  # noqa: SLF001
        assert self._fs(active)._geometry_faces(  # noqa: SLF001
            geom, SimpleNamespace(frame_index=None)
        ) is None                                                        # unknown frame
        inactive = self._fs(FaceMap(identities=(Identity("a", normalize([1, 0, 0])),)))
        assert inactive._geometry_faces(geom, at0) is None               # map inactive  # noqa: SLF001
        gen = self._fs(active)
        gen._supports_multi_source = False  # noqa: SLF001
        assert gen._geometry_faces(geom, at0) is None                    # generic backend  # noqa: SLF001

    def test_trace_reports_unassigned_routing(self, capsys):
        # SINNER2_GEOM_TRACE diagnostic: with the map ACTIVE (identity "a" has a
        # source) but the frame's geometry face tagged to the UNASSIGNED "b", the
        # face routes to nothing → srcAssigned=n. This is the exact signature of
        # the reported "Precompute ON → no swap" (built faces that don't route).
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        fm = FaceMap(identities=(
            Identity("a", normalize([1, 0, 0]), source_path="/s.png"),  # active
            Identity("b", normalize([0, 1, 0])),                        # no source
        ))
        fs = self._fs(fm)
        fs._mapped_sources = {"/s.png": object()}  # noqa: SLF001
        fs._geom_trace_n = 0  # noqa: SLF001
        # The geometry face on this frame belongs to "b" (unassigned).
        faces = fs._geometry_faces(  # noqa: SLF001
            self._geom(5, ident_id="b"), SimpleNamespace(frame_index=5)
        )
        assert faces is not None
        fs._trace_geometry(faces, fm, SimpleNamespace(frame_index=5))  # noqa: SLF001
        err = capsys.readouterr().err
        assert "[geom]" in err and "built=1" in err
        assert "match=b" in err and "srcAssigned=n" in err
        assert "assignedIdentities=['a']" in err

    def test_trace_reports_assigned_routing(self, capsys):
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path="/s.png"),)
        )
        fs = self._fs(fm)
        fs._mapped_sources = {"/s.png": object()}  # noqa: SLF001 — prepared source
        fs._geom_trace_n = 0  # noqa: SLF001
        faces = fs._geometry_faces(self._geom(5), SimpleNamespace(frame_index=5))  # noqa: SLF001
        fs._trace_geometry(faces, fm, SimpleNamespace(frame_index=5))  # noqa: SLF001
        err = capsys.readouterr().err
        assert "srcAssigned=Y" in err and "srcPrepared=Y" in err

    def test_process_uses_geometry_and_skips_detection(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper, tmp_path
    ):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.processor import ChainContext

        a = tmp_path / "alice.png"
        cv2.imwrite(str(a), np.full((32, 32, 3), 10, np.uint8))
        src_a = object()
        # setup consumes: main source [obj], mapped source for alice [src_a].
        stub_insightface_app.get.side_effect = [[object()], [src_a]]
        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path=str(a)),),
            threshold=0.5,
        )
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_compensation=False),
            face_map=fm,
        )
        fs.setup()
        fs.set_geometry(self._geom(3))
        get_calls = stub_insightface_app.get.call_count
        fs.process(_blank(), ChainContext(frame_index=3))
        # NO further detection — geometry supplied the face.
        assert stub_insightface_app.get.call_count == get_calls
        # And it was swapped with alice's mapped source.
        assert stub_inswapper.get.call_count == 1
        assert stub_inswapper.get.call_args.args[2] is src_a

    def test_routing_switches_to_det_rec_when_no_embeddings(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper, tmp_path
    ):
        # No geometry + active map + a detection-only analyser (no embeddings) →
        # the swap must call analyse_det_rec so routing has embeddings to match;
        # the default full-pack analyser (embeddings) keeps the fast analyse path.
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        a = tmp_path / "alice.png"
        cv2.imwrite(str(a), np.full((32, 32, 3), 10, np.uint8))
        stub_insightface_app.get.side_effect = [[object()], [object()]]  # setup only
        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path=str(a)),),
            threshold=0.5,
        )
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_compensation=False),
            face_map=fm,
        )
        fs.setup()
        called = []
        fs._analyser.analyse = lambda f: called.append("analyse") or []  # noqa: SLF001
        fs._analyser.analyse_det_rec = lambda f: called.append("det_rec") or []  # noqa: SLF001
        # No-embeddings analyser (e.g. yoloface) → det+rec.
        fs._analyser.provides_embeddings = lambda: False  # noqa: SLF001
        fs.process(_blank())
        assert called == ["det_rec"]
        # Full-pack analyser (embeddings) → keep the cached analyse path.
        called.clear()
        fs._analyser.provides_embeddings = lambda: True  # noqa: SLF001
        fs.process(_blank())
        assert called == ["analyse"]

    def test_uncovered_frame_falls_back_to_live_detection(
        self, source_image, models_dir, stub_insightface_app, stub_inswapper, tmp_path
    ):
        from types import SimpleNamespace

        from sinner2.pipeline.face_map import FaceMap, Identity, normalize
        from sinner2.pipeline.processor import ChainContext

        a = tmp_path / "alice.png"
        cv2.imwrite(str(a), np.full((32, 32, 3), 10, np.uint8))
        stranger = SimpleNamespace(  # detected live, but matches no identity
            normed_embedding=np.asarray(normalize([0, 0, 1]), np.float32),
            bbox=np.array([0, 0, 4, 4], float), kps=np.zeros((5, 2), np.float32),
            det_score=0.9, sex=None,
        )
        stub_insightface_app.get.side_effect = [[object()], [object()], [stranger]]
        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0, 0]), source_path=str(a)),),
            threshold=0.5,
        )
        fs = FaceSwapper(
            source=Source(path=source_image),
            params=_params(rotation_compensation=False),
            face_map=fm,
        )
        fs.setup()
        fs.set_geometry(self._geom(3))  # only frame 3 has geometry
        get_calls = stub_insightface_app.get.call_count
        fs.process(_blank(), ChainContext(frame_index=99))  # uncovered → fall back
        assert stub_insightface_app.get.call_count == get_calls + 1  # re-detected live
        stub_inswapper.get.assert_not_called()  # the stranger matched nothing → no swap
