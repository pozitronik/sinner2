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
