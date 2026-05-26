from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from sinner2.config.source import Source
from sinner2.pipeline import face_analyser
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors import face_swapper
from sinner2.pipeline.processors.face_swapper import FaceSwapper, FaceSwapperParams
from sinner2.types import Frame


@pytest.fixture
def stub_insightface_app(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    stub.get = MagicMock(return_value=[MagicMock(name="Face")])
    monkeypatch.setattr(face_analyser, "_get_shared_face_analysis", lambda: stub)
    face_analyser.reset_shared_face_analysis()
    return stub


@pytest.fixture
def stub_inswapper(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    swapper = MagicMock()
    swapper.get = MagicMock(side_effect=lambda f, *a, **k: f)
    monkeypatch.setattr(face_swapper, "_load_inswapper", lambda _path: swapper)
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


class TestFaceSwapperParams:
    def test_defaults(self):
        p = FaceSwapperParams()
        assert p.detection_interval == 1
        assert p.many_faces is True

    def test_rejects_zero_interval(self):
        with pytest.raises(Exception):
            FaceSwapperParams(detection_interval=0)


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
