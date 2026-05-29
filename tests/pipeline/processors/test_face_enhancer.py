from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors import face_enhancer
from sinner2.pipeline.processors.face_enhancer import FaceEnhancer, FaceEnhancerParams
from sinner2.types import Frame


@pytest.fixture
def stub_restorer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    restorer = MagicMock()
    restorer.enhance = MagicMock(side_effect=lambda f, **_k: ([], [], f))
    monkeypatch.setattr(
        face_enhancer, "_load_restorer", lambda _path, _upscale, _device: restorer
    )
    return restorer


@pytest.fixture
def models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    md = tmp_path / "models"
    md.mkdir()
    (md / "GFPGANv1.4.pth").write_bytes(b"x")
    monkeypatch.setenv("SINNER2_MODELS_DIR", str(md))
    return md


def _blank() -> Frame:
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestFaceEnhancerParams:
    def test_defaults(self):
        p = FaceEnhancerParams()
        assert p.upscale == 1
        assert p.only_center_face is False

    def test_rejects_zero_upscale(self):
        with pytest.raises(Exception):
            FaceEnhancerParams(upscale=0)

    def test_rejects_excessive_upscale(self):
        with pytest.raises(Exception):
            FaceEnhancerParams(upscale=10)


class TestFaceEnhancer:
    def test_name(self):
        assert FaceEnhancer.name == "FaceEnhancer"

    def test_compliant_with_processor_protocol(self):
        assert isinstance(FaceEnhancer(), Processor)

    def test_process_before_setup_raises(self):
        fe = FaceEnhancer()
        with pytest.raises(RuntimeError, match="before setup"):
            fe.process(_blank())

    def test_setup_enables_process(self, models_dir: Path, stub_restorer: MagicMock):
        fe = FaceEnhancer()
        fe.setup()
        out = fe.process(_blank())
        assert out.shape == (10, 10, 3)

    def test_process_falls_back_to_original_when_restored_is_none(
        self, models_dir: Path, stub_restorer: MagicMock
    ):
        stub_restorer.enhance = MagicMock(return_value=([], [], None))
        fe = FaceEnhancer()
        fe.setup()
        original = _blank()
        out = fe.process(original)
        assert out is original

    def test_release_makes_process_raise(self, models_dir: Path, stub_restorer: MagicMock):
        fe = FaceEnhancer()
        fe.setup()
        fe.release()
        with pytest.raises(RuntimeError):
            fe.process(_blank())

    def test_upscale_passed_to_restorer(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        captured: dict[str, int] = {}

        def fake_load(_path: Path, upscale: int, _device: str) -> MagicMock:
            captured["upscale"] = upscale
            return MagicMock(enhance=MagicMock(return_value=([], [], None)))

        monkeypatch.setattr(face_enhancer, "_load_restorer", fake_load)

        fe = FaceEnhancer(params=FaceEnhancerParams(upscale=2))
        fe.setup()
        assert captured["upscale"] == 2

    def test_device_follows_cuda_availability(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import torch

        captured: dict[str, str] = {}

        def fake_load(_path: Path, _upscale: int, device: str) -> MagicMock:
            captured["device"] = device
            return MagicMock(enhance=MagicMock(return_value=([], [], None)))

        monkeypatch.setattr(face_enhancer, "_load_restorer", fake_load)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        FaceEnhancer().setup()
        assert captured["device"] == "cpu"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        FaceEnhancer().setup()
        assert captured["device"] == "cuda"

    def test_only_center_face_passed_to_enhance(
        self, models_dir: Path, stub_restorer: MagicMock
    ):
        fe = FaceEnhancer(params=FaceEnhancerParams(only_center_face=True))
        fe.setup()
        fe.process(_blank())
        _, kwargs = stub_restorer.enhance.call_args
        assert kwargs["only_center_face"] is True
