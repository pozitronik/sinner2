from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from sinner2.pipeline import face_analyser
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors import face_enhancer
from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    FaceEnhancer,
    FaceEnhancerParams,
)
from sinner2.types import Frame


@pytest.fixture(autouse=True)
def stub_face_detection(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Rotation compensation (on by default) makes the enhancer detect faces;
    stub the shared insightface model so tests never load/download it. Detects
    nothing by default — rotation tests override `.get`."""
    app = MagicMock()
    app.get = MagicMock(return_value=[])
    monkeypatch.setattr(
        face_analyser, "_get_shared_face_analysis", lambda *a, **k: app
    )
    face_analyser.reset_shared_face_analysis()
    return app


@pytest.fixture
def stub_restorer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    restorer = MagicMock()
    restorer.enhance = MagicMock(side_effect=lambda f, **_k: ([], [], f))
    monkeypatch.setattr(
        face_enhancer,
        "_load_restorer",
        lambda _path, _upscale, _device, fp16=False: restorer,
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


def _gfpgan(**overrides) -> FaceEnhancerParams:
    """Params pinned to the torch-GFPGAN flavor. The params default is now
    GFPGAN_ONNX (the faster path), while these tests exercise the torch
    pipeline (loader / fp16 / device / rotation) specifically."""
    overrides.setdefault("model", EnhancerModel.GFPGAN)
    return FaceEnhancerParams(**overrides)


class TestFaceEnhancerParams:
    def test_defaults(self):
        p = FaceEnhancerParams()
        assert p.model is EnhancerModel.GFPGAN_ONNX  # the fast ONNX export
        assert p.upscale == 1
        assert p.only_center_face is False
        assert p.fp16 is True

    def test_rejects_zero_upscale(self):
        with pytest.raises(Exception):
            FaceEnhancerParams(upscale=0)

    def test_rejects_excessive_upscale(self):
        with pytest.raises(Exception):
            FaceEnhancerParams(upscale=10)


class TestFaceEnhancer:
    def test_name(self):
        assert FaceEnhancer.name == "FaceEnhancer"

    def test_not_thread_safe(self):
        assert FaceEnhancer.thread_safe is False

    def test_compliant_with_processor_protocol(self):
        assert isinstance(FaceEnhancer(), Processor)

    def test_process_before_setup_raises(self):
        fe = FaceEnhancer()
        with pytest.raises(RuntimeError, match="before setup"):
            fe.process(_blank())

    def test_setup_enables_process(self, models_dir: Path, stub_restorer: MagicMock):
        fe = FaceEnhancer(params=_gfpgan())
        fe.setup()
        out = fe.process(_blank())
        assert out.shape == (10, 10, 3)

    def test_process_falls_back_to_original_when_restored_is_none(
        self, models_dir: Path, stub_restorer: MagicMock
    ):
        stub_restorer.enhance = MagicMock(return_value=([], [], None))
        fe = FaceEnhancer(params=_gfpgan())
        fe.setup()
        original = _blank()
        out = fe.process(original)
        assert out is original

    def test_release_makes_process_raise(self, models_dir: Path, stub_restorer: MagicMock):
        fe = FaceEnhancer(params=_gfpgan())
        fe.setup()
        fe.release()
        with pytest.raises(RuntimeError):
            fe.process(_blank())

    def test_upscale_passed_to_restorer(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        captured: dict[str, int] = {}

        def fake_load(
            _path: Path, upscale: int, _device: str, fp16: bool = False
        ) -> MagicMock:
            captured["upscale"] = upscale
            return MagicMock(enhance=MagicMock(return_value=([], [], None)))

        monkeypatch.setattr(face_enhancer, "_load_restorer", fake_load)

        fe = FaceEnhancer(params=_gfpgan(upscale=2))
        fe.setup()
        assert captured["upscale"] == 2

    def test_device_follows_cuda_availability(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import torch

        captured: dict[str, str] = {}

        def fake_load(_path: Path, _upscale: int, device, fp16: bool = False) -> MagicMock:
            captured["device"] = device
            return MagicMock(enhance=MagicMock(return_value=([], [], None)))

        monkeypatch.setattr(face_enhancer, "_load_restorer", fake_load)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        FaceEnhancer(params=_gfpgan()).setup()  # device="auto" → resolves to cpu
        assert captured["device"].type == "cpu"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        FaceEnhancer(params=_gfpgan()).setup()
        assert captured["device"].type == "cuda"

    def test_explicit_cpu_device_overrides_available_cuda(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import torch

        captured: dict = {}

        def fake_load(_path, _upscale, device, fp16=False):
            captured["device"] = device
            return MagicMock(enhance=MagicMock(return_value=([], [], None)))

        monkeypatch.setattr(face_enhancer, "_load_restorer", fake_load)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        FaceEnhancer(device="cpu", params=_gfpgan()).setup()
        assert captured["device"].type == "cpu"

    def test_only_center_face_passed_to_enhance(
        self, models_dir: Path, stub_restorer: MagicMock
    ):
        fe = FaceEnhancer(params=_gfpgan(only_center_face=True))
        fe.setup()
        fe.process(_blank())
        _, kwargs = stub_restorer.enhance.call_args
        assert kwargs["only_center_face"] is True

    def test_release_returns_cuda_memory_when_on_gpu(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # nvidia-smi only reflects a freed model if torch's caching allocator
        # hands the blocks back — release() must call empty_cache on CUDA so a
        # realtime worker-count decrease visibly frees VRAM.
        import torch

        monkeypatch.setattr(
            face_enhancer,
            "_load_restorer",
            lambda *a, **k: MagicMock(enhance=MagicMock(return_value=([], [], None))),
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        empties: list[int] = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empties.append(1))
        fe = FaceEnhancer(device="cuda", params=_gfpgan())
        fe.setup()
        fe.release()
        assert empties == [1]

    def test_release_skips_cuda_cache_on_cpu(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import torch

        monkeypatch.setattr(
            face_enhancer,
            "_load_restorer",
            lambda *a, **k: MagicMock(enhance=MagicMock(return_value=([], [], None))),
        )
        empties: list[int] = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empties.append(1))
        fe = FaceEnhancer(device="cpu", params=_gfpgan())
        fe.setup()
        fe.release()
        assert empties == []


class TestFp16:
    """GFPGAN half precision: active only on CUDA, gates the autocast wrapper,
    and is passed to the loader so it can half the generator weights."""

    def _capture_load(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        captured: dict = {}

        def fake_load(_path, _upscale, _device, fp16=False):
            captured["fp16"] = fp16
            return MagicMock(enhance=MagicMock(return_value=([], [], None)))

        monkeypatch.setattr(face_enhancer, "_load_restorer", fake_load)
        return captured

    def test_fp16_passed_to_loader_on_cuda(self, models_dir, monkeypatch):
        import torch

        captured = self._capture_load(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        fe = FaceEnhancer(device="cuda", params=_gfpgan(fp16=True))
        fe.setup()
        assert captured["fp16"] is True
        assert fe._fp16 is True  # noqa: SLF001

    def test_fp16_disabled_on_cpu_even_when_requested(self, models_dir, monkeypatch):
        captured = self._capture_load(monkeypatch)
        # Explicit CPU device → fp16 is a no-op, so it's forced off.
        fe = FaceEnhancer(device="cpu", params=_gfpgan(fp16=True))
        fe.setup()
        assert captured["fp16"] is False
        assert fe._fp16 is False  # noqa: SLF001

    def test_fp16_param_false_disables_on_cuda(self, models_dir, monkeypatch):
        import torch

        captured = self._capture_load(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        fe = FaceEnhancer(device="cuda", params=_gfpgan(fp16=False))
        fe.setup()
        assert captured["fp16"] is False
        assert fe._fp16 is False  # noqa: SLF001

    def test_autocast_context_gating(self):
        import contextlib

        import torch

        fe = FaceEnhancer()
        fe._fp16 = False  # noqa: SLF001
        assert isinstance(fe._gfpgan_autocast(), contextlib.nullcontext)  # noqa: SLF001
        fe._fp16 = True  # noqa: SLF001
        assert isinstance(fe._gfpgan_autocast(), torch.autocast)  # noqa: SLF001


class TestRotationCompensation:
    def test_uprights_tilted_face(
        self, models_dir, stub_restorer, stub_face_detection, monkeypatch
    ):
        called: list = []
        monkeypatch.setattr(
            face_enhancer,
            "enhance_with_uprighting",
            lambda result, *a, **k: called.append(True) or result,
        )
        tilted = MagicMock()
        tilted.bbox = np.array([2, 2, 8, 8], float)
        tilted.kps = np.array([[3, 3], [7, 7]], float)  # 45° roll
        stub_face_detection.get.return_value = [tilted]
        fe = FaceEnhancer(
            params=_gfpgan(rotation_compensation=True, rotation_threshold_deg=15)
        )
        fe.setup()
        fe.process(_blank())
        assert called == [True]

    def test_skips_upright_face(
        self, models_dir, stub_restorer, stub_face_detection, monkeypatch
    ):
        called: list = []
        monkeypatch.setattr(
            face_enhancer,
            "enhance_with_uprighting",
            lambda result, *a, **k: called.append(True) or result,
        )
        upright = MagicMock()
        upright.bbox = np.array([2, 2, 8, 8], float)
        upright.kps = np.array([[3, 5], [7, 5]], float)  # level eyes → 0°
        stub_face_detection.get.return_value = [upright]
        fe = FaceEnhancer(params=_gfpgan())  # rotation on by default
        fe.setup()
        fe.process(_blank())
        assert called == []

    def test_disabled_skips_detection(
        self, models_dir, stub_restorer, stub_face_detection
    ):
        fe = FaceEnhancer(params=_gfpgan(rotation_compensation=False))
        fe.setup()
        assert fe._analyser is None  # noqa: SLF001 — not built when disabled
        fe.process(_blank())
        stub_face_detection.get.assert_not_called()


class TestCodeFormerBackend:
    def test_codeformer_model_uses_onnx_backend(
        self, models_dir, stub_face_detection, monkeypatch
    ):
        class _StubBackend:
            def __init__(self, *a, **k):
                pass

            def setup(self):
                pass

            def enhance(self, img):
                return img

        monkeypatch.setattr(face_enhancer, "CodeFormerBackend", _StubBackend)
        fe = FaceEnhancer(
            params=FaceEnhancerParams(
                model=EnhancerModel.CODEFORMER, rotation_compensation=False
            )
        )
        fe.setup()
        assert fe._codeformer is not None  # noqa: SLF001
        out = fe.process(_blank())
        assert out.shape == _blank().shape

    def test_gfpgan_onnx_is_default(
        self, models_dir, stub_face_detection, monkeypatch
    ):
        # Default-construct → the GFPGAN-ONNX flavor through PlainBfrBackend
        # (the fast path), not the torch restorer and not CodeFormer.
        captured: dict = {}

        class _StubBackend:
            def __init__(self, model_file, *a, **k):
                captured["model_file"] = model_file

            def setup(self):
                pass

        monkeypatch.setattr(face_enhancer, "PlainBfrBackend", _StubBackend)
        fe = FaceEnhancer()
        fe.setup()
        assert fe._bfr is not None  # noqa: SLF001
        assert fe._restorer is None  # noqa: SLF001 — torch GFPGAN not loaded
        assert fe._codeformer is None  # noqa: SLF001
        assert captured["model_file"] == "gfpgan_1.4.onnx"

    def test_release_evicts_codeformer_session(
        self, models_dir, stub_face_detection, monkeypatch
    ):
        released: list[int] = []

        class _StubBackend:
            def __init__(self, *a, **k):
                pass

            def setup(self):
                pass

            def enhance(self, img):
                return img

            def release(self):
                released.append(1)

        monkeypatch.setattr(face_enhancer, "CodeFormerBackend", _StubBackend)
        fe = FaceEnhancer(
            params=FaceEnhancerParams(
                model=EnhancerModel.CODEFORMER, rotation_compensation=False
            )
        )
        fe.setup()
        fe.release()
        assert released == [1]  # backend.release() ran (evicts the ONNX session)
        assert fe._codeformer is None  # noqa: SLF001


class TestPlainBfrBackends:
    """GPEN-512 and RestoreFormer++ route through the shared PlainBfrBackend."""

    class _StubBackend:
        instances: list = []

        def __init__(self, model_file, *a, **k):
            self.model_file = model_file
            self.released = 0
            TestPlainBfrBackends._StubBackend.instances.append(self)

        def setup(self):
            pass

        def enhance(self, img):
            return img

        def release(self):
            self.released += 1

    @pytest.mark.parametrize(
        "model, expected_file",
        [
            (EnhancerModel.GFPGAN_ONNX, "gfpgan_1.4.onnx"),
            (EnhancerModel.GPEN_512, "gpen_bfr_512.onnx"),
            (EnhancerModel.GPEN_1024, "gpen_bfr_1024.onnx"),
            (EnhancerModel.GPEN_2048, "gpen_bfr_2048.onnx"),
            (EnhancerModel.RESTOREFORMER_PP, "restoreformer_plus_plus.onnx"),
        ],
    )
    def test_model_uses_bfr_backend_with_right_file(
        self, models_dir, stub_face_detection, monkeypatch, model, expected_file
    ):
        self._StubBackend.instances = []
        monkeypatch.setattr(face_enhancer, "PlainBfrBackend", self._StubBackend)
        fe = FaceEnhancer(
            params=FaceEnhancerParams(model=model, rotation_compensation=False)
        )
        fe.setup()
        assert fe._bfr is not None  # noqa: SLF001
        assert fe._codeformer is None  # noqa: SLF001
        assert self._StubBackend.instances[-1].model_file == expected_file
        out = fe.process(_blank())
        assert out.shape == _blank().shape

    def test_release_evicts_bfr_session(
        self, models_dir, stub_face_detection, monkeypatch
    ):
        self._StubBackend.instances = []
        monkeypatch.setattr(face_enhancer, "PlainBfrBackend", self._StubBackend)
        fe = FaceEnhancer(
            params=FaceEnhancerParams(
                model=EnhancerModel.GPEN_512, rotation_compensation=False
            )
        )
        fe.setup()
        backend = self._StubBackend.instances[-1]
        fe.release()
        assert backend.released == 1
        assert fe._bfr is None  # noqa: SLF001


class TestRotationSkippedForOnnxRestorers:
    """CodeFormer / GPEN / RestoreFormer already remove in-plane roll via their
    per-face estimate_norm alignment, so the GFPGAN-specific uprighting pass is
    wasted work and must be skipped for them (it re-detects + re-restores the
    heaviest op per tilted face)."""

    class _StubBackend:
        def __init__(self, *a, **k):
            pass

        def setup(self):
            pass

        def enhance(self, img):
            return img

    def test_codeformer_skips_uprighting(
        self, models_dir, stub_face_detection, monkeypatch
    ):
        monkeypatch.setattr(face_enhancer, "CodeFormerBackend", self._StubBackend)
        called: list = []
        monkeypatch.setattr(
            face_enhancer,
            "enhance_with_uprighting",
            lambda result, *a, **k: called.append(True) or result,
        )
        tilted = MagicMock()
        tilted.bbox = np.array([2, 2, 8, 8], float)
        tilted.kps = np.array([[3, 3], [7, 7]], float)  # 45° roll
        stub_face_detection.get.return_value = [tilted]
        fe = FaceEnhancer(
            params=FaceEnhancerParams(
                model=EnhancerModel.CODEFORMER,
                rotation_compensation=True,
                rotation_threshold_deg=15,
            )
        )
        fe.setup()
        fe.process(_blank())
        assert called == []  # uprighting skipped for the ONNX restorer


def test_enhancer_onnx_model_file_mapping():
    from sinner2.pipeline.processors.face_enhancer import enhancer_onnx_model_file

    assert enhancer_onnx_model_file(EnhancerModel.GFPGAN) is None
    assert enhancer_onnx_model_file(EnhancerModel.GFPGAN_ONNX) == "gfpgan_1.4.onnx"
    assert enhancer_onnx_model_file(EnhancerModel.CODEFORMER) == "codeformer.onnx"
    assert enhancer_onnx_model_file(EnhancerModel.GPEN_512) == "gpen_bfr_512.onnx"
    assert enhancer_onnx_model_file(EnhancerModel.GPEN_1024) == "gpen_bfr_1024.onnx"
    assert enhancer_onnx_model_file(EnhancerModel.GPEN_2048) == "gpen_bfr_2048.onnx"
    assert (
        enhancer_onnx_model_file(EnhancerModel.RESTOREFORMER_PP)
        == "restoreformer_plus_plus.onnx"
    )
