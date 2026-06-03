from unittest.mock import MagicMock

import numpy as np
import pytest

from sinner2.pipeline import face_analyser
from sinner2.pipeline.face_analyser import FaceAnalyser, reset_shared_face_analysis
from sinner2.types import Frame


@pytest.fixture
def stub_insightface(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    stub.get = MagicMock(return_value=[MagicMock(name="Face")])
    monkeypatch.setattr(face_analyser, "_get_shared_face_analysis", lambda *a, **k: stub)
    return stub


@pytest.fixture(autouse=True)
def _reset_shared() -> None:
    reset_shared_face_analysis()


def _blank_frame() -> Frame:
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestFaceAnalyser:
    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError):
            FaceAnalyser(detection_interval=0)

    def test_interval_1_detects_every_frame(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=1)
        for _ in range(5):
            a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 5

    def test_interval_3_detects_every_third_frame(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=3)
        for _ in range(9):
            a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 3

    def test_returned_faces_are_cached_between_detections(self, stub_insightface: MagicMock):
        face1 = MagicMock(name="face1")
        face2 = MagicMock(name="face2")
        stub_insightface.get.side_effect = [[face1], [face2]]

        a = FaceAnalyser(detection_interval=3)
        assert a.analyse(_blank_frame()) == [face1]
        assert a.analyse(_blank_frame()) == [face1]
        assert a.analyse(_blank_frame()) == [face1]
        assert a.analyse(_blank_frame()) == [face2]

    def test_analyse_uncached_bypasses_cache(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=10)
        for _ in range(3):
            a.analyse_uncached(_blank_frame())
        assert stub_insightface.get.call_count == 3

    def test_reset_cache_forces_next_detection(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=10)
        a.analyse(_blank_frame())
        a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 1

        a.reset_cache()
        a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 2

    def test_returned_list_is_a_copy(self, stub_insightface: MagicMock):
        face = MagicMock(name="face")
        stub_insightface.get.return_value = [face]

        a = FaceAnalyser(detection_interval=1)
        faces = a.analyse(_blank_frame())
        faces.append("extra")
        assert a.analyse(_blank_frame()) == [face]


class TestDetectionSize:
    """The face-detector input size (det_size) is configurable: smaller =
    faster detection (may miss small/distant faces). It threads from the
    swapper params through FaceAnalyser into insightface's prepare(), and is
    aligned to a multiple of 32 (SCRFD's strides)."""

    def _capture_det_size(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        import sys
        import types

        captured: dict = {}

        class FakeFaceAnalysis:
            def __init__(self, name=None, providers=None, provider_options=None, **kw):
                pass

            def prepare(self, ctx_id=0, det_size=None):  # noqa: ARG002
                captured["det_size"] = det_size

        app_mod = types.ModuleType("insightface.app")
        app_mod.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.app", app_mod)
        return captured

    def test_default_det_size_is_640(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._capture_det_size(monkeypatch)
        face_analyser._get_shared_face_analysis(None)
        assert captured["det_size"] == (640, 640)

    def test_custom_det_size_forwarded(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._capture_det_size(monkeypatch)
        face_analyser._get_shared_face_analysis(None, det_size=320)
        assert captured["det_size"] == (320, 320)

    def test_det_size_aligned_to_multiple_of_32(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._capture_det_size(monkeypatch)
        face_analyser._get_shared_face_analysis(None, det_size=300)
        assert captured["det_size"] == (288, 288)

    def test_tiny_det_size_clamped_to_minimum(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._capture_det_size(monkeypatch)
        face_analyser._get_shared_face_analysis(None, det_size=1)
        assert captured["det_size"] == (32, 32)

    def test_face_analyser_threads_det_size_into_shared_build(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        captured: dict = {}

        def fake_shared(providers=None, det_size=640):
            captured["det_size"] = det_size
            stub = MagicMock()
            stub.get = MagicMock(return_value=[])
            return stub

        monkeypatch.setattr(face_analyser, "_get_shared_face_analysis", fake_shared)
        FaceAnalyser(detection_interval=1, detection_size=256).analyse(_blank_frame())
        assert captured["det_size"] == 256


class TestSharedFaceAnalysisProviders:
    """The shared insightface detector is built with the same tuned CUDA
    provider options as the swapper (FaceAnalysis forwards provider_options to
    each sub-model's ORT session)."""

    def test_forwards_tuned_provider_options(self, monkeypatch: pytest.MonkeyPatch):
        import sys
        import types

        captured: dict = {}

        class FakeFaceAnalysis:
            def __init__(self, name=None, providers=None, provider_options=None, **kw):
                captured["providers"] = providers
                captured["provider_options"] = provider_options

            def prepare(self, ctx_id=0, det_size=None):  # noqa: ARG002
                pass

        app_mod = types.ModuleType("insightface.app")
        app_mod.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.app", app_mod)

        face_analyser._get_shared_face_analysis(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert captured["providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert captured["provider_options"][0]["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert captured["provider_options"][1] == {}

    def test_detector_does_not_use_tensorrt(self, monkeypatch: pytest.MonkeyPatch):
        # The detector pack stays on CUDA even when the swapper uses TensorRT —
        # else buffalo_l's 5 sub-models each compile their own engine.
        import sys
        import types

        captured: dict = {}

        class FakeFaceAnalysis:
            def __init__(self, name=None, providers=None, provider_options=None, **kw):
                captured["providers"] = providers

            def prepare(self, ctx_id=0, det_size=None):  # noqa: ARG002
                pass

        app_mod = types.ModuleType("insightface.app")
        app_mod.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.app", app_mod)

        face_analyser._get_shared_face_analysis(
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert "TensorrtExecutionProvider" not in captured["providers"]
        assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _capture(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        import sys
        import types

        captured: dict = {}

        class FakeFaceAnalysis:
            def __init__(self, name=None, providers=None, provider_options=None, **kw):
                captured["providers"] = providers

            def prepare(self, ctx_id=0, det_size=None):  # noqa: ARG002
                pass

        app_mod = types.ModuleType("insightface.app")
        app_mod.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.app", app_mod)
        return captured

    def test_empty_providers_stay_empty(self, monkeypatch: pytest.MonkeyPatch):
        # User selected NO providers → detector gets [] (ORT → CPU), NOT a
        # substituted default.
        captured = self._capture(monkeypatch)
        face_analyser._get_shared_face_analysis([])
        assert captured["providers"] == []

    def test_none_providers_use_default(self, monkeypatch: pytest.MonkeyPatch):
        # None = unspecified (e.g. a programmatic caller) → platform default.
        captured = self._capture(monkeypatch)
        face_analyser._get_shared_face_analysis(None)
        assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_tensorrt_only_falls_back_to_default_for_detector(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Only TRT picked → detector can't use it; falls back to the GPU default
        # (NOT empty — that's the deliberate exception to "empty stays empty").
        captured = self._capture(monkeypatch)
        face_analyser._get_shared_face_analysis(["TensorrtExecutionProvider"])
        assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
