from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from sinner2.pipeline import face_analyser
from sinner2.pipeline.face_analyser import FaceAnalyser, reset_shared_face_analysis
from sinner2.types import Frame


class TestLoadNotifier:
    def teardown_method(self):
        face_analyser.set_load_notifier(None)
        reset_shared_face_analysis()

    def test_notify_calls_installed_notifier(self):
        seen: list[str] = []
        face_analyser.set_load_notifier(seen.append)
        face_analyser._notify_load("hi")  # noqa: SLF001
        face_analyser._notify_load("")  # noqa: SLF001
        assert seen == ["hi", ""]

    def test_notify_noop_without_listener(self):
        face_analyser.set_load_notifier(None)
        face_analyser._notify_load("nobody")  # noqa: SLF001  # must not raise

    def test_notify_swallows_listener_errors(self):
        face_analyser.set_load_notifier(lambda _m: (_ for _ in ()).throw(RuntimeError()))
        face_analyser._notify_load("x")  # noqa: SLF001  # must not propagate

    def test_first_run_flags_download_then_clears(self, monkeypatch, tmp_path):
        import insightface.app

        from sinner2.pipeline import model_cache

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(model_cache, "get_models_dir", lambda: models_dir)

        class _Fake:
            def __init__(self, **kw):
                pass

            def prepare(self, **kw):
                pass

        monkeypatch.setattr(insightface.app, "FaceAnalysis", _Fake)
        seen: list[str] = []
        face_analyser.set_load_notifier(seen.append)
        reset_shared_face_analysis()
        face_analyser._get_shared_face_analysis()  # noqa: SLF001
        assert seen and seen[0].startswith("Downloading")
        assert seen[-1] == ""

    def test_present_pack_does_not_flag(self, monkeypatch, tmp_path):
        import insightface.app

        from sinner2.pipeline import model_cache

        models_dir = tmp_path / "models"
        (models_dir / "buffalo_l").mkdir(parents=True)  # clean location present
        monkeypatch.setattr(model_cache, "get_models_dir", lambda: models_dir)

        class _Fake:
            def __init__(self, **kw):
                pass

            def prepare(self, **kw):
                pass

        monkeypatch.setattr(insightface.app, "FaceAnalysis", _Fake)
        seen: list[str] = []
        face_analyser.set_load_notifier(seen.append)
        reset_shared_face_analysis()
        face_analyser._get_shared_face_analysis()  # noqa: SLF001
        assert seen == []  # nothing to download → no flag


class TestBuffaloPackLocation:
    def test_default_models_dir_uses_clean_path(self):
        # Models dir named "models" → root is its parent so insightface's
        # hardcoded /models segment yields <models_dir>/buffalo_l, not doubled.
        models = Path("/app/models")
        root, pack = face_analyser._buffalo_root_and_pack(models)  # noqa: SLF001
        assert root == Path("/app")
        assert pack == Path("/app/models/buffalo_l")

    def test_custom_models_dir_keeps_nested(self):
        models = Path("/data/sinnermodels")  # not named "models"
        root, pack = face_analyser._buffalo_root_and_pack(models)  # noqa: SLF001
        assert root == models
        assert pack == Path("/data/sinnermodels/models/buffalo_l")


@pytest.fixture
def stub_insightface(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    stub.get = MagicMock(return_value=[MagicMock(name="Face")])
    stub.det_model.detect = MagicMock(
        return_value=(
            np.array([[1.0, 2.0, 30.0, 40.0, 0.9]], np.float32),
            np.array([[[3.0, 4.0]] * 5], np.float32),
        )
    )
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


class TestDetectionOnly:
    """detection_only mode: the shared pack's det model alone, no aux models.
    For consumers that only align by keypoints (the ONNX restorer backends) —
    buffalo_l's .get() runs four extra models per face they never read."""

    def test_uses_det_model_not_get(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_only=True)
        faces = a.analyse(_blank_frame())
        stub_insightface.det_model.detect.assert_called_once()
        stub_insightface.get.assert_not_called()
        assert len(faces) == 1

    def test_faces_carry_bbox_kps_score(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_only=True)
        face = a.analyse(_blank_frame())[0]
        assert np.allclose(face.bbox, [1.0, 2.0, 30.0, 40.0])
        assert face.kps.shape == (5, 2)
        assert face.det_score == pytest.approx(0.9)
        # FaceLite: deliberately NO sex/pose — getattr-guarded paths degrade.
        assert not hasattr(face, "sex")

    def test_does_not_provide_gender(self, stub_insightface: MagicMock):
        assert FaceAnalyser(detection_only=True).provides_gender() is False
        assert FaceAnalyser().provides_gender() is True

    def test_interval_caching_still_applies(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=3, detection_only=True)
        for _ in range(9):
            a.analyse(_blank_frame())
        assert stub_insightface.det_model.detect.call_count == 3

    def test_no_faces_detected(self, stub_insightface: MagicMock):
        stub_insightface.det_model.detect.return_value = (
            np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32),
        )
        a = FaceAnalyser(detection_only=True)
        assert a.analyse(_blank_frame()) == []


class TestScanDetectorChoice:
    """analyse_det_rec with a chosen detector: the CUSTOM detector finds the
    faces and ArcFace (from the shared pack) adds the embedding — so a faster
    detection-only detector can still drive identity clustering."""

    def test_custom_detector_plus_arcface(self, stub_insightface, monkeypatch):
        from sinner2.pipeline import detectors as det_mod
        from sinner2.pipeline.detectors import DetectorModel, FaceLite

        stub_det = MagicMock()
        stub_det.detect.return_value = [
            FaceLite(
                bbox=np.array([1.0, 2.0, 3.0, 4.0], np.float32),
                kps=np.array([[3.0, 4.0]] * 5, np.float32),
                det_score=0.8,
            ),
        ]
        monkeypatch.setattr(det_mod, "build_detector", lambda *a, **k: stub_det)
        rec = stub_insightface.models.get.return_value  # the recognition stub

        a = FaceAnalyser(detector=DetectorModel.YOLOFACE)
        faces = a.analyse_det_rec(_blank_frame())

        # The CHOSEN detector found the faces — NOT buffalo_l's det_model…
        stub_det.detect.assert_called_once()
        stub_insightface.det_model.detect.assert_not_called()
        # …and ArcFace ran per face to add the embedding.
        assert rec.get.call_count == 1
        assert len(faces) == 1 and faces[0].det_score == pytest.approx(0.8)

    def test_buffalo_l_still_uses_det_model(self, stub_insightface):
        # No custom detector → the existing buffalo_l det_model + ArcFace path.
        rec = stub_insightface.models.get.return_value
        a = FaceAnalyser()  # buffalo_l
        a.analyse_det_rec(_blank_frame())
        stub_insightface.det_model.detect.assert_called_once()
        assert rec.get.call_count == 1

    def test_release_releases_standalone_detector(self, stub_insightface, monkeypatch):
        from sinner2.pipeline import detectors as det_mod
        from sinner2.pipeline.detectors import DetectorModel

        stub_det = MagicMock()
        monkeypatch.setattr(det_mod, "build_detector", lambda *a, **k: stub_det)
        a = FaceAnalyser(detector=DetectorModel.YOLOFACE)
        a.release()
        stub_det.release.assert_called_once()  # the detector's session is freed
        assert a._detector is None  # noqa: SLF001
        a.release()  # idempotent

    def test_release_is_noop_for_buffalo_l(self, stub_insightface):
        # buffalo_l is a shared singleton (no standalone detector) — release()
        # must NOT touch it.
        a = FaceAnalyser()
        a.release()
        assert a._detector is None  # noqa: SLF001


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


class TestAlternativeDetector:
    """A standalone detector handles per-frame TARGET detection; the SOURCE
    face (analyse_uncached) stays on buffalo_l for its embedding."""

    class _StubDetector:
        def __init__(self):
            self.setups = 0
            self.detects = 0

        def setup(self):
            self.setups += 1

        def detect(self, _frame):
            self.detects += 1
            return [MagicMock(name="lite")]

    def test_analyse_routes_to_detector_and_eager_setup(self, monkeypatch):
        from sinner2.pipeline import detectors

        stub = self._StubDetector()
        monkeypatch.setattr(detectors, "build_detector", lambda m, p=None, size=640: stub)
        a = FaceAnalyser(detector=detectors.DetectorModel.YOLOFACE)
        assert stub.setups == 1  # loaded eagerly at construction (no worker race)
        a.analyse(_blank_frame())
        assert stub.detects == 1
        assert a.provides_gender() is False

    def test_detection_size_forwarded_to_standalone_detector(self, monkeypatch):
        # Regression: detection_size was dropped on the standalone-detector path
        # (FaceAnalyser called build_detector without it), so the knob was dead
        # for yolo/scrfd.
        from sinner2.pipeline import detectors

        captured: dict = {}

        def fake_build(m, p=None, size=640):
            captured["size"] = size
            return self._StubDetector()

        monkeypatch.setattr(detectors, "build_detector", fake_build)
        FaceAnalyser(detector=detectors.DetectorModel.SCRFD_2_5G, detection_size=256)
        assert captured["size"] == 256

    def test_analyse_uncached_stays_buffalo_l(
        self, monkeypatch, stub_insightface: MagicMock
    ):
        from sinner2.pipeline import detectors

        monkeypatch.setattr(
            detectors, "build_detector", lambda m, p=None, size=640: self._StubDetector()
        )
        a = FaceAnalyser(detector=detectors.DetectorModel.YOLOFACE)
        a.analyse_uncached(_blank_frame())
        # Source path uses the shared pack, not the standalone detector.
        assert stub_insightface.get.call_count == 1

    def test_buffalo_l_default_has_no_detector_and_provides_gender(self):
        a = FaceAnalyser()
        assert a.provides_gender() is True


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


class TestModelDownloadRoot:
    """buffalo_l must download into the project models dir, not insightface's
    default ~/.insightface. Regression: FaceAnalysis was built without root=,
    so the pack landed outside the chosen models folder."""

    def _capture_root(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        import sys
        import types

        captured: dict = {}

        class FakeFaceAnalysis:
            def __init__(
                self, name=None, root=None, providers=None, provider_options=None, **kw
            ):
                captured["root"] = root

            def prepare(self, ctx_id=0, det_size=None):  # noqa: ARG002
                pass

        app_mod = types.ModuleType("insightface.app")
        app_mod.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.app", app_mod)
        return captured

    def test_download_root_is_project_models_dir(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        from sinner2.pipeline import model_cache

        monkeypatch.setattr(model_cache, "get_models_dir", lambda: tmp_path)
        captured = self._capture_root(monkeypatch)
        face_analyser._get_shared_face_analysis(None)
        assert captured["root"] == str(tmp_path)
