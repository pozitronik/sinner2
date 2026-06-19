from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.pipeline import model_cache
from sinner2.pipeline.model_cache import (
    available_onnx_providers,
    build_provider_options,
    build_session_options,
    clear_session_cache,
    get_actual_providers,
    get_model_path,
    get_models_dir,
    get_onnx_session,
    get_onnx_session_io,
    record_actual_providers,
    reset_actual_providers,
)


class TestGetModelsDir:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        assert get_models_dir() == tmp_path

    def test_env_override_wins_even_when_cwd_changes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        monkeypatch.chdir(tmp_path.parent)
        assert get_models_dir() == tmp_path

    def test_default_is_project_relative(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.delenv("SINNER2_MODELS_DIR", raising=False)
        monkeypatch.chdir(tmp_path)  # cwd should not influence the result
        from sinner2.pipeline import model_cache

        expected = Path(model_cache.__file__).resolve().parents[3] / "models"
        assert get_models_dir() == expected


class TestGetModelPath:
    def test_returns_existing_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        f = tmp_path / "weights.onnx"
        f.write_bytes(b"x")
        assert get_model_path("weights.onnx") == f

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            get_model_path("missing.onnx")

    def test_directory_does_not_count_as_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "weights.onnx").mkdir()
        with pytest.raises(FileNotFoundError):
            get_model_path("weights.onnx")


class TestSessionCache:
    def test_caches_per_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        import onnxruntime

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"x")

        call_count = 0

        def fake_session(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return MagicMock()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)

        s1 = get_onnx_session("m.onnx")
        s2 = get_onnx_session("m.onnx")
        assert s1 is s2
        assert call_count == 1

    def test_caches_per_path_and_providers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # rank 35: the same ONNX file requested with DIFFERENT execution
        # providers must NOT share a session (the second consumer would silently
        # get the first's EP). Same (path, providers) still caches.
        import onnxruntime

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"x")

        built: list[tuple] = []

        def fake_session(*_a, providers=None, **_k):
            built.append(tuple(providers or ()))
            return MagicMock()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)

        s1 = get_onnx_session(
            "m.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        s2 = get_onnx_session("m.onnx", providers=["CPUExecutionProvider"])
        assert s1 is not s2  # different EP profile → distinct session
        assert built == [
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
            ("CPUExecutionProvider",),
        ]
        s3 = get_onnx_session("m.onnx", providers=["CPUExecutionProvider"])
        assert s3 is s2  # same (path, providers) → cached, no rebuild
        assert len(built) == 2

    def test_different_models_have_different_sessions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import onnxruntime

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "a.onnx").write_bytes(b"a")
        (tmp_path / "b.onnx").write_bytes(b"b")

        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: MagicMock())

        s1 = get_onnx_session("a.onnx")
        s2 = get_onnx_session("b.onnx")
        assert s1 is not s2

    def test_clear_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        import onnxruntime

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"x")

        call_count = 0

        def fake_session(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return MagicMock()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)

        get_onnx_session("m.onnx")
        clear_session_cache()
        get_onnx_session("m.onnx")
        assert call_count == 2

    def test_release_evicts_only_the_named_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import onnxruntime

        from sinner2.pipeline.model_cache import release_onnx_session

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "a.onnx").write_bytes(b"a")
        (tmp_path / "b.onnx").write_bytes(b"b")

        counts = {"a.onnx": 0, "b.onnx": 0}

        def fake_session(path: str, *_a: object, **_k: object) -> MagicMock:
            counts[Path(path).name] += 1
            return MagicMock()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)

        get_onnx_session("a.onnx")
        get_onnx_session("b.onnx")
        release_onnx_session("a.onnx")        # evict only 'a'
        get_onnx_session("a.onnx")            # reloads
        get_onnx_session("b.onnx")            # still cached
        assert counts["a.onnx"] == 2          # built, evicted, rebuilt
        assert counts["b.onnx"] == 1          # untouched

    def test_release_unknown_session_is_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from sinner2.pipeline.model_cache import release_onnx_session

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        release_onnx_session("never_loaded.onnx")  # must not raise

    def test_release_refcounts_shared_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # N per-worker CodeFormer backends share ONE cached session. One backend
        # releasing must NOT evict the session out from under the others — only
        # the LAST release evicts.
        import onnxruntime

        from sinner2.pipeline.model_cache import release_onnx_session

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"x")
        builds = {"n": 0}

        def fake(*_a: object, **_k: object) -> MagicMock:
            builds["n"] += 1
            return MagicMock()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake)

        s1 = get_onnx_session("m.onnx")  # build, refcount 1
        s2 = get_onnx_session("m.onnx")  # cached, refcount 2
        assert s1 is s2 and builds["n"] == 1
        release_onnx_session("m.onnx")   # refcount 1 — still in use, NOT evicted
        s3 = get_onnx_session("m.onnx")  # still cached (not rebuilt)
        assert s3 is s1 and builds["n"] == 1
        release_onnx_session("m.onnx")   # refcount 1
        release_onnx_session("m.onnx")   # refcount 0 → evict
        get_onnx_session("m.onnx")       # rebuild
        assert builds["n"] == 2


class TestSessionTuning:
    """Every ONNX session is built with tuned CUDA provider options (cuDNN algo
    search + arena strategy) and a graph-optimized SessionOptions. The options
    are applied centrally so the swapper, detector, codeformer, and converters
    are all tuned identically — and this is the seam TensorRT later reuses."""

    @pytest.fixture(autouse=True)
    def _reset_trt_fp16(self):
        from sinner2.pipeline.model_cache import set_tensorrt_fp16

        yield
        set_tensorrt_fp16(False)  # module default; don't leak across tests

    def test_cuda_gets_tuned_options(self):
        opts = build_provider_options(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert len(opts) == 2  # aligned 1:1 with providers (ORT requires it)
        assert opts[0]["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert opts[0]["arena_extend_strategy"] == "kSameAsRequested"
        assert opts[1] == {}  # CPU EP: no options

    def test_cpu_only_gets_empty_options(self):
        opts = build_provider_options(["CPUExecutionProvider"])
        assert opts == [{}]

    def test_tensorrt_gets_engine_cache_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from sinner2.pipeline.model_cache import set_tensorrt_fp16

        # Keep the engine cache out of the repo's models/ dir during tests.
        monkeypatch.setenv("SINNER2_TRT_CACHE_DIR", str(tmp_path / "trt"))
        set_tensorrt_fp16(True)
        opts = build_provider_options(
            ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        )
        # ORT's TRT EP wants "True"/"False" strings (not "1"/"0").
        assert opts[0]["trt_engine_cache_enable"] == "True"
        assert opts[0]["trt_timing_cache_enable"] == "True"
        assert opts[0]["trt_fp16_enable"] == "True"
        assert opts[0]["trt_engine_cache_path"] == str(tmp_path / "trt")
        # CUDA still tuned in the same list.
        assert opts[1]["cudnn_conv_algo_search"] == "EXHAUSTIVE"

    def test_tensorrt_fp16_toggle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from sinner2.pipeline.model_cache import set_tensorrt_fp16

        monkeypatch.setenv("SINNER2_TRT_CACHE_DIR", str(tmp_path / "trt"))
        try:
            set_tensorrt_fp16(True)
            opts = build_provider_options(["TensorrtExecutionProvider"])
            assert opts[0]["trt_fp16_enable"] == "True"
            set_tensorrt_fp16(False)
            opts = build_provider_options(["TensorrtExecutionProvider"])
            assert "trt_fp16_enable" not in opts[0]  # fp16 off → flag omitted
        finally:
            set_tensorrt_fp16(False)  # restore module default (off) for other tests

    def test_options_are_fresh_copies(self):
        # Each call returns independent dicts — mutating one must not corrupt
        # the shared module constant for the next session.
        first = build_provider_options(["CUDAExecutionProvider"])
        first[0]["arena_extend_strategy"] = "kNextPowerOfTwo"
        second = build_provider_options(["CUDAExecutionProvider"])
        assert second[0]["arena_extend_strategy"] == "kSameAsRequested"

    def test_session_options_optimize_graph(self):
        import onnxruntime as ort

        so = build_session_options()
        assert so.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def test_get_onnx_session_applies_tuning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import onnxruntime

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"x")
        captured: dict = {}

        def fake_session(_path: str, **kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)
        get_onnx_session("m.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert captured["providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert captured["provider_options"][0]["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert captured["provider_options"][1] == {}
        assert captured["sess_options"] is not None

    def test_get_onnx_session_empty_providers_stay_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import onnxruntime

        clear_session_cache()
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"x")
        captured: dict = {}

        def fake_session(_path: str, **kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)
        get_onnx_session("m.onnx", [])  # explicit "no providers"
        # NOT substituted with the CUDA+CPU default — passed through empty.
        assert captured["providers"] == []
        assert captured["provider_options"] == []


class TestInsightfaceCache:
    """The insightface swap model (inswapper / reswapper) is cached by
    (path, providers) so a session rebuild — every source/target change —
    reuses the resident model instead of allocating a fresh ORT session and
    stacking VRAM. insightface.model_zoo.get_model has no cache of its own."""

    @staticmethod
    def _stub_get_model(monkeypatch, counter):
        import sys
        import types

        def fake_get_model(path, *_a, **_k):
            counter["n"] += 1
            return MagicMock(name=f"model:{path}")

        mz = types.ModuleType("insightface.model_zoo")
        mz.get_model = fake_get_model  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.model_zoo", mz)

    def test_reuses_model_for_same_path_and_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from sinner2.pipeline.model_cache import get_insightface_swap_model

        clear_session_cache()
        counter = {"n": 0}
        self._stub_get_model(monkeypatch, counter)

        p = Path("/models/inswapper_128.onnx")
        m1 = get_insightface_swap_model(p, ["CPUExecutionProvider"])
        m2 = get_insightface_swap_model(p, ["CPUExecutionProvider"])
        assert m1 is m2
        assert counter["n"] == 1  # built once, reused on the second rebuild

    def test_different_providers_build_distinct_models(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from sinner2.pipeline.model_cache import get_insightface_swap_model

        clear_session_cache()
        counter = {"n": 0}
        self._stub_get_model(monkeypatch, counter)

        p = Path("/models/inswapper_128.onnx")
        m1 = get_insightface_swap_model(p, ["CUDAExecutionProvider"])
        m2 = get_insightface_swap_model(p, ["CPUExecutionProvider"])
        assert m1 is not m2
        assert counter["n"] == 2

    def test_clear_session_cache_evicts_insightface_model(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from sinner2.pipeline.model_cache import get_insightface_swap_model

        clear_session_cache()
        counter = {"n": 0}
        self._stub_get_model(monkeypatch, counter)

        p = Path("/models/inswapper_128.onnx")
        get_insightface_swap_model(p, ["CPUExecutionProvider"])
        clear_session_cache()  # the providers-change path drops it
        get_insightface_swap_model(p, ["CPUExecutionProvider"])
        assert counter["n"] == 2  # rebuilt after the clear

    def test_forwards_tuned_provider_options(self, monkeypatch: pytest.MonkeyPatch):
        import sys
        import types

        from sinner2.pipeline.model_cache import get_insightface_swap_model

        clear_session_cache()
        captured: dict = {}

        def fake_get_model(_path, *_a, **kw):
            captured.update(kw)
            return MagicMock()

        mz = types.ModuleType("insightface.model_zoo")
        mz.get_model = fake_get_model  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.model_zoo", mz)

        get_insightface_swap_model(
            Path("/models/inswapper_128.onnx"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        assert captured["providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        # The inswapper gets the same CUDA tuning as every other model.
        assert captured["provider_options"][0]["arena_extend_strategy"] == (
            "kSameAsRequested"
        )

    def test_empty_providers_stay_empty(self, monkeypatch: pytest.MonkeyPatch):
        import sys
        import types

        from sinner2.pipeline.model_cache import get_insightface_swap_model

        clear_session_cache()
        captured: dict = {}

        def fake_get_model(_path, *_a, **kw):
            captured.update(kw)
            return MagicMock()

        mz = types.ModuleType("insightface.model_zoo")
        mz.get_model = fake_get_model  # type: ignore[attr-defined]
        pkg = sys.modules.get("insightface") or types.ModuleType("insightface")
        monkeypatch.setitem(sys.modules, "insightface", pkg)
        monkeypatch.setitem(sys.modules, "insightface.model_zoo", mz)

        get_insightface_swap_model(Path("/models/inswapper_128.onnx"), [])
        # "No providers" passed through, not substituted with the default.
        assert captured["providers"] == []
        assert captured["provider_options"] == []


class TestAvailableProviders:
    """available_onnx_providers reports what ORT built in. Providers are no
    longer global state — each processor receives them via its execution
    profile — so model_cache only exposes discovery + the `actual` record."""

    def test_available_returns_a_list(self):
        # We can't assert exact contents (depends on ORT install), but
        # CPU is always built-in. Confirm at minimum that.
        providers = available_onnx_providers()
        assert isinstance(providers, list)
        assert "CPUExecutionProvider" in providers


class TestActualProviders:
    """`actual` is what ORT really wired up at session-load time —
    distinct from `active` (what we requested) because ORT silently
    falls back when a requested provider can't initialise (e.g.
    TensorRT EP loads but nvinfer is missing → ORT uses CPU)."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        reset_actual_providers()
        yield
        reset_actual_providers()

    def test_initially_none(self):
        assert get_actual_providers() is None

    def test_record_stores_tuple(self):
        record_actual_providers(["CPUExecutionProvider"])
        assert get_actual_providers() == ("CPUExecutionProvider",)

    def test_reset_clears_actual(self):
        # Test-only reset (no global provider state clears it anymore —
        # the next real session-load overwrites it instead).
        record_actual_providers(["CUDAExecutionProvider"])
        reset_actual_providers()
        assert get_actual_providers() is None

    def test_record_overwrites(self):
        # Last-write-wins — each new session-load supersedes the
        # previous, even if user didn't change selection.
        record_actual_providers(["CUDAExecutionProvider"])
        record_actual_providers(["CPUExecutionProvider"])
        assert get_actual_providers() == ("CPUExecutionProvider",)


class TestDeleteModelEvictsCaches:
    def test_delete_force_evicts_session_and_insightface(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # delete_model must FORCE-evict the cached session(s) regardless of
        # refcount (the file is going away). Borrowing release_onnx_session's
        # decrement left a dangling entry for an N-consumer session (refcount
        # corruption) and never touched the insightface cache, so a deleted
        # inswapper/reswapper stayed resident.
        from sinner2.pipeline import model_cache as mc

        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        clear_session_cache()
        name = "inswapper_128.onnx"
        path = get_models_dir() / name
        skey = (path, ("CUDAExecutionProvider",))
        mc._session_cache[skey] = MagicMock()  # noqa: SLF001
        mc._session_refcount[skey] = 3  # noqa: SLF001  (3 live consumers)
        mc._insightface_cache[skey] = MagicMock()  # noqa: SLF001

        mc.delete_model(name)

        # Force-evicted regardless of refcount; all provider variants for the
        # path are gone from both caches.
        assert all(k[0] != path for k in mc._session_cache)  # noqa: SLF001
        assert all(k[0] != path for k in mc._session_refcount)  # noqa: SLF001
        assert all(k[0] != path for k in mc._insightface_cache)  # noqa: SLF001
        clear_session_cache()


def _io(name: str) -> MagicMock:
    m = MagicMock()
    m.name = name  # MagicMock(name=...) sets the repr, not .name — set it explicitly
    return m


class TestGetOnnxSessionIo:
    """The shared session+I/O-name helper used by the single-in/single-out
    ONNX processors (detectors, occlusion maskers)."""

    def test_returns_session_and_first_input_output_names(self, monkeypatch):
        fake = MagicMock()
        fake.get_inputs.return_value = [_io("in0"), _io("in1")]
        fake.get_outputs.return_value = [_io("out0"), _io("out1")]
        monkeypatch.setattr(model_cache, "get_onnx_session", lambda *a, **k: fake)

        session, in_name, out_name = get_onnx_session_io("m.onnx")
        assert session is fake
        assert in_name == "in0"   # FIRST input
        assert out_name == "out0"  # FIRST output

    def test_forwards_name_and_providers(self, monkeypatch):
        captured = {}
        fake = MagicMock()
        fake.get_inputs.return_value = [_io("i")]
        fake.get_outputs.return_value = [_io("o")]

        def _spy(name, providers=None):
            captured["name"], captured["providers"] = name, providers
            return fake

        monkeypatch.setattr(model_cache, "get_onnx_session", _spy)
        get_onnx_session_io("yolo.onnx", providers=["CUDAExecutionProvider"])
        assert captured == {"name": "yolo.onnx", "providers": ["CUDAExecutionProvider"]}
