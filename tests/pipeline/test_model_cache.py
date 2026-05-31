from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.pipeline.model_cache import (
    available_onnx_providers,
    clear_session_cache,
    get_actual_providers,
    get_model_path,
    get_models_dir,
    get_onnx_session,
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
