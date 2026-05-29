from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.pipeline.model_cache import (
    _DEFAULT_PROVIDERS,
    available_onnx_providers,
    clear_session_cache,
    get_active_providers,
    get_actual_providers,
    get_model_path,
    get_models_dir,
    get_onnx_session,
    record_actual_providers,
    set_active_providers,
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


class TestActiveProviders:
    """Provider state surfaced by model_cache. Used by FaceAnalyser and
    FaceSwapper at construction time; mutated by the GUI controller
    when the user picks a new set of ONNX execution providers."""

    @pytest.fixture(autouse=True)
    def _reset_providers(self):
        # Each test starts and ends with the default state.
        set_active_providers(None)
        yield
        set_active_providers(None)

    def test_available_returns_a_list(self):
        # We can't assert exact contents (depends on ORT install), but
        # CPU is always built-in. Confirm at minimum that.
        providers = available_onnx_providers()
        assert isinstance(providers, list)
        assert "CPUExecutionProvider" in providers

    def test_default_active_is_defaults(self):
        assert get_active_providers() == _DEFAULT_PROVIDERS

    def test_set_changes_active(self):
        set_active_providers(["CPUExecutionProvider"])
        assert get_active_providers() == ("CPUExecutionProvider",)

    def test_set_none_reverts_to_default(self):
        set_active_providers(["CPUExecutionProvider"])
        set_active_providers(None)
        assert get_active_providers() == _DEFAULT_PROVIDERS

    def test_set_empty_reverts_to_default(self):
        # Empty list = "user unchecked everything" — falling back to
        # the defaults is the only sensible answer (CPU is always
        # available; refusing to run is worse UX than running on CPU).
        set_active_providers([])
        assert get_active_providers() == _DEFAULT_PROVIDERS

    def test_set_clears_session_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # Sessions are bound to providers at construction. Changing
        # providers must invalidate any cached sessions.
        monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
        (tmp_path / "m.onnx").write_bytes(b"\0")
        call_count = [0]

        def fake_session(*args, **kwargs):
            call_count[0] += 1
            return MagicMock()

        import sinner2.pipeline.model_cache as mc

        monkeypatch.setattr(
            "onnxruntime.InferenceSession",
            fake_session,
        )
        monkeypatch.setattr(mc, "_preload_bundled_cuda_libs", lambda: None)
        clear_session_cache()
        get_onnx_session("m.onnx")
        # Same model again: cache hit, no extra call.
        get_onnx_session("m.onnx")
        assert call_count[0] == 1
        # Provider change clears the cache, next get rebuilds.
        set_active_providers(["CPUExecutionProvider"])
        get_onnx_session("m.onnx")
        assert call_count[0] == 2

    def test_set_resets_shared_face_analysis(self, monkeypatch: pytest.MonkeyPatch):
        # The shared FaceAnalysis singleton picks providers at
        # construction; set_active_providers must drop it so the next
        # caller rebuilds with the new providers.
        from sinner2.pipeline import face_analyser

        face_analyser._shared_app = "sentinel"  # type: ignore[assignment]  # any non-None value
        set_active_providers(["CPUExecutionProvider"])
        assert face_analyser._shared_app is None  # noqa: SLF001


class TestActualProviders:
    """`actual` is what ORT really wired up at session-load time —
    distinct from `active` (what we requested) because ORT silently
    falls back when a requested provider can't initialise (e.g.
    TensorRT EP loads but nvinfer is missing → ORT uses CPU)."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        set_active_providers(None)
        # Clear actual via the public re-route (set_active_providers
        # clears it as a side effect; there's no separate clear API).
        yield
        set_active_providers(None)

    def test_initially_none(self):
        assert get_actual_providers() is None

    def test_record_stores_tuple(self):
        record_actual_providers(["CPUExecutionProvider"])
        assert get_actual_providers() == ("CPUExecutionProvider",)

    def test_set_active_clears_actual(self):
        # When user picks new providers, the "what's truly loaded"
        # value becomes meaningless until the next session-load. Avoid
        # showing stale truth by clearing it.
        record_actual_providers(["CUDAExecutionProvider"])
        set_active_providers(["TensorrtExecutionProvider"])
        assert get_actual_providers() is None

    def test_record_overwrites(self):
        # Last-write-wins — each new session-load supersedes the
        # previous, even if user didn't change selection.
        record_actual_providers(["CUDAExecutionProvider"])
        record_actual_providers(["CPUExecutionProvider"])
        assert get_actual_providers() == ("CPUExecutionProvider",)
