from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.pipeline.model_cache import (
    clear_session_cache,
    get_model_path,
    get_models_dir,
    get_onnx_session,
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
