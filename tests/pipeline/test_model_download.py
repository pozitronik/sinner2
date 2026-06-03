"""Tests for model presence-detection + download (mocked HTTP)."""
from __future__ import annotations

import pytest

from sinner2.pipeline import model_cache


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
    return tmp_path


class _FakeResponse:
    """Minimal stand-in for urlopen()'s response context manager."""

    def __init__(self, data: bytes, total: int | None = None) -> None:
        self._data = data
        self._pos = 0
        # total=None → omit Content-Length (server didn't send it)
        self.headers = {} if total is None else {"Content-Length": str(total)}

    def read(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class TestModelSources:
    def test_required_models_registered_with_urls(self):
        # The required pair drives the first-run download.
        assert set(model_cache.REQUIRED_MODELS) == {
            "inswapper_128.onnx",
            "GFPGANv1.4.pth",
        }
        # Every registered model (required + optional upscaler) has a URL.
        for name in model_cache.REQUIRED_MODELS:
            assert name in model_cache.MODEL_SOURCES
        for url in model_cache.MODEL_SOURCES.values():
            assert url.startswith("https://")

    def test_optional_upscaler_models_registered_but_not_required(self):
        assert "RealESRGAN_x4plus.pth" in model_cache.MODEL_SOURCES
        assert "RealESRGAN_x4plus.pth" not in model_cache.REQUIRED_MODELS

    def test_onnx_enhancer_models_registered_lazy(self):
        # GPEN-512 + RestoreFormer++ are optional (download on selection), like
        # CodeFormer — registered with URLs but not in the required set.
        for name in ("gpen_bfr_512.onnx", "restoreformer_plus_plus.onnx"):
            assert name in model_cache.MODEL_SOURCES
            assert name not in model_cache.REQUIRED_MODELS


class TestMissingModels:
    def test_all_required_missing_in_empty_dir(self, models_dir):
        # Only the required models are reported missing — optional upscaler
        # weights download lazily, so they don't gate first-run setup.
        assert set(model_cache.missing_models()) == set(model_cache.REQUIRED_MODELS)

    def test_present_models_drop_out(self, models_dir):
        (models_dir / "inswapper_128.onnx").write_bytes(b"weights")
        assert model_cache.missing_models() == ["GFPGANv1.4.pth"]
        (models_dir / "GFPGANv1.4.pth").write_bytes(b"weights")
        assert model_cache.missing_models() == []

    def test_zero_byte_file_counts_as_missing(self, models_dir):
        (models_dir / "inswapper_128.onnx").write_bytes(b"")
        assert "inswapper_128.onnx" in model_cache.missing_models()


class TestDownloadModel:
    def test_writes_file_and_reports_progress(self, models_dir, monkeypatch):
        data = b"x" * 1000
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *_a, **_k: _FakeResponse(data, len(data))
        )
        seen: list[tuple[int, int]] = []
        model_cache.download_model(
            "inswapper_128.onnx", on_progress=lambda d, t: seen.append((d, t))
        )
        dest = models_dir / "inswapper_128.onnx"
        assert dest.read_bytes() == data
        assert seen[0] == (0, 1000)
        assert seen[-1] == (1000, 1000)
        # The temp file is gone (renamed into place).
        assert not (models_dir / "inswapper_128.onnx.part").exists()

    def test_unknown_model_raises(self, models_dir):
        with pytest.raises(ValueError, match="no download URL"):
            model_cache.download_model("nope.bin")

    def test_cancel_midstream_leaves_no_files(self, models_dir, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *_a, **_k: _FakeResponse(b"x" * 9999, 9999)
        )
        model_cache.download_model("inswapper_128.onnx", should_cancel=lambda: True)
        assert not (models_dir / "inswapper_128.onnx").exists()
        assert not (models_dir / "inswapper_128.onnx.part").exists()

    def test_error_midstream_cleans_up_part(self, models_dir, monkeypatch):
        class _FailingResponse(_FakeResponse):
            def __init__(self):
                super().__init__(b"x" * 9999, 9999)
                self._calls = 0

            def read(self, _n):
                # Small fixed chunks so the loop iterates a few times before
                # the failure (the real chunk size would swallow it in one go).
                self._calls += 1
                if self._calls > 2:
                    raise OSError("connection reset")
                return b"x" * 1000

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *_a, **_k: _FailingResponse()
        )
        with pytest.raises(RuntimeError, match="connection reset") as exc:
            model_cache.download_model("inswapper_128.onnx")
        # The URL is surfaced so a download failure is debuggable.
        assert "https://" in str(exc.value)
        assert not (models_dir / "inswapper_128.onnx").exists()
        assert not (models_dir / "inswapper_128.onnx.part").exists()

    def test_no_content_length_still_downloads(self, models_dir, monkeypatch):
        data = b"y" * 500
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *_a, **_k: _FakeResponse(data, total=None)
        )
        model_cache.download_model("GFPGANv1.4.pth")
        assert (models_dir / "GFPGANv1.4.pth").read_bytes() == data
