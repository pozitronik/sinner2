import sys

import pytest

from sinner2.pipeline import model_cache


@pytest.fixture(autouse=True)
def _reset_flag():
    original = model_cache._cuda_preloaded
    yield
    model_cache._cuda_preloaded = original


class TestPreloadBundledCudaLibs:
    def test_windows_imports_torch(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "win32")
        model_cache._cuda_preloaded = False
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
        attempted: list[str] = []

        def patched(name: str, *args, **kwargs):
            if name == "torch":
                attempted.append(name)
                return type(sys)("torch")  # stub module
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", patched)
        model_cache._preload_bundled_cuda_libs()
        assert attempted == ["torch"]

    def test_windows_skips_when_torch_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "win32")
        model_cache._cuda_preloaded = False
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def patched(name: str, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", patched)
        # Must not raise
        model_cache._preload_bundled_cuda_libs()

    def test_no_op_on_other_platforms(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        model_cache._cuda_preloaded = False
        # Neither Linux nor Windows branch runs — must not raise
        model_cache._preload_bundled_cuda_libs()

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch):
        model_cache._cuda_preloaded = False
        attempts: list[int] = []

        def fake_import(_name: str):
            attempts.append(1)
            raise ImportError("no nvidia for test")

        monkeypatch.setattr("builtins.__import__", lambda name, *a, **k: fake_import(name) if name == "nvidia" else __import__(name, *a, **k))
        model_cache._preload_bundled_cuda_libs()
        model_cache._preload_bundled_cuda_libs()
        model_cache._preload_bundled_cuda_libs()
        # nvidia should have been attempted at most once
        assert len(attempts) <= 1

    def test_returns_silently_when_nvidia_missing(self, monkeypatch: pytest.MonkeyPatch):
        model_cache._cuda_preloaded = False
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def patched(name: str, *args, **kwargs):
            if name == "nvidia":
                raise ImportError("no nvidia in this test env")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", patched)
        # Must not raise
        model_cache._preload_bundled_cuda_libs()
