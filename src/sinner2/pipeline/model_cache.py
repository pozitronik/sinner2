import ctypes
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnxruntime as ort

ProgressCallback = Callable[[int, int], None]
"""bytes_done, bytes_total — invoked during lazy model download."""

_DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

_session_cache: dict[Path, "ort.InferenceSession"] = {}
_session_lock = threading.RLock()
_cuda_preloaded = False


def _preload_bundled_cuda_libs() -> None:
    """Make torch's bundled CUDA libs findable by ORT's later dlopens.

    Two platforms, two mechanisms:
      - **Linux**: ctypes.CDLL every .so under `nvidia/*/lib/` as RTLD_GLOBAL
        so cuDNN/cuBLAS symbols live on the global namespace. The dynamic
        linker doesn't auto-search those package dirs.
      - **Windows**: import torch, which calls `os.add_dll_directory(torch/lib)`
        and registers the bundled CUDA DLLs with the loader for subsequent
        LoadLibraryEx calls (Python 3.8+ on Windows ignores PATH for
        extension-module DLL loads).

    Idempotent. Both branches are no-ops if the relevant package isn't
    installed (CPU-only environments).
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return
    _cuda_preloaded = True
    if sys.platform == "win32":
        try:
            import torch  # noqa: F401  # registers torch/lib with the DLL loader
        except ImportError:
            pass
        return
    if sys.platform != "linux":
        return
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except ImportError:
        return
    paths = list(getattr(nvidia, "__path__", []))
    for prefix in paths:
        for lib_dir in Path(prefix).glob("*/lib"):
            for so in lib_dir.glob("*.so*"):
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _project_models_dir() -> Path:
    """models/ resolved relative to this package's install location.

    For an editable install (`pip install -e .`) this resolves to the repo's
    top-level models/. For a non-editable wheel install the resolution lands
    inside site-packages and won't be useful — set SINNER2_MODELS_DIR in that
    case.
    """
    return Path(__file__).resolve().parents[3] / "models"


def get_models_dir() -> Path:
    """Resolve the models directory.

    SINNER2_MODELS_DIR env var takes precedence (deployment-time override);
    otherwise defaults to `<repo>/models/` resolved from this package's
    install location. CWD-relative was tried in earlier drafts and is wrong
    for any launcher that doesn't `cd` into the project first.
    """
    env = os.environ.get("SINNER2_MODELS_DIR")
    if env:
        return Path(env)
    return _project_models_dir()


def get_model_path(name: str, on_progress: ProgressCallback | None = None) -> Path:
    """Return a local path to the named model file.

    Raises FileNotFoundError if the file is not present. on_progress is
    reserved for lazy-download integration; currently unused — bring your own
    weights via the models dir.
    """
    path = get_models_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"model file not found: {path}\n"
            f"hint: set SINNER2_MODELS_DIR or place the file at {get_models_dir()}/{name}"
        )
    return path


def get_onnx_session(
    name: str,
    providers: list[str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> "ort.InferenceSession":
    """Return a cached ONNX InferenceSession for the named model.

    Sessions are cached at module scope, keyed by absolute path. Loading the
    same model from different Processor instances is free after the first
    call — heavy weights stay resident in GPU memory across the chain.
    """
    _preload_bundled_cuda_libs()
    import onnxruntime as ort

    path = get_model_path(name, on_progress=on_progress)
    with _session_lock:
        cached = _session_cache.get(path)
        if cached is not None:
            return cached
        session = ort.InferenceSession(
            str(path),
            providers=list(providers) if providers else list(_DEFAULT_PROVIDERS),
        )
        _session_cache[path] = session
        return session


def clear_session_cache() -> None:
    """Drop all cached ONNX sessions. Test-only; releases GPU memory."""
    with _session_lock:
        _session_cache.clear()
