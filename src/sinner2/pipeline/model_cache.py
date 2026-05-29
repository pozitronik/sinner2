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
# User-selected ONNX providers. None = use the platform-default tuple
# (CUDA then CPU). Set via set_active_providers() from the GUI; read by
# anything constructing an ORT session or an insightface model so the
# user's choice flows through every code path that loads ONNX weights.
_active_providers: tuple[str, ...] | None = None
# What ORT actually wired up at session-construction time. Differs from
# _active_providers when a requested provider can't initialise — e.g.
# TensorRT EP DLL loads (so it appears in get_available_providers) but
# nvinfer is missing, and ORT falls back to CPU silently. Surfaced to
# the GUI so the status bar can tell the user the truth.
_actual_providers: tuple[str, ...] | None = None
_providers_lock = threading.RLock()


def available_onnx_providers() -> list[str]:
    """Providers ONNX Runtime reports as built-in for this install.

    Lazy import + lazy call — ORT initialisation isn't free, and we'd
    rather not pay it just for importing this module. Result is not
    cached because the GUI may want to refresh after a runtime change
    (rare, but harmless to allow)."""
    import onnxruntime as ort

    return list(ort.get_available_providers())


def get_active_providers() -> tuple[str, ...]:
    """Effective provider list for new ORT session / insightface model
    creations. Falls back to _DEFAULT_PROVIDERS when the user hasn't
    picked anything yet."""
    with _providers_lock:
        return _active_providers if _active_providers else _DEFAULT_PROVIDERS


def record_actual_providers(providers: list[str] | tuple[str, ...]) -> None:
    """Called by processors after an ORT session loads to surface the
    list of providers ORT actually wired up. May differ from what was
    requested if a provider failed to initialise (missing runtime libs,
    GPU absent, etc.) — in that case ORT silently falls back to the
    next provider in the list. The GUI uses this to show the truth
    rather than the user's intent."""
    global _actual_providers
    with _providers_lock:
        _actual_providers = tuple(providers)


def get_actual_providers() -> tuple[str, ...] | None:
    """What ORT actually used at the most recent session-load, or None
    if no session has loaded since startup. None means "nothing loaded
    yet; use get_active_providers() as the best guess"."""
    with _providers_lock:
        return _actual_providers


def set_active_providers(providers: tuple[str, ...] | list[str] | None) -> None:
    """Replace the active providers and invalidate caches that captured
    the old list. None reverts to the default tuple.

    Side effects: clears the InferenceSession cache (sessions are bound
    to providers at creation time) and the FaceAnalysis singleton
    (insightface picks providers at FaceAnalysis() construction). Callers
    rebuild the chain after this so processors re-load with the new
    providers picked up via get_active_providers().
    """
    global _active_providers, _actual_providers
    with _providers_lock:
        _active_providers = tuple(providers) if providers else None
        # Clear actual too — the next session-load will record what
        # ORT really picked. Without this, the GUI would show stale
        # "actual" during the brief window between selection change
        # and session rebuild.
        _actual_providers = None
    clear_session_cache()
    # Imported here (not at module top) to avoid a circular import —
    # face_analyser imports model_cache for get_active_providers.
    from sinner2.pipeline import face_analyser

    face_analyser.reset_shared_face_analysis()


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
            providers=list(providers) if providers else list(get_active_providers()),
        )
        _session_cache[path] = session
        return session


def clear_session_cache() -> None:
    """Drop all cached ONNX sessions. Test-only; releases GPU memory."""
    with _session_lock:
        _session_cache.clear()
