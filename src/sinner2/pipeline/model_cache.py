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

# Model files with their download URLs, keyed by the exact filename the
# processors look up. The first two are REQUIRED (downloaded up front); the
# Real-ESRGAN upscaler models are OPTIONAL — registered here only so the
# upscaler can fetch them lazily on first enable (see REQUIRED_MODELS below).
MODEL_SOURCES: dict[str, str] = {
    "inswapper_128.onnx": (
        "https://github.com/pozitronik/sinner/releases/download/v200823/"
        "inswapper_128.onnx"
    ),
    "GFPGANv1.4.pth": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/"
        "GFPGANv1.4.pth"
    ),
    "realesr-general-x4v3.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "realesr-general-x4v3.pth"
    ),
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth"
    ),
    "RealESRGAN_x2plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
        "RealESRGAN_x2plus.pth"
    ),
    # BiSeNet face-parser for occlusion-aware masking. URL taken verbatim from
    # facexlib's parsing/__init__.py (the bisenet weight is on the v0.2.0 tag —
    # NOT v0.2.2, which hosts the parsenet variant).
    "parsing_bisenet.pth": (
        "https://github.com/xinntao/facexlib/releases/download/v0.2.0/"
        "parsing_bisenet.pth"
    ),
    # ParseNet variant — lighter/faster (GFPGAN's default parser). Note the
    # release tag differs from bisenet's (v0.2.2 vs v0.2.0).
    "parsing_parsenet.pth": (
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/"
        "parsing_parsenet.pth"
    ),
    # CodeFormer face-restorer (ONNX, facefusion-assets). The `w` fidelity is a
    # scalar model input. ~377 MB.
    "codeformer.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/codeformer.onnx"
    ),
    # ---- Alternative face-swap models (optional; lazy on selection) ----
    # ReSwapper: clean-room reproduction of inswapper, drop-in via insightface's
    # INSwapper class (same ArcFace embedding contract). Saved under our own
    # filename; the upstream asset is reswapper-1019500.onnx. ~554 MB.
    "reswapper_128.onnx": (
        "https://huggingface.co/somanchiu/reswapper/resolve/main/"
        "reswapper-1019500.onnx"
    ),
    # Ghost / SimSwap / UniFace: facefusion-style ONNX swappers (256px), driven
    # by GenericOnnxSwapper. Ghost + SimSwap also need a crossface embedding
    # converter (below). All from facefusion-assets, verified.
    "ghost_1_256.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/ghost_1_256.onnx"
    ),
    "ghost_2_256.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/ghost_2_256.onnx"
    ),
    "ghost_3_256.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/ghost_3_256.onnx"
    ),
    "simswap_256.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/simswap_256.onnx"
    ),
    "uniface_256.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/uniface_256.onnx"
    ),
    # ArcFace→model embedding converters (facefusion crossface). Companion to
    # ghost / simswap respectively; ~22 MB each.
    "crossface_ghost.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.4.0/crossface_ghost.onnx"
    ),
    "crossface_simswap.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.4.0/crossface_simswap.onnx"
    ),
}

# Downloaded up front by the first-run GUI flow. Optional models (the upscaler
# weights) are fetched lazily by their processor, so users who never upscale
# don't pay the download.
REQUIRED_MODELS: tuple[str, ...] = (
    "inswapper_128.onnx",
    "GFPGANv1.4.pth",
)

_DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

_session_cache: dict[Path, "ort.InferenceSession"] = {}
_session_lock = threading.RLock()
_cuda_preloaded = False
# What ORT actually wired up at session-construction time. Differs from the
# REQUESTED providers when one can't initialise — e.g. the TensorRT EP DLL
# loads (so it appears in get_available_providers) but nvinfer is missing, and
# ORT silently falls back to CPU. Recorded by processors after a session loads
# and surfaced to the GUI so the status bar can show the truth. Providers
# themselves are no longer global state — each processor receives its
# execution profile explicitly (see config.execution).
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
    yet; use the requested providers as the best guess"."""
    with _providers_lock:
        return _actual_providers


def reset_actual_providers() -> None:
    """Drop the recorded `actual` list. Test-only — the value is otherwise
    overwritten by the next session-load via record_actual_providers()."""
    global _actual_providers
    with _providers_lock:
        _actual_providers = None


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

    Raises FileNotFoundError if the file is not present. Downloading is handled
    up front by the GUI (see gui/model_download.py + download_model below), not
    lazily here, so processing never blocks mid-setup on a multi-hundred-MB
    fetch.
    """
    path = get_models_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"model file not found: {path}\n"
            f"hint: set SINNER2_MODELS_DIR or place the file at {get_models_dir()}/{name}"
        )
    return path


def _model_present(name: str) -> bool:
    # Present AND non-empty — a zero-byte leftover (e.g. an aborted manual
    # copy) shouldn't count as installed.
    path = get_models_dir() / name
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def model_present(name: str) -> bool:
    """Whether the named model file is present (and non-empty)."""
    return _model_present(name)


def missing_models() -> list[str]:
    """Which of the REQUIRED model files aren't present in the models dir.
    Optional models (upscaler weights) are excluded — they download lazily."""
    return [name for name in REQUIRED_MODELS if not _model_present(name)]


def download_model(
    name: str,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Stream the named model from its URL into the models dir.

    Writes to a ``.part`` file and renames on completion, so a partial or
    cancelled download never leaves a fake-complete model file behind.
    ``on_progress(bytes_done, bytes_total)`` is called as data arrives
    (bytes_total is 0 if the server doesn't send Content-Length). Returns
    early (cleaning up) if ``should_cancel()`` becomes true; raises on any
    network / filesystem error.
    """
    import urllib.request

    url = MODEL_SOURCES.get(name)
    if url is None:
        raise ValueError(f"no download URL registered for model {name!r}")
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / name
    part = dest.with_name(dest.name + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "sinner2"})
    try:
        # timeout bounds a stalled connection so read() can't block forever —
        # lets a cancel (checked between chunks) take effect within ~timeout.
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            if on_progress is not None:
                on_progress(0, total)
            with open(part, "wb") as out:
                while True:
                    if should_cancel is not None and should_cancel():
                        out.close()
                        part.unlink(missing_ok=True)
                        return
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
        part.replace(dest)  # atomic: final name appears only on full success
    except Exception as exc:
        part.unlink(missing_ok=True)
        # Surface the URL — the raw HTTPError ("404: Not Found") doesn't name
        # what it hit, so the GUI's failure message was undebuggable.
        raise RuntimeError(f"failed to download {name} from {url}: {exc}") from exc
    except BaseException:
        part.unlink(missing_ok=True)  # KeyboardInterrupt / SystemExit — clean up + propagate
        raise


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
