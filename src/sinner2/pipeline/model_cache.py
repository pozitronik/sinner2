import ctypes
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    # SwinIR real-SR "M" x4 (torch/.pth via basicsr swinir_arch). Verified to
    # load strict against the _build_swinir config. ~64 MB.
    "swinir_realsr_m_x4.pth": (
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
        "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    ),
    # ONNX whole-frame upscalers (facefusion-assets) — HAT + community ESRGAN
    # models. Single dynamic image input/output, RGB [0,1] (verified). Lazy.
    "real_hatgan_x4.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/real_hatgan_x4.onnx"
    ),
    "ultra_sharp_x4.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/ultra_sharp_x4.onnx"
    ),
    "span_kendata_x4.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/span_kendata_x4.onnx"
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
    # GFPGAN v1.4 as ONNX (facefusion export) — same generator as the bundled
    # .pth but driven by PlainBfrBackend (per-face align → restore → paste)
    # instead of the torch+facexlib pipeline, which is several times slower
    # (its detect + full-frame paste dominated the frame cost). Same [-1,1]
    # 512px contract as GPEN. Lazy on selection.
    "gfpgan_1.4.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/gfpgan_1.4.onnx"
    ),
    # GPEN-BFR-512 + RestoreFormer++ — plain ONNX face restorers (no knobs),
    # higher detail than GFPGAN-512. facefusion-assets, verified. Single image
    # input/output in [-1,1]; driven by PlainBfrBackend. Lazy on selection.
    "gpen_bfr_512.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/gpen_bfr_512.onnx"
    ),
    "gpen_bfr_1024.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/gpen_bfr_1024.onnx"
    ),
    "gpen_bfr_2048.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/gpen_bfr_2048.onnx"
    ),
    "restoreformer_plus_plus.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/restoreformer_plus_plus.onnx"
    ),
    # ---- Alternative target detectors (optional; lazy on selection) ----
    # Detection-only models (box + 5 keypoints) — faster than buffalo_l's full
    # pack for per-frame target detection. yoloface parsing verified against
    # buffalo_l; scrfd runs through insightface's own SCRFD class.
    "yoloface_8n.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/yoloface_8n.onnx"
    ),
    "scrfd_2.5g.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/"
        "models-3.0.0/scrfd_2.5g.onnx"
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

# Tuned CUDA execution-provider options, applied to EVERY ONNX session we build
# (swapper, detector, codeformer, the crossface converters). Centralized here so
# the whole pipeline is tuned identically, and so this stays the single seam a
# TensorRT EP later slots its engine-cache options into.
#   - cudnn_conv_algo_search=EXHAUSTIVE: benchmark cuDNN conv algorithms once and
#     reuse the fastest. The inswapper/SCRFD/codeformer nets are fixed-shape, so
#     the one-time search at first inference amortizes across the whole session /
#     batch — typically a few-to-15% steady-state win on the conv-heavy models.
#   - arena_extend_strategy=kSameAsRequested: grow the device arena by exactly
#     what's requested instead of rounding up to the next power of two — lower
#     peak VRAM and steadier arena growth (matters with the per-worker footprint
#     and the arena jitter seen across session swaps).
# Strings (ORT parses provider-option values as strings). Override here, not at
# call sites, so every model stays consistent.
_CUDA_PROVIDER_OPTIONS: dict[str, str] = {
    "cudnn_conv_algo_search": "EXHAUSTIVE",
    "arena_extend_strategy": "kSameAsRequested",
}


# TensorRT engine precision. fp32 (off) by design: inswapper_128's TRT fp16
# engine produces a corrupted swap (max abs error ~0.93 vs CUDA, measured on the
# 5090), so there is deliberately NO GUI toggle for it. fp32 TRT matches CUDA and
# is still faster. The SINNER2_TENSORRT_FP16 env var is an escape hatch for
# experimenting with a different, fp16-clean swap model — not a user-facing knob.
# Baked into the built engine, so a change needs a new (precision-specific) build.
_tensorrt_fp16 = os.environ.get("SINNER2_TENSORRT_FP16", "").lower() in (
    "1", "true", "yes", "on",
)
_tensorrt_preloaded = False


def set_tensorrt_fp16(enabled: bool) -> None:
    """Programmatic override for TensorRT fp16 (used by tests / advanced
    callers). Affects engines built AFTER the change (the flag is compiled into
    the engine); existing cached engines keep their precision until the cache
    directory is cleared. No GUI toggle exposes this — fp16 corrupts the bundled
    inswapper model."""
    global _tensorrt_fp16
    _tensorrt_fp16 = bool(enabled)


def get_trt_cache_dir() -> Path:
    """Directory for compiled TensorRT engines + the timing cache.

    Persistent so the slow first-run engine build is paid ONCE — later launches
    load the cached engine in ~1s. `SINNER2_TRT_CACHE_DIR` overrides; defaults to
    `<models>/trt_engines/`. Engines are GPU / driver / TRT-version specific;
    ORT rebuilds them automatically (and re-caches) when any of those change.
    """
    env = os.environ.get("SINNER2_TRT_CACHE_DIR")
    if env:
        return Path(env)
    return get_models_dir() / "trt_engines"


def tensorrt_engine_cached() -> bool:
    """True if at least one compiled TensorRT engine already exists in the cache
    dir, so the next TRT session LOADS one (fast) rather than COMPILING.

    Lets the GUI skip the 'compiling…' modal when there's nothing to compile —
    e.g. after toggling the provider off and on again, where TRT is no longer the
    active provider but the engine is still on disk. (`.engine` files only; the
    `.timing` cache alone doesn't mean an engine is built.)
    """
    cache = get_trt_cache_dir()
    try:
        return cache.is_dir() and any(cache.glob("*.engine"))
    except OSError:
        return False


def _tensorrt_libs_dir() -> Path | None:
    """Locate the `tensorrt_libs` package dir (where the `tensorrt-cu12` wheel
    drops nvinfer_10.dll etc.) WITHOUT importing it (find_spec doesn't run the
    package __init__, which loads the heavy bindings)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("tensorrt_libs")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    if spec.origin:
        return Path(spec.origin).parent
    return None


def _preload_tensorrt_libs() -> None:
    """Make the TensorRT runtime DLLs loadable by ORT's TensorRT EP.

    ORT's onnxruntime_providers_tensorrt.dll depends on nvinfer_10.dll, which in
    turn depends on the CUDA runtime (cudart/cuBLAS/cuDNN). Two things must be on
    the loader's search path before the EP loads, or it fails with "nvinfer_10.dll
    missing" (Error 126 — really a transitive-dependency miss) and ORT silently
    falls back to CUDA:
      1. the CUDA runtime — handled by _preload_bundled_cuda_libs (torch/lib),
      2. the TensorRT libs — registered by importing `tensorrt`, which is how the
         NVIDIA wheel wires its own DLL directory (a bare os.add_dll_directory of
         tensorrt_libs isn't enough — nvinfer still can't find cudart).
    Importing tensorrt is heavier than a path tweak, but it only happens when the
    user has actually selected the TRT provider. No-op when TensorRT isn't
    installed.
    """
    global _tensorrt_preloaded
    if _tensorrt_preloaded:
        return
    _tensorrt_preloaded = True
    # CUDA runtime first — nvinfer links it.
    _preload_bundled_cuda_libs()
    try:
        import tensorrt  # noqa: F401  registers tensorrt_libs with the loader
        return
    except ImportError:
        pass
    # Fallback if the bindings can't import but the libs are present.
    if sys.platform == "win32":
        libs = _tensorrt_libs_dir()
        if libs is not None and libs.is_dir():
            try:
                os.add_dll_directory(str(libs))
            except (OSError, AttributeError):
                pass


def _tensorrt_provider_options() -> dict[str, str]:
    """Engine-cache + precision options for the TensorRT EP. Builds (once) a
    persistent engine + timing cache so the slow compile is one-time, and
    enables fp16 unless the GUI toggled it off. Also ensures the TRT DLLs are on
    the loader path (so the EP actually engages instead of falling back)."""
    _preload_tensorrt_libs()
    cache = get_trt_cache_dir()
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    # ORT's TensorRT EP parses boolean options as the strings "True"/"False"
    # (NOT "1"/"0") — a "1" is rejected and the whole EP fails to construct,
    # falling back to CUDA.
    opts = {
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(cache),
        "trt_timing_cache_enable": "True",
    }
    if _tensorrt_fp16:
        opts["trt_fp16_enable"] = "True"
    return opts


def build_provider_options(providers: list[str]) -> list[dict[str, str]]:
    """Per-provider options aligned 1:1 with ``providers``.

    CUDA gets the tuned cuDNN/arena options; TensorRT gets engine-cache + fp16
    options (and triggers the DLL preload); every other EP (CPU) gets an empty
    dict. ORT requires provider_options to be the same length as providers when
    supplied, hence the 1:1 mapping.
    """
    out: list[dict[str, str]] = []
    for p in providers:
        if p == "CUDAExecutionProvider":
            out.append(dict(_CUDA_PROVIDER_OPTIONS))
        elif p == "TensorrtExecutionProvider":
            out.append(_tensorrt_provider_options())
        else:
            out.append({})
    return out


def build_session_options() -> "ort.SessionOptions":
    """Tuned SessionOptions for a direct InferenceSession (codeformer / generic
    swappers / converters).

    Graph optimization at ALL (operator fusion + constant folding) and memory-
    pattern planning on. insightface builds its OWN sessions and forwards only
    provider options (not sess_options), so the swap/detect models get the
    provider tuning above plus ORT's default SessionOptions — which already
    defaults graph optimization to ALL, so they lose nothing meaningful.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    return so


_session_cache: dict[tuple[Path, tuple[str, ...]], "ort.InferenceSession"] = {}
# Per-path consumer refcount for _session_cache. A single cached session can be
# SHARED by several consumers (notably the N per-worker CodeFormer backends, each
# of which get_onnx_session's the same path). release_onnx_session must only
# evict when the LAST consumer releases — otherwise one worker's release pulls
# the session out from under the others and a later get rebuilds a duplicate
# (VRAM leak). get bumps the count; release decrements and evicts at 0.
_session_refcount: dict[tuple[Path, tuple[str, ...]], int] = {}
# Cached insightface swap models (inswapper / reswapper), keyed by
# (path, providers). insightface.model_zoo.get_model builds a BRAND-NEW ORT
# session on every call and never caches, so without this a source/target
# change — which rebuilds the swapper — would allocate a fresh inswapper
# session each time, its VRAM stacking up cycle after cycle. Caching here makes
# the swap model a process-wide resident, the same way the buffalo_l detector
# already is. Keyed by providers so a providers change (which calls
# clear_session_cache) rebuilds it on the new EP list.
_insightface_cache: dict[tuple[Path, tuple[str, ...]], Any] = {}
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


def model_size_on_disk(name: str) -> int:
    """Bytes the installed model occupies, or 0 if it isn't present."""
    path = get_models_dir() / name
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _force_evict_caches(path: Path) -> None:
    """Force-drop a model's cached session(s) regardless of refcount — used when
    the file is going away (delete_model). The consumer-refcount API
    (release_onnx_session) only evicts at the LAST release, so for an N-consumer
    session it would leave a dangling entry and skew the count; and it never
    touches the insightface cache (keyed by (path, providers)), so a deleted
    inswapper/reswapper would stay resident. This pops both, for every provider
    variant of the path."""
    import gc

    evicted = False
    with _session_lock:
        for key in [k for k in _session_cache if k[0] == path]:
            _session_cache.pop(key, None)
            _session_refcount.pop(key, None)
            evicted = True
        for key in [k for k in _insightface_cache if k[0] == path]:
            _insightface_cache.pop(key, None)
            evicted = True
    if evicted:
        gc.collect()


def delete_model(name: str) -> bool:
    """Delete a model's file from the models dir. Returns True if a file was
    removed (False if it wasn't there). Force-evicts any cached session(s) so a
    re-download re-loads cleanly and no stale model lingers in VRAM. The caller
    is responsible for confirming — especially for REQUIRED models, which the
    app needs to run."""
    path = get_models_dir() / name
    _force_evict_caches(path)
    try:
        if path.is_file():
            path.unlink()
            return True
    except OSError:
        pass
    return False


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
    # None → platform default; an explicit [] stays empty (ORT then uses its CPU
    # last-resort). Only an unspecified (None) list falls back to the default.
    names = list(_DEFAULT_PROVIDERS) if providers is None else list(providers)
    # Key by (path, providers): the same ONNX file requested with a DIFFERENT EP
    # profile must get its own session, not silently inherit the first caller's
    # providers (matches how _insightface_cache is keyed).
    key = (path, tuple(names))
    with _session_lock:
        cached = _session_cache.get(key)
        if cached is not None:
            _session_refcount[key] = _session_refcount.get(key, 0) + 1
            return cached
        session = ort.InferenceSession(
            str(path),
            sess_options=build_session_options(),
            providers=names,
            provider_options=build_provider_options(names),
        )
        _session_cache[key] = session
        _session_refcount[key] = 1
        return session


def release_onnx_session(name: str, providers: list[str] | None = None) -> None:
    """Drop one consumer's hold on the cached ONNX session for ``name``; evict +
    free its device memory only when the LAST consumer releases.

    Sessions are keyed by (path, providers), so pass the SAME ``providers`` used
    to acquire the session (None → the platform default, matching get). Called by
    a processor's release() when its feature is disabled so the model doesn't
    linger in VRAM (ORT frees a session's GPU arena when the last reference
    drops). A session can be shared (N per-worker CodeFormer backends), so this
    is refcounted: each get_onnx_session bumped the count, each release decrements
    it, and only the final release pops + collects. No-op if nothing is cached
    for that (name, providers). Re-enabling reloads via get_onnx_session.
    """
    import gc

    path = get_models_dir() / name
    names = list(_DEFAULT_PROVIDERS) if providers is None else list(providers)
    key = (path, tuple(names))
    with _session_lock:
        if key not in _session_cache:
            return
        remaining = _session_refcount.get(key, 1) - 1
        if remaining > 0:
            _session_refcount[key] = remaining
            return  # other consumers still hold it
        _session_refcount.pop(key, None)
        session = _session_cache.pop(key, None)
    if session is None:
        return
    # Last consumer gone — drop the local ref and force a collection so ORT's
    # session destructor runs now (freeing the CUDA arena) rather than at a
    # later GC.
    del session
    gc.collect()


def get_insightface_swap_model(
    path: Path, providers: list[str] | None = None
) -> Any:
    """Return a cached insightface swap model (INSwapper-family) for ``path``.

    insightface's ``model_zoo.get_model`` builds a fresh ORT session every call
    and keeps no cache of its own, so the swapper's ``setup()`` would otherwise
    allocate a new inswapper session on every session rebuild (each source /
    target change), leaking VRAM cycle after cycle. We cache the wrapped model
    by (path, providers) and reuse it — a rebuild then re-binds the resident
    model instead of loading another copy. The model stays resident across
    swapper teardown (FaceSwapper.release just drops its local ref); the
    providers-change path clears this via ``clear_session_cache``.
    """
    from insightface.model_zoo import get_model

    # None → platform default; an explicit [] stays empty (→ ORT CPU last-resort).
    eps = _DEFAULT_PROVIDERS if providers is None else tuple(providers)
    key = (path, eps)
    with _session_lock:
        cached = _insightface_cache.get(key)
        if cached is not None:
            return cached
        # insightface forwards providers + provider_options straight to ORT's
        # InferenceSession, so the inswapper gets the same CUDA tuning as every
        # other model (sess_options aren't forwarded, but ORT's default already
        # optimizes the graph at ALL).
        model = get_model(
            str(path),
            providers=list(eps),
            provider_options=build_provider_options(list(eps)),
        )
        _insightface_cache[key] = model
        return model


def clear_session_cache() -> None:
    """Drop all cached ONNX sessions + insightface swap models. Releases GPU
    memory. Used by the providers-change path (so models rebuild on the new EP
    list) and by tests."""
    with _session_lock:
        _session_cache.clear()
        _session_refcount.clear()
        _insightface_cache.clear()
