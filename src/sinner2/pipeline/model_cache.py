import os
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


def get_models_dir() -> Path:
    """Resolve the models directory.

    SINNER2_MODELS_DIR env var takes precedence; otherwise defaults to
    `<cwd>/models`. The env var is the deployment-time override; CWD-relative
    is the dev-time default.
    """
    env = os.environ.get("SINNER2_MODELS_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "models"


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
