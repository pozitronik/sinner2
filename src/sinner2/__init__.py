from importlib.metadata import PackageNotFoundError, version

from sinner2.pipeline.model_cache import _preload_bundled_cuda_libs as _preload

# Run on package import so insightface / gfpgan / direct ORT users all find
# the bundled CUDA libs without depending on LD_LIBRARY_PATH or the venv's
# activate script. Idempotent and Linux-only; no-op elsewhere.
_preload()

try:
    # Git tags are the source of truth; hatch-vcs stamps this into the installed
    # distribution metadata at build/install time.
    __version__ = version("sinner2")
except PackageNotFoundError:
    # Running from a source tree that was never installed (no dist metadata) —
    # e.g. tests run straight off the checkout. Fall back to a sentinel.
    __version__ = "0.0.0+unknown"
