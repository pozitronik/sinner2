from sinner2.pipeline.model_cache import _preload_bundled_cuda_libs as _preload

# Run on package import so insightface / gfpgan / direct ORT users all find
# the bundled CUDA libs without depending on LD_LIBRARY_PATH or the venv's
# activate script. Idempotent and Linux-only; no-op elsewhere.
_preload()

__version__ = "0.1.0"
