"""Hardware → install-variant decision.

Maps a detected SystemInfo to the recommended pyproject extra (cpu / cuda /
cuda118 / mac-arm) with a human-readable reason. Pure + stdlib-only.

Driver thresholds (by major version) are conservative cross-platform minimums:
  - CUDA 12.8 (the `cuda` extra, cu128 wheels) needs driver branch ~525+.
  - CUDA 11.8 (the `cuda118` extra, cu118 wheels) needs driver branch ~450+.
Blackwell GPUs (compute capability >= 12.0, e.g. RTX 50xx) have NO cu118
kernels, so an old driver means they can't use the GPU at all → CPU + "update
the driver".
"""
from __future__ import annotations

from dataclasses import dataclass

from installer.detect import SystemInfo

_CUDA128_MIN_DRIVER_MAJOR = 525
_CUDA118_MIN_DRIVER_MAJOR = 450

# Every selectable variant (the wizard lists these so the user can override the
# recommendation — e.g. force CPU to skip the driver hassle).
ALL_VARIANTS = ("cuda", "cuda118", "cpu", "mac-arm")


@dataclass(frozen=True)
class Recommendation:
    variant: str  # one of ALL_VARIANTS
    reason: str
    # GPU present but unusable as-is (driver too old) → we fall back to CPU but
    # tell the user a driver update would unlock the GPU.
    gpu_blocked: bool = False


def _driver_major(version: str | None) -> int | None:
    if not version:
        return None
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return None


def _cc_at_least(cc: str | None, minimum: tuple[int, int]) -> bool:
    if not cc:
        return False
    try:
        parts = tuple(int(x) for x in cc.split("."))
    except ValueError:
        return False
    return parts >= minimum


def recommend(info: SystemInfo) -> Recommendation:
    if info.os == "macos":
        if info.arch == "arm64":
            return Recommendation("mac-arm", "Apple Silicon detected")
        return Recommendation("cpu", "Intel Mac — no CUDA; CPU build")

    if not info.gpus:
        return Recommendation("cpu", "No NVIDIA GPU detected — CPU build")

    gpu = info.gpus[0]
    major = _driver_major(gpu.driver_version)
    is_blackwell = _cc_at_least(gpu.compute_capability, (12, 0))

    if major is None:
        return Recommendation(
            "cpu",
            f"{gpu.name}: couldn't read the driver version — using CPU",
            gpu_blocked=True,
        )
    if major >= _CUDA128_MIN_DRIVER_MAJOR:
        return Recommendation(
            "cuda", f"{gpu.name}, driver {gpu.driver_version} → CUDA 12.8 build"
        )
    if is_blackwell:
        return Recommendation(
            "cpu",
            f"{gpu.name} needs driver {_CUDA128_MIN_DRIVER_MAJOR}+ for CUDA 12.8 "
            f"(it has no CUDA 11.8 support); update the driver to use the GPU",
            gpu_blocked=True,
        )
    if major >= _CUDA118_MIN_DRIVER_MAJOR:
        return Recommendation(
            "cuda118",
            f"{gpu.name}, driver {gpu.driver_version} → CUDA 11.8 build "
            "(driver too old for 12.8)",
        )
    return Recommendation(
        "cpu",
        f"{gpu.name} driver {gpu.driver_version} is too old for CUDA; "
        "update the driver or use the CPU build",
        gpu_blocked=True,
    )
