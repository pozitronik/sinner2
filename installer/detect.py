"""System detection: OS, architecture, and NVIDIA GPU/driver.

The shell-out to nvidia-smi is kept thin; the output PARSING is a pure function
(parse_gpus) so it's unit-testable without a GPU. stdlib-only.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver_version: str  # e.g. "560.94"
    compute_capability: str | None  # e.g. "12.0" (sm_120); None if not queryable


@dataclass(frozen=True)
class SystemInfo:
    os: str  # "windows" | "linux" | "macos"
    arch: str  # "x86_64" | "arm64" | other platform.machine() value
    gpus: tuple[GpuInfo, ...]
    is_wsl: bool

    @property
    def has_nvidia_gpu(self) -> bool:
        return bool(self.gpus)

    @property
    def driver_version(self) -> str | None:
        return self.gpus[0].driver_version if self.gpus else None


def detect_os() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def detect_arch() -> str:
    """Normalise platform.machine() to the values we branch on."""
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def detect_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def parse_gpus(csv_output: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=name,driver_version,compute_cap
    --format=csv,noheader` output. Tolerates the 2-column form (older drivers
    without compute_cap) and skips blank/garbage lines."""
    gpus: list[GpuInfo] = []
    for line in csv_output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        cc = parts[2] if len(parts) >= 3 and parts[2] and parts[2] != "[N/A]" else None
        gpus.append(
            GpuInfo(name=parts[0], driver_version=parts[1], compute_capability=cc)
        )
    return gpus


def _run_nvidia_smi() -> str | None:
    """Query nvidia-smi; None if it's absent or fails. Retries without
    compute_cap for older drivers that don't support that field."""
    queries = (
        "name,driver_version,compute_cap",
        "name,driver_version",  # fallback for older nvidia-smi
    )
    for query in queries:
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    return None


def detect() -> SystemInfo:
    smi = _run_nvidia_smi()
    gpus = tuple(parse_gpus(smi)) if smi else ()
    return SystemInfo(
        os=detect_os(),
        arch=detect_arch(),
        gpus=gpus,
        is_wsl=detect_wsl(),
    )
