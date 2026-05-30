"""Install operations.

Command CONSTRUCTION is pure (testable); execution is a thin subprocess
wrapper. The install follows the recipe that's known to work (see the
wsl-gpu-setup notes): create a venv, install the right torch build from the
matching PyTorch index, install the app + its extra, patch basicsr for modern
torchvision, and make onnxruntime-gpu win over the plain onnxruntime that
insightface drags in. stdlib-only.
"""
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

# PyTorch wheel index per variant. mac-arm uses the default PyPI wheels.
_TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu128",
    "cuda118": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
    "mac-arm": None,
}

_BASICSR_OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
_BASICSR_NEW = "from torchvision.transforms.functional import rgb_to_grayscale"


def is_gpu_variant(variant: str) -> bool:
    return variant in ("cuda", "cuda118")


def torch_index_url(variant: str) -> str | None:
    return _TORCH_INDEX.get(variant)


def venv_python(venv_dir: Path, os: str) -> Path:
    if os == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def site_packages_dir(venv_dir: Path, os: str, py_tag: str = "python3.12") -> Path:
    if os == "windows":
        return venv_dir / "Lib" / "site-packages"
    return venv_dir / "lib" / py_tag / "site-packages"


# ---- Command builders (pure) ----

def create_venv_command(uv: str, venv_dir: Path, python: str = "3.12") -> list[str]:
    return [uv, "venv", "--python", python, str(venv_dir)]


def torch_install_command(uv: str, py: Path, variant: str) -> list[str]:
    cmd = [uv, "pip", "install", "--python", str(py), "torch", "torchvision"]
    index = torch_index_url(variant)
    if index:
        cmd += ["--index-url", index]
    return cmd


def app_install_command(
    uv: str, py: Path, variant: str, project_dir: str = "."
) -> list[str]:
    return [uv, "pip", "install", "--python", str(py), "-e", f"{project_dir}[{variant},gui]"]


def ort_gpu_reinstall_command(uv: str, py: Path) -> list[str]:
    # insightface pulls plain onnxruntime over onnxruntime-gpu (same import
    # name → last one wins). Force the GPU build to win.
    return [uv, "pip", "install", "--python", str(py), "--reinstall", "--no-deps", "onnxruntime-gpu"]


# ---- basicsr patch (pure file op) ----

def find_basicsr_degradations(site_packages: Path) -> Path | None:
    candidate = site_packages / "basicsr" / "data" / "degradations.py"
    return candidate if candidate.is_file() else None


def apply_basicsr_patch(degradations_py: Path) -> bool:
    """basicsr 1.4.2 imports the removed torchvision.transforms.functional_tensor.
    Rewrite the import. Idempotent — returns False if already patched/absent."""
    text = degradations_py.read_text(encoding="utf-8")
    if _BASICSR_OLD not in text:
        return False
    degradations_py.write_text(
        text.replace(_BASICSR_OLD, _BASICSR_NEW), encoding="utf-8"
    )
    return True


# ---- Execution (thin) ----

def run(cmd: Sequence[str], env: dict[str, str] | None = None) -> int:
    """Run a command, streaming its output to the console. Returns the exit
    code. UV_HTTP_TIMEOUT is bumped so the multi-hundred-MB torch download
    doesn't time out on slow links."""
    import os

    merged = {**os.environ, "UV_HTTP_TIMEOUT": "300"}
    if env:
        merged.update(env)
    return subprocess.run(list(cmd), env=merged).returncode
