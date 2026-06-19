"""Per-processor execution profiles.

Separates HOW a processor runs (hardware + concurrency) from WHAT it does
(its ``*Params`` model). The two processors live on different frameworks, so
each exposes its framework's native knobs rather than a forced common
abstraction:

  - FaceSwapper / FaceAnalyser run on ONNX Runtime  -> ``providers`` (ordered EP list)
  - FaceEnhancer (GFPGAN) runs on PyTorch           -> ``device``

``workers`` is the one shared concept: how many parallel units the BATCH stage
runner gives this processor. It is realised differently depending on the
processor's ``thread_safe`` flag — a thread-safe processor shares one instance
across N threads; a non-thread-safe one gets N independent instances. Realtime
is frame-major (a single pool), so ``workers`` is batch-only; the build params
(providers / device) apply to both modes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from sinner2.config.base import SinnerBaseModel

if TYPE_CHECKING:
    import torch


DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class ExecutionProfile(SinnerBaseModel):
    """Concurrency knob shared by all processors (consumed by batch)."""

    workers: int = 1


class OnnxExecution(ExecutionProfile):
    """Execution config for ONNX Runtime processors (FaceSwapper)."""

    providers: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ONNX_PROVIDERS)
    )


class TorchExecution(ExecutionProfile):
    """Execution config for PyTorch processors (FaceEnhancer / GFPGAN)."""

    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N"


class HybridExecution(TorchExecution):
    """Execution config for processors that switch framework by MODEL — the
    enhancer and upscaler each have a torch backend (GFPGAN / Real-ESRGAN, via
    ``device``) AND ONNX-runtime backends (CodeFormer / GPEN / RestoreFormer++ /
    GFPGAN-ONNX / HAT / fp16 exports, via ``providers``). Both knobs are carried;
    the processor uses whichever its selected model needs. Subclasses
    TorchExecution so the ``device`` field + any isinstance checks still hold;
    adds the ONNX provider list with the standard CUDA→CPU default."""

    providers: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ONNX_PROVIDERS)
    )


def resolve_torch_device(spec: str) -> "torch.device":
    """Map a device spec to a torch.device, falling back to CPU when CUDA is
    requested but unavailable (so a stale config can't crash — the enhancer
    warns when it lands on CPU)."""
    import torch

    s = (spec or "auto").strip().lower()
    if s in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s.startswith("cuda"):
        return torch.device(s if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def available_torch_devices() -> list[tuple[str, str]]:
    """``[(value, label), ...]`` for a device selector: Auto, CPU, then each
    CUDA GPU by name. Never raises — returns at least Auto + CPU."""
    devices = [
        ("auto", "Auto (CUDA if available, else CPU)"),
        ("cpu", "CPU"),
    ]
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(
                    (f"cuda:{i}", f"cuda:{i} — {torch.cuda.get_device_name(i)}")
                )
    except Exception:  # noqa: BLE001
        pass
    return devices
