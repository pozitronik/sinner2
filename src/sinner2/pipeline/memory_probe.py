"""Memory probes: live device VRAM (NVIDIA, via pynvml) + process RAM (psutil),
plus a registry of per-model load footprints measured at load time.

All optional and best-effort: no pynvml / no NVIDIA device → VRAM reads as None;
no psutil → RAM reads as None; a measurement failure never affects the wrapped
model load. Nothing here imports Qt, so it stays headless + unit-testable.

The honest split this feature rests on: device VRAM / process RAM are MEASURED
(the real counters, not estimates), and a model's footprint is the MEASURED
delta its load adds — not a prediction. Runtime memory (the ORT arena, the CUDA
context, per-worker copies) can't be predicted ahead of time, so we measure it.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

_lock = threading.Lock()
# None = not yet tried; True/False = whether NVML init succeeded (cached so we
# don't repeatedly pay a failing import on a non-NVIDIA box).
_nvml_ready: "bool | None" = None
# Guards against attributing a VRAM delta while ANOTHER load is being measured
# concurrently (the device counter is global — overlapping loads would pollute
# each other). A concurrent load is simply not measured; it still loads.
_measuring = False
# Has any GPU-touching load been measured yet — the FIRST one also pays the
# one-time CUDA context tax (cuDNN/cuBLAS), so its delta is inflated; flagged.
_any_gpu_load = False


@dataclass(frozen=True)
class ModelFootprint:
    """What loading one model ADDED, measured at load time."""

    label: str
    vram_bytes: "int | None"  # device VRAM delta, or None when no GPU was read
    ram_bytes: int            # process RSS delta (0 when psutil absent)
    first_load: bool          # this load also paid the one-time CUDA context tax


def _ensure_nvml() -> bool:
    global _nvml_ready
    if _nvml_ready is not None:
        return _nvml_ready
    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_ready = True
    except Exception:  # noqa: BLE001 — no pynvml / no driver / no device
        _nvml_ready = False
    return _nvml_ready


def device_vram(index: int = 0) -> "tuple[int, int] | None":
    """``(used, total)`` bytes for a CUDA device, or None when unavailable.

    ``used`` is the WHOLE device (every process), which is what "how close to
    OOM" actually depends on — the GPU is shared."""
    if not _ensure_nvml():
        return None
    try:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used), int(info.total)
    except Exception:  # noqa: BLE001
        return None


def process_ram() -> "int | None":
    """Resident set size of THIS process in bytes, or None when psutil absent."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


def _gb(n: int) -> float:
    return n / (1024 ** 3)


def format_memory(vram: "tuple[int, int] | None", ram: "int | None") -> str:
    """A compact one-line readout from raw probe values — e.g.
    ``"VRAM 6.2 / 24 GB · RAM 3.1 GB"``. Shows only what's available (RAM-only
    off CUDA); empty string when neither probe is available (the status cell
    then hides itself)."""
    parts = []
    if vram is not None:
        used, total = vram
        parts.append(f"VRAM {_gb(used):.1f} / {_gb(total):.0f} GB")
    if ram is not None:
        parts.append(f"RAM {_gb(ram):.1f} GB")
    return " · ".join(parts)


def model_footprints() -> "dict[str, ModelFootprint]":
    """A snapshot of every model footprint measured so far (label → footprint)."""
    with _lock:
        return dict(_footprints)


def reset_footprints() -> None:
    """Clear the registry (tests; or a fresh session)."""
    global _any_gpu_load
    with _lock:
        _footprints.clear()
        _any_gpu_load = False


_footprints: "dict[str, ModelFootprint]" = {}


@contextmanager
def measure_model_load(label: str) -> Iterator[None]:
    """Record the VRAM/RAM delta a model load adds, keyed by ``label``.

    Best-effort: a measurement error never disturbs the load. If another load is
    already being measured (the global VRAM counter would be polluted), this one
    is left UNmeasured — the load still proceeds; we just keep what numbers we
    can trust over guessing."""
    global _measuring, _any_gpu_load
    with _lock:
        concurrent = _measuring
        if not concurrent:
            _measuring = True
    if concurrent:
        yield
        return
    try:
        vram_before = device_vram()
        ram_before = process_ram()
        first = not _any_gpu_load and vram_before is not None
    except Exception:  # noqa: BLE001 — a misbehaving probe must not break loads
        vram_before, ram_before, first = None, None, False
    try:
        yield
    finally:
        try:
            vram_after = device_vram()
            ram_after = process_ram()
            vram_delta = (
                vram_after[0] - vram_before[0]
                if vram_before is not None and vram_after is not None
                else None
            )
            ram_delta = (
                ram_after - ram_before
                if ram_before is not None and ram_after is not None
                else 0
            )
            with _lock:
                if vram_before is not None:
                    _any_gpu_load = True
                _footprints[label] = ModelFootprint(
                    label=label,
                    vram_bytes=vram_delta,
                    ram_bytes=ram_delta,
                    first_load=first,
                )
        except Exception:  # noqa: BLE001 — measurement must never break a load
            pass
        finally:
            with _lock:
                _measuring = False
