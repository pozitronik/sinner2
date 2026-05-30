"""Post-install verification — the real check that the chosen variant works.

A small probe script runs inside the INSTALLED venv (where torch/onnxruntime
live) and prints JSON facts; `interpret` turns those facts into pass/fail
checks with remediation hints. interpret + parse are pure (testable); only the
probe execution shells out. This is also the standalone troubleshooter.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Runs in the venv python; prints one JSON object. Each import is guarded so a
# missing/broken package becomes a recorded error, not a crash.
_PROBE = r"""
import json, sys
d = {"python": "%d.%d.%d" % sys.version_info[:3]}
try:
    import torch
    d["torch"] = torch.__version__
    d["torch_cuda"] = bool(torch.cuda.is_available())
    d["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e:
    d["torch_error"] = repr(e)
try:
    import onnxruntime as ort
    d["ort"] = ort.__version__
    d["ort_providers"] = list(ort.get_available_providers())
except Exception as e:
    d["ort_error"] = repr(e)
try:
    import sinner2  # noqa: F401
    d["sinner2"] = True
except Exception as e:
    d["sinner2_error"] = repr(e)
print(json.dumps(d))
"""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_probe_output(stdout: str) -> dict:
    """Last non-empty stdout line is the JSON (anything before is noise)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {}


def interpret(data: dict, variant: str) -> list[CheckResult]:
    gpu = variant in ("cuda", "cuda118")
    results: list[CheckResult] = []

    py = data.get("python", "?")
    results.append(CheckResult("Python 3.12", py.startswith("3.12"), py))

    if "torch_error" in data:
        results.append(CheckResult("PyTorch", False, data["torch_error"]))
    else:
        results.append(CheckResult("PyTorch", True, data.get("torch", "?")))
        if gpu:
            ok = bool(data.get("torch_cuda"))
            detail = (
                data.get("device")
                or "torch.cuda.is_available() is False — check the NVIDIA driver"
            )
            results.append(CheckResult("CUDA via torch", ok, detail))

    if "ort_error" in data:
        results.append(CheckResult("ONNX Runtime", False, data["ort_error"]))
    else:
        results.append(CheckResult("ONNX Runtime", True, data.get("ort", "?")))
        if gpu:
            providers = data.get("ort_providers", [])
            ok = "CUDAExecutionProvider" in providers
            detail = (
                ", ".join(providers)
                if ok
                else f"CUDAExecutionProvider missing (have: {', '.join(providers) or 'none'}) "
                "— re-run with Repair, or reinstall onnxruntime-gpu"
            )
            results.append(CheckResult("CUDA execution provider", ok, detail))

    if data.get("sinner2"):
        results.append(CheckResult("sinner2 import", True, "ok"))
    else:
        results.append(
            CheckResult("sinner2 import", False, data.get("sinner2_error", "not importable"))
        )
    return results


def all_ok(results: list[CheckResult]) -> bool:
    return all(r.ok for r in results)


def run_doctor(python_exe: Path, variant: str) -> list[CheckResult]:
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return [CheckResult("probe", False, f"couldn't run the venv python: {exc}")]
    data = parse_probe_output(proc.stdout)
    if not data:
        return [CheckResult("probe", False, proc.stderr.strip() or "no probe output")]
    return interpret(data, variant)
