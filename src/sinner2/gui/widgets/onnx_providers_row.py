"""Shared ONNX execution-provider selector strip.

A ``[ ]Tensor [ ]CUDA [ ]CPU`` checkbox row for one ONNX-using processor, added
to a QFormLayout under an "ONNX Providers" label so it sits in the field column
like any other row. ORT tries the checked providers in the order shown; an ONNX
model can't run on zero providers, so unchecking everything forces CPU back on
(the floor). ``changed`` fires on any user toggle.

Used by both the live settings panel (QProcessorControls) and the batch-task
dialog. The batch dialog passes ``preferred`` so a task's requested EPs render
FIRST in their saved priority order — and a requested-but-unavailable EP (e.g. a
CUDA task edited on a CPU-only box) still renders, so editing round-trips it
instead of silently dropping it.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.pipeline.model_cache import available_onnx_providers

_TRT_PROVIDER = "TensorrtExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"
_TRT_TIP = (
    "TensorRT: compiles a GPU-specific engine for this model — typically 2–3×\n"
    "faster than plain CUDA. The FIRST run after enabling builds the engine\n"
    "(tens of seconds, one-time) and caches it to disk. Needs the TensorRT\n"
    "runtime; falls back to CUDA if it's missing."
)
_GENERIC_PROVIDER_TIP = (
    "ONNX execution provider. Multiple may be checked; ORT tries them in the\n"
    "order shown. You can't run on no provider — unchecking everything forces\n"
    "CPU back on (the floor). Applies immediately (rebuilds the session)."
)


def _short_provider_label(prov: str) -> str:
    """'CUDAExecutionProvider' → 'CUDA'; 'TensorrtExecutionProvider' → 'Tensor'."""
    name = prov.replace("ExecutionProvider", "")
    return "Tensor" if name == "Tensorrt" else name


class OnnxProvidersRow(QWidget):
    """The ``[ ]Tensor [ ]CUDA [ ]CPU`` checkbox strip for one ONNX-using
    processor. Forces CPU on when everything is unchecked (an ONNX model can't
    run on zero providers). ``changed`` fires on any user toggle.

    The strip always renders in ORT's own priority order (the order
    ``available_onnx_providers()`` reports — TensorRT first, CPU last), so it
    reads identically here and in the live panel. ``preferred`` (the batch
    dialog passes a task's saved provider list) does NOT reorder the available
    EPs; it only adds checkboxes for any requested EP the platform doesn't
    expose (e.g. a CUDA task edited on a CPU-only box), inserted at its position
    relative to the other requested EPs so the saved priority round-trips
    instead of being dropped or shoved to the end."""

    changed = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        preferred: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._tooltips: dict[str, str] = {}
        try:
            available = available_onnx_providers()
        except Exception:  # noqa: BLE001 — broken ORT install → render the defaults
            available = list(DEFAULT_ONNX_PROVIDERS)
        # Render the platform's EPs in ORT's priority order (TensorRT first,
        # CPU last) — identical to the live panel. Any requested EP the platform
        # doesn't expose is INSERTED before the first later requested EP that is
        # placed, so a saved priority like [CUDA, CPU] edited on a CPU-only box
        # round-trips as [CUDA, CPU] rather than [CPU, CUDA].
        ordered: list[str] = list(available)
        placed: set[str] = set(ordered)
        wanted = list(preferred or [])
        for idx, prov in enumerate(wanted):
            if prov in placed:
                continue
            pos = len(ordered)
            for later in wanted[idx + 1:]:
                if later in placed:
                    pos = ordered.index(later)
                    break
            ordered.insert(pos, prov)
            placed.add(prov)
        default_active = set(DEFAULT_ONNX_PROVIDERS)
        for prov in ordered:
            tip = _TRT_TIP if prov == _TRT_PROVIDER else _GENERIC_PROVIDER_TIP
            self._tooltips[prov] = tip
            cb = QCheckBox(_short_provider_label(prov))
            cb.setToolTip(tip)
            cb.setChecked(prov in default_active)
            cb.toggled.connect(self._on_toggled)
            layout.addWidget(cb)
            self._checkboxes[prov] = cb
        layout.addStretch(1)

    def _force_cpu_if_empty(self) -> None:
        if any(cb.isChecked() for cb in self._checkboxes.values()):
            return
        cpu = self._checkboxes.get(_CPU_PROVIDER)
        if cpu is not None:
            cpu.blockSignals(True)
            cpu.setChecked(True)
            cpu.blockSignals(False)

    def _on_toggled(self) -> None:
        self._force_cpu_if_empty()
        self.changed.emit()

    def selected(self) -> list[str]:
        """Checked providers in the platform's preference order. Non-empty."""
        return [p for p, cb in self._checkboxes.items() if cb.isChecked()]

    def set_selected(self, providers: list[str]) -> None:
        """Reflect a restored selection WITHOUT firing ``changed``."""
        wanted = set(providers)
        for p, cb in self._checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(p in wanted)
            cb.blockSignals(False)
        self._force_cpu_if_empty()

    def checkboxes(self) -> dict[str, QCheckBox]:
        return self._checkboxes

    def mark_failed(self, failed: set[str]) -> None:
        """Red strikethrough on providers ORT couldn't initialise; empty clears."""
        for name, cb in self._checkboxes.items():
            if name in failed:
                cb.setStyleSheet(
                    "QCheckBox { color: #d94545; text-decoration: line-through; }"
                )
                cb.setToolTip(
                    f"{name} failed to initialise — ORT fell back to a\n"
                    "lower-priority provider (its runtime libs are likely missing)."
                )
            else:
                cb.setStyleSheet("")
                cb.setToolTip(self._tooltips.get(name, ""))
