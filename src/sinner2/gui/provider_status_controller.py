"""Provider-status view logic for the main window.

Owns: the effective-ONNX-provider status-bar cell, the failed-provider highlight
(requested-but-not-loaded EPs go red), the post-async-rebuild highlight poll
(``set_chain`` records the real providers on a worker thread, so highlighting
synchronously would compare against the previous session), and the one-time
TensorRT engine-build modal.

A QObject because it owns QTimers. Holds references to the controller +
processor/status-bar widgets; the TRT dialog is parented to the WINDOW so it's
modal to it and discoverable via ``window.findChildren``. Window-state truth
(get_actual_providers / tensorrt_engine_cached) is read through ``model_cache``
so tests can patch the module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer
from PySide6.QtWidgets import QProgressDialog, QWidget

from sinner2.pipeline import model_cache

if TYPE_CHECKING:
    from sinner2.gui.player_controller import PlayerController
    from sinner2.gui.widgets.processor_controls import QProcessorControls

_TRT_PROVIDER = "TensorrtExecutionProvider"


class ProviderStatusController(QObject):
    def __init__(
        self,
        *,
        window: QWidget,
        controller: "PlayerController",
        processors: "QProcessorControls",
        providers_panel: Any,  # status-bar value panel (has set_value)
        status_bar: Any,  # QStatusActionBar (has show_message)
    ) -> None:
        super().__init__(window)
        self._window = window
        self._controller = controller
        self._processors = processors
        self._providers_panel = providers_panel
        self._status_bar = status_bar
        self._trt_wait_active = False
        self._highlight_timer: QTimer | None = None

    def refresh_label(self) -> None:
        # Trim the trailing "ExecutionProvider" suffix so the status-bar cell
        # stays short.
        providers = self._controller.effective_onnx_providers()
        short = [p.removesuffix("ExecutionProvider") or p for p in providers]
        self._providers_panel.set_value(", ".join(short) if short else "")

    def highlight_failed(self) -> None:
        """Mark requested-but-not-loaded providers red on the widget + pop a
        transient status message. The mismatch signals a provider's runtime libs
        are missing (TensorRT EP loaded but no nvinfer, etc.). Styling is a view
        concern, so it lives here, not on the controller."""
        # Runs at every point provider TRUTH is (re)known — including async-rebuild
        # poll completions — so refresh the EP cell here too (the synchronous
        # refresh elsewhere sees the OLD session's recorded providers).
        self.refresh_label()
        requested = set(self._processors.swapper_providers())
        actual = set(self._controller.effective_onnx_providers())
        # Empty requested = user unchecked everything → defaults; nothing to flag.
        if not requested:
            self._processors.mark_providers_failed(set())
            return
        failed = requested - actual
        self._processors.mark_providers_failed(failed)
        if failed:
            short_failed = ", ".join(
                p.removesuffix("ExecutionProvider") for p in failed
            )
            short_actual = ", ".join(
                p.removesuffix("ExecutionProvider")
                for p in self._controller.effective_onnx_providers()
            )
            self._status_bar.show_message(
                f"ONNX provider(s) failed to load: {short_failed}. "
                f"ORT is using: {short_actual}",
                7000,
            )

    def schedule_highlight_refresh(self) -> None:
        """Refresh the failed-provider highlight AFTER the async chain rebuild
        records what ORT actually wired up. Poll ``get_actual_providers()`` (non-
        modal) until it changes from the pre-rebuild snapshot, a newer toggle
        supersedes this request, the session goes away, or a short timeout
        backstops a same-providers rebuild — then highlight against the truth."""
        requested = tuple(self._processors.swapper_providers())
        if not requested or self._controller.executor() is None:
            # Defaults in use, or no live session to rebuild — highlight now.
            self.highlight_failed()
            return
        before = model_cache.get_actual_providers()
        # Replace any in-flight refresh so rapid toggles don't stack timers.
        if self._highlight_timer is not None:
            self._highlight_timer.stop()
        elapsed = QElapsedTimer()
        elapsed.start()
        timer = QTimer(self)
        timer.setInterval(150)
        self._highlight_timer = timer

        def _poll() -> None:
            actual = model_cache.get_actual_providers()
            rebuilt = actual != before  # the rebuilt session recorded its EPs
            superseded = tuple(self._processors.swapper_providers()) != requested
            gone = self._controller.executor() is None
            if rebuilt or superseded or gone or elapsed.elapsed() > 8000:
                timer.stop()
                if self._highlight_timer is timer:
                    self._highlight_timer = None
                # A newer toggle owns the highlight now — don't fight it.
                if not superseded:
                    self.highlight_failed()

        timer.timeout.connect(_poll)
        timer.start()

    def wait_for_tensorrt_build(self) -> bool:
        """If TensorRT is requested but no session has loaded it yet, a (possibly
        slow, one-time) engine build is about to run. Show a modal busy dialog
        until TRT shows up in the ACTUAL recorded providers (build done) or we
        give up, then refresh the highlight. Returns True if it took over the wait
        (caller must NOT also highlight); False otherwise.

        Uses model_cache.get_actual_providers() — the truly-loaded list — NOT the
        controller's effective list, which falls back to the REQUESTED list before
        any session loads (so at launch it would wrongly skip the dialog)."""
        trt = _TRT_PROVIDER
        if trt not in self._processors.swapper_providers():
            return False
        if self._controller.executor() is None:
            return False
        if self._trt_wait_active:
            # A wait is already showing (re-entered via sessionScratchDirChanged
            # while the build runs) — don't stack a second dialog + timer.
            return True
        before = model_cache.get_actual_providers()
        if before is not None and trt in before:
            return False  # a session already built + loaded TRT this run
        if model_cache.tensorrt_engine_cached():
            return False  # engine already compiled on disk → fast load, no modal
        self._trt_wait_active = True
        dialog = QProgressDialog(
            "Compiling the TensorRT engine for the swap model.\n"
            "One-time step (about 30 seconds); cached for next time.",
            "", 0, 0, self._window,
        )
        dialog.setWindowTitle("TensorRT")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setCancelButton(None)  # the compile can't be interrupted
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.show()
        elapsed = QElapsedTimer()
        elapsed.start()
        timer = QTimer(self)
        timer.setInterval(400)

        def _poll() -> None:
            actual = model_cache.get_actual_providers()
            built = actual is not None and trt in actual
            # A DIFFERENT session recorded providers without TRT → it fell back
            # (engine failed to load) → stop waiting and let the red show truth.
            fell_back = actual is not None and actual != before and trt not in actual
            gone = (
                self._controller.executor() is None
                or trt not in self._processors.swapper_providers()
            )
            if built or fell_back or gone or elapsed.elapsed() > 75_000:
                self._trt_wait_active = False
                timer.stop()
                dialog.close()
                self.highlight_failed()

        timer.timeout.connect(_poll)
        timer.start()
        return True
