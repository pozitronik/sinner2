"""First-run model download flow (GUI).

If the required model files are missing, ask whether to download them, then
fetch them with a cancellable progress dialog. On failure or decline, point
the user at the models dir to place them by hand. Driven once at startup from
gui/__main__.py.

Threading: the download runs on a QThread. The worker's signals are delivered
to a controller QObject that lives on the GUI thread, so every widget touch
(progress bar, closing the dialog) happens on the GUI thread — touching a
widget from the worker thread deadlocks the modal event loop. Cancellation
goes the other way (GUI → worker) via a threading.Event set directly, because
the worker's own event loop is blocked inside the blocking download and
couldn't service a queued slot call.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from sinner2.pipeline import model_cache


class _DownloadWorker(QObject):
    progress = Signal(str, int, int)  # name, bytes_done, bytes_total
    finished = Signal(bool, str)  # ok, error ("" | "cancelled" | message)

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self._names = names
        self._cancel = threading.Event()

    def cancel(self) -> None:
        # Called from the GUI thread (DirectConnection); Event is thread-safe.
        self._cancel.set()

    def run(self) -> None:
        try:
            for name in self._names:
                model_cache.download_model(
                    name,
                    on_progress=(
                        lambda done, total, n=name: self.progress.emit(n, done, total)
                    ),
                    should_cancel=self._cancel.is_set,
                )
                if self._cancel.is_set():
                    self.finished.emit(False, "cancelled")
                    return
            self.finished.emit(True, "")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class _DownloadController(QObject):
    """Lives on the GUI thread; receives the worker's cross-thread signals and
    drives the dialog (which may only be touched from the GUI thread)."""

    def __init__(self, dialog: QProgressDialog) -> None:
        super().__init__()
        self._dialog = dialog
        self.ok = False
        self.error = ""

    @Slot(str, int, int)
    def on_progress(self, name: str, done: int, total: int) -> None:
        mb = 1024 * 1024
        if total > 0:
            self._dialog.setValue(int(done * 100 / total))
            self._dialog.setLabelText(
                f"Downloading {name}\n{done // mb} / {total // mb} MB"
            )
        else:
            self._dialog.setLabelText(f"Downloading {name}\n{done // mb} MB")

    @Slot(bool, str)
    def on_finished(self, ok: bool, error: str) -> None:
        self.ok = ok
        self.error = error
        self._dialog.close()  # ends the modal exec() — on the GUI thread


def ensure_models_present(parent: QWidget) -> None:
    """If model files are missing, offer to download them. Blocks (modal)
    until the download completes, is cancelled, or the user declines."""
    missing = model_cache.missing_models()
    if not missing:
        return

    models_dir = model_cache.get_models_dir()
    names = ", ".join(missing)
    answer = QMessageBox.question(
        parent,
        "Download models?",
        f"Required model file(s) are missing:\n\n    {names}\n\n"
        "Download them now from the official sources?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if answer != QMessageBox.StandardButton.Yes:
        QMessageBox.information(
            parent,
            "Models needed",
            "Processing won't work until the model files are present.\n\n"
            f"Place them in:\n{models_dir}",
        )
        return

    dialog = QProgressDialog("Preparing download…", "Cancel", 0, 100, parent)
    dialog.setWindowTitle("Downloading models")
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(0)
    # Keep the dialog up until the worker actually stops (autoClose would hide
    # it the instant Cancel is clicked, while the download is still unwinding).
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)

    thread = QThread(parent)
    worker = _DownloadWorker(missing)
    worker.moveToThread(thread)
    controller = _DownloadController(dialog)  # GUI-thread affinity

    # Worker → GUI: auto-connection is queued (cross-thread) onto the GUI loop.
    worker.progress.connect(controller.on_progress)
    worker.finished.connect(controller.on_finished)
    # GUI → worker: Direct so the cancel flag is set immediately, even though
    # the worker's event loop is blocked inside the download.
    dialog.canceled.connect(worker.cancel, Qt.ConnectionType.DirectConnection)
    dialog.canceled.connect(lambda: dialog.setLabelText("Cancelling…"))
    thread.started.connect(worker.run)
    thread.start()

    dialog.exec()
    thread.quit()
    thread.wait()
    worker.deleteLater()

    if not controller.ok and controller.error not in ("", "cancelled"):
        QMessageBox.warning(
            parent,
            "Download failed",
            f"Could not download the models:\n{controller.error}\n\n"
            f"You can place them manually in:\n{models_dir}",
        )
