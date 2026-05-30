"""First-run model download flow (GUI).

If the required model files are missing, ask whether to download them, then
fetch them with a cancellable progress dialog. On failure or decline, point
the user at the models dir to place them by hand. Driven once at startup from
gui/__main__.py.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from sinner2.pipeline import model_cache


class _DownloadWorker(QObject):
    progress = Signal(str, int, int)  # name, bytes_done, bytes_total
    finished = Signal(bool, str)  # ok, error ("" | "cancelled" | message)

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self._names = names
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            for name in self._names:
                model_cache.download_model(
                    name,
                    on_progress=(
                        lambda done, total, n=name: self.progress.emit(n, done, total)
                    ),
                    should_cancel=lambda: self._cancel,
                )
                if self._cancel:
                    self.finished.emit(False, "cancelled")
                    return
            self.finished.emit(True, "")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


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
    result = {"ok": False, "error": ""}

    def on_progress(name: str, done: int, total: int) -> None:
        mb = 1024 * 1024
        if total > 0:
            dialog.setValue(int(done * 100 / total))
            dialog.setLabelText(f"Downloading {name}\n{done // mb} / {total // mb} MB")
        else:
            dialog.setLabelText(f"Downloading {name}\n{done // mb} MB")

    def on_finished(ok: bool, error: str) -> None:
        result["ok"], result["error"] = ok, error
        dialog.close()  # ends the modal exec()

    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    dialog.canceled.connect(worker.cancel)
    dialog.canceled.connect(lambda: dialog.setLabelText("Cancelling…"))
    thread.started.connect(worker.run)
    thread.start()

    dialog.exec()
    thread.quit()
    thread.wait()
    worker.deleteLater()

    if not result["ok"] and result["error"] not in ("", "cancelled"):
        QMessageBox.warning(
            parent,
            "Download failed",
            f"Could not download the models:\n{result['error']}\n\n"
            f"You can place them manually in:\n{models_dir}",
        )
