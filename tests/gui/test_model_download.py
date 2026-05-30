"""Tests for the first-run model-download GUI flow."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox, QWidget

from PySide6.QtWidgets import QProgressDialog

from sinner2.gui import model_download
from sinner2.gui.model_download import (
    _DownloadController,
    _DownloadWorker,
    ensure_models_present,
)


@pytest.fixture
def parent(qtbot):
    w = QWidget()
    qtbot.addWidget(w)
    return w


class TestEnsureModelsPresent:
    def test_noop_when_nothing_missing(self, parent, monkeypatch):
        monkeypatch.setattr(model_download.model_cache, "missing_models", lambda: [])
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: asked.append(1)
        )
        ensure_models_present(parent)
        assert asked == []  # never prompted

    def test_decline_shows_hint_and_skips_download(self, parent, monkeypatch):
        monkeypatch.setattr(
            model_download.model_cache,
            "missing_models",
            lambda: ["inswapper_128.onnx"],
        )
        monkeypatch.setattr(
            model_download.model_cache, "get_models_dir", lambda: Path("/models")
        )
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *a, **k: QMessageBox.StandardButton.No,
        )
        info: list = []
        monkeypatch.setattr(
            QMessageBox, "information", lambda *a, **k: info.append(a)
        )
        downloaded: list = []
        monkeypatch.setattr(
            model_download.model_cache,
            "download_model",
            lambda *a, **k: downloaded.append(a),
        )
        ensure_models_present(parent)
        assert downloaded == []  # no download attempted
        assert info  # manual-placement hint shown


class TestDownloadWorker:
    def test_finishes_ok_when_all_download(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            model_download.model_cache, "download_model", lambda *a, **k: None
        )
        worker = _DownloadWorker(["inswapper_128.onnx", "GFPGANv1.4.pth"])
        with qtbot.waitSignal(worker.finished, timeout=1000) as blocker:
            worker.run()
        assert blocker.args == [True, ""]

    def test_propagates_error(self, qtbot, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("network down")

        monkeypatch.setattr(model_download.model_cache, "download_model", boom)
        worker = _DownloadWorker(["inswapper_128.onnx"])
        with qtbot.waitSignal(worker.finished, timeout=1000) as blocker:
            worker.run()
        assert blocker.args[0] is False
        assert "network down" in blocker.args[1]

    def test_reports_cancelled(self, qtbot, monkeypatch):
        # download_model returns without writing when cancelled; the worker
        # then reports "cancelled" rather than success.
        monkeypatch.setattr(
            model_download.model_cache, "download_model", lambda *a, **k: None
        )
        worker = _DownloadWorker(["inswapper_128.onnx"])
        worker.cancel()
        with qtbot.waitSignal(worker.finished, timeout=1000) as blocker:
            worker.run()
        assert blocker.args == [False, "cancelled"]


class TestDownloadController:
    """The controller lives on the GUI thread and is the only thing that
    touches the dialog (worker signals are delivered here, not to closures
    running on the worker thread)."""

    def test_on_progress_updates_dialog(self, qtbot):
        dialog = QProgressDialog("", "Cancel", 0, 100)
        qtbot.addWidget(dialog)
        ctrl = _DownloadController(dialog)
        mb = 1024 * 1024
        ctrl.on_progress("model.bin", 50 * mb, 100 * mb)
        assert dialog.value() == 50
        assert "model.bin" in dialog.labelText()
        assert "50 / 100 MB" in dialog.labelText()

    def test_on_finished_records_result(self, qtbot):
        dialog = QProgressDialog("", "Cancel", 0, 100)
        qtbot.addWidget(dialog)
        ctrl = _DownloadController(dialog)
        ctrl.on_finished(False, "boom")
        assert ctrl.ok is False
        assert ctrl.error == "boom"
