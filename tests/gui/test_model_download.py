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
            model_download, "confirm", lambda *a, **k: asked.append(1) or False
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
        monkeypatch.setattr(model_download, "confirm", lambda *a, **k: False)
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

    def test_progress_signal_carries_over_2gb_without_overflow(self, qtbot):
        # Regression: progress was Signal(str, int, int) → shiboken packs a
        # Python int into a C++ 32-bit signed int (max ~2.1 GB), so a model
        # download past 2 GB raised OverflowError on emit and stalled the
        # progress bar. The byte fields must carry full Python ints (object).
        worker = _DownloadWorker(["big_model.pth"])
        received: list = []
        worker.progress.connect(
            lambda name, done, total: received.append((name, done, total))
        )
        big = 3 * 1024 ** 3  # 3 GB — past the 2**31 C-int ceiling
        worker.progress.emit("big_model.pth", big, big)
        assert received == [("big_model.pth", big, big)]


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


class TestJoinDownloadThread:
    def test_clean_join_deletes_worker(self):
        from unittest.mock import MagicMock

        from sinner2.gui.model_download import _join_download_thread

        thread = MagicMock()
        thread.wait.return_value = True
        worker = MagicMock()
        _join_download_thread(thread, worker, timeout_ms=5000)
        thread.quit.assert_called_once()
        thread.wait.assert_called_once_with(5000)
        worker.deleteLater.assert_called_once()

    def test_overrun_defers_cleanup_to_background(self):
        # A stuck download (blocked in a chunk read up to the socket timeout)
        # must not freeze the GUI on an unbounded wait: bound it, and defer
        # cleanup to thread.finished rather than deleting a running thread now.
        from unittest.mock import MagicMock

        from sinner2.gui.model_download import _join_download_thread

        thread = MagicMock()
        thread.wait.return_value = False  # didn't finish within the budget
        worker = MagicMock()
        _join_download_thread(thread, worker, timeout_ms=5000)
        worker.deleteLater.assert_not_called()  # not deleted while still running
        assert thread.finished.connect.call_count >= 1  # background cleanup wired
