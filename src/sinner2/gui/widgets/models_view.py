"""Models management tab — see every downloadable model, its install status +
size, and download / delete / repair it.

Layout: a header (models dir, installed count + disk usage, bulk actions) over a
table grouped by category. Downloads run in the background (non-modal) one at a
time through a queue; the row's Status cell shows live progress. The actual
fetch is `model_cache.download_model` (cancellable, byte-progress) wrapped in a
QThread worker — the same engine the first-run flow uses.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sinner2.gui.confirm import confirm
from sinner2.pipeline import model_cache
from sinner2.pipeline.models_catalog import (
    ModelCategory,
    catalog_entries,
    model_info,
)

_COL_MODEL, _COL_CATEGORY, _COL_STATUS, _COL_SIZE, _COL_MEMORY = range(5)
_HEADERS = ["Model", "Category", "Status", "Size", "Memory"]
_ROLE_FILE = Qt.ItemDataRole.UserRole
_ROLE_SORT = Qt.ItemDataRole.UserRole + 1  # comparable value driving header sort
_MB = 1024 * 1024
_GB = 1024 ** 3
_MEMORY_TOOLTIP = (
    "VRAM this model added when it loaded — measured live on THIS machine "
    "(fills in as you use models), not a prediction. Torch models (GFPGAN, "
    "parsers) load once PER WORKER, so multiply by the worker count. '*' marks "
    "the first GPU model loaded — its number also includes the one-time CUDA "
    "context (cuDNN/cuBLAS). Needs nvidia-ml-py."
)


def _fmt_mb(num_bytes: int) -> str:
    mb = num_bytes / _MB
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _fmt_footprint(fp) -> "tuple[str, int, str]":
    """(text, sort-bytes, tooltip) for a measured model footprint, or
    ("", 0, "") when nothing measurable was recorded."""
    if fp.vram_bytes is not None and fp.vram_bytes > 0:
        text = f"+{fp.vram_bytes / _GB:.2f} GB"
        sort_bytes = fp.vram_bytes
        tip = "Measured VRAM this model added when it loaded."
    elif fp.ram_bytes > 0:
        text = f"+{fp.ram_bytes / _GB:.2f} GB RAM"
        sort_bytes = fp.ram_bytes
        tip = "Measured RAM this model added (no GPU was measured)."
    else:
        return "", 0, ""
    if fp.first_load:
        text += " *"
        tip += " * includes the one-time CUDA context (cuDNN/cuBLAS)."
    return text, sort_bytes, tip


class _SortItem(QStandardItem):
    """A cell that sorts by its _ROLE_SORT value (numbers numerically, status by
    rank, size by bytes) rather than the displayed text."""

    def __lt__(self, other: QStandardItem) -> bool:
        a, b = self.data(_ROLE_SORT), other.data(_ROLE_SORT)
        if a is None or b is None:
            return super().__lt__(other)
        return bool(a < b)


class _DownloadWorker(QObject):
    """Downloads ONE model on its own thread (cancellable, byte-progress)."""

    progress = Signal(int, int)   # done, total
    finished = Signal(bool, str)  # ok, error ("" | "cancelled" | message)

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            model_cache.download_model(
                self._name,
                on_progress=lambda d, t: self.progress.emit(d, t),
                should_cancel=self._cancel.is_set,
            )
        except Exception as exc:  # network / disk / unknown-model
            self.finished.emit(False, str(exc))
            return
        if self._cancel.is_set():
            self.finished.emit(False, "cancelled")
        else:
            self.finished.emit(True, "")


class QModelsView(QWidget):
    """The Models tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ---- header ----
        self._summary = QLabel("")
        self._summary.setStyleSheet("color: gray;")
        self._download_missing_btn = QPushButton("Download all missing")
        self._download_missing_btn.clicked.connect(self._on_download_all_missing)
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(self._open_models_folder)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        header = QHBoxLayout()
        header.addWidget(self._summary, stretch=1)
        header.addWidget(self._download_missing_btn)
        header.addWidget(open_btn)
        header.addWidget(refresh_btn)

        # ---- table ----
        self._model = QStandardItemModel(0, len(_HEADERS), self)
        self._model.setHorizontalHeaderLabels(_HEADERS)
        mem_header = self._model.horizontalHeaderItem(_COL_MEMORY)
        if mem_header is not None:
            mem_header.setToolTip(_MEMORY_TOOLTIP)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.doubleClicked.connect(self._on_double_click)
        # User-resizable columns + click-to-sort headers.
        self._table.setSortingEnabled(True)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self._table, stretch=1)

        # ---- download queue ----
        self._queue: list[str] = []
        self._current: str | None = None
        self._thread: QThread | None = None
        self._worker: _DownloadWorker | None = None

        self._populate()

        # Per-model footprints fill in as models load (in the background, while
        # the tab is open) — poll the registry so the Memory column updates live.
        self._mem_timer = QTimer(self)
        self._mem_timer.setInterval(2000)
        self._mem_timer.timeout.connect(self._refresh_memory_cells)
        self._mem_timer.start()

    # ---- population / refresh ----

    def _populate(self) -> None:
        # Sorting off while appending so each new row stays at the end (where
        # _refresh_row expects it); re-enabled + sorted once the table is built.
        self._table.setSortingEnabled(False)
        self._model.removeRows(0, self._model.rowCount())
        cats = list(ModelCategory)
        for info in catalog_entries():
            name_item = _SortItem(info.display_name)
            name_item.setData(info.filename, _ROLE_FILE)
            name_item.setData(info.display_name.lower(), _ROLE_SORT)
            tip = info.description + (f"\nLicense: {info.license}" if info.license else "")
            tip += "\n\nDouble-click for details."
            name_item.setToolTip(tip)
            cat_item = _SortItem(info.category.value)
            # Sort the Category column category-first, name-second (keeps groups
            # together AND ordered within a group).
            cat_item.setData(
                f"{cats.index(info.category):02d}:{info.display_name.lower()}",
                _ROLE_SORT,
            )
            row = [
                name_item, cat_item, _SortItem(""), _SortItem(""), _SortItem("")
            ]
            self._model.appendRow(row)
            self._refresh_row(self._model.rowCount() - 1)
        self._refresh_memory_cells()
        # Group by category initially; user can re-sort any column.
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(_COL_CATEGORY, Qt.SortOrder.AscendingOrder)
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(
            _COL_MODEL, max(self._table.columnWidth(_COL_MODEL), 200)
        )
        self._refresh_summary()

    def refresh(self) -> None:
        """Re-scan presence/sizes from disk (e.g. after a manual file drop)."""
        for row in range(self._model.rowCount()):
            if self._row_file(row) != self._current:
                self._refresh_row(row)
        self._refresh_summary()

    def _refresh_row(self, row: int) -> None:
        name = self._row_file(row)
        present = model_cache.model_present(name)
        required = name in model_cache.REQUIRED_MODELS
        info = model_info(name)
        if present:
            size = model_cache.model_size_on_disk(name)
            self._set_status(row, "✓ Installed", 0)
            self._set_size(row, _fmt_mb(size), size)
        else:
            self._set_status(
                row,
                "Required — missing" if required else "Not installed",
                4 if required else 3,
            )
            approx = info.size_mb * _MB if info else 0
            self._set_size(row, f"~{info.size_mb} MB" if info else "", approx)

    def _set_status(self, row: int, text: str, rank: int) -> None:
        item = self._model.item(row, _COL_STATUS)
        item.setText(text)
        item.setData(rank, _ROLE_SORT)
        item.setToolTip("")

    def _set_size(self, row: int, text: str, sort_bytes: int) -> None:
        item = self._model.item(row, _COL_SIZE)
        item.setText(text)
        item.setData(sort_bytes, _ROLE_SORT)

    def _refresh_memory_cells(self) -> None:
        """Fill the Memory column from the live per-model footprint registry —
        cells appear as each model is loaded (measured), keyed by filename."""
        from sinner2.pipeline.memory_probe import model_footprints

        footprints = model_footprints()
        for row in range(self._model.rowCount()):
            item = self._model.item(row, _COL_MEMORY)
            if item is None:
                continue
            fp = footprints.get(self._row_file(row))
            text, sort_bytes, tip = (
                _fmt_footprint(fp) if fp is not None else ("", 0, "")
            )
            item.setText(text)
            item.setData(sort_bytes, _ROLE_SORT)
            item.setToolTip(tip)

    def _refresh_summary(self) -> None:
        files = [self._row_file(r) for r in range(self._model.rowCount())]
        installed = [f for f in files if model_cache.model_present(f)]
        used = sum(model_cache.model_size_on_disk(f) for f in installed)
        self._summary.setText(
            f"Installed {len(installed)}/{len(files)}  ·  {_fmt_mb(used)} on disk"
            f"  ·  {model_cache.get_models_dir()}"
        )
        self._download_missing_btn.setEnabled(len(installed) < len(files))

    # ---- helpers ----

    def _row_file(self, row: int) -> str:
        return self._model.item(row, _COL_MODEL).data(_ROLE_FILE)

    def _row_for(self, name: str) -> int:
        for row in range(self._model.rowCount()):
            if self._row_file(row) == name:
                return row
        return -1

    # ---- context menu ----

    def _on_context_menu(self, pos) -> None:
        index = self._table.indexAt(pos)
        if not index.isValid():
            return
        name = self._row_file(index.row())
        present = model_cache.model_present(name)
        downloading = name == self._current or name in self._queue
        menu = QMenu(self)
        if downloading:
            menu.addAction("Cancel download", lambda: self._cancel(name))
        elif present:
            menu.addAction("Re-download (repair)", lambda: self._redownload(name))
            menu.addAction("Delete", lambda: self._delete(name))
        else:
            menu.addAction("Download", lambda: self._enqueue([name]))
        menu.addSeparator()
        menu.addAction("Details…", lambda: self._show_details(name))
        menu.addAction("Reveal in folder", self._open_models_folder)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _on_double_click(self, index) -> None:
        if index.isValid():
            self._show_details(self._row_file(index.row()))

    def _details_html(self, name: str) -> str:
        info = model_info(name)
        if info is None:
            return ""
        url = model_cache.MODEL_SOURCES.get(name, "")
        if model_cache.model_present(name):
            size = model_cache.model_size_on_disk(name)
            status = f"Installed ({_fmt_mb(size)})"
            location = str(model_cache.get_models_dir() / name)
        else:
            status = f"Not installed (~{info.size_mb} MB download)"
            location = ""
        rows = [
            ("Category", info.category.value),
            ("Description", info.description),
            ("License", info.license or "—"),
            ("Status", status),
        ]
        if location:
            rows.append(("Location", location))
        rows.append(("File", name))
        rows.append(("Source", f'<a href="{url}">{url}</a>'))
        return "<br>".join(f"<b>{k}:</b> {v}" for k, v in rows)

    def _show_details(self, name: str) -> None:
        info = model_info(name)
        if info is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(info.display_name)
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)
        label = QLabel(self._details_html(name))
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    # ---- actions ----

    def _open_models_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(model_cache.get_models_dir())))

    def _on_download_all_missing(self) -> None:
        missing = [
            self._row_file(r)
            for r in range(self._model.rowCount())
            if not model_cache.model_present(self._row_file(r))
            and self._row_file(r) not in self._queue
            and self._row_file(r) != self._current
        ]
        if not missing:
            return
        total_mb = sum((model_info(m).size_mb if model_info(m) else 0) for m in missing)
        if confirm(
            self, "download_all_missing", "Download all missing",
            f"Download {len(missing)} model(s), about {total_mb} MB?",
        ):
            self._enqueue(missing)

    def _delete(self, name: str) -> None:
        required = name in model_cache.REQUIRED_MODELS
        msg = f"Delete {name}?"
        if required:
            msg += "\n\nThis model is REQUIRED — the app won't work without it."
        # A REQUIRED model can never be suppressed/auto-confirmed — silently
        # auto-deleting it would brick the app.
        if confirm(
            self, "delete_model", "Delete model", msg, suppressible=not required
        ):
            model_cache.delete_model(name)
            self._refresh_row(self._row_for(name))
            self._refresh_summary()

    def _redownload(self, name: str) -> None:
        model_cache.delete_model(name)
        self._enqueue([name])

    # ---- download queue ----

    def _enqueue(self, names: list[str]) -> None:
        for name in names:
            if name != self._current and name not in self._queue:
                self._queue.append(name)
                row = self._row_for(name)
                if row >= 0:
                    self._set_status(row, "Queued…", 2)
        self._pump()

    def _pump(self) -> None:
        if self._current is not None or not self._queue:
            return
        name = self._queue.pop(0)
        self._current = name
        self._worker = _DownloadWorker(name)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        row = self._row_for(name)
        if row >= 0:
            self._set_status(row, "Downloading…", 1)
        self._thread.start()

    def _cancel(self, name: str) -> None:
        if name in self._queue:
            self._queue.remove(name)
            self._refresh_row(self._row_for(name))
        elif name == self._current and self._worker is not None:
            self._worker.cancel()

    def _on_progress(self, done: int, total: int) -> None:
        if self._current is None:
            return
        row = self._row_for(self._current)
        if row < 0:
            return
        pct = f"{done * 100 // total}%" if total else "…"
        self._set_status(row, f"Downloading {pct}", 1)

    def _on_finished(self, ok: bool, error: str) -> None:
        name = self._current
        self._teardown_thread()
        self._current = None
        if name is not None:
            row = self._row_for(name)
            if row >= 0:
                if ok or error == "cancelled":
                    self._refresh_row(row)
                else:
                    self._set_status(row, "Failed", 5)
                    self._model.item(row, _COL_STATUS).setToolTip(error)
        self._refresh_summary()
        self._pump()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None

    def shutdown(self) -> None:
        """Cancel any in-flight download and join the thread — call before the
        app quits so a long fetch doesn't outlive the GUI."""
        self._mem_timer.stop()
        self._queue.clear()
        if self._worker is not None:
            self._worker.cancel()
        self._teardown_thread()
        self._current = None
