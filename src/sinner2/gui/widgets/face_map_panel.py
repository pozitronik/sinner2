"""The Faces side-panel tab: discover and map people in a multi-face target.

Top: the scan SETTINGS (stride, workers, preview, age/sex), then the Analyze
button + progress. Below: the discovered people as a SORTABLE TABLE — face
thumbnail, id, appearances, age, sex, assigned source. Click a row to jump the
preview to that person's first frame; multi-select rows to assign one source to
many or exclude them. Pure view: the FaceMap is owned upstream (controller);
this widget renders it and emits intent signals.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sinner2.pipeline.face_map import FaceMap, Identity

_THUMB = 56
# Columns.
_C_FACE, _C_ID, _C_APP, _C_AGE, _C_SEX, _C_SOURCE = range(6)
_HEADERS = ("Face", "ID", "Appears", "Age", "Sex", "Source")
_ID_ROLE = Qt.ItemDataRole.UserRole + 1


class _NumItem(QStandardItem):
    """A table cell that DISPLAYS text but SORTS by a number (so 'Appears' / 'Age'
    sort numerically, and an unknown age sorts as -1)."""

    def __init__(self, value: int | None, text: str | None = None) -> None:
        super().__init__("" if (value is None and text is None) else (text or str(value)))
        self._value = -1 if value is None else int(value)
        self.setEditable(False)

    def __lt__(self, other: object) -> bool:
        return self._value < getattr(other, "_value", -1)


class QFaceMapPanel(QWidget):
    analyzeRequested = Signal(int)              # stride
    cancelRequested = Signal()
    navigateRequested = Signal(int)            # frame index of a clicked person
    deleteIdentitiesRequested = Signal(list)   # ids to exclude

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._face_map = FaceMap.empty()
        self._analyzing = False
        # Hold the icon-bearing items per id so thumbnails update in place
        # regardless of the current sort order.
        self._face_items: dict[str, QStandardItem] = {}
        self._source_items: dict[str, QStandardItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ---- Settings (Analyze comes AFTER, per the redesign) ----
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("stride"))
        self._stride = QSpinBox()
        self._stride.setRange(1, 300)
        self._stride.setValue(15)
        self._stride.setToolTip("Sample every Nth frame. Larger = faster scan.")
        row1.addWidget(self._stride)
        row1.addWidget(QLabel("workers"))
        self._workers = QSpinBox()
        self._workers.setRange(1, 16)
        self._workers.setValue(4)
        self._workers.setToolTip(
            "Parallel detection threads — more keeps the GPU busier."
        )
        row1.addWidget(self._workers)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._preview_check = QCheckBox("Preview")
        self._preview_check.setChecked(True)
        self._preview_check.setToolTip(
            "Show the frames being scanned on the preview while analyzing."
        )
        row2.addWidget(self._preview_check)
        self._demographics_check = QCheckBox("Detect age/sex")
        self._demographics_check.setToolTip(
            "Also run the gender/age model (slower) to fill the Age/Sex columns. "
            "Off = fast detection + recognition only."
        )
        row2.addWidget(self._demographics_check)
        row2.addStretch(1)
        layout.addLayout(row2)

        self._analyze_btn = QPushButton("Analyze faces")
        self._analyze_btn.setToolTip("Scan the target to discover the people in it.")
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        layout.addWidget(self._analyze_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # ---- Findings table ----
        self._model = QStandardItemModel(0, len(_HEADERS), self)
        self._model.setHorizontalHeaderLabels(_HEADERS)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setIconSize(QSize(_THUMB, _THUMB))
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setDefaultSectionSize(_THUMB + 8)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_C_ID, QHeaderView.ResizeMode.Stretch)
        self._table.clicked.connect(self._on_row_clicked)
        layout.addWidget(self._table, stretch=1)

        bottom = QHBoxLayout()
        self._exclude_btn = QPushButton("Exclude selected")
        self._exclude_btn.setToolTip("Remove the selected people from the list.")
        self._exclude_btn.clicked.connect(self._on_exclude)
        bottom.addWidget(self._exclude_btn)
        self._hint = QLabel("Select a person, then click a Sources tile to assign.")
        self._hint.setStyleSheet("color: #888;")
        bottom.addWidget(self._hint, stretch=1)
        layout.addLayout(bottom)

    # ---- Settings accessors ----

    def stride(self) -> int:
        return self._stride.value()

    def workers(self) -> int:
        return self._workers.value()

    def preview_enabled(self) -> bool:
        return self._preview_check.isChecked()

    def detect_demographics(self) -> bool:
        return self._demographics_check.isChecked()

    def face_map(self) -> FaceMap:
        return self._face_map

    # ---- Selection ----

    def selected_identities(self) -> list[str]:
        ids: list[str] = []
        for index in self._table.selectionModel().selectedRows():
            ident = self._model.item(index.row(), _C_FACE)
            if ident is not None:
                ids.append(str(ident.data(_ID_ROLE)))
        return ids

    def select_identity(self, identity_id: str | None) -> None:
        """Programmatically select a single person (e.g. after clicking its
        face on the preview)."""
        self._table.clearSelection()
        if identity_id is None:
            return
        for row in range(self._model.rowCount()):
            item = self._model.item(row, _C_FACE)
            if item is not None and str(item.data(_ID_ROLE)) == identity_id:
                self._table.selectRow(row)
                return

    # ---- Catalog rendering ----

    def set_face_map(self, face_map: FaceMap) -> None:
        self._face_map = face_map
        was_sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)  # don't re-sort mid-rebuild
        self._model.removeRows(0, self._model.rowCount())
        self._face_items = {}
        self._source_items = {}
        for ident in face_map.identities:
            self._append_row(ident)
        self._table.setSortingEnabled(was_sorting)
        self._hint.setVisible(not face_map.is_empty())

    def _append_row(self, ident: Identity) -> None:
        face = QStandardItem()
        face.setEditable(False)
        face.setData(ident.id, _ID_ROLE)
        face.setText("?")
        name = ident.label or f"P-{ident.id[:6]}"
        id_item = QStandardItem(name)
        id_item.setEditable(False)
        app_item = _NumItem(ident.occurrences)
        age_item = _NumItem(ident.age, "—" if ident.age is None else f"{ident.age}")
        sex_item = QStandardItem(ident.sex or "—")
        sex_item.setEditable(False)
        source = QStandardItem()
        source.setEditable(False)
        if ident.source_path:
            source.setText(Path(ident.source_path).name[:10])
            source.setToolTip(ident.source_path)
        self._model.appendRow(
            [face, id_item, app_item, age_item, sex_item, source]
        )
        self._face_items[ident.id] = face
        self._source_items[ident.id] = source

    def set_target_thumbnail(self, identity_id: str, pixmap: QPixmap) -> None:
        item = self._face_items.get(identity_id)
        if item is not None:
            item.setText("")
            item.setIcon(QIcon(pixmap))

    def set_source_thumbnail(
        self, identity_id: str, pixmap: QPixmap | None, name: str | None
    ) -> None:
        item = self._source_items.get(identity_id)
        if item is None:
            return
        if pixmap is not None:
            item.setIcon(QIcon(pixmap))
        item.setText(name[:10] if name else "")
        item.setToolTip(name or "")

    # ---- Analysis state ----

    def set_analyzing(self, on: bool) -> None:
        self._analyzing = on
        self._analyze_btn.setText("Cancel" if on else "Analyze faces")
        for w in (
            self._stride, self._workers, self._preview_check,
            self._demographics_check,
        ):
            w.setEnabled(not on)
        self._progress.setVisible(on)
        if on:
            self._progress.setRange(0, 0)

    def set_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)

    # ---- Internals ----

    def _on_analyze_clicked(self) -> None:
        if self._analyzing:
            self.cancelRequested.emit()
        else:
            self.analyzeRequested.emit(self._stride.value())

    def _on_row_clicked(self, index) -> None:  # type: ignore[no-untyped-def]
        item = self._model.item(index.row(), _C_FACE)
        if item is None:
            return
        ident = self._face_map_identity(str(item.data(_ID_ROLE)))
        if ident is not None and ident.first_frame is not None:
            self.navigateRequested.emit(ident.first_frame)

    def _on_exclude(self) -> None:
        ids = self.selected_identities()
        if ids:
            self.deleteIdentitiesRequested.emit(ids)

    def _face_map_identity(self, identity_id: str) -> Identity | None:
        for ident in self._face_map.identities:
            if ident.id == identity_id:
                return ident
        return None
