"""The Faces panel: discover and map people in a multi-face target.

A subpanel of the Sources tab (revealed by its "Faces" toggle). Top: the scan
SETTINGS (stride, workers, preview, age/sex), then Analyze + progress. Below:
the discovered people as a SORTABLE TABLE — face thumbnail, id, appearances,
age, sex, assigned source. Click a row to jump the preview to that person's
first frame; ✕ removes a row. With a face row selected, clicking a tile in the
adjacent sources library assigns that source to it. Pure view: the FaceMap is
owned upstream (controller); this widget renders it and emits intents.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import (
    QIcon,
    QKeySequence,
    QPixmap,
    QShortcut,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sinner2.pipeline.face_map import FaceMap, Identity

_THUMB = 56
# Columns: Face · Source · Age · Sex · Appears · Score · Roll · Yaw · Pitch.
# The id isn't a column — it rides on the Face item's data (selection key); rows
# are deleted with the Delete key (no per-row ✕).
(_C_FACE, _C_SOURCE, _C_AGE, _C_SEX, _C_APP,
 _C_SCORE, _C_ROLL, _C_YAW, _C_PITCH) = range(9)
_HEADERS = ("Face", "Source", "Age", "Sex", "Appears",
            "Score", "Roll", "Yaw", "Pitch")
# Initial column widths (user-resizable from here).
_WIDTHS = {_C_FACE: 66, _C_SOURCE: 120, _C_AGE: 44, _C_SEX: 44, _C_APP: 60,
           _C_SCORE: 56, _C_ROLL: 52, _C_YAW: 52, _C_PITCH: 52}
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


class _FloatItem(QStandardItem):
    """A cell that DISPLAYS formatted text but SORTS by a float (score / pose).
    None → "—", sorting below every real value."""

    def __init__(self, value: float | None, fmt: str = "{:.2f}") -> None:
        super().__init__("—" if value is None else fmt.format(value))
        self._value = float("-inf") if value is None else float(value)
        self.setEditable(False)
        self.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

    def __lt__(self, other: object) -> bool:
        return self._value < getattr(other, "_value", float("-inf"))


class QFaceMapPanel(QWidget):
    analyzeRequested = Signal(int)              # stride
    cancelRequested = Signal()
    resetRequested = Signal()                  # clear catalog + scan progress
    navigateRequested = Signal(int)            # frame index of a clicked person
    deleteIdentitiesRequested = Signal(list)   # ids to exclude
    mergeIdentitiesRequested = Signal(list)    # ids to fold into one
    selectionChanged = Signal()                # table row selection changed
    settingsChanged = Signal()                 # a scan-settings control changed

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._face_map = FaceMap.empty()
        self._analyzing = False
        # True only while restore_settings() programmatically sets the controls,
        # so seeding persisted values on startup doesn't echo back as a "user
        # changed a setting" persist (and doesn't fire before the window wires up).
        self._restoring = False
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
        self._precompute_check = QCheckBox("Precompute")
        self._precompute_check.setChecked(True)
        self._precompute_check.setToolTip(
            "After the scan, build the per-frame map so live playback + render "
            "SKIP detection (a full-frame pass — the slow part). Off = catalog "
            "only; playback detects live (per-identity routing still works)."
        )
        row2.addWidget(self._precompute_check)
        row2.addStretch(1)
        layout.addLayout(row2)

        # ---- Detection (the scan's OWN settings, independent of the live
        # swapper). The detector is fixed to buffalo_l: identity matching needs
        # ArcFace embeddings, which the faster detection-only detectors can't
        # produce — so this is shown, not chosen. ----
        det_group = QGroupBox("Detection")
        det_box = QVBoxLayout(det_group)
        det_row1 = QHBoxLayout()
        det_label = QLabel("Detector: buffalo_l")
        det_label.setToolTip(
            "Analysis always uses buffalo_l — identity matching needs ArcFace "
            "embeddings, which the detection-only detectors don't produce. "
            "Execution providers come from the global execution settings."
        )
        det_row1.addWidget(det_label)
        det_row1.addWidget(QLabel("size"))
        self._det_size = QSpinBox()
        self._det_size.setRange(64, 2048)
        self._det_size.setSingleStep(32)
        self._det_size.setValue(640)
        self._det_size.setToolTip(
            "Detector input size (px, aligned to a multiple of 32). Larger finds "
            "smaller/distant faces but scans slower."
        )
        det_row1.addWidget(self._det_size)
        det_row1.addStretch(1)
        det_box.addLayout(det_row1)
        det_row2 = QHBoxLayout()
        self._refine_check = QCheckBox("Refine keypoints (2dfan4)")
        self._refine_check.setToolTip(
            "Bake 2dfan4-refined keypoints into the per-frame map — steadier "
            "alignment on tilted faces. Adds time to the scan."
        )
        det_row2.addWidget(self._refine_check)
        det_row2.addWidget(QLabel("min score"))
        self._refine_score = QDoubleSpinBox()
        self._refine_score.setRange(0.0, 1.0)
        self._refine_score.setSingleStep(0.05)
        self._refine_score.setValue(0.5)
        self._refine_score.setToolTip(
            "Skip refinement when 2dfan4's confidence is below this."
        )
        det_row2.addWidget(self._refine_score)
        det_row2.addStretch(1)
        det_box.addLayout(det_row2)
        det_row3 = QHBoxLayout()
        self._bake_angle_check = QCheckBox("Bake face angle (2dfan4)")
        self._bake_angle_check.setChecked(True)
        self._bake_angle_check.setToolTip(
            "Bake a steady per-face tilt angle into the per-frame map so rotation "
            "compensation works during detection-free playback. Without it, the "
            "Pose / Landmark-68 angle sources fall back to the noisier keypoint "
            "angle there (a rebuilt face has no pose estimate)."
        )
        det_row3.addWidget(self._bake_angle_check)
        det_row3.addStretch(1)
        det_box.addLayout(det_row3)
        layout.addWidget(det_group)

        # Persist scan settings across restarts: any change emits settingsChanged
        # (the owner saves them); restore_settings() seeds them on startup.
        self._stride.valueChanged.connect(self._on_settings_changed)
        self._workers.valueChanged.connect(self._on_settings_changed)
        self._preview_check.toggled.connect(self._on_settings_changed)
        self._demographics_check.toggled.connect(self._on_settings_changed)
        self._precompute_check.toggled.connect(self._on_settings_changed)
        self._det_size.valueChanged.connect(self._on_settings_changed)
        self._refine_check.toggled.connect(self._on_settings_changed)
        self._refine_score.valueChanged.connect(self._on_settings_changed)
        self._bake_angle_check.toggled.connect(self._on_settings_changed)

        analyze_row = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze faces")
        self._analyze_btn.setToolTip("Scan the target to discover the people in it.")
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        analyze_row.addWidget(self._analyze_btn, stretch=1)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip(
            "Clear the catalog + scan progress so the next Analyze starts fresh."
        )
        self._reset_btn.clicked.connect(self.resetRequested)
        analyze_row.addWidget(self._reset_btn)
        layout.addLayout(analyze_row)

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
        # User-resizable columns: Interactive + sensible initial widths.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        for col, width in _WIDTHS.items():
            self._table.setColumnWidth(col, width)
        # A single click both selects the row (→ selectionChanged → highlight the
        # person's geometry box) AND jumps the preview to their first appearance.
        self._table.clicked.connect(self._on_cell_clicked)
        self._table.selectionModel().selectionChanged.connect(
            lambda *_: self.selectionChanged.emit()
        )
        # Delete key removes the selected people (replaces the per-row ✕); Ctrl+M
        # merges the selection. Both scoped to the table (fire only when focused).
        self._delete_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete), self._table
        )
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._delete_shortcut.activated.connect(self._on_delete_shortcut)
        self._merge_shortcut = QShortcut(QKeySequence("Ctrl+M"), self._table)
        self._merge_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._merge_shortcut.activated.connect(self._on_merge_shortcut)
        # Right-click → Merge / Delete on the selection.
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._table, stretch=1)

    # ---- Settings accessors ----

    def stride(self) -> int:
        return self._stride.value()

    def workers(self) -> int:
        return self._workers.value()

    def preview_enabled(self) -> bool:
        return self._preview_check.isChecked()

    def detect_demographics(self) -> bool:
        return self._demographics_check.isChecked()

    def precompute_geometry(self) -> bool:
        return self._precompute_check.isChecked()

    def detection_size(self) -> int:
        return self._det_size.value()

    def landmark_refine(self) -> bool:
        return self._refine_check.isChecked()

    def landmark_min_score(self) -> float:
        return self._refine_score.value()

    def bake_angle(self) -> bool:
        return self._bake_angle_check.isChecked()

    def _on_settings_changed(self, *_: object) -> None:
        if not self._restoring:
            self.settingsChanged.emit()

    def restore_settings(
        self,
        *,
        stride: int | None = None,
        workers: int | None = None,
        preview: bool | None = None,
        demographics: bool | None = None,
        precompute: bool | None = None,
        detection_size: int | None = None,
        landmark_refine: bool | None = None,
        landmark_min_score: float | None = None,
        bake_angle: bool | None = None,
    ) -> None:
        """Seed the scan-settings controls from persisted values (None = keep the
        default). Runs silently — no settingsChanged echo during restore."""
        self._restoring = True
        try:
            if stride is not None:
                self._stride.setValue(int(stride))
            if workers is not None:
                self._workers.setValue(int(workers))
            if preview is not None:
                self._preview_check.setChecked(bool(preview))
            if demographics is not None:
                self._demographics_check.setChecked(bool(demographics))
            if precompute is not None:
                self._precompute_check.setChecked(bool(precompute))
            if detection_size is not None:
                self._det_size.setValue(int(detection_size))
            if landmark_refine is not None:
                self._refine_check.setChecked(bool(landmark_refine))
            if landmark_min_score is not None:
                self._refine_score.setValue(float(landmark_min_score))
            if bake_angle is not None:
                self._bake_angle_check.setChecked(bool(bake_angle))
        finally:
            self._restoring = False

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

    def note_face_map(self, face_map: FaceMap) -> None:
        """Record a new FaceMap WITHOUT rebuilding the table — for in-place edits
        (a source assignment) that change existing cells, not the row set."""
        self._face_map = face_map

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

    def _append_row(self, ident: Identity) -> None:
        face = QStandardItem()
        face.setEditable(False)
        face.setData(ident.id, _ID_ROLE)
        face.setText("?")
        face.setToolTip(ident.label or f"P-{ident.id[:6]}")
        source = QStandardItem()
        source.setEditable(False)
        source.setToolTip("Double-click to pick a source for this face.")
        if ident.source_path:
            # Thumbnail only — no filename text (set via set_source_thumbnail).
            source.setToolTip(ident.source_path)
        age_item = _NumItem(ident.age, "—" if ident.age is None else f"{ident.age}")
        sex_item = QStandardItem(ident.sex or "—")
        sex_item.setEditable(False)
        app_item = _NumItem(ident.occurrences)
        # Score (det confidence, always present) + pose (degrees; only the full
        # buffalo_l pack fills them — "—" in fast det+rec mode).
        score_item = _FloatItem(ident.det_score, "{:.2f}")
        roll_item = _FloatItem(ident.roll, "{:.0f}°")
        yaw_item = _FloatItem(ident.yaw, "{:.0f}°")
        pitch_item = _FloatItem(ident.pitch, "{:.0f}°")
        self._model.appendRow(
            [face, source, age_item, sex_item, app_item,
             score_item, roll_item, yaw_item, pitch_item]
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
        # Thumbnail only — the filename lives in the tooltip (item 3).
        item.setToolTip(name or "")

    # ---- Analysis state ----

    def set_analyzing(self, on: bool) -> None:
        self._analyzing = on
        self._analyze_btn.setText("Cancel" if on else "Analyze faces")
        # Reset is disabled mid-scan: clearing the catalog while the job runs
        # would be overwritten by its finishing `finished` apply (cancel first).
        for w in (
            self._stride, self._workers, self._preview_check,
            self._demographics_check, self._precompute_check, self._reset_btn,
            self._det_size, self._refine_check, self._refine_score,
            self._bake_angle_check,
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

    def _row_id(self, row: int) -> str | None:
        item = self._model.item(row, _C_FACE)
        return str(item.data(_ID_ROLE)) if item is not None else None

    def _on_cell_clicked(self, index) -> None:  # type: ignore[no-untyped-def]
        ident_id = self._row_id(index.row())
        if ident_id is None:
            return
        ident = self._face_map_identity(ident_id)
        if ident is not None and ident.first_frame is not None:
            self.navigateRequested.emit(ident.first_frame)

    def _on_delete_shortcut(self) -> None:
        """Delete key (while the table has focus) removes the selected people."""
        ids = self.selected_identities()
        if ids:
            self.deleteIdentitiesRequested.emit(ids)

    def _on_merge_shortcut(self) -> None:
        """Ctrl+M folds the selected people into one (needs 2+ selected)."""
        ids = self.selected_identities()
        if len(ids) >= 2:
            self.mergeIdentitiesRequested.emit(ids)

    def _on_context_menu(self, pos) -> None:  # type: ignore[no-untyped-def]
        """Right-click → Merge (2+ selected) / Delete (1+ selected)."""
        ids = self.selected_identities()
        if not ids:
            return
        menu = QMenu(self._table)
        merge = menu.addAction("Merge\tCtrl+M")
        merge.setEnabled(len(ids) >= 2)
        merge.triggered.connect(lambda: self.mergeIdentitiesRequested.emit(ids))
        delete = menu.addAction("Delete\tDel")
        delete.triggered.connect(lambda: self.deleteIdentitiesRequested.emit(ids))
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _face_map_identity(self, identity_id: str) -> Identity | None:
        for ident in self._face_map.identities:
            if ident.id == identity_id:
                return ident
        return None
