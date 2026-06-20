"""The Faces panel: discover and map people in a multi-face target.

A subpanel of the Sources tab (revealed by its "Face map" toggle). Top: one
Scan group — detector + size + preview, age/sex, workers + stride, refine +
min score, precompute + bake angle, and Analyze / Reset — then a progress bar.
Below: the discovered people as a SORTABLE
TABLE — face thumbnail, source, age, sex, appearances, score, pose. Click a row
to jump the preview to that person's first frame and highlight their face;
Delete (or right-click) removes rows, Ctrl+M merges them. With a face row
selected, clicking a tile in the adjacent sources library assigns that source to
it. Pure view: the FaceMap is owned upstream (controller); this widget renders
it and emits intents.
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
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
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

from sinner2.gui.widgets.face_detection_overlay import FaceDetection
from sinner2.pipeline.detectors import DetectorModel
from sinner2.pipeline.face_map import FaceMap, Identity

# Scan detectors: buffalo_l (full pack → also gender/age) or a faster
# detection-only model paired with ArcFace for the clustering embedding.
_SCAN_DETECTORS: list[tuple[str, str]] = [
    (DetectorModel.BUFFALO_L.value, "buffalo_l (full pack, gender + age)"),
    (DetectorModel.YOLOFACE.value, "YOLOFace 8n (fast, detection-only)"),
    (DetectorModel.SCRFD_2_5G.value, "SCRFD 2.5g (fast, detection-only)"),
]


def _field_row(*widgets: QWidget, grow: QWidget | None = None) -> QWidget:
    """A tight horizontal strip of controls for ONE QFormLayout field column —
    lets related controls pair on a row while the form keeps the labels aligned.
    ``grow`` stretches that widget to fill; otherwise a trailing stretch
    left-aligns the row."""
    container = QWidget()
    h = QHBoxLayout(container)
    h.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        h.addWidget(w, 1 if w is grow else 0)
    if grow is None:
        h.addStretch(1)
    return container

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
    # frame index of a clicked person + a drawable FaceDetection (box in native
    # frame coords + the scanned age/sex/angle) at that frame, or None when no box
    # is catalogued for it.
    navigateRequested = Signal(int, object)
    deleteIdentitiesRequested = Signal(list)   # ids to exclude
    mergeIdentitiesRequested = Signal(list)    # ids to fold into one
    selectionChanged = Signal()                # table row selection changed
    settingsChanged = Signal()                 # a scan-settings control changed
    useFaceMapToggled = Signal(bool)           # the in-panel routing toggle
    showOverlayToggled = Signal(bool)          # show/hide the face-map preview overlay

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

        # ---- One Scan group, aligned like the settings tab: a QFormLayout puts
        # every label in a right-aligned column with the fields lined up; related
        # fields pair on a row (via _field_row), and the build options sit under a
        # divider. ----
        scan_group = QGroupBox("Face scanner")
        form = QFormLayout(scan_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        # "Use face map" routing toggle — the same switch as in the Face
        # recognition settings (kept in sync), surfaced here where you build the
        # map. Enabled once a map exists; doesn't gate the scanner controls.
        self._use_face_map_check = QCheckBox("Use face map")
        self._use_face_map_check.setEnabled(False)
        self._use_face_map_check.setToolTip(
            "Route playback through this target's face map — each person swapped "
            "with their mapped source — instead of the single global source. "
            "Enabled once a map exists; mirrors the 'Use face map' switch in the "
            "Face recognition settings. Remembered per target."
        )
        self._use_face_map_check.toggled.connect(self.useFaceMapToggled)
        self._use_face_map_check.toggled.connect(self._update_show_overlay_enabled)
        # The face-map overlay (boxes + selected-identity highlight) is managed
        # here, not by F8 — this toggles it so you can get a clean preview after
        # assigning. Only meaningful while routing is on, so grayed otherwise.
        self._show_overlay_check = QCheckBox("Show overlay")
        self._show_overlay_check.setChecked(True)
        self._show_overlay_check.setEnabled(False)
        self._show_overlay_check.setToolTip(
            "Show detected-face boxes + the selected person's highlight on the "
            "preview while the face map is in use. Turn off for a clean view "
            "after you've assigned sources."
        )
        self._show_overlay_check.toggled.connect(self.showOverlayToggled)
        form.addRow(_field_row(self._use_face_map_check, self._show_overlay_check))
        top_divider = QFrame()
        top_divider.setFrameShape(QFrame.Shape.HLine)
        top_divider.setFrameShadow(QFrame.Shadow.Sunken)
        form.addRow(top_divider)

        self._detector = QComboBox()
        for value, label in _SCAN_DETECTORS:
            self._detector.addItem(label, value)
        self._detector.setCurrentIndex(0)  # buffalo_l
        self._detector.setToolTip(
            "Face detector for the scan. buffalo_l (the full pack) also yields "
            "gender/age. yoloface / scrfd are faster detection-only models — "
            "ArcFace still adds the recognition embedding so clustering works, "
            "but they can't produce age/sex (that needs buffalo_l). EPs follow "
            "the swapper's ONNX provider selection. Downloads on first use."
        )
        form.addRow("Detector", self._detector)

        self._det_size = QSpinBox()
        # Same range/step as the live swapper's detection size (multiples of 32),
        # but a SEPARATE setting: this one is the offline scan's detector input,
        # the swapper's is for live playback.
        self._det_size.setRange(128, 1280)
        self._det_size.setSingleStep(32)
        self._det_size.setValue(640)
        self._det_size.setToolTip(
            "Detector input size for the SCAN (px, multiples of 32) — separate "
            "from the swapper's live detection size. Larger finds smaller/distant "
            "faces but scans slower."
        )
        self._workers = QSpinBox()
        self._workers.setRange(1, 16)
        self._workers.setValue(4)
        self._workers.setToolTip(
            "Parallel detection threads — more keeps the GPU busier."
        )
        form.addRow("Size", _field_row(
            self._det_size, QLabel("Workers"), self._workers
        ))

        self._stride = QSpinBox()
        self._stride.setRange(1, 300)
        self._stride.setValue(15)
        self._stride.setToolTip("Sample every Nth frame. Larger = faster scan.")
        self._preview_check = QCheckBox("Preview")
        self._preview_check.setChecked(True)
        self._preview_check.setToolTip(
            "Show the frames being scanned on the preview while analyzing."
        )
        form.addRow("Stride", _field_row(self._stride, self._preview_check))

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        form.addRow(divider)

        self._demographics_check = QCheckBox("Detect age/sex/angle")
        self._demographics_check.setToolTip(
            "Also run the gender/age model (slower) to fill the Age/Sex columns "
            "and the head-angle (roll/yaw/pitch) readout. Off = fast detection + "
            "recognition only. Needs buffalo_l."
        )
        form.addRow(self._demographics_check)

        self._refine_check = QCheckBox("Refine keypoints")
        self._refine_check.setToolTip(
            "Bake 2dfan4-refined keypoints into the per-frame map — steadier "
            "alignment on tilted faces. Adds time to the scan."
        )
        self._refine_min_label = QLabel("min")
        self._refine_score = QDoubleSpinBox()
        self._refine_score.setRange(0.0, 1.0)
        self._refine_score.setSingleStep(0.05)
        self._refine_score.setValue(0.5)
        self._refine_score.setToolTip(
            "Only with 'Refine keypoints' on: skip the 2dfan4 refinement for a "
            "face when 2dfan4's landmark confidence is below this. This is the "
            "landmark-refinement threshold — NOT the face-detection 'Score' in "
            "the list, and it never drops a face from the list."
        )
        form.addRow(_field_row(
            self._refine_check, self._refine_min_label, self._refine_score
        ))

        self._precompute_check = QCheckBox("Precompute map")
        self._precompute_check.setChecked(True)
        self._precompute_check.setToolTip(
            "After the scan, build the per-frame map so live playback + render "
            "SKIP detection (a full-frame pass — the slow part). Off = catalog "
            "only; playback detects live (per-identity routing still works)."
        )
        self._bake_angle_check = QCheckBox("Bake face angle")
        self._bake_angle_check.setChecked(True)
        self._bake_angle_check.setToolTip(
            "Bake a steady per-face tilt angle (2dfan4) into the per-frame map so "
            "rotation compensation works during detection-free playback. Without "
            "it, the Pose / Landmark-68 angle sources fall back to the noisier "
            "keypoint angle there (a rebuilt face has no pose estimate)."
        )
        form.addRow(_field_row(self._precompute_check, self._bake_angle_check))

        self._batch_recognition_check = QCheckBox("Batch recognition")
        self._batch_recognition_check.setChecked(True)
        self._batch_recognition_check.setToolTip(
            "Recognise faces in cross-frame batches (one ArcFace call per ~32 "
            "faces instead of per face) — a faster scan, identical catalog. "
            "Turn off only to isolate a recognition issue."
        )
        form.addRow(self._batch_recognition_check)

        self._analyze_btn = QPushButton("Scan for faces")
        self._analyze_btn.setToolTip("Scan the target to discover the people in it.")
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip(
            "Clear the catalog + scan progress so the next Analyze starts fresh."
        )
        self._reset_btn.clicked.connect(self.resetRequested)
        form.addRow(_field_row(
            self._analyze_btn, self._reset_btn, grow=self._analyze_btn
        ))
        layout.addWidget(scan_group)

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
        self._batch_recognition_check.toggled.connect(self._on_settings_changed)
        self._detector.currentIndexChanged.connect(self._on_settings_changed)
        # "min score" only matters when refinement is on — gray it with the
        # checkbox so the dependency is visible. "Detect age/sex/angle" needs buffalo_l,
        # so it grays for the detection-only detectors.
        self._refine_check.toggled.connect(self._update_refine_rows)
        self._detector.currentIndexChanged.connect(self._update_detector_dependent)
        self._update_refine_rows()
        self._update_detector_dependent()

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
        # Polish: zebra rows (palette-driven, theme-aware), no grid lines or
        # corner button, full-row hover/focus — a cleaner findings list.
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.setCornerButtonEnabled(False)
        self._table.setFrameShape(QFrame.Shape.NoFrame)
        self._table.verticalHeader().setDefaultSectionSize(_THUMB + 8)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        # User-resizable columns: Interactive + sensible initial widths.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.setHighlightSections(False)  # don't bold the sorted section oddly
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        _hfont = header.font()
        _hfont.setBold(True)
        header.setFont(_hfont)
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

    def batch_recognition(self) -> bool:
        return self._batch_recognition_check.isChecked()

    def detector(self) -> DetectorModel:
        return DetectorModel(self._detector.currentData())

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

    def _update_refine_rows(self) -> None:
        """Gray 'Refine min score' (label + box) unless 'Refine keypoints' is on —
        it does nothing otherwise, so the dependency is visible."""
        on = self._refine_check.isChecked()
        self._refine_min_label.setEnabled(on)
        self._refine_score.setEnabled(on)

    def _update_detector_dependent(self) -> None:
        """Demographics (age/sex/angle) need buffalo_l's genderage pack — gray +
        clear 'Detect age/sex/angle' for the detection-only detectors."""
        full = self._detector.currentData() == DetectorModel.BUFFALO_L.value
        self._demographics_check.setEnabled(full)
        if not full and self._demographics_check.isChecked():
            self._demographics_check.setChecked(False)

    def restore_settings(
        self,
        *,
        stride: int | None = None,
        workers: int | None = None,
        preview: bool | None = None,
        demographics: bool | None = None,
        precompute: bool | None = None,
        detector: str | None = None,
        detection_size: int | None = None,
        landmark_refine: bool | None = None,
        landmark_min_score: float | None = None,
        bake_angle: bool | None = None,
        batch_recognition: bool | None = None,
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
            if detector is not None:
                idx = self._detector.findData(detector)
                if idx >= 0:
                    self._detector.setCurrentIndex(idx)
            if detection_size is not None:
                self._det_size.setValue(int(detection_size))
            if landmark_refine is not None:
                self._refine_check.setChecked(bool(landmark_refine))
            if landmark_min_score is not None:
                self._refine_score.setValue(float(landmark_min_score))
            if bake_angle is not None:
                self._bake_angle_check.setChecked(bool(bake_angle))
            if batch_recognition is not None:
                self._batch_recognition_check.setChecked(bool(batch_recognition))
        finally:
            self._restoring = False
        self._update_refine_rows()  # reflect the restored refine state
        self._update_detector_dependent()  # …and the restored detector

    def face_map(self) -> FaceMap:
        return self._face_map

    # ---- "Use face map" routing toggle (synced with the settings switch) ----

    def use_face_map(self) -> bool:
        return self._use_face_map_check.isChecked()

    def show_overlay(self) -> bool:
        return self._show_overlay_check.isChecked()

    def set_use_face_map(self, on: bool) -> None:
        """Reflect the routing state WITHOUT emitting (the owner syncs it from the
        settings switch / restore / auto-on)."""
        self._use_face_map_check.blockSignals(True)
        self._use_face_map_check.setChecked(bool(on))
        self._use_face_map_check.blockSignals(False)
        self._update_show_overlay_enabled()

    def set_face_map_available(self, available: bool) -> None:
        """Enable the 'Use face map' toggle only once a map exists for the target
        (you can't route through a map that isn't built). The scanner controls
        stay usable regardless — this gates only the toggle."""
        self._use_face_map_check.setEnabled(bool(available))

    def _update_show_overlay_enabled(self) -> None:
        """'Show overlay' only does anything while the map is in use — gray it
        when routing is off so the dependency is visible."""
        self._show_overlay_check.setEnabled(self._use_face_map_check.isChecked())

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
        source.setToolTip("Select this row, then click a source in the library "
                          "to assign it to this face.")
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
        self._analyze_btn.setText("Cancel" if on else "Scan for faces")
        # Reset is disabled mid-scan: clearing the catalog while the job runs
        # would be overwritten by its finishing `finished` apply (cancel first).
        # The findings table is disabled too — a merge/delete mid-scan would be
        # overwritten by the finishing apply. The Analyze/Cancel button stays
        # enabled (the panel itself isn't disabled during a scan, see
        # main_window._on_face_analysis_active) so Cancel is reachable.
        for w in (
            self._stride, self._workers, self._preview_check,
            self._demographics_check, self._precompute_check, self._reset_btn,
            self._detector, self._det_size, self._refine_check, self._refine_score,
            self._bake_angle_check, self._table,
        ):
            w.setEnabled(not on)
        self._progress.setVisible(on)
        if on:
            self._progress.setRange(0, 0)
        else:
            # Re-apply the inter-control dependencies the blanket re-enable undid.
            self._update_refine_rows()
            self._update_detector_dependent()

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
        if ident is None:
            return
        # Navigate to the EARLIEST occurrence (first_frame) and draw its box + the
        # scanned age/sex/angle when the scan stored a box there. Fall back to the
        # clearest occurrence (ref) for scans predating first_bbox, then to a
        # box-less jump.
        if ident.first_frame is not None and ident.first_bbox is not None:
            self.navigateRequested.emit(
                ident.first_frame, self._catalog_face(ident, ident.first_bbox)
            )
        elif ident.ref_frame is not None and ident.ref_bbox is not None:
            self.navigateRequested.emit(
                ident.ref_frame, self._catalog_face(ident, ident.ref_bbox)
            )
        elif ident.first_frame is not None:
            self.navigateRequested.emit(ident.first_frame, None)

    @staticmethod
    def _catalog_face(
        ident: Identity, bbox: tuple[float, float, float, float]
    ) -> FaceDetection:
        """A drawable detection from the catalog: the box plus the identity's
        scanned age/sex/score/angle, so the overlay shows the same readout as the
        Faces table (the pose is the clearest occurrence's, matching the table)."""
        pose: tuple[float, float, float] | None = None
        if ident.pitch is not None and ident.yaw is not None and ident.roll is not None:
            pose = (ident.pitch, ident.yaw, ident.roll)
        return FaceDetection(
            bbox=bbox, score=ident.det_score, sex=ident.sex, age=ident.age, pose=pose,
        )

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
