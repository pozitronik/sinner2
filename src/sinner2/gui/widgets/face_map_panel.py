"""The Faces side-panel tab: discover and map people in a multi-face target.

Top: an Analyze control (strided catalog scan) + progress. Below: one card per
discovered identity — a representative thumbnail, occurrence count, the assigned
source (thumbnail), and a delete button. Selecting a card and clicking a Sources
tile assigns that source (wired in the main window). Pure view: the FaceMap is
owned upstream (controller); this widget renders it and emits intent signals.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sinner2.pipeline.face_map import FaceMap, Identity

_THUMB = 64
_SOURCE_THUMB = 44


class _IdentityCard(QFrame):
    """One discovered identity: [target thumb · name/count · source thumb · ✕]."""

    selected = Signal(str)         # identity id
    deleteRequested = Signal(str)  # identity id

    def __init__(self, identity: Identity, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._id = identity.id
        self._is_selected = False
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self._thumb = QLabel("?")
        self._thumb.setFixedSize(_THUMB, _THUMB)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("border: 1px solid #555;")
        layout.addWidget(self._thumb)

        info = QVBoxLayout()
        info.setSpacing(1)
        name = identity.label or f"Person {identity.id[:4]}"
        self._name = QLabel(name)
        self._name.setStyleSheet("font-weight: bold;")
        info.addWidget(self._name)
        self._count = QLabel(f"{identity.occurrences} appearance(s)")
        self._count.setStyleSheet("color: #999;")
        info.addWidget(self._count)
        layout.addLayout(info, stretch=1)

        self._source = QLabel("no source")
        self._source.setFixedSize(_SOURCE_THUMB, _SOURCE_THUMB)
        self._source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source.setStyleSheet("border: 1px dashed #777; color: #999;")
        self._source.setToolTip("Select this card, then click a Sources tile to assign.")
        layout.addWidget(self._source)

        delete = QToolButton()
        delete.setText("✕")
        delete.setToolTip("Remove this identity")
        delete.setAutoRaise(True)
        delete.clicked.connect(lambda: self.deleteRequested.emit(self._id))
        layout.addWidget(delete)
        self._apply_style()

    def identity_id(self) -> str:
        return self._id

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self._thumb.setPixmap(
            pixmap.scaled(
                _THUMB, _THUMB,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_source(self, pixmap: QPixmap | None, name: str | None) -> None:
        if pixmap is not None:
            self._source.setPixmap(
                pixmap.scaled(
                    _SOURCE_THUMB, _SOURCE_THUMB,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._source.setStyleSheet("border: 1px solid #2e9e3f;")
            self._source.setToolTip(name or "")
        else:
            self._source.clear()
            self._source.setText("no source")
            self._source.setStyleSheet("border: 1px dashed #777; color: #999;")

    def set_selected(self, on: bool) -> None:
        self._is_selected = on
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            "_IdentityCard { background: #3a3000; border: 1px solid #c8a000; }"
            if self._is_selected
            else ""
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.selected.emit(self._id)
        super().mousePressEvent(event)


class QFaceMapPanel(QWidget):
    analyzeRequested = Signal(int)         # stride
    cancelRequested = Signal()
    identitySelected = Signal(str)         # id ("" = none)
    deleteIdentityRequested = Signal(str)  # id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._face_map = FaceMap.empty()
        self._cards: dict[str, _IdentityCard] = {}
        self._selected: str | None = None
        self._analyzing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        controls = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze faces")
        self._analyze_btn.setToolTip(
            "Scan the target to discover the distinct people in it."
        )
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        controls.addWidget(self._analyze_btn)
        controls.addWidget(QLabel("stride"))
        self._stride = QSpinBox()
        self._stride.setRange(1, 300)
        self._stride.setValue(15)
        self._stride.setToolTip(
            "Sample every Nth frame. Larger = faster scan, may miss brief "
            "appearances."
        )
        controls.addWidget(self._stride)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Scrollable card list.
        self._list_host = QWidget()
        self._list = QVBoxLayout(self._list_host)
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(4)
        self._list.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._list_host)
        layout.addWidget(scroll, stretch=1)

        self._hint = QLabel(
            "No faces yet.\nClick “Analyze faces” to discover the people in this "
            "target, then assign a source to each."
        )
        self._hint.setWordWrap(True)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet("color: #888;")
        layout.addWidget(self._hint)

    # ---- Public API ----

    def stride(self) -> int:
        return self._stride.value()

    def selected_identity(self) -> str | None:
        return self._selected

    def face_map(self) -> FaceMap:
        return self._face_map

    def set_face_map(self, face_map: FaceMap) -> None:
        """Rebuild the cards from a catalog. Selection is preserved if the
        identity still exists, else cleared."""
        self._face_map = face_map
        # Drop old cards.
        for card in self._cards.values():
            self._list.removeWidget(card)
            card.deleteLater()
        self._cards = {}
        for ident in face_map.identities:
            card = _IdentityCard(ident)
            card.selected.connect(self._on_card_selected)
            card.deleteRequested.connect(self.deleteIdentityRequested)
            # Show the assigned source as a name immediately (thumb arrives via
            # set_source_thumbnail when the main window has loaded the pixmap).
            if ident.source_path:
                from pathlib import Path

                card.set_source(None, None)
                card._source.setText(Path(ident.source_path).name[:6])  # noqa: SLF001
            self._cards[ident.id] = card
            self._list.insertWidget(self._list.count() - 1, card)
        if self._selected not in self._cards:
            self._selected = None
        self._refresh_selection()
        self._hint.setVisible(face_map.is_empty())

    def set_target_thumbnail(self, identity_id: str, pixmap: QPixmap) -> None:
        card = self._cards.get(identity_id)
        if card is not None:
            card.set_thumbnail(pixmap)

    def set_source_thumbnail(
        self, identity_id: str, pixmap: QPixmap | None, name: str | None
    ) -> None:
        card = self._cards.get(identity_id)
        if card is not None:
            card.set_source(pixmap, name)

    def set_analyzing(self, on: bool) -> None:
        self._analyzing = on
        self._analyze_btn.setText("Cancel" if on else "Analyze faces")
        self._stride.setEnabled(not on)
        self._progress.setVisible(on)
        if on:
            self._progress.setRange(0, 0)  # busy until the first progress tick

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

    def _on_card_selected(self, identity_id: str) -> None:
        self._selected = None if self._selected == identity_id else identity_id
        self._refresh_selection()
        self.identitySelected.emit(self._selected or "")

    def _refresh_selection(self) -> None:
        for cid, card in self._cards.items():
            card.set_selected(cid == self._selected)
