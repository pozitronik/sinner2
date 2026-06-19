"""Folder-mirror view for the media library: the thumbnail grid split into
collapsible per-folder sections (the disk folders, as-is).

It shares the library's ONE item model (so thumbnails are generated once): each
section is an auto-height grid over a `_FolderProxy` that filters the library's
sort/filter proxy down to the files in that folder. Sections stack in a scroll
area; their headers fold/unfold. The flat grid stays the default; the library
toggles between the two.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    QModelIndex,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sinner2.library.library_model import ROLE_PATH, item_cell_size

_GRID_SPACING = 8


def _section_label(folder: Path, roots: list[Path]) -> str:
    """A readable label for a folder section: its path relative to the nearest
    containing root (so nested folders read as 'footage/b-roll'), or its name."""
    best: Path | None = None
    for root in roots:
        try:
            rel = folder.relative_to(root)
        except ValueError:
            continue
        if best is None or len(rel.parts) < len(best.parts):
            best = rel
    if best is None or not best.parts:
        return folder.name or str(folder)
    return str(best)


class _FolderProxy(QSortFilterProxyModel):
    """Filters the library's sort/filter proxy down to one folder's files."""

    def __init__(self, folder: Path, source: QSortFilterProxyModel) -> None:
        super().__init__()
        self._folder = folder
        self.setSourceModel(source)

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        src = self.sourceModel()
        path = src.data(src.index(row, 0, parent), ROLE_PATH)
        return path is not None and Path(path).parent == self._folder


class _AutoHeightGrid(QListView):
    """An IconMode grid that sizes its HEIGHT to fit all its items at the current
    width (no inner scrollbar) — so a column of these scrolls as one in the outer
    scroll area instead of nesting scrollers."""

    def __init__(self, dim: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setSpacing(_GRID_SPACING)
        self.setUniformItemSizes(True)
        self.setWordWrap(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._dim = dim
        self.setIconSize(QSize(dim, dim))
        self.setGridSize(item_cell_size(dim))

    def set_dim(self, dim: int) -> None:
        self._dim = dim
        self.setIconSize(QSize(dim, dim))
        self.setGridSize(item_cell_size(dim))
        self.recompute_height()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.recompute_height()

    def recompute_height(self) -> None:
        model = self.model()
        n = model.rowCount() if model is not None else 0
        cell = item_cell_size(self._dim)
        cw = cell.width() + _GRID_SPACING
        ch = cell.height() + _GRID_SPACING
        avail = max(cw, self.viewport().width())
        cols = max(1, avail // cw)
        rows = (n + cols - 1) // cols if n else 0
        self.setFixedHeight(rows * ch + _GRID_SPACING if rows else 0)


class _FolderSection(QWidget):
    """One collapsible folder: a header toggle + an auto-height grid."""

    pathActivated = Signal(object)  # Path

    def __init__(
        self, folder: Path, label: str, proxy: _FolderProxy, dim: int
    ) -> None:
        super().__init__()
        self._folder = folder
        self._proxy = proxy
        count = proxy.rowCount()
        self._header = QToolButton()
        self._header.setText(f"▾  {label}   ({count})")
        self._header.setToolTip(str(folder))
        self._header.setCheckable(True)
        self._header.setChecked(True)  # expanded by default
        self._header.setAutoRaise(True)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._header.setStyleSheet("QToolButton { font-weight: bold; }")
        self._header.toggled.connect(self._on_toggled)
        self._label = label

        self._grid = _AutoHeightGrid(dim)
        self._grid.setModel(proxy)
        self._grid.clicked.connect(self._emit)
        self._grid.activated.connect(self._emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._grid)

    def _on_toggled(self, expanded: bool) -> None:
        self._header.setText(
            f"{'▾' if expanded else '▸'}  {self._label}   "
            f"({self._proxy.rowCount()})"
        )
        self._grid.setVisible(expanded)

    def _emit(self, index: QModelIndex) -> None:
        path = index.data(ROLE_PATH)
        if path is not None:
            self.pathActivated.emit(Path(path))

    def grid(self) -> _AutoHeightGrid:
        return self._grid

    def set_dim(self, dim: int) -> None:
        self._grid.set_dim(dim)

    def clear_selection(self) -> None:
        self._grid.clearSelection()


class QFolderGridView(QScrollArea):
    """The folder-mirror view: a scroll of collapsible folder sections built from
    the library's (filtered/sorted) proxy. Emits pathSelected on a tile click —
    the same signal the flat grid uses."""

    pathSelected = Signal(object)  # Path

    def __init__(
        self, source_proxy: QSortFilterProxyModel, display_dim: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source_proxy
        self._dim = display_dim
        self._roots: list[Path] = []
        self._sections: list[_FolderSection] = []
        self._active = False  # only auto-rebuild while folder mode is shown
        self.setWidgetResizable(True)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 4, 4, 4)
        self._body_layout.setSpacing(6)
        self._body_layout.addStretch(1)
        self.setWidget(self._body)

        # Rebuild on (debounced) changes to the source proxy — a scan batch, a
        # filter/sort change, or a clear — so the sections track the model.
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(150)
        self._rebuild_timer.timeout.connect(self.rebuild)
        for sig in (
            source_proxy.rowsInserted, source_proxy.rowsRemoved,
            source_proxy.modelReset, source_proxy.layoutChanged,
        ):
            sig.connect(self._schedule_rebuild)

    def set_active(self, active: bool) -> None:
        """Folder mode shown/hidden. Auto-rebuilds (on model changes) run only
        while active, so flat mode pays nothing for the section machinery."""
        self._active = bool(active)

    def set_roots(self, roots: list[Path]) -> None:
        self._roots = list(roots)

    def set_display_dim(self, dim: int) -> None:
        self._dim = dim
        for section in self._sections:
            section.set_dim(dim)

    def _schedule_rebuild(self, *_: object) -> None:
        if self._active:
            self._rebuild_timer.start()

    def rebuild(self) -> None:
        """Re-derive the folder sections from the current (filtered) proxy rows.
        Cheap-ish: groups by parent dir, one section + proxy per folder."""
        # Tear the old sections down.
        for section in self._sections:
            section.setParent(None)
            section.deleteLater()
        self._sections = []

        # Distinct parent folders present in the (filtered) proxy, sorted.
        src = self._source
        folders: set[Path] = set()
        for row in range(src.rowCount()):
            path = src.data(src.index(row, 0), ROLE_PATH)
            if path is not None:
                folders.add(Path(path).parent)

        insert_at = self._body_layout.count() - 1  # before the trailing stretch
        for folder in sorted(folders, key=lambda p: str(p).lower()):
            proxy = _FolderProxy(folder, src)
            section = _FolderSection(
                folder, _section_label(folder, self._roots), proxy, self._dim
            )
            section.pathActivated.connect(self._on_path_activated)
            self._body_layout.insertWidget(insert_at, section)
            insert_at += 1
            self._sections.append(section)
            section.grid().recompute_height()

    def _on_path_activated(self, path: object) -> None:
        # Single-selection across the whole view: clear the other sections so the
        # highlighted tile reads as the one global selection.
        sender = self.sender()
        for section in self._sections:
            if section is not sender:
                section.clear_selection()
        self.pathSelected.emit(path)
