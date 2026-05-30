"""Library item model + sort/filter proxy.

The item model holds one row per library entry; thumbnails are generated
off the main thread and applied via a queued signal so worker callbacks
never touch widgets directly. The proxy exposes a small typed API (sort
field, filter text) over QSortFilterProxyModel's built-ins.

Roles:
  DisplayRole       — caption (what the view's text shows)
  DecorationRole    — current thumbnail pixmap (placeholder until ready)
  ROLE_PATH         — Path to the source file (str — Qt models prefer
                      bare strings; the wrapper converts at the boundary)
  ROLE_NAME         — lowercased filename for case-insensitive name sort
  ROLE_PIXEL_COUNT  — int (source resolution, not thumb)
  ROLE_MOD_DATE     — float (mtime)
  ROLE_FILE_SIZE    — int (bytes)

Filter operates on caption (case-insensitive substring) so the user can
type any visible text and narrow the grid live.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QObject, QSize, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QPixmap, QStandardItem, QStandardItemModel

from sinner2.library.thumbnail_generator import (
    ThumbnailError,
    ThumbnailGenerator,
    ThumbnailOutcome,
    ThumbnailResult,
)


ROLE_PATH = Qt.ItemDataRole.UserRole + 1
ROLE_NAME = Qt.ItemDataRole.UserRole + 2
ROLE_PIXEL_COUNT = Qt.ItemDataRole.UserRole + 3
ROLE_MOD_DATE = Qt.ItemDataRole.UserRole + 4
ROLE_FILE_SIZE = Qt.ItemDataRole.UserRole + 5
ROLE_ERROR = Qt.ItemDataRole.UserRole + 6
# JPEG path of the cached thumbnail. Stored so set_display_dim can
# re-scale from the cached extraction without re-running the generator.
ROLE_JPEG_PATH = Qt.ItemDataRole.UserRole + 7


class SortField(str, Enum):
    NAME = "name"
    PATH = "path"
    DATE = "date"
    SIZE = "size"
    PIXELS = "pixels"


_SORT_ROLE_BY_FIELD: dict[SortField, int] = {
    SortField.NAME: ROLE_NAME,
    SortField.PATH: ROLE_PATH,
    SortField.DATE: ROLE_MOD_DATE,
    SortField.SIZE: ROLE_FILE_SIZE,
    SortField.PIXELS: ROLE_PIXEL_COUNT,
}


# Caption area scales with display_dim so larger tiles get room for
# more wrapped lines and the proportion icon-to-text stays balanced.
# Floor is two short lines at the default font (~48px); above that it
# grows roughly 1px caption per 4px icon, capped to keep aspect from
# becoming taller than wide on very large tiles.
_CAPTION_HEIGHT_MIN = 48
_CAPTION_HEIGHT_MAX_FACTOR = 0.33
_CAPTION_DIM_FACTOR = 0.25
# Horizontal padding around the icon inside its cell.
_ITEM_H_PADDING = 16


def _caption_height(display_dim: int) -> int:
    scaled = int(display_dim * _CAPTION_DIM_FACTOR)
    capped = int(display_dim * _CAPTION_HEIGHT_MAX_FACTOR)
    return max(_CAPTION_HEIGHT_MIN, min(scaled, capped))


def _placeholder_pixmap(size: int) -> QPixmap:
    """Solid-grey square shown until the real thumbnail arrives."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.gray)
    return pix


def _scale_pixmap_to_dim(pix: QPixmap, dim: int) -> QPixmap:
    """Fit pixmap within dim×dim preserving aspect. Returns the input
    unchanged when it's already small enough — Qt would do the same
    on paint but we want a stable sizeHint."""
    if pix.isNull():
        return pix
    if pix.width() <= dim and pix.height() <= dim:
        return pix
    return pix.scaled(
        dim,
        dim,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def item_cell_size(display_dim: int) -> QSize:
    """Per-item cell footprint at the given display dim. The view's
    gridSize and per-item sizeHint must agree on this — otherwise
    Qt's IconMode layout falls back to using the pixmap's intrinsic
    size and tiles overlap when the cached pixmap is larger than the
    display dim (the typical case: extraction at 384, display at 128)."""
    return QSize(display_dim + _ITEM_H_PADDING, display_dim + _caption_height(display_dim))


class LibraryItemModel(QStandardItemModel):
    """Item model + path-keyed lookup + thumbnail-completion plumbing.

    Add/remove paths via add_path/remove_path. Thumbnails are produced
    by a ThumbnailGenerator on worker threads; results land here via
    a queued Qt signal so the view doesn't observe partial state.
    """

    # Signals carry primitive types only (Path is fine) so cross-thread
    # queueing works without registering custom metatypes.
    _thumbnailReady = Signal(object, str, str, int)  # path, jpeg_path, caption, pixel_count
    _thumbnailFailed = Signal(object, str)  # path, reason

    def __init__(
        self,
        generator: ThumbnailGenerator,
        *,
        display_dim: int = 128,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._generator = generator
        self._items: dict[Path, QStandardItem] = {}
        # display_dim is the SIZE THE PIXMAP IS RENDERED AT in the grid.
        # Independent of generator.thumb_dim (the EXTRACTION size on
        # disk). The model pre-scales pixmaps to display_dim so the
        # view's IconMode layout has a stable per-item size; without
        # this, items with the larger extracted pixmap (384px) get cells
        # sized for 384, and tiles overlap when the view's iconSize is
        # set to a smaller value (128px default).
        self._display_dim = display_dim
        self._placeholder = _placeholder_pixmap(display_dim)
        # QueuedConnection because the worker thread emits — main thread
        # consumes. Without explicit queueing Qt would auto-pick Direct
        # in single-thread contexts and Queued only across threads;
        # being explicit removes guesswork in tests.
        self._thumbnailReady.connect(
            self._apply_thumbnail, type=Qt.ConnectionType.QueuedConnection
        )
        self._thumbnailFailed.connect(
            self._apply_failure, type=Qt.ConnectionType.QueuedConnection
        )

    def display_dim(self) -> int:
        return self._display_dim

    def set_display_dim(self, dim: int) -> None:
        """Re-scale every cached thumbnail to the new dim. Cheap because
        we keep the JPEG path per item and QPixmap.scaled is in-memory —
        no re-extraction. Items still waiting for first generation get
        a refreshed placeholder so their cell footprint matches."""
        if dim == self._display_dim:
            return
        self._display_dim = dim
        self._placeholder = _placeholder_pixmap(dim)
        cell = item_cell_size(dim)
        for row in range(self.rowCount()):
            item = self.item(row)
            if item is None:
                continue
            jpeg_path = item.data(ROLE_JPEG_PATH)
            if jpeg_path:
                pix = QPixmap(str(jpeg_path))
                scaled = _scale_pixmap_to_dim(pix, dim)
                item.setData(scaled, Qt.ItemDataRole.DecorationRole)
            else:
                # Thumbnail still pending — show resized placeholder so
                # the cell footprint matches the others.
                item.setData(self._placeholder, Qt.ItemDataRole.DecorationRole)
            item.setSizeHint(cell)

    def paths(self) -> list[Path]:
        # Preserve insertion order (QStandardItemModel uses a list internally).
        return [
            Path(self.item(row).data(ROLE_PATH))
            for row in range(self.rowCount())
        ]

    def has_path(self, path: Path) -> bool:
        return path in self._items

    def add_path(self, path: Path) -> bool:
        """Add one entry. Returns False on duplicate (dedupe is by
        Path equality — case-sensitive on POSIX, OS-default on Windows)."""
        if path in self._items:
            return False
        item = QStandardItem()
        item.setEditable(False)
        item.setData(path.name, Qt.ItemDataRole.DisplayRole)
        item.setData(self._placeholder, Qt.ItemDataRole.DecorationRole)
        # Tooltip shows the full path before the thumbnail callback can
        # set a richer caption. Without this, a clipped caption on a
        # small tile leaves no way to see what the file actually is.
        item.setData(str(path), Qt.ItemDataRole.ToolTipRole)
        # Explicit per-item sizeHint — the view's uniformItemSizes(True)
        # uses the FIRST item's hint, but we set every item's hint
        # anyway so a set_display_dim mid-session re-sizes them all
        # cleanly.
        item.setSizeHint(item_cell_size(self._display_dim))
        item.setData(str(path), ROLE_PATH)
        item.setData(path.name.lower(), ROLE_NAME)
        # Stat now so sort works before the thumbnail completes — the
        # thumbnail callback refreshes these with the same values.
        try:
            st = path.stat()
            item.setData(float(st.st_mtime), ROLE_MOD_DATE)
            item.setData(int(st.st_size), ROLE_FILE_SIZE)
        except OSError:
            item.setData(0.0, ROLE_MOD_DATE)
            item.setData(0, ROLE_FILE_SIZE)
        item.setData(0, ROLE_PIXEL_COUNT)  # filled in by thumbnail meta
        self.appendRow(item)
        self._items[path] = item
        self._generator.submit(path, self._on_thumb_outcome)
        return True

    def add_paths(self, paths: Iterable[Path]) -> int:
        """Bulk add; returns count actually added (post-dedupe)."""
        added = 0
        for p in paths:
            if self.add_path(p):
                added += 1
        return added

    def remove_path(self, path: Path) -> bool:
        item = self._items.pop(path, None)
        if item is None:
            return False
        # QStandardItem.row() returns -1 if the item was already removed
        # — guard so a double-remove is a silent no-op.
        row = item.row()
        if row >= 0:
            self.removeRow(row)
        return True

    def clear_paths(self) -> None:
        self.clear()
        self._items.clear()

    def _on_thumb_outcome(self, outcome: ThumbnailOutcome) -> None:
        """Called on a worker thread by the generator. Re-emits as a
        queued signal so the apply runs on the GUI thread."""
        if isinstance(outcome, ThumbnailResult):
            self._thumbnailReady.emit(
                outcome.source,
                str(outcome.jpeg_path),
                outcome.meta.caption,
                outcome.meta.pixel_count,
            )
        else:
            assert isinstance(outcome, ThumbnailError)
            self._thumbnailFailed.emit(outcome.source, outcome.reason)

    def _apply_thumbnail(
        self, path: Path, jpeg_path: str, caption: str, pixel_count: int
    ) -> None:
        item = self._items.get(path)
        if item is None:
            return
        pix = QPixmap(jpeg_path)
        if not pix.isNull():
            # Pre-scale to display_dim with KeepAspectRatio so the
            # view doesn't get a 384px pixmap when the cell is 128px.
            # Stored jpeg_path lets us rescale on later display_dim
            # changes without touching the generator.
            item.setData(
                _scale_pixmap_to_dim(pix, self._display_dim),
                Qt.ItemDataRole.DecorationRole,
            )
            item.setData(jpeg_path, ROLE_JPEG_PATH)
        item.setData(caption, Qt.ItemDataRole.DisplayRole)
        # Tooltip is the full caption (filename + dimensions) plus the
        # full path on its own line — small tiles will elide / wrap-
        # clip the display text, so the tooltip is the only place the
        # user can read the complete info.
        path_str = item.data(ROLE_PATH) or ""
        item.setData(f"{caption}\n{path_str}", Qt.ItemDataRole.ToolTipRole)
        item.setData(int(pixel_count), ROLE_PIXEL_COUNT)
        # Clear any prior error marker.
        item.setData(None, ROLE_ERROR)

    def _apply_failure(self, path: Path, reason: str) -> None:
        item = self._items.get(path)
        if item is None:
            return
        item.setData(reason, ROLE_ERROR)
        # Caption surfaces the error so the user can see why a tile is
        # blank — beats a silent failure that looks like "loading…" forever.
        item.setData(f"{path.name} — {reason}", Qt.ItemDataRole.DisplayRole)
        item.setData(
            f"{path}\n{reason}", Qt.ItemDataRole.ToolTipRole
        )


class LibrarySortFilterProxy(QSortFilterProxyModel):
    """Sort by a SortField; filter by case-insensitive substring against caption."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sort_field = SortField.NAME
        self.setSortRole(_SORT_ROLE_BY_FIELD[self._sort_field])
        self.setFilterRole(Qt.ItemDataRole.DisplayRole)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setDynamicSortFilter(True)

    def set_sort_field(self, field: SortField) -> None:
        if field == self._sort_field:
            return
        self._sort_field = field
        self.setSortRole(_SORT_ROLE_BY_FIELD[field])
        # setSortRole alone does NOT re-sort an already-sorted proxy,
        # and sort() is a no-op when (column, order) are unchanged —
        # Qt's optimization. invalidate() clears the cached order so
        # the subsequent sort() actually re-runs against the new role.
        order = self.sortOrder()
        if order == Qt.SortOrder.AscendingOrder or order == Qt.SortOrder.DescendingOrder:
            self.invalidate()
            self.sort(0, order)
        else:
            self.sort(0, Qt.SortOrder.AscendingOrder)

    def sort_field(self) -> SortField:
        return self._sort_field

    def set_filter_text(self, text: str) -> None:
        # Substring filter — use FixedString matching so regex special
        # characters in filenames don't cause surprises.
        self.setFilterFixedString(text)
