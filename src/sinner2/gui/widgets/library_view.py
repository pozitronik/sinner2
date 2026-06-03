"""Library view widget — grid of media thumbnails, drag-drop, sort/filter.

One widget composes:
  - QListView in IconMode for the grid (built-in scroll + keyboard nav)
  - QSortFilterProxyModel for live sort/filter
  - Filter line edit + sort dropdown + sort-direction button + Add button
  - Drag-drop accept (files and folders, folders recursively scanned)

The accept predicate decides what counts — sources accept images only,
targets accept images and videos. Folders are scanned regardless and
filtered per-file, so dragging a mixed folder onto the targets library
yields the images and videos and silently drops the rest.

Emits pathSelected(Path) when the user clicks (or activates with Enter)
a tile. Caller wires that to whichever 'load this' action makes sense.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QAction,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sinner2.library.library_model import (
    ROLE_PATH,
    LibraryItemModel,
    LibrarySortFilterProxy,
    SortField,
    item_cell_size,
)
from sinner2.library.media_kind import is_media
from sinner2.library.thumbnail_generator import ThumbnailGenerator


_SORT_FIELD_LABELS: list[tuple[SortField, str]] = [
    (SortField.NAME, "Name"),
    (SortField.PATH, "Path"),
    (SortField.DATE, "Date"),
    (SortField.SIZE, "Size"),
    (SortField.PIXELS, "Pixels"),
]

# Batch size for streaming discovered files back to the GUI. Per-file
# emits cost a Qt signal hop each — at thousands of files on a network
# share, that adds up. A batch of 64 amortises the overhead while still
# letting the user see progress within the first few hundred ms.
_SCAN_BATCH_SIZE = 64

# Display-dim range. The generator extracts at a fixed larger size and
# Qt downscales the QPixmap to whatever display_dim is in effect, so
# the user can resize tiles live without re-extracting. Max is capped
# at the generator's extraction size at runtime (upscaling past the
# source resolution would only blur).
_DISPLAY_DIM_MIN = 64
_DISPLAY_DIM_STEP = 32
_DISPLAY_DIM_DEFAULT = 128


# Keys that relocate the grid cursor. A press of one of these that actually
# moves the current item applies the file under it immediately (sinner1
# parity — no Enter needed). Enter and mouse clicks still apply via the
# view's activated/clicked signals.
_NAV_KEYS = frozenset(
    {
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
        Qt.Key.Key_Home,
        Qt.Key.Key_End,
        Qt.Key.Key_PageUp,
        Qt.Key.Key_PageDown,
    }
)


class _NavigatingListView(QListView):
    """QListView that emits `navigated` whenever an arrow/page/home/end key
    actually moves the current item.

    The caller wires this to its apply-the-current-file action so keyboard
    navigation loads the tile under the cursor live. We compare the current
    index before and after letting the base class move the cursor and only
    emit when it genuinely changed — so a key pressed against a grid edge
    (no movement), a non-navigation key, and any programmatic model change
    all stay silent and never auto-load a file the user didn't pick.
    """

    navigated = Signal(object)  # the new current QModelIndex (proxy index)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() not in _NAV_KEYS:
            super().keyPressEvent(event)
            return
        before = self.currentIndex()
        super().keyPressEvent(event)
        after = self.currentIndex()
        if after.isValid() and after != before:
            self.navigated.emit(after)


class _FolderScanWorker(QObject):
    """Walks paths off the GUI thread, emitting accepted files in batches.

    Lives on a QThread so the GIL releases between rglob iterations and
    the GUI can keep painting. A user-set _cancel flag interrupts the
    walk between batches — important on a slow network share where a
    single os.scandir can take seconds.
    """

    batch = Signal(list)  # list[Path]
    finished = Signal(int)  # total files emitted

    def __init__(self, roots: list[Path], accept) -> None:
        super().__init__()
        self._roots = roots
        self._accept = accept
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        total = 0
        pending: list[Path] = []
        try:
            for root in self._roots:
                if self._cancel:
                    break
                try:
                    if root.is_file() and self._accept(root):
                        pending.append(root)
                    elif root.is_dir():
                        for child in root.rglob("*"):
                            if self._cancel:
                                break
                            try:
                                if child.is_file() and self._accept(child):
                                    pending.append(child)
                                    if len(pending) >= _SCAN_BATCH_SIZE:
                                        total += len(pending)
                                        self.batch.emit(pending)
                                        pending = []
                            except OSError:
                                # Per-entry permission/dead-symlink errors
                                # must not abort the whole scan.
                                continue
                except OSError:
                    continue
            if pending:
                total += len(pending)
                self.batch.emit(pending)
        finally:
            self.finished.emit(total)


class QLibraryView(QWidget):
    """Self-contained library widget. One per source/target tab.

    Two parallel lists matter:
      - `roots`: the entries the user added (folder OR file, as-added).
        This is what gets persisted to settings — a folder dropped onto
        the library lands as a single root, not as the (potentially
        thousands of) files inside.
      - `paths`: the expanded files currently displayed in the grid.
        Folders contribute their accepted children; files contribute
        themselves.
    """

    pathSelected = Signal(Path)
    pathsChanged = Signal(list)  # list[Path] — emitted on any expanded-grid mutation
    rootsChanged = Signal(list)  # list[Path] — emitted when user-added roots change (persist this)
    displayDimChanged = Signal(int)  # emitted when user resizes thumbnails
    sortChanged = Signal()  # emitted when the sort field or direction changes

    def __init__(
        self,
        generator: ThumbnailGenerator,
        *,
        accept: Callable[[Path], bool] = is_media,
        file_dialog_filter: str = "Media files (*)",
        display_dim: int = _DISPLAY_DIM_DEFAULT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._accept = accept
        self._file_dialog_filter = file_dialog_filter
        # Display dim — how large the tile is rendered. Generator's
        # thumb_dim is what's actually extracted/cached and forms our
        # upper bound (no upscaling past the source pixmap, which would
        # only blur). Clamp on every update via _clamp_display_dim.
        self._generator = generator
        self._display_dim = self._clamp_display_dim(display_dim)
        # Model needs the display_dim up-front so the first batch of
        # items lands with correct sizeHints and pre-scaled placeholders.
        self._model = LibraryItemModel(
            generator, display_dim=self._display_dim, parent=self
        )
        self._proxy = LibrarySortFilterProxy(parent=self)
        self._proxy.setSourceModel(self._model)

        # Top control bar.
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter…")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._proxy.set_filter_text)

        self._sort_combo = QComboBox()
        for field, label in _SORT_FIELD_LABELS:
            self._sort_combo.addItem(label, field)
        self._sort_combo.currentIndexChanged.connect(self._on_sort_field_changed)

        self._sort_dir_button = QToolButton()
        self._sort_dir_button.setText("▲")  # ascending by default
        self._sort_dir_button.setToolTip("Sort direction (ascending/descending)")
        self._sort_dir_button.clicked.connect(self._toggle_sort_direction)

        # Add button is a split-button: main click adds files (the
        # common case), arrow opens a menu with the Folder choice.
        # Folder ingestion runs on a background thread so big network
        # shares don't freeze the GUI.
        self._add_button = QToolButton()
        self._add_button.setText("Add…")
        self._add_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        self._add_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self._add_button.setToolTip(
            "Add files. Use the dropdown to add a folder (recursively scanned)."
        )
        self._add_button.clicked.connect(self._open_add_files_dialog)
        add_menu = QMenu(self._add_button)
        files_action = QAction("Files…", add_menu)
        files_action.triggered.connect(self._open_add_files_dialog)
        folder_action = QAction("Folder…", add_menu)
        folder_action.triggered.connect(self._open_add_folder_dialog)
        add_menu.addAction(files_action)
        add_menu.addAction(folder_action)
        self._add_button.setMenu(add_menu)

        # Scanning indicator — only shown while a background scan is
        # active. Keeps the user aware that work is happening even
        # though the GUI stays responsive.
        self._scan_label = QLabel("")
        self._scan_label.setStyleSheet("color: gray;")

        # Thumbnail size buttons. Ctrl+Wheel on the grid does the same
        # thing — the buttons make the feature discoverable without
        # requiring a keyboard hint.
        self._shrink_button = QToolButton()
        self._shrink_button.setText("−")
        self._shrink_button.setToolTip(
            f"Smaller thumbnails (Ctrl+Wheel down). Step: {_DISPLAY_DIM_STEP}px."
        )
        self._shrink_button.clicked.connect(
            lambda: self.set_display_dim(self._display_dim - _DISPLAY_DIM_STEP)
        )
        self._grow_button = QToolButton()
        self._grow_button.setText("+")
        self._grow_button.setToolTip(
            f"Larger thumbnails (Ctrl+Wheel up). Step: {_DISPLAY_DIM_STEP}px."
        )
        self._grow_button.clicked.connect(
            lambda: self.set_display_dim(self._display_dim + _DISPLAY_DIM_STEP)
        )

        # Clear button — sinner1's "Library → Clear" menu equivalent.
        # Destructive (wipes user-curated roots), so confirms first.
        # Tooltip names what gets wiped so the user isn't surprised
        # by losing the entry list across launches.
        self._clear_button = QToolButton()
        self._clear_button.setText("Clear")
        self._clear_button.setToolTip(
            "Remove every entry from this library (folder roots + files).\n"
            "Cleared list persists across launches — re-add via Add… or drag-drop."
        )
        self._clear_button.clicked.connect(self._confirm_clear)

        controls = QHBoxLayout()
        controls.addWidget(self._filter_edit, stretch=1)
        controls.addWidget(self._scan_label)
        controls.addWidget(self._shrink_button)
        controls.addWidget(self._grow_button)
        controls.addWidget(self._sort_combo)
        controls.addWidget(self._sort_dir_button)
        controls.addWidget(self._add_button)
        controls.addWidget(self._clear_button)

        # The grid itself.
        self._list = _NavigatingListView()
        self._list.setModel(self._proxy)
        self._list.setViewMode(QListView.ViewMode.IconMode)
        self._list.setResizeMode(QListView.ResizeMode.Adjust)
        self._list.setMovement(QListView.Movement.Static)
        self._list.setSpacing(8)
        self._list.setUniformItemSizes(True)
        self._list.setWordWrap(True)
        self._list.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self._list.activated.connect(self._on_activated)
        self._list.clicked.connect(self._on_activated)
        # Arrow-key navigation applies the tile under the cursor immediately,
        # routing through the same handler as click/Enter.
        self._list.navigated.connect(self._on_activated)
        # Apply current display_dim to icon + grid sizes. Same helper
        # called by set_display_dim later so resize is just a config
        # change — Qt re-renders the cached QPixmap at the new size on
        # the next paint.
        self._apply_display_dim_to_view()
        # Ctrl+Wheel on the grid resizes thumbnails. We install an
        # event filter rather than subclass QListView for surface
        # minimalism.
        self._list.viewport().installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._list, stretch=1)

        self.setAcceptDrops(True)

        # Apply the initial sort so the grid is ordered from the start.
        self._proxy.sort(0, Qt.SortOrder.AscendingOrder)

        # Track all live scan workers and their threads so shutdown()
        # / clear() can cancel them in bulk — without this, a folder
        # scan outlives the GUI or the user's clear and keeps streaming
        # entries into the wiped model. The label switches to
        # "Scanning…" while any scan is live.
        self._scan_jobs: list[tuple[QThread, _FolderScanWorker]] = []
        # Bumped each time scans are cancelled (via clear() or
        # shutdown()). Batches that were emitted before the bump carry
        # their start-time epoch and get discarded by the handler
        # when stale. Without this, a queued batch signal fires AFTER
        # clear() emptied the model and re-populates it from a scan
        # the user just cancelled.
        self._scan_cancel_epoch = 0

        # Roots: user-added entries (folders or files, as-added). This
        # is the list that gets persisted; the grid's `paths` are
        # derived by re-expanding roots on restore.
        self._roots: list[Path] = []

    # ---- Public API ----

    def paths(self) -> list[Path]:
        return self._model.paths()

    def add_paths(self, paths: Iterable[Path]) -> int:
        """Add a batch of paths. Returns the count actually added (post-
        accept-filter and post-dedupe). Folders are NOT expanded here;
        call ingest_files_and_folders for that."""
        accepted = [p for p in paths if self._accept(p)]
        added = self._model.add_paths(accepted)
        if added:
            self.pathsChanged.emit(self._model.paths())
        return added

    def roots(self) -> list[Path]:
        return list(self._roots)

    # ---- Display dim ----

    def sort_field(self) -> str:
        """Current sort field as its str token (for persistence)."""
        data = self._sort_combo.currentData()
        return data.value if isinstance(data, SortField) else str(data)

    def sort_order(self) -> str:
        """'asc' or 'desc'."""
        return (
            "desc"
            if self._proxy.sortOrder() == Qt.SortOrder.DescendingOrder
            else "asc"
        )

    def set_sort(self, field: str | None, order: str | None) -> None:
        """Silent restore of the sort field + direction (no sortChanged)."""
        if field is not None:
            try:
                sort_field = SortField(field)
            except ValueError:
                sort_field = None
            if sort_field is not None:
                self._sort_combo.blockSignals(True)
                for i in range(self._sort_combo.count()):
                    if self._sort_combo.itemData(i) == sort_field:
                        self._sort_combo.setCurrentIndex(i)
                        break
                self._sort_combo.blockSignals(False)
                self._proxy.set_sort_field(sort_field)
        sort_order = (
            Qt.SortOrder.DescendingOrder
            if order == "desc"
            else Qt.SortOrder.AscendingOrder
        )
        self._proxy.sort(0, sort_order)
        self._sort_dir_button.setText(
            "▲" if sort_order == Qt.SortOrder.AscendingOrder else "▼"
        )

    def display_dim(self) -> int:
        return self._display_dim

    def set_display_dim(self, dim: int) -> None:
        new_dim = self._clamp_display_dim(dim)
        if new_dim == self._display_dim:
            return
        self._display_dim = new_dim
        self._apply_display_dim_to_view()
        self.displayDimChanged.emit(new_dim)

    def _clamp_display_dim(self, dim: int) -> int:
        # Lower bound: _DISPLAY_DIM_MIN (anything smaller is useless).
        # Upper bound: the generator's extraction size (upscaling past
        # the source pixmap would just blur the tile).
        upper = self._generator.thumb_dim
        snapped = max(_DISPLAY_DIM_MIN, min(int(dim), upper))
        # Snap to the step grid so +/- buttons land cleanly on the
        # same values regardless of starting point.
        snapped = (snapped // _DISPLAY_DIM_STEP) * _DISPLAY_DIM_STEP
        return max(_DISPLAY_DIM_MIN, snapped)

    def _apply_display_dim_to_view(self) -> None:
        dim = self._display_dim
        self._list.setIconSize(QSize(dim, dim))
        # Grid cell footprint must agree with the model's per-item
        # sizeHint (see item_cell_size) — otherwise the view falls back
        # to the pixmap's intrinsic size for some metrics and tiles
        # overlap when the extracted pixmap is larger than display_dim.
        self._list.setGridSize(item_cell_size(dim))
        # Propagate to the model so existing items get rescaled
        # pixmaps from their cached JPEGs + updated size hints.
        if hasattr(self, "_model"):
            self._model.set_display_dim(dim)

    def set_paths(self, paths: Iterable[Path]) -> None:
        """Replace the entire library with the given paths (silent —
        does NOT emit pathsChanged, used by restore-from-settings).
        Does NOT touch roots — see set_roots for that."""
        # Cancel any in-flight folder scan FIRST (bump the epoch) so its late
        # batches don't repopulate the grid we're about to replace — same guard
        # clear() applies.
        self._cancel_active_scans()
        self._model.clear_paths()
        accepted = [p for p in paths if self._accept(p)]
        self._model.add_paths(accepted)

    def set_roots(self, roots: Iterable[Path]) -> None:
        """Silent restore of user-added entries. Each root is validated
        (files: must exist; folders: must be a directory) and then
        re-expanded — files added to the grid directly, folders scanned
        in the background. Does NOT emit rootsChanged or pathsChanged
        (restore mustn't round-trip back into persist)."""
        # Cancel any in-flight scan FIRST (bump the epoch) so a prior scan's late
        # batches don't survive into the new library — and so the new scan
        # started below carries the bumped epoch (without this they share an
        # epoch and the old batches pass the staleness check and reinsert the
        # previous folder's files). Mirrors clear().
        self._cancel_active_scans()
        self._roots = []
        self._model.clear_paths()
        immediate_files: list[Path] = []
        folders: list[Path] = []
        for p in roots:
            try:
                if p.is_dir():
                    self._roots.append(p)
                    folders.append(p)
                elif p.is_file() and self._accept(p):
                    self._roots.append(p)
                    immediate_files.append(p)
                # Else: file that no longer exists or is rejected —
                # silently drop from roots so the persisted state self-
                # heals on the next save.
            except OSError:
                continue
        if immediate_files:
            self._model.add_paths(immediate_files)
        if folders:
            self._start_background_scan(folders)

    def remove_path(self, path: Path) -> bool:
        """Remove a single path from the grid AND from roots if present.

        Note: a file that came from a folder root's expansion stays in
        roots (the folder is the root, not the file). Removing such a
        file removes it from the grid for this session but it will
        re-appear on next launch when the folder is re-scanned. To
        permanently exclude, remove the parent folder root or actually
        delete the file from disk."""
        ok = self._model.remove_path(path)
        if not ok:
            return False
        self.pathsChanged.emit(self._model.paths())
        if path in self._roots:
            self._roots.remove(path)
            self.rootsChanged.emit(list(self._roots))
        return True

    def clear(self) -> None:
        had_paths = bool(self._model.paths())
        had_roots = bool(self._roots)
        # Stop any in-flight folder scan FIRST so late batches don't
        # repopulate the just-cleared model. Cooperative cancel —
        # workers exit on their next loop tick, late batches are
        # filtered out by the epoch bump.
        self._cancel_active_scans()
        if not (had_paths or had_roots):
            return
        self._model.clear_paths()
        self._roots = []
        if had_paths:
            self.pathsChanged.emit([])
        if had_roots:
            self.rootsChanged.emit([])

    def ingest_files_and_folders(self, paths: Iterable[Path]) -> None:
        """Add files; for folders, recursively scan IN A BACKGROUND THREAD
        so a slow network share doesn't freeze the GUI. Each path becomes
        a `root` (the persistence unit) — a folder root stays a single
        entry in settings even when it expands to thousands of files in
        the grid.

        Does NOT return a count — call sites that want to know what
        landed should subscribe to pathsChanged.
        """
        paths_list = list(paths)
        new_roots: list[Path] = []
        immediate_files: list[Path] = []
        folder_roots: list[Path] = []
        for p in paths_list:
            if p in self._roots:
                continue  # already-added root — silently dedupe
            try:
                if p.is_dir():
                    new_roots.append(p)
                    folder_roots.append(p)
                elif p.is_file() and self._accept(p):
                    new_roots.append(p)
                    immediate_files.append(p)
            except OSError:
                continue
        if not new_roots:
            return
        self._roots.extend(new_roots)
        if immediate_files:
            self.add_paths(immediate_files)
        if folder_roots:
            self._start_background_scan(folder_roots)
        self.rootsChanged.emit(list(self._roots))

    # ---- Background scanning ----

    def _start_background_scan(self, roots: list[Path]) -> None:
        thread = QThread(self)
        worker = _FolderScanWorker(roots, self._accept)
        # Tag the worker with the cancel epoch it started under so
        # _on_scan_batch can discard stale batches (cancel/clear that
        # runs between emit and dispatch).
        worker.start_epoch = self._scan_cancel_epoch  # type: ignore[attr-defined]
        self._scan_jobs.append((thread, worker))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # IMPORTANT: connect to BOUND METHODS of self, not lambdas.
        # PySide6's AutoConnection routes the slot to the receiver
        # QObject's thread — for a bound method that's `self` (the
        # view, on the GUI thread). A lambda has no QObject context,
        # so even with explicit Qt.QueuedConnection the slot wrapper
        # attaches to the sender (the worker, on its own thread),
        # producing "QThread::wait: Thread tried to wait on itself"
        # warnings when _on_scan_finished calls thread.wait().
        # Worker / epoch / thread are resolved inside the slot via
        # self.sender() (the worker QObject) — valid as long as the
        # worker hasn't been deleteLater'd yet, which it can't be
        # because we're inside its own signal dispatch.
        worker.batch.connect(self._on_scan_batch)
        worker.finished.connect(self._on_scan_worker_finished)
        self._scan_label.setText("Scanning…")
        thread.start()

    def _on_scan_batch(self, paths: list[Path]) -> None:
        """Worker → GUI thread (via Qt's auto routing for bound-method
        slots). Discard batches from a scan that was cancelled before
        the queued signal could dispatch — without this, a clear()
        during an active scan races with a few in-flight batches and
        the wiped library refills itself."""
        worker = self.sender()
        start_epoch = getattr(worker, "start_epoch", 0)
        if start_epoch < self._scan_cancel_epoch:
            return
        self.add_paths(paths)

    def _cancel_active_scans(self) -> None:
        """Cooperatively cancel every live scan worker. Does NOT wait
        for the worker threads to exit — the worker's cancel flag is
        checked at the top of every rglob iteration, so it exits on
        the next loop tick. Late-emitted batches are filtered out by
        the bumped scan-cancel epoch.

        For shutdown, the caller must additionally join + delete the
        threads (shutdown() does this); for clear() it's enough to
        let them die in the background."""
        self._scan_cancel_epoch += 1
        for _thread, worker in self._scan_jobs:
            worker.cancel()

    def _on_scan_worker_finished(self, _count: int) -> None:
        """Worker → GUI thread. Looks the worker up via sender() and
        joins / deletes its thread on this side, where thread.wait()
        is correctly waiting on a DIFFERENT thread (the worker's
        own). Replaces a previous lambda-based connection that ran
        the slot on the worker thread itself."""
        worker = self.sender()
        if not isinstance(worker, _FolderScanWorker):
            return
        # Find the matching (thread, worker) entry.
        thread: QThread | None = None
        for t, w in self._scan_jobs:
            if w is worker:
                thread = t
                break
        if thread is None:
            return
        thread.quit()
        thread.wait(1000)
        worker.deleteLater()
        thread.deleteLater()
        self._scan_jobs = [j for j in self._scan_jobs if j[1] is not worker]
        if not self._scan_jobs:
            self._scan_label.setText("")

    def shutdown(self) -> None:
        """Cancel every live folder scan and wait briefly for the QThreads
        to exit. MUST be called before the QApplication quits; without
        this, a long network folder walk outlives the GUI and the
        process stays alive until the walk completes naturally."""
        # Snapshot the list because _on_scan_finished may mutate it
        # if a worker happens to finish during cancel.
        jobs = list(self._scan_jobs)
        # Bump the cancel epoch + set worker.cancel flags via the
        # shared helper — keeps clear() and shutdown() symmetric so a
        # late batch never repopulates after either action.
        self._cancel_active_scans()
        for thread, worker in jobs:
            thread.quit()
            # Generous wait: the worker checks the cancel flag once per
            # file iteration, so on a slow network share the in-flight
            # os.scandir() may take seconds to return. 5s is enough for
            # any realistic single batch; past that we drop the wait
            # and let the daemon-promoted threads die with the process.
            thread.wait(5000)
            # Queue deletion. _on_scan_finished may also fire later
            # via a queued signal; deleteLater is idempotent here
            # because Qt processes only one pending deletion per object.
            worker.deleteLater()
            thread.deleteLater()
        # Clear the registry ourselves: _on_scan_finished does that
        # too, but it runs through Qt's event loop and we don't pump
        # events during shutdown. Without this manual clear the
        # registry would still hold the (defunct) entries when the
        # test or main_window checks `_scan_jobs` post-shutdown.
        self._scan_jobs = []
        self._scan_label.setText("")

    # ---- Internal slots ----

    def _on_sort_field_changed(self, idx: int) -> None:
        # QComboBox.itemData round-trips userData through a QVariant,
        # which strips the Enum type from str-mixin enums — we get back
        # a plain `str` even though we stored a `SortField`. Coerce
        # back via SortField(value); the dropdown can't hold a value
        # that isn't a member, so the ValueError branch is paranoia.
        value = self._sort_combo.itemData(idx)
        if value is None:
            return
        try:
            field = SortField(value)
        except ValueError:
            return
        self._proxy.set_sort_field(field)
        self.sortChanged.emit()

    def _toggle_sort_direction(self) -> None:
        new_order = (
            Qt.SortOrder.DescendingOrder
            if self._proxy.sortOrder() == Qt.SortOrder.AscendingOrder
            else Qt.SortOrder.AscendingOrder
        )
        self._proxy.sort(0, new_order)
        self._sort_dir_button.setText(
            "▲" if new_order == Qt.SortOrder.AscendingOrder else "▼"
        )
        self.sortChanged.emit()

    def _on_activated(self, proxy_index) -> None:
        path_str = self._proxy.data(proxy_index, ROLE_PATH)
        if path_str:
            self.pathSelected.emit(Path(path_str))

    def _confirm_clear(self) -> None:
        # Skip the confirmation for an already-empty library — pressing
        # the button when there's nothing to clear (and no scan is in
        # flight) is a clean no-op.
        visible_count = len(self._model.paths())
        roots_count = len(self._roots)
        scanning = bool(self._scan_jobs)
        if visible_count == 0 and roots_count == 0 and not scanning:
            return
        # Show the visible grid count — that's what the user is
        # looking at and what they expect to disappear. Folder roots
        # show as "1 entry" in roots but expand to hundreds of files
        # in the grid; reporting the root count is misleading. Mention
        # the scan if one's in flight so the user knows their cancel
        # is doing double duty.
        scan_note = " (a scan is still in progress and will be cancelled)" if scanning else ""
        if visible_count == 1:
            msg = f"Remove the 1 entry from this library{scan_note}?"
        else:
            msg = f"Remove all {visible_count} entries from this library{scan_note}?"
        reply = QMessageBox.question(
            self,
            "Clear library",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.clear()

    def _open_add_files_dialog(self) -> None:
        paths_str, _ = QFileDialog.getOpenFileNames(
            self,
            "Add files to library",
            "",
            self._file_dialog_filter,
        )
        if not paths_str:
            return
        self.ingest_files_and_folders([Path(p) for p in paths_str])

    def _open_add_folder_dialog(self) -> None:
        folder_str = QFileDialog.getExistingDirectory(
            self,
            "Add folder to library",
            "",
        )
        if not folder_str:
            return
        self.ingest_files_and_folders([Path(folder_str)])

    # ---- Ctrl+Wheel resize ----

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Intercept Ctrl+Wheel on the list viewport to resize tiles.

        We install on the viewport (not the list itself) so the event
        arrives BEFORE the default scroll handling. Returning True
        consumes the event so the grid doesn't scroll at the same time
        as resizing.
        """
        if (
            event.type() == QEvent.Type.Wheel
            and watched is self._list.viewport()
        ):
            assert isinstance(event, QWheelEvent)
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # angleDelta().y() > 0 = wheel up = grow; < 0 = shrink.
                # One typical notch is 120, but trackpads emit small
                # deltas — treat any positive/negative as one step so
                # the gesture is predictable.
                delta = event.angleDelta().y()
                if delta > 0:
                    self.set_display_dim(self._display_dim + _DISPLAY_DIM_STEP)
                elif delta < 0:
                    self.set_display_dim(self._display_dim - _DISPLAY_DIM_STEP)
                return True
        return super().eventFilter(watched, event)

    # ---- Drag-drop ----

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        paths: list[Path] = []
        for url in urls:
            local = url.toLocalFile()
            if local:
                paths.append(Path(local))
        if paths:
            self.ingest_files_and_folders(paths)
            event.acceptProposedAction()
