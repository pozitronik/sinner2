from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

_RECENTS_MAX = 10


class QPathPicker(QWidget):
    """One labeled path slot with a Load split-button and drag-drop accept.

    Emits pathChanged when the path is set (via Load dialog, recent menu,
    drag-drop, or set_path). Empty / None never emits — only valid paths
    do, so connected slots don't have to filter.

    Recents: each successful set_path prepends the path to a bounded
    list (deduped, capped at _RECENTS_MAX). The Load button is a
    QToolButton in MenuButtonPopup mode — main click opens the file
    dialog; the arrow shows a menu of recent paths plus a Clear entry.
    recentsChanged fires whenever the list mutates so the caller can
    persist it.
    """

    pathChanged = Signal(Path)
    recentsChanged = Signal(list)  # list[Path]

    def __init__(self, label_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._recents: list[Path] = []
        self._label = QLabel(label_text)
        self._display = QLineEdit()
        self._display.setReadOnly(True)

        self._load_button = QToolButton()
        self._load_button.setText("Load…")
        # MenuButtonPopup: split button — main area click runs the
        # default action (Browse), the small arrow shows the menu.
        self._load_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._load_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._load_button.clicked.connect(self._open_dialog)
        self._recent_menu = QMenu(self._load_button)
        self._load_button.setMenu(self._recent_menu)
        self._rebuild_recent_menu()

        layout = QHBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._display, stretch=1)
        layout.addWidget(self._load_button)

        self.setAcceptDrops(True)

    def path(self) -> Path | None:
        return self._path

    def recents(self) -> list[Path]:
        return list(self._recents)

    def set_path(self, path: Path | None) -> None:
        self._path = path
        self._display.setText(str(path) if path is not None else "")
        if path is not None:
            self._push_recent(path)
            self.pathChanged.emit(path)

    def set_recents(self, paths: list[Path]) -> None:
        """Replace the recents list (used by startup restore). Does NOT
        emit recentsChanged — restore shouldn't trigger another persist
        of the same value we just loaded."""
        self._recents = []
        seen: set[str] = set()
        for p in paths:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            self._recents.append(p)
            if len(self._recents) >= _RECENTS_MAX:
                break
        self._rebuild_recent_menu()

    def clear_recents(self) -> None:
        if not self._recents:
            return
        self._recents = []
        self._rebuild_recent_menu()
        self.recentsChanged.emit([])

    def _push_recent(self, path: Path) -> None:
        # Most-recent-first. Dedupe by string equality (case-sensitive
        # in Python — fine on Linux; on Windows two different cases of
        # the same path resolve to the same file but show as separate
        # entries here. Good enough for v1; can normalize later if it
        # becomes noisy.)
        path_str = str(path)
        new_list = [path] + [p for p in self._recents if str(p) != path_str]
        new_list = new_list[:_RECENTS_MAX]
        if [str(p) for p in new_list] == [str(p) for p in self._recents]:
            # Path was already the top entry — no change worth signaling.
            return
        self._recents = new_list
        self._rebuild_recent_menu()
        self.recentsChanged.emit(list(self._recents))

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        if self._recents:
            for path in self._recents:
                action = QAction(str(path), self._recent_menu)
                # Lambda captures path via default arg so each action
                # binds its own path (not the last loop value).
                action.triggered.connect(lambda _checked=False, p=path: self.set_path(p))
                self._recent_menu.addAction(action)
            self._recent_menu.addSeparator()
            clear_action = QAction("Clear recents", self._recent_menu)
            clear_action.triggered.connect(self.clear_recents)
            self._recent_menu.addAction(clear_action)
        else:
            empty_action = QAction("(no recent files)", self._recent_menu)
            empty_action.setEnabled(False)
            self._recent_menu.addAction(empty_action)

    def _open_dialog(self) -> None:
        # Start the dialog in the directory of the currently selected
        # path so the OS native dialog (and its own Recent/MRU list)
        # lands in the right neighborhood. Falls back to the most-recent
        # entry's parent, then to default.
        if self._path is not None:
            start_dir = str(self._path.parent)
        elif self._recents:
            start_dir = str(self._recents[0].parent)
        else:
            start_dir = ""
        path_str, _ = QFileDialog.getOpenFileName(self, "Select file", start_dir)
        if path_str:
            self.set_path(Path(path_str))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.set_path(Path(local))
            event.acceptProposedAction()


class QSourceTargetPanel(QWidget):
    """Composes Source and Target pickers stacked vertically."""

    sourceChanged = Signal(Path)
    targetChanged = Signal(Path)
    sourceRecentsChanged = Signal(list)  # list[Path]
    targetRecentsChanged = Signal(list)
    addToBatchRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = QPathPicker("Source:")
        self._target = QPathPicker("Target:")
        self._source.pathChanged.connect(self.sourceChanged)
        self._target.pathChanged.connect(self.targetChanged)
        self._source.recentsChanged.connect(self.sourceRecentsChanged)
        self._target.recentsChanged.connect(self.targetRecentsChanged)

        # "Add to batch" lives next to the picker row — the natural
        # spot once the user has loaded a source + target and tuned
        # settings via the Settings tab. Disabled when source or
        # target is unset (no point in batching a half-configured task).
        self._add_to_batch = QToolButton()
        self._add_to_batch.setText("Add to batch")
        self._add_to_batch.setToolTip(
            "Save the current source + target + settings as a batch "
            "task. Edit / run from the Batch tab."
        )
        self._add_to_batch.setEnabled(False)
        self._add_to_batch.clicked.connect(self.addToBatchRequested)
        self._source.pathChanged.connect(self._update_add_to_batch_enabled)
        self._target.pathChanged.connect(self._update_add_to_batch_enabled)

        layout = QVBoxLayout(self)
        layout.addWidget(self._source)
        layout.addWidget(self._target)
        # Right-aligned button row below the pickers.
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._add_to_batch)
        layout.addLayout(button_row)

    def _update_add_to_batch_enabled(self, _path: Path) -> None:
        # Both must be set for the batch task to have meaning.
        ok = self.source_path() is not None and self.target_path() is not None
        self._add_to_batch.setEnabled(ok)

    def source_path(self) -> Path | None:
        return self._source.path()

    def target_path(self) -> Path | None:
        return self._target.path()

    def set_source(self, path: Path | None) -> None:
        self._source.set_path(path)

    def set_target(self, path: Path | None) -> None:
        self._target.set_path(path)

    def source_recents(self) -> list[Path]:
        return self._source.recents()

    def target_recents(self) -> list[Path]:
        return self._target.recents()

    def set_source_recents(self, paths: list[Path]) -> None:
        self._source.set_recents(paths)

    def set_target_recents(self, paths: list[Path]) -> None:
        self._target.set_recents(paths)
