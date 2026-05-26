from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QPathPicker(QWidget):
    """One labeled path slot with a Load button and drag-drop accept.

    Emits pathChanged when the path is set (via Load dialog, drag-drop, or
    set_path). Empty / None never emits — only valid paths do, so connected
    slots don't have to filter.
    """

    pathChanged = Signal(Path)

    def __init__(self, label_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._label = QLabel(label_text)
        self._display = QLineEdit()
        self._display.setReadOnly(True)
        self._load_button = QPushButton("Load…")
        self._load_button.clicked.connect(self._open_dialog)

        layout = QHBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._display, stretch=1)
        layout.addWidget(self._load_button)

        self.setAcceptDrops(True)

    def path(self) -> Path | None:
        return self._path

    def set_path(self, path: Path | None) -> None:
        self._path = path
        self._display.setText(str(path) if path is not None else "")
        if path is not None:
            self.pathChanged.emit(path)

    def _open_dialog(self) -> None:
        # Start the dialog in the directory of the currently selected path so
        # the OS native dialog (and its own Recent/MRU list) lands in the
        # right neighborhood. Falls back to default when no path is set yet.
        start_dir = str(self._path.parent) if self._path is not None else ""
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = QPathPicker("Source:")
        self._target = QPathPicker("Target:")
        self._source.pathChanged.connect(self.sourceChanged)
        self._target.pathChanged.connect(self.targetChanged)

        layout = QVBoxLayout(self)
        layout.addWidget(self._source)
        layout.addWidget(self._target)

    def source_path(self) -> Path | None:
        return self._source.path()

    def target_path(self) -> Path | None:
        return self._target.path()

    def set_source(self, path: Path | None) -> None:
        self._source.set_path(path)

    def set_target(self, path: Path | None) -> None:
        self._target.set_path(path)
