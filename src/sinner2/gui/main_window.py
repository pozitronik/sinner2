from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.gui.widgets.source_target_panel import QSourceTargetPanel
from sinner2.gui.widgets.transport_controls import QTransportControls


class SinnerMainWindow(QMainWindow):
    """The player surface: frame display on top, transport, then source/target.

    All real work lives on PlayerController; this class is layout, keyboard
    shortcuts, and error dialogs. Closing the window tears down the player.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("sinner2")
        self.resize(960, 720)

        self._display = QFrameDisplayWidget()
        self._transport = QTransportControls()
        self._pickers = QSourceTargetPanel()
        self._processors = QProcessorControls()

        central = QWidget()
        top = QHBoxLayout()
        top.addWidget(self._display, stretch=1)
        top.addWidget(self._processors)
        layout = QVBoxLayout(central)
        layout.addLayout(top, stretch=1)
        layout.addWidget(self._transport)
        layout.addWidget(self._pickers)
        self.setCentralWidget(central)

        self.statusBar().showMessage("ready")

        self._controller = PlayerController(self._display, self._transport, parent=self)
        self._controller.errorOccurred.connect(self._show_error)

        self._pickers.sourceChanged.connect(self._reload_player)
        self._pickers.targetChanged.connect(self._reload_player)
        self._processors.configChanged.connect(self._on_processor_config_changed)
        # Seed the controller with the widget defaults so the first session uses them.
        self._on_processor_config_changed()

    def _reload_player(self, _path: Path) -> None:
        self._controller.set_source_and_target(
            self._pickers.source_path(), self._pickers.target_path()
        )

    def _on_processor_config_changed(self) -> None:
        self._controller.apply_chain_config(
            swapper_params=self._processors.swapper_params(),
            enhancer_params=self._processors.enhancer_params(),
            enhancer_enabled=self._processors.enhancer_enabled(),
        )

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        QMessageBox.critical(self, "sinner2", message)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        executor = self._controller.executor()
        key = event.key()
        if executor is None:
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_Space:
            if executor.is_playing.get():
                executor.pause()
            else:
                executor.play()
            return
        if key == Qt.Key.Key_Left:
            executor.seek(max(0, executor.current_frame.get() - 1))
            return
        if key == Qt.Key.Key_Right:
            executor.seek(executor.current_frame.get() + 1)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._controller.shutdown()
        super().closeEvent(event)
