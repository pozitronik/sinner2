from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

_SEEK_DEBOUNCE_MS = 100


class QTransportControls(QWidget):
    """Play/pause toggle + scrub slider + frame counter.

    Emits signals for user-initiated actions; the main window wires those to
    `executor.play() / pause() / seek()`. State is reflected via setter slots
    that observable bridges call. Slider value changes from set_current_frame
    are signal-blocked so reflecting the executor's position never loops back
    out as a seek request.
    """

    playRequested = Signal()
    pauseRequested = Signal()
    seekRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_playing = False
        self._frame_count = 0

        self._play_button = QPushButton("Play")
        self._play_button.clicked.connect(self._on_play_clicked)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)
        # Debounce rapid drag events. Without this, each slider tick during
        # a drag fires a seek that drains the worker queue, so the worker
        # never finishes any frame mid-drag.
        self._seek_debounce = QTimer(self)
        self._seek_debounce.setSingleShot(True)
        self._seek_debounce.setInterval(_SEEK_DEBOUNCE_MS)
        self._seek_debounce.timeout.connect(self._fire_debounced_seek)
        self._pending_seek: int | None = None

        self._label = QLabel("0 / 0")
        self._label.setMinimumWidth(80)

        layout = QHBoxLayout(self)
        layout.addWidget(self._play_button)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)

    def set_frame_count(self, count: int) -> None:
        self._frame_count = max(0, count)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, count - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._update_label(0)

    @Slot(int)
    def set_current_frame(self, frame: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._update_label(frame)

    @Slot(bool)
    def set_is_playing(self, is_playing: bool) -> None:
        self._is_playing = bool(is_playing)
        self._play_button.setText("Pause" if self._is_playing else "Play")

    def _on_play_clicked(self) -> None:
        if self._is_playing:
            self.pauseRequested.emit()
        else:
            self.playRequested.emit()

    def _on_slider_moved(self, value: int) -> None:
        # Coalesce rapid drag updates — the debounce restarts on each tick.
        self._pending_seek = value
        self._seek_debounce.start()

    def _fire_debounced_seek(self) -> None:
        if self._pending_seek is not None:
            self.seekRequested.emit(self._pending_seek)
            self._pending_seek = None

    def _on_slider_released(self) -> None:
        # Always fire the final position on release, even if a debounce is
        # in flight, so the user's intended position lands deterministically.
        self._seek_debounce.stop()
        self._pending_seek = None
        self.seekRequested.emit(self._slider.value())

    def _update_label(self, frame: int) -> None:
        self._label.setText(f"{frame} / {max(0, self._frame_count - 1)}")
