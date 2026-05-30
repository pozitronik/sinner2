from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QWidget,
)

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
    volumeChanged = Signal(int)  # 0-100
    addToBatchRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_playing = False
        self._frame_count = 0
        # While the user is dragging the slider, we ignore programmatic
        # set_current_frame updates from the playback observable. Otherwise
        # the 30 Hz playback tick yanks the slider back to the play head
        # mid-drag and the user can't scrub.
        self._user_dragging = False

        self._play_button = QPushButton("Play")
        self._play_button.clicked.connect(self._on_play_clicked)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
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

        # "Add to batch": capture the current source + target + settings as a
        # batch task. Disabled (driven by the main window) until a source AND
        # target are loaded. Sits where the Mute toggle used to — mute is
        # redundant with dragging the volume slider to zero.
        self._add_to_batch = QToolButton()
        # A green "+" reads as an add control among the playback widgets. Plain
        # text + stylesheet (not the ➕ emoji) so the green is guaranteed
        # regardless of the platform's emoji font; greyed when disabled so the
        # green doubles as a "source + target loaded, ready to add" cue.
        self._add_to_batch.setText("+")
        self._add_to_batch.setStyleSheet(
            "QToolButton { color: #2e9e3f; font-weight: bold; font-size: 16px; }"
            "QToolButton:disabled { color: #888888; }"
        )
        self._add_to_batch.setToolTip(
            "Add to batch: save the current source + target + settings as a "
            "task. Edit / run it from the Batch tab."
        )
        self._add_to_batch.setEnabled(False)
        self._add_to_batch.clicked.connect(self.addToBatchRequested)

        # Audio volume (0 = silent, which replaces the old mute toggle).
        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setRange(0, 100)
        self._volume.setValue(100)
        self._volume.setFixedWidth(100)
        self._volume.setToolTip("Audio volume (0 = silent). Affects only the playback path.")
        self._volume.valueChanged.connect(self.volumeChanged)

        layout = QHBoxLayout(self)
        # Zero horizontal margins so the controls line up with the display
        # above; tight vertical margins keep the row short.
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self._play_button)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)
        layout.addWidget(self._add_to_batch)
        layout.addWidget(self._volume)

    def set_frame_count(self, count: int) -> None:
        self._frame_count = max(0, count)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, count - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._update_label(0)

    @Slot(int)
    def set_current_frame(self, frame: int) -> None:
        if self._user_dragging:
            return
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

    def _on_slider_pressed(self) -> None:
        self._user_dragging = True

    def _on_slider_moved(self, value: int) -> None:
        # Update label live so user sees the target frame while dragging.
        self._update_label(value)
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
        self._user_dragging = False
        self.seekRequested.emit(self._slider.value())

    def _update_label(self, frame: int) -> None:
        self._label.setText(f"{frame} / {max(0, self._frame_count - 1)}")

    # ---- Audio control state ----

    def volume(self) -> int:
        return self._volume.value()

    def set_volume_silently(self, value: int) -> None:
        """Apply a persisted volume without re-emitting volumeChanged.
        Used during startup restore so the controller isn't notified of
        a "change" that isn't really one."""
        self._volume.blockSignals(True)
        self._volume.setValue(max(0, min(100, value)))
        self._volume.blockSignals(False)

    def set_audio_enabled(self, enabled: bool) -> None:
        """Enable/disable the volume control. Called after each new session
        loads, based on whether the target has an audio track."""
        self._volume.setEnabled(enabled)

    def set_add_to_batch_enabled(self, enabled: bool) -> None:
        """Enable the Add-to-batch button. The main window drives this from
        the source/target picker state (both must be loaded)."""
        self._add_to_batch.setEnabled(enabled)
