from PySide6.QtCore import QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QWidget,
)

from sinner2.gui.session_capabilities import SessionCapabilities, SessionKind
from sinner2.pipeline.sections import SectionSet

_SEEK_DEBOUNCE_MS = 100

# Section band colours: a calm green for ordinary sections, an amber for the
# selected one (the one [ / ] edit) and the pending in-point marker.
_SECTION_COLOR = QColor(70, 160, 90, 170)
_SECTION_SELECTED_COLOR = QColor(240, 180, 60, 220)
_PENDING_COLOR = QColor(240, 180, 60)


class _SectionSlider(QSlider):
    """Scrub slider that paints the selected timeline sections as bands over its
    groove, plus a marker for a half-finished (in-point set, no out-point yet)
    section. Pure view: the section STATE lives on QTransportControls, which
    pushes it here via set_overlay()."""

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self._ranges: list[tuple[int, int]] = []
        self._selected: int | None = None
        self._pending: int | None = None

    def set_overlay(
        self,
        ranges: list[tuple[int, int]],
        selected: int | None,
        pending: int | None,
    ) -> None:
        self._ranges = ranges
        self._selected = selected
        self._pending = pending
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self.maximum() <= self.minimum():
            return  # no usable range to map frames onto
        if not self._ranges and self._pending is None:
            return
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderGroove, self,
        )
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderHandle, self,
        )
        span = groove.width() - handle.width()
        x0 = groove.x() + handle.width() / 2.0

        def px(frame: int) -> float:
            return x0 + QStyle.sliderPositionFromValue(
                self.minimum(), self.maximum(), frame, span
            )

        painter = QPainter(self)
        band_h = 6
        band_top = groove.center().y() - band_h // 2 + 1
        for i, (start, end) in enumerate(self._ranges):
            xa, xb = px(start), px(end)
            color = (
                _SECTION_SELECTED_COLOR if i == self._selected else _SECTION_COLOR
            )
            painter.fillRect(
                QRectF(xa, band_top, max(2.0, xb - xa), band_h), color
            )
        if self._pending is not None:
            xp = px(self._pending)
            painter.fillRect(
                QRectF(xp - 1.0, groove.y(), 2.0, groove.height()), _PENDING_COLOR
            )
        painter.end()


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
    sectionsChanged = Signal(object)  # emits the new SectionSet

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_playing = False
        self._frame_count = 0
        # Target fps for the time readout; 0 → no timeline fps known, so the
        # label shows frames only (e.g. a camera, or before a session loads).
        self._fps = 0.0
        # While the user is dragging the slider, we ignore programmatic
        # set_current_frame updates from the playback observable. Otherwise
        # the 30 Hz playback tick yanks the slider back to the play head
        # mid-drag and the user can't scrub.
        self._user_dragging = False

        # Section selection state. The SectionSet is the committed set of
        # included ranges; _pending_in is a started-but-not-closed in-point;
        # _selected_index is the band [ / ] currently edit (None = none, so [
        # starts a new section instead of nudging one).
        self._sections = SectionSet.empty()
        self._pending_in: int | None = None
        self._selected_index: int | None = None

        self._play_button = QPushButton("Play")
        self._play_button.clicked.connect(self._on_play_clicked)

        self._slider = _SectionSlider(Qt.Orientation.Horizontal)
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
        self._label.setMinimumWidth(150)

        # "Add to batch": capture the current source + target + settings as a
        # batch task. Sits at the FAR LEFT of the row. The whole transport is
        # disabled by the main window until a source AND target are loaded, so
        # this button rides the parent's enabled state (greyed → green).
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
        layout.addWidget(self._add_to_batch)
        layout.addWidget(self._play_button)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._label)
        layout.addWidget(self._volume)

    def set_frame_count(self, count: int) -> None:
        self._frame_count = max(0, count)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, count - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._update_label(0)

    def set_fps(self, fps: float) -> None:
        """Target frame rate for the time readout. 0 leaves the label showing
        frames only (no timeline fps, e.g. a camera or before a session)."""
        self._fps = max(0.0, float(fps))
        self._update_label(self._slider.value())

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
        # Highlight the band the playhead is over so the user can see which
        # section [ / ] will edit before releasing.
        self._update_selection_to(value)
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
        # Clicking/landing on a band selects it (so [ / ] nudge that band);
        # landing in a gap clears the selection (so [ starts a new section).
        self._update_selection_to(self._slider.value())
        self.seekRequested.emit(self._slider.value())

    # ---- Section selection ([ / ] editing) ----

    def sections(self) -> SectionSet:
        return self._sections

    def selected_index(self) -> int | None:
        return self._selected_index

    def pending_in(self) -> int | None:
        return self._pending_in

    def set_sections(self, sections: SectionSet) -> None:
        """Reflect an externally-set selection (restore / clear) WITHOUT emitting
        — clears any in-progress in-point and selection."""
        self._sections = sections
        self._pending_in = None
        self._selected_index = None
        self._refresh_section_overlay()

    def mark_in(self, frame: int) -> None:
        """`[` — set a section's START to ``frame``. With a band selected, nudge
        that band's start (1-frame precise); otherwise begin a new section by
        marking the in-point (closed later by `]`)."""
        if self._selected_index is not None and self._selected_index < len(
            self._sections.ranges
        ):
            _, end = self._sections.ranges[self._selected_index]
            self._sections = self._sections.with_range_replaced(
                self._selected_index, frame, end
            )
            self._reselect_at(frame)
            self._refresh_section_overlay()
            self._emit_sections()
        else:
            self._pending_in = frame
            self._refresh_section_overlay()

    def mark_out(self, frame: int) -> None:
        """`]` — set a section's END to ``frame``. With a band selected, nudge
        that band's end; otherwise commit the pending in-point → a new section.
        No-op when neither a band nor an in-point is active."""
        if self._selected_index is not None and self._selected_index < len(
            self._sections.ranges
        ):
            start, _ = self._sections.ranges[self._selected_index]
            self._sections = self._sections.with_range_replaced(
                self._selected_index, start, frame
            )
            self._reselect_at(frame)
            self._refresh_section_overlay()
            self._emit_sections()
        elif self._pending_in is not None:
            self._sections = self._sections.with_added(self._pending_in, frame)
            self._pending_in = None
            # Cleared, NOT selected: the next [ starts a fresh section rather
            # than nudging the one just committed.
            self._selected_index = None
            self._refresh_section_overlay()
            self._emit_sections()

    def delete_selected(self) -> None:
        """Remove the selected band (Delete). No-op when nothing is selected."""
        if self._selected_index is None:
            return
        self._sections = self._sections.without_index(self._selected_index)
        self._selected_index = None
        self._refresh_section_overlay()
        self._emit_sections()

    def clear_sections(self) -> None:
        """Drop all sections (back to whole-timeline playback)."""
        if self._sections.is_empty() and self._pending_in is None:
            return
        self._sections = SectionSet.empty()
        self._pending_in = None
        self._selected_index = None
        self._refresh_section_overlay()
        self._emit_sections()

    def _reselect_at(self, frame: int) -> None:
        """Re-resolve the selection by frame after an edit re-normalized the
        ranges (a nudge can merge bands, shifting indices)."""
        self._selected_index = self._sections.index_at(frame)

    def _update_selection_to(self, frame: int) -> None:
        idx = self._sections.index_at(frame)
        if idx != self._selected_index:
            self._selected_index = idx
            self._refresh_section_overlay()

    def _refresh_section_overlay(self) -> None:
        self._slider.set_overlay(
            list(self._sections.ranges), self._selected_index, self._pending_in
        )

    def _emit_sections(self) -> None:
        self.sectionsChanged.emit(self._sections)

    def _update_label(self, frame: int) -> None:
        last = max(0, self._frame_count - 1)
        if self._fps > 0:
            # Time prefix when the timeline fps is known; frames stay as the
            # precise position. "0:12 / 1:30   360 / 2699".
            self._label.setText(
                f"{self._fmt_time(frame / self._fps)} / "
                f"{self._fmt_time(last / self._fps)}   {frame} / {last}"
            )
        else:
            self._label.setText(f"{frame} / {last}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

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

    # ---- Capability-driven gating ----

    def apply_capabilities(self, caps: SessionCapabilities) -> None:
        """Gate each control by the active target's capabilities. A file target
        is seekable/finite/maybe-audio; a camera target is none of those but
        still play/pause-able (stop/start); NONE disables everything."""
        self._play_button.setEnabled(caps.can_play_pause)
        self._slider.setEnabled(caps.seekable)
        self._label.setVisible(caps.has_timeline)
        self.set_audio_enabled(caps.has_audio)
        self._add_to_batch.setEnabled(caps.kind is SessionKind.FILE)
