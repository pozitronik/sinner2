import pytest

from sinner2.gui.session_capabilities import SessionCapabilities
from sinner2.gui.widgets.transport_controls import QTransportControls


@pytest.fixture
def widget(qtbot):
    w = QTransportControls()
    qtbot.addWidget(w)
    return w


class TestApplyCapabilities:
    def test_file_caps_enable_seek_label_audio_batch(self, widget):
        widget.apply_capabilities(SessionCapabilities.for_file(has_audio=True))
        assert widget._play_button.isEnabled()        # noqa: SLF001
        assert widget._slider.isEnabled()             # noqa: SLF001
        assert widget._label.isVisibleTo(widget)      # noqa: SLF001
        assert widget._volume.isEnabled()             # noqa: SLF001
        assert widget._add_to_batch.isEnabled()       # noqa: SLF001

    def test_file_without_audio_disables_volume_only(self, widget):
        widget.apply_capabilities(SessionCapabilities.for_file(has_audio=False))
        assert widget._slider.isEnabled()             # noqa: SLF001
        assert not widget._volume.isEnabled()         # noqa: SLF001

    def test_camera_caps_disable_seek_label_audio_batch_keep_play(self, widget):
        widget.apply_capabilities(SessionCapabilities.for_camera())
        assert widget._play_button.isEnabled()        # noqa: SLF001 stop/start
        assert not widget._slider.isEnabled()         # noqa: SLF001 no seek
        assert not widget._label.isVisibleTo(widget)  # noqa: SLF001 no frames
        assert not widget._volume.isEnabled()         # noqa: SLF001 no audio
        assert not widget._add_to_batch.isEnabled()   # noqa: SLF001 file-only

    def test_none_caps_disable_everything(self, widget):
        widget.apply_capabilities(SessionCapabilities.none())
        assert not widget._play_button.isEnabled()    # noqa: SLF001
        assert not widget._slider.isEnabled()         # noqa: SLF001
        assert not widget._volume.isEnabled()         # noqa: SLF001
        assert not widget._add_to_batch.isEnabled()   # noqa: SLF001


class TestTransportControls:
    def test_initial_button_label(self, widget):
        assert widget._play_button.text() == "Play"  # noqa: SLF001

    def test_initial_counter_label(self, widget):
        assert widget._label.text() == "0 / 0"  # noqa: SLF001

    def test_set_frame_count_updates_slider_and_label(self, widget):
        widget.set_frame_count(100)
        assert widget._slider.maximum() == 99  # noqa: SLF001
        assert widget._label.text() == "0 / 99"  # noqa: SLF001

    def test_set_current_frame_updates_label_only(self, widget):
        widget.set_frame_count(100)
        widget.set_current_frame(42)
        assert widget._label.text() == "42 / 99"  # noqa: SLF001
        assert widget._slider.value() == 42  # noqa: SLF001


class TestTimeReadout:
    def test_frames_only_without_fps(self, widget):
        widget.set_frame_count(100)
        widget.set_current_frame(42)
        assert widget._label.text() == "42 / 99"  # noqa: SLF001 — no fps → frames only

    def test_time_prefix_added_with_fps(self, widget):
        widget.set_frame_count(2700)
        widget.set_fps(30.0)
        widget.set_current_frame(360)
        text = widget._label.text()  # noqa: SLF001
        assert "0:12" in text  # 360 / 30 = 12s
        assert "360 / 2699" in text  # frame position preserved

    def test_set_fps_zero_reverts_to_frames_only(self, widget):
        widget.set_frame_count(2700)
        widget.set_fps(30.0)
        widget.set_current_frame(360)
        widget.set_fps(0.0)
        assert widget._label.text() == "360 / 2699"  # noqa: SLF001

    def test_fmt_time_minutes_and_hours(self):
        from sinner2.gui.widgets.transport_controls import QTransportControls

        assert QTransportControls._fmt_time(12) == "0:12"  # noqa: SLF001
        assert QTransportControls._fmt_time(90) == "1:30"  # noqa: SLF001
        assert QTransportControls._fmt_time(3725) == "1:02:05"  # noqa: SLF001

    def test_set_is_playing_changes_button_label(self, widget):
        widget.set_is_playing(True)
        assert widget._play_button.text() == "Pause"  # noqa: SLF001
        widget.set_is_playing(False)
        assert widget._play_button.text() == "Play"  # noqa: SLF001

    def test_play_click_emits_play_when_stopped(self, widget, qtbot):
        with qtbot.waitSignal(widget.playRequested, timeout=1000):
            widget._play_button.click()  # noqa: SLF001

    def test_play_click_emits_pause_when_playing(self, widget, qtbot):
        widget.set_is_playing(True)
        with qtbot.waitSignal(widget.pauseRequested, timeout=1000):
            widget._play_button.click()  # noqa: SLF001

    def test_slider_release_emits_seek_with_value(self, widget, qtbot):
        widget.set_frame_count(100)
        widget._slider.setValue(50)  # noqa: SLF001
        with qtbot.waitSignal(widget.seekRequested, timeout=1000) as blocker:
            widget._slider.sliderReleased.emit()  # noqa: SLF001
        assert blocker.args == [50]

    def test_slider_drag_emits_seek_after_debounce(self, widget, qtbot):
        widget.set_frame_count(100)
        with qtbot.waitSignal(widget.seekRequested, timeout=1000) as blocker:
            widget._slider.sliderMoved.emit(33)  # noqa: SLF001
        assert blocker.args == [33]

    def test_slider_drag_coalesces_rapid_moves(self, widget, qtbot):
        widget.set_frame_count(100)
        with qtbot.waitSignal(widget.seekRequested, timeout=1000) as blocker:
            widget._slider.sliderMoved.emit(10)  # noqa: SLF001
            widget._slider.sliderMoved.emit(20)  # noqa: SLF001
            widget._slider.sliderMoved.emit(30)  # noqa: SLF001
        # Only the last emitted value should reach seekRequested.
        assert blocker.args == [30]

    def test_set_current_frame_does_not_emit_seek(self, widget, qtbot):
        widget.set_frame_count(100)
        with qtbot.assertNotEmitted(widget.seekRequested, wait=100):
            widget.set_current_frame(50)

    def test_set_frame_count_does_not_emit_seek(self, widget, qtbot):
        with qtbot.assertNotEmitted(widget.seekRequested, wait=100):
            widget.set_frame_count(100)



class TestAddToBatch:
    def test_follows_parent_enabled_state(self, widget):
        # The button rides the transport's enabled state (the main window
        # disables the whole transport until a source + target are loaded).
        widget.setEnabled(False)
        assert widget._add_to_batch.isEnabled() is False  # noqa: SLF001
        widget.setEnabled(True)
        assert widget._add_to_batch.isEnabled() is True  # noqa: SLF001

    def test_click_emits_request(self, widget, qtbot):
        with qtbot.waitSignal(widget.addToBatchRequested, timeout=1000):
            widget._add_to_batch.click()  # noqa: SLF001


class TestSectionEditing:
    """[ / ] section state machine: mark in/out → commit, select-and-nudge,
    delete, clear, and the sectionsChanged signal."""

    def test_mark_in_sets_pending_without_committing(self, widget):
        widget.mark_in(50)
        assert widget.pending_in() == 50
        assert widget.sections().is_empty()

    def test_mark_in_out_commits_a_section(self, widget, qtbot):
        widget.mark_in(50)
        with qtbot.waitSignal(widget.sectionsChanged) as blocker:
            widget.mark_out(120)
        from sinner2.pipeline.sections import SectionSet

        assert blocker.args[0] == SectionSet.of([(50, 120)])
        assert widget.sections().ranges == ((50, 120),)
        # Pending cleared; selection NOT set (so next [ starts a new section).
        assert widget.pending_in() is None
        assert widget.selected_index() is None

    def test_multiple_sections(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        widget.mark_in(180)
        widget.mark_out(240)
        assert widget.sections().ranges == ((50, 120), (180, 240))

    def test_remark_in_moves_pending(self, widget):
        widget.mark_in(50)
        widget.mark_in(60)  # no selection → just moves the in-point
        assert widget.pending_in() == 60
        widget.mark_out(120)
        assert widget.sections().ranges == ((60, 120),)

    def test_mark_out_without_in_is_noop(self, widget):
        widget.mark_out(120)
        assert widget.sections().is_empty()

    def test_select_then_mark_in_nudges_start(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        widget.mark_in(180)
        widget.mark_out(240)
        # Select section 1 (the [180,240] band) by landing the playhead in it.
        widget._update_selection_to(200)  # noqa: SLF001
        assert widget.selected_index() == 1
        widget.mark_in(175)  # nudge its start 180 → 175
        assert widget.sections().ranges == ((50, 120), (175, 240))

    def test_select_then_mark_out_nudges_end(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        widget._update_selection_to(80)  # select section 0
        widget.mark_out(110)  # nudge end 120 → 110
        assert widget.sections().ranges == ((50, 110),)

    def test_delete_selected(self, widget, qtbot):
        widget.mark_in(50)
        widget.mark_out(120)
        widget.mark_in(180)
        widget.mark_out(240)
        widget._update_selection_to(60)  # select section 0
        with qtbot.waitSignal(widget.sectionsChanged):
            widget.delete_selected()
        assert widget.sections().ranges == ((180, 240),)
        assert widget.selected_index() is None

    def test_delete_without_selection_is_noop(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        widget.delete_selected()  # nothing selected
        assert widget.sections().ranges == ((50, 120),)

    def test_clear_sections(self, widget, qtbot):
        widget.mark_in(50)
        widget.mark_out(120)
        with qtbot.waitSignal(widget.sectionsChanged) as blocker:
            widget.clear_sections()
        assert widget.sections().is_empty()
        assert blocker.args[0].is_empty()

    def test_set_sections_does_not_emit(self, widget):
        from sinner2.pipeline.sections import SectionSet

        fired = []
        widget.sectionsChanged.connect(lambda s: fired.append(s))
        widget.set_sections(SectionSet.of([(10, 20)]))
        assert widget.sections().ranges == ((10, 20),)
        assert fired == []  # restore path is silent
        assert widget.pending_in() is None

    def test_nudge_can_merge_bands(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        widget.mark_in(180)
        widget.mark_out(240)
        widget._update_selection_to(200)  # select section 1
        widget.mark_in(100)  # pull its start into section 0 → they merge
        assert widget.sections().ranges == ((50, 240),)
        # Selection re-resolved onto the merged band.
        assert widget.selected_index() == 0


class TestSectionMenuAndCancel:
    def test_cancel_pending_clears_marker(self, widget):
        widget.mark_in(50)
        assert widget.pending_in() == 50
        widget.cancel_pending()
        assert widget.pending_in() is None

    def test_cancel_pending_without_pending_is_noop(self, widget):
        widget.cancel_pending()  # must not raise
        assert widget.pending_in() is None

    def test_menu_mark_in_out_at_clicked_frame(self, widget):
        # The menu targets the CLICKED frame, not the playhead.
        widget._menu_mark_in(40)   # noqa: SLF001
        widget._menu_mark_out(95)  # noqa: SLF001
        assert widget.sections().ranges == ((40, 95),)

    def test_menu_mark_in_on_band_nudges_it(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        # Right-click inside the band and "set in-point here" → nudge start.
        widget._menu_mark_in(60)  # noqa: SLF001
        assert widget.sections().ranges == ((60, 120),)

    def test_menu_remove_section_under_cursor(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        widget.mark_in(180)
        widget.mark_out(240)
        widget._menu_remove(200)  # noqa: SLF001 — inside the second band
        assert widget.sections().ranges == ((50, 120),)


class TestSliderKeyIgnore:
    def test_slider_ignores_arrow_keys(self, widget):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        ev = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Left,
            Qt.KeyboardModifier.NoModifier,
        )
        widget._slider.keyPressEvent(ev)  # noqa: SLF001
        # Ignored → bubbles up to the main window's frame-step handler.
        assert not ev.isAccepted()


class TestSectionPainting:
    def test_overlay_pushed_to_slider(self, widget):
        widget.mark_in(50)
        widget.mark_out(120)
        # The slider received the band for painting.
        assert widget._slider._ranges == [(50, 120)]  # noqa: SLF001

    def test_paints_without_crash(self, widget, qtbot):
        widget.set_frame_count(300)
        widget.mark_in(50)
        widget.mark_out(120)
        widget.show()
        qtbot.waitExposed(widget)
        widget._slider.repaint()  # noqa: SLF001 — exercises paintEvent


class TestProcessingVisualiser:
    def test_bar_hidden_by_default(self, widget):
        assert not widget.visualiser_visible()

    def test_toggle_visibility(self, widget):
        widget.set_visualiser_visible(True)
        assert widget.visualiser_visible()
        widget.set_visualiser_visible(False)
        assert not widget.visualiser_visible()

    def test_set_frame_states_feeds_the_bar(self, widget):
        from sinner2.pipeline.realtime.frame_state import FrameState

        states = bytes([int(FrameState.READY_MEM)] * 50)
        widget.set_frame_states(states, 50)
        assert widget._frame_state_bar._frame_count == 50  # noqa: SLF001
        assert widget._frame_state_bar._states == states   # noqa: SLF001

    def test_current_frame_drives_bar_playhead(self, widget):
        widget.set_frame_count(100)
        widget.set_current_frame(42)
        assert widget._frame_state_bar._playhead == 42      # noqa: SLF001

    def test_new_target_clears_the_bar(self, widget):
        from sinner2.pipeline.realtime.frame_state import FrameState

        widget.set_frame_states(bytes([int(FrameState.READY_MEM)] * 10), 10)
        widget.set_frame_count(200)  # new target
        assert widget._frame_state_bar._frame_count == 0    # noqa: SLF001

    def test_bar_click_forwards_as_seek(self, widget, qtbot):
        from PySide6.QtCore import QPoint, Qt

        from sinner2.pipeline.realtime.frame_state import FrameState

        widget.set_visualiser_visible(True)
        widget.set_frame_states(bytes([int(FrameState.READY_MEM)] * 200), 200)
        bar = widget._frame_state_bar  # noqa: SLF001
        bar.resize(100, 16)
        with qtbot.waitSignal(widget.seekRequested, timeout=1000) as blocker:
            qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(50, 8))
        assert blocker.args[0] == 100  # halfway → frame 100 of 200
