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
