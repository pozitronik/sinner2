import pytest

from sinner2.gui.widgets.transport_controls import QTransportControls


@pytest.fixture
def widget(qtbot):
    w = QTransportControls()
    qtbot.addWidget(w)
    return w


class TestAudioControls:
    def test_initial_volume_full(self, widget):
        assert widget.volume() == 100

    def test_initial_not_muted(self, widget):
        assert widget.muted() is False

    def test_volume_slider_emits_signal(self, widget, qtbot):
        with qtbot.waitSignal(widget.volumeChanged, timeout=1000) as blocker:
            widget._volume.setValue(50)  # noqa: SLF001
        assert blocker.args == [50]

    def test_mute_checkbox_emits_signal(self, widget, qtbot):
        with qtbot.waitSignal(widget.mutedChanged, timeout=1000) as blocker:
            widget._mute.setChecked(True)  # noqa: SLF001
        assert blocker.args == [True]

    def test_silent_setters_do_not_emit(self, widget, qtbot):
        # Used during startup restore — pushing a persisted value into
        # the widget must NOT round-trip back out as a "user change."
        with qtbot.assertNotEmitted(widget.volumeChanged, wait=100):
            widget.set_volume_silently(33)
        assert widget.volume() == 33
        with qtbot.assertNotEmitted(widget.mutedChanged, wait=100):
            widget.set_muted_silently(True)
        assert widget.muted() is True

    def test_set_audio_enabled_toggles_controls(self, widget):
        widget.set_audio_enabled(False)
        assert not widget._mute.isEnabled()  # noqa: SLF001
        assert not widget._volume.isEnabled()  # noqa: SLF001
        widget.set_audio_enabled(True)
        assert widget._mute.isEnabled()  # noqa: SLF001
        assert widget._volume.isEnabled()  # noqa: SLF001

    def test_silent_volume_clamps_out_of_range(self, widget):
        widget.set_volume_silently(-50)
        assert widget.volume() == 0
        widget.set_volume_silently(999)
        assert widget.volume() == 100
