import pytest

from sinner2.gui.widgets.processor_controls import QProcessorControls


@pytest.fixture
def widget(qtbot):
    w = QProcessorControls()
    qtbot.addWidget(w)
    return w


class TestQProcessorControls:
    def test_default_swapper_params(self, widget):
        params = widget.swapper_params()
        assert params.detection_interval == 1
        assert params.many_faces is True

    def test_default_enhancer_params(self, widget):
        params = widget.enhancer_params()
        assert params.upscale == 1
        assert params.only_center_face is False

    def test_default_enhancer_enabled(self, widget):
        assert widget.enhancer_enabled() is True

    def test_changing_detection_interval_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._detection_interval.setValue(5)  # noqa: SLF001
        assert widget.swapper_params().detection_interval == 5

    def test_toggling_many_faces_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._many_faces.setChecked(False)  # noqa: SLF001
        assert widget.swapper_params().many_faces is False

    def test_changing_upscale_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._upscale.setValue(2)  # noqa: SLF001
        assert widget.enhancer_params().upscale == 2

    def test_toggling_center_face_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._only_center_face.setChecked(True)  # noqa: SLF001
        assert widget.enhancer_params().only_center_face is True

    def test_toggling_enhancer_group_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._enhancer_box.setChecked(False)  # noqa: SLF001
        assert widget.enhancer_enabled() is False

    def test_default_strategy_is_best_effort(self, widget):
        from sinner2.pipeline.skip_strategy import BestEffortStrategy

        assert isinstance(widget.skip_strategy(), BestEffortStrategy)

    def test_changing_strategy_emits_config_changed(self, widget, qtbot):
        with qtbot.waitSignal(widget.configChanged, timeout=1000):
            widget._strategy_combo.setCurrentIndex(1)  # noqa: SLF001
        from sinner2.pipeline.skip_strategy import SyncedStrategy

        assert isinstance(widget.skip_strategy(), SyncedStrategy)
