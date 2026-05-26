from pathlib import Path

import pytest

from sinner2.gui.widgets.source_target_panel import QPathPicker, QSourceTargetPanel


class TestQPathPicker:
    @pytest.fixture
    def picker(self, qtbot):
        p = QPathPicker("Test:")
        qtbot.addWidget(p)
        return p

    def test_initial_path_is_none(self, picker):
        assert picker.path() is None

    def test_set_path_updates_display(self, picker, tmp_path):
        picker.set_path(tmp_path / "x.png")
        assert picker._display.text() == str(tmp_path / "x.png")  # noqa: SLF001

    def test_set_path_emits_path_changed(self, picker, qtbot, tmp_path):
        p = tmp_path / "x.png"
        with qtbot.waitSignal(picker.pathChanged, timeout=1000) as blocker:
            picker.set_path(p)
        assert blocker.args == [p]

    def test_set_none_does_not_emit(self, picker, qtbot):
        with qtbot.assertNotEmitted(picker.pathChanged, wait=100):
            picker.set_path(None)

    def test_set_none_clears_display(self, picker, tmp_path):
        picker.set_path(tmp_path / "x.png")
        picker.set_path(None)
        assert picker._display.text() == ""  # noqa: SLF001


class TestQSourceTargetPanel:
    @pytest.fixture
    def panel(self, qtbot):
        p = QSourceTargetPanel()
        qtbot.addWidget(p)
        return p

    def test_initial_paths_are_none(self, panel):
        assert panel.source_path() is None
        assert panel.target_path() is None

    def test_set_source_propagates(self, panel, qtbot, tmp_path):
        p = tmp_path / "face.png"
        with qtbot.waitSignal(panel.sourceChanged, timeout=1000) as blocker:
            panel.set_source(p)
        assert blocker.args == [p]
        assert panel.source_path() == p

    def test_set_target_propagates(self, panel, qtbot, tmp_path):
        p = tmp_path / "video.mp4"
        with qtbot.waitSignal(panel.targetChanged, timeout=1000) as blocker:
            panel.set_target(p)
        assert blocker.args == [p]
        assert panel.target_path() == p

    def test_source_change_does_not_affect_target(self, panel, qtbot, tmp_path):
        with qtbot.assertNotEmitted(panel.targetChanged, wait=100):
            panel.set_source(tmp_path / "x.png")
