
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

    def test_set_target_visible_toggles_only_target(self, panel):
        panel.set_target_visible(False)
        assert not panel._target.isVisibleTo(panel)  # noqa: SLF001
        assert panel._source.isVisibleTo(panel)      # noqa: SLF001 — source stays
        panel.set_target_visible(True)
        assert panel._target.isVisibleTo(panel)      # noqa: SLF001

    def test_set_source_enabled_locks_only_source(self, panel):
        panel.set_source_enabled(False)
        assert panel._source.isEnabled() is False    # noqa: SLF001 — locked
        assert panel._target.isEnabled() is True     # noqa: SLF001 — target stays
        panel.set_source_enabled(True)
        assert panel._source.isEnabled() is True      # noqa: SLF001

    def test_use_camera_button_emits_camera_requested(self, panel, qtbot):
        with qtbot.waitSignal(panel.cameraRequested, timeout=1000):
            panel._use_camera.click()  # noqa: SLF001

    def test_source_and_target_edits_are_equal_width(self, panel, qtbot):
        # The camera button on the target row would otherwise make the target
        # edit narrower; the source Load button is widened to compensate so both
        # path edits render the same width.
        panel.resize(400, 80)
        panel.show()
        qtbot.waitExposed(panel)
        src_edit = panel._source._display  # noqa: SLF001
        tgt_edit = panel._target._display  # noqa: SLF001
        assert abs(src_edit.width() - tgt_edit.width()) <= 1

    def test_source_load_button_wider_than_target_load(self, panel, qtbot):
        panel.resize(400, 80)
        panel.show()
        qtbot.waitExposed(panel)
        src_load = panel._source._load_button  # noqa: SLF001
        tgt_load = panel._target._load_button  # noqa: SLF001
        # Source spends the camera's space on its Load button → strictly wider.
        assert src_load.width() > tgt_load.width()


class TestQPathPickerRecents:
    @pytest.fixture
    def picker(self, qtbot):
        p = QPathPicker("Test:")
        qtbot.addWidget(p)
        return p

    def test_initial_recents_empty(self, picker):
        assert picker.recents() == []

    def test_set_path_pushes_into_recents(self, picker, tmp_path):
        a = tmp_path / "a.png"
        picker.set_path(a)
        assert picker.recents() == [a]

    def test_recents_most_recent_first(self, picker, tmp_path):
        a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
        picker.set_path(a)
        picker.set_path(b)
        picker.set_path(c)
        assert picker.recents() == [c, b, a]

    def test_recents_dedupe(self, picker, tmp_path):
        # Re-selecting the same file shouldn't duplicate it — moves to top.
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        picker.set_path(a)
        picker.set_path(b)
        picker.set_path(a)
        assert picker.recents() == [a, b]

    def test_recents_capped(self, picker, tmp_path):
        # Cap is _RECENTS_MAX (10). Adding more rolls out the oldest.
        for i in range(15):
            picker.set_path(tmp_path / f"f{i:02d}.png")
        recents = picker.recents()
        assert len(recents) == 10
        # Most recent should be the last one added.
        assert recents[0] == tmp_path / "f14.png"
        # Oldest in recents should be the 5th-from-last added.
        assert recents[-1] == tmp_path / "f05.png"

    def test_set_path_emits_recents_changed(self, picker, qtbot, tmp_path):
        a = tmp_path / "a.png"
        with qtbot.waitSignal(picker.recentsChanged, timeout=1000) as blocker:
            picker.set_path(a)
        assert blocker.args == [[a]]

    def test_same_top_path_does_not_re_emit(self, picker, qtbot, tmp_path):
        a = tmp_path / "a.png"
        picker.set_path(a)
        # Re-setting the same top entry — pathChanged still fires (the
        # caller may have refreshed) but recents didn't change, so no
        # recentsChanged emission.
        with qtbot.assertNotEmitted(picker.recentsChanged, wait=100):
            picker.set_path(a)

    def test_set_recents_does_not_emit(self, picker, qtbot, tmp_path):
        # Restore path: applies persisted list silently — emitting
        # would round-trip through main_window and re-save the same
        # value, plus risk feedback loops.
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        with qtbot.assertNotEmitted(picker.recentsChanged, wait=100):
            picker.set_recents([a, b])
        assert picker.recents() == [a, b]

    def test_set_recents_dedupes_and_caps(self, picker, tmp_path):
        # Bad persisted lists (with duplicates or oversize) should be
        # normalised on restore.
        paths = [tmp_path / f"f{i}.png" for i in range(15)]
        # Inject duplicates.
        picker.set_recents([paths[0], paths[0], *paths])
        recents = picker.recents()
        assert len(recents) == 10
        assert len(set(str(p) for p in recents)) == 10  # all distinct

    def test_clear_recents(self, picker, qtbot, tmp_path):
        picker.set_path(tmp_path / "a.png")
        picker.set_path(tmp_path / "b.png")
        with qtbot.waitSignal(picker.recentsChanged, timeout=1000) as blocker:
            picker.clear_recents()
        assert blocker.args == [[]]
        assert picker.recents() == []

    def test_clear_recents_no_op_when_empty(self, picker, qtbot):
        with qtbot.assertNotEmitted(picker.recentsChanged, wait=100):
            picker.clear_recents()


class TestQSourceTargetPanelRecents:
    @pytest.fixture
    def panel(self, qtbot):
        p = QSourceTargetPanel()
        qtbot.addWidget(p)
        return p

    def test_source_recents_isolated_from_target(self, panel, tmp_path):
        panel.set_source(tmp_path / "src.png")
        assert panel.source_recents() == [tmp_path / "src.png"]
        assert panel.target_recents() == []

    def test_source_recents_changed_signal(self, panel, qtbot, tmp_path):
        with qtbot.waitSignal(panel.sourceRecentsChanged, timeout=1000) as blocker:
            panel.set_source(tmp_path / "x.png")
        assert blocker.args == [[tmp_path / "x.png"]]

    def test_target_recents_changed_signal(self, panel, qtbot, tmp_path):
        with qtbot.waitSignal(panel.targetRecentsChanged, timeout=1000) as blocker:
            panel.set_target(tmp_path / "v.mp4")
        assert blocker.args == [[tmp_path / "v.mp4"]]

    def test_set_recents_round_trip(self, panel, tmp_path):
        panel.set_source_recents([tmp_path / "a.png", tmp_path / "b.png"])
        panel.set_target_recents([tmp_path / "x.mp4"])
        assert panel.source_recents() == [tmp_path / "a.png", tmp_path / "b.png"]
        assert panel.target_recents() == [tmp_path / "x.mp4"]
