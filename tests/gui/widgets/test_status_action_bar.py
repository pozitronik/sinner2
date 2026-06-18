import pytest
from PySide6.QtWidgets import QLabel

from sinner2.gui.widgets.status_action_bar import QStatusActionBar


@pytest.fixture
def bar(qtbot):
    b = QStatusActionBar()
    qtbot.addWidget(b)
    return b


class TestStatusActionBar:
    def test_toggle_buttons_are_checkable(self, bar):
        for button in (
            bar.on_top_button,
            bar.stats_button,
            bar.visualiser_button,
            bar.fullscreen_button,
            bar.side_panel_button,
        ):
            assert button.isCheckable()

    def test_action_buttons_are_not_checkable(self, bar):
        assert not bar.rotate_button.isCheckable()
        assert not bar.save_button.isCheckable()

    def test_every_button_has_a_tooltip(self, bar):
        for button in (
            bar.on_top_button,
            bar.stats_button,
            bar.visualiser_button,
            bar.rotate_button,
            bar.fullscreen_button,
            bar.side_panel_button,
            bar.save_button,
        ):
            assert button.toolTip()

    def test_show_message_is_persistent_by_default(self, bar, qtbot):
        bar.show_message("hello")
        assert bar.current_message() == "hello"
        qtbot.wait(60)
        assert bar.current_message() == "hello"  # no timeout → stays

    def test_timed_message_clears_itself(self, bar, qtbot):
        bar.show_message("temp", 50)
        assert bar.current_message() == "temp"
        qtbot.waitUntil(lambda: bar.current_message() == "", timeout=2000)

    def test_add_permanent_widget_appends(self, bar):
        before = bar._layout.count()  # noqa: SLF001
        bar.add_permanent_widget(QLabel("indicator"))
        assert bar._layout.count() == before + 1  # noqa: SLF001


class TestStatusPanels:
    def test_add_panel_appends_and_returns_panel(self, bar):
        before = bar._layout.count()  # noqa: SLF001
        panel = bar.add_panel("⏱", "throughput", min_width=60)
        assert bar._layout.count() == before + 1  # noqa: SLF001
        assert panel.toolTip() == "throughput"

    def test_panel_hidden_until_it_has_a_value(self, bar, qtbot):
        panel = bar.add_panel("⏱", "throughput")
        bar.show()
        qtbot.waitExposed(bar)
        assert not panel.isVisible()  # empty → hidden (no blank cell)
        assert panel.value() == ""

    def test_set_value_shows_panel_with_icon_and_value(self, bar, qtbot):
        panel = bar.add_panel("⏱", "throughput")
        bar.show()
        qtbot.waitExposed(bar)
        panel.set_value("30.0 fps")
        assert panel.isVisible()
        assert panel.value() == "30.0 fps"  # value() drops the icon prefix
        assert "30.0 fps" in panel._value.text()  # noqa: SLF001
        assert "⏱" in panel._value.text()  # noqa: SLF001 — icon prefixed

    def test_clearing_value_hides_panel_again(self, bar, qtbot):
        panel = bar.add_panel("⏱", "throughput")
        bar.show()
        qtbot.waitExposed(bar)
        panel.set_value("30.0 fps")
        assert panel.isVisible()
        panel.set_value("")
        assert not panel.isVisible()
        assert panel.value() == ""

    def test_panel_without_icon_shows_bare_value(self, bar):
        panel = bar.add_panel(tooltip="plain")
        panel.set_value("hello")
        assert panel._value.text() == "hello"  # noqa: SLF001 — no icon prefix


class TestPanelVisibilityMenu:
    def test_only_keyed_panels_are_registered(self, bar):
        bar.add_panel("⏱", "tip", key="fps", label="FPS")
        bar.add_panel("▦", "tip")  # no key → not menu-toggleable
        assert [p.key() for p in bar._panels] == ["fps"]  # noqa: SLF001

    def test_user_hidden_panel_stays_hidden_with_value(self, bar, qtbot):
        panel = bar.add_panel("⏱", "tip", key="fps", label="FPS")
        bar.show()
        qtbot.waitExposed(bar)
        panel.set_value("30 fps")
        assert panel.isVisible()
        bar.set_panel_user_visible("fps", False)
        assert not panel.isVisible()  # hidden despite a live value
        assert panel.value() == "30 fps"  # value retained while hidden
        bar.set_panel_user_visible("fps", True)
        assert panel.isVisible()

    def test_user_visible_panel_still_hidden_without_value(self, bar, qtbot):
        panel = bar.add_panel("⏱", "tip", key="fps", label="FPS")
        bar.show()
        qtbot.waitExposed(bar)
        bar.set_panel_user_visible("fps", True)
        assert not panel.isVisible()  # no value → still hidden

    def test_hidden_panel_keys_reflects_toggles(self, bar):
        bar.add_panel("a", key="a", label="A")
        bar.add_panel("b", key="b", label="B")
        assert bar.hidden_panel_keys() == []
        bar.set_panel_user_visible("a", False)
        assert bar.hidden_panel_keys() == ["a"]

    def test_toggle_panel_applies_and_emits(self, bar, qtbot):
        panel = bar.add_panel("⏱", "tip", key="fps", label="FPS")
        panel.set_value("x")
        with qtbot.waitSignal(bar.panelVisibilityChanged) as blocker:
            bar._toggle_panel(panel, False)  # noqa: SLF001
        assert blocker.args == ["fps", False]
        assert panel.user_visible() is False

    def test_set_panel_user_visible_unknown_key_is_noop(self, bar):
        bar.add_panel("⏱", key="fps", label="FPS")
        bar.set_panel_user_visible("nope", False)  # must not raise
        assert bar.hidden_panel_keys() == []
