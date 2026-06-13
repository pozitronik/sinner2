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
