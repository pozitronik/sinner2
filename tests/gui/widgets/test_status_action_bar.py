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
            bar.face_button,
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
            bar.face_button,
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
