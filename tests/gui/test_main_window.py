import pytest

from sinner2.gui.main_window import SinnerMainWindow


@pytest.fixture
def window(qtbot):
    w = SinnerMainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


class TestSinnerMainWindow:
    def test_constructs_with_expected_title(self, window):
        assert window.windowTitle() == "sinner2"

    def test_has_central_widget(self, window):
        assert window.centralWidget() is not None

    def test_status_bar_has_ready_message(self, window):
        assert "ready" in window.statusBar().currentMessage()

    def test_no_executor_until_source_and_target_set(self, window):
        assert window._controller.executor() is None  # noqa: SLF001
