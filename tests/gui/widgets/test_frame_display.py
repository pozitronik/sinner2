import threading

import numpy as np
import pytest

from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.types import Frame


def _bgr(h: int = 10, w: int = 10, value: int = 128) -> Frame:
    return np.full((h, w, 3), value, dtype=np.uint8)


@pytest.fixture
def widget(qtbot):
    w = QFrameDisplayWidget()
    qtbot.addWidget(w)
    return w


class TestQFrameDisplayWidget:
    def test_initial_pixmap_is_none(self, widget):
        assert widget._pixmap is None  # noqa: SLF001

    def test_show_frame_eventually_sets_pixmap(self, widget, qtbot):
        widget.show_frame(_bgr())
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001

    def test_pixmap_dimensions_match_frame(self, widget, qtbot):
        widget.show_frame(_bgr(h=50, w=80))
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001
        assert widget._pixmap.width() == 80  # noqa: SLF001
        assert widget._pixmap.height() == 50  # noqa: SLF001

    def test_bgr_red_renders_as_red(self, widget, qtbot):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # R channel (BGR index 2)
        widget.show_frame(frame)
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001
        image = widget._pixmap.toImage()  # noqa: SLF001
        pixel = image.pixelColor(5, 5)
        assert pixel.red() == 255
        assert pixel.green() == 0
        assert pixel.blue() == 0

    def test_show_frame_from_worker_thread(self, widget, qtbot):
        def worker() -> None:
            widget.show_frame(_bgr(h=20, w=30, value=200))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001
        assert widget._pixmap.width() == 30  # noqa: SLF001
        assert widget._pixmap.height() == 20  # noqa: SLF001

    def test_subsequent_frame_replaces_previous(self, widget, qtbot):
        widget.show_frame(_bgr(h=10, w=10))
        qtbot.waitUntil(lambda: widget._pixmap is not None and widget._pixmap.width() == 10, timeout=1000)  # noqa: SLF001
        widget.show_frame(_bgr(h=40, w=60))
        qtbot.waitUntil(lambda: widget._pixmap is not None and widget._pixmap.width() == 60, timeout=1000)  # noqa: SLF001

    def test_non_contiguous_input_still_works(self, widget, qtbot):
        base = np.zeros((10, 10, 6), dtype=np.uint8)
        sliced = base[:, :, :3]  # non-contiguous view
        sliced[:, :, 2] = 255
        widget.show_frame(sliced)
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001
        pixel = widget._pixmap.toImage().pixelColor(5, 5)  # noqa: SLF001
        assert pixel.red() == 255
