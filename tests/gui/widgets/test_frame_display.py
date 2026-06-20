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

    def test_show_frame_on_gui_thread_is_synchronous(self, widget):
        # On the GUI thread (the live path, already hopped over by its
        # controller) the pixmap updates IMMEDIATELY — no event-loop spin —
        # so the redundant GUI→GUI queued hop is gone. No qtbot.wait here.
        widget.show_frame(_bgr(h=12, w=24))
        assert widget._pixmap is not None  # noqa: SLF001 — set synchronously
        assert widget._pixmap.width() == 24  # noqa: SLF001

    def test_subsequent_frame_replaces_previous(self, widget, qtbot):
        widget.show_frame(_bgr(h=10, w=10))
        qtbot.waitUntil(lambda: widget._pixmap is not None and widget._pixmap.width() == 10, timeout=1000)  # noqa: SLF001,E501
        widget.show_frame(_bgr(h=40, w=60))
        qtbot.waitUntil(lambda: widget._pixmap is not None and widget._pixmap.width() == 60, timeout=1000)  # noqa: SLF001,E501

    def test_non_contiguous_input_still_works(self, widget, qtbot):
        base = np.zeros((10, 10, 6), dtype=np.uint8)
        sliced = base[:, :, :3]  # non-contiguous view
        sliced[:, :, 2] = 255
        widget.show_frame(sliced)
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001
        pixel = widget._pixmap.toImage().pixelColor(5, 5)  # noqa: SLF001
        assert pixel.red() == 255


class TestRotation:
    """Rotation is display-only: the underlying pixel buffer is left
    intact (so save-current-frame and the executor both see un-rotated
    content), but paint applies a quarter-turn transform."""

    def test_initial_rotation_is_zero(self, widget):
        assert widget.rotation() == 0

    def test_set_rotation_clamps_to_valid_quarter_turns(self, widget):
        widget.set_rotation(45)  # not a quarter turn — snap to 0
        assert widget.rotation() == 0
        widget.set_rotation(90)
        assert widget.rotation() == 90
        widget.set_rotation(720)  # out of range — snap to 0
        assert widget.rotation() == 0

    def test_cycle_rotation_advances_through_quarter_turns(self, widget):
        # Cycle: 0 → 90 → 180 → 270 → 0
        seq = [widget.cycle_rotation() for _ in range(5)]
        assert seq == [90, 180, 270, 0, 90]

    def test_current_pixmap_applies_rotation(self, widget, qtbot):
        # Source frame is 20 wide × 50 tall. After 90° rotation the
        # current pixmap should be 50 wide × 20 tall.
        widget.show_frame(_bgr(h=50, w=20))
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001
        widget.set_rotation(90)
        out = widget.current_pixmap()
        assert out is not None
        assert out.width() == 50
        assert out.height() == 20

    def test_current_pixmap_none_before_first_frame(self, widget):
        # Save-current-frame should be a no-op when nothing's on screen.
        assert widget.current_pixmap() is None

    def test_rotation_does_not_mutate_source_pixmap(
        self, widget, qtbot
    ):
        # The underlying _pixmap stays at the source dimensions even
        # after rotation — only the rotated copy returned by
        # current_pixmap differs. Important so executor's pixel buffer
        # isn't accidentally aliased through the display.
        widget.show_frame(_bgr(h=50, w=20))
        qtbot.waitUntil(lambda: widget._pixmap is not None, timeout=1000)  # noqa: SLF001
        widget.set_rotation(180)
        assert widget._pixmap.width() == 20  # noqa: SLF001
        assert widget._pixmap.height() == 50  # noqa: SLF001


class TestRotatedPixmapCache:
    """The rotated render source must be cached, not re-allocated on every
    paint. paintEvent fires once per displayed frame (30-60 fps); a full-res
    SmoothTransformation rotation each time is wasteful. _rotated_source() is
    the cached seam both paintEvent and current_pixmap render through."""

    def test_rotation_zero_returns_source_pixmap(self, widget):
        widget._on_frame_ready(_bgr(h=50, w=20), 0)  # noqa: SLF001
        assert widget._rotated_source() is widget._pixmap  # noqa: SLF001

    def test_rotated_source_is_cached_across_calls(self, widget):
        widget._on_frame_ready(_bgr(h=50, w=20), 0)  # noqa: SLF001
        widget.set_rotation(90)
        # Two consecutive calls with no source/rotation change must return the
        # SAME underlying pixmap (same cacheKey) — recomputing would allocate
        # a fresh rotated pixmap each time and yield different cacheKeys.
        first = widget._rotated_source()  # noqa: SLF001
        second = widget._rotated_source()  # noqa: SLF001
        assert first.cacheKey() == second.cacheKey()

    def test_rotated_source_recomputes_on_rotation_change(self, widget):
        widget._on_frame_ready(_bgr(h=50, w=20), 0)  # noqa: SLF001
        widget.set_rotation(90)
        key_90 = widget._rotated_source().cacheKey()  # noqa: SLF001
        widget.set_rotation(180)
        assert widget._rotated_source().cacheKey() != key_90  # noqa: SLF001

    def test_rotated_source_recomputes_on_new_frame(self, widget):
        widget._on_frame_ready(_bgr(h=50, w=20), 0)  # noqa: SLF001
        widget.set_rotation(90)
        key_first = widget._rotated_source().cacheKey()  # noqa: SLF001
        widget._on_frame_ready(_bgr(h=50, w=20, value=200), 1)  # noqa: SLF001
        assert widget._rotated_source().cacheKey() != key_first  # noqa: SLF001
