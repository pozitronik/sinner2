"""Unit tests for FullscreenController in isolation (lightweight fakes for the
window, chrome widgets, transport, fullscreen bar and central layout).

The window-level integration is covered by test_main_window's Fullscreen* classes;
here we pin the controller's own contract: the redundant-call guard, the
enter/exit choreography, and the maximized-vs-normal restore decision.
"""
from __future__ import annotations

from sinner2.gui.fullscreen_controller import FullscreenController


class _Widget:
    def __init__(self, visible: bool = True) -> None:
        self._visible = visible

    def isVisible(self) -> bool:
        return self._visible

    def setVisible(self, v: bool) -> None:
        self._visible = v

    def show(self) -> None:
        self._visible = True


class _Bar:
    def __init__(self) -> None:
        self.events: list[str] = []

    def attach(self, w: object) -> None:
        self.events.append("attach")

    def detach(self, w: object) -> None:
        self.events.append("detach")

    def begin(self) -> None:
        self.events.append("begin")

    def end(self) -> None:
        self.events.append("end")


class _Layout:
    def __init__(self) -> None:
        self.ops: list[tuple] = []

    def removeWidget(self, w: object) -> None:
        self.ops.append(("remove", w))

    def insertWidget(self, i: int, w: object) -> None:
        self.ops.append(("insert", i, w))

    def indexOf(self, w: object) -> int:
        return 0  # splitter at index 0 → transport re-homes at slot 1


class _Window:
    def __init__(self, maximized: bool = False) -> None:
        self._maximized = maximized
        self.calls: list[str] = []

    def isMaximized(self) -> bool:
        return self._maximized

    def showFullScreen(self) -> None:
        self.calls.append("fs")

    def showNormal(self) -> None:
        self.calls.append("normal")

    def showMaximized(self) -> None:
        self.calls.append("max")


def _make(maximized: bool = False):
    window = _Window(maximized=maximized)
    bar, layout = _Bar(), _Layout()
    chrome = [_Widget(), _Widget()]
    transport, splitter = _Widget(), _Widget()
    ctrl = FullscreenController(
        window, fs_controls=bar, chrome=chrome, central_layout=layout,
        transport=transport, top_splitter=splitter,
    )
    return ctrl, window, bar, layout, chrome, transport


class TestGuard:
    def test_set_fullscreen_false_when_already_windowed_is_noop(self):
        ctrl, window, bar, _layout, _chrome, _t = _make()
        ctrl.set_fullscreen(False)
        assert ctrl.is_fullscreen is False
        assert window.calls == [] and bar.events == []

    def test_set_fullscreen_true_twice_enters_once(self):
        ctrl, window, _bar, _layout, _chrome, _t = _make()
        ctrl.set_fullscreen(True)
        ctrl.set_fullscreen(True)  # redundant — guarded
        assert window.calls == ["fs"]


class TestEnter:
    def test_enter_hides_chrome_moves_transport_and_goes_fullscreen(self):
        ctrl, window, bar, layout, chrome, transport = _make()
        ctrl.enter()
        assert ctrl.is_fullscreen is True
        assert all(not w.isVisible() for w in chrome)  # chrome hidden
        assert ("remove", transport) in layout.ops    # pulled from the layout
        assert bar.events == ["attach", "begin"]
        assert window.calls == ["fs"]


class TestExit:
    def test_exit_restores_chrome_visibility_and_rehomes_transport(self):
        ctrl, window, bar, layout, chrome, transport = _make()
        chrome[0].setVisible(True)
        chrome[1].setVisible(False)  # a chrome widget that was already hidden
        ctrl.enter()
        ctrl.exit()
        assert ctrl.is_fullscreen is False
        assert chrome[0].isVisible() is True   # restored to pre-fullscreen state
        assert chrome[1].isVisible() is False
        assert ("insert", 1, transport) in layout.ops  # re-homed at splitter+1
        assert bar.events[-2:] == ["end", "detach"]

    def test_exit_restores_normal_when_entered_windowed(self):
        ctrl, window, *_ = _make(maximized=False)
        ctrl.enter()
        ctrl.exit()
        assert "normal" in window.calls and "max" not in window.calls

    def test_exit_restores_maximized_when_entered_maximized(self):
        ctrl, window, *_ = _make(maximized=True)
        ctrl.enter()
        ctrl.exit()
        assert "max" in window.calls and "normal" not in window.calls
