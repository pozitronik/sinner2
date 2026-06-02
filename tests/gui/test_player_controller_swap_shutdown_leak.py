"""Shutdown racing a completing swap must not leak the new session resources.

Regression: on a successful swap the worker thread swaps the new world into the
live executor and emits a queued _sessionSwapReady; the GUI slot adopts the new
write_executor + store into the controller's refs. shutdown() joins the worker
and runs _teardown_session BEFORE that queued slot fires (the loop isn't spun
during close), so it tears down the OLD (already-shut-down) refs and the NEW
bundle's threads + file handles leak. The fix drains the unconsumed bundle.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from tests.audio.test_audio_backend import FakeAudioBackend


class _FakeWriteExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[bool] = []

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


class _FakeStore:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def widgets(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return display, transport


def _controller(widgets) -> PlayerController:
    display, transport = widgets

    def factory(_name: AudioBackendName) -> FakeAudioBackend:
        return FakeAudioBackend()

    return PlayerController(
        frame_display=display,
        transport=transport,
        audio_backend_factory=factory,
    )


class TestShutdownDrainsUnconsumedSwap:
    def test_unconsumed_bundle_is_released_on_shutdown(self, widgets):
        ctrl = _controller(widgets)
        we = _FakeWriteExecutor()
        store = _FakeStore()
        # Simulate a swap that completed on the worker but whose GUI slot never
        # ran (shutdown raced it): the new bundle is stashed, unconsumed.
        ctrl._last_swap_bundle = SimpleNamespace(  # noqa: SLF001
            write_executor=we, session_store=store
        )

        ctrl.shutdown()

        assert we.shutdown_calls == [True]  # new write executor torn down
        assert store.closed == 1  # new store closed

    def test_no_stashed_bundle_is_a_noop(self, widgets):
        ctrl = _controller(widgets)
        assert ctrl._last_swap_bundle is None  # noqa: SLF001
        ctrl.shutdown()  # must not raise
