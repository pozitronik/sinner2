"""A failed async source/target swap must not leave audio paused against a
still-playing video.

Regression: _begin_swap pauses audio for the swap window; on success
_adopt_swapped_bundle restores it. But the failure branch of
_on_session_swap_ready only emitted errorOccurred — the old session keeps
producing frames while audio stays paused (A/V desync). The fix re-syncs audio
to the still-live executor's play/seek state.
"""
from __future__ import annotations

import pytest

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.gui.player_controller import PlayerController, _SwapOutcome
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from tests.audio.test_audio_backend import FakeAudioBackend


class _Obs:
    def __init__(self, value: object) -> None:
        self._value = value

    def get(self) -> object:
        return self._value


class _FakeExecutor:
    def __init__(self, frame: int, playing: bool) -> None:
        self.current_frame = _Obs(frame)
        self.is_playing = _Obs(playing)

    def stop(self) -> None:
        pass


@pytest.fixture
def widgets(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return display, transport


@pytest.fixture
def factory():
    instances: list[FakeAudioBackend] = []

    def make(_name: AudioBackendName) -> FakeAudioBackend:
        b = FakeAudioBackend()
        instances.append(b)
        return b

    make.instances = instances  # type: ignore[attr-defined]
    return make


def _controller(widgets, factory) -> PlayerController:
    display, transport = widgets
    return PlayerController(
        frame_display=display,
        transport=transport,
        audio_backend_factory=factory,
    )


class TestSwapFailureRestoresAudio:
    def test_failed_swap_resumes_audio_for_playing_session(
        self, widgets, factory, tmp_path
    ):
        ctrl = _controller(widgets, factory)
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"x")
        backend = ctrl.audio_backend()
        assert backend is not None
        backend.load(target)
        backend.pause()  # _begin_swap paused it for the swap window
        ctrl._current_target_path = target  # noqa: SLF001
        ctrl._target_fps = 30.0  # noqa: SLF001
        # Old session is still live and PLAYING (the failed swap left it alone).
        ctrl._executor = _FakeExecutor(frame=90, playing=True)  # noqa: SLF001

        ctrl._on_session_swap_ready(_SwapOutcome(error="no face found"))  # noqa: SLF001

        assert backend.is_playing is True  # audio resumed with the video
        assert backend.position_s == pytest.approx(3.0)  # 90 / 30 fps
        ctrl._executor = None  # noqa: SLF001
        ctrl.shutdown()

    def test_failed_swap_keeps_audio_paused_for_paused_session(
        self, widgets, factory, tmp_path
    ):
        ctrl = _controller(widgets, factory)
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"x")
        backend = ctrl.audio_backend()
        assert backend is not None
        backend.load(target)
        backend.pause()
        ctrl._current_target_path = target  # noqa: SLF001
        ctrl._target_fps = 30.0  # noqa: SLF001
        ctrl._executor = _FakeExecutor(frame=0, playing=False)  # noqa: SLF001

        ctrl._on_session_swap_ready(_SwapOutcome(error="boom"))  # noqa: SLF001

        assert backend.is_playing is False  # stays paused — session is paused
        ctrl._executor = None  # noqa: SLF001
        ctrl.shutdown()
