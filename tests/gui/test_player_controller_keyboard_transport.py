"""The keyboard transport shortcuts (space / arrows / Home / End) must drive
the SAME audio-aware path as the transport buttons.

Regression: main_window.keyPressEvent called executor.pause()/play()/seek()
DIRECTLY, bypassing PlayerController._on_pause/_on_play/_on_seek (which keep the
audio backend in lock-step). So spacebar paused the video but left audio
playing (whenever the play button didn't have focus to consume the key), and
arrow-key seeks desynced audio. toggle_playback() / seek_to() are the public
audio-aware entry points the shortcuts now use.
"""
from __future__ import annotations

import pytest

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from tests.audio.test_audio_backend import FakeAudioBackend


class _Obs:
    def __init__(self, value: bool) -> None:
        self._value = value

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = value


class _FakeExecutor:
    def __init__(self, playing: bool) -> None:
        self.is_playing = _Obs(playing)
        self.calls: list = []

    def play(self) -> None:
        self.calls.append("play")
        self.is_playing.set(True)

    def pause(self) -> None:
        self.calls.append("pause")
        self.is_playing.set(False)

    def seek(self, frame: int) -> None:
        self.calls.append(("seek", frame))

    def stop(self) -> None:
        pass


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


def _loaded_backend(ctrl):
    from pathlib import Path

    backend = ctrl.audio_backend()
    backend.load(Path("clip.mp4"))  # is_loaded() now True
    return backend


class TestTogglePlayback:
    def test_toggle_while_playing_pauses_both(self, widgets):
        ctrl = _controller(widgets)
        backend = _loaded_backend(ctrl)
        backend.play()
        ctrl._executor = _FakeExecutor(playing=True)  # noqa: SLF001
        ctrl.toggle_playback()
        assert "pause" in ctrl._executor.calls  # noqa: SLF001  video paused
        assert backend.is_playing is False  # audio paused too — the reported bug
        ctrl._executor = None  # noqa: SLF001
        ctrl.shutdown()

    def test_toggle_while_paused_plays_both(self, widgets):
        ctrl = _controller(widgets)
        backend = _loaded_backend(ctrl)
        backend.pause()
        ctrl._executor = _FakeExecutor(playing=False)  # noqa: SLF001
        ctrl.toggle_playback()
        assert "play" in ctrl._executor.calls  # noqa: SLF001
        assert backend.is_playing is True
        ctrl._executor = None  # noqa: SLF001
        ctrl.shutdown()

    def test_toggle_no_session_is_noop(self, widgets):
        ctrl = _controller(widgets)
        ctrl.toggle_playback()  # _executor is None — must not raise
        ctrl.shutdown()


class TestSeekTo:
    def test_seek_to_seeks_audio(self, widgets):
        ctrl = _controller(widgets)
        backend = _loaded_backend(ctrl)
        ctrl._executor = _FakeExecutor(playing=False)  # noqa: SLF001
        ctrl._target_fps = 30.0  # noqa: SLF001
        ctrl.seek_to(90)
        assert ("seek", 90) in ctrl._executor.calls  # noqa: SLF001
        assert backend.position_s == pytest.approx(3.0)  # 90 / 30 fps
        ctrl._executor = None  # noqa: SLF001
        ctrl.shutdown()
