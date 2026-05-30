"""Tests for the AudioBackend protocol surface + factory.

Per-backend behavior tests (e.g. QtMediaAudioBackend playing actual
audio) belong in manual QA — too fragile to test reliably across
machines without real media + audio hardware. These tests pin the
contract any backend must satisfy via a fake implementation."""

from pathlib import Path

import pytest

from sinner2.audio.audio_backend import (
    AudioBackend,
    build_audio_backend,
)


class FakeAudioBackend:
    """Records every call. Useful for testing wiring in the controller
    without depending on a real audio device or QApplication."""

    def __init__(self) -> None:
        self.loaded: Path | None = None
        self.is_playing = False
        self.position_s = 0.0
        self.volume = 1.0
        self.muted = False
        self.shut_down = False
        self.calls: list[tuple[str, object]] = []

    def load(self, media_path: Path) -> None:
        self.loaded = media_path
        self.calls.append(("load", media_path))

    def play(self) -> None:
        self.is_playing = True
        self.calls.append(("play", None))

    def pause(self) -> None:
        self.is_playing = False
        self.calls.append(("pause", None))

    def seek_seconds(self, seconds: float) -> None:
        self.position_s = max(0.0, seconds)
        self.calls.append(("seek", seconds))

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))
        self.calls.append(("set_volume", volume))

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)
        self.calls.append(("set_muted", muted))

    def is_loaded(self) -> bool:
        return self.loaded is not None

    def has_audio(self) -> bool:
        return self.loaded is not None

    def shutdown(self) -> None:
        self.shut_down = True
        self.calls.append(("shutdown", None))


class TestProtocolConformance:
    def test_fake_backend_conforms(self):
        # If the runtime_checkable protocol is well-specified, a fake
        # that implements every method satisfies isinstance.
        assert isinstance(FakeAudioBackend(), AudioBackend)


class TestFactoryDispatch:
    def test_unknown_backend_raises(self):
        # AudioBackendName is an Enum; passing something not in it fails.
        class NotABackend:
            pass

        with pytest.raises(ValueError):
            build_audio_backend(NotABackend())  # type: ignore[arg-type]


class TestFakeBackendBehavior:
    """Smoke tests for the FakeAudioBackend itself — used by other
    tests in this package, so it needs to be obviously correct."""

    def test_initial_state(self):
        b = FakeAudioBackend()
        assert not b.is_loaded()
        assert not b.has_audio()
        assert b.volume == 1.0
        assert not b.muted

    def test_load_then_play(self):
        b = FakeAudioBackend()
        b.load(Path("/x.mp4"))
        b.play()
        assert b.is_loaded()
        assert b.is_playing

    def test_volume_clamped(self):
        b = FakeAudioBackend()
        b.set_volume(-0.5)
        assert b.volume == 0.0
        b.set_volume(2.0)
        assert b.volume == 1.0

    def test_records_call_order(self):
        b = FakeAudioBackend()
        b.load(Path("/x"))
        b.play()
        b.pause()
        b.seek_seconds(3.5)
        names = [c[0] for c in b.calls]
        assert names == ["load", "play", "pause", "seek"]
