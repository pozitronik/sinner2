"""Unit tests for AudioController — the audio-backend helper extracted from
PlayerController (Phase 3.1). A fake backend captures the calls; no Qt, no real
audio device. Pins the lazy-vs-non-lazy construction semantics + the guarded
mirror operations.
"""
from __future__ import annotations

from pathlib import Path

from sinner2.audio.audio_backend import AudioBackendName
from sinner2.gui.audio_controller import AudioController


class _FakeBackend:
    def __init__(self) -> None:
        self.volume: float | None = None
        self.loaded = False
        self.loaded_path: Path | None = None
        self.played = False
        self.paused = False
        self.sought: float | None = None
        self.shut = False

    def set_volume(self, v: float) -> None:
        self.volume = v

    def load(self, p: Path) -> None:
        self.loaded = True
        self.loaded_path = p

    def is_loaded(self) -> bool:
        return self.loaded

    def play(self) -> None:
        self.played = True

    def pause(self) -> None:
        self.paused = True

    def seek_seconds(self, s: float) -> None:
        self.sought = s

    def shutdown(self) -> None:
        self.shut = True


def _factory_returning(*backends):
    it = iter(backends)
    return lambda _name: next(it)


# ---- lazy construction + volume replay ----

def test_ensure_backend_constructs_once_and_replays_volume():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.cache_initial_volume(40)  # no backend yet → cached, not applied
    assert b.volume is None
    assert ac.ensure_backend() is b
    assert b.volume == 0.4  # replayed on construction
    assert ac.ensure_backend() is b  # cached, not reconstructed


def test_ensure_backend_reports_error_and_returns_none_on_failure():
    errors = []

    def boom(_name):
        raise RuntimeError("no device")

    ac = AudioController(factory=boom, on_error=errors.append)
    assert ac.ensure_backend() is None
    assert ac.backend is None
    assert len(errors) == 1 and "audio backend init failed" in errors[0]


# ---- volume: lazy change vs non-lazy startup ----

def test_set_volume_lazy_constructs_and_clamps():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.set_volume(150)  # clamps to 100, constructs the backend
    assert ac.backend is b
    assert ac.volume == 100
    assert b.volume == 1.0


def test_cache_initial_volume_does_not_construct():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.cache_initial_volume(-10)  # clamps to 0, must NOT build a backend
    assert ac.backend is None
    assert ac.volume == 0


# ---- backend switching ----

def test_switch_backend_constructs_and_carries_volume_when_none():
    # Only one backend type exists today, so a "switch" exercises the construct
    # path: with no backend yet, switch_backend builds one (changed=True) and the
    # cached volume is replayed into it. (The shut-down-old branch is defensive
    # for a future second backend; the teardown is covered by test_shutdown.)
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.cache_initial_volume(50)
    changed = ac.switch_backend(AudioBackendName.QT)
    assert changed is True
    assert ac.backend is b
    assert b.volume == 0.5  # cached volume replayed into the constructed backend


def test_switch_backend_noop_when_same_name():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.ensure_backend()
    assert ac.switch_backend(ac.name) is False
    assert b.shut is False


# ---- teardown ----

def test_shutdown_clears_backend():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.ensure_backend()
    ac.shutdown()
    assert b.shut is True
    assert ac.backend is None


# ---- guarded mirror operations ----

def test_mirror_ops_noop_until_loaded():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.ensure_backend()  # constructed but not loaded
    ac.play_if_loaded()
    ac.pause_if_loaded()
    ac.seek_if_loaded(3.0)
    assert b.played is False and b.paused is False and b.sought is None
    ac.load(Path("/m.mp4"))
    assert b.loaded_path == Path("/m.mp4")
    ac.play_if_loaded()
    ac.seek_if_loaded(3.0)
    assert b.played is True and b.sought == 3.0


def test_load_noop_without_backend():
    ac = AudioController(factory=_factory_returning(_FakeBackend()))
    ac.load(Path("/m.mp4"))  # no backend ensured → no-op, no crash
    assert ac.backend is None


# ---- restore_state ----

def test_restore_state_seeks_and_plays_when_loaded():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.ensure_backend()
    ac.load(Path("/m.mp4"))
    ac.restore_state(target_fps=30.0, frame=60, playing=True)
    assert b.sought == 2.0  # 60 / 30
    assert b.played is True


def test_restore_state_pauses_when_not_playing_and_skips_seek_without_fps():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.ensure_backend()
    ac.load(Path("/m.mp4"))
    ac.restore_state(target_fps=0.0, frame=60, playing=False)
    assert b.sought is None  # fps <= 0 → no seek
    assert b.paused is True


def test_restore_state_noop_when_not_loaded():
    b = _FakeBackend()
    ac = AudioController(factory=_factory_returning(b))
    ac.ensure_backend()  # not loaded
    ac.restore_state(target_fps=30.0, frame=60, playing=True)
    assert b.sought is None and b.played is False
