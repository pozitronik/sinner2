"""Audio backend Protocol + factory.

Sinner2's player owns an AudioBackend that plays the target video's audio
track in sync with the (processed) frames the worker is producing. The
controller drives the backend via play/pause/seek matching the same
events the executor handles — sync is "video-master" today: the Timeline
ticks wall-clock and audio plays at 1×, so both stay aligned as long as
they started from the same position at the same instant.

The Protocol is intentionally tiny so a second backend (pygame.mixer
fed by ffmpeg-extracted PCM, python-vlc bound to the file, etc.) can be
added without touching the controller. Pick the backend via
`AudioBackendName` in settings; the factory below dispatches.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class AudioBackendName(str, Enum):
    """Identifier persisted in settings so the user can pick a backend."""

    QT = "qt"


@runtime_checkable
class AudioBackend(Protocol):
    """Plays the audio track of a single media file with play/pause/seek/volume.

    Lifecycle: construct → load(path) on each new session → play/pause/seek
    as the user interacts → shutdown() at app exit. load() may be called
    repeatedly to switch media; the implementation reuses underlying
    decoder/output objects when possible.

    Threading: all calls happen on the GUI thread. Backends that need
    cross-thread work (e.g. running their own decoder) must marshal
    internally; the controller never spawns threads on the backend's behalf.
    """

    def load(self, media_path: Path) -> None:
        """Switch to this media file. Stops any current playback. No-op
        when the path equals the currently loaded one. Implementations
        may detect "no audio track" lazily; callers should not assume
        has_audio() is truthy until at least one play() attempt."""

    def reload(self) -> None:
        """Re-point at the currently-loaded media, re-arming any deferred
        play/seek. Unlike load(), does NOT skip when the path is unchanged.
        Used after an async session swap so a resume runs through the same
        load→ready path a media switch uses (a bare play() to resume a
        just-paused player is unreliable on some backends). No-op if nothing
        is loaded."""

    def play(self) -> None: ...

    def pause(self) -> None: ...

    def seek_seconds(self, seconds: float) -> None:
        """Set position in seconds. Bounded to [0, duration] internally."""

    def set_volume(self, volume: float) -> None:
        """Linear 0.0-1.0. Caller is responsible for any perceptual curve."""

    def set_muted(self, muted: bool) -> None: ...

    def is_loaded(self) -> bool:
        """True after a successful load() and before any teardown."""

    def has_audio(self) -> bool:
        """False for image targets, videos without an audio track, or
        when the implementation hasn't probed yet. Used by the UI to
        gray out audio controls."""

    def shutdown(self) -> None:
        """Release any underlying resources. Idempotent."""


def build_audio_backend(name: AudioBackendName) -> AudioBackend:
    """Construct a backend by name. Adding a new backend means:
       1. implementing the AudioBackend Protocol,
       2. adding an entry to AudioBackendName,
       3. dispatching here.
    The controller picks one up by name and is otherwise backend-agnostic."""
    if name is AudioBackendName.QT:
        # Lazy import so absence of PySide6 in unit-test environments
        # doesn't break factory imports.
        from sinner2.audio.qt_audio_backend import QtMediaAudioBackend

        return QtMediaAudioBackend()
    raise ValueError(f"unknown audio backend: {name!r}")
