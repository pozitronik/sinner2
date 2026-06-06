"""Audio-backend concern for the realtime player.

Owns the audio backend lifecycle (lazy construction, switch, shutdown), the
cached volume, and the guarded backend operations the session + transport mirror
to it (load / play / pause / seek). Extracted from PlayerController (Phase 3.1)
so the controller no longer carries the backend plumbing inline.

The session-restore state (which frame, playing or not, the target fps) is NOT
owned here — that's session state shared with the swap path. The controller
gathers it from the live executor and passes it into `restore_state`, so this
helper stays free of executor/session knowledge.

Lazy vs non-lazy is preserved exactly from the original: a volume *change*
(`set_volume`) constructs the backend if needed (so the slider always reaches a
backend), but the startup `cache_initial_volume` does not (no backend exists yet
on first launch). `restore_state` and the `*_if_loaded` mirrors operate on the
already-constructed backend only.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sinner2.audio.audio_backend import (
    AudioBackend,
    AudioBackendName,
    build_audio_backend,
)


class AudioController:
    """The audio backend + volume, with guarded mirror operations. Qt-free;
    construction failures are reported through the `on_error` callback."""

    def __init__(
        self,
        factory: Callable[[AudioBackendName], AudioBackend] | None = None,
        on_error: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        self._factory = factory or build_audio_backend
        self._on_error = on_error
        # Constructed lazily — some backends (QtMultimedia) need a QApplication
        # to exist first, and the controller may be built before that.
        self._backend: AudioBackend | None = None
        self._name: AudioBackendName = AudioBackendName.QT
        self._volume: int = 100

    @property
    def backend(self) -> AudioBackend | None:
        """The current backend WITHOUT constructing one (None until ensured)."""
        return self._backend

    @property
    def name(self) -> AudioBackendName:
        return self._name

    @property
    def volume(self) -> int:
        return self._volume

    def ensure_backend(self) -> AudioBackend | None:
        """Lazy accessor — constructs on first request so the QApplication
        exists by then. Returns None if construction failed (reported via
        on_error). Replays the cached volume so the backend matches the UI state
        the user set before any media loaded."""
        if self._backend is None:
            try:
                self._backend = self._factory(self._name)
                self._backend.set_volume(self._volume / 100.0)
            except Exception as exc:
                self._on_error(f"audio backend init failed: {exc}")
                self._backend = None
        return self._backend

    def switch_backend(self, name: AudioBackendName) -> bool:
        """Swap to a different backend: shut the old one down and reconstruct so
        the new one picks up the cached volume. Returns True if it actually
        changed (the caller should then reload the current media into it); a
        no-op (and False) when the name is unchanged and a backend exists."""
        if name is self._name and self._backend is not None:
            return False
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None
        self._name = name
        self.ensure_backend()
        return True

    def shutdown(self) -> None:
        """Tear the backend down (session teardown). Safe when none exists."""
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None

    def set_volume(self, value: int) -> None:
        """Cache + apply a volume change (0-100). Constructs the backend if
        needed so the slider always reaches one."""
        self._volume = max(0, min(100, value))
        backend = self.ensure_backend()
        if backend is not None:
            backend.set_volume(self._volume / 100.0)

    def cache_initial_volume(self, value: int) -> None:
        """Cache the persisted startup volume WITHOUT constructing a backend
        (there's none on first launch); apply immediately if one already exists."""
        self._volume = max(0, min(100, value))
        if self._backend is not None:
            self._backend.set_volume(self._volume / 100.0)

    def load(self, target_path: Path) -> None:
        """Load media into the current backend (no-op when none exists)."""
        if self._backend is not None:
            self._backend.load(target_path)

    def play_if_loaded(self) -> None:
        if self._backend is not None and self._backend.is_loaded():
            self._backend.play()

    def pause_if_loaded(self) -> None:
        if self._backend is not None and self._backend.is_loaded():
            self._backend.pause()

    def seek_if_loaded(self, seconds: float) -> None:
        if self._backend is not None and self._backend.is_loaded():
            self._backend.seek_seconds(seconds)

    def restore_state(self, target_fps: float, frame: int, playing: bool) -> None:
        """Re-point the backend at a restored session position + play state
        (after a swap/reconfigure). No-op until the backend has media loaded;
        QtMediaAudioBackend arms a pending seek/play if the codec isn't ready, so
        issuing these immediately is safe — they apply on LoadedMedia."""
        backend = self._backend
        if backend is None or not backend.is_loaded():
            return
        if target_fps > 0:
            backend.seek_seconds(frame / target_fps)
        if playing:
            backend.play()
        else:
            backend.pause()
