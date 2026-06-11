"""Unit tests for QtMediaAudioBackend's deferred play / seek logic.

QMediaPlayer.setSource is asynchronous: between setSource() and the
mediaStatusChanged(LoadedMedia) callback, play() and setPosition() are
silently dropped on Windows Media Foundation. This was the root cause
of "audio disappears on target switch while playing" — change_target
loads new media then immediately calls play(), and the play landed
in the dead zone.

These tests exercise the backend's state machine WITHOUT loading real
media (which would need a hardware audio device and platform-specific
codecs). We construct the backend, prime the internal player, then
directly invoke _on_media_status_changed to simulate the async load
finishing.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtMultimedia import QMediaPlayer

from sinner2.audio.qt_audio_backend import QtMediaAudioBackend


@pytest.fixture
def backend(qtbot) -> QtMediaAudioBackend:
    b = QtMediaAudioBackend()
    yield b
    b.shutdown()


def _prime_player(backend: QtMediaAudioBackend) -> MagicMock:
    """Construct the underlying player (lazy) and replace it with a
    mock so we can assert which Qt calls were made without launching
    the real Media Foundation pipeline."""
    backend._ensure_player()  # noqa: SLF001 — touches the private builder
    fake = MagicMock()
    backend._player = fake  # noqa: SLF001
    return fake


class TestInvalidMedia:
    def test_invalid_media_clears_loaded_path_so_reload_works(self, backend):
        # A broken file → InvalidMedia must reset _loaded_path so is_loaded()
        # reports False AND a later load() of the SAME path actually re-loads
        # (load() early-returns while _loaded_path matches) — otherwise audio is
        # dead for that path for the rest of the session.
        _prime_player(backend)
        backend.load(Path("/broken.mp4"))
        assert backend.is_loaded() is True
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.InvalidMedia
        )
        assert backend.is_loaded() is False


class TestDeferredPlay:
    def test_play_before_ready_is_deferred(self, backend):
        fake = _prime_player(backend)
        # Backend doesn't think media is ready yet. play() must NOT
        # call the underlying player.play() — that's the broken path.
        backend.play()
        fake.play.assert_not_called()

    def test_loaded_status_arms_pending_play(self, backend):
        fake = _prime_player(backend)
        backend.play()
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        fake.play.assert_called_once()
        # Pending flag cleared so a subsequent Loaded event doesn't
        # re-trigger play.
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        fake.play.assert_called_once()

    def test_buffered_status_also_arms(self, backend):
        # MF sometimes skips LoadedMedia and goes directly to
        # BufferedMedia. Both must arm pending play.
        fake = _prime_player(backend)
        backend.play()
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.BufferedMedia
        )
        fake.play.assert_called_once()

    def test_pause_cancels_pending_play(self, backend):
        # User asks to play, then changes mind before load completes.
        # When media finishes loading, play must NOT auto-fire.
        fake = _prime_player(backend)
        backend.play()
        backend.pause()
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        fake.play.assert_not_called()

    def test_play_after_ready_calls_player_immediately(self, backend):
        fake = _prime_player(backend)
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        backend.play()
        fake.play.assert_called_once()


class TestDeferredSeek:
    def test_seek_before_ready_is_deferred(self, backend):
        fake = _prime_player(backend)
        backend.seek_seconds(5.0)
        fake.setPosition.assert_not_called()

    def test_loaded_status_applies_pending_seek(self, backend):
        fake = _prime_player(backend)
        backend.seek_seconds(5.0)
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        fake.setPosition.assert_called_once_with(5000)

    def test_seek_after_ready_calls_player_immediately(self, backend):
        fake = _prime_player(backend)
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        backend.seek_seconds(2.5)
        fake.setPosition.assert_called_once_with(2500)

    def test_negative_seek_clamps_to_zero(self, backend):
        fake = _prime_player(backend)
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        backend.seek_seconds(-3.0)
        fake.setPosition.assert_called_once_with(0)


class TestLoadResetsState:
    def test_load_clears_ready_and_pending(self, backend, tmp_path: Path):
        # Existing media ready, pending play armed for a moment, then
        # a load() of a NEW target must reset everything: the new
        # source is loading, so subsequent play() should defer again.
        fake = _prime_player(backend)
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        assert backend._media_ready is True  # noqa: SLF001
        # Switch target. The real call would invoke setSource on the
        # player; with our mock that just records, but the backend
        # MUST flip _media_ready back to False so the next play()
        # gets deferred.
        backend.load(tmp_path / "different.mp4")
        assert backend._media_ready is False  # noqa: SLF001
        backend.play()
        # Now the player.play() must NOT have been called again — it
        # waits for the next LoadedMedia event for the new source.
        fake.play.assert_not_called()


class TestReload:
    """reload() CLEARS the source then re-sets it (clear-then-set), so a resume
    after an async source swap goes through a real LoadedMedia transition. A
    plain same-URL setSource leaves the player Stopped@0 and the deferred play
    never fires — the "audio dies on a source change" bug (verified by
    scripts/audio_resume_test.py: same-URL SILENT, clear-then-set RESUMES)."""

    def test_reload_clears_then_reissues_source(self, backend, tmp_path: Path):
        fake = _prime_player(backend)
        backend.load(tmp_path / "clip.mp4")
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        assert backend._media_ready is True  # noqa: SLF001
        fake.setSource.reset_mock()
        backend.reload()
        # Cleared first (empty URL) THEN re-set to the real file — the clear is
        # what forces Qt to actually reload an unchanged source.
        assert fake.setSource.call_count == 2
        assert fake.setSource.call_args_list[0].args[0].isEmpty()
        assert fake.setSource.call_args_list[1].args[0].toLocalFile().endswith(
            "clip.mp4"
        )
        assert backend._media_ready is False  # noqa: SLF001
        # A play() now defers until the re-load reports ready (the proven path).
        backend.play()
        fake.play.assert_not_called()
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        fake.play.assert_called_once()

    def test_reload_without_loaded_media_is_noop(self, backend):
        fake = _prime_player(backend)
        backend.reload()
        fake.setSource.assert_not_called()


class TestInvalidMediaClearsPending:
    def test_invalid_media_drops_pending_play(self, backend):
        fake = _prime_player(backend)
        backend.play()
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.InvalidMedia
        )
        # Pending play must NOT fire on an invalid source.
        fake.play.assert_not_called()
        # And a subsequent Loaded event must not retroactively play.
        backend._on_media_status_changed(  # noqa: SLF001
            QMediaPlayer.MediaStatus.LoadedMedia
        )
        # LoadedMedia after Invalid resets ready, but pending was
        # cleared by Invalid — so still no play.
        fake.play.assert_not_called()
