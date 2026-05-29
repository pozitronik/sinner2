"""QtMultimedia-backed AudioBackend.

Uses QMediaPlayer + QAudioOutput. No video sink wired — we want only
the audio track. Format coverage is whatever QtMultimedia/Media
Foundation supports out of the box (MP4 h264+aac, MOV, MKV, common
WAV/MP3 sidecar files). Less than VLC but zero extra installs.

Calls are expected from the GUI thread. QMediaPlayer instances must
live on the thread that hosts their Qt event loop, so we construct
them lazily in load() rather than in __init__ (which may run before a
QApplication exists in some test contexts).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class QtMediaAudioBackend(QObject):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player: QMediaPlayer | None = None
        self._output: QAudioOutput | None = None
        self._loaded_path: Path | None = None
        # Cached settings applied to a freshly created player/output. Lets
        # the controller call set_volume/set_muted before load() without
        # losing the value.
        self._pending_volume: float = 1.0
        self._pending_muted: bool = False
        # QMediaPlayer.setSource is asynchronous: after it returns the
        # player is in LoadingMedia, not Loaded. play() called in that
        # window can silently fail on Windows Media Foundation (the
        # symptom: switching target while playing → audio disappears
        # until the user pauses + plays again two or three times). We
        # track ready/pending state and arm play via mediaStatusChanged
        # so a play() during load is honoured the moment media is ready.
        self._media_ready: bool = False
        self._pending_play: bool = False
        self._pending_position_ms: int | None = None

    def _ensure_player(self) -> tuple[QMediaPlayer, QAudioOutput]:
        if self._player is None:
            output = QAudioOutput(self)
            output.setVolume(self._pending_volume)
            output.setMuted(self._pending_muted)
            player = QMediaPlayer(self)
            player.setAudioOutput(output)
            player.mediaStatusChanged.connect(self._on_media_status_changed)
            self._player = player
            self._output = output
        assert self._output is not None
        return self._player, self._output

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        """Apply any deferred play / seek the moment media is usable.

        LoadedMedia or BufferedMedia mean the codec is up and seeks /
        play() commands will actually be honoured. Earlier statuses are
        ignored (the deferred state stays armed until we see a usable one).
        InvalidMedia surfaces the error and clears pending state so
        nothing waits forever.
        """
        if self._player is None:
            return
        if status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            self._media_ready = True
            if self._pending_position_ms is not None:
                self._player.setPosition(self._pending_position_ms)
                self._pending_position_ms = None
            if self._pending_play:
                self._pending_play = False
                self._player.play()
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            # Cannot honour pending play/seek on broken media; drop them
            # so nothing dangles for the next load.
            self._pending_play = False
            self._pending_position_ms = None
            self._media_ready = False

    def load(self, media_path: Path) -> None:
        if self._loaded_path == media_path:
            return
        player, _ = self._ensure_player()
        player.stop()
        # Reset readiness — any play()/seek() that arrives during the
        # window between setSource() and mediaStatusChanged(Loaded) will
        # be queued by _on_media_status_changed.
        self._media_ready = False
        self._pending_play = False
        self._pending_position_ms = None
        player.setSource(QUrl.fromLocalFile(str(media_path)))
        self._loaded_path = media_path

    def play(self) -> None:
        if self._player is None:
            return
        if not self._media_ready:
            self._pending_play = True
            return
        self._player.play()

    def pause(self) -> None:
        if self._player is None:
            return
        # A pause() before media is ready cancels any pending auto-play
        # so the caller's intent ("don't play") is honoured when load
        # finishes — without this, a fast load/pause/play sequence could
        # play twice.
        self._pending_play = False
        self._player.pause()

    def seek_seconds(self, seconds: float) -> None:
        if self._player is None:
            return
        # QMediaPlayer position is in milliseconds; clamp negatives to 0
        # (occasional caller may pass -1 to mean "before start").
        ms = max(0, int(seconds * 1000))
        if not self._media_ready:
            # Seek before LoadedMedia is silently ignored by MF on
            # Windows. Defer until the player reports it's ready.
            self._pending_position_ms = ms
            return
        self._player.setPosition(ms)

    def set_volume(self, volume: float) -> None:
        clamped = max(0.0, min(1.0, volume))
        self._pending_volume = clamped
        if self._output is not None:
            self._output.setVolume(clamped)

    def set_muted(self, muted: bool) -> None:
        self._pending_muted = bool(muted)
        if self._output is not None:
            self._output.setMuted(bool(muted))

    def is_loaded(self) -> bool:
        return self._loaded_path is not None and self._player is not None

    def has_audio(self) -> bool:
        if self._player is None:
            return False
        # hasAudio() polls the loaded media's track set; reliable only
        # after the player has progressed past LoadingMedia status. The
        # UI uses this to gate audio controls but doesn't depend on it
        # for correctness — set_muted on a no-audio source is a no-op.
        return bool(self._player.hasAudio())

    def shutdown(self) -> None:
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
            self._player = None
        if self._output is not None:
            self._output = None
        self._loaded_path = None
