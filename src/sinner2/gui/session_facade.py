"""SessionFacade — the single active session over the file + camera engines.

main_window binds the transport, keyboard, and display to this facade instead of
juggling two controllers. The facade routes every call to whichever engine owns
the active target (file -> PlayerController, camera -> LiveController), keeps
exactly ONE active at a time (set_target tears the other down), and republishes a
unified capability + signal surface. It does NOT reimplement the audio-aware
transport logic — that stays authoritative in PlayerController; the facade only
chooses the engine.

Stage 4: file targets are fully wired; the camera branch (set_target with a
CameraConfig, and the camera hot-reconfigure) is stubbed and lands in Stage 6.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from sinner2.gui.live_controller import LiveController
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.processor_snapshot import ProcessorParamsSnapshot
from sinner2.gui.session_capabilities import (
    CameraConfig,
    FileTarget,
    SessionCapabilities,
    SessionKind,
)


class SessionFacade(QObject):
    capabilitiesChanged = Signal(object)  # SessionCapabilities, on activation
    errorOccurred = Signal(str)
    sessionSwitching = Signal(bool)

    def __init__(
        self,
        player: PlayerController,
        live: LiveController,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._player = player
        self._live = live
        self._active_kind = SessionKind.NONE
        self._source_path: Path | None = None
        self._file_target_path: Path | None = None
        self._snapshot: ProcessorParamsSnapshot | None = None
        player.errorOccurred.connect(self.errorOccurred)
        live.errorOccurred.connect(self.errorOccurred)
        player.sessionSwitching.connect(self.sessionSwitching)

    # ---- target / source / settings (routed to the active engine) ----

    def set_source(self, source_path: Path) -> None:
        """The face to apply. Hot-applies to the active session, or seeds the
        next file build when a target is already chosen."""
        self._source_path = source_path
        if self._active_kind is SessionKind.CAMERA:
            self._update_camera()
            return
        if self._player.executor() is not None:
            self._player.change_source(source_path)
        elif self._file_target_path is not None:
            self._player.set_source_and_target(source_path, self._file_target_path)
            self._emit_caps()

    def set_target(self, descriptor: FileTarget | CameraConfig) -> None:
        """Choose the target. A file path routes to the file engine (build on
        first load, async swap when one is already active)."""
        if isinstance(descriptor, CameraConfig):
            raise NotImplementedError("camera target lands in Stage 6")
        if not isinstance(descriptor, FileTarget):
            return
        self._active_kind = SessionKind.FILE
        self._file_target_path = descriptor.path
        if self._player.executor() is not None:
            self._player.change_target(descriptor.path)
        elif self._source_path is not None:
            self._player.set_source_and_target(self._source_path, descriptor.path)
        self._emit_caps()

    def apply_settings(self, snapshot: ProcessorParamsSnapshot) -> None:
        """Hot-apply processor/exec settings to the active session."""
        self._snapshot = snapshot
        if self._active_kind is SessionKind.CAMERA:
            self._update_camera()
            return
        self._player.apply_session_config(**snapshot.to_session_config())
        self._player.set_video_backend(snapshot.video_backend)
        self._player.set_reader_pool_size(snapshot.reader_pool_size)
        self._player.set_processing_scale(snapshot.processing_scale)

    def _update_camera(self) -> None:
        # Stage 6: live.update(source=self._source_path, snapshot=self._snapshot).
        pass

    # ---- transport (delegate; the player keeps the audio-aware logic) ----

    def toggle_playback(self) -> None:
        if self._active_kind is SessionKind.CAMERA:
            self._live.toggle_playback()
        else:
            self._player.toggle_playback()

    def play(self) -> None:
        if self._active_kind is SessionKind.CAMERA:
            if not self._live.is_running():
                self._live.toggle_playback()
        else:
            self._player.play()

    def pause(self) -> None:
        if self._active_kind is SessionKind.CAMERA:
            if self._live.is_running():
                self._live.toggle_playback()
        else:
            self._player.pause()

    def seek_to(self, frame: int) -> None:
        if self.capabilities().seekable:
            self._player.seek_to(frame)

    # ---- introspection ----

    def capabilities(self) -> SessionCapabilities:
        if self._active_kind is SessionKind.FILE:
            return self._player.capabilities()
        if self._active_kind is SessionKind.CAMERA:
            return SessionCapabilities.for_camera()
        return SessionCapabilities.none()

    def is_active(self) -> bool:
        return self._active_kind is not SessionKind.NONE

    def active_kind(self) -> SessionKind:
        return self._active_kind

    def player(self) -> PlayerController:
        """Escape hatch for inherently file-only panels (cache, save-frame,
        metrics) that reach the file engine directly."""
        return self._player

    def shutdown(self) -> None:
        self._live.stop()
        self._player.shutdown()

    def _emit_caps(self) -> None:
        self.capabilitiesChanged.emit(self.capabilities())
