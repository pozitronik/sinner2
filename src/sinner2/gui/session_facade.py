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

from collections.abc import Callable
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
        snapshot_provider: Callable[[], ProcessorParamsSnapshot] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._player = player
        self._live = live
        # Pulls the current processor settings when starting / hot-updating the
        # camera (the live engine builds its chain from a snapshot, like the
        # file engine does). Injected by main_window; None until then.
        self._snapshot_provider = snapshot_provider
        self._active_kind = SessionKind.NONE
        self._source_path: Path | None = None
        self._file_target_path: Path | None = None
        self._camera_config: CameraConfig | None = None
        player.errorOccurred.connect(self.errorOccurred)
        live.errorOccurred.connect(self.errorOccurred)
        player.sessionSwitching.connect(self.sessionSwitching)

    # ---- target / source / settings (routed to the active engine) ----

    def set_source(self, source_path: Path) -> None:
        """The face to apply. Hot-applies to the active session, or seeds the
        next file build when a target is already chosen."""
        self._source_path = source_path
        if self._active_kind is SessionKind.CAMERA:
            # Fast path: re-point the live swapper without a chain rebuild (the
            # enhancer/upscaler worker instances survive). Settings changes still
            # go through apply_settings → _update_camera (full rebuild).
            self._live.set_source(source_path)
            return
        if self._player.executor() is not None:
            self._player.change_source(source_path)
        elif self._file_target_path is not None:
            self._player.set_source_and_target(source_path, self._file_target_path)
            self._emit_caps()

    def set_target(self, descriptor: FileTarget | CameraConfig) -> None:
        """Choose the target. A file path routes to the file engine (build on
        first load, async swap when one is active); a CameraConfig tears down the
        file session and starts the camera."""
        if isinstance(descriptor, CameraConfig):
            self._activate_camera(descriptor)
            return
        if not isinstance(descriptor, FileTarget):
            return
        if self._active_kind is SessionKind.CAMERA:
            self._live.stop()  # leaving the camera for a file target
        self._active_kind = SessionKind.FILE
        self._file_target_path = descriptor.path
        if self._player.executor() is not None:
            self._player.change_target(descriptor.path)
        elif self._source_path is not None:
            self._player.set_source_and_target(self._source_path, descriptor.path)
        self._emit_caps()

    def _activate_camera(self, config: CameraConfig) -> None:
        # Tear down the file session (keeps the reusable audio backend), then
        # start the camera — auto-start, like a file target shows frame 0.
        self._player.deactivate()
        self._active_kind = SessionKind.CAMERA
        self._camera_config = config
        self._start_camera()
        self._emit_caps()

    def deactivate_camera(self) -> None:
        """Leave camera mode (the 📹 toggle was turned off): stop the camera and
        drop to NONE — the file session was torn down on activation, so
        re-selecting a file target rebuilds it. Emits capabilitiesChanged so the
        file-only chrome (the Execution group, the target picker) restores."""
        if self._active_kind is not SessionKind.CAMERA:
            return
        self._live.stop()
        self._active_kind = SessionKind.NONE
        self._emit_caps()

    def _start_camera(self) -> None:
        if self._snapshot_provider is None or self._source_path is None:
            return  # need a face + a settings source to build the live chain
        cfg = self._camera_config
        if cfg is None:
            return
        self._live.start(
            source_path=self._source_path,
            snapshot=self._snapshot_provider(),
            device=cfg.device, width=cfg.width, height=cfg.height,
            fps=cfg.fps, workers=cfg.workers, mjpeg_port=cfg.mjpeg_port,
        )

    def apply_settings(self, snapshot: ProcessorParamsSnapshot) -> None:
        """Hot-apply processor/exec settings to the active session."""
        if self._active_kind is SessionKind.CAMERA:
            self._update_camera()
            return
        self._player.apply_session_config(**snapshot.to_session_config())
        self._player.set_video_backend(snapshot.video_backend)
        self._player.set_reader_pool_size(snapshot.reader_pool_size)
        self._player.set_processing_scale(snapshot.processing_scale)

    def _update_camera(self) -> None:
        # Hot-apply the current face + settings to the running camera chain.
        if self._snapshot_provider is None or self._source_path is None:
            return
        self._live.update(
            source_path=self._source_path, snapshot=self._snapshot_provider()
        )

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
        if self._active_kind is SessionKind.CAMERA:
            return SessionCapabilities.for_camera()
        # File caps follow the player's actual session (for_file when an executor
        # exists, else none) — robust to sessions built without a set_target call.
        return self._player.capabilities()

    def is_active(self) -> bool:
        if self._active_kind is SessionKind.CAMERA:
            return True
        return self._player.executor() is not None

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
