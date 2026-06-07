"""SessionFacade (Stage 4): routes the transport + target/source/settings to the
file engine, re-emits the unified signals, and gates seek by capability. Camera
target activation is stubbed here (lands in Stage 6). Uses lightweight QObject
stubs with REAL signals so both routing (recorded calls) and signal re-emit are
checkable without a real executor/camera.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal

from sinner2.gui.session_capabilities import (
    CameraConfig,
    FileTarget,
    SessionCapabilities,
    SessionKind,
)
from sinner2.gui.session_facade import SessionFacade


class _StubPlayer(QObject):
    errorOccurred = Signal(str)
    sessionSwitching = Signal(bool)

    def __init__(self):
        super().__init__()
        self._executor = None
        self._caps = SessionCapabilities.none()
        self.calls: list = []

    def executor(self):
        return self._executor

    def capabilities(self):
        return self._caps

    def set_source_and_target(self, s, t):
        self.calls.append(("sst", s, t))

    def change_target(self, t):
        self.calls.append(("ct", t))

    def change_source(self, s):
        self.calls.append(("cs", s))

    def play(self):
        self.calls.append(("play",))

    def pause(self):
        self.calls.append(("pause",))

    def toggle_playback(self):
        self.calls.append(("toggle",))

    def seek_to(self, f):
        self.calls.append(("seek", f))

    def apply_session_config(self, **kw):
        self.calls.append(("cfg", kw))

    def set_video_backend(self, b):
        self.calls.append(("vb", b))

    def set_reader_pool_size(self, n):
        self.calls.append(("rps", n))

    def set_processing_scale(self, s):
        self.calls.append(("scale", s))

    def deactivate(self):
        self.calls.append(("deactivate",))

    def shutdown(self):
        self.calls.append(("shutdown",))


class _StubLive(QObject):
    errorOccurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.calls: list = []

    def is_running(self):
        return self._running

    def start(self, **kwargs):
        self.calls.append(("start", kwargs))
        self._running = True

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))

    def set_source(self, source_path):
        self.calls.append(("set_source", source_path))

    def toggle_playback(self):
        self.calls.append(("toggle",))
        self._running = not self._running

    def stop(self):
        self.calls.append(("stop",))
        self._running = False


class _Snap:
    video_backend = "cv2"
    reader_pool_size = 3
    processing_scale = 0.5

    def to_session_config(self):
        return {"a": 1}


@pytest.fixture
def trio(qtbot):
    player, live = _StubPlayer(), _StubLive()
    facade = SessionFacade(player, live, snapshot_provider=lambda: _Snap())
    return facade, player, live


SRC = Path("face.jpg")
TGT = Path("clip.mp4")


# ---- file target build / swap ----

def test_set_source_then_target_builds_first_load(trio):
    facade, player, _ = trio
    facade.set_source(SRC)
    assert player.calls == []          # no target yet → wait
    facade.set_target(FileTarget(TGT))
    assert ("sst", SRC, TGT) in player.calls
    assert facade.active_kind() is SessionKind.FILE


def test_set_target_while_active_swaps_target(trio):
    facade, player, _ = trio
    facade.set_source(SRC)
    player._executor = object()        # a live session exists
    facade.set_target(FileTarget(TGT))
    assert ("ct", TGT) in player.calls          # async swap, not a fresh build
    assert ("sst", SRC, TGT) not in player.calls


def test_set_source_while_active_swaps_source(trio):
    facade, player, _ = trio
    facade.set_target(FileTarget(TGT))  # target first
    player._executor = object()
    facade.set_source(SRC)
    assert ("cs", SRC) in player.calls


# ---- transport delegation + seek gating ----

def test_transport_routes_to_player_for_file(trio):
    facade, player, _ = trio
    facade.set_target(FileTarget(TGT))
    facade.play()
    facade.pause()
    facade.toggle_playback()
    assert ("play",) in player.calls
    assert ("pause",) in player.calls
    assert ("toggle",) in player.calls


def test_seek_no_ops_when_no_target(trio):
    facade, player, _ = trio
    facade.seek_to(5)
    assert ("seek", 5) not in player.calls   # NONE caps → not seekable


def test_seek_routes_when_file_seekable(trio):
    facade, player, _ = trio
    facade.set_target(FileTarget(TGT))
    player._executor = object()
    player._caps = SessionCapabilities.for_file(has_audio=True)
    facade.seek_to(7)
    assert ("seek", 7) in player.calls


# ---- settings ----

def test_apply_settings_routes_to_player(trio):
    facade, player, _ = trio
    facade.set_target(FileTarget(TGT))
    facade.apply_settings(_Snap())
    kinds = [c[0] for c in player.calls]
    assert "cfg" in kinds and "vb" in kinds and "rps" in kinds and "scale" in kinds


# ---- capabilities ----

def test_capabilities_none_then_file(trio):
    facade, player, _ = trio
    assert facade.capabilities().kind is SessionKind.NONE
    facade.set_target(FileTarget(TGT))
    player._executor = object()
    player._caps = SessionCapabilities.for_file(has_audio=False)
    assert facade.capabilities().kind is SessionKind.FILE


# ---- signal re-emit ----

def test_reemits_player_and_live_errors(trio, qtbot):
    facade, player, live = trio
    seen: list = []
    facade.errorOccurred.connect(seen.append)
    player.errorOccurred.emit("boom-file")
    live.errorOccurred.emit("boom-live")
    assert seen == ["boom-file", "boom-live"]


def test_reemits_session_switching(trio):
    facade, player, _ = trio
    seen: list = []
    facade.sessionSwitching.connect(seen.append)
    player.sessionSwitching.emit(True)
    player.sessionSwitching.emit(False)
    assert seen == [True, False]


def test_capabilities_changed_emitted_on_target(trio):
    facade, _, _ = trio
    seen: list = []
    facade.capabilitiesChanged.connect(seen.append)
    facade.set_target(FileTarget(TGT))
    assert len(seen) == 1


# ---- lifecycle ----

def test_shutdown_stops_live_then_player(trio):
    facade, player, live = trio
    facade.shutdown()
    assert ("stop",) in live.calls
    assert ("shutdown",) in player.calls


# ---- camera target ----

def test_camera_target_deactivates_file_and_starts_camera(trio):
    facade, player, live = trio
    facade.set_source(SRC)              # a face is required to build the chain
    facade.set_target(CameraConfig(device=2, mjpeg_port=9000))
    assert ("deactivate",) in player.calls         # file engine torn down
    assert facade.active_kind() is SessionKind.CAMERA
    starts = [c for c in live.calls if c[0] == "start"]
    assert starts and starts[0][1]["device"] == 2  # auto-started with the config
    assert starts[0][1]["source_path"] == SRC


def test_camera_caps_are_for_camera(trio):
    facade, _, _ = trio
    facade.set_source(SRC)
    facade.set_target(CameraConfig())
    assert facade.capabilities().kind is SessionKind.CAMERA
    assert facade.is_active()


def test_file_target_after_camera_stops_the_camera(trio):
    facade, player, live = trio
    facade.set_source(SRC)
    facade.set_target(CameraConfig())
    facade.set_target(FileTarget(TGT))
    assert ("stop",) in live.calls
    assert facade.active_kind() is SessionKind.FILE


def test_source_change_uses_fast_path_not_rebuild_when_camera(trio):
    facade, _, live = trio
    facade.set_source(SRC)
    facade.set_target(CameraConfig())
    other = Path("face2.jpg")
    facade.set_source(other)
    # Fast path: live.set_source (no chain rebuild), NOT live.update.
    sets = [c for c in live.calls if c[0] == "set_source"]
    assert sets and sets[-1][1] == other
    assert not any(c[0] == "update" for c in live.calls)


def test_settings_change_rebuilds_running_camera(trio):
    facade, _, live = trio
    facade.set_source(SRC)
    facade.set_target(CameraConfig())
    facade.apply_settings(_Snap())
    # Settings DO rebuild the chain (enhancer/upscaler params may change).
    assert any(c[0] == "update" for c in live.calls)


def test_toggle_playback_routes_to_camera(trio):
    facade, player, live = trio
    facade.set_source(SRC)
    facade.set_target(CameraConfig())
    facade.toggle_playback()  # Space = stop/start the camera
    assert ("toggle",) in live.calls
    assert ("toggle",) not in player.calls


def test_camera_does_not_start_without_a_source(trio):
    facade, player, live = trio
    facade.set_target(CameraConfig())  # no source set
    assert ("deactivate",) in player.calls
    assert facade.active_kind() is SessionKind.CAMERA   # camera is the target…
    assert not any(c[0] == "start" for c in live.calls)  # …but can't start yet
