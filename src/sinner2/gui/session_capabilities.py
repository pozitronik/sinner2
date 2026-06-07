"""Value objects for the single-session model.

A session has one TARGET, which is either a file (image/video) or a camera. The
target's *kind* and *capabilities* — not a global mode — decide what the UI
exposes (seek/timeline/audio for a file; none of those for a camera). These are
plain immutable value objects consumed by the SessionFacade, the transport
controls, and main_window.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SessionKind(str, Enum):
    NONE = "none"      # no target loaded
    FILE = "file"      # image/video target (seekable, finite, maybe audio)
    CAMERA = "camera"  # live camera target (no seek/timeline/audio)


@dataclass(frozen=True)
class SessionCapabilities:
    """What the active target supports — drives per-control UI gating."""

    kind: SessionKind
    seekable: bool       # slider + seek keys
    has_timeline: bool   # finite frame_count → frame counter
    has_audio: bool      # volume control
    can_play_pause: bool  # play/pause button + Space

    @property
    def label(self) -> str:
        return self.kind.value

    @classmethod
    def none(cls) -> "SessionCapabilities":
        return cls(
            kind=SessionKind.NONE,
            seekable=False,
            has_timeline=False,
            has_audio=False,
            can_play_pause=False,
        )

    @classmethod
    def for_file(cls, *, has_audio: bool) -> "SessionCapabilities":
        return cls(
            kind=SessionKind.FILE,
            seekable=True,
            has_timeline=True,
            has_audio=has_audio,
            can_play_pause=True,
        )

    @classmethod
    def for_camera(cls) -> "SessionCapabilities":
        # A camera is non-seekable + has no audio; play/pause maps to stop/start.
        return cls(
            kind=SessionKind.CAMERA,
            seekable=False,
            has_timeline=False,
            has_audio=False,
            can_play_pause=True,
        )


@dataclass(frozen=True)
class FileTarget:
    """A media-file target (image or video). The facade routes this to the file
    engine, which wraps it in the validating `config.target.Target`."""

    path: Path


@dataclass(frozen=True)
class CameraConfig:
    """A camera target. The facade routes this to the live engine; it never
    enters the file `Target` / reader / cache machinery."""

    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    workers: int = 1
    mjpeg_port: int = 8080
