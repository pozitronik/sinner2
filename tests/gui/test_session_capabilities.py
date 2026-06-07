"""Tests for the session value objects (SessionKind / SessionCapabilities /
FileTarget / CameraConfig). Pure value objects — no Qt."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from sinner2.gui.session_capabilities import (
    CameraConfig,
    FileTarget,
    SessionCapabilities,
    SessionKind,
)


def test_none_caps_disable_everything():
    caps = SessionCapabilities.none()
    assert caps.kind is SessionKind.NONE
    assert not caps.seekable
    assert not caps.has_timeline
    assert not caps.has_audio
    assert not caps.can_play_pause
    assert caps.label == "none"


def test_for_file_with_audio():
    caps = SessionCapabilities.for_file(has_audio=True)
    assert caps.kind is SessionKind.FILE
    assert caps.seekable
    assert caps.has_timeline
    assert caps.has_audio
    assert caps.can_play_pause
    assert caps.label == "file"


def test_for_file_without_audio():
    caps = SessionCapabilities.for_file(has_audio=False)
    assert caps.seekable and caps.has_timeline and caps.can_play_pause
    assert not caps.has_audio


def test_for_camera_is_non_seekable_no_audio_but_playable():
    caps = SessionCapabilities.for_camera()
    assert caps.kind is SessionKind.CAMERA
    assert not caps.seekable
    assert not caps.has_timeline
    assert not caps.has_audio
    assert caps.can_play_pause  # Space = stop/start the camera
    assert caps.label == "camera"


def test_capabilities_are_frozen():
    caps = SessionCapabilities.none()
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.seekable = True  # type: ignore[misc]


def test_file_target_holds_path():
    t = FileTarget(Path("/x/y.mp4"))
    assert t.path == Path("/x/y.mp4")


def test_camera_config_defaults_and_overrides():
    assert CameraConfig() == CameraConfig(
        device=0, width=1280, height=720, fps=30, workers=1, mjpeg_port=8080
    )
    c = CameraConfig(device=2, width=640, height=480, fps=60, workers=3,
                     mjpeg_port=9000)
    assert (c.device, c.width, c.height, c.fps, c.workers, c.mjpeg_port) == (
        2, 640, 480, 60, 3, 9000
    )


def test_camera_config_is_frozen():
    c = CameraConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.device = 5  # type: ignore[misc]
