"""Tests for CameraSource (Stage 2) via an injected fake capture — no real
webcam needed. Pins latest-wins, resize-to-requested-size, the open-failure
path, capture release on stop, and device enumeration."""
from __future__ import annotations

import time

import numpy as np

from sinner2.pipeline.live.camera_source import CameraSource, available_cameras


class _FakeCapture:
    def __init__(self, frames, opened=True):
        self._frames = list(frames)
        self._opened = opened
        self._i = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or not self._frames:
            return False, None
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return True, frame

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 0.0

    def release(self):
        self.released = True


def _factory_for(frames, opened=True):
    created = []

    def factory(_device, _w, _h):
        cap = _FakeCapture(frames, opened=opened)
        created.append(cap)
        return cap

    return factory, created


def test_reads_latest_frame():
    a = np.full((48, 64, 3), 10, np.uint8)
    b = np.full((48, 64, 3), 200, np.uint8)
    factory, _ = _factory_for([a, b])
    src = CameraSource(0, width=64, height=48, capture_factory=factory)
    src.start()
    assert src.wait_ready(timeout=5) is True
    time.sleep(0.05)
    frame = src.read()
    src.stop()
    assert frame is not None and frame.shape == (48, 64, 3)
    assert int(frame[0, 0, 0]) == 200  # latest (b), not a


def test_frames_seen_increments_while_capturing():
    factory, _ = _factory_for([np.zeros((48, 64, 3), np.uint8)])
    src = CameraSource(0, width=64, height=48, capture_factory=factory)
    src.start()
    src.wait_ready(timeout=5)
    time.sleep(0.05)
    src.stop()
    assert src.frames_seen > 0


def test_resizes_to_requested_size():
    factory, _ = _factory_for([np.full((100, 80, 3), 128, np.uint8)])
    src = CameraSource(0, width=64, height=48, capture_factory=factory)
    src.start()
    src.wait_ready(timeout=5)
    time.sleep(0.05)
    frame = src.read()
    src.stop()
    assert frame is not None and frame.shape == (48, 64, 3)


def test_open_failure_sets_error_and_reads_none():
    factory, created = _factory_for([], opened=False)
    src = CameraSource(7, capture_factory=factory)
    src.start()
    assert src.wait_ready(timeout=5) is False
    assert src.opened is False
    assert src.error is not None and "7" in src.error
    assert src.read() is None
    src.stop()
    assert created[0].released is True


def test_stop_releases_capture():
    factory, created = _factory_for([np.zeros((48, 64, 3), np.uint8)])
    src = CameraSource(0, width=64, height=48, capture_factory=factory)
    src.start()
    src.wait_ready(timeout=5)
    time.sleep(0.02)
    src.stop()
    assert created[0].released is True


def test_available_cameras_returns_working_indices():
    def factory(device, _w, _h):
        usable = device in (0, 2)
        frame = np.zeros((4, 4, 3), np.uint8)
        return _FakeCapture([frame] if usable else [], opened=usable)

    assert available_cameras(max_probe=4, capture_factory=factory) == [0, 2]
