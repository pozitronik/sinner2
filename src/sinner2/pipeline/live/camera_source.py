"""Live capture source for the live-camera mode.

A background thread pulls frames from a capture device (webcam index, or any
path/URL cv2.VideoCapture accepts) and keeps only the LATEST — consumers read it
at their own pace (latency-first: a dropped frame is fine, lag is not). Unlike
`CV2VideoTargetReader` there's no frame_count / seek: a live stream is infinite.

The capture object is built through an injectable factory so the loop is testable
without a real camera (mirrors PlayerController's session_factory hook).
"""
from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2

from sinner2.types import Frame

CaptureFactory = Callable[[Any, int, int], Any]


def _open(device: Any, backend: int | None) -> Any:
    return cv2.VideoCapture(device) if backend is None \
        else cv2.VideoCapture(device, backend)


def _default_capture(device: Any, width: int, height: int) -> Any:
    # Keep the first backend that not only OPENS but actually DELIVERS a frame.
    # A Windows webcam frequently "opens" on one backend yet returns no frames
    # (camera LED on, but cv2.read() fails) -> a black/frozen preview. So we
    # test-read each backend and pick one that yields a frame. DirectShow is
    # usually best; fall back to Media Foundation, then the platform default.
    if sys.platform == "win32" and isinstance(device, int):
        backends: list[int | None] = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    else:
        backends = [None]
    chosen: Any = None
    for backend in backends:
        cap = _open(device, backend)
        if cap.isOpened():
            ok, _frame = cap.read()
            if ok and _frame is not None:
                chosen = cap
                break
        cap.release()
    if chosen is None:
        # Nothing delivered a frame; hand back the first attempt so CameraSource
        # reports the failure (opened-but-no-frames is caught by frames_seen).
        chosen = _open(device, backends[0])
    if width > 0:
        chosen.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        chosen.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return chosen


class CameraSource:
    """Latest-frame producer over a capture device. `read()` returns the most
    recent frame (sized to width x height), or None until the first one arrives /
    if the device failed to open (see `opened` + `error`)."""

    def __init__(
        self,
        device: Any = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.device = device
        self.w = width
        self.h = height
        self.fps = max(1, fps)
        self._capture_factory = capture_factory or _default_capture
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._opened = False
        self._error: str | None = None
        self._frames_seen = 0
        self._ready = threading.Event()  # set once the open attempt has resolved

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def frames_seen(self) -> int:
        """How many frames have been captured (0 = opened but nothing delivered)."""
        return self._frames_seen

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Block until the device open attempt resolves; True if it opened."""
        self._ready.wait(timeout)
        return self._opened

    def read(self) -> Frame | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _store(self, frame: Frame) -> None:
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))  # type: ignore[assignment]
        with self._lock:
            self._latest = frame
            self._frames_seen += 1

    def _run(self) -> None:
        cap = self._capture_factory(self.device, self.w, self.h)
        self._opened = bool(cap.isOpened())
        if not self._opened:
            self._error = f"could not open capture device {self.device!r}"
            self._ready.set()
            cap.release()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)  # transient hiccup / no frame yet
                    continue
                self._store(frame)
        finally:
            cap.release()


def available_cameras(
    max_probe: int = 5, *, capture_factory: CaptureFactory | None = None
) -> list[int]:
    """Best-effort list of working camera indices (probes 0..max_probe-1).

    Cross-platform device *naming* is messy; this only returns the indices that
    open AND deliver a frame. Good enough for an MVP device picker."""
    factory = capture_factory or _default_capture
    found: list[int] = []
    for i in range(max_probe):
        cap = factory(i, 0, 0)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
        finally:
            cap.release()
    return found
