"""Output sinks for the live-camera mode.

A `FrameSink` consumes processed frames *pushed* from the live loop and exposes
them to outside consumers. `MjpegSink` serves them over MJPEG-HTTP (browser /
VLC / OBS) — driver-free, validated standalone first in `scripts/stream_test.py`.
Other sinks (RTSP via a bundled MediaMTX, WebRTC) can slot in behind the same
protocol later without touching the live loop.

Note: an MJPEG/IP stream is NOT a virtual webcam — it won't appear in the camera
dropdown of Zoom / Teams / Discord (that needs a virtual-cam driver). This is the
driver-free path, for stream-aware consumers.
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol, runtime_checkable

import cv2

from sinner2.types import Frame


@runtime_checkable
class FrameSink(Protocol):
    """Consumes processed frames from the live loop and exposes them to outside
    consumers. `push` is called once per processed frame on the loop thread;
    implementations must be cheap + non-blocking there (do I/O on their own
    thread)."""

    def start(self) -> None:
        ...

    def push(self, frame: Frame) -> None:
        ...

    def stop(self) -> None:
        ...

    def describe(self) -> str:
        """Human-readable endpoint (e.g. the URL) for the GUI to display."""
        ...


_PAGE = b"""<!doctype html><title>sinner2 live</title>
<body style="margin:0;background:#111;text-align:center">
<img src="/stream.mjpg" style="max-width:100%;height:auto">
</body>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a) -> None:  # quieter than the default per-request log
        pass

    def do_GET(self) -> None:  # noqa: N802 (stdlib API name)
        sink: MjpegSink = self.server.sink  # type: ignore[attr-defined]
        if self.path in ("/", "/index.html"):
            self._send(_PAGE, "text/html")
        elif self.path == "/snapshot.jpg":
            data = sink.encode_latest()
            if data is None:
                self.send_error(503, "no frame yet")
            else:
                self._send(data, "image/jpeg")
        elif self.path == "/stream.mjpg":
            self._stream(sink)
        else:
            self.send_error(404)

    def _send(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, sink: MjpegSink) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        interval = 1.0 / sink.fps
        try:
            while True:
                data = sink.encode_latest()
                if data is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], sink: MjpegSink) -> None:
        super().__init__(addr, _Handler)
        self.sink = sink


class MjpegSink:
    """Serves the latest pushed frame over MJPEG-HTTP.

    `push` just swaps an in-memory latest-frame (cheap, non-blocking); each
    connected client's handler thread encodes + sends at ~`fps`, so a slow
    client backpressures only its own thread, never the live loop.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        fps: int = 30,
        jpeg_quality: int = 80,
    ) -> None:
        self._host = host
        self._port = port
        self.fps = max(1, fps)
        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        self._latest: Frame | None = None
        self._version = 0  # bumped on each push; tags the encoded cache
        self._encoded: bytes | None = None  # JPEG of _latest, shared by clients
        self._encoded_version = -1
        self._lock = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _Server((self._host, self._port), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def push(self, frame: Frame) -> None:
        with self._lock:
            self._latest = frame
            self._version += 1  # invalidates the encoded cache; re-encode lazily

    def encode_latest(self) -> bytes | None:
        with self._lock:
            # Encode-once-share: N client handler threads (and a static scene
            # that isn't pushing new frames) reuse one JPEG instead of each
            # re-encoding the same frame every tick.
            if self._encoded is not None and self._encoded_version == self._version:
                return self._encoded
            frame = self._latest
            version = self._version
        if frame is None:
            return None
        # Encode OUTSIDE the lock so a slow encode can't block push() (the live
        # loop). A rare race where two clients both encode a freshly-pushed frame
        # just wastes one encode; the cache write below keeps only the latest.
        ok, jpg = cv2.imencode(".jpg", frame, self._jpeg_params)
        data = jpg.tobytes() if ok else None
        with self._lock:
            if data is not None and version == self._version:
                self._encoded = data
                self._encoded_version = version
        return data

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    @property
    def port(self) -> int:
        """The actually-bound port (resolves an ephemeral port=0 after start)."""
        if self._server is not None:
            return self._server.server_address[1]
        return self._port

    def describe(self) -> str:
        return f"http://localhost:{self.port}/  (MJPEG stream: /stream.mjpg)"
