#!/usr/bin/env python3
"""Standalone MJPEG-over-HTTP test bench for the planned Sinner2 live-webcam mode.

Serves a live frame source over MJPEG so you can confirm which of your apps can
consume it (browser, VLC, OBS) and eyeball end-to-end latency, before we fold an
MJPEG sink into Sinner2.

RTSP was dropped for now: ffmpeg's `-rtsp_flags listen` is unreliable (it failed
even on a pure ffmpeg test source), and a robust RTSP path would need a bundled
MediaMTX binary -- a possible later addition, not needed for the MJPEG path.

Note: this is NOT a virtual webcam -- it won't appear in the camera dropdown of
Zoom / Teams / Discord (that needs a virtual-cam driver). It's the driver-free
IP-stream path, for stream-aware consumers.

Requirements: numpy, opencv-python (already in the sinner2 venv).

Examples:
  python scripts/stream_test.py                 # synthetic test pattern
  python scripts/stream_test.py --source 0      # webcam index 0
  python scripts/stream_test.py --source clip.mp4
  python scripts/stream_test.py --width 1280 --height 720 --fps 30

Then open a consumer at the URLs it prints, e.g.:
  http://localhost:8080/             (browser preview page)
  http://localhost:8080/stream.mjpg  (VLC: Open Network Stream)
  http://localhost:8080/snapshot.jpg (single JPEG)
"""
from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


class FrameSource:
    """Background producer of the latest frame (always sized to width x height).

    A single capture/generate thread keeps `_latest` fresh; any number of MJPEG
    clients read it at their own pace -- the same latest-wins model the real live
    mode will use.
    """

    def __init__(self, spec: str, width: int, height: int, fps: int) -> None:
        self.spec = spec
        self.w = width
        self.h = height
        self.fps = fps
        self._latest = np.zeros((height, width, 3), np.uint8)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def read(self) -> np.ndarray:
        with self._lock:
            return self._latest.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _set(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))
        with self._lock:
            self._latest = frame

    def _run(self) -> None:
        if self.spec == "test":
            self._run_synthetic()
        else:
            self._run_capture()

    def _run_synthetic(self) -> None:
        grad = np.tile(np.linspace(0, 255, self.w, dtype=np.uint8), (self.h, 1))
        interval = 1.0 / self.fps
        i = 0
        while not self._stop.is_set():
            colored = cv2.applyColorMap(np.roll(grad, (i * 4) % self.w, axis=1),
                                        cv2.COLORMAP_TURBO)
            x = (i * 8) % self.w
            cv2.rectangle(colored, (x, 0), (min(x + 6, self.w), self.h),
                          (255, 255, 255), -1)
            t = datetime.now()
            clock = t.strftime("%H:%M:%S.") + f"{t.microsecond // 1000:03d}"
            for color, thick in ((0, 0, 0), 6), ((255, 255, 255), 2):
                cv2.putText(colored, clock, (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, thick)
            cv2.putText(colored, f"frame {i}  {self.w}x{self.h}@{self.fps}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            self._set(colored)
            i += 1
            time.sleep(interval)

    def _run_capture(self) -> None:
        index: int | str = int(self.spec) if self.spec.isdigit() else self.spec
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        if not cap.isOpened():
            print(f"[source] could not open {self.spec!r}; falling back to test pattern")
            cap.release()
            self._run_synthetic()
            return
        is_file = not str(self.spec).isdigit()
        interval = 1.0 / self.fps
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_file:  # loop the clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)  # transient webcam hiccup
                continue
            self._set(frame)
            if is_file:
                time.sleep(interval)  # webcams pace themselves; files don't
        cap.release()


# --------------------------- MJPEG over HTTP ---------------------------------

_PAGE = b"""<!doctype html><title>stream_test</title>
<body style="margin:0;background:#111;text-align:center">
<img src="/stream.mjpg" style="max-width:100%;height:auto">
</body>"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_a) -> None:  # quieter than the default per-request log
        pass

    def do_GET(self) -> None:  # noqa: N802 (stdlib API name)
        src: FrameSource = self.server.source  # type: ignore[attr-defined]
        if self.path in ("/", "/index.html"):
            self._send_bytes(_PAGE, "text/html")
        elif self.path == "/snapshot.jpg":
            ok, jpg = cv2.imencode(".jpg", src.read())
            if ok:
                self._send_bytes(jpg.tobytes(), "image/jpeg")
        elif self.path == "/stream.mjpg":
            self._stream(src)
        else:
            self.send_error(404)

    def _send_bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, src: FrameSource) -> None:
        print(f"[mjpeg] client connected: {self.client_address[0]}")
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        interval = 1.0 / src.fps
        try:
            while True:
                ok, jpg = cv2.imencode(".jpg", src.read())
                if not ok:
                    continue
                data = jpg.tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[mjpeg] client disconnected: {self.client_address[0]}")


class _MJPEGServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], source: FrameSource) -> None:
        super().__init__(addr, _MJPEGHandler)
        self.source = source


# ------------------------------- main ----------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="test",
                   help="'test' (pattern), a webcam index like 0, or a video path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--host", default="0.0.0.0", help="bind address")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    source = FrameSource(args.source, args.width, args.height, args.fps)
    source.start()

    server = _MJPEGServer((args.host, args.port), source)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"\nsource: {args.source!r}  {args.width}x{args.height}@{args.fps}\n")
    print(f"MJPEG  browser : http://localhost:{args.port}/")
    print(f"       stream  : http://localhost:{args.port}/stream.mjpg"
          "   (VLC: Open Network Stream)")
    print(f"       snapshot: http://localhost:{args.port}/snapshot.jpg")
    if args.host == "0.0.0.0":
        print("\n(bound on all interfaces -- reachable from your LAN via this "
              "machine's IP)")
    print("\nCtrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        server.shutdown()
        source.stop()


if __name__ == "__main__":
    main()
