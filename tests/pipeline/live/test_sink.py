"""Tests for the live-mode output sinks (Stage 1).

MjpegSink is exercised over a real loopback socket (the same in-process HTTP
smoke that validated the standalone bench): push a frame, then fetch the
snapshot / stream / page and assert real JPEG bytes come back.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request

import numpy as np

from sinner2.pipeline.live.sink import FrameSink, MjpegSink


def _sink() -> MjpegSink:
    return MjpegSink(host="127.0.0.1", port=0, fps=60)


def test_mjpeg_sink_satisfies_framesink_protocol():
    assert isinstance(_sink(), FrameSink)


def test_mjpeg_sink_serves_pushed_frame():
    sink = _sink()
    sink.start()
    try:
        sink.push(np.full((48, 64, 3), 128, np.uint8))
        time.sleep(0.1)
        base = f"http://127.0.0.1:{sink.port}"

        snap = urllib.request.urlopen(base + "/snapshot.jpg", timeout=5).read()
        assert snap[:2] == b"\xff\xd8"  # JPEG SOI

        resp = urllib.request.urlopen(base + "/stream.mjpg", timeout=5)
        chunk = resp.read(4000)
        resp.close()
        assert b"--frame" in chunk and b"\xff\xd8" in chunk

        page = urllib.request.urlopen(base + "/", timeout=5).read()
        assert b"stream.mjpg" in page
    finally:
        sink.stop()


def test_snapshot_is_503_before_first_push():
    sink = _sink()
    sink.start()
    try:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{sink.port}/snapshot.jpg", timeout=5
            )
            raise AssertionError("expected 503 before any frame is pushed")
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
    finally:
        sink.stop()


def test_push_swaps_latest_frame():
    sink = _sink()
    sink.push(np.zeros((8, 8, 3), np.uint8))
    first = sink.encode_latest()
    sink.push(np.full((8, 8, 3), 255, np.uint8))
    second = sink.encode_latest()
    assert first is not None and second is not None and first != second


def test_encode_latest_caches_until_next_push(monkeypatch):
    # Encode-once-share: repeated encode_latest() calls (N clients) reuse one
    # JPEG until a new frame is pushed.
    import sinner2.pipeline.live.sink as sink_mod

    calls = [0]
    real = sink_mod.cv2.imencode

    def counting(ext, frame, params):
        calls[0] += 1
        return real(ext, frame, params)

    monkeypatch.setattr(sink_mod.cv2, "imencode", counting)
    sink = _sink()
    sink.push(np.zeros((8, 8, 3), np.uint8))
    a = sink.encode_latest()
    b = sink.encode_latest()  # cache hit — no second encode
    assert a is not None and a == b
    assert calls[0] == 1
    sink.push(np.full((8, 8, 3), 255, np.uint8))  # invalidates the cache
    c = sink.encode_latest()
    assert calls[0] == 2 and c != a


def test_describe_reports_url():
    sink = _sink()
    sink.start()
    try:
        assert f":{sink.port}" in sink.describe()
    finally:
        sink.stop()
