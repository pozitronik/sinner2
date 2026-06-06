"""Tests for LiveLoop (Stage 3) with stub source / chain / sinks — no camera, no
HTTP. Pins: chain runs in order, result reaches every sink + the preview
callback, the source + sinks get started/stopped, and a throwing processor
doesn't kill the loop.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from sinner2.pipeline.live.live_loop import LiveLoop


class _StubSource:
    def __init__(self, frame):
        self._frame = frame
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def read(self):
        return self._frame

    def stop(self):
        self.stopped = True


class _SpyProcessor:
    def __init__(self, name, delta):
        self.name = name
        self._delta = delta
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return (frame.astype(int) + self._delta).astype(np.uint8)


class _SpySink:
    def __init__(self):
        self.pushed = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def push(self, frame):
        self.pushed.append(frame)

    def stop(self):
        self.stopped = True

    def describe(self):
        return "spy"


def _run_briefly(loop: LiveLoop, seconds: float = 0.1) -> None:
    loop.start()
    time.sleep(seconds)
    loop.stop()


def test_processes_chain_in_order_to_all_sinks_and_preview():
    src = _StubSource(np.zeros((8, 8, 3), np.uint8))
    p1, p2 = _SpyProcessor("a", 1), _SpyProcessor("b", 2)
    sink_a, sink_b = _SpySink(), _SpySink()
    previews: list = []
    loop = LiveLoop(src, [p1, p2], [sink_a, sink_b],
                    on_frame=previews.append, fps=200)
    _run_briefly(loop)

    assert p1.calls > 0 and p2.calls > 0
    assert sink_a.pushed and sink_b.pushed and previews
    # chain applied in order: 0 + 1 + 2 == 3
    assert int(sink_a.pushed[-1][0, 0, 0]) == 3
    assert int(previews[-1][0, 0, 0]) == 3
    assert loop.frames_processed > 0


def test_lifecycle_starts_and_stops_source_and_sinks():
    src = _StubSource(np.zeros((4, 4, 3), np.uint8))
    sink = _SpySink()
    loop = LiveLoop(src, [], [sink], fps=200)
    _run_briefly(loop, 0.05)
    assert src.started and src.stopped
    assert sink.started and sink.stopped


def test_none_frame_is_skipped():
    class _NoneSource(_StubSource):
        def read(self):
            return None

    sink = _SpySink()
    loop = LiveLoop(_NoneSource(None), [], [sink], fps=200)
    _run_briefly(loop, 0.05)
    assert sink.pushed == []  # nothing to push until a frame arrives
    assert loop.frames_processed == 0


def test_throwing_processor_does_not_kill_loop():
    class _Flaky:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def process(self, frame):
            self.calls += 1
            raise RuntimeError("boom")

    src = _StubSource(np.zeros((4, 4, 3), np.uint8))
    sink = _SpySink()
    flaky = _Flaky()
    loop = LiveLoop(src, [flaky], [sink], fps=200)
    _run_briefly(loop)
    assert flaky.calls > 1          # kept running past the first failure
    assert loop.errors > 0
    assert sink.pushed == []        # failed frames are not pushed


def test_stop_is_clean_when_never_started_frames():
    # Source that blocks returning None forever -> stop must still return promptly.
    src = _StubSource(None)
    loop = LiveLoop(src, [], [_SpySink()], fps=200)
    loop.start()
    time.sleep(0.02)
    done = threading.Event()
    threading.Thread(target=lambda: (loop.stop(), done.set()), daemon=True).start()
    assert done.wait(timeout=3.0), "stop() did not return promptly"
