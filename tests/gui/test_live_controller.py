"""Tests for LiveController (Stage 5a) with injected fake camera + sink and an
all-processors-off snapshot (empty chain → no model loading). Pins: start runs +
emits processed frames as a queued signal, stop tears down, bad source surfaces
an error, double-start is a no-op.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sinner2.gui.live_controller import LiveController
from sinner2.gui.widgets.processor_controls import QProcessorControls


class _StubCam:
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
        return "http://localhost:0/"


@pytest.fixture
def off_snapshot(qtbot):
    w = QProcessorControls()
    qtbot.addWidget(w)
    return dataclasses.replace(
        w.snapshot(),
        swapper_enabled=False,
        enhancer_enabled=False,
        upscaler_enabled=False,
    )


@pytest.fixture
def source_file(tmp_path):
    p = tmp_path / "face.jpg"
    p.write_bytes(b"")  # Source only validates existence
    return p


def _controller(cam, sink):
    return LiveController(
        camera_factory=lambda device, w, h, fps: cam,
        sink_factory=lambda port, fps: sink,
    )


def test_start_runs_and_emits_processed_frames(qtbot, off_snapshot, source_file):
    cam = _StubCam(np.full((8, 8, 3), 7, np.uint8))
    sink = _SpySink()
    ctrl = _controller(cam, sink)
    running = []
    ctrl.runningChanged.connect(running.append)
    try:
        with qtbot.waitSignal(ctrl.frameReady, timeout=2000):
            ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        assert ctrl.is_running()
        assert running == [True]
        assert cam.started and sink.started
        assert sink.pushed  # at least one frame pushed before the signal fired
    finally:
        ctrl.stop()


def test_stop_tears_down(qtbot, off_snapshot, source_file):
    cam = _StubCam(np.zeros((8, 8, 3), np.uint8))
    sink = _SpySink()
    ctrl = _controller(cam, sink)
    ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
    ctrl.stop()
    assert not ctrl.is_running()
    assert cam.stopped and sink.stopped
    assert ctrl.sink_url() is None


def test_invalid_source_emits_error(qtbot, off_snapshot, tmp_path):
    cam = _StubCam(np.zeros((8, 8, 3), np.uint8))
    ctrl = _controller(cam, _SpySink())
    with qtbot.waitSignal(ctrl.errorOccurred, timeout=2000):
        ctrl.start(
            source_path=tmp_path / "missing.jpg",
            snapshot=off_snapshot,
            mjpeg_port=0,
        )
    assert not ctrl.is_running()
    assert cam.started is False  # never got as far as building the camera


def test_double_start_is_noop(qtbot, off_snapshot, source_file):
    cams = []

    def cam_factory(device, w, h, fps):
        cam = _StubCam(np.zeros((4, 4, 3), np.uint8))
        cams.append(cam)
        return cam

    ctrl = LiveController(
        camera_factory=cam_factory, sink_factory=lambda port, fps: _SpySink()
    )
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        assert len(cams) == 1  # second start ignored while running
    finally:
        ctrl.stop()


def test_camera_open_failure_surfaces_error(off_snapshot, source_file):
    class _FailCam:
        opened = False
        error = "could not open capture device 9"

        def start(self):
            pass

        def read(self):
            return None

        def stop(self):
            pass

    cam = _FailCam()
    ctrl = LiveController(
        camera_factory=lambda *a: cam, sink_factory=lambda *a: _SpySink()
    )
    errors: list = []
    ctrl.errorOccurred.connect(errors.append)
    ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
    ctrl._check_camera()  # invoke the delayed open-failure check directly
    assert errors and "open" in errors[0].lower()
    assert not ctrl.is_running()


def test_camera_opened_but_no_frames_surfaces_error(off_snapshot, source_file):
    class _NoFramesCam:
        opened = True
        frames_seen = 0

        def start(self):
            pass

        def read(self):
            return None

        def stop(self):
            pass

    ctrl = LiveController(
        camera_factory=lambda *a: _NoFramesCam(), sink_factory=lambda *a: _SpySink()
    )
    errors: list = []
    ctrl.errorOccurred.connect(errors.append)
    ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
    ctrl._check_camera()
    assert errors and "no frames" in errors[0].lower()
    assert not ctrl.is_running()


def test_update_hot_swaps_chain_while_running(
    qtbot, off_snapshot, source_file, tmp_path
):
    from unittest.mock import MagicMock

    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        ctrl._loop.set_chain = MagicMock()  # type: ignore[union-attr]
        other = tmp_path / "face2.jpg"
        other.write_bytes(b"")
        ctrl.update(source_path=other, snapshot=off_snapshot)
        ctrl._loop.set_chain.assert_called_once()  # type: ignore[union-attr]
    finally:
        ctrl.stop()


def test_update_when_not_running_is_noop(off_snapshot, source_file):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    ctrl.update(source_path=source_file, snapshot=off_snapshot)  # must not crash
    assert not ctrl.is_running()


def test_update_bad_source_emits_error_but_keeps_running(
    qtbot, off_snapshot, source_file, tmp_path
):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    errors: list = []
    ctrl.errorOccurred.connect(errors.append)
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        ctrl.update(source_path=tmp_path / "missing.jpg", snapshot=off_snapshot)
        assert errors and "source" in errors[0].lower()
        assert ctrl.is_running()  # bad swap surfaced, session stayed up
    finally:
        ctrl.stop()


def test_sink_url_while_running(qtbot, off_snapshot, source_file):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        assert ctrl.sink_url() == "http://localhost:0/"
    finally:
        ctrl.stop()


def test_measured_fps_zero_when_not_running(off_snapshot, source_file):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    assert ctrl.measured_fps() == 0.0


def test_processing_fps_signal_emits_while_running(qtbot, off_snapshot, source_file):
    cam = _StubCam(np.zeros((8, 8, 3), np.uint8))
    ctrl = _controller(cam, _SpySink())
    try:
        with qtbot.waitSignal(ctrl.processingFpsChanged, timeout=2000) as blocker:
            ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        assert isinstance(blocker.args[0], float)
        assert blocker.args[0] >= 0.0
    finally:
        ctrl.stop()


def test_fps_timer_stops_on_stop(qtbot, off_snapshot, source_file):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
    qtbot.wait(50)
    ctrl.stop()
    seen: list = []
    ctrl.processingFpsChanged.connect(seen.append)
    qtbot.wait(300)  # well past the 200ms timer interval
    assert seen == []  # no emissions once stopped


def _spy_build_chain(captured):
    def _spy(source, **kwargs):
        captured.update(kwargs)
        return []  # empty chain -> raw passthrough, no model load
    return _spy


def test_detection_sink_passed_to_chain(off_snapshot, source_file, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("sinner2.gui.live_controller.build_chain",
                        _spy_build_chain(captured))
    sink = object()
    ctrl = LiveController(
        camera_factory=lambda *a: _StubCam(np.zeros((4, 4, 3), np.uint8)),
        sink_factory=lambda *a: _SpySink(),
        detection_sink=sink,
    )
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        assert captured.get("detection_sink") is sink  # forwarded, not None
    finally:
        ctrl.stop()


def test_set_detection_sink_before_start(off_snapshot, source_file, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("sinner2.gui.live_controller.build_chain",
                        _spy_build_chain(captured))
    sink = object()
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    ctrl.set_detection_sink(sink)
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        assert captured.get("detection_sink") is sink
    finally:
        ctrl.stop()


def test_start_passes_worker_count_to_loop(off_snapshot, source_file, monkeypatch):
    captured: dict = {}

    class _FakeLoop:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

        def stop(self):
            pass

        def measured_fps(self):
            return 0.0

    monkeypatch.setattr("sinner2.gui.live_controller.LiveLoop", _FakeLoop)
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot,
                   workers=5, mjpeg_port=0)
        assert captured.get("workers") == 5
    finally:
        ctrl.stop()


def test_set_worker_count_delegates_to_loop(off_snapshot, source_file, monkeypatch):
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake.measured_fps.return_value = 0.0
    monkeypatch.setattr("sinner2.gui.live_controller.LiveLoop",
                        lambda *a, **k: fake)
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        ctrl.set_worker_count(6)
        fake.set_worker_count.assert_called_once_with(6)
    finally:
        ctrl.stop()


def test_set_worker_count_noop_when_not_running(off_snapshot, source_file):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    ctrl.set_worker_count(4)  # no live session -> must not raise
    assert not ctrl.is_running()


def test_toggle_playback_stops_running_camera(qtbot, off_snapshot, source_file):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
    assert ctrl.is_running()
    ctrl.toggle_playback()  # Space = stop the capture
    assert not ctrl.is_running()


def test_toggle_playback_restarts_stopped_camera(qtbot, off_snapshot, source_file):
    ctrl = LiveController(
        camera_factory=lambda *a: _StubCam(np.zeros((4, 4, 3), np.uint8)),
        sink_factory=lambda *a: _SpySink(),
    )
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        ctrl.toggle_playback()  # stop
        assert not ctrl.is_running()
        ctrl.toggle_playback()  # restart with the remembered config
        assert ctrl.is_running()
    finally:
        ctrl.stop()


def test_toggle_playback_noop_when_never_started(off_snapshot, source_file):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    ctrl.toggle_playback()  # no remembered config -> no-op, no crash
    assert not ctrl.is_running()


def test_update_refreshes_restart_source(qtbot, off_snapshot, source_file, tmp_path):
    ctrl = _controller(_StubCam(np.zeros((4, 4, 3), np.uint8)), _SpySink())
    try:
        ctrl.start(source_path=source_file, snapshot=off_snapshot, mjpeg_port=0)
        other = tmp_path / "face2.jpg"
        other.write_bytes(b"")
        ctrl.update(source_path=other, snapshot=off_snapshot)
        # A later Space-restart uses the updated face, not the original.
        assert ctrl._restart_args["source_path"] == other  # noqa: SLF001
    finally:
        ctrl.stop()
