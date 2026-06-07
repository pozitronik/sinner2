"""Coordinates a live-camera session: CameraSource → chain → LiveLoop → MjpegSink
+ a preview signal.

A sibling of PlayerController for the live path. It builds the chain from a
`ProcessorParamsSnapshot` (the same widget snapshot the file path uses) with the
chosen source face, wires a CameraSource into a LiveLoop feeding an MjpegSink,
and re-emits each processed frame as a queued Qt signal so the GUI preview
updates on the GUI thread. Capture/sink construction is injectable for tests.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from sinner2.config.source import Source
from sinner2.gui.processor_snapshot import ProcessorParamsSnapshot
from sinner2.pipeline.chain_builder import build_chain
from sinner2.pipeline.live.camera_source import CameraSource
from sinner2.pipeline.live.live_loop import LiveLoop
from sinner2.pipeline.live.sink import MjpegSink
from sinner2.types import Frame

CameraFactory = Callable[[Any, int, int, int], Any]
SinkFactory = Callable[[int, int], MjpegSink]


class LiveController(QObject):
    frameReady = Signal(object)   # processed Frame — queued to the GUI thread
    errorOccurred = Signal(str)
    runningChanged = Signal(bool)
    processingFpsChanged = Signal(float)  # measured live throughput (emitted fps)

    def __init__(
        self,
        *,
        camera_factory: CameraFactory | None = None,
        sink_factory: SinkFactory | None = None,
        detection_sink: object | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._camera_factory: CameraFactory = camera_factory or (
            lambda device, w, h, fps: CameraSource(device, w, h, fps)
        )
        self._sink_factory: SinkFactory = sink_factory or (
            lambda port, fps: MjpegSink(port=port, fps=fps)
        )
        # Shared FaceDetectionSink (same object the file path uses) so the live
        # chain's swapper publishes detections to the GUI overlay + crop probe.
        self._detection_sink = detection_sink
        self._loop: LiveLoop | None = None
        self._sink: MjpegSink | None = None
        self._camera: Any = None
        # Poll the loop's measured fps (~5 Hz) and re-emit on the GUI thread.
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(200)
        self._fps_timer.timeout.connect(self._emit_fps)

    def set_detection_sink(self, sink: object | None) -> None:
        """Wire the shared detection sink (call before start). Mirrors
        PlayerController.set_detection_sink so the live overlay works too."""
        self._detection_sink = sink

    def measured_fps(self) -> float:
        return self._loop.measured_fps() if self._loop is not None else 0.0

    def _emit_fps(self) -> None:
        self.processingFpsChanged.emit(self.measured_fps())

    def set_worker_count(self, n: int) -> None:
        """Resize the live worker pool while running (no-op when stopped)."""
        if self._loop is not None:
            self._loop.set_worker_count(n)

    def is_running(self) -> bool:
        return self._loop is not None

    def start(
        self,
        *,
        source_path: Path,
        snapshot: ProcessorParamsSnapshot,
        device: Any = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        workers: int = 1,
        mjpeg_port: int = 8080,
    ) -> None:
        """Build + start a live session. Source is the face to apply (a still);
        the camera is the live target. No-op if already running."""
        if self._loop is not None:
            return
        chain = self._build_chain(source_path, snapshot)
        if chain is None:
            return
        camera = self._camera_factory(device, width, height, fps)
        self._camera = camera
        self._sink = self._sink_factory(mjpeg_port, fps)
        self._loop = LiveLoop(
            camera, chain, [self._sink], on_frame=self._emit_frame,
            fps=fps, workers=workers,
        )
        try:
            self._loop.start()
        except Exception as exc:  # noqa: BLE001 — e.g. MJPEG port already in use
            self.errorOccurred.emit(f"failed to start live session: {exc}")
            self._loop = None
            self._sink = None
            self._camera = None
            return
        print(f"[live] MJPEG sink: {self._sink.describe()}", file=sys.stderr)
        self._fps_timer.start()
        self.runningChanged.emit(True)
        # The device opens on the capture thread; surface a failure shortly after
        # (non-blocking) so a bad camera shows an error instead of a blank panel.
        QTimer.singleShot(1500, self._check_camera)

    def update(
        self, *, source_path: Path, snapshot: ProcessorParamsSnapshot
    ) -> None:
        """Hot-apply a new source face / processor settings to the running
        session by rebuilding the chain and swapping it in. No-op if not
        running. The new chain's models load on a side thread; the live feed
        keeps running the current chain until the swap completes."""
        if self._loop is None:
            return
        chain = self._build_chain(source_path, snapshot)
        if chain is None:
            return
        self._loop.set_chain(chain)

    def _build_chain(
        self, source_path: Path, snapshot: ProcessorParamsSnapshot
    ) -> list[Any] | None:
        try:
            source = Source(path=source_path)
        except Exception as exc:  # noqa: BLE001 — surface bad source to the GUI
            self.errorOccurred.emit(f"invalid source: {exc}")
            return None
        return build_chain(
            source,
            swapper_enabled=snapshot.swapper_enabled,
            swapper_params=snapshot.swapper_params,
            swapper_providers=snapshot.swapper_providers,
            detection_sink=self._detection_sink,
            enhancer_enabled=snapshot.enhancer_enabled,
            enhancer_params=snapshot.enhancer_params,
            enhancer_device=snapshot.enhancer_device,
            upscaler_enabled=snapshot.upscaler_enabled,
            upscaler_params=snapshot.upscaler_params,
            upscaler_device=snapshot.upscaler_device,
        )

    def _check_camera(self) -> None:
        cam = self._camera
        if cam is None or self._loop is None:
            return
        if not getattr(cam, "opened", True):
            self.errorOccurred.emit(
                getattr(cam, "error", None) or "camera failed to open"
            )
            self.stop()
        elif getattr(cam, "frames_seen", 1) == 0:
            self.errorOccurred.emit(
                "camera opened but delivered no frames — try Refresh to pick a "
                "working camera, or a lower resolution"
            )
            self.stop()

    def _emit_frame(self, frame: Frame) -> None:
        # Called on the loop thread; the queued Signal hops it to the GUI thread.
        self.frameReady.emit(frame)

    def stop(self) -> None:
        if self._loop is None:
            return
        self._fps_timer.stop()
        self._loop.stop()
        self._loop = None
        self._sink = None
        self._camera = None
        self.runningChanged.emit(False)

    def sink_url(self) -> str | None:
        return self._sink.describe() if self._sink is not None else None
