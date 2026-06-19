"""Camera settings panel (the ⚙️ Settings dialog's Camera tab).

Camera-only settings: capture device + resolution/fps/workers + MJPEG port, the
served-URL readout, and the "Allow camera mode" gate that reveals the 📹 toggle
button in the main window (the single source of truth for camera mode). Start/
stop is driven by THAT button, not here. Device probing is on-demand (Refresh),
never at construction — probing opens each camera, which is slow and can fight
other apps.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sinner2.pipeline.live.camera_source import available_cameras
from sinner2.pipeline.live.live_loop import MAX_LIVE_WORKERS


class QLiveView(QWidget):
    allowCameraToggled = Signal(bool)  # the gate → show/hide the 📹 button
    workersChanged = Signal(int)  # live worker-pool resize while running
    configChanged = Signal()  # device/res/fps/workers/port changed → persist

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._running = False

        # The gate: reveals the 📹 camera toggle in the main window. Camera mode
        # is opt-in, so it's off (and the button hidden) by default.
        self._allow_camera = QCheckBox("Allow camera mode")
        self._allow_camera.setToolTip(
            "Show the 📹 camera toggle in the main window so you can switch to "
            "live-camera mode. Off = camera mode hidden."
        )
        self._allow_camera.toggled.connect(self.allowCameraToggled)

        form = QFormLayout()

        self._device = QComboBox()
        # Only index 0 by default — the real device list isn't known until a
        # probe (Refresh), so don't imply more cameras than actually exist.
        self._device.addItem("Camera 0", 0)
        self._refresh = QPushButton("Refresh")
        self._refresh.clicked.connect(self._refresh_devices)
        device_row = QHBoxLayout()
        device_row.addWidget(self._device, stretch=1)
        device_row.addWidget(self._refresh)
        device_box = QWidget()
        device_box.setLayout(device_row)
        form.addRow("Camera", device_box)

        self._width = self._spin(160, 3840, 1280, step=16)
        self._height = self._spin(120, 2160, 720, step=16)
        self._fps = self._spin(1, 60, 30)
        self._workers = self._spin(1, MAX_LIVE_WORKERS, 1)
        self._workers.setToolTip(
            "Parallel processing threads for the live chain. More can raise "
            "throughput on a heavy chain (swap + enhance + upscale); adjustable "
            "while running."
        )
        self._workers.valueChanged.connect(self.workersChanged)
        self._port = self._spin(1, 65535, 8080)
        form.addRow("Width", self._width)
        form.addRow("Height", self._height)
        form.addRow("FPS", self._fps)
        form.addRow("Workers", self._workers)
        form.addRow("MJPEG port", self._port)
        # Any config change persists (main_window saves it to settings).
        self._device.currentIndexChanged.connect(self.configChanged)
        for spin in (self._width, self._height, self._fps, self._workers,
                     self._port):
            spin.valueChanged.connect(self.configChanged)

        self._url = QLabel("—")
        self._url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._url.setWordWrap(True)

        root = QVBoxLayout(self)
        root.addWidget(self._allow_camera)
        root.addLayout(form)
        root.addWidget(QLabel("Serving:"))
        root.addWidget(self._url)
        root.addStretch(1)

    @staticmethod
    def _spin(lo: int, hi: int, value: int, step: int = 1) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    # ---- read-back (main_window reads these on start) ----

    def device(self) -> int:
        return int(self._device.currentData())

    def width(self) -> int:
        return self._width.value()

    def height(self) -> int:
        return self._height.value()

    def fps(self) -> int:
        return self._fps.value()

    def workers(self) -> int:
        return self._workers.value()

    def port(self) -> int:
        return self._port.value()

    def set_config(
        self, *, device: int, width: int, height: int, fps: int,
        workers: int, mjpeg_port: int,
    ) -> None:
        """Restore persisted camera config WITHOUT emitting configChanged."""
        idx = self._device.findData(device)
        self._device.blockSignals(True)
        if idx < 0:  # not in the (un-probed) list yet — add it so it's selectable
            self._device.addItem(f"Camera {device}", device)
            idx = self._device.findData(device)
        self._device.setCurrentIndex(idx)
        self._device.blockSignals(False)
        for spin, value in (
            (self._width, width), (self._height, height), (self._fps, fps),
            (self._workers, workers), (self._port, mjpeg_port),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def allow_camera(self) -> bool:
        return self._allow_camera.isChecked()

    def set_allow_camera(self, on: bool) -> None:
        """Reflect the persisted gate WITHOUT emitting allowCameraToggled."""
        self._allow_camera.blockSignals(True)
        self._allow_camera.setChecked(bool(on))
        self._allow_camera.blockSignals(False)

    # ---- state driven by LiveController.runningChanged ----

    def set_running(self, running: bool) -> None:
        # Lock the device/resolution + the gate while a camera session runs
        # (workers stays adjustable live). Stop is via the 📹 button.
        self._running = running
        for w in (self._device, self._refresh, self._width, self._height,
                  self._fps, self._port, self._allow_camera):
            w.setEnabled(not running)

    def set_url(self, url: str | None) -> None:
        self._url.setText(url or "—")

    def _refresh_devices(self) -> None:
        found = available_cameras(max_probe=8)
        if not found:
            return
        current = self._device.currentData()
        self._device.clear()
        for i in found:
            self._device.addItem(f"Camera {i}", i)
        idx = self._device.findData(current)
        if idx >= 0:
            self._device.setCurrentIndex(idx)
