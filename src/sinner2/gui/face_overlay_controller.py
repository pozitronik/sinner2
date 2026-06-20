"""Face-detection overlay for the main window (the threaded half of the overlay
work; the metrics overlay is its own controller).

Owns the detection overlay widget, the FaceDetectionSink, the FaceDetectionProbe
running on its own QThread (so the live preview never stalls), and the poll
timer — plus the view logic that resolves WHICH overlay is up (face-map vs the
F8 diagnostic) and drives it. Shared face-map / scan state (`_use_face_map`,
`_faces_mode`, `_face_analyzing`) and the processor/controller/face-map
collaborators are READ through the window; this owns only overlay state.

`stop()` joins the probe thread (called from the window's closeEvent). The sink
is exposed so the session/controller/camera wiring can publish swapper
detections to the same sink the overlay polls.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal

from sinner2.gui.face_detection_probe import FaceDetectionProbe, FaceDetectionSink
from sinner2.gui.widgets.face_detection_overlay import (
    FaceDetection,
    QFaceDetectionOverlay,
)

if TYPE_CHECKING:
    from sinner2.types import Frame

# Set SINNER2_OVERLAY_TRACE=1 to log the detection-overlay poll state on each tick.
_OVERLAY_TRACE = bool(os.environ.get("SINNER2_OVERLAY_TRACE"))
_PROBE_INTERVAL_S = 0.15  # ~6 Hz: enough to track, cheap enough to stay smooth
_THREAD_JOIN_WAIT_MS = 2000
_THREAD_JOIN_MAX_WAITS = 15  # ~30s worst case before giving up and logging


class FaceOverlayController(QObject):
    _requestDetection = Signal(object, int, int)  # frame, width, height

    def __init__(self, window) -> None:  # type: ignore[no-untyped-def]
        super().__init__(window)
        self._window = window
        self._face_overlay_on = False
        self._comparison_on = False
        self._last_probe_feed = 0.0
        # Last frame handed to the display, kept so enabling the overlay can
        # detect the current frame immediately (e.g. while paused).
        self._last_displayed_frame: Frame | None = None
        # Frame index of the detections the overlay last DREW (from the sink). A
        # click-to-pick is rejected if the sink has advanced since (stale boxes).
        self._overlay_drawn_frame: int | None = None
        # A box drawn straight from the scan catalog when navigating to a found
        # face (show_catalog_face). PINNED: an EMPTY live result for the same
        # frame must not wipe it (a cached frame the swapper skips publishes
        # nothing; a single live re-detect can miss a face the scan caught). A
        # NON-empty live result supersedes it; the next seek clears it.
        self._pinned_box: tuple[float, float, float, float] | None = None
        self._face_overlay = QFaceDetectionOverlay(parent=window._display)
        window._display.set_face_overlay(self._face_overlay)
        self._face_overlay.faceClicked.connect(self._on_overlay_face_clicked)
        # The swapper publishes its pre-swap detections here; the probe fills it
        # in the swapper-off case. A timer polls it while the overlay is on.
        self._detection_sink = FaceDetectionSink()
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setInterval(int(_PROBE_INTERVAL_S * 1000))
        self._overlay_timer.timeout.connect(self._overlay_tick)
        self._detection_probe = FaceDetectionProbe(
            providers=window._settings.swapper_providers,
            detection_size=window._settings.swapper_detection_size or 640,
            sink=self._detection_sink,
        )
        self._detection_thread = QThread(self)
        self._detection_probe.moveToThread(self._detection_thread)
        self._detection_thread.start()
        self._requestDetection.connect(
            self._detection_probe.analyze, Qt.ConnectionType.QueuedConnection
        )
        self._detection_probe.detectionsReady.connect(
            self._on_detections, Qt.ConnectionType.QueuedConnection
        )
        window._display.frameDisplayed.connect(self._feed_detection_probe)

    @property
    def sink(self) -> FaceDetectionSink:
        """The detection sink — published to by the swapper (via the session /
        controller / camera) and polled by the overlay."""
        return self._detection_sink

    def configure_probe(self, providers, detection_size) -> None:  # type: ignore[no-untyped-def]
        """Keep the probe on the same providers/size as the swapper (a providers
        change resets the shared analyser; a stale probe list would rebuild it
        on the wrong EPs)."""
        self._detection_probe.configure(providers, detection_size)

    def stop(self) -> bool:
        """Quit + bounded-join the probe thread for shutdown. Returns False if it
        didn't stop (caller logs rather than destroy a running thread). The first
        detection can be lazily building buffalo_l (> 2s), so wait in increments."""
        thread = self._detection_thread
        thread.quit()
        waits = 0
        while thread.isRunning() and waits < _THREAD_JOIN_MAX_WAITS:
            thread.wait(_THREAD_JOIN_WAIT_MS)
            waits += 1
        return not thread.isRunning()

    # ---- mode resolution ----

    def _overlay_active(self) -> bool:
        return (
            self._face_map_overlay_on() or self._diagnostic_overlay_on()
        ) and not self._window._face_analyzing

    def _face_map_overlay_on(self) -> bool:
        return (
            self._window._use_face_map
            and self._window._faces_mode
            and self._window._face_map_panel.show_overlay()
        )

    def _diagnostic_overlay_on(self) -> bool:
        return self._face_overlay_on and not self._window._use_face_map

    # ---- state orchestration ----

    def _refresh_overlay_state(self) -> None:
        face_map = self._face_map_overlay_on() and not self._window._face_analyzing
        active = self._overlay_active()
        self._face_overlay.set_pick_enabled(face_map)  # pick = face-map overlay only
        self._refresh_overlay_modes()
        if active:
            self._face_overlay.setGeometry(self._window._display.rect())
            self._face_overlay.show()
            if not self._overlay_timer.isActive():
                self._overlay_timer.start()
            self._refresh_overlay_now()
        else:
            self._overlay_timer.stop()
            self._face_overlay.hide()
            self._face_overlay.clear()

    def _refresh_overlay_now(self) -> None:
        if self._window._processors.swapper_enabled():
            self._overlay_tick()
        elif self._last_displayed_frame is not None:
            self._submit_to_probe(self._last_displayed_frame)

    def _refresh_overlay_modes(self) -> None:
        # Comparison crops only for the diagnostic overlay (F8, editor closed)
        # with the toggle on — the swapper extracts them only then.
        comparison = self._diagnostic_overlay_on() and self._comparison_on
        self._detection_sink.set_wants_crops(comparison)
        self._face_overlay.set_comparison(comparison)

    def _refresh_face_highlight(self) -> None:
        bbox = (
            self._window._face_map_ctl.selected_face_bbox()
            if self._face_map_overlay_on() else None
        )
        self._face_overlay.set_highlight(bbox)

    def _clear_overlay_for_seek(self) -> None:
        if not self._overlay_active():
            return
        self._pinned_box = None
        self._detection_sink.clear()
        self._face_overlay.clear()

    def show_catalog_face(
        self, bbox: tuple[float, float, float, float], frame_w: int, frame_h: int
    ) -> None:
        """Draw a single face box straight from the scan catalog (no live
        detection) when navigating to a found face, so its box shows even on a
        cached frame the swapper skips — instead of relying on a live re-detect
        that can miss it. ``bbox`` is in native frame-pixel space (``frame_w`` ×
        ``frame_h``). Pins the box so an empty live result won't wipe it."""
        if not self._overlay_active():
            return
        box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        self._pinned_box = box
        self._face_overlay.set_detections(
            [FaceDetection(bbox=box)], frame_w, frame_h
        )
        self._face_overlay.set_highlight(box)

    # ---- toggles / restore ----

    def _apply_face_overlay_visible(self, on: bool) -> None:
        self._face_overlay_on = on
        self._refresh_overlay_state()

    def _set_face_overlay_visible(self, on: bool) -> None:
        self._apply_face_overlay_visible(on)
        if on:
            if self._window._processors.swapper_enabled():
                msg = "Face-detection overlay on — showing the face swapper's detections"
            else:
                msg = "Face-detection overlay on (F8)"
            self._window._status_bar.show_message(msg, 4000)
        self._window._update_settings(face_overlay_visible=on)

    def _restore_face_overlay_state(self) -> None:
        visible = bool(self._window._settings.face_overlay_visible)
        self._apply_face_overlay_visible(visible)
        self._window._processors.set_overlay_checked(visible)

    def _set_comparison_visible(self, on: bool) -> None:
        self._comparison_on = on
        self._refresh_overlay_modes()
        if on:
            # Force one reprocess so the current (paused) frame's crops publish now.
            executor = self._window._controller.executor()
            if executor is not None:
                current = executor.current_frame.get()
                if current >= 0:
                    executor.seek(current)
            if not (self._diagnostic_overlay_on() and self._window._processors.swapper_enabled()):
                self._window._status_bar.show_message(
                    "Comparison needs the face overlay (F8, Faces editor closed) "
                    "and the swapper on",
                    4000,
                )
        self._window._update_settings(face_comparison_visible=on)

    def _restore_comparison_state(self) -> None:
        on = bool(self._window._settings.face_comparison_visible)
        self._comparison_on = on
        self._window._processors.set_comparison_checked(on)
        self._refresh_overlay_modes()

    def _on_show_overlay_toggled(self, _on: bool) -> None:
        self._refresh_overlay_state()
        self._refresh_face_highlight()

    def _on_overlay_face_clicked(self, bbox: object) -> None:
        self._window._face_map_ctl.on_face_clicked(
            bbox, self._overlay_drawn_frame  # type: ignore[arg-type]
        )

    # ---- detection feed / poll ----

    def _overlay_tick(self) -> None:
        if _OVERLAY_TRACE:
            self._trace_overlay()
        if not self._overlay_active() or not self._window._processors.swapper_enabled():
            return
        latest = self._detection_sink.latest_detections()
        if latest is not None:
            detections, w, h = latest
            # A non-empty live result supersedes a pinned catalog box; an empty
            # one must not wipe it (see show_catalog_face).
            if detections or self._pinned_box is None:
                self._pinned_box = None
                self._face_overlay.set_detections(detections, w, h)
                raw = self._detection_sink.latest_raw()
                self._overlay_drawn_frame = raw[3] if raw is not None else None
                self._refresh_face_highlight()
        if self._comparison_on:
            crops = self._detection_sink.latest_crops()
            if crops is not None:
                pairs, w, h = crops
                self._face_overlay.set_crop_pairs(pairs, w, h)

    def _trace_overlay(self) -> None:
        import sys

        latest = self._detection_sink.latest_detections()
        sink_n = len(latest[0]) if latest is not None else None
        ex = self._window._controller.executor()
        print(
            f"[overlay] f8={self._face_overlay_on} "
            f"facesMode={self._window._faces_mode} "
            f"analyzing={self._window._face_analyzing} "
            f"swapper={self._window._processors.swapper_enabled()} "
            f"sinkFaces={sink_n} "
            f"shownFaces={len(self._face_overlay._detections)} "  # noqa: SLF001
            f"curFrame={ex.current_frame.get() if ex is not None else None}",
            file=sys.stderr, flush=True,
        )

    def _feed_detection_probe(self, frame: "Frame") -> None:
        # Always remember the latest frame; only probe when the overlay is on AND
        # the swapper is off (swapper-on uses its published detections instead).
        self._last_displayed_frame = frame
        if not self._overlay_active() or self._window._processors.swapper_enabled():
            return
        import time as _time

        if _time.monotonic() - self._last_probe_feed < _PROBE_INTERVAL_S:
            return
        self._submit_to_probe(frame)

    def _submit_to_probe(self, frame: "Frame") -> None:
        import time as _time

        self._last_probe_feed = _time.monotonic()
        h, w = frame.shape[:2]
        # Copy so the producer can't mutate the buffer under the probe thread.
        self._requestDetection.emit(frame.copy(), w, h)

    def _on_detections(self, detections: object, width: int, height: int) -> None:
        if not self._overlay_active():
            return
        # Keep a pinned catalog box rather than let an empty live re-detect wipe
        # it (see show_catalog_face); a non-empty result supersedes it.
        if not detections and self._pinned_box is not None:
            return
        self._pinned_box = None
        self._face_overlay.set_detections(detections, width, height)  # type: ignore[arg-type]
        raw = self._detection_sink.latest_raw()
        self._overlay_drawn_frame = raw[3] if raw is not None else None
        self._refresh_face_highlight()
