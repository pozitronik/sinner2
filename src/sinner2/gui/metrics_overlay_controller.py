"""The floating metrics overlay (FPS / buffer / latency) for the main window.

Owns the write/drop rate trackers and the show/hide/restore/reposition + per-tick
sample logic for the ``QMetricsOverlay`` widget. The widget itself stays owned by
the window (it's a child of the display and the window's resizeEvent repositions
it); this holds a reference to operate on it.

The player controller is reached through a getter because the overlay is built
before the controller exists (the overlay's snapshot_fn is a deferred callback).
This is the metrics half of the planned OverlayController — the threaded
face-detection overlay is a separate extraction.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sinner2.gui.widgets.metrics_overlay import CumulativeRateTracker, MetricsSample

if TYPE_CHECKING:
    from sinner2.config.settings import Settings
    from sinner2.gui.player_controller import PlayerController

_MARGIN_PX = 8


class MetricsOverlayController:
    def __init__(
        self,
        *,
        controller_getter: "Callable[[], PlayerController]",
        update_settings: Callable[..., None],
        settings_getter: "Callable[[], Settings]",
        write_rate: Any = None,
        drop_rate: Any = None,
    ) -> None:
        self._controller_getter = controller_getter
        self._update_settings = update_settings
        self._settings_getter = settings_getter
        # Cumulative trackers: injectable so tests can assert reset()/update().
        self._write_rate = write_rate if write_rate is not None else CumulativeRateTracker()
        self._drop_rate = drop_rate if drop_rate is not None else CumulativeRateTracker()
        self._overlay: Any = None

    def set_overlay(self, overlay: Any) -> None:
        """Hand over the window-owned QMetricsOverlay to operate on."""
        self._overlay = overlay

    def reset_rates(self) -> None:
        # Reset so the first reading after a (re-)show is a fresh baseline, not a
        # delta smeared over the whole interval the overlay was hidden (its timer
        # is stopped while hidden, freezing the trackers).
        self._write_rate.reset()
        self._drop_rate.reset()

    def reposition(self) -> None:
        # Anchor at the display's top-left with a small margin.
        hint = self._overlay.sizeHint()
        self._overlay.setGeometry(_MARGIN_PX, _MARGIN_PX, hint.width(), hint.height())

    def set_visible(self, on: bool) -> None:
        if on:
            self.reset_rates()
        self._overlay.setVisible(on)
        if on:
            self.reposition()
        self._update_settings(metrics_overlay_visible=on)

    def restore_state(self) -> bool:
        """Apply the persisted visibility to the overlay; returns it so the
        window can reflect it on the stats toggle button."""
        visible = bool(self._settings_getter().metrics_overlay_visible)
        if visible:
            self.reset_rates()
            self.reposition()
        self._overlay.setVisible(visible)
        return visible

    def sample(self) -> "MetricsSample | None":
        # Called by the overlay's QTimer (~10 Hz). None when no session is active
        # so the overlay shows the placeholder.
        executor = self._controller_getter().executor()
        if executor is None:
            self._write_rate.reset()
            self._drop_rate.reset()
            return None
        import time as _time

        now = _time.monotonic()
        buf_metrics = executor.metrics.get()
        write_fps = self._write_rate.update(buf_metrics.write_completed, now)
        drop_fps = self._drop_rate.update(buf_metrics.write_dropped, now)
        return MetricsSample(
            timestamp=now,
            read_fps=executor.reads_per_second(),
            process_fps=executor.processing_fps.get(),
            write_fps=write_fps,
            drop_fps=drop_fps,
            cache_hit_ratio=buf_metrics.cache_hit_ratio,
            memory_used_mb=buf_metrics.memory_used_bytes / 1024 / 1024,
            work_outstanding=0,  # not surfaced by executor today; placeholder
            work_capacity=0,
            write_outstanding=buf_metrics.write_outstanding,
            write_capacity=buf_metrics.write_max_outstanding,
            total_drops=buf_metrics.write_dropped,
            last_completed=executor.last_completed_frame(),
            processor_timings=executor.processor_timings(),
        )
