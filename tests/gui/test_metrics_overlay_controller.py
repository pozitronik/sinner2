"""Unit tests for MetricsOverlayController (the floating metrics-overlay logic
extracted from SinnerMainWindow)."""
from __future__ import annotations

from unittest.mock import MagicMock

from sinner2.config.settings import Settings
from sinner2.gui.metrics_overlay_controller import MetricsOverlayController


def _make(executor=None, settings=None):
    wr, dr, overlay = MagicMock(), MagicMock(), MagicMock()
    controller = MagicMock()
    controller.executor.return_value = executor
    updates: list[dict] = []
    ctl = MetricsOverlayController(
        controller_getter=lambda: controller,
        update_settings=lambda **k: updates.append(k),
        settings_getter=lambda: settings or Settings(),
        write_rate=wr,
        drop_rate=dr,
    )
    ctl.set_overlay(overlay)
    return ctl, wr, dr, overlay, updates


class TestSetVisible:
    def test_show_resets_rates_shows_repositions_persists(self):
        ctl, wr, dr, overlay, updates = _make()
        ctl.set_visible(True)
        wr.reset.assert_called_once()
        dr.reset.assert_called_once()
        overlay.setVisible.assert_called_once_with(True)
        overlay.setGeometry.assert_called_once()  # repositioned
        assert updates == [{"metrics_overlay_visible": True}]

    def test_hide_does_not_reset_or_reposition(self):
        ctl, wr, _dr, overlay, updates = _make()
        ctl.set_visible(False)
        wr.reset.assert_not_called()
        overlay.setGeometry.assert_not_called()
        overlay.setVisible.assert_called_once_with(False)
        assert updates == [{"metrics_overlay_visible": False}]


class TestRestoreState:
    def test_returns_persisted_visibility_and_applies(self):
        ctl, _wr, _dr, overlay, _ = _make(
            settings=Settings(metrics_overlay_visible=True)
        )
        assert ctl.restore_state() is True
        overlay.setVisible.assert_called_once_with(True)

    def test_hidden_when_unset(self):
        ctl, _wr, _dr, overlay, _ = _make(settings=Settings())
        assert ctl.restore_state() is False
        overlay.setVisible.assert_called_once_with(False)


class TestSample:
    def test_returns_none_and_resets_rates_when_no_session(self):
        ctl, wr, dr, _o, _ = _make(executor=None)
        assert ctl.sample() is None
        wr.reset.assert_called_once()
        dr.reset.assert_called_once()

    def test_builds_sample_from_executor_metrics(self):
        executor = MagicMock()
        executor.reads_per_second.return_value = 25.0
        executor.processing_fps.get.return_value = 30.0
        executor.last_completed_frame.return_value = 42
        executor.processor_timings.return_value = {}
        bm = executor.metrics.get.return_value
        bm.write_completed = 100
        bm.write_dropped = 2
        bm.cache_hit_ratio = 0.9
        bm.memory_used_bytes = 1024 * 1024 * 10
        bm.write_outstanding = 1
        bm.write_max_outstanding = 4
        ctl, wr, dr, _o, _ = _make(executor=executor)
        wr.update.return_value = 30.0
        dr.update.return_value = 0.0
        sample = ctl.sample()
        assert sample is not None
        assert sample.write_fps == 30.0
        assert sample.cache_hit_ratio == 0.9
        assert sample.memory_used_mb == 10.0
