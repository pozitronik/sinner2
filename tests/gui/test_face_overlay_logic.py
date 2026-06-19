"""Characterization tests for SinnerMainWindow's face-detection overlay logic.

This surface (mode resolution, the overlay-state orchestrator, the probe/sink
feed decisions, comparison + highlight) had NO direct coverage. These tests pin
the CURRENT behaviour so the planned extraction into a face-detection
OverlayController is provably faithful. They drive a bare window
(object.__new__, no QObject init) with mocked collaborators — exactly the
state-flag + widget surface the logic reads.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from sinner2.config.settings import Settings
from sinner2.gui import main_window as mw


def _win(**overrides):
    win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
    # State flags (defaults = everything off / idle).
    win._face_overlay_on = False
    win._use_face_map = False
    win._faces_mode = False
    win._face_analyzing = False
    win._comparison_on = False
    win._batch_active = False
    win._overlay_drawn_frame = None
    win._last_displayed_frame = None
    win._last_probe_feed = 0.0
    # Mocked collaborators.
    win._face_overlay = MagicMock()
    win._detection_sink = MagicMock()
    win._face_map_panel = MagicMock()
    win._face_map_panel.show_overlay.return_value = True
    win._face_map_ctl = MagicMock()
    win._processors = MagicMock()
    win._processors.swapper_enabled.return_value = False
    win._controller = MagicMock()
    win._status_bar = MagicMock()
    win._display = MagicMock()
    win._overlay_timer = MagicMock()
    for key, value in overrides.items():
        setattr(win, key, value)
    return win


class TestModeResolution:
    def test_face_map_overlay_needs_use_map_and_faces_mode_and_show_overlay(self):
        # All three on → face-map overlay on.
        win = _win(_use_face_map=True, _faces_mode=True)
        assert win._face_map_overlay_on() is True
        # Any one off → off.
        assert _win(_use_face_map=False, _faces_mode=True)._face_map_overlay_on() is False
        assert _win(_use_face_map=True, _faces_mode=False)._face_map_overlay_on() is False
        win_no_show = _win(_use_face_map=True, _faces_mode=True)
        win_no_show._face_map_panel.show_overlay.return_value = False
        assert win_no_show._face_map_overlay_on() is False

    def test_diagnostic_overlay_is_f8_and_not_face_map_mode(self):
        assert _win(_face_overlay_on=True, _use_face_map=False)._diagnostic_overlay_on() is True
        # F8 is suppressed in face-map mode (the face-map overlay owns the surface).
        assert _win(_face_overlay_on=True, _use_face_map=True)._diagnostic_overlay_on() is False
        assert _win(_face_overlay_on=False)._diagnostic_overlay_on() is False

    def test_overlay_active_when_either_overlay_wants_it_and_not_scanning(self):
        assert _win(_face_overlay_on=True)._overlay_active() is True  # diagnostic
        assert _win(_use_face_map=True, _faces_mode=True)._overlay_active() is True  # face-map
        # A running scan owns the display → overlay forced down.
        assert _win(_face_overlay_on=True, _face_analyzing=True)._overlay_active() is False
        assert _win()._overlay_active() is False  # nothing wants it


class TestRefreshOverlayState:
    def test_active_shows_overlay_and_starts_timer(self):
        win = _win(_face_overlay_on=True)
        win._refresh_overlay_modes = MagicMock()
        win._refresh_overlay_now = MagicMock()
        win._overlay_timer.isActive.return_value = False
        win._refresh_overlay_state()
        win._face_overlay.show.assert_called_once()
        win._overlay_timer.start.assert_called_once()
        win._face_overlay.set_pick_enabled.assert_called_once_with(False)  # diagnostic, not pickable
        win._refresh_overlay_now.assert_called_once()

    def test_inactive_hides_overlay_and_stops_timer(self):
        win = _win()  # nothing wants the overlay
        win._refresh_overlay_modes = MagicMock()
        win._refresh_overlay_state()
        win._face_overlay.hide.assert_called_once()
        win._face_overlay.clear.assert_called_once()
        win._overlay_timer.stop.assert_called_once()

    def test_face_map_mode_makes_overlay_pickable(self):
        win = _win(_use_face_map=True, _faces_mode=True)
        win._refresh_overlay_modes = MagicMock()
        win._refresh_overlay_now = MagicMock()
        win._overlay_timer.isActive.return_value = True
        win._refresh_overlay_state()
        win._face_overlay.set_pick_enabled.assert_called_once_with(True)


class TestRefreshOverlayModes:
    def test_comparison_wanted_only_for_diagnostic_overlay_with_toggle_on(self):
        win = _win(_face_overlay_on=True, _comparison_on=True)  # diagnostic + comparison
        win._refresh_overlay_modes()
        win._detection_sink.set_wants_crops.assert_called_once_with(True)
        win._face_overlay.set_comparison.assert_called_once_with(True)

    def test_no_comparison_crops_in_face_map_mode(self):
        win = _win(_use_face_map=True, _faces_mode=True, _comparison_on=True)
        win._refresh_overlay_modes()
        win._detection_sink.set_wants_crops.assert_called_once_with(False)


class TestClearForSeek:
    def test_clears_sink_and_overlay_when_active(self):
        win = _win(_face_overlay_on=True)
        win._clear_overlay_for_seek()
        win._detection_sink.clear.assert_called_once()
        win._face_overlay.clear.assert_called_once()

    def test_noop_when_overlay_down(self):
        win = _win()  # inactive
        win._clear_overlay_for_seek()
        win._detection_sink.clear.assert_not_called()


class TestFaceHighlight:
    def test_highlights_selected_bbox_in_face_map_mode(self):
        win = _win(_use_face_map=True, _faces_mode=True)
        win._face_map_ctl.selected_face_bbox.return_value = (1, 2, 3, 4)
        win._refresh_face_highlight()
        win._face_overlay.set_highlight.assert_called_once_with((1, 2, 3, 4))

    def test_clears_highlight_outside_face_map_mode(self):
        win = _win(_face_overlay_on=True)  # diagnostic, not face-map
        win._refresh_face_highlight()
        win._face_overlay.set_highlight.assert_called_once_with(None)


class TestApplyAndRestore:
    def test_apply_sets_flag_and_refreshes(self):
        win = _win()
        win._refresh_overlay_state = MagicMock()
        win._apply_face_overlay_visible(True)
        assert win._face_overlay_on is True
        win._refresh_overlay_state.assert_called_once()

    def test_restore_face_overlay_applies_persisted_and_checks_button(self):
        win = _win(_settings=Settings(face_overlay_visible=True))
        win._refresh_overlay_state = MagicMock()
        win._restore_face_overlay_state()
        assert win._face_overlay_on is True
        win._processors.set_overlay_checked.assert_called_once_with(True)

    def test_restore_comparison_applies_persisted_and_checks_box(self):
        win = _win(_settings=Settings(face_comparison_visible=True))
        win._refresh_overlay_modes = MagicMock()
        win._restore_comparison_state()
        assert win._comparison_on is True
        win._processors.set_comparison_checked.assert_called_once_with(True)
        win._refresh_overlay_modes.assert_called_once()


class TestComparisonVisible:
    def test_set_comparison_sets_flag_and_refreshes(self, monkeypatch):
        monkeypatch.setattr(mw.user_settings, "save", lambda _s: None)
        win = _win(_settings=Settings())
        win._refresh_overlay_modes = MagicMock()
        win._controller.executor.return_value = None  # no live reprocess
        win._set_comparison_visible(True)
        assert win._comparison_on is True
        win._refresh_overlay_modes.assert_called_once()
        assert win._settings.face_comparison_visible is True  # persisted


class TestProbeFeed:
    def test_always_remembers_latest_frame(self):
        win = _win()  # overlay down
        win._submit_to_probe = MagicMock()
        frame = MagicMock()
        win._feed_detection_probe(frame)
        assert win._last_displayed_frame is frame
        win._submit_to_probe.assert_not_called()  # overlay off → no detection cost

    def test_submits_when_overlay_on_and_swapper_off(self):
        win = _win(_face_overlay_on=True)  # active, swapper off
        win._submit_to_probe = MagicMock()
        win._feed_detection_probe(MagicMock())
        win._submit_to_probe.assert_called_once()

    def test_does_not_probe_when_swapper_on(self):
        win = _win(_face_overlay_on=True)
        win._processors.swapper_enabled.return_value = True  # swapper publishes instead
        win._submit_to_probe = MagicMock()
        win._feed_detection_probe(MagicMock())
        win._submit_to_probe.assert_not_called()


class TestOverlayTick:
    def test_swapper_on_polls_sink_and_records_drawn_frame(self):
        win = _win(_face_overlay_on=True)
        win._processors.swapper_enabled.return_value = True
        win._refresh_face_highlight = MagicMock()
        win._detection_sink.latest_detections.return_value = (["det"], 640, 480)
        win._detection_sink.latest_raw.return_value = ("a", "b", "c", 42)
        win._overlay_tick()
        win._face_overlay.set_detections.assert_called_once_with(["det"], 640, 480)
        assert win._overlay_drawn_frame == 42

    def test_noop_when_swapper_off(self):
        win = _win(_face_overlay_on=True)  # swapper off → probe path, not tick
        win._overlay_tick()
        win._face_overlay.set_detections.assert_not_called()


class TestOnDetections:
    def test_draws_when_active_and_records_drawn_frame(self):
        win = _win(_face_overlay_on=True)
        win._detection_sink.latest_raw.return_value = ("a", "b", "c", 7)
        win._on_detections(["d"], 320, 240)
        win._face_overlay.set_detections.assert_called_once_with(["d"], 320, 240)
        assert win._overlay_drawn_frame == 7

    def test_noop_when_overlay_down(self):
        win = _win()  # inactive
        win._on_detections(["d"], 320, 240)
        win._face_overlay.set_detections.assert_not_called()


class TestFaceClicked:
    def test_routes_pick_with_drawn_frame(self):
        win = _win(_overlay_drawn_frame=42)
        win._on_overlay_face_clicked((10, 20, 30, 40))
        win._face_map_ctl.on_face_clicked.assert_called_once_with((10, 20, 30, 40), 42)
