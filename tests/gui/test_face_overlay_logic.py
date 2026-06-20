"""Characterization tests for FaceOverlayController (the face-detection overlay).

These pin mode resolution, the overlay-state orchestrator, the probe/sink feed
decisions, comparison + highlight, tick and click routing. They were written
against SinnerMainWindow BEFORE the extraction and migrated here verbatim (same
assertions) — so they prove the move into FaceOverlayController is faithful.

The controller is driven via object.__new__ (no QObject init / no real thread)
with a mock window for the shared face-map/scan state it reads, and mocked
overlay/sink/timer for the state it owns.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from sinner2.config.settings import Settings
from sinner2.gui.face_overlay_controller import FaceOverlayController

# Keys the controller READS through the window vs the state it OWNS.
_WINDOW_READ = {"_use_face_map", "_faces_mode", "_face_analyzing", "_settings"}


def _ctl(**overrides):
    win = MagicMock()
    win._use_face_map = False
    win._faces_mode = False
    win._face_analyzing = False
    win._face_map_panel.show_overlay.return_value = True
    win._processors.swapper_enabled.return_value = False
    ctl = FaceOverlayController.__new__(FaceOverlayController)
    ctl._window = win
    ctl._face_overlay = MagicMock()
    ctl._detection_sink = MagicMock()
    ctl._overlay_timer = MagicMock()
    ctl._face_overlay_on = False
    ctl._comparison_on = False
    ctl._overlay_drawn_frame = None
    ctl._pinned_box = None
    ctl._last_displayed_frame = None
    ctl._last_probe_feed = 0.0
    for key, value in overrides.items():
        setattr(win if key in _WINDOW_READ else ctl, key, value)
    return ctl, win


class TestModeResolution:
    def test_face_map_overlay_needs_use_map_and_faces_mode_and_show_overlay(self):
        ctl, _ = _ctl(_use_face_map=True, _faces_mode=True)
        assert ctl._face_map_overlay_on() is True
        assert _ctl(_use_face_map=False, _faces_mode=True)[0]._face_map_overlay_on() is False
        assert _ctl(_use_face_map=True, _faces_mode=False)[0]._face_map_overlay_on() is False
        ctl_no, win = _ctl(_use_face_map=True, _faces_mode=True)
        win._face_map_panel.show_overlay.return_value = False
        assert ctl_no._face_map_overlay_on() is False

    def test_diagnostic_overlay_is_f8_and_not_face_map_mode(self):
        assert _ctl(_face_overlay_on=True, _use_face_map=False)[0]._diagnostic_overlay_on() is True
        assert _ctl(_face_overlay_on=True, _use_face_map=True)[0]._diagnostic_overlay_on() is False
        assert _ctl(_face_overlay_on=False)[0]._diagnostic_overlay_on() is False

    def test_overlay_active_when_either_overlay_wants_it_and_not_scanning(self):
        assert _ctl(_face_overlay_on=True)[0]._overlay_active() is True
        assert _ctl(_use_face_map=True, _faces_mode=True)[0]._overlay_active() is True
        assert _ctl(_face_overlay_on=True, _face_analyzing=True)[0]._overlay_active() is False
        assert _ctl()[0]._overlay_active() is False


class TestRefreshOverlayState:
    def test_active_shows_overlay_and_starts_timer(self):
        ctl, _ = _ctl(_face_overlay_on=True)
        ctl._refresh_overlay_modes = MagicMock()
        ctl._refresh_overlay_now = MagicMock()
        ctl._overlay_timer.isActive.return_value = False
        ctl._refresh_overlay_state()
        ctl._face_overlay.show.assert_called_once()
        ctl._overlay_timer.start.assert_called_once()
        ctl._face_overlay.set_pick_enabled.assert_called_once_with(False)
        ctl._refresh_overlay_now.assert_called_once()

    def test_inactive_hides_overlay_and_stops_timer(self):
        ctl, _ = _ctl()
        ctl._refresh_overlay_modes = MagicMock()
        ctl._refresh_overlay_state()
        ctl._face_overlay.hide.assert_called_once()
        ctl._face_overlay.clear.assert_called_once()
        ctl._overlay_timer.stop.assert_called_once()

    def test_face_map_mode_makes_overlay_pickable(self):
        ctl, _ = _ctl(_use_face_map=True, _faces_mode=True)
        ctl._refresh_overlay_modes = MagicMock()
        ctl._refresh_overlay_now = MagicMock()
        ctl._overlay_timer.isActive.return_value = True
        ctl._refresh_overlay_state()
        ctl._face_overlay.set_pick_enabled.assert_called_once_with(True)


class TestRefreshOverlayModes:
    def test_comparison_wanted_only_for_diagnostic_overlay_with_toggle_on(self):
        ctl, _ = _ctl(_face_overlay_on=True, _comparison_on=True)
        ctl._refresh_overlay_modes()
        ctl._detection_sink.set_wants_crops.assert_called_once_with(True)
        ctl._face_overlay.set_comparison.assert_called_once_with(True)

    def test_no_comparison_crops_in_face_map_mode(self):
        ctl, _ = _ctl(_use_face_map=True, _faces_mode=True, _comparison_on=True)
        ctl._refresh_overlay_modes()
        ctl._detection_sink.set_wants_crops.assert_called_once_with(False)


class TestClearForSeek:
    def test_clears_sink_and_overlay_when_active(self):
        ctl, _ = _ctl(_face_overlay_on=True)
        ctl._clear_overlay_for_seek()
        ctl._detection_sink.clear.assert_called_once()
        ctl._face_overlay.clear.assert_called_once()

    def test_noop_when_overlay_down(self):
        ctl, _ = _ctl()
        ctl._clear_overlay_for_seek()
        ctl._detection_sink.clear.assert_not_called()


class TestFaceHighlight:
    def test_highlights_selected_bbox_in_face_map_mode(self):
        ctl, win = _ctl(_use_face_map=True, _faces_mode=True)
        win._face_map_ctl.selected_face_bbox.return_value = (1, 2, 3, 4)
        ctl._refresh_face_highlight()
        ctl._face_overlay.set_highlight.assert_called_once_with((1, 2, 3, 4))

    def test_clears_highlight_outside_face_map_mode(self):
        ctl, _ = _ctl(_face_overlay_on=True)
        ctl._refresh_face_highlight()
        ctl._face_overlay.set_highlight.assert_called_once_with(None)


class TestApplyAndRestore:
    def test_apply_sets_flag_and_refreshes(self):
        ctl, _ = _ctl()
        ctl._refresh_overlay_state = MagicMock()
        ctl._apply_face_overlay_visible(True)
        assert ctl._face_overlay_on is True
        ctl._refresh_overlay_state.assert_called_once()

    def test_restore_face_overlay_applies_persisted_and_checks_button(self):
        ctl, win = _ctl(_settings=Settings(face_overlay_visible=True))
        ctl._refresh_overlay_state = MagicMock()
        ctl._restore_face_overlay_state()
        assert ctl._face_overlay_on is True
        win._processors.set_overlay_checked.assert_called_once_with(True)

    def test_restore_comparison_applies_persisted_and_checks_box(self):
        ctl, win = _ctl(_settings=Settings(face_comparison_visible=True))
        ctl._refresh_overlay_modes = MagicMock()
        ctl._restore_comparison_state()
        assert ctl._comparison_on is True
        win._processors.set_comparison_checked.assert_called_once_with(True)
        ctl._refresh_overlay_modes.assert_called_once()


class TestComparisonVisible:
    def test_set_comparison_sets_flag_and_persists(self):
        ctl, win = _ctl(_settings=Settings())
        ctl._refresh_overlay_modes = MagicMock()
        win._controller.executor.return_value = None  # no live reprocess
        ctl._set_comparison_visible(True)
        assert ctl._comparison_on is True
        ctl._refresh_overlay_modes.assert_called_once()
        win._update_settings.assert_called_once_with(face_comparison_visible=True)


class TestProbeFeed:
    def test_always_remembers_latest_frame(self):
        ctl, _ = _ctl()  # overlay down
        ctl._submit_to_probe = MagicMock()
        frame = MagicMock()
        ctl._feed_detection_probe(frame)
        assert ctl._last_displayed_frame is frame
        ctl._submit_to_probe.assert_not_called()

    def test_submits_when_overlay_on_and_swapper_off(self):
        ctl, _ = _ctl(_face_overlay_on=True)
        ctl._submit_to_probe = MagicMock()
        ctl._feed_detection_probe(MagicMock())
        ctl._submit_to_probe.assert_called_once()

    def test_does_not_probe_when_swapper_on(self):
        ctl, win = _ctl(_face_overlay_on=True)
        win._processors.swapper_enabled.return_value = True
        ctl._submit_to_probe = MagicMock()
        ctl._feed_detection_probe(MagicMock())
        ctl._submit_to_probe.assert_not_called()


class TestOverlayTick:
    def test_swapper_on_polls_sink_and_records_drawn_frame(self):
        ctl, win = _ctl(_face_overlay_on=True)
        win._processors.swapper_enabled.return_value = True
        ctl._refresh_face_highlight = MagicMock()
        ctl._detection_sink.latest_detections.return_value = (["det"], 640, 480)
        ctl._detection_sink.latest_raw.return_value = ("a", "b", "c", 42)
        ctl._overlay_tick()
        ctl._face_overlay.set_detections.assert_called_once_with(["det"], 640, 480)
        assert ctl._overlay_drawn_frame == 42

    def test_noop_when_swapper_off(self):
        ctl, _ = _ctl(_face_overlay_on=True)
        ctl._overlay_tick()
        ctl._face_overlay.set_detections.assert_not_called()


class TestOnDetections:
    def test_draws_when_active_and_records_drawn_frame(self):
        ctl, _ = _ctl(_face_overlay_on=True)
        ctl._refresh_face_highlight = MagicMock()
        ctl._detection_sink.latest_raw.return_value = ("a", "b", "c", 7)
        ctl._on_detections(["d"], 320, 240)
        ctl._face_overlay.set_detections.assert_called_once_with(["d"], 320, 240)
        assert ctl._overlay_drawn_frame == 7

    def test_noop_when_overlay_down(self):
        ctl, _ = _ctl()
        ctl._on_detections(["d"], 320, 240)
        ctl._face_overlay.set_detections.assert_not_called()


class TestFaceClicked:
    def test_routes_pick_with_drawn_frame(self):
        ctl, win = _ctl(_overlay_drawn_frame=42)
        ctl._on_overlay_face_clicked((10, 20, 30, 40))
        win._face_map_ctl.on_face_clicked.assert_called_once_with((10, 20, 30, 40), 42)


class TestCatalogFace:
    """show_catalog_face draws the scanned box on navigate and PINS it so an
    empty live re-detect (or a cached frame the swapper skips) can't wipe it,
    while a real detection supersedes it."""

    def test_draws_and_pins_when_active(self):
        ctl, _ = _ctl(_face_overlay_on=True)
        ctl.show_catalog_face((1.0, 2.0, 3.0, 4.0), 640, 480)
        dets, w, h = ctl._face_overlay.set_detections.call_args.args
        assert len(dets) == 1 and dets[0].bbox == (1.0, 2.0, 3.0, 4.0)
        assert (w, h) == (640, 480)
        ctl._face_overlay.set_highlight.assert_called_once_with((1.0, 2.0, 3.0, 4.0))
        assert ctl._pinned_box == (1.0, 2.0, 3.0, 4.0)

    def test_noop_when_overlay_down(self):
        ctl, _ = _ctl()  # overlay inactive
        ctl.show_catalog_face((1.0, 2.0, 3.0, 4.0), 640, 480)
        ctl._face_overlay.set_detections.assert_not_called()
        assert ctl._pinned_box is None

    def test_empty_detections_keep_the_pin(self):
        ctl, _ = _ctl(_face_overlay_on=True, _pinned_box=(1.0, 2.0, 3.0, 4.0))
        ctl._refresh_face_highlight = MagicMock()
        ctl._on_detections([], 640, 480)
        ctl._face_overlay.set_detections.assert_not_called()
        assert ctl._pinned_box == (1.0, 2.0, 3.0, 4.0)

    def test_nonempty_detections_supersede_the_pin(self):
        ctl, _ = _ctl(_face_overlay_on=True, _pinned_box=(1.0, 2.0, 3.0, 4.0))
        ctl._refresh_face_highlight = MagicMock()
        ctl._detection_sink.latest_raw.return_value = ("a", "b", "c", 9)
        ctl._on_detections(["d"], 320, 240)
        ctl._face_overlay.set_detections.assert_called_once_with(["d"], 320, 240)
        assert ctl._pinned_box is None

    def test_seek_clears_the_pin(self):
        ctl, _ = _ctl(_face_overlay_on=True, _pinned_box=(1.0, 2.0, 3.0, 4.0))
        ctl._clear_overlay_for_seek()
        assert ctl._pinned_box is None
        ctl._detection_sink.clear.assert_called_once()

    def test_overlay_tick_keeps_pin_on_empty_sink(self):
        ctl, win = _ctl(_face_overlay_on=True, _pinned_box=(1.0, 2.0, 3.0, 4.0))
        win._processors.swapper_enabled.return_value = True
        ctl._refresh_face_highlight = MagicMock()
        ctl._detection_sink.latest_detections.return_value = ([], 640, 480)
        ctl._overlay_tick()
        ctl._face_overlay.set_detections.assert_not_called()
        assert ctl._pinned_box == (1.0, 2.0, 3.0, 4.0)
