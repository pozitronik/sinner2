from pathlib import Path

import pytest

from sinner2.gui.main_window import SinnerMainWindow


@pytest.fixture
def window(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Point settings at an isolated tmp file so the user's real
    # ~/.../settings.json doesn't bleed into the test (any saved
    # source/target paths would auto-start a real model-loading
    # session — privacy leak + ~7s setup cost + this very test would
    # fail because `executor is not None` after the auto-start).
    settings_path = tmp_path / "settings.json"
    # Pre-write a Settings file that pins the batch store + global
    # output dir into tmp_path so the test doesn't pick up real
    # batch state from <install>/batch/ (which would accumulate
    # across test runs).
    import json as _json

    settings_path.write_text(
        _json.dumps(
            {
                "batch_store_path": str(tmp_path / "batch_store"),
                "batch_global_output_path": str(tmp_path / "batch_out"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(settings_path))
    w = SinnerMainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


class TestSinnerMainWindow:
    def test_constructs_with_expected_title(self, window):
        assert window.windowTitle() == "sinner2"

    def test_has_central_widget(self, window):
        assert window.centralWidget() is not None

    def test_status_bar_has_ready_message(self, window):
        assert "ready" in window._status_bar.current_message()  # noqa: SLF001

    def test_no_executor_until_source_and_target_set(self, window):
        assert window._controller.executor() is None  # noqa: SLF001


class TestModelDownloadDialog:
    def test_event_shows_then_hides_busy_dialog(self, window):
        assert window._model_load_dialog is None  # noqa: SLF001
        window._on_model_load_event("Downloading face-analysis models…")  # noqa: SLF001
        dlg = window._model_load_dialog  # noqa: SLF001
        assert dlg is not None
        assert dlg.maximum() == 0 and dlg.minimum() == 0  # indeterminate busy
        assert "Downloading" in dlg.labelText()
        window._on_model_load_event("")  # noqa: SLF001 — download done
        assert window._model_load_dialog is None  # noqa: SLF001

    def test_notifier_installed_on_face_analyser(self, window):
        from sinner2.pipeline import face_analyser

        # The window wired its relay's emit as the global notifier at startup.
        assert face_analyser._load_notifier is not None  # noqa: SLF001


class TestModelsTab:
    def test_side_panel_has_models_tab(self, window):
        panel = window._side_panel  # noqa: SLF001
        titles = [panel.tabText(i) for i in range(panel.count())]
        assert "Models" in titles

    def test_models_view_wired(self, window):
        assert window._side_panel.models_view() is window._models_view  # noqa: SLF001
        # The catalog is fully listed.
        from sinner2.pipeline.models_catalog import MODEL_CATALOG

        assert window._models_view._model.rowCount() == len(MODEL_CATALOG)  # noqa: SLF001


class TestStaysOnTop:
    def test_f12_toggles_window_stays_on_top_flag(self, window):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        # Initial: not stays-on-top.
        assert not (
            window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        )
        evt = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_F12,
            Qt.KeyboardModifier.NoModifier,
        )
        window.keyPressEvent(evt)
        assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        # Toggle back.
        window.keyPressEvent(evt)
        assert not (
            window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        )

    def test_corner_button_sets_flag(self, window):
        from PySide6.QtCore import Qt

        window._status_bar.on_top_button.setChecked(True)  # noqa: SLF001
        assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        window._status_bar.on_top_button.setChecked(False)  # noqa: SLF001
        assert not (window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)

    def test_f12_keeps_corner_button_in_sync(self, window):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        assert window._status_bar.on_top_button.isChecked() is False  # noqa: SLF001
        evt = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_F12, Qt.KeyboardModifier.NoModifier
        )
        window.keyPressEvent(evt)
        assert window._status_bar.on_top_button.isChecked() is True  # noqa: SLF001
        window.keyPressEvent(evt)
        assert window._status_bar.on_top_button.isChecked() is False  # noqa: SLF001


class TestFullscreenRestore:
    """F11 exit must return to the PRE-fullscreen window state — a window that
    was maximized comes back maximized, not dropped to a restored size."""

    def _spy_show_methods(self, window, monkeypatch, *, maximized: bool):
        calls: list[str] = []
        monkeypatch.setattr(window, "isMaximized", lambda: maximized)
        monkeypatch.setattr(window, "showFullScreen", lambda: calls.append("fs"))
        monkeypatch.setattr(window, "showMaximized", lambda: calls.append("max"))
        monkeypatch.setattr(window, "showNormal", lambda: calls.append("normal"))
        return calls

    def test_exit_restores_maximized_when_entered_maximized(
        self, window, monkeypatch
    ):
        calls = self._spy_show_methods(window, monkeypatch, maximized=True)
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  # enter
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  # exit
        assert "max" in calls
        assert "normal" not in calls

    def test_exit_restores_normal_when_entered_windowed(
        self, window, monkeypatch
    ):
        calls = self._spy_show_methods(window, monkeypatch, maximized=False)
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  # enter
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  # exit
        assert "normal" in calls
        assert "max" not in calls


class TestAltEnterFullscreen:
    """Alt+Enter toggles fullscreen, same as F11 / the corner button."""

    @staticmethod
    def _alt_enter(window):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        window.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.AltModifier,
            )
        )

    def test_alt_enter_enters_and_exits_fullscreen(self, window, monkeypatch):
        # Don't actually flip real window state in the test env.
        monkeypatch.setattr(window, "isMaximized", lambda: False)
        monkeypatch.setattr(window, "showFullScreen", lambda: None)
        monkeypatch.setattr(window, "showNormal", lambda: None)
        monkeypatch.setattr(window, "showMaximized", lambda: None)

        assert window._is_fullscreen is False  # noqa: SLF001
        self._alt_enter(window)
        assert window._is_fullscreen is True  # noqa: SLF001
        self._alt_enter(window)
        assert window._is_fullscreen is False  # noqa: SLF001

    def test_alt_enter_keeps_fullscreen_button_in_sync(self, window, monkeypatch):
        monkeypatch.setattr(window, "showFullScreen", lambda: None)
        monkeypatch.setattr(window, "showNormal", lambda: None)
        monkeypatch.setattr(window, "showMaximized", lambda: None)
        monkeypatch.setattr(window, "isMaximized", lambda: False)

        self._alt_enter(window)
        assert window._status_bar.fullscreen_button.isChecked() is True  # noqa: SLF001


class TestFullscreenControlBar:
    """In fullscreen the transport row is moved into the auto-hiding bottom
    bar (so playback stays reachable) and handed back on exit."""

    @staticmethod
    def _stub_show(window, monkeypatch):
        monkeypatch.setattr(window, "isMaximized", lambda: False)
        monkeypatch.setattr(window, "showFullScreen", lambda: None)
        monkeypatch.setattr(window, "showNormal", lambda: None)
        monkeypatch.setattr(window, "showMaximized", lambda: None)

    def test_enter_moves_transport_into_bar(self, window, monkeypatch):
        self._stub_show(window, monkeypatch)
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  enter
        assert (
            window._transport.parentWidget() is window._fs_controls  # noqa: SLF001
        )

    def test_exit_returns_transport_to_main_layout(self, window, monkeypatch):
        self._stub_show(window, monkeypatch)
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  enter
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  exit
        # Back under the central widget, and present in the central layout.
        assert (
            window._transport.parentWidget() is window.centralWidget()  # noqa: SLF001
        )
        layout = window._central_layout  # noqa: SLF001
        widgets = {layout.itemAt(i).widget() for i in range(layout.count())}
        assert window._transport in widgets  # noqa: SLF001
        assert window._transport.isVisible() or not window.isVisible()  # noqa: SLF001

    def test_bar_starts_hidden_on_enter(self, window, monkeypatch):
        self._stub_show(window, monkeypatch)
        window._status_bar.fullscreen_button.toggle()  # noqa: SLF001  enter
        assert window._fs_controls.is_revealed() is False  # noqa: SLF001


class TestStatusActionButtons:
    """The bottom-bar buttons drive the same actions as the shortcuts and stay
    in sync with them."""

    def test_stats_button_toggles_overlay(self, window):
        initial = window._metrics_overlay.isHidden()  # noqa: SLF001
        window._status_bar.stats_button.toggle()  # noqa: SLF001
        assert window._metrics_overlay.isHidden() is not initial  # noqa: SLF001

    def test_f4_keeps_stats_button_in_sync(self, window):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        evt = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_F4, Qt.KeyboardModifier.NoModifier
        )
        window.keyPressEvent(evt)
        assert window._status_bar.stats_button.isChecked() is True  # noqa: SLF001

    def test_overlay_checkbox_toggles_overlay(self, window):
        initial_hidden = window._face_overlay.isHidden()  # noqa: SLF001
        initial_on = window._face_overlay_on  # noqa: SLF001
        window._processors._overlay_enabled.toggle()  # noqa: SLF001
        assert window._face_overlay.isHidden() is not initial_hidden  # noqa: SLF001
        assert window._face_overlay_on is not initial_on  # noqa: SLF001

    def test_declining_upscaler_download_reverts_toggle(self, window, monkeypatch):
        # Enabling the upscaler when its model is missing must NOT silently
        # download — declining reverts the toggle.
        monkeypatch.setattr(window, "_ensure_upscaler_model", lambda: False)
        window._processors._upscaler_box.setChecked(True)  # noqa: SLF001
        assert window._processors.upscaler_enabled() is False  # reverted

    def test_accepting_upscaler_download_keeps_toggle(self, window, monkeypatch):
        monkeypatch.setattr(window, "_ensure_upscaler_model", lambda: True)
        window._processors._upscaler_box.setChecked(True)  # noqa: SLF001
        assert window._processors.upscaler_enabled() is True

    def test_f8_keeps_overlay_checkbox_in_sync(self, window):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        evt = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_F8, Qt.KeyboardModifier.NoModifier
        )
        window.keyPressEvent(evt)
        assert window._processors.face_overlay_enabled() is True  # noqa: SLF001
        assert window._face_overlay_on is True  # noqa: SLF001

    def test_enabling_overlay_probes_current_frame_when_swapper_off(
        self, window, monkeypatch
    ):
        # Swapper off → enabling while paused must probe the last shown frame
        # at once, not wait for a new one.
        import numpy as np

        monkeypatch.setattr(window._processors, "swapper_enabled", lambda: False)
        window._last_displayed_frame = np.zeros(  # noqa: SLF001
            (10, 20, 3), dtype=np.uint8
        )
        seen: list = []
        window._requestDetection.connect(  # noqa: SLF001
            lambda _f, w, h: seen.append((w, h))
        )
        window._processors._overlay_enabled.toggle()  # noqa: SLF001  # enable
        assert seen == [(20, 10)]  # (w, h) of the stored frame

    def test_enabling_overlay_uses_swapper_detections_when_swapper_on(
        self, window, monkeypatch
    ):
        from sinner2.gui.widgets.face_detection_overlay import FaceDetection

        monkeypatch.setattr(window._processors, "swapper_enabled", lambda: True)
        det = FaceDetection(bbox=(1.0, 2.0, 3.0, 4.0))
        monkeypatch.setattr(
            window._detection_sink,  # noqa: SLF001
            "latest_detections",
            lambda: ([det], 20, 10),
        )
        window._processors._overlay_enabled.toggle()  # noqa: SLF001  # enable
        assert window._face_overlay._detections == [det]  # noqa: SLF001
        assert window._face_overlay._frame_size == (20, 10)  # noqa: SLF001

    def test_disabling_overlay_keeps_sink_for_immediate_reenable(self, window):
        # The swapper publishes continuously; turning the overlay off must NOT
        # wipe the sink, so re-enabling while paused shows boxes at once.
        from types import SimpleNamespace

        import numpy as np

        window._detection_sink.publish(  # noqa: SLF001
            [SimpleNamespace(bbox=np.array([0.0, 0.0, 10.0, 10.0]))], 20, 10
        )
        window._set_face_overlay_visible(True)  # noqa: SLF001
        window._set_face_overlay_visible(False)  # noqa: SLF001
        assert window._detection_sink.latest_detections() is not None  # noqa: SLF001

    def test_seek_clears_stale_overlay_when_active(self, window, monkeypatch):
        # A seek is a discontinuity: with the overlay UP, drop the boxes + the
        # sink so a box from the old position can't linger "stuck" on the new
        # frame (it repopulates from the next detection).
        from types import SimpleNamespace

        import numpy as np

        from sinner2.gui.widgets.face_detection_overlay import FaceDetection

        seeks = []
        monkeypatch.setattr(window._session, "seek_to", seeks.append)  # noqa: SLF001
        window._face_overlay_on = True  # noqa: SLF001 — overlay active
        window._detection_sink.publish(  # noqa: SLF001
            [SimpleNamespace(bbox=np.array([0.0, 0.0, 10.0, 10.0]))], 20, 10
        )
        window._face_overlay.set_detections(  # noqa: SLF001
            [FaceDetection(bbox=(0.0, 0.0, 10.0, 10.0))], 20, 10
        )
        window._on_seek_requested(7)  # noqa: SLF001
        assert window._detection_sink.latest_detections() is None  # noqa: SLF001
        assert window._face_overlay._detections == []  # noqa: SLF001
        assert seeks == [7]  # still seeks the active session

    def test_seek_keeps_sink_when_overlay_inactive(self, window, monkeypatch):
        # Overlay DOWN: leave the sink alone (the swapper keeps it warm for an
        # immediate re-enable) — only seek.
        from types import SimpleNamespace

        import numpy as np

        seeks = []
        monkeypatch.setattr(window._session, "seek_to", seeks.append)  # noqa: SLF001
        window._face_overlay_on = False  # noqa: SLF001
        window._faces_mode = False  # noqa: SLF001 — editor closed → overlay down
        window._detection_sink.publish(  # noqa: SLF001
            [SimpleNamespace(bbox=np.array([0.0, 0.0, 10.0, 10.0]))], 20, 10
        )
        window._on_seek_requested(3)  # noqa: SLF001
        assert window._detection_sink.latest_detections() is not None  # noqa: SLF001
        assert seeks == [3]

    def test_face_map_overlay_gated_by_toggle(self, window, monkeypatch):
        # The "Use face map" toggle is the gate: only with it ON (+ editor open)
        # is the overlay the FACE-MAP overlay (pick + highlight) and the F8
        # diagnostic suppressed. With it OFF, F8 owns the surface (no pick).
        monkeypatch.setattr(
            window._face_map_ctl, "set_mode_active", lambda on: None  # noqa: SLF001
        )
        monkeypatch.setattr(
            window._face_map_ctl, "set_use_for_playback", lambda on: None  # noqa: SLF001
        )
        window._face_overlay_on = True  # noqa: SLF001 — F8 on
        window._on_faces_mode_toggled(True)  # noqa: SLF001 — editor open, toggle still OFF
        assert window._face_map_overlay_on() is False  # noqa: SLF001 — OFF = no face-map overlay
        assert window._diagnostic_overlay_on() is True  # noqa: SLF001 — F8 diagnostic
        assert window._face_overlay._pick_enabled is False  # noqa: SLF001 — no pick when OFF
        window._set_use_face_map(True)  # noqa: SLF001 — flip the gate ON
        # isHidden (not isVisible) — headless, ancestors aren't shown.
        assert window._face_overlay.isHidden() is False  # noqa: SLF001
        assert window._face_map_overlay_on() is True  # noqa: SLF001
        assert window._diagnostic_overlay_on() is False  # noqa: SLF001 — F8 yields
        assert window._face_overlay._pick_enabled is True  # noqa: SLF001 — face-map pick

    def test_analysis_forces_overlay_down(self, window, monkeypatch):
        # A scan owns the display: the overlay is fully down regardless of mode,
        # and comes back when it finishes.
        monkeypatch.setattr(
            window._face_map_ctl, "set_mode_active", lambda on: None  # noqa: SLF001
        )
        monkeypatch.setattr(
            window._face_map_ctl, "set_use_for_playback", lambda on: None  # noqa: SLF001
        )
        window._on_faces_mode_toggled(True)  # noqa: SLF001 — editor open
        window._set_use_face_map(True)  # noqa: SLF001 — face-map mode → overlay up
        assert window._face_overlay.isHidden() is False  # noqa: SLF001
        window._on_face_analysis_active(True)  # noqa: SLF001 — scan starts
        assert window._face_overlay.isHidden() is True  # noqa: SLF001
        assert window._overlay_timer.isActive() is False  # noqa: SLF001
        window._on_face_analysis_active(False)  # noqa: SLF001 — scan done
        assert window._face_overlay.isHidden() is False  # noqa: SLF001 — mode still on
        assert window._overlay_timer.isActive() is True  # noqa: SLF001

    def test_comparison_checkbox_drives_wants_crops(self, window):
        window._processors._overlay_enabled.toggle()  # noqa: SLF001  # overlay on
        window._processors._comparison_enabled.toggle()  # noqa: SLF001  # comparison on
        assert window._detection_sink.wants_crops() is True  # noqa: SLF001
        assert window._face_overlay._comparison_on is True  # noqa: SLF001
        window._processors._comparison_enabled.toggle()  # noqa: SLF001  # off
        assert window._detection_sink.wants_crops() is False  # noqa: SLF001

    def test_enabling_comparison_turns_on_the_overlay(self, window):
        # The two are linked: comparison draws ON the overlay, so checking it
        # with the overlay OFF must enable the overlay (else the toggle looks
        # broken). Crops are then wanted.
        assert window._processors.face_overlay_enabled() is False  # noqa: SLF001
        window._processors._comparison_enabled.toggle()  # noqa: SLF001  # comparison on
        assert window._processors.face_overlay_enabled() is True  # noqa: SLF001
        assert window._comparison_on is True  # noqa: SLF001
        assert window._detection_sink.wants_crops() is True  # noqa: SLF001

    def test_disabling_overlay_turns_off_comparison(self, window):
        # Turning the overlay off must drop comparison too — nothing to draw on.
        window._processors._comparison_enabled.toggle()  # noqa: SLF001  # both on (linked)
        assert window._processors.face_comparison_enabled() is True  # noqa: SLF001
        window._processors._overlay_enabled.toggle()  # noqa: SLF001  # overlay off
        assert window._processors.face_comparison_enabled() is False  # noqa: SLF001
        assert window._comparison_on is False  # noqa: SLF001
        assert window._detection_sink.wants_crops() is False  # noqa: SLF001

    def test_library_zoom_is_independent_per_panel(self, window):
        src = window._side_panel.sources_library()  # noqa: SLF001
        tgt = window._side_panel.targets_library()  # noqa: SLF001
        tgt_before = tgt.display_dim()
        src.set_display_dim(src.display_dim() + 32)  # change source only
        assert tgt.display_dim() == tgt_before  # target NOT synced
        assert window._settings.library_sources_display_dim == src.display_dim()  # noqa: SLF001

    def test_library_sort_persists_per_panel(self, window):
        tgt = window._side_panel.targets_library()  # noqa: SLF001
        tgt.set_sort("date", "asc")  # silent restore
        tgt._toggle_sort_direction()  # noqa: SLF001  # asc → desc, fires sortChanged
        assert window._settings.library_targets_sort_field == "date"  # noqa: SLF001
        assert window._settings.library_targets_sort_order == "desc"  # noqa: SLF001

    def test_hint_reflects_swapper_state(self, window, monkeypatch):
        monkeypatch.setattr(window._processors, "swapper_enabled", lambda: True)
        window._set_face_overlay_visible(True)  # noqa: SLF001
        assert "swapper" in window._status_bar.current_message().lower()  # noqa: SLF001

        monkeypatch.setattr(window._processors, "swapper_enabled", lambda: False)
        window._set_face_overlay_visible(True)  # noqa: SLF001
        assert "swapper" not in window._status_bar.current_message().lower()  # noqa: SLF001

    def test_side_panel_button_hides_panel(self, window):
        assert window._side_panel.isHidden() is False  # noqa: SLF001  # default shown
        window._status_bar.side_panel_button.toggle()  # noqa: SLF001
        assert window._side_panel.isHidden() is True  # noqa: SLF001

    def test_rotate_button_cycles_rotation(self, window):
        assert window._display.rotation() == 0  # noqa: SLF001
        window._status_bar.rotate_button.click()  # noqa: SLF001
        assert window._display.rotation() == 90  # noqa: SLF001


class TestRerenderFromCurrent:
    def test_request_delegates_to_controller(self, window, monkeypatch):
        called: list = []
        monkeypatch.setattr(
            window._controller,  # noqa: SLF001
            "rerender_from_current",
            lambda: called.append(1),
        )
        window._processors.rerenderRequested.emit()  # noqa: SLF001
        assert called == [1]


class TestSeekAndQueueShortcuts:
    @staticmethod
    def _press(window, key, ctrl=False):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        mod = (
            Qt.KeyboardModifier.ControlModifier
            if ctrl
            else Qt.KeyboardModifier.NoModifier
        )
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, mod))

    def _stub_executor(self, window, monkeypatch):
        from unittest.mock import MagicMock

        ex = MagicMock()
        ex.frame_count.return_value = 100
        ex.current_frame.get.return_value = 50
        monkeypatch.setattr(
            window._controller, "executor", lambda: ex  # noqa: SLF001
        )
        # The seek shortcuts now route through controller.seek_to() -> _on_seek,
        # which uses the controller's _executor attribute (same object as the
        # accessor in the real app), so the mock must back both.
        monkeypatch.setattr(window._controller, "_executor", ex)  # noqa: SLF001
        return ex

    def test_home_seeks_to_start(self, window, monkeypatch):
        from PySide6.QtCore import Qt

        ex = self._stub_executor(window, monkeypatch)
        self._press(window, Qt.Key.Key_Home)
        ex.seek.assert_called_once_with(0)

    def test_end_seeks_to_last_frame(self, window, monkeypatch):
        from PySide6.QtCore import Qt

        ex = self._stub_executor(window, monkeypatch)
        self._press(window, Qt.Key.Key_End)
        ex.seek.assert_called_once_with(99)

    def test_ctrl_enter_sends_to_batch(self, window, tmp_path, monkeypatch):
        from PySide6.QtCore import Qt

        monkeypatch.setattr(
            window._controller,  # noqa: SLF001
            "set_source_and_target",
            lambda *a, **k: None,
        )
        src = tmp_path / "s.png"
        src.write_bytes(b"x")
        tgt = tmp_path / "t.mp4"
        tgt.write_bytes(b"x")
        window._pickers.set_source(src)  # noqa: SLF001
        window._pickers.set_target(tgt)  # noqa: SLF001
        before = len(window._batch_store.list())  # noqa: SLF001
        self._press(window, Qt.Key.Key_Return, ctrl=True)
        assert len(window._batch_store.list()) == before + 1  # noqa: SLF001


class TestSectionShortcuts:
    @staticmethod
    def _press(window, key):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        window.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        )

    def _stub_executor(self, window, monkeypatch, *, frame):
        from unittest.mock import MagicMock

        ex = MagicMock()
        ex.frame_count.return_value = 300
        ex.current_frame.get.return_value = frame
        monkeypatch.setattr(window._controller, "executor", lambda: ex)  # noqa: SLF001
        monkeypatch.setattr(window._controller, "_executor", ex)  # noqa: SLF001
        return ex

    def test_bracket_keys_create_section(self, window, monkeypatch):
        from PySide6.QtCore import Qt
        from sinner2.pipeline.sections import SectionSet

        ex = self._stub_executor(window, monkeypatch, frame=50)
        self._press(window, Qt.Key.Key_BracketLeft)   # in at 50
        ex.current_frame.get.return_value = 120
        self._press(window, Qt.Key.Key_BracketRight)  # out at 120 → commit
        assert window._transport.sections() == SectionSet.of([(50, 120)])  # noqa: SLF001
        # Pushed to the executor for live trimming.
        ex.set_sections.assert_called_with(SectionSet.of([(50, 120)]))  # noqa: SLF001

    def test_delete_key_removes_selected_section(self, window, monkeypatch):
        from PySide6.QtCore import Qt

        ex = self._stub_executor(window, monkeypatch, frame=50)
        self._press(window, Qt.Key.Key_BracketLeft)
        ex.current_frame.get.return_value = 120
        self._press(window, Qt.Key.Key_BracketRight)
        # Select the band, then delete it.
        window._transport._update_selection_to(60)  # noqa: SLF001
        self._press(window, Qt.Key.Key_Delete)
        assert window._transport.sections().is_empty()  # noqa: SLF001

    def test_restore_sections_for_unknown_target_clears(self, window, monkeypatch):
        from sinner2.pipeline.sections import SectionSet

        ex = self._stub_executor(window, monkeypatch, frame=0)
        window._transport.set_sections(SectionSet.of([(10, 20)]))  # noqa: SLF001
        window._restore_sections_for_target(Path("never_seen.mp4"))  # noqa: SLF001
        assert window._transport.sections().is_empty()  # noqa: SLF001
        ex.set_sections.assert_called_with(SectionSet.empty())  # noqa: SLF001

    def test_sections_persist_and_restore_per_target(self, window, monkeypatch):
        from sinner2.pipeline.sections import SectionSet

        self._stub_executor(window, monkeypatch, frame=0)
        tgt = Path("clip.mp4")
        monkeypatch.setattr(window._pickers, "target_path", lambda: tgt)  # noqa: SLF001
        # An edit persists under the target...
        window._on_sections_changed(SectionSet.of([(30, 90)]))  # noqa: SLF001
        assert window._settings.sections_by_target == {  # noqa: SLF001
            "clip.mp4": [[30, 90]]
        }
        # ...and reloading that target restores it.
        window._transport.set_sections(SectionSet.empty())  # noqa: SLF001
        window._restore_sections_for_target(tgt)  # noqa: SLF001
        assert window._transport.sections() == SectionSet.of([(30, 90)])  # noqa: SLF001

    def test_clearing_sections_forgets_target_entry(self, window, monkeypatch):
        from sinner2.pipeline.sections import SectionSet

        self._stub_executor(window, monkeypatch, frame=0)
        tgt = Path("clip.mp4")
        monkeypatch.setattr(window._pickers, "target_path", lambda: tgt)  # noqa: SLF001
        window._on_sections_changed(SectionSet.of([(30, 90)]))  # noqa: SLF001
        window._on_sections_changed(SectionSet.empty())  # noqa: SLF001 — cleared
        assert window._settings.sections_by_target is None  # noqa: SLF001

    def test_ctrl_arrow_steps_100_frames(self, window, monkeypatch):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        ex = self._stub_executor(window, monkeypatch, frame=500)
        ex.frame_count.return_value = 2000  # room to step forward
        seeks = []
        monkeypatch.setattr(window._session, "seek_to", seeks.append)  # noqa: SLF001
        window.keyPressEvent(QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Right,
            Qt.KeyboardModifier.ControlModifier,
        ))
        assert seeks == [600]  # 500 + 100

    def test_shift_arrow_steps_10_frames(self, window, monkeypatch):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        ex = self._stub_executor(window, monkeypatch, frame=500)
        ex.frame_count.return_value = 2000
        seeks = []
        monkeypatch.setattr(window._session, "seek_to", seeks.append)  # noqa: SLF001
        window.keyPressEvent(QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Left,
            Qt.KeyboardModifier.ShiftModifier,
        ))
        assert seeks == [490]  # 500 - 10

    def test_add_to_batch_carries_sections(self, window, tmp_path, monkeypatch):
        from sinner2.pipeline.sections import SectionSet

        monkeypatch.setattr(
            window._controller, "set_source_and_target", lambda *a, **k: None  # noqa: SLF001
        )
        src = tmp_path / "s.png"
        src.write_bytes(b"x")
        tgt = tmp_path / "t.mp4"
        tgt.write_bytes(b"x")
        window._pickers.set_source(src)  # noqa: SLF001
        window._pickers.set_target(tgt)  # noqa: SLF001
        window._transport.set_sections(SectionSet.of([(30, 90), (150, 200)]))  # noqa: SLF001
        window._on_add_to_batch()  # noqa: SLF001
        task = window._batch_store.list()[0]  # noqa: SLF001
        assert task.sections == [[30, 90], [150, 200]]

    def test_add_to_batch_no_sections_leaves_none(self, window, tmp_path, monkeypatch):
        monkeypatch.setattr(
            window._controller, "set_source_and_target", lambda *a, **k: None  # noqa: SLF001
        )
        src = tmp_path / "s.png"
        src.write_bytes(b"x")
        tgt = tmp_path / "t.mp4"
        tgt.write_bytes(b"x")
        window._pickers.set_source(src)  # noqa: SLF001
        window._pickers.set_target(tgt)  # noqa: SLF001
        window._on_add_to_batch()  # noqa: SLF001
        assert window._batch_store.list()[0].sections is None  # noqa: SLF001


class TestFaceMappingWiring:
    def test_faces_toggle_opens_editor(self, window):
        # The Sources-tab "Face map" toggle opens the EDITOR. It does NOT lock the
        # source picker or enable picking on its own — pick/highlight need the
        # "Use face map" gate ON (the editor alone is just single-source + panel).
        window._on_faces_mode_toggled(True)  # noqa: SLF001
        assert window._faces_mode is True  # noqa: SLF001
        assert window._face_overlay._pick_enabled is False  # noqa: SLF001 — gate still off
        assert window._pickers._source.isEnabled() is True  # noqa: SLF001 — not locked
        window._on_faces_mode_toggled(False)  # noqa: SLF001
        assert window._faces_mode is False  # noqa: SLF001

    def test_use_face_map_switch_drives_routing(self, window, monkeypatch):
        # D6: the Face-detector "Use face map" switch routes playback with the
        # editor closed, locks the source, and persists per target.
        routing, persisted = [], []
        monkeypatch.setattr(
            window._face_map_ctl, "set_mode_active",  # noqa: SLF001
            lambda on: routing.append(on),
        )
        monkeypatch.setattr(
            window._face_map_ctl, "set_use_for_playback",  # noqa: SLF001
            lambda on: persisted.append(on),
        )
        window._on_use_face_map_toggled(True)  # noqa: SLF001
        assert window._use_face_map is True and routing[-1] is True  # noqa: SLF001
        assert persisted == [True]
        assert window._pickers._source.isEnabled() is False  # noqa: SLF001 — locked
        window._on_use_face_map_toggled(False)  # noqa: SLF001
        assert routing[-1] is False
        assert window._pickers._source.isEnabled() is True  # noqa: SLF001

    def test_opening_editor_does_not_change_routing(self, window, monkeypatch):
        # Item 1: opening the editor must NOT flip routing on — the switch owns
        # routing and only unlocks once a map is built.
        routing = []
        monkeypatch.setattr(
            window._face_map_ctl, "set_mode_active",  # noqa: SLF001
            lambda on: routing.append(on),
        )
        window._on_faces_mode_toggled(True)  # noqa: SLF001 — open editor, no map
        assert window._use_face_map is False  # noqa: SLF001 — switch untouched
        assert window._processors.use_face_map() is False  # noqa: SLF001

    def test_switch_unlocks_only_when_a_map_is_built(self, window):
        # Item 1: enabled ⟺ a catalog exists — never just because the editor is
        # open. The processor widget reflects it.
        window._on_faces_mode_toggled(True)  # noqa: SLF001 — editor open, no map
        assert window._processors._use_face_map.isEnabled() is False  # noqa: SLF001
        window._on_map_availability_changed(True)  # noqa: SLF001 — map built
        assert window._processors._use_face_map.isEnabled() is True  # noqa: SLF001

    def test_map_unavailable_disables_and_clears_routing(self, window, monkeypatch):
        routing = []
        monkeypatch.setattr(
            window._face_map_ctl, "set_mode_active",  # noqa: SLF001
            lambda on: routing.append(on),
        )
        monkeypatch.setattr(
            window._face_map_ctl, "set_use_for_playback", lambda on: None  # noqa: SLF001
        )
        window._on_map_availability_changed(True)  # noqa: SLF001
        window._set_use_face_map(True)  # noqa: SLF001 — routing on
        window._on_map_availability_changed(False)  # noqa: SLF001 — map gone (reset)
        assert window._use_face_map is False and routing[-1] is False  # noqa: SLF001
        assert window._processors._use_face_map.isEnabled() is False  # noqa: SLF001

    def test_fresh_analysis_turns_routing_on(self, window, monkeypatch):
        # Issue 1: after a scan builds a catalog the "Use face map" switch flips
        # ON by itself — the user shouldn't have to toggle it every analysis.
        routing = []
        monkeypatch.setattr(
            window._face_map_ctl, "set_mode_active",  # noqa: SLF001
            lambda on: routing.append(on),
        )
        monkeypatch.setattr(
            window._face_map_ctl, "set_use_for_playback", lambda on: None  # noqa: SLF001
        )
        window._on_analysis_produced_map(True)  # noqa: SLF001
        assert window._use_face_map is True and routing[-1] is True  # noqa: SLF001
        # An empty scan leaves routing untouched (nothing to route).
        window._set_use_face_map(False)  # noqa: SLF001
        routing.clear()
        window._on_analysis_produced_map(False)  # noqa: SLF001
        assert window._use_face_map is False  # noqa: SLF001

    def test_restore_preference_reflects_and_routes(self, window, monkeypatch):
        routing = []
        monkeypatch.setattr(
            window._face_map_ctl, "set_mode_active",  # noqa: SLF001
            lambda on: routing.append(on),
        )
        window._on_use_for_playback_restored(True)  # noqa: SLF001
        assert window._use_face_map is True  # noqa: SLF001
        assert window._processors.use_face_map() is True  # noqa: SLF001
        assert routing[-1] is True

    def test_analysis_active_pauses_and_locks(self, window, monkeypatch):
        paused = []
        locks = []
        monkeypatch.setattr(window._session, "pause", lambda: paused.append(1))  # noqa: SLF001
        monkeypatch.setattr(
            window, "_set_editing_locked", lambda on: locks.append(on)  # noqa: SLF001
        )
        window._on_face_analysis_active(True)  # noqa: SLF001
        assert paused == [1]
        assert locks == [True]
        window._on_face_analysis_active(False)  # noqa: SLF001
        assert locks == [True, False]  # no batch running → unlocks

    def test_library_click_off_mode_sets_global_source(self, window, monkeypatch):
        window._faces_mode = False  # noqa: SLF001
        src_calls = []
        monkeypatch.setattr(
            window._pickers, "set_source", lambda p: src_calls.append(p)  # noqa: SLF001
        )
        window._on_library_source_selected(Path("/s.png"))  # noqa: SLF001
        assert src_calls == [Path("/s.png")]

    def test_library_click_in_mode_assigns_to_selection(self, window):
        from sinner2.pipeline.face_map import FaceMap, Identity, normalize

        fm = FaceMap(
            identities=(
                Identity("a", normalize([1, 0, 0])),
                Identity("b", normalize([0, 1, 0])),
            )
        )
        window._face_map_ctl._catalog = fm  # noqa: SLF001 — controller is the authority
        window._face_map_ctl._mode_active = True  # noqa: SLF001 — routing pushed live
        window._controller.set_face_map(fm)  # noqa: SLF001
        window._face_map_panel.set_face_map(fm)  # noqa: SLF001
        window._face_map_panel._table.selectAll()  # noqa: SLF001 — both rows
        window._faces_mode = True  # noqa: SLF001
        window._on_library_source_selected(Path("/src/alice.png"))  # noqa: SLF001
        by_id = {
            i.id: i.source_path
            for i in window._controller.face_map().identities  # noqa: SLF001
        }
        assert by_id["a"] == str(Path("/src/alice.png"))
        assert by_id["b"] == str(Path("/src/alice.png"))

    def test_library_click_in_mode_without_selection_nudges(self, window, monkeypatch):
        from sinner2.pipeline.face_map import FaceMap

        calls = []
        monkeypatch.setattr(
            window._face_map_ctl, "assign_source",  # noqa: SLF001
            lambda ids, p: calls.append((ids, p)),
        )
        window._face_map_panel.set_face_map(FaceMap.empty())  # noqa: SLF001
        window._faces_mode = True  # noqa: SLF001
        window._on_library_source_selected(Path("/src/x.png"))  # noqa: SLF001
        assert calls == []  # nothing selected → no assignment
        assert "Select one or more faces" in window._status_bar.current_message()  # noqa: SLF001

    def test_library_click_locked_during_batch(self, window, monkeypatch):
        calls = []
        src = []
        monkeypatch.setattr(
            window._face_map_ctl, "assign_source",  # noqa: SLF001
            lambda ids, p: calls.append((ids, p)),
        )
        monkeypatch.setattr(
            window._pickers, "set_source", lambda p: src.append(p)  # noqa: SLF001
        )
        window._faces_mode = True  # noqa: SLF001
        window._batch_active = True  # noqa: SLF001
        window._on_library_source_selected(Path("/src/x.png"))  # noqa: SLF001
        assert calls == [] and src == []  # editing locked → nothing happens

    def test_selecting_face_highlights_only_its_box(self, window, monkeypatch):
        # Highlight needs the face-map overlay active = toggle ON + editor open.
        window._faces_mode = True  # noqa: SLF001
        window._use_face_map = True  # noqa: SLF001 — gate on
        monkeypatch.setattr(
            window._face_map_ctl, "selected_face_bbox",  # noqa: SLF001
            lambda: (1.0, 2.0, 3.0, 4.0),
        )
        calls = []
        monkeypatch.setattr(
            window._face_overlay, "set_highlight", lambda b: calls.append(b)  # noqa: SLF001
        )
        window._refresh_face_highlight()  # noqa: SLF001
        assert calls == [(1.0, 2.0, 3.0, 4.0)]
        window._use_face_map = False  # noqa: SLF001 — gate off → highlight cleared
        window._refresh_face_highlight()  # noqa: SLF001
        assert calls[-1] is None

    def test_selection_kicks_probe_when_swapper_off(self, window, monkeypatch):
        # Point 2: selecting a face must highlight it reliably even with the
        # swapper OFF — so the selection kicks a fresh detection of the current
        # frame (the highlight reads the sink the probe fills). Swapper ON uses
        # the swapper's published detections, so no probe is needed.
        import numpy as np

        monkeypatch.setattr(window._processors, "swapper_enabled", lambda: False)
        monkeypatch.setattr(
            window._face_map_ctl, "selected_face_bbox", lambda: None  # noqa: SLF001
        )
        window._faces_mode = True  # noqa: SLF001
        window._use_face_map = True  # noqa: SLF001 — face-map overlay active
        window._face_analyzing = False  # noqa: SLF001
        window._last_displayed_frame = np.zeros((10, 20, 3), np.uint8)  # noqa: SLF001
        probed = []
        window._requestDetection.connect(  # noqa: SLF001
            lambda _f, w, h: probed.append((w, h))
        )
        window._on_face_selection_changed()  # noqa: SLF001
        assert probed == [(20, 10)]  # fresh detection kicked
        # Swapper ON → rely on the published sink, no probe.
        monkeypatch.setattr(window._processors, "swapper_enabled", lambda: True)
        probed.clear()
        window._on_face_selection_changed()  # noqa: SLF001
        assert probed == []

    def test_live_running_disables_faces_toggle(self, window):
        window._on_live_running(True)  # noqa: SLF001
        assert window._side_panel.faces_mode() is False  # noqa: SLF001
        assert window._side_panel._faces_toggle.isEnabled() is False  # noqa: SLF001
        window._on_live_running(False)  # noqa: SLF001
        assert window._side_panel._faces_toggle.isEnabled() is True  # noqa: SLF001

    def test_reset_confirmed_resets(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(
            window._face_map_ctl, "reset_catalog", lambda: calls.append(1)  # noqa: SLF001
        )
        monkeypatch.setattr("sinner2.gui.main_window.confirm", lambda *a, **k: True)
        window._on_face_map_reset()  # noqa: SLF001
        assert calls == [1]

    def test_reset_declined_does_nothing(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(
            window._face_map_ctl, "reset_catalog", lambda: calls.append(1)  # noqa: SLF001
        )
        monkeypatch.setattr("sinner2.gui.main_window.confirm", lambda *a, **k: False)
        window._on_face_map_reset()  # noqa: SLF001
        assert calls == []

    def test_reset_confirm_is_suppressible(self, window, monkeypatch):
        # D3: the reset prompt now offers "Don't ask me again".
        seen = {}
        monkeypatch.setattr(
            "sinner2.gui.main_window.confirm",
            lambda *a, **k: seen.update(k) or True,
        )
        window._on_face_map_reset()  # noqa: SLF001
        assert seen.get("suppressible") is True


class TestRotationShortcut:
    def test_r_key_cycles_rotation(self, window):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        assert window._display.rotation() == 0  # noqa: SLF001
        evt = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.NoModifier,
        )
        window.keyPressEvent(evt)
        assert window._display.rotation() == 90  # noqa: SLF001
        for expected in (180, 270, 0):
            window.keyPressEvent(evt)
            assert window._display.rotation() == expected  # noqa: SLF001


class TestTransportGating:
    """Transport controls gate on the active session's capabilities."""

    def test_disabled_without_session(self, window):
        assert not window._transport._play_button.isEnabled()  # noqa: SLF001
        assert not window._transport._slider.isEnabled()  # noqa: SLF001

    def test_file_session_enables_play_and_seek(self, window, monkeypatch):
        from unittest.mock import MagicMock

        ex = MagicMock()
        monkeypatch.setattr(window._controller, "executor", lambda: ex)  # noqa: SLF001
        monkeypatch.setattr(window._controller, "_executor", ex)  # noqa: SLF001
        window._refresh_transport_enabled()  # noqa: SLF001
        assert window._transport._play_button.isEnabled()  # noqa: SLF001
        assert window._transport._slider.isEnabled()  # noqa: SLF001


class TestBatchIntegration:
    def test_add_to_batch_with_no_paths_is_noop(self, window):
        # Pickers empty → addToBatchRequested handler short-circuits.
        # The button itself is disabled in this state, but the handler
        # should still be defensive (signals can theoretically arrive
        # via testing harnesses).
        window._on_add_to_batch()  # noqa: SLF001
        assert len(window._batch_store.list()) == 0  # noqa: SLF001

    def test_add_to_batch_persists_task_and_appends_row(
        self, window, qtbot, tmp_path, monkeypatch
    ):
        # Stub the controller's heavy session-build call (it loads
        # insightface / GFPGAN, takes 30+ seconds cold and slows the
        # test suite to a crawl). The batch capture path doesn't need
        # an actual session — just the picker state.
        monkeypatch.setattr(
            window._controller,  # noqa: SLF001
            "set_source_and_target",
            lambda *a, **k: None,
        )
        # Drop fake source + target files so the pickers accept them.
        src = tmp_path / "src.png"
        src.write_bytes(b"x")
        tgt = tmp_path / "tgt.mp4"
        tgt.write_bytes(b"x")
        window._pickers.set_source(src)  # noqa: SLF001
        window._pickers.set_target(tgt)  # noqa: SLF001
        window._on_add_to_batch()  # noqa: SLF001
        tasks = window._batch_store.list()  # noqa: SLF001
        assert len(tasks) == 1
        assert tasks[0].source_path == src
        assert tasks[0].target_path == tgt
        # Row also landed in the view.
        assert window._batch_view._model.rowCount() == 1  # noqa: SLF001

    def test_add_to_batch_uses_defaults_template_not_preview(
        self, window, tmp_path, monkeypatch
    ):
        # Batch is decoupled from the live preview: a new task carries the
        # Batch Defaults template's config + ONLY the picker source/target.
        monkeypatch.setattr(
            window._controller,  # noqa: SLF001
            "set_source_and_target",
            lambda *a, **k: None,
        )
        window._batch_defaults = window._batch_defaults.model_copy(  # noqa: SLF001
            update={
                "swapper_model": "uniface_256",
                "processing_scale": 0.5,
                "enhancer_enabled": False,
            }
        )
        src = tmp_path / "src.png"
        src.write_bytes(b"x")
        tgt = tmp_path / "tgt.mp4"
        tgt.write_bytes(b"x")
        window._pickers.set_source(src)  # noqa: SLF001
        window._pickers.set_target(tgt)  # noqa: SLF001
        window._on_add_to_batch()  # noqa: SLF001
        tasks = window._batch_store.list()  # noqa: SLF001
        assert len(tasks) == 1
        t = tasks[0]
        assert t.source_path == src and t.target_path == tgt
        # Config came from the defaults template, not the preview.
        assert t.swapper_model == "uniface_256"
        assert t.processing_scale == 0.5
        assert t.enhancer_enabled is False
        assert t.status.value == "pending"

    def test_batch_settings_persists_defaults_and_paths(
        self, window, monkeypatch
    ):
        from sinner2.batch import defaults as batch_defaults
        from sinner2.gui import main_window as mw

        class _FakeSettingsDialog:
            class DialogCode:
                Accepted = 1
                Rejected = 0

            def __init__(
                self, template, parent=None, *, defaults_mode,
                store_path, global_output_path,
            ):
                assert defaults_mode is True
                self._template = template

            def exec(self):
                return self.DialogCode.Accepted

            def to_task(self):
                return self._template.model_copy(
                    update={"swapper_model": "ghost_2_256"}
                )

            def store_path(self):
                return "/new/store"

            def global_output_path(self):
                return "/new/out"

        monkeypatch.setattr(mw, "QBatchTaskDialog", _FakeSettingsDialog)
        window._on_batch_settings()  # noqa: SLF001
        # Template updated in memory + persisted to disk.
        assert window._batch_defaults.swapper_model == "ghost_2_256"  # noqa: SLF001
        reloaded = batch_defaults.load_defaults(
            window._batch_defaults_path  # noqa: SLF001
        )
        assert reloaded.swapper_model == "ghost_2_256"
        # Queue-wide paths persisted into settings + applied to the queue.
        assert window._settings.batch_store_path == "/new/store"  # noqa: SLF001
        assert window._settings.batch_global_output_path == "/new/out"  # noqa: SLF001
        assert window._batch_queue._global_output_dir == Path("/new/out")  # noqa: SLF001

    def test_batch_settings_rejected_changes_nothing(self, window, monkeypatch):
        from sinner2.gui import main_window as mw

        before_model = window._batch_defaults.swapper_model  # noqa: SLF001

        class _RejectDialog:
            class DialogCode:
                Accepted = 1
                Rejected = 0

            def __init__(self, *a, **k):
                pass

            def exec(self):
                return self.DialogCode.Rejected

            def to_task(self):  # pragma: no cover - must not be called
                raise AssertionError("to_task on a rejected dialog")

        monkeypatch.setattr(mw, "QBatchTaskDialog", _RejectDialog)
        window._on_batch_settings()  # noqa: SLF001
        assert window._batch_defaults.swapper_model == before_model  # noqa: SLF001

    def test_batch_running_locks_editing_surface(self, window, monkeypatch):
        # DaVinci-style: a running batch locks transport + pickers + settings
        # + libraries, but the Batch tab stays interactive. A mock file session
        # makes the transport live before the lock + re-enabled after.
        from unittest.mock import MagicMock

        ex = MagicMock()
        ex.frame_count.return_value = 100      # resync_transport reads these on idle
        ex.current_frame.get.return_value = 0
        monkeypatch.setattr(window._controller, "executor", lambda: ex)  # noqa: SLF001
        monkeypatch.setattr(window._controller, "_executor", ex)  # noqa: SLF001
        window._refresh_transport_enabled()  # noqa: SLF001
        assert window._transport._play_button.isEnabled()  # noqa: SLF001

        window._batch_queue.taskStarted.emit("x")  # noqa: SLF001
        assert not window._transport._play_button.isEnabled()  # noqa: SLF001 locked
        assert not window._pickers.isEnabled()  # noqa: SLF001
        assert not window._processors.isEnabled()  # noqa: SLF001
        assert not window._side_panel.sources_library().isEnabled()  # noqa: SLF001
        assert not window._side_panel.targets_library().isEnabled()  # noqa: SLF001
        assert window._batch_view.isEnabled()  # noqa: SLF001  queue stays usable
        window._batch_queue.queueIdle.emit()  # noqa: SLF001
        assert window._transport._play_button.isEnabled()  # noqa: SLF001 re-enabled
        assert window._pickers.isEnabled()  # noqa: SLF001
        assert window._processors.isEnabled()  # noqa: SLF001

    def test_failure_surfaces_consolidated_error_at_idle(
        self, window, monkeypatch
    ):
        # A failure is collected during the run and surfaced in ONE error dialog
        # when the queue goes idle (no modal spam per task), and the editing
        # lock always releases on idle so the user can recover without restart.
        errors: list[str] = []
        monkeypatch.setattr(window, "_show_error", errors.append)  # noqa: SLF001
        window._batch_queue.taskStarted.emit("x")  # noqa: SLF001  fresh run
        window._batch_queue.taskFailed.emit("x", "ffmpeg not found")  # noqa: SLF001
        assert errors == []  # nothing modal mid-run
        window._batch_queue.queueIdle.emit()  # noqa: SLF001
        assert len(errors) == 1
        assert "ffmpeg not found" in errors[0]
        assert not window._batch_active  # noqa: SLF001  unlocked
        assert window._pickers.isEnabled()  # noqa: SLF001

    def test_no_failures_shows_no_error_dialog(self, window, monkeypatch):
        errors: list[str] = []
        monkeypatch.setattr(window, "_show_error", errors.append)  # noqa: SLF001
        window._batch_queue.taskStarted.emit("x")  # noqa: SLF001
        window._batch_queue.queueIdle.emit()  # noqa: SLF001
        assert errors == []  # clean run → no error popup

    def test_config_change_ignored_while_batch_active(
        self, window, monkeypatch
    ):
        # The live re-render path must be inert during a render — otherwise
        # toggling a param would clobber the batch preview.
        calls: list = []
        monkeypatch.setattr(
            window._controller,  # noqa: SLF001
            "apply_session_config",
            lambda **k: calls.append(k),
        )
        window._batch_active = True  # noqa: SLF001
        window._on_processor_config_changed()  # noqa: SLF001
        assert calls == []

    def test_batch_preview_shows_frame_on_display(self, window, qtbot):
        import numpy as np

        frame = np.full((12, 12, 3), 90, dtype=np.uint8)
        window._batch_queue.taskPreview.emit("x", frame)  # noqa: SLF001
        qtbot.waitUntil(
            lambda: window._display._pixmap is not None,  # noqa: SLF001
            timeout=1000,
        )


class TestSaveCurrentFrame:
    def test_save_no_op_when_no_frame_displayed(self, window, monkeypatch):
        # No frame loaded → status message says so + dialog never opens.
        prompted: list[object] = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **k: (prompted.append(True), ("", ""))[1],
        )
        window._save_current_frame()  # noqa: SLF001
        assert prompted == []
        assert "No frame" in window._status_bar.current_message()  # noqa: SLF001

    def test_save_writes_pixmap_to_disk(
        self, window, qtbot, tmp_path, monkeypatch
    ):
        import numpy as np

        from sinner2.types import Frame

        # Push a frame into the display so current_pixmap is non-None.
        frame: Frame = np.full((20, 30, 3), 200, dtype=np.uint8)
        window._display.show_frame(frame)  # noqa: SLF001
        qtbot.waitUntil(
            lambda: window._display._pixmap is not None,  # noqa: SLF001
            timeout=1000,
        )
        out = tmp_path / "snap.png"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), ""),
        )
        window._save_current_frame()  # noqa: SLF001
        assert out.is_file()
        assert out.stat().st_size > 0
        assert "Saved" in window._status_bar.current_message()  # noqa: SLF001


class TestTensorRTBuildWait:
    """_wait_for_tensorrt_build shows a modal 'compiling' dialog whenever a TRT
    engine build is about to run (TRT requested, a live session, and no session
    has actually recorded TRT yet) — keyed off the REAL recorded providers
    (get_actual_providers), so it also fires at launch (actual = None), not just
    on a toggle."""

    def _set(self, window, monkeypatch, *, requested, actual, has_executor, cached=False):
        from unittest.mock import MagicMock

        from sinner2.pipeline import model_cache
        monkeypatch.setattr(window._processors, "swapper_providers", lambda: requested)  # noqa: SLF001
        monkeypatch.setattr(model_cache, "get_actual_providers", lambda: actual)
        monkeypatch.setattr(model_cache, "tensorrt_engine_cached", lambda: cached)
        monkeypatch.setattr(  # noqa: SLF001
            window._controller,
            "executor",
            lambda: (MagicMock() if has_executor else None),
        )

    def test_no_wait_when_trt_not_requested(self, window, monkeypatch):
        self._set(window, monkeypatch, requested=["CUDAExecutionProvider"],
                  actual=("CUDAExecutionProvider",), has_executor=True)
        assert window._wait_for_tensorrt_build() is False  # noqa: SLF001

    def test_no_wait_when_trt_already_recorded(self, window, monkeypatch):
        self._set(window, monkeypatch, requested=["TensorrtExecutionProvider"],
                  actual=("TensorrtExecutionProvider", "CUDAExecutionProvider"), has_executor=True)
        assert window._wait_for_tensorrt_build() is False  # noqa: SLF001

    def test_no_wait_when_no_session(self, window, monkeypatch):
        self._set(window, monkeypatch, requested=["TensorrtExecutionProvider"],
                  actual=("CUDAExecutionProvider",), has_executor=False)
        assert window._wait_for_tensorrt_build() is False  # noqa: SLF001

    def test_shows_modal_on_toggle(self, window, monkeypatch):
        # Live CUDA session, user just enabled TRT → actual still CUDA → wait.
        from PySide6.QtWidgets import QProgressDialog
        self._set(window, monkeypatch, requested=["TensorrtExecutionProvider"],
                  actual=("CUDAExecutionProvider",), has_executor=True)
        assert window._wait_for_tensorrt_build() is True  # noqa: SLF001
        dialogs = window.findChildren(QProgressDialog)
        assert dialogs, "a modal compile dialog should be shown"
        for d in dialogs:
            d.close()

    def test_shows_modal_at_launch_when_nothing_recorded(self, window, monkeypatch):
        # The reported bug: launch with TRT persisted + no cached engine. No
        # session has recorded providers yet (actual = None) → must still wait.
        from PySide6.QtWidgets import QProgressDialog
        self._set(window, monkeypatch, requested=["TensorrtExecutionProvider"],
                  actual=None, has_executor=True)
        assert window._wait_for_tensorrt_build() is True  # noqa: SLF001
        dialogs = window.findChildren(QProgressDialog)
        assert dialogs, "a modal compile dialog should be shown at launch too"
        for d in dialogs:
            d.close()

    def test_reentrant_call_does_not_stack_a_second_dialog(self, window, monkeypatch):
        # Mid-build, an in-flight async swap completing emits
        # sessionScratchDirChanged and re-enters this function; the guards all
        # pass during a FIRST build (TRT not recorded yet, no engine on disk),
        # so without an active-wait flag a second dialog + timer stack over the
        # first (audit rank 37).
        from PySide6.QtWidgets import QProgressDialog
        self._set(window, monkeypatch, requested=["TensorrtExecutionProvider"],
                  actual=("CUDAExecutionProvider",), has_executor=True)
        assert window._wait_for_tensorrt_build() is True  # noqa: SLF001
        assert window._wait_for_tensorrt_build() is True  # noqa: SLF001 — took over, no new dialog
        visible = [
            d for d in window.findChildren(QProgressDialog) if d.isVisible()
        ]
        assert len(visible) == 1
        for d in window.findChildren(QProgressDialog):
            d.close()

    def test_no_modal_when_engine_already_cached(self, window, monkeypatch):
        # Toggling TRT off then on: it's not the active provider, but the engine
        # is already compiled on disk → fast load → NO modal flash.
        from PySide6.QtWidgets import QProgressDialog
        self._set(window, monkeypatch, requested=["TensorrtExecutionProvider"],
                  actual=("CUDAExecutionProvider",), has_executor=True, cached=True)
        assert window._wait_for_tensorrt_build() is False  # noqa: SLF001
        assert not window.findChildren(QProgressDialog)


class TestProviderHighlightDeferred:
    """A provider toggle rebuilds the chain via the ASYNC set_chain, so the
    'actual' providers ORT wired up aren't recorded until the dispatcher thread
    finishes the rebuild. The failed-provider highlight must wait for that
    recording — otherwise it compares the NEW request against the PREVIOUS
    session's providers and flashes a spurious red until the next toggle
    (the reported bug: re-checking a provider goes red, clears a toggle later)."""

    def _wire(self, window, monkeypatch, *, requested, state):
        from sinner2.pipeline import model_cache
        monkeypatch.setattr(window._processors, "swapper_providers",  # noqa: SLF001
                            lambda: requested)
        monkeypatch.setattr(window._controller, "executor", lambda: object())  # noqa: SLF001
        monkeypatch.setattr(model_cache, "get_actual_providers",
                            lambda: state["actual"])
        monkeypatch.setattr(window._controller, "effective_onnx_providers",  # noqa: SLF001
                            lambda: state["actual"] or ())
        marked: list[set] = []
        monkeypatch.setattr(window._processors, "mark_providers_failed",  # noqa: SLF001
                            lambda s: marked.append(set(s)))
        return marked

    def test_no_spurious_red_before_rebuild_records(self, window, qtbot, monkeypatch):
        # Stale actual is missing the just-requested TRT (rebuild not done yet).
        # The highlight must NOT fire on this stale value.
        req = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
               "CPUExecutionProvider"]
        state = {"actual": ("CUDAExecutionProvider", "CPUExecutionProvider")}
        marked = self._wire(window, monkeypatch, requested=req, state=state)
        window._schedule_provider_highlight_refresh()  # noqa: SLF001
        assert marked == []  # nothing highlighted on the stale snapshot
        # Async rebuild lands: ORT actually wired up all three.
        state["actual"] = tuple(req)
        qtbot.waitUntil(lambda: bool(marked), timeout=3000)
        assert marked[-1] == set()  # highlighted against truth → no failure

    def test_genuine_fallback_still_marks_red(self, window, qtbot, monkeypatch):
        # before = a working TRT session; the rebuild really falls back to
        # CUDA+CPU → the highlight must still go red (post-rebuild, not spurious).
        req = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
               "CPUExecutionProvider"]
        state = {"actual": tuple(req)}
        marked = self._wire(window, monkeypatch, requested=req, state=state)
        window._schedule_provider_highlight_refresh()  # noqa: SLF001
        state["actual"] = ("CUDAExecutionProvider", "CPUExecutionProvider")
        qtbot.waitUntil(lambda: bool(marked), timeout=3000)
        assert "TensorrtExecutionProvider" in marked[-1]

    def test_no_session_highlights_immediately(self, window, monkeypatch):
        # No live session → nothing to wait for → highlight right away.
        from sinner2.pipeline import model_cache
        monkeypatch.setattr(window._processors, "swapper_providers",  # noqa: SLF001
                            lambda: ["CUDAExecutionProvider"])
        monkeypatch.setattr(window._controller, "executor", lambda: None)  # noqa: SLF001
        monkeypatch.setattr(model_cache, "get_actual_providers", lambda: None)
        monkeypatch.setattr(window._controller, "effective_onnx_providers",  # noqa: SLF001
                            lambda: ("CUDAExecutionProvider",))
        marked: list[set] = []
        monkeypatch.setattr(window._processors, "mark_providers_failed",  # noqa: SLF001
                            lambda s: marked.append(set(s)))
        window._schedule_provider_highlight_refresh()  # noqa: SLF001
        assert marked == [set()]  # immediate, no deferral

    def test_highlight_refreshes_ep_status_panel(self, window, monkeypatch):
        # Regression: the EP status-bar cell stayed stale after a provider change
        # because only the checkbox highlight got the post-async-rebuild refresh.
        # _highlight_failed_providers runs at every "truth known" point, so it
        # must also update the EP cell to the effective providers.
        monkeypatch.setattr(window._processors, "swapper_providers",  # noqa: SLF001
                            lambda: ["CUDAExecutionProvider"])
        monkeypatch.setattr(window._controller, "effective_onnx_providers",  # noqa: SLF001
                            lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"))
        monkeypatch.setattr(window._processors, "mark_providers_failed",  # noqa: SLF001
                            lambda s: None)
        window._highlight_failed_providers()  # noqa: SLF001
        assert window._providers_panel.value() == "CUDA, CPU"  # noqa: SLF001


class TestKeyboardTransportIsAudioAware:
    """Spacebar / arrows / Home / End must route through the controller's
    audio-aware methods, not the executor directly (else audio desyncs)."""

    def _press(self, window, key):
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtGui import QKeyEvent

        ev = QKeyEvent(
            QKeyEvent.Type.KeyPress, key, _Qt.KeyboardModifier.NoModifier
        )
        window.keyPressEvent(ev)

    def test_space_routes_through_toggle_playback(self, window, monkeypatch):
        from unittest.mock import MagicMock
        from PySide6.QtCore import Qt as _Qt

        monkeypatch.setattr(window._controller, "executor", lambda: MagicMock())  # noqa: SLF001,E501
        called = []
        monkeypatch.setattr(  # noqa: SLF001
            window._controller, "toggle_playback", lambda: called.append(True)
        )
        self._press(window, _Qt.Key.Key_Space)
        assert called == [True]

    def test_arrow_routes_through_seek_to(self, window, monkeypatch):
        from unittest.mock import MagicMock
        from PySide6.QtCore import Qt as _Qt

        ex = MagicMock()
        ex.current_frame.get.return_value = 10
        monkeypatch.setattr(window._controller, "executor", lambda: ex)  # noqa: SLF001
        seeks = []
        monkeypatch.setattr(  # noqa: SLF001
            window._controller, "seek_to", lambda f: seeks.append(f)
        )
        self._press(window, _Qt.Key.Key_Right)
        assert seeks == [11]


class TestUpdateSettingsResilience:
    """A failed settings write must NOT leave the in-memory copy stale (rank 34):
    self._settings is the base for every later model_copy, so a save() OSError
    that prevented the assignment would corrupt all subsequent persistence."""

    def test_memory_stays_authoritative_when_save_fails(self, monkeypatch):
        from sinner2.config.settings import Settings
        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._settings = Settings()  # noqa: SLF001

        def boom(_s):
            raise OSError("disk full")

        monkeypatch.setattr(mw.user_settings, "save", boom)
        # Must not raise, and must update the in-memory settings anyway.
        win._update_settings(source_path="/new/src.png")  # noqa: SLF001
        assert win._settings.source_path == "/new/src.png"  # noqa: SLF001


class TestFaceAnalyzeSettingsPersistence:
    """D2: the Faces scan settings (stride/workers/preview/age-sex/precompute)
    persist across restarts via the Settings model."""

    def test_persist_writes_all_fields(self, monkeypatch):
        from unittest.mock import MagicMock

        from sinner2.config.settings import Settings
        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._settings = Settings()  # noqa: SLF001
        monkeypatch.setattr(mw.user_settings, "save", lambda _s: None)
        panel = MagicMock()
        panel.stride.return_value = 9
        panel.workers.return_value = 3
        panel.preview_enabled.return_value = False
        panel.detect_demographics.return_value = True
        panel.precompute_geometry.return_value = False
        panel.detection_size.return_value = 800
        panel.landmark_refine.return_value = True
        panel.landmark_min_score.return_value = 0.7
        panel.bake_angle.return_value = False
        win._face_map_panel = panel  # noqa: SLF001
        win._persist_face_analyze_settings()  # noqa: SLF001
        s = win._settings  # noqa: SLF001
        assert s.face_analyze_stride == 9
        assert s.face_analyze_workers == 3
        assert s.face_analyze_preview is False
        assert s.face_analyze_demographics is True
        assert s.face_analyze_precompute is False
        assert s.face_analyze_detection_size == 800
        assert s.face_analyze_landmark_refine is True
        assert s.face_analyze_landmark_min_score == 0.7
        assert s.face_analyze_bake_angle is False


class TestMetricsRateResetOnShow:
    """Re-showing the metrics overlay must reset the write/drop rate trackers
    (rank 38): the overlay timer stops while hidden, freezing the trackers, so
    the first reading after a re-show would otherwise be a delta smeared over the
    whole hidden interval."""

    def _win(self, monkeypatch):
        from unittest.mock import MagicMock

        from sinner2.config.settings import Settings
        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._settings = Settings()  # noqa: SLF001
        win._write_rate = MagicMock()  # noqa: SLF001
        win._drop_rate = MagicMock()  # noqa: SLF001
        win._metrics_overlay = MagicMock()  # noqa: SLF001
        monkeypatch.setattr(mw.user_settings, "save", lambda _s: None)
        return win

    def test_show_resets_rate_trackers(self, monkeypatch):
        win = self._win(monkeypatch)
        win._set_stats_visible(True)  # noqa: SLF001
        win._write_rate.reset.assert_called_once()  # noqa: SLF001
        win._drop_rate.reset.assert_called_once()  # noqa: SLF001

    def test_hide_does_not_reset(self, monkeypatch):
        win = self._win(monkeypatch)
        win._set_stats_visible(False)  # noqa: SLF001
        win._write_rate.reset.assert_not_called()  # noqa: SLF001
        win._drop_rate.reset.assert_not_called()  # noqa: SLF001


class TestSessionSwitchingDisablesControls:
    """During an async source/target swap the processor panel must be disabled
    too — not just the transport — else a config change mid-swap reaches the
    controller and is silently overwritten by reconfigure_from."""

    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._transport = MagicMock()  # noqa: SLF001
        win._processors = MagicMock()  # noqa: SLF001
        win._status_bar = MagicMock()  # noqa: SLF001
        win._session = MagicMock()  # noqa: SLF001
        win._batch_active = False  # noqa: SLF001
        return win

    def test_switching_disables_processor_panel_and_transport(self):
        win = self._win()
        win._on_session_switching(True)  # noqa: SLF001
        win._processors.setEnabled.assert_called_with(False)  # noqa: SLF001
        # Transport gated off via capabilities (none) rather than whole-widget.
        win._transport.apply_capabilities.assert_called()  # noqa: SLF001

    def test_ready_re_enables_processor_panel_and_transport(self):
        win = self._win()
        win._on_session_switching(False)  # noqa: SLF001
        win._processors.setEnabled.assert_called_with(True)  # noqa: SLF001
        # _refresh_transport_enabled re-applies the active session's caps.
        win._transport.apply_capabilities.assert_called()  # noqa: SLF001


class TestLiveMode:
    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._live = MagicMock()  # noqa: SLF001
        win._live_view = MagicMock()  # noqa: SLF001
        win._pickers = MagicMock()  # noqa: SLF001
        win._processors = MagicMock()  # noqa: SLF001
        win._status_bar = MagicMock()  # noqa: SLF001
        win._transport = MagicMock()  # noqa: SLF001
        win._controller = MagicMock()  # noqa: SLF001
        win._session = MagicMock()  # noqa: SLF001
        win._side_panel = MagicMock()  # noqa: SLF001 — faces toggle gating
        win._batch_active = False  # noqa: SLF001
        win._models_confirmed = True  # noqa: SLF001 — deferred confirm already done
        win._update_settings = MagicMock()  # noqa: SLF001 — persist no-op
        return win

    def test_use_camera_without_source_shows_message_and_no_activation(self):
        win = self._win()
        win._pickers.source_path.return_value = None  # noqa: SLF001
        win._on_use_camera()  # noqa: SLF001
        win._session.set_target.assert_not_called()  # noqa: SLF001
        win._status_bar.show_message.assert_called()  # noqa: SLF001

    def test_use_camera_with_source_sets_camera_target(self, tmp_path):
        from sinner2.gui.session_capabilities import CameraConfig

        win = self._win()
        src = tmp_path / "face.png"
        src.write_bytes(b"x")
        win._pickers.source_path.return_value = src  # noqa: SLF001
        for getter, value in (
            ("device", 2), ("width", 640), ("height", 480),
            ("fps", 24), ("workers", 3), ("port", 9000),
        ):
            getattr(win._live_view, getter).return_value = value  # noqa: SLF001
        win._on_use_camera()  # noqa: SLF001
        win._session.set_target.assert_called_once()  # noqa: SLF001
        cfg = win._session.set_target.call_args.args[0]  # noqa: SLF001
        assert isinstance(cfg, CameraConfig)
        assert cfg.device == 2 and cfg.mjpeg_port == 9000

    def test_running_updates_view_and_transport(self):
        win = self._win()
        win._live.sink_url.return_value = "http://localhost:8080/"  # noqa: SLF001
        win._on_live_running(True)  # noqa: SLF001
        win._live_view.set_running.assert_called_with(True)  # noqa: SLF001
        win._live_view.set_url.assert_called_with("http://localhost:8080/")  # noqa: SLF001
        win._transport.apply_capabilities.assert_called()  # noqa: SLF001

    def test_stopped_updates_view(self):
        win = self._win()
        win._on_live_running(False)  # noqa: SLF001
        win._live_view.set_running.assert_called_with(False)  # noqa: SLF001
        win._transport.apply_capabilities.assert_called()  # noqa: SLF001


class TestCapabilityChromeAndFps:
    """The transport + Settings chrome + FPS label follow the active session's
    capabilities (no more File/Live mode flag)."""

    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._session = MagicMock()  # noqa: SLF001
        win._transport = MagicMock()  # noqa: SLF001
        win._processors = MagicMock()  # noqa: SLF001
        win._fps_panel = MagicMock()  # noqa: SLF001
        win._batch_active = False  # noqa: SLF001
        return win

    def test_camera_caps_disable_seek_and_hide_file_groups(self):
        from sinner2.gui.session_capabilities import (
            SessionCapabilities,
            SessionKind,
        )

        win = self._win()
        win._session.active_kind.return_value = SessionKind.CAMERA  # noqa: SLF001
        win._on_capabilities_changed(SessionCapabilities.for_camera())  # noqa: SLF001
        win._transport.apply_capabilities.assert_called_once()  # noqa: SLF001
        win._processors.set_file_only_visible.assert_called_with(False)  # noqa: SLF001

    def test_file_caps_show_file_groups(self):
        from sinner2.gui.session_capabilities import (
            SessionCapabilities,
            SessionKind,
        )

        win = self._win()
        win._session.active_kind.return_value = SessionKind.FILE  # noqa: SLF001
        win._on_capabilities_changed(  # noqa: SLF001
            SessionCapabilities.for_file(has_audio=True)
        )
        win._processors.set_file_only_visible.assert_called_with(True)  # noqa: SLF001

    def test_live_fps_label_only_updates_for_camera(self):
        from sinner2.gui.session_capabilities import SessionKind

        win = self._win()
        win._session.active_kind.return_value = SessionKind.FILE  # noqa: SLF001
        win._update_live_fps_label(12.3)  # noqa: SLF001
        win._fps_panel.set_value.assert_not_called()  # noqa: SLF001
        win._session.active_kind.return_value = SessionKind.CAMERA  # noqa: SLF001
        win._update_live_fps_label(12.3)  # noqa: SLF001
        win._fps_panel.set_value.assert_called_with("12.3 fps")  # noqa: SLF001

    def test_file_fps_label_ignored_for_camera(self):
        from sinner2.gui.session_capabilities import SessionKind

        win = self._win()
        win._session.active_kind.return_value = SessionKind.CAMERA  # noqa: SLF001
        win._update_fps_label(30.0)  # noqa: SLF001
        win._fps_panel.set_value.assert_not_called()  # noqa: SLF001


class TestPathIsFileHelper:
    """_path_is_file must report an UNREACHABLE location as "not a file" rather
    than raising — on Windows a detached drive makes is_file() raise OSError, and
    that exception, hit during startup restore, would abort the whole launch."""

    def test_missing_path_returns_false(self, tmp_path):
        from sinner2.gui.main_window import _path_is_file

        assert _path_is_file(tmp_path / "nope.jpg") is False

    def test_existing_file_returns_true(self, tmp_path):
        from sinner2.gui.main_window import _path_is_file

        p = tmp_path / "f.bin"
        p.write_bytes(b"x")
        assert _path_is_file(p) is True

    def test_oserror_is_swallowed_to_false(self):
        from unittest.mock import MagicMock

        from sinner2.gui.main_window import _path_is_file

        fake = MagicMock()
        fake.is_file.side_effect = OSError("WinError 21: device not ready")
        assert _path_is_file(fake) is False


class TestStartupResilienceWithUnreachablePaths:
    """A persisted source/target on a detached drive (is_file raises OSError)
    must not crash startup — the path is skipped, the window comes up, and no
    session is built."""

    def test_window_constructs_when_persisted_path_raises(
        self, qtbot, monkeypatch, tmp_path
    ):
        import json as _json
        from pathlib import Path as _Path

        bad = r"Z:\detached\face.jpg"
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(
            _json.dumps({"source_path": bad, "target_path": bad}),
            encoding="utf-8",
        )
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(settings_path))

        # Simulate a detached drive: is_file() RAISES for the persisted path
        # (as Windows does for unready media) but behaves normally elsewhere.
        real_is_file = _Path.is_file

        def fake_is_file(self):
            if str(self) == bad:
                raise OSError("WinError 21: device not ready")
            return real_is_file(self)

        monkeypatch.setattr(_Path, "is_file", fake_is_file)

        from sinner2.gui.main_window import SinnerMainWindow

        win = SinnerMainWindow()  # must NOT raise
        qtbot.addWidget(win)
        assert win._controller.executor() is None  # noqa: SLF001
        assert win._pending_initial_target is None  # noqa: SLF001
        win.close()


class TestDeferredInitialSession:
    """The first session build (restored source+target) is deferred to showEvent
    so the window paints before model loading starts."""

    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._session = MagicMock()  # noqa: SLF001
        win._restoring_paths = False  # noqa: SLF001
        win._pending_initial_target = None  # noqa: SLF001
        win._initial_session_started = False  # noqa: SLF001
        win._models_confirmed = True  # noqa: SLF001 — deferred confirm already done
        win._refresh_transport_enabled = MagicMock()  # noqa: SLF001
        win._highlight_failed_providers = MagicMock()  # noqa: SLF001
        # A target change restores that target's remembered sections (transport
        # + executor) and face-map catalog; with no saved entry it clears them.
        win._transport = MagicMock()  # noqa: SLF001
        win._controller = MagicMock()  # noqa: SLF001
        win._face_map_ctl = MagicMock()  # noqa: SLF001
        win._settings = MagicMock()  # noqa: SLF001
        win._settings.sections_by_target = None  # noqa: SLF001
        return win

    def test_target_change_during_restore_defers_build(self):
        win = self._win()
        win._restoring_paths = True  # noqa: SLF001
        win._on_target_changed(Path("clip.mp4"))  # noqa: SLF001
        # Recorded, NOT built.
        assert win._pending_initial_target == Path("clip.mp4")  # noqa: SLF001
        win._session.set_target.assert_not_called()  # noqa: SLF001

    def test_target_change_outside_restore_builds_immediately(self):
        from sinner2.gui.session_capabilities import FileTarget

        win = self._win()
        win._on_target_changed(Path("clip.mp4"))  # noqa: SLF001
        win._session.set_target.assert_called_once_with(  # noqa: SLF001
            FileTarget(Path("clip.mp4"))
        )
        assert win._pending_initial_target is None  # noqa: SLF001

    def test_deferred_start_builds_pending_target(self):
        from sinner2.gui.session_capabilities import FileTarget

        win = self._win()
        win._pending_initial_target = Path("clip.mp4")  # noqa: SLF001
        win._start_deferred_initial_session()  # noqa: SLF001
        win._session.set_target.assert_called_once_with(  # noqa: SLF001
            FileTarget(Path("clip.mp4"))
        )
        # Consumed (can't fire twice) and chrome refreshed.
        assert win._pending_initial_target is None  # noqa: SLF001
        win._refresh_transport_enabled.assert_called_once()  # noqa: SLF001
        win._highlight_failed_providers.assert_called_once()  # noqa: SLF001

    def test_deferred_start_no_pending_is_noop(self):
        win = self._win()
        win._start_deferred_initial_session()  # noqa: SLF001
        win._session.set_target.assert_not_called()  # noqa: SLF001

    def test_show_schedules_deferred_build_once(self, qtbot, monkeypatch, tmp_path):
        # Restore a real source+target so a build IS pending, then assert showing
        # the window does NOT build during construction and schedules exactly one
        # deferred build that runs on the event loop.
        import json as _json

        src = tmp_path / "face.png"
        src.write_bytes(b"x")
        tgt = tmp_path / "clip.mp4"
        tgt.write_bytes(b"x")
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(
            _json.dumps({"source_path": str(src), "target_path": str(tgt)}),
            encoding="utf-8",
        )
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(settings_path))

        from sinner2.gui.main_window import SinnerMainWindow

        win = SinnerMainWindow()
        qtbot.addWidget(win)
        # Deferred: nothing built during __init__, but the target is recorded.
        assert win._controller.executor() is None  # noqa: SLF001
        assert win._pending_initial_target == tgt  # noqa: SLF001

        # Don't actually load models — stub the build the deferred starter calls.
        from unittest.mock import MagicMock

        win._session = MagicMock()  # noqa: SLF001
        win.show()
        qtbot.waitUntil(
            lambda: win._pending_initial_target is None, timeout=2000  # noqa: SLF001
        )
        win._session.set_target.assert_called_once()  # noqa: SLF001
        win.close()


class TestFullscreenTransportRehome:
    def test_fullscreen_exit_rehomes_transport_below_splitter(
        self, window, monkeypatch
    ):
        # Exit-fullscreen must re-home the transport relative to the splitter
        # index, not a hardcoded slot.
        monkeypatch.setattr(window, "showFullScreen", lambda: None)
        monkeypatch.setattr(window, "showMaximized", lambda: None)
        monkeypatch.setattr(window, "showNormal", lambda: None)
        monkeypatch.setattr(window, "isMaximized", lambda: False)
        layout = window._central_layout  # noqa: SLF001
        window._enter_fullscreen()  # noqa: SLF001
        window._exit_fullscreen()  # noqa: SLF001
        splitter_idx = layout.indexOf(window._top_splitter)  # noqa: SLF001
        assert layout.indexOf(window._transport) == splitter_idx + 1  # noqa: SLF001


class TestDeferredModelConfirm:
    """Settings restore must NOT pop the blocking model-download confirm (headless
    that hangs window construction; for a user it's a startup nag). The prompt is
    deferred to the first session build — "keep selection, prompt on first use"."""

    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._batch_active = False  # noqa: SLF001
        win._restoring_settings = False  # noqa: SLF001
        win._models_confirmed = False  # noqa: SLF001
        win._processors = MagicMock()  # noqa: SLF001
        win._session = MagicMock()  # noqa: SLF001
        win._detection_probe = MagicMock()  # noqa: SLF001
        win._confirm_optional_models = MagicMock()  # noqa: SLF001
        win._refresh_providers_label = MagicMock()  # noqa: SLF001
        win._refresh_session_indicators = MagicMock()  # noqa: SLF001
        win._wait_for_tensorrt_build = MagicMock(return_value=False)  # noqa: SLF001
        win._schedule_provider_highlight_refresh = MagicMock()  # noqa: SLF001
        snap = win._processors.snapshot.return_value  # noqa: SLF001
        snap.swapper_providers = []
        snap.swapper_params.detection_size = 640
        return win

    def test_restore_skips_confirm_but_still_applies(self):
        win = self._win()
        win._restoring_settings = True  # noqa: SLF001
        win._on_processor_config_changed()  # noqa: SLF001
        win._confirm_optional_models.assert_not_called()  # noqa: SLF001
        win._session.apply_settings.assert_called_once()  # noqa: SLF001 — config seeded
        assert win._models_confirmed is False  # noqa: SLF001 — not yet confirmed

    def test_user_change_runs_confirm(self):
        win = self._win()
        win._on_processor_config_changed()  # noqa: SLF001
        win._confirm_optional_models.assert_called_once()  # noqa: SLF001
        assert win._models_confirmed is True  # noqa: SLF001

    def test_deferred_confirm_runs_once_then_gated(self):
        win = self._win()
        win._ensure_models_confirmed_before_build()  # noqa: SLF001
        win._confirm_optional_models.assert_called_once()  # noqa: SLF001
        assert win._models_confirmed is True  # noqa: SLF001
        win._ensure_models_confirmed_before_build()  # noqa: SLF001 — gated
        win._confirm_optional_models.assert_called_once()  # noqa: SLF001

    def test_deferred_confirm_noop_when_already_confirmed(self):
        win = self._win()
        win._models_confirmed = True  # noqa: SLF001
        win._ensure_models_confirmed_before_build()  # noqa: SLF001
        win._confirm_optional_models.assert_not_called()  # noqa: SLF001


class TestCacheStatsAsync:
    """The cache size/count is a stat-walk of every cache dir — run off the GUI
    thread (queued result), skipped during close, and a stale walk dropped by gen."""

    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._closing = False  # noqa: SLF001
        win._cache_stats_gen = 5  # noqa: SLF001
        win._processors = MagicMock()  # noqa: SLF001
        win._controller = MagicMock()  # noqa: SLF001
        win._cache_stats_cb = MagicMock()  # noqa: SLF001
        return win

    def test_apply_sets_label_for_current_gen(self):
        win = self._win()
        win._apply_cache_stats((5, 3, 2 * 1024**3, 50 * 1024**3))  # noqa: SLF001
        text = win._processors.set_cache_stats_text.call_args[0][0]  # noqa: SLF001
        assert "3 entries" in text

    def test_apply_ignores_superseded_gen(self):
        win = self._win()
        win._apply_cache_stats((4, 9, 0, 0))  # noqa: SLF001 — gen 4 < current 5
        win._processors.set_cache_stats_text.assert_not_called()  # noqa: SLF001

    def test_apply_skipped_when_closing(self):
        win = self._win()
        win._closing = True  # noqa: SLF001
        win._apply_cache_stats((5, 3, 0, 0))  # noqa: SLF001
        win._processors.set_cache_stats_text.assert_not_called()  # noqa: SLF001

    def test_refresh_skipped_when_closing_does_no_walk(self):
        win = self._win()
        win._closing = True  # noqa: SLF001
        win._refresh_cache_stats()  # noqa: SLF001
        win._processors.set_invalidate_enabled.assert_not_called()  # noqa: SLF001
        win._controller.cache_manager.assert_not_called()  # noqa: SLF001 — no walk

    def test_refresh_walks_off_thread_and_reports(self, qtbot):
        from unittest.mock import MagicMock

        win = self._win()
        win._controller.executor.return_value = None  # noqa: SLF001
        mgr = MagicMock()
        mgr.list_entries.return_value = []
        mgr.free_disk_bytes.return_value = 1024
        win._controller.cache_manager.return_value = mgr  # noqa: SLF001
        win._refresh_cache_stats()  # noqa: SLF001 — returns immediately (dispatches a thread)
        win._processors.set_invalidate_enabled.assert_called_once()  # noqa: SLF001
        qtbot.waitUntil(lambda: win._cache_stats_cb.called, timeout=2000)  # noqa: SLF001
        payload = win._cache_stats_cb.call_args[0][0]  # noqa: SLF001
        assert payload[0] == 6 and payload[1] == 0  # noqa: SLF001 — gen bumped, 0 entries


class TestStatusBarInfoPanels:
    """Resolution / display-fps / workers / drops panels composed from the new
    controller signals + accessors."""

    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._session = MagicMock()  # noqa: SLF001
        win._controller = MagicMock()  # noqa: SLF001
        win._processors = MagicMock()  # noqa: SLF001
        win._resolution_panel = MagicMock()  # noqa: SLF001
        win._workers_panel = MagicMock()  # noqa: SLF001
        win._display_fps_panel = MagicMock()  # noqa: SLF001
        win._drops_panel = MagicMock()  # noqa: SLF001
        win._native_size = None  # noqa: SLF001
        win._frames_skipped = 0  # noqa: SLF001
        win._write_dropped = 0  # noqa: SLF001
        return win

    def test_resolution_native_and_source_fps(self):
        win = self._win()
        win._native_size = (1920, 1080)  # noqa: SLF001
        win._controller.target_fps.return_value = 30.0  # noqa: SLF001
        win._processors.processing_scale.return_value = 1.0  # noqa: SLF001
        win._update_resolution_panel()  # noqa: SLF001
        win._resolution_panel.set_value.assert_called_with("1920×1080 @30")  # noqa: SLF001

    def test_resolution_shows_downscaled_size(self):
        win = self._win()
        win._native_size = (1920, 1080)  # noqa: SLF001
        win._controller.target_fps.return_value = 30.0  # noqa: SLF001
        win._processors.processing_scale.return_value = 0.5  # noqa: SLF001
        win._update_resolution_panel()  # noqa: SLF001
        win._resolution_panel.set_value.assert_called_with(  # noqa: SLF001
            "1920×1080 @30 → 960×540"
        )

    def test_resolution_hidden_without_size(self):
        win = self._win()
        win._update_resolution_panel()  # noqa: SLF001
        win._resolution_panel.set_value.assert_called_with("")  # noqa: SLF001

    def test_workers_shown_only_with_active_session(self):
        win = self._win()
        win._native_size = None  # noqa: SLF001 — resolution hides, we test workers
        win._controller.executor.return_value = object()  # noqa: SLF001
        win._controller.applied_worker_count.return_value = 4  # noqa: SLF001
        win._refresh_session_indicators()  # noqa: SLF001
        win._workers_panel.set_value.assert_called_with("4")  # noqa: SLF001

    def test_workers_hidden_without_session(self):
        win = self._win()
        win._controller.executor.return_value = None  # noqa: SLF001
        win._refresh_session_indicators()  # noqa: SLF001
        win._workers_panel.set_value.assert_called_with("")  # noqa: SLF001

    def test_drops_combines_skip_and_drop(self):
        win = self._win()
        win._frames_skipped = 12  # noqa: SLF001
        win._write_dropped = 3  # noqa: SLF001
        win._update_drops_panel()  # noqa: SLF001
        win._drops_panel.set_value.assert_called_with("12 skip · 3 drop")  # noqa: SLF001

    def test_drops_hidden_when_none(self):
        win = self._win()
        win._update_drops_panel()  # noqa: SLF001
        win._drops_panel.set_value.assert_called_with("")  # noqa: SLF001

    def test_frames_skipped_signal_updates_drops(self):
        win = self._win()
        win._on_frames_skipped(7)  # noqa: SLF001
        win._drops_panel.set_value.assert_called_with("7 skip")  # noqa: SLF001

    def test_native_size_none_clears_session_cells(self):
        from unittest.mock import MagicMock

        win = self._win()
        win._frames_skipped = 5  # noqa: SLF001
        win._write_dropped = 2  # noqa: SLF001
        win._refresh_session_indicators = MagicMock()  # noqa: SLF001
        win._on_native_size_changed(None)  # noqa: SLF001
        win._display_fps_panel.set_value.assert_called_with("")  # noqa: SLF001
        assert win._frames_skipped == 0  # noqa: SLF001
        assert win._write_dropped == 0  # noqa: SLF001

    def test_display_fps_gated_to_non_camera(self):
        from sinner2.gui.session_capabilities import SessionKind

        win = self._win()
        win._session.active_kind.return_value = SessionKind.CAMERA  # noqa: SLF001
        win._update_display_fps_label(30.0)  # noqa: SLF001
        win._display_fps_panel.set_value.assert_not_called()  # noqa: SLF001
        win._session.active_kind.return_value = SessionKind.FILE  # noqa: SLF001
        win._update_display_fps_label(30.0)  # noqa: SLF001
        win._display_fps_panel.set_value.assert_called_with("30.0 fps")  # noqa: SLF001


class TestStatusPanelMenuPersistence:
    """Right-click panel-visibility menu: choices persist to settings and are
    re-applied to the panels on the next launch."""

    def test_status_panels_restored_from_settings(self, qtbot, tmp_path, monkeypatch):
        import json

        from sinner2.gui.main_window import SinnerMainWindow

        settings_path = tmp_path / "settings.json"
        settings_path.write_text(
            json.dumps({"status_panels_hidden": ["fps", "drops"]}), encoding="utf-8"
        )
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(settings_path))
        w = SinnerMainWindow()
        qtbot.addWidget(w)
        assert not w._fps_panel.user_visible()  # noqa: SLF001 — hidden per settings
        assert not w._drops_panel.user_visible()  # noqa: SLF001
        assert w._cache_panel.user_visible()  # noqa: SLF001 — not listed → visible

    def test_toggle_persists_hidden_keys(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._status_bar = MagicMock()  # noqa: SLF001
        win._status_bar.hidden_panel_keys.return_value = ["fps", "drops"]  # noqa: SLF001
        win._update_settings = MagicMock()  # noqa: SLF001
        win._on_panel_visibility_changed("fps", False)  # noqa: SLF001
        win._update_settings.assert_called_once_with(  # noqa: SLF001
            status_panels_hidden=["fps", "drops"]
        )

    def test_toggle_with_nothing_hidden_persists_none(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._status_bar = MagicMock()  # noqa: SLF001
        win._status_bar.hidden_panel_keys.return_value = []  # noqa: SLF001
        win._update_settings = MagicMock()  # noqa: SLF001
        win._on_panel_visibility_changed("fps", True)  # noqa: SLF001
        win._update_settings.assert_called_once_with(status_panels_hidden=None)  # noqa: SLF001
