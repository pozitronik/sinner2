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
    """The whole transport row is inactive until a source AND target load."""

    def test_disabled_without_source_and_target(self, window):
        assert window._transport.isEnabled() is False  # noqa: SLF001

    def test_enables_only_when_both_loaded(self, window, tmp_path, monkeypatch):
        monkeypatch.setattr(
            window._controller,  # noqa: SLF001
            "set_source_and_target",
            lambda *a, **k: None,
        )
        src = tmp_path / "s.png"
        src.write_bytes(b"x")
        window._pickers.set_source(src)  # noqa: SLF001
        assert window._transport.isEnabled() is False  # noqa: SLF001  # source only
        tgt = tmp_path / "t.mp4"
        tgt.write_bytes(b"x")
        window._pickers.set_target(tgt)  # noqa: SLF001
        assert window._transport.isEnabled() is True  # noqa: SLF001  # both


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

    def test_batch_running_locks_editing_surface(
        self, window, tmp_path, monkeypatch
    ):
        # DaVinci-style: a running batch locks transport + pickers + settings
        # + libraries, but the Batch tab stays interactive.
        # Load a source + target (controller stubbed to skip the heavy session
        # build) so the transport is live before the lock and re-enables after.
        monkeypatch.setattr(
            window._controller,  # noqa: SLF001
            "set_source_and_target",
            lambda *a, **k: None,
        )
        src = tmp_path / "src.png"
        src.write_bytes(b"x")
        tgt = tmp_path / "tgt.mp4"
        tgt.write_bytes(b"x")
        window._pickers.set_source(src)  # noqa: SLF001
        window._pickers.set_target(tgt)  # noqa: SLF001
        assert window._transport.isEnabled()  # noqa: SLF001  # source+target loaded

        window._batch_queue.taskStarted.emit("x")  # noqa: SLF001
        assert not window._transport.isEnabled()  # noqa: SLF001
        assert not window._pickers.isEnabled()  # noqa: SLF001
        assert not window._processors.isEnabled()  # noqa: SLF001
        assert not window._side_panel.sources_library().isEnabled()  # noqa: SLF001
        assert not window._side_panel.targets_library().isEnabled()  # noqa: SLF001
        assert window._batch_view.isEnabled()  # noqa: SLF001  queue stays usable
        window._batch_queue.queueIdle.emit()  # noqa: SLF001
        assert window._transport.isEnabled()  # noqa: SLF001  # re-enabled
        assert window._pickers.isEnabled()  # noqa: SLF001
        assert window._processors.isEnabled()  # noqa: SLF001

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
        return win

    def test_switching_disables_processor_panel(self):
        win = self._win()
        win._on_session_switching(True)  # noqa: SLF001
        win._processors.setEnabled.assert_called_with(False)  # noqa: SLF001
        win._transport.setEnabled.assert_called_with(False)  # noqa: SLF001

    def test_ready_re_enables_processor_panel(self):
        win = self._win()
        win._on_session_switching(False)  # noqa: SLF001
        win._processors.setEnabled.assert_called_with(True)  # noqa: SLF001
        win._transport.setEnabled.assert_called_with(True)  # noqa: SLF001


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
        return win

    def test_start_without_source_shows_message_and_no_start(self):
        win = self._win()
        win._pickers.source_path.return_value = None  # noqa: SLF001
        win._on_live_start()  # noqa: SLF001
        win._live.start.assert_not_called()  # noqa: SLF001
        win._status_bar.show_message.assert_called()  # noqa: SLF001

    def test_start_with_source_delegates_with_snapshot(self, tmp_path):
        win = self._win()
        src = tmp_path / "face.png"
        src.write_bytes(b"x")
        win._pickers.source_path.return_value = src  # noqa: SLF001
        for getter, value in (
            ("device", 0), ("width", 1280), ("height", 720),
            ("fps", 30), ("port", 8080),
        ):
            getattr(win._live_view, getter).return_value = value  # noqa: SLF001
        win._on_live_start()  # noqa: SLF001
        win._live.start.assert_called_once()  # noqa: SLF001
        win._controller.pause.assert_called_once()  # noqa: SLF001  live takes over
        kwargs = win._live.start.call_args.kwargs  # noqa: SLF001
        assert kwargs["source_path"] == src
        assert kwargs["snapshot"] is win._processors.snapshot.return_value  # noqa: SLF001
        assert kwargs["device"] == 0 and kwargs["mjpeg_port"] == 8080


    def test_start_without_source_does_not_pause(self):
        win = self._win()
        win._pickers.source_path.return_value = None  # noqa: SLF001
        win._on_live_start()  # noqa: SLF001
        win._controller.pause.assert_not_called()  # noqa: SLF001

    def test_running_disables_transport_and_updates_view(self):
        win = self._win()
        win._live.sink_url.return_value = "http://localhost:8080/"  # noqa: SLF001
        win._on_live_running(True)  # noqa: SLF001
        win._live_view.set_running.assert_called_with(True)  # noqa: SLF001
        win._live_view.set_url.assert_called_with("http://localhost:8080/")  # noqa: SLF001
        win._transport.setEnabled.assert_called_with(False)  # noqa: SLF001

    def test_stopped_re_enables_transport(self):
        win = self._win()
        win._on_live_running(False)  # noqa: SLF001
        win._live_view.set_running.assert_called_with(False)  # noqa: SLF001
        win._transport.setEnabled.assert_called_with(True)  # noqa: SLF001


class TestModeToggle:
    def _win(self):
        from unittest.mock import MagicMock

        from sinner2.gui import main_window as mw

        win = mw.SinnerMainWindow.__new__(mw.SinnerMainWindow)
        win._mode = "file"  # noqa: SLF001
        win._status_bar = MagicMock()  # noqa: SLF001  carries the mode_button
        win._controller = MagicMock()  # noqa: SLF001
        win._live = MagicMock()  # noqa: SLF001
        win._transport = MagicMock()  # noqa: SLF001
        win._pickers = MagicMock()  # noqa: SLF001
        win._processors = MagicMock()  # noqa: SLF001
        win._side_panel = MagicMock()  # noqa: SLF001
        win._fps_label = MagicMock()  # noqa: SLF001
        win._refresh_transport_enabled = MagicMock()  # noqa: SLF001
        return win

    def test_switch_to_live_pauses_file_and_hides_file_chrome(self):
        win = self._win()
        win._controller.executor.return_value = object()  # noqa: SLF001
        win._set_mode("live")  # noqa: SLF001
        assert win._mode == "live"  # noqa: SLF001
        win._controller.pause.assert_called_once()  # noqa: SLF001 paused, not torn down
        win._live.stop.assert_not_called()  # noqa: SLF001
        win._transport.setVisible.assert_called_with(False)  # noqa: SLF001
        win._pickers.set_target_visible.assert_called_with(False)  # noqa: SLF001
        win._processors.set_file_only_visible.assert_called_with(False)  # noqa: SLF001
        win._side_panel.set_mode.assert_called_with("live")  # noqa: SLF001

    def test_switch_to_live_does_not_autostart_camera(self):
        win = self._win()
        win._controller.executor.return_value = None  # noqa: SLF001
        win._set_mode("live")  # noqa: SLF001
        win._live.start.assert_not_called()  # noqa: SLF001  user presses Start

    def test_switch_to_file_stops_live_and_restores_chrome(self):
        win = self._win()
        win._mode = "live"  # noqa: SLF001
        win._set_mode("file")  # noqa: SLF001
        assert win._mode == "file"  # noqa: SLF001
        win._live.stop.assert_called_once()  # noqa: SLF001
        win._transport.setVisible.assert_called_with(True)  # noqa: SLF001
        win._pickers.set_target_visible.assert_called_with(True)  # noqa: SLF001
        win._processors.set_file_only_visible.assert_called_with(True)  # noqa: SLF001
        win._side_panel.set_mode.assert_called_with("file")  # noqa: SLF001
        win._refresh_transport_enabled.assert_called_once()  # noqa: SLF001

    def test_same_mode_is_noop(self):
        win = self._win()  # already "file"
        win._set_mode("file")  # noqa: SLF001
        win._side_panel.set_mode.assert_not_called()  # noqa: SLF001
        win._controller.pause.assert_not_called()  # noqa: SLF001
        win._live.stop.assert_not_called()  # noqa: SLF001

    def test_switch_clears_fps_label(self):
        win = self._win()
        win._controller.executor.return_value = None  # noqa: SLF001
        win._set_mode("live")  # noqa: SLF001
        win._fps_label.setText.assert_called_with("--- fps")  # noqa: SLF001

    def test_live_fps_label_only_updates_in_live_mode(self):
        win = self._win()  # file mode
        win._update_live_fps_label(12.3)  # noqa: SLF001
        win._fps_label.setText.assert_not_called()  # noqa: SLF001
        win._mode = "live"  # noqa: SLF001
        win._update_live_fps_label(12.3)  # noqa: SLF001
        win._fps_label.setText.assert_called_with("12.3 fps")  # noqa: SLF001

    def test_file_fps_label_ignored_in_live_mode(self):
        win = self._win()
        win._mode = "live"  # noqa: SLF001
        win._update_fps_label(30.0)  # noqa: SLF001
        win._fps_label.setText.assert_not_called()  # noqa: SLF001

    def test_default_mode_is_file_with_button_unchecked_and_live_tab_hidden(
        self, window
    ):
        # Real window: defaults to file mode; the status-bar toggle is unchecked.
        assert window._mode == "file"  # noqa: SLF001
        assert not window._status_bar.mode_button.isChecked()  # noqa: SLF001
        live_idx = window._side_panel.indexOf(window._live_view)  # noqa: SLF001
        assert not window._side_panel.isTabVisible(live_idx)  # noqa: SLF001

    def test_mode_button_toggles_mode(self, window):
        assert window._mode == "file"  # noqa: SLF001
        window._status_bar.mode_button.setChecked(True)  # noqa: SLF001 fires toggled
        assert window._mode == "live"  # noqa: SLF001
        window._status_bar.mode_button.setChecked(False)  # noqa: SLF001
        assert window._mode == "file"  # noqa: SLF001

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
