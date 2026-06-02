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
        monkeypatch.setattr(window._controller, "executor", lambda: (MagicMock() if has_executor else None))  # noqa: SLF001

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
