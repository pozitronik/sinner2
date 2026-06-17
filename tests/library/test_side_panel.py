"""Integration tests for QSidePanel.

Verifies the side panel exposes the expected accessors, owns its own
thumbnail generator (no leakage across instances), and provides the
processors widget at the right tab index.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.gui.widgets.processor_controls import QProcessorControls
from sinner2.gui.widgets.side_panel import QSidePanel
from sinner2.library.media_kind import is_image, is_media


@pytest.fixture
def side_panel(qtbot, tmp_path: Path) -> QSidePanel:
    panel = QSidePanel(
        thumbnail_cache_dir=tmp_path / "thumbs",
        processors=QProcessorControls(),
        thumb_extract_dim=256,
        thumb_display_dim=128,
        thumb_workers=2,
    )
    qtbot.addWidget(panel)
    yield panel
    panel.shutdown()


class TestQSidePanel:
    def test_three_tabs_in_expected_order(self, side_panel):
        assert side_panel.count() == 3
        assert side_panel.tabText(0) == "Settings"
        assert side_panel.tabText(1) == "Sources"
        assert side_panel.tabText(2) == "Targets"

    def test_processors_accessor_returns_provided_instance(
        self, qtbot, tmp_path
    ):
        proc = QProcessorControls()
        panel = QSidePanel(
            thumbnail_cache_dir=tmp_path / "thumbs", processors=proc
        )
        qtbot.addWidget(panel)
        try:
            assert panel.processors() is proc
        finally:
            panel.shutdown()

    def test_sources_library_uses_image_filter(self, side_panel):
        # Cheapest assertion: the accept predicate on sources is is_image,
        # on targets is is_media. We can verify by direct attribute peek.
        assert side_panel.sources_library()._accept is is_image  # noqa: SLF001
        assert side_panel.targets_library()._accept is is_media  # noqa: SLF001

    def test_libraries_are_independent_widgets(self, side_panel):
        # Adding to sources must not propagate to targets.
        assert side_panel.sources_library() is not side_panel.targets_library()
        assert side_panel.sources_library().paths() == []
        assert side_panel.targets_library().paths() == []

    def test_faces_subpanel_lives_on_sources_tab_no_separate_tab(
        self, qtbot, tmp_path
    ):
        from PySide6.QtWidgets import QWidget

        faces = QWidget()
        panel = QSidePanel(
            thumbnail_cache_dir=tmp_path / "thumbs",
            processors=QProcessorControls(),
            face_map_panel=faces,
        )
        qtbot.addWidget(panel)
        try:
            # No separate "Faces" tab — it's a subpanel of Sources, hidden until
            # the toggle, and exposed via face_map_panel().
            assert "Faces" not in [panel.tabText(i) for i in range(panel.count())]
            assert panel.face_map_panel() is faces
            assert faces.isHidden() is True
            assert panel.faces_mode() is False
            with qtbot.waitSignal(panel.facesModeToggled) as blocker:
                panel._faces_toggle.setChecked(True)  # noqa: SLF001
            assert blocker.args == [True]
            assert panel.faces_mode() is True
            assert faces.isHidden() is False
        finally:
            panel.shutdown()

    def test_set_faces_available_disables_and_clears_mode(self, qtbot, tmp_path):
        from PySide6.QtWidgets import QWidget

        faces = QWidget()
        panel = QSidePanel(
            thumbnail_cache_dir=tmp_path / "thumbs",
            processors=QProcessorControls(),
            face_map_panel=faces,
        )
        qtbot.addWidget(panel)
        try:
            panel._faces_toggle.setChecked(True)  # noqa: SLF001
            with qtbot.waitSignal(panel.facesModeToggled) as blocker:
                panel.set_faces_available(False)  # camera session → file-only off
            assert blocker.args == [False]  # mode cleared
            assert panel.faces_mode() is False
            assert panel._faces_toggle.isEnabled() is False  # noqa: SLF001
        finally:
            panel.shutdown()

    def test_open_face_map_editor_selects_tab_and_toggles(self, qtbot, tmp_path):
        from PySide6.QtWidgets import QWidget

        faces = QWidget()
        panel = QSidePanel(
            thumbnail_cache_dir=tmp_path / "thumbs",
            processors=QProcessorControls(),
            face_map_panel=faces,
        )
        qtbot.addWidget(panel)
        try:
            with qtbot.waitSignal(panel.facesModeToggled) as blocker:
                panel.open_face_map_editor()
            assert blocker.args == [True] and panel.faces_mode() is True
            assert panel.tabText(panel.currentIndex()) == "Sources"
            # No-op when face-mapping is unavailable (e.g. camera).
            panel._faces_toggle.setChecked(False)  # noqa: SLF001
            panel.set_faces_available(False)
            panel.open_face_map_editor()
            assert panel.faces_mode() is False
        finally:
            panel.shutdown()

    def test_set_display_dim_mirrors_to_both_libraries(self, side_panel):
        # The side panel's set_display_dim is what main_window calls
        # to keep source/target tiles in sync; verify it actually
        # propagates to both views (regression catcher for a future
        # refactor that might decouple them).
        side_panel.set_display_dim(96)
        assert side_panel.sources_library().display_dim() == 96
        assert side_panel.targets_library().display_dim() == 96
        # And display_dim() accessor reads from either view (they're
        # in sync, so reading from sources is canonical).
        assert side_panel.display_dim() == 96

    def test_set_editing_locked_toggles_editing_tabs(self, side_panel):
        side_panel.set_editing_locked(True)
        assert not side_panel.processors().isEnabled()
        assert not side_panel.sources_library().isEnabled()
        assert not side_panel.targets_library().isEnabled()
        side_panel.set_editing_locked(False)
        assert side_panel.processors().isEnabled()
        assert side_panel.sources_library().isEnabled()
        assert side_panel.targets_library().isEnabled()

    def test_set_mode_toggles_live_and_file_tabs(self, qtbot, tmp_path):
        from typing import cast

        from PySide6.QtWidgets import QWidget

        from sinner2.gui.widgets.batch_view import QBatchView

        live = QWidget()
        batch = QWidget()
        panel = QSidePanel(
            thumbnail_cache_dir=tmp_path / "thumbs",
            processors=QProcessorControls(),
            batch_view=cast(QBatchView, batch),  # set_mode only toggles its tab
            live_view=live,
        )
        qtbot.addWidget(panel)
        try:
            ti = panel.indexOf(panel.targets_library())
            bi = panel.indexOf(batch)
            li = panel.indexOf(live)
            panel.set_mode("live")
            assert not panel.isTabVisible(ti)       # no file target in live
            assert not panel.isTabVisible(bi)       # batch is file-only
            assert panel.isTabVisible(li)           # live shown
            assert panel.currentWidget() is live    # and selected
            panel.set_mode("file")
            assert panel.isTabVisible(ti)           # targets restored
            assert panel.isTabVisible(bi)           # batch restored
            assert not panel.isTabVisible(li)       # live hidden
        finally:
            panel.shutdown()
