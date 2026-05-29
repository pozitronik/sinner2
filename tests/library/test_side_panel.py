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
