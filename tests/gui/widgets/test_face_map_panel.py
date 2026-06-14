"""Tests for the Faces panel (face-mapping UI)."""
from __future__ import annotations

import pytest

from sinner2.gui.widgets.face_map_panel import QFaceMapPanel
from sinner2.pipeline.face_map import FaceMap, Identity, normalize


@pytest.fixture
def panel(qtbot):
    p = QFaceMapPanel()
    qtbot.addWidget(p)
    return p


def _map(*ids):
    return FaceMap(
        identities=tuple(
            Identity(i, normalize([1, 0, 0]), occurrences=n) for i, n in ids
        )
    )


class TestCards:
    def test_empty_shows_hint(self, panel):
        assert panel._hint.isVisibleTo(panel)  # noqa: SLF001
        assert panel.face_map().is_empty()

    def test_set_face_map_builds_cards_and_hides_hint(self, panel):
        panel.set_face_map(_map(("a", 5), ("b", 3)))
        assert set(panel._cards) == {"a", "b"}  # noqa: SLF001
        assert not panel._hint.isVisibleTo(panel)  # noqa: SLF001

    def test_rebuild_replaces_cards(self, panel):
        panel.set_face_map(_map(("a", 1), ("b", 1)))
        panel.set_face_map(_map(("c", 1)))
        assert set(panel._cards) == {"c"}  # noqa: SLF001


class TestSelection:
    def test_click_selects_and_emits(self, panel, qtbot):
        panel.set_face_map(_map(("a", 1), ("b", 1)))
        with qtbot.waitSignal(panel.identitySelected) as blocker:
            panel._cards["a"].selected.emit("a")  # noqa: SLF001 — simulate click
        assert blocker.args == ["a"]
        assert panel.selected_identity() == "a"

    def test_reclick_deselects(self, panel):
        panel.set_face_map(_map(("a", 1)))
        panel._on_card_selected("a")  # noqa: SLF001
        assert panel.selected_identity() == "a"
        panel._on_card_selected("a")  # noqa: SLF001 — toggle off
        assert panel.selected_identity() is None

    def test_selection_cleared_when_identity_gone(self, panel):
        panel.set_face_map(_map(("a", 1)))
        panel._on_card_selected("a")  # noqa: SLF001
        panel.set_face_map(_map(("b", 1)))  # 'a' removed
        assert panel.selected_identity() is None


class TestAnalyze:
    def test_analyze_emits_stride(self, panel, qtbot):
        panel._stride.setValue(20)  # noqa: SLF001
        with qtbot.waitSignal(panel.analyzeRequested) as blocker:
            panel._analyze_btn.click()  # noqa: SLF001
        assert blocker.args == [20]

    def test_analyzing_state_toggles_to_cancel(self, panel, qtbot):
        panel.set_analyzing(True)
        assert panel._analyze_btn.text() == "Cancel"  # noqa: SLF001
        assert panel._progress.isVisibleTo(panel)  # noqa: SLF001
        with qtbot.waitSignal(panel.cancelRequested):
            panel._analyze_btn.click()  # noqa: SLF001

    def test_progress_updates_bar(self, panel):
        panel.set_analyzing(True)
        panel.set_progress(3, 10)
        assert panel._progress.maximum() == 10  # noqa: SLF001
        assert panel._progress.value() == 3  # noqa: SLF001

    def test_done_hides_progress(self, panel):
        panel.set_analyzing(True)
        panel.set_analyzing(False)
        assert not panel._progress.isVisibleTo(panel)  # noqa: SLF001
        assert panel._analyze_btn.text() == "Analyze faces"  # noqa: SLF001


class TestDemographicsAndGrouping:
    def test_demographics_default_off(self, panel):
        assert panel.detect_demographics() is False

    def test_card_shows_sex_and_age(self, panel):
        fm = FaceMap(
            identities=(Identity("a", normalize([1, 0]), occurrences=5, sex="M", age=34),)
        )
        panel.set_face_map(fm)
        text = panel._cards["a"]._count.text()  # noqa: SLF001
        assert "M" in text and "34y" in text

    def test_group_by_sex_orders_identities(self, panel):
        fm = FaceMap(
            identities=(
                Identity("a", normalize([1, 0, 0]), occurrences=5, sex="F"),
                Identity("b", normalize([0, 1, 0]), occurrences=3, sex="M"),
                Identity("c", normalize([0, 0, 1]), occurrences=9),  # unknown
            )
        )
        panel.set_face_map(fm)
        panel._group_check.setChecked(True)  # noqa: SLF001 — triggers re-sort
        order = [i.id for i in panel._ordered_identities()]  # noqa: SLF001
        assert order == ["b", "a", "c"]  # M, then F, then unknown


class TestDelete:
    def test_delete_emits(self, panel, qtbot):
        panel.set_face_map(_map(("a", 1)))
        with qtbot.waitSignal(panel.deleteIdentityRequested) as blocker:
            panel._cards["a"].deleteRequested.emit("a")  # noqa: SLF001
        assert blocker.args == ["a"]
