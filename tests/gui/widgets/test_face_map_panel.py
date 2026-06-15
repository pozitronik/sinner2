"""Tests for the Faces panel (sortable findings table)."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from sinner2.gui.widgets.face_map_panel import (
    _C_APP,
    _C_FACE,
    _C_SEX,
    _ID_ROLE,
    QFaceMapPanel,
)
from sinner2.pipeline.face_map import FaceMap, Identity, normalize


@pytest.fixture
def panel(qtbot):
    p = QFaceMapPanel()
    qtbot.addWidget(p)
    return p


def _map(*specs):
    """specs: (id, occurrences, sex, age, first_frame)."""
    idents = []
    for i, (cid, occ, sex, age, first) in enumerate(specs):
        idents.append(
            Identity(cid, normalize([1, 0, i + 1]), occurrences=occ, sex=sex, age=age,
                     first_frame=first)
        )
    return FaceMap(identities=tuple(idents))


def _row_ids(panel):
    return [
        panel._model.item(r, _C_FACE).data(_ID_ROLE)  # noqa: SLF001
        for r in range(panel._model.rowCount())  # noqa: SLF001
    ]


class TestTable:
    def test_set_face_map_populates_rows(self, panel):
        panel.set_face_map(_map(("a", 5, "M", 30, 0), ("b", 3, "F", 25, 10)))
        assert panel._model.rowCount() == 2  # noqa: SLF001
        assert set(_row_ids(panel)) == {"a", "b"}

    def test_columns_show_demographics(self, panel):
        panel.set_face_map(_map(("a", 5, "M", 30, 0)))
        assert panel._model.item(0, _C_SEX).text() == "M"  # noqa: SLF001
        assert panel._model.item(0, _C_APP).text() == "5"  # noqa: SLF001

    def test_rebuild_replaces(self, panel):
        panel.set_face_map(_map(("a", 1, None, None, 0)))
        panel.set_face_map(_map(("c", 1, None, None, 0)))
        assert _row_ids(panel) == ["c"]


class TestSorting:
    def test_sort_by_appearances_numeric(self, panel):
        # 9 must sort below 100 (numeric, not lexical "100" < "9").
        panel.set_face_map(
            _map(("a", 9, None, None, 0), ("b", 100, None, None, 0))
        )
        panel._table.sortByColumn(_C_APP, Qt.SortOrder.AscendingOrder)  # noqa: SLF001
        assert _row_ids(panel) == ["a", "b"]
        panel._table.sortByColumn(_C_APP, Qt.SortOrder.DescendingOrder)  # noqa: SLF001
        assert _row_ids(panel) == ["b", "a"]


class TestSelection:
    def test_selected_identities_multi(self, panel):
        panel.set_face_map(_map(("a", 1, None, None, 0), ("b", 1, None, None, 0)))
        panel._table.selectRow(0)  # noqa: SLF001
        panel._table.selectionModel().select(  # noqa: SLF001
            panel._model.index(1, 0),  # noqa: SLF001
            panel._table.selectionModel().SelectionFlag.Select  # noqa: SLF001
            | panel._table.selectionModel().SelectionFlag.Rows,  # noqa: SLF001
        )
        assert set(panel.selected_identities()) == {"a", "b"}

    def test_select_identity_programmatic(self, panel):
        panel.set_face_map(_map(("a", 1, None, None, 0), ("b", 1, None, None, 0)))
        panel.select_identity("b")
        assert panel.selected_identities() == ["b"]


class TestNavigate:
    def test_row_click_navigates_to_first_frame(self, panel, qtbot):
        panel.set_face_map(_map(("a", 1, None, None, 77)))
        with qtbot.waitSignal(panel.navigateRequested) as blocker:
            panel._on_row_clicked(panel._model.index(0, 0))  # noqa: SLF001
        assert blocker.args == [77]


class TestExclude:
    def test_exclude_emits_selected_ids(self, panel, qtbot):
        panel.set_face_map(_map(("a", 1, None, None, 0), ("b", 1, None, None, 0)))
        panel._table.selectRow(0)  # noqa: SLF001
        with qtbot.waitSignal(panel.deleteIdentitiesRequested) as blocker:
            panel._exclude_btn.click()  # noqa: SLF001
        assert blocker.args[0] == ["a"]


class TestAnalyzeControls:
    def test_analyze_emits_stride(self, panel, qtbot):
        panel._stride.setValue(20)  # noqa: SLF001
        with qtbot.waitSignal(panel.analyzeRequested) as blocker:
            panel._analyze_btn.click()  # noqa: SLF001
        assert blocker.args == [20]

    def test_analyzing_toggles_cancel(self, panel, qtbot):
        panel.set_analyzing(True)
        assert panel._analyze_btn.text() == "Cancel"  # noqa: SLF001
        with qtbot.waitSignal(panel.cancelRequested):
            panel._analyze_btn.click()  # noqa: SLF001

    def test_progress_updates(self, panel):
        panel.set_analyzing(True)
        panel.set_progress(3, 10)
        assert panel._progress.maximum() == 10  # noqa: SLF001
        assert panel._progress.value() == 3  # noqa: SLF001

    def test_settings_accessors(self, panel):
        assert panel.stride() == 15
        assert panel.workers() == 4
        assert panel.preview_enabled() is True
        assert panel.detect_demographics() is False
