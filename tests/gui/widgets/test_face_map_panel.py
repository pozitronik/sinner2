"""Tests for the Faces panel (sortable findings table)."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView

from sinner2.gui.widgets.face_map_panel import (
    _C_APP,
    _C_FACE,
    _C_PITCH,
    _C_ROLL,
    _C_SCORE,
    _C_SEX,
    _C_YAW,
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


def _row_of(panel, ident_id):
    return _row_ids(panel).index(ident_id)


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

    def test_score_and_pose_columns(self, panel):
        panel.set_face_map(FaceMap(identities=(
            Identity("a", normalize([1, 0]), occurrences=1,
                     det_score=0.95, roll=12.0, yaw=-5.0, pitch=3.0),
            Identity("b", normalize([0, 1]), occurrences=1),  # no score/pose
        )))
        ra = _row_of(panel, "a")
        assert panel._model.item(ra, _C_SCORE).text() == "0.95"  # noqa: SLF001
        assert panel._model.item(ra, _C_ROLL).text() == "12°"  # noqa: SLF001
        assert panel._model.item(ra, _C_YAW).text() == "-5°"  # noqa: SLF001
        assert panel._model.item(ra, _C_PITCH).text() == "3°"  # noqa: SLF001
        rb = _row_of(panel, "b")
        assert panel._model.item(rb, _C_SCORE).text() == "—"  # noqa: SLF001
        assert panel._model.item(rb, _C_ROLL).text() == "—"  # noqa: SLF001


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
    def test_selection_change_emits_signal(self, panel, qtbot):
        panel.set_face_map(_map(("a", 1, None, None, 0), ("b", 1, None, None, 0)))
        with qtbot.waitSignal(panel.selectionChanged):
            panel._table.selectRow(0)  # noqa: SLF001

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
        # A single click jumps the preview to that person's first appearance.
        panel.set_face_map(_map(("a", 1, None, None, 77)))
        with qtbot.waitSignal(panel.navigateRequested) as blocker:
            panel._on_cell_clicked(panel._model.index(0, _C_FACE))  # noqa: SLF001
        assert blocker.args == [77]


class TestExclude:
    def test_delete_shortcut_emits_selected(self, panel, qtbot):
        # The Delete key (no per-row ✕) removes the SELECTED people.
        panel.set_face_map(_map(("a", 1, None, None, 0), ("b", 1, None, None, 0)))
        panel.select_identity("b")
        with qtbot.waitSignal(panel.deleteIdentitiesRequested) as blocker:
            panel._on_delete_shortcut()  # noqa: SLF001
        assert blocker.args[0] == ["b"]

    def test_delete_shortcut_noop_without_selection(self, panel):
        panel.set_face_map(_map(("a", 1, None, None, 0)))
        panel._table.clearSelection()  # noqa: SLF001
        fired = []
        panel.deleteIdentitiesRequested.connect(fired.append)
        panel._on_delete_shortcut()  # noqa: SLF001
        assert fired == []

    def test_merge_shortcut_needs_two(self, panel, qtbot):
        panel.set_face_map(_map(("a", 1, None, None, 0), ("b", 1, None, None, 0)))
        fired = []
        panel.mergeIdentitiesRequested.connect(fired.append)
        # One selected → no merge.
        panel.select_identity("a")
        panel._on_merge_shortcut()  # noqa: SLF001
        assert fired == []
        # Two selected → merge emits both ids.
        panel._table.selectAll()  # noqa: SLF001
        panel._on_merge_shortcut()  # noqa: SLF001
        assert len(fired) == 1 and set(fired[0]) == {"a", "b"}


class TestReset:
    def test_reset_button_emits(self, panel, qtbot):
        with qtbot.waitSignal(panel.resetRequested):
            panel._reset_btn.click()  # noqa: SLF001

    def test_reset_disabled_while_analyzing(self, panel):
        panel.set_analyzing(True)
        assert panel._reset_btn.isEnabled() is False  # noqa: SLF001
        panel.set_analyzing(False)
        assert panel._reset_btn.isEnabled() is True  # noqa: SLF001


class TestColumns:
    def test_columns_are_user_resizable(self, panel):
        header = panel._table.horizontalHeader()  # noqa: SLF001
        assert (
            header.sectionResizeMode(_C_FACE)
            == QHeaderView.ResizeMode.Interactive
        )
        assert not header.stretchLastSection()


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
        assert panel.precompute_geometry() is True  # default on

    def test_precompute_toggle_and_lock(self, panel):
        panel._precompute_check.setChecked(False)  # noqa: SLF001
        assert panel.precompute_geometry() is False
        panel.set_analyzing(True)
        assert panel._precompute_check.isEnabled() is False  # noqa: SLF001
        panel.set_analyzing(False)
        assert panel._precompute_check.isEnabled() is True  # noqa: SLF001

    def test_restore_settings_seeds_without_emitting(self, panel, qtbot):
        fired = []
        panel.settingsChanged.connect(lambda: fired.append(1))
        panel.restore_settings(
            stride=7, workers=2, preview=False, demographics=True, precompute=False
        )
        assert panel.stride() == 7
        assert panel.workers() == 2
        assert panel.preview_enabled() is False
        assert panel.detect_demographics() is True
        assert panel.precompute_geometry() is False
        assert fired == []  # restore is silent — no persist echo

    def test_restore_settings_none_keeps_default(self, panel):
        panel.restore_settings(stride=None, workers=None)
        assert panel.stride() == 15  # unchanged default
        assert panel.workers() == 4

    def test_detection_group_accessors_and_defaults(self, panel):
        # D1/D5: the scan's own Detection settings, independent of the swapper.
        assert panel.detection_size() == 640
        assert panel.landmark_refine() is False
        assert panel.landmark_min_score() == 0.5
        assert panel.bake_angle() is True  # D5 default on

    def test_restore_detection_settings(self, panel, qtbot):
        fired = []
        panel.settingsChanged.connect(lambda: fired.append(1))
        panel.restore_settings(
            detection_size=960, landmark_refine=True, landmark_min_score=0.8,
            bake_angle=False,
        )
        assert panel.detection_size() == 960
        assert panel.landmark_refine() is True
        assert panel.landmark_min_score() == 0.8
        assert panel.bake_angle() is False
        assert fired == []  # silent restore

    def test_detection_controls_locked_while_analyzing(self, panel):
        panel.set_analyzing(True)
        assert panel._det_size.isEnabled() is False  # noqa: SLF001
        assert panel._refine_check.isEnabled() is False  # noqa: SLF001
        assert panel._refine_score.isEnabled() is False  # noqa: SLF001
        assert panel._bake_angle_check.isEnabled() is False  # noqa: SLF001
        panel.set_analyzing(False)
        assert panel._det_size.isEnabled() is True  # noqa: SLF001

    def test_refine_min_score_grays_with_refine(self, panel):
        # 'Refine min score' is gated by 'Refine keypoints' (it does nothing
        # otherwise) — the dependency is shown by graying.
        assert panel.landmark_refine() is False  # default off
        assert panel._refine_score.isEnabled() is False  # noqa: SLF001 — grayed
        assert panel._refine_min_label.isEnabled() is False  # noqa: SLF001
        panel._refine_check.setChecked(True)  # noqa: SLF001
        assert panel._refine_score.isEnabled() is True  # noqa: SLF001
        assert panel._refine_min_label.isEnabled() is True  # noqa: SLF001
        panel._refine_check.setChecked(False)  # noqa: SLF001
        assert panel._refine_score.isEnabled() is False  # noqa: SLF001

    def test_detector_default_and_demographics_gate(self, panel):
        from sinner2.pipeline.detectors import DetectorModel
        # Default buffalo_l → demographics available.
        assert panel.detector() is DetectorModel.BUFFALO_L
        assert panel._demographics_check.isEnabled() is True  # noqa: SLF001
        panel._demographics_check.setChecked(True)  # noqa: SLF001
        # A detection-only detector can't do gender/age → gray + clear it.
        idx = panel._detector.findData(DetectorModel.YOLOFACE.value)  # noqa: SLF001
        panel._detector.setCurrentIndex(idx)  # noqa: SLF001
        assert panel.detector() is DetectorModel.YOLOFACE
        assert panel._demographics_check.isEnabled() is False  # noqa: SLF001
        assert panel.detect_demographics() is False  # cleared

    def test_restore_detector(self, panel):
        from sinner2.pipeline.detectors import DetectorModel
        panel.restore_settings(detector=DetectorModel.SCRFD_2_5G.value)
        assert panel.detector() is DetectorModel.SCRFD_2_5G
        assert panel._demographics_check.isEnabled() is False  # noqa: SLF001

    def test_refine_dependency_survives_restore_and_analyze(self, panel):
        # Restoring refine=on enables the min score; an analyze cycle re-applies it.
        panel.restore_settings(landmark_refine=True)
        assert panel._refine_score.isEnabled() is True  # noqa: SLF001
        panel.set_analyzing(True)
        assert panel._refine_score.isEnabled() is False  # noqa: SLF001 — locked
        panel.set_analyzing(False)
        assert panel._refine_score.isEnabled() is True  # noqa: SLF001 — refine still on

    def test_user_change_emits_settings_changed(self, panel, qtbot):
        fired = []
        panel.settingsChanged.connect(lambda: fired.append(1))
        panel._stride.setValue(42)  # noqa: SLF001 — simulate a user edit
        panel._precompute_check.setChecked(False)  # noqa: SLF001
        assert len(fired) == 2


class TestUseFaceMapToggle:
    """The in-panel 'Use face map' checkbox: a SEPARATE switch (not the group
    title) synced with the settings one; enabled only once a map exists; never
    grays the scanner controls."""

    def test_default_off_and_disabled(self, panel):
        assert panel.use_face_map() is False
        assert panel._use_face_map_check.isEnabled() is False  # noqa: SLF001 — no map

    def test_available_gates_only_the_toggle(self, panel):
        panel.set_face_map_available(True)
        assert panel._use_face_map_check.isEnabled() is True  # noqa: SLF001
        # The scanner controls stay usable regardless of availability.
        assert panel._detector.isEnabled() is True  # noqa: SLF001
        assert panel._analyze_btn.isEnabled() is True  # noqa: SLF001
        panel.set_face_map_available(False)
        assert panel._use_face_map_check.isEnabled() is False  # noqa: SLF001
        assert panel._analyze_btn.isEnabled() is True  # noqa: SLF001 — still usable

    def test_user_toggle_emits(self, panel, qtbot):
        panel.set_face_map_available(True)
        with qtbot.waitSignal(panel.useFaceMapToggled) as blocker:
            panel._use_face_map_check.setChecked(True)  # noqa: SLF001
        assert blocker.args == [True]

    def test_set_use_face_map_is_silent(self, panel):
        fired = []
        panel.useFaceMapToggled.connect(fired.append)
        panel.set_use_face_map(True)
        assert panel.use_face_map() is True
        assert fired == []  # synced from the owner — no echo

    def test_show_overlay_default_on_and_gated_by_use_face_map(self, panel):
        # Defaults on (you see faces when editing); grayed unless routing is on
        # (it only controls the overlay in face-map mode).
        assert panel.show_overlay() is True
        assert panel._show_overlay_check.isEnabled() is False  # noqa: SLF001 — routing off
        panel._use_face_map_check.setChecked(True)  # noqa: SLF001
        assert panel._show_overlay_check.isEnabled() is True  # noqa: SLF001
        panel._use_face_map_check.setChecked(False)  # noqa: SLF001
        assert panel._show_overlay_check.isEnabled() is False  # noqa: SLF001

    def test_show_overlay_toggle_emits(self, panel, qtbot):
        panel._use_face_map_check.setChecked(True)  # noqa: SLF001 — enable it
        with qtbot.waitSignal(panel.showOverlayToggled) as blocker:
            panel._show_overlay_check.setChecked(False)  # noqa: SLF001
        assert blocker.args == [False]
