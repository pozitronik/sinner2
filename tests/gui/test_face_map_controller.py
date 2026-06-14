"""Tests for FaceMapController — the face-mapping coordination logic."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sinner2.gui.face_map_controller import (
    FaceMapController,
    _carry_over_assignments,
)
from sinner2.pipeline.face_map import FaceMap, Identity, normalize
from sinner2.pipeline.face_map_store import face_map_path, save_face_map


def _ident(i, vec, **kw):
    return Identity(i, normalize(vec), **kw)


@pytest.fixture
def ctrl(qtbot, tmp_path):
    panel = MagicMock()
    panel.selected_identity.return_value = None
    panel.workers.return_value = 4
    panel.preview_enabled.return_value = True
    player = MagicMock()
    player.face_map.return_value = FaceMap.empty()
    sink = SimpleNamespace(_latest=None)
    c = FaceMapController(
        panel=panel,
        player=player,
        detection_sink=sink,
        store_dir=tmp_path / "face_maps",
        target_path=lambda: Path("/v/clip.mp4"),
        providers=lambda: ["CPUExecutionProvider"],
        detection_size=lambda: 640,
        current_frame=lambda: 100,
        status=lambda *a: None,
        extract_target_thumb=lambda t, i: "THUMB",
        load_source_thumb=lambda p: "SRC",
    )
    c._panel, c._player, c._sink = panel, player, sink  # handles for assertions
    yield c
    c.shutdown()


class TestAnalyze:
    def test_no_target_aborts(self, qtbot, tmp_path):
        panel = MagicMock()
        player = MagicMock()
        player.face_map.return_value = FaceMap.empty()
        c = FaceMapController(
            panel=panel, player=player, detection_sink=SimpleNamespace(_latest=None),
            store_dir=tmp_path, target_path=lambda: None,
            providers=lambda: None, detection_size=lambda: 640,
            current_frame=lambda: 0,
        )
        try:
            c._on_analyze_requested(15)
            panel.set_analyzing.assert_called_with(False)
        finally:
            c.shutdown()

    def test_requests_analysis_with_params(self, ctrl):
        fired = []
        ctrl._requestAnalysis.connect(lambda *a: fired.append(a))
        ctrl._on_analyze_requested(20)
        ctrl._panel.set_analyzing.assert_called_with(True)
        # target, stride, threshold, providers, det_size, sections, preview, workers.
        # show_preview is None in this fixture → preview False regardless.
        assert fired == [
            (str(Path("/v/clip.mp4")), 20, 0.5, ["CPUExecutionProvider"], 640,
             None, False, 4)
        ]

    def test_analyze_emits_analyzing_active(self, ctrl):
        states = []
        ctrl.analyzingChanged.connect(states.append)
        ctrl._on_analyze_requested(15)
        ctrl._on_analysis_finished(FaceMap.empty())
        assert states == [True, False]

    def test_passes_sections_and_preview_when_enabled(self, qtbot, tmp_path):
        from sinner2.pipeline.sections import SectionSet

        panel = MagicMock()
        panel.preview_enabled.return_value = True
        panel.workers.return_value = 3
        player = MagicMock()
        player.face_map.return_value = FaceMap.empty()
        secs = SectionSet.of([(2, 5)])
        c = FaceMapController(
            panel=panel, player=player, detection_sink=SimpleNamespace(_latest=None),
            store_dir=tmp_path, target_path=lambda: Path("/v.mp4"),
            providers=lambda: None, detection_size=lambda: 640,
            current_frame=lambda: 0, sections=lambda: secs,
            show_preview=lambda _f: None,
        )
        try:
            fired = []
            c._requestAnalysis.connect(lambda *a: fired.append(a))
            c._on_analyze_requested(3)
            assert fired[0][5] == secs   # sections forwarded
            assert fired[0][6] is True   # preview requested
            assert fired[0][7] == 3      # workers forwarded
        finally:
            c.shutdown()


class TestCancel:
    def test_cancel_is_direct_and_synchronous(self, qtbot, tmp_path):
        from sinner2.gui.widgets.face_map_panel import QFaceMapPanel

        panel = QFaceMapPanel()
        qtbot.addWidget(panel)
        player = MagicMock()
        player.face_map.return_value = FaceMap.empty()
        c = FaceMapController(
            panel=panel, player=player, detection_sink=SimpleNamespace(_latest=None),
            store_dir=tmp_path, target_path=lambda: Path("/v.mp4"),
            providers=lambda: None, detection_size=lambda: 640, current_frame=lambda: 0,
        )
        try:
            c._job._cancel.clear()
            panel.cancelRequested.emit()
            # Direct connection → the Event is set on THIS thread, immediately
            # (a queued connection couldn't, since the job thread is mid-scan).
            assert c._job._cancel.is_set()
        finally:
            c.shutdown()

    def test_finished_applies_and_persists(self, ctrl, tmp_path):
        fm = FaceMap(identities=(_ident("a", [1, 0, 0]),))
        ctrl._on_analysis_finished(fm)
        ctrl._panel.set_analyzing.assert_called_with(False)
        ctrl._player.set_face_map.assert_called_once()
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert len(applied.identities) == 1
        # Sidecar written.
        assert face_map_path(Path("/v/clip.mp4"), tmp_path / "face_maps").is_file()

    def test_failed_resets_state(self, ctrl):
        ctrl._on_analysis_failed("boom")
        ctrl._panel.set_analyzing.assert_called_with(False)
        ctrl._player.set_face_map.assert_not_called()


class TestEdits:
    def test_delete_identity(self, ctrl):
        ctrl._player.face_map.return_value = FaceMap(
            identities=(_ident("a", [1, 0]), _ident("b", [0, 1]))
        )
        ctrl._on_delete_identity("a")
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert [i.id for i in applied.identities] == ["b"]

    def test_assign_source_to_selected(self, ctrl):
        ctrl._player.face_map.return_value = FaceMap(identities=(_ident("a", [1, 0]),))
        ctrl._panel.selected_identity.return_value = "a"
        assert ctrl.assign_source_to_selected(Path("/src/alice.png")) is True
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert applied.identities[0].source_path == str(Path("/src/alice.png"))

    def test_assign_without_selection_is_noop(self, ctrl):
        ctrl._panel.selected_identity.return_value = None
        assert ctrl.assign_source_to_selected(Path("/s.png")) is False
        ctrl._player.set_face_map.assert_not_called()


class TestFaceClick:
    def test_click_selects_matching_identity(self, ctrl):
        ctrl._player.face_map.return_value = FaceMap(
            identities=(_ident("a", [1, 0, 0]),), threshold=0.5
        )
        face = SimpleNamespace(
            normed_embedding=normalize([1, 0, 0]), bbox=(0, 0, 10, 10)
        )
        ctrl._sink._latest = ([face], 100, 100)
        ctrl.on_face_clicked((1, 1, 9, 9))
        ctrl._panel.select_identity.assert_called_once_with("a")
        ctrl._player.set_face_map.assert_not_called()  # selection, not capture

    def test_click_captures_new_identity(self, ctrl):
        ctrl._player.face_map.return_value = FaceMap(
            identities=(_ident("a", [1, 0, 0]),), threshold=0.5
        )
        stranger = SimpleNamespace(
            normed_embedding=normalize([0, 0, 1]), bbox=(0, 0, 10, 10)
        )
        ctrl._sink._latest = ([stranger], 100, 100)
        ctrl.on_face_clicked((1, 1, 9, 9))
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert len(applied.identities) == 2  # captured the stranger
        ctrl._panel.select_identity.assert_called_once()

    def test_click_without_embedding_is_skipped(self, ctrl):
        face = SimpleNamespace(normed_embedding=None, bbox=(0, 0, 10, 10))
        ctrl._sink._latest = ([face], 100, 100)
        ctrl.on_face_clicked((1, 1, 9, 9))
        ctrl._player.set_face_map.assert_not_called()


class TestRestore:
    def test_restores_sidecar(self, ctrl, tmp_path):
        fm = FaceMap(identities=(_ident("a", [1, 0], source_path="/s.png"),))
        save_face_map(face_map_path(Path("/v/clip.mp4"), tmp_path / "face_maps"), fm)
        ctrl.restore_for_target(Path("/v/clip.mp4"))
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert applied.identities[0].source_path == "/s.png"

    def test_restore_missing_applies_empty(self, ctrl):
        ctrl.restore_for_target(Path("/v/unseen.mp4"))
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert applied.is_empty()


class TestCarryOver:
    def test_reattaches_sources_by_centroid(self):
        old = FaceMap(
            identities=(
                _ident("old_a", [1, 0, 0], source_path="/alice.png"),
                _ident("old_b", [0, 1, 0], source_path="/bob.png"),
            ),
            threshold=0.5,
        )
        fresh = FaceMap(
            identities=(
                _ident("new_1", [0.97, 0.03, 0]),  # ~ alice
                _ident("new_2", [0.02, 0.98, 0]),  # ~ bob
                _ident("new_3", [0, 0, 1]),        # new person
            ),
            threshold=0.5,
        )
        merged = _carry_over_assignments(old, fresh)
        by_id = {i.id: i.source_path for i in merged.identities}
        assert by_id["new_1"] == "/alice.png"
        assert by_id["new_2"] == "/bob.png"
        assert by_id["new_3"] is None

    def test_empty_old_is_passthrough(self):
        fresh = FaceMap(identities=(_ident("a", [1, 0]),))
        assert _carry_over_assignments(FaceMap.empty(), fresh) == fresh
