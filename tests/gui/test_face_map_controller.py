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
    panel.selected_identities.return_value = []
    panel.workers.return_value = 4
    panel.preview_enabled.return_value = True
    panel.detect_demographics.return_value = False  # fast by default
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
        ctrl._requestAnalysis.connect(fired.append)
        ctrl._on_analyze_requested(20)
        ctrl._panel.set_analyzing.assert_called_with(True)
        req = fired[0]
        assert req.target_path == str(Path("/v/clip.mp4"))
        assert req.stride == 20
        assert req.providers == ["CPUExecutionProvider"]
        assert req.detection_size == 640
        assert req.preview is False  # show_preview is None in this fixture
        assert req.workers == 4
        assert req.fast is True       # demographics off → fast
        assert req.start_index == 0   # no prior progress → fresh

    def test_analyze_emits_analyzing_active(self, ctrl):
        states = []
        ctrl.analyzingChanged.connect(states.append)
        ctrl._on_analyze_requested(15)
        ctrl._on_analysis_finished(FaceMap.empty(), 0, 0)
        assert states == [True, False]

    def test_passes_sections_and_preview_when_enabled(self, qtbot, tmp_path):
        from sinner2.pipeline.sections import SectionSet

        panel = MagicMock()
        panel.preview_enabled.return_value = True
        panel.workers.return_value = 3
        panel.detect_demographics.return_value = True  # full pack → fast=False
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
            c._requestAnalysis.connect(fired.append)
            c._on_analyze_requested(3)
            assert fired[0].sections == secs
            assert fired[0].preview is True
            assert fired[0].workers == 3
            assert fired[0].fast is False  # demographics on → not fast
        finally:
            c.shutdown()


class TestResumeRequest:
    def test_resumes_from_saved_progress(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_store import progress_path, save_progress

        sig = ctrl._scan_signature(15, None)
        save_progress(
            progress_path(Path("/v/clip.mp4"), tmp_path / "face_maps"), sig, 3, 10
        )
        fired = []
        ctrl._requestAnalysis.connect(fired.append)
        ctrl._on_analyze_requested(15)
        assert fired[0].start_index == 3  # resumed from where it stopped
        assert ctrl._resuming is True

    def test_fresh_when_signature_differs(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_store import progress_path, save_progress

        # Saved progress for stride 99; requesting stride 15 → can't resume.
        save_progress(
            progress_path(Path("/v/clip.mp4"), tmp_path / "face_maps"),
            ctrl._scan_signature(99, None), 3, 10,
        )
        fired = []
        ctrl._requestAnalysis.connect(fired.append)
        ctrl._on_analyze_requested(15)
        assert fired[0].start_index == 0
        assert ctrl._resuming is False


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
        ctrl._on_analysis_finished(fm, 5, 5)
        ctrl._panel.set_analyzing.assert_called_with(False)
        ctrl._player.set_face_map.assert_called_once()
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert len(applied.identities) == 1
        # Catalog + progress sidecars written.
        assert face_map_path(Path("/v/clip.mp4"), tmp_path / "face_maps").is_file()
        from sinner2.pipeline.face_map_store import load_progress, progress_path

        prog = load_progress(progress_path(Path("/v/clip.mp4"), tmp_path / "face_maps"))
        assert prog["scanned"] == 5 and prog["total"] == 5

    def test_failed_resets_state(self, ctrl):
        ctrl._on_analysis_failed("boom")
        ctrl._panel.set_analyzing.assert_called_with(False)
        ctrl._player.set_face_map.assert_not_called()


class TestEdits:
    def test_delete_identities(self, ctrl):
        ctrl._player.face_map.return_value = FaceMap(
            identities=(_ident("a", [1, 0]), _ident("b", [0, 1]), _ident("c", [0, 0, 1]))
        )
        ctrl._on_delete_identities(["a", "c"])  # exclude multiple
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert [i.id for i in applied.identities] == ["b"]

    def test_assign_one_source_to_many_selected(self, ctrl):
        ctrl._player.face_map.return_value = FaceMap(
            identities=(_ident("a", [1, 0]), _ident("b", [0, 1]))
        )
        ctrl._panel.selected_identities.return_value = ["a", "b"]
        assert ctrl.assign_source_to_selected(Path("/src/alice.png")) is True
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert applied.identities[0].source_path == str(Path("/src/alice.png"))
        assert applied.identities[1].source_path == str(Path("/src/alice.png"))

    def test_assign_without_selection_is_noop(self, ctrl):
        ctrl._panel.selected_identities.return_value = []
        assert ctrl.assign_source_to_selected(Path("/s.png")) is False
        ctrl._player.set_face_map.assert_not_called()

    def test_navigate_forwards_frame(self, qtbot, tmp_path):
        navs = []
        panel = MagicMock()
        player = MagicMock()
        player.face_map.return_value = FaceMap.empty()
        c = FaceMapController(
            panel=panel, player=player, detection_sink=SimpleNamespace(_latest=None),
            store_dir=tmp_path, target_path=lambda: Path("/v.mp4"),
            providers=lambda: None, detection_size=lambda: 640,
            current_frame=lambda: 0, navigate=navs.append,
        )
        try:
            c._on_navigate(42)
            assert navs == [42]
        finally:
            c.shutdown()

    def test_position_forwards_to_set_position(self, qtbot, tmp_path):
        positions = []
        panel = MagicMock()
        player = MagicMock()
        player.face_map.return_value = FaceMap.empty()
        c = FaceMapController(
            panel=panel, player=player, detection_sink=SimpleNamespace(_latest=None),
            store_dir=tmp_path, target_path=lambda: Path("/v.mp4"),
            providers=lambda: None, detection_size=lambda: 640,
            current_frame=lambda: 0, set_position=positions.append,
        )
        try:
            c._on_position(123)
            assert positions == [123]
        finally:
            c.shutdown()


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
