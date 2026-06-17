"""Tests for FaceMapController — the face-mapping coordination logic."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sinner2.gui.face_detection_probe import FaceDetectionSink
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
    panel.precompute_geometry.return_value = True
    player = MagicMock()
    player.face_map.return_value = FaceMap.empty()
    sink = FaceDetectionSink()  # real sink: tests publish() raw faces into it
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
    c._mode_active = True  # mode ON so routing is pushed to the (mock) player
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
        assert req.compute_geometry is True  # precompute checkbox on

    def test_precompute_unchecked_skips_geometry(self, ctrl):
        ctrl._panel.precompute_geometry.return_value = False
        fired = []
        ctrl._requestAnalysis.connect(fired.append)
        ctrl._on_analyze_requested(15)
        assert fired[0].compute_geometry is False  # catalog only, no phase 2

    def test_request_carries_detector(self, ctrl):
        from sinner2.pipeline.detectors import DetectorModel
        ctrl._detector_choice = lambda: DetectorModel.YOLOFACE  # noqa: SLF001
        fired = []
        ctrl._requestAnalysis.connect(fired.append)
        ctrl._on_analyze_requested(15)
        assert fired[0].detector is DetectorModel.YOLOFACE  # threaded to the scan

    def test_analyze_emits_analyzing_active(self, ctrl):
        states = []
        ctrl.analyzingChanged.connect(states.append)
        ctrl._on_analyze_requested(15)
        ctrl._on_analysis_finished(FaceMap.empty(), None, 0, 0)
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

    def test_scan_signature_includes_clustering_params(self, ctrl):
        # The fingerprint must change when ANYTHING that shapes the embeddings
        # changes — resuming across such a change would seed the old catalog
        # with faces from a different pipeline and corrupt the identities.
        from sinner2.pipeline.detectors import DetectorModel

        base = ctrl._scan_signature(15, None)
        assert ctrl._scan_signature(15, None) == base  # stable when unchanged

        ctrl._detector_choice = lambda: DetectorModel.YOLOFACE  # noqa: SLF001
        assert ctrl._scan_signature(15, None) != base  # different detector
        ctrl._detector_choice = lambda: None  # noqa: SLF001

        ctrl._detection_size = lambda: 320  # noqa: SLF001
        assert ctrl._scan_signature(15, None) != base  # different det size
        ctrl._detection_size = lambda: 640  # noqa: SLF001

        ctrl._panel.detect_demographics.return_value = True  # full pack on
        assert ctrl._scan_signature(15, None) != base  # different embed path
        ctrl._panel.detect_demographics.return_value = False

        ctrl._catalog = ctrl._catalog.with_threshold(0.8)
        assert ctrl._scan_signature(15, None) != base  # different threshold


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
        ctrl._on_analysis_finished(fm, None, 5, 5)
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
        ctrl._catalog = FaceMap(
            identities=(_ident("a", [1, 0]), _ident("b", [0, 1]), _ident("c", [0, 0, 1]))
        )
        ctrl._on_delete_identities(["a", "c"])  # exclude multiple
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert [i.id for i in applied.identities] == ["b"]

    def test_deleting_last_identity_announces_routing_off(self, ctrl):
        # Removing the final face empties the map → routing force-disables; the
        # user must be told (the source picker silently unlocks otherwise).
        msgs: list[str] = []
        ctrl._status = lambda m, *a: msgs.append(m)
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0]),))
        ctrl._on_delete_identities(["a"])
        assert ctrl._catalog.is_empty()
        assert any("routing off" in m for m in msgs)

    def test_deleting_non_last_identity_is_silent(self, ctrl):
        # Removing one of several faces does NOT empty the map → no message.
        msgs: list[str] = []
        ctrl._status = lambda m, *a: msgs.append(m)
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0]), _ident("b", [0, 1])))
        ctrl._on_delete_identities(["a"])
        assert not msgs

    def test_deleting_last_identity_silent_when_routing_off(self, ctrl):
        # Routing already off → emptying the map flips nothing visible, so no
        # message (only an on→off routing flip is worth announcing).
        msgs: list[str] = []
        ctrl._status = lambda m, *a: msgs.append(m)
        ctrl._mode_active = False
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0]),))
        ctrl._on_delete_identities(["a"])
        assert not msgs

    def test_merge_identities(self, ctrl):
        ctrl._catalog = FaceMap(identities=(  # noqa: SLF001
            _ident("a", [1, 0], occurrences=4),
            _ident("b", [0.9, 0.1], occurrences=1, source_path="/s.png"),
            _ident("c", [0, 1], occurrences=2),
        ))
        ctrl._on_merge_identities(["a", "b"])  # noqa: SLF001
        # b folded into a → catalog has a + c; a kept b's source + summed occ.
        assert [i.id for i in ctrl._catalog.identities] == ["a", "c"]  # noqa: SLF001
        a = ctrl._catalog.identities[0]  # noqa: SLF001
        assert a.occurrences == 5 and a.source_path == "/s.png"
        ctrl._panel.select_identity.assert_called_with("a")  # noqa: SLF001

    def test_merge_needs_two_is_noop(self, ctrl):
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0]),))  # noqa: SLF001
        ctrl._player.set_face_map.reset_mock()  # noqa: SLF001
        ctrl._on_merge_identities(["a"])  # noqa: SLF001 — <2 valid
        ctrl._player.set_face_map.assert_not_called()  # noqa: SLF001

    def test_assign_one_source_to_many_ids(self, ctrl):
        # One source onto a multi-row selection (the embedded-sources gesture).
        ctrl._catalog = FaceMap(
            identities=(_ident("a", [1, 0]), _ident("b", [0, 1]))
        )
        assert ctrl.assign_source(["a", "b"], Path("/src/alice.png")) is True
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert applied.identities[0].source_path == str(Path("/src/alice.png"))
        assert applied.identities[1].source_path == str(Path("/src/alice.png"))

    def test_assign_source_empty_ids_is_noop(self, ctrl):
        assert ctrl.assign_source([], Path("/s.png")) is False
        ctrl._player.set_face_map.assert_not_called()

    def test_assign_source_does_not_re_extract_targets(self, ctrl, monkeypatch):
        # Perf guard: assigning must NOT re-extract target thumbnails (the slow
        # per-identity video decode that froze the UI) — only update the source.
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0]),))
        extracts = []
        monkeypatch.setattr(
            ctrl, "_extract_target", lambda t, i: extracts.append(i)  # noqa: SLF001
        )
        assert ctrl.assign_source(["a"], Path("/src/x.png")) is True
        assert extracts == []  # no video re-extraction
        ctrl._panel.set_source_thumbnail.assert_called()  # source cell updated


class TestReset:
    def test_reset_clears_catalog_and_progress(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_store import (
            progress_path,
            save_progress,
        )

        store = tmp_path / "face_maps"
        target = Path("/v/clip.mp4")
        save_face_map(
            face_map_path(target, store),
            FaceMap(identities=(_ident("a", [1, 0], source_path="/s.png"),)),
        )
        save_progress(progress_path(target, store), "15|[]", 3, 10)
        ctrl._resuming = True

        ctrl.reset_catalog()

        applied = ctrl._player.set_face_map.call_args.args[0]
        assert applied.is_empty()
        assert not face_map_path(target, store).exists()   # catalog gone
        assert not progress_path(target, store).exists()   # progress gone
        assert ctrl._resuming is False

    def test_reset_without_target_is_safe(self, qtbot, tmp_path):
        player = MagicMock()
        player.face_map.return_value = FaceMap.empty()
        c = FaceMapController(
            panel=MagicMock(), player=player,
            detection_sink=SimpleNamespace(_latest=None),
            store_dir=tmp_path, target_path=lambda: None,
            providers=lambda: None, detection_size=lambda: 640,
            current_frame=lambda: 0,
        )
        try:
            c.reset_catalog()  # no target → no sidecar work, no crash
            applied = player.set_face_map.call_args.args[0]
            assert applied.is_empty()
        finally:
            c.shutdown()

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
        ctrl._catalog = FaceMap(
            identities=(_ident("a", [1, 0, 0]),), threshold=0.5
        )
        face = SimpleNamespace(
            normed_embedding=normalize([1, 0, 0]), bbox=(0, 0, 10, 10)
        )
        ctrl._sink.publish([face], 100, 100)
        ctrl.on_face_clicked((1, 1, 9, 9))
        ctrl._panel.select_identity.assert_called_once_with("a")
        ctrl._player.set_face_map.assert_not_called()  # selection, not capture

    def test_click_captures_new_identity(self, ctrl):
        ctrl._catalog = FaceMap(
            identities=(_ident("a", [1, 0, 0]),), threshold=0.5
        )
        stranger = SimpleNamespace(
            normed_embedding=normalize([0, 0, 1]), bbox=(0, 0, 10, 10)
        )
        ctrl._sink.publish([stranger], 100, 100)
        ctrl.on_face_clicked((1, 1, 9, 9))
        applied = ctrl._player.set_face_map.call_args.args[0]
        assert len(applied.identities) == 2  # captured the stranger
        ctrl._panel.select_identity.assert_called_once()

    def test_click_without_embedding_is_skipped(self, ctrl):
        face = SimpleNamespace(normed_embedding=None, bbox=(0, 0, 10, 10))
        ctrl._sink.publish([face], 100, 100)
        ctrl.on_face_clicked((1, 1, 9, 9))
        ctrl._player.set_face_map.assert_not_called()

    def test_stale_click_is_rejected(self, ctrl):
        # The sink advanced to frame 7, but the boxes the user clicked were drawn
        # at frame 5 → reject the pick (don't capture a face off an unseen frame).
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0, 0]),), threshold=0.5)
        stranger = SimpleNamespace(
            normed_embedding=normalize([0, 0, 1]), bbox=(0, 0, 10, 10)
        )
        ctrl._sink.publish([stranger], 100, 100, frame_index=7)
        ctrl.on_face_clicked((1, 1, 9, 9), expected_frame=5)
        ctrl._player.set_face_map.assert_not_called()  # stale → no capture
        ctrl._panel.select_identity.assert_not_called()

    def test_fresh_click_with_matching_frame_proceeds(self, ctrl):
        # Sink frame == the frame whose boxes were drawn → the pick proceeds.
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0, 0]),), threshold=0.5)
        face = SimpleNamespace(
            normed_embedding=normalize([1, 0, 0]), bbox=(0, 0, 10, 10)
        )
        ctrl._sink.publish([face], 100, 100, frame_index=5)
        ctrl.on_face_clicked((1, 1, 9, 9), expected_frame=5)
        ctrl._panel.select_identity.assert_called_once_with("a")

    def test_untagged_sink_is_not_gated(self, ctrl):
        # No frame index on the sink (e.g. the probe path) → freshness can't be
        # judged, so the pick proceeds rather than being silently dropped.
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0, 0]),), threshold=0.5)
        face = SimpleNamespace(
            normed_embedding=normalize([1, 0, 0]), bbox=(0, 0, 10, 10)
        )
        ctrl._sink.publish([face], 100, 100)  # frame_index defaults None
        ctrl.on_face_clicked((1, 1, 9, 9), expected_frame=5)
        ctrl._panel.select_identity.assert_called_once_with("a")


class TestSelectedFaceBbox:
    def test_returns_matching_face_box(self, ctrl):
        ctrl._panel.selected_identities.return_value = ["a"]
        ctrl._catalog = FaceMap(
            identities=(_ident("a", [1, 0, 0]),), threshold=0.5
        )
        match = SimpleNamespace(normed_embedding=normalize([1, 0, 0]), bbox=(1, 2, 3, 4))
        other = SimpleNamespace(normed_embedding=normalize([0, 1, 0]), bbox=(5, 6, 7, 8))
        ctrl._sink.publish([other, match], 100, 100)
        assert ctrl.selected_face_bbox() == (1.0, 2.0, 3.0, 4.0)

    def test_none_unless_exactly_one_selected(self, ctrl):
        ctrl._panel.selected_identities.return_value = []
        assert ctrl.selected_face_bbox() is None
        ctrl._panel.selected_identities.return_value = ["a", "b"]
        assert ctrl.selected_face_bbox() is None

    def test_none_when_no_face_matches(self, ctrl):
        ctrl._panel.selected_identities.return_value = ["a"]
        ctrl._catalog = FaceMap(
            identities=(_ident("a", [1, 0, 0]),), threshold=0.5
        )
        stranger = SimpleNamespace(normed_embedding=normalize([0, 0, 1]), bbox=(1, 2, 3, 4))
        ctrl._sink.publish([stranger], 100, 100)
        assert ctrl.selected_face_bbox() is None  # below threshold

    def _geom(self, *faces):
        from sinner2.pipeline.face_map_geometry import FrameGeometry
        return FrameGeometry(faces={100: tuple(faces)}, frame_count=1)

    def _gf(self, ident_id, bbox, emb=()):
        from sinner2.pipeline.face_map_geometry import GeomFace
        kps = tuple((0.0, 0.0) for _ in range(5))
        return GeomFace(ident_id, bbox, kps, embedding=tuple(emb))

    def test_highlight_comes_from_sink_not_geometry(self, ctrl):
        # The overlay draws the SINK's boxes; the highlight must come from those
        # same detections — NOT geometry@current_frame, which is a different
        # clock and floats the highlight off the boxes under playhead lag. Here
        # geometry points at a DIFFERENT box than the sink; the sink box (the one
        # actually drawn) must win.
        ctrl._panel.selected_identities.return_value = ["a"]
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0, 0]),), threshold=0.5)
        ctrl._geometry = self._geom(self._gf("a", (10.0, 20.0, 30.0, 40.0)))
        sink_face = SimpleNamespace(normed_embedding=normalize([1, 0, 0]), bbox=(1, 2, 3, 4))
        ctrl._sink.publish([sink_face], 100, 100)
        assert ctrl.selected_face_bbox() == (1.0, 2.0, 3.0, 4.0)  # the drawn box

    def test_no_highlight_when_sink_empty_even_with_geometry(self, ctrl):
        # No drawn boxes (sink empty) → no highlight, even though geometry has a
        # box for the identity — highlighting it would float over nothing.
        ctrl._panel.selected_identities.return_value = ["a"]
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0, 0]),), threshold=0.5)
        ctrl._geometry = self._geom(self._gf("a", (10.0, 20.0, 30.0, 40.0)))
        ctrl._sink.clear()
        assert ctrl.selected_face_bbox() is None


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

    def test_restore_loads_geometry_sidecar(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_geometry import (
            FrameGeometry, GeomFace, geometry_path, save_geometry,
        )

        store = tmp_path / "face_maps"
        kps = tuple((float(i), 0.0) for i in range(5))
        save_geometry(
            geometry_path(Path("/v/clip.mp4"), store),
            FrameGeometry(faces={0: (GeomFace("a", (0., 0., 4., 4.), kps),)}, frame_count=1),
        )
        ctrl.restore_for_target(Path("/v/clip.mp4"))
        assert ctrl._geometry is not None  # noqa: SLF001
        assert ctrl._geometry.face_count() == 1  # noqa: SLF001


class TestGeometryActivation:
    def _geom(self):
        from sinner2.pipeline.face_map_geometry import FrameGeometry, GeomFace

        kps = tuple((float(i), 0.0) for i in range(5))
        return FrameGeometry(
            faces={0: (GeomFace("a", (0., 0., 4., 4.), kps),)}, frame_count=1
        )

    def test_finished_persists_geometry_and_applies_when_active(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_geometry import geometry_path, load_geometry

        ctrl.set_mode_active(True)
        geom = self._geom()
        ctrl._on_analysis_finished(  # noqa: SLF001
            FaceMap(identities=(_ident("a", [1, 0]),)), geom, 1, 1
        )
        # NPZ persisted next to the catalog…
        assert load_geometry(
            geometry_path(Path("/v/clip.mp4"), tmp_path / "face_maps")
        ) is not None
        # …and applied to the live swapper (mode active).
        ctrl._player.set_geometry.assert_called_with(geom)  # noqa: SLF001

    def test_mode_toggle_applies_then_clears_geometry(self, ctrl):
        ctrl._geometry = self._geom()  # noqa: SLF001
        ctrl.set_mode_active(True)
        ctrl._player.set_geometry.assert_called_with(ctrl._geometry)  # noqa: SLF001
        ctrl.set_mode_active(False)
        ctrl._player.set_geometry.assert_called_with(None)  # noqa: SLF001

    def test_finished_reports_geometry_in_status(self, ctrl):
        # The geometry summary now goes to the STATUS BAR (no panel caption), so
        # the user can still tell build-time (0 faces) from runtime (N faces).
        seen = []
        ctrl._status = lambda msg, *_a: seen.append(msg)  # noqa: SLF001
        ctrl._on_analysis_finished(  # noqa: SLF001
            FaceMap(identities=(_ident("a", [1, 0]),)), self._geom(), 1, 1
        )
        assert "1 faces" in seen[-1] and "1 frames" in seen[-1]

    def test_finished_status_no_geometry(self, ctrl):
        seen = []
        ctrl._status = lambda msg, *_a: seen.append(msg)  # noqa: SLF001
        ctrl._on_analysis_finished(FaceMap.empty(), None, 0, 0)  # noqa: SLF001
        assert "Precompute off" in seen[-1]

    def test_finished_status_empty_geometry(self, ctrl):
        from sinner2.pipeline.face_map_geometry import FrameGeometry

        seen = []
        ctrl._status = lambda msg, *_a: seen.append(msg)  # noqa: SLF001
        ctrl._on_analysis_finished(  # noqa: SLF001
            FaceMap.empty(), FrameGeometry.empty(), 1, 1
        )
        assert "0 faces" in seen[-1]

    def test_finished_signals_produced_map(self, ctrl):
        # A fresh scan that found someone asks the GUI to turn routing ON (so the
        # map drives playback at once). Empty scan → no auto-routing.
        seen = []
        ctrl.analysisProducedMap.connect(seen.append)
        ctrl._on_analysis_finished(  # noqa: SLF001
            FaceMap(identities=(_ident("a", [1, 0]),)), None, 1, 1
        )
        assert seen[-1] is True
        ctrl._on_analysis_finished(FaceMap.empty(), None, 0, 0)  # noqa: SLF001
        assert seen[-1] is False

    def test_produced_map_emitted_after_catalog_applied(self, ctrl):
        # Ordering matters: routing-on (driven by this signal) cascades into
        # set_face_map/set_geometry, so the catalog + geometry must already be
        # applied when it fires — assert the apply happened before the emit.
        order = []
        ctrl._player.set_face_map.side_effect = (  # noqa: SLF001
            lambda *_a, **_k: order.append("apply")
        )
        ctrl.set_mode_active(True)  # so _apply pushes to the player
        ctrl.analysisProducedMap.connect(lambda _b: order.append("produced"))
        ctrl._on_analysis_finished(  # noqa: SLF001
            FaceMap(identities=(_ident("a", [1, 0]),)), None, 1, 1
        )
        assert order and order[-1] == "produced" and "apply" in order

    def test_mode_off_clears_catalog_routing(self, ctrl):
        # The bug: leaving Faces mode left the active map on the swapper, jamming
        # normal swapping. Mode off must push the EMPTY map (global source again).
        ctrl._catalog = FaceMap(  # noqa: SLF001
            identities=(_ident("a", [1, 0], source_path="/s"),)
        )
        ctrl.set_mode_active(True)
        assert ctrl._player.set_face_map.call_args.args[0].is_active()  # noqa: SLF001
        ctrl.set_mode_active(False)
        assert ctrl._player.set_face_map.call_args.args[0].is_empty()  # noqa: SLF001

    def test_mode_on_arms_map_without_assignments(self, ctrl):
        # Point 1 fix: mode on with NO source assigned still routes (armed), so
        # unmapped faces show the original instead of the single global source.
        ctrl._catalog = FaceMap(identities=(_ident("a", [1, 0]),))  # noqa: SLF001
        ctrl.set_mode_active(True)
        pushed = ctrl._player.set_face_map.call_args.args[0]  # noqa: SLF001
        assert pushed.armed is True and pushed.is_active() is True
        # …but the stored catalog stays unarmed (clean to persist / carry over).
        assert ctrl._catalog.armed is False  # noqa: SLF001

    def test_inactive_mode_pushes_neither_geometry_nor_routing(self, ctrl):
        # Mode OFF: analysis persists the catalog + geometry but pushes NOTHING
        # live (empty face map + no geometry) — so a plain swap isn't jammed.
        ctrl._mode_active = False  # noqa: SLF001 — override the mode-on fixture
        ctrl._on_analysis_finished(  # noqa: SLF001
            FaceMap(identities=(_ident("a", [1, 0], source_path="/s"),)), self._geom(), 1, 1
        )
        for c in ctrl._player.set_geometry.call_args_list:  # noqa: SLF001
            assert c.args[0] is None  # geometry never pushed
        # The face map pushed to the swapper is EMPTY (routing cleared)…
        assert ctrl._player.set_face_map.call_args.args[0].is_empty()  # noqa: SLF001
        # …but the catalog is still held for when mode turns on.
        assert ctrl._catalog.is_active()  # noqa: SLF001

    def test_reset_deletes_geometry_sidecar(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_geometry import (
            geometry_path, load_geometry, save_geometry,
        )

        ctrl.set_mode_active(True)
        gp = geometry_path(Path("/v/clip.mp4"), tmp_path / "face_maps")
        save_geometry(gp, self._geom())
        ctrl._geometry = load_geometry(gp)  # noqa: SLF001
        ctrl.reset_catalog()
        assert not gp.exists()
        ctrl._player.set_geometry.assert_called_with(None)  # noqa: SLF001


class TestUseForPlayback:
    """D6: per-target 'use the map for playback' preference + availability."""

    def test_apply_emits_availability(self, ctrl):
        seen = []
        ctrl.mapAvailabilityChanged.connect(seen.append)
        ctrl._apply(  # noqa: SLF001
            FaceMap(identities=(_ident("a", [1, 0]),)), persist=False
        )
        assert seen[-1] is True
        ctrl._apply(FaceMap.empty(), persist=False)  # noqa: SLF001
        assert seen[-1] is False

    def test_set_use_for_playback_persists(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_store import load_use_map, use_map_path

        p = use_map_path(Path("/v/clip.mp4"), tmp_path / "face_maps")
        ctrl.set_use_for_playback(True)
        assert load_use_map(p) is True
        ctrl.set_use_for_playback(False)
        assert load_use_map(p) is False

    def test_restore_emits_saved_preference(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_store import (
            face_map_path, save_face_map, save_use_map, use_map_path,
        )

        store = tmp_path / "face_maps"
        save_face_map(
            face_map_path(Path("/v/clip.mp4"), store),
            FaceMap(identities=(_ident("a", [1, 0], source_path="/s"),)),
        )
        save_use_map(use_map_path(Path("/v/clip.mp4"), store), True)
        seen = []
        ctrl.useForPlaybackRestored.connect(seen.append)
        ctrl.restore_for_target(Path("/v/clip.mp4"))
        assert seen[-1] is True

    def test_restore_off_when_no_catalog(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_store import save_use_map, use_map_path

        # A stray 'use' marker with no catalog must NOT route (nothing to map).
        save_use_map(use_map_path(Path("/v/clip.mp4"), tmp_path / "face_maps"), True)
        seen = []
        ctrl.useForPlaybackRestored.connect(seen.append)
        ctrl.restore_for_target(Path("/v/clip.mp4"))
        assert seen[-1] is False

    def test_restore_does_not_force_routing_on(self, ctrl, tmp_path):
        # Restore must honour the SAVED preference (useForPlaybackRestored), never
        # the fresh-scan auto-on path — otherwise reopening a target with the map
        # turned off would re-enable it.
        from sinner2.pipeline.face_map_store import (
            face_map_path, save_face_map, save_use_map, use_map_path,
        )

        store = tmp_path / "face_maps"
        save_face_map(
            face_map_path(Path("/v/clip.mp4"), store),
            FaceMap(identities=(_ident("a", [1, 0], source_path="/s"),)),
        )
        save_use_map(use_map_path(Path("/v/clip.mp4"), store), False)
        produced = []
        ctrl.analysisProducedMap.connect(produced.append)
        ctrl.restore_for_target(Path("/v/clip.mp4"))
        assert produced == []  # restore never fires the fresh-scan auto-on

    def test_reset_clears_preference(self, ctrl, tmp_path):
        from sinner2.pipeline.face_map_store import load_use_map, use_map_path

        p = use_map_path(Path("/v/clip.mp4"), tmp_path / "face_maps")
        ctrl.set_use_for_playback(True)
        ctrl.reset_catalog()
        assert load_use_map(p) is False


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
