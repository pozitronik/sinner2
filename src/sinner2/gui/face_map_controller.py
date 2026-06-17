"""Coordinates the face-mapping UI: the analysis job, the Faces panel, the live
player, the detection sink, and per-target sidecar persistence.

A focused collaborator (like AudioController / CacheController) so the main
window only instantiates it and forwards a couple of events (a library-source
click, an overlay face click, a target change). Everything else — running the
strided scan on a worker thread, applying the resulting catalog to the live
swapper, extracting card thumbnails, saving/restoring the sidecar — lives here.

Dependencies are injected so the handler logic is testable without Qt media,
cv2, or real models.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, Signal

from sinner2.gui.face_map_job import AnalysisRequest, FaceMapAnalysisJob
from sinner2.pipeline.face_map import FaceMap, Identity, cosine
from sinner2.pipeline.face_map_geometry import (
    FrameGeometry,
    delete_geometry,
    geometry_path,
    load_geometry,
    save_geometry,
)
from sinner2.pipeline.face_map_store import (
    face_map_path,
    load_face_map,
    load_progress,
    load_use_map,
    progress_path,
    save_face_map,
    save_progress,
    save_use_map,
    use_map_path,
)

# (identity_id) -> a target thumbnail object (a QPixmap in the app; anything in
# tests). Returns None when the crop can't be extracted.
TargetThumbFn = Callable[[Path, Identity], Any]
SourceThumbFn = Callable[[str], Any]


class FaceMapController(QObject):
    """Owns the analysis job + thread and the face-map editing flow."""

    _requestAnalysis = Signal(object)  # an AnalysisRequest
    analyzingChanged = Signal(bool)    # the GUI locks editing while a scan runs
    mapAvailabilityChanged = Signal(bool)  # a usable catalog exists (non-empty)
    useForPlaybackRestored = Signal(bool)  # per-target "use the map" pref on load
    analysisProducedMap = Signal(bool)  # a FRESH scan built a catalog (turn routing on)

    def __init__(
        self,
        *,
        panel: Any,
        player: Any,
        detection_sink: Any,
        store_dir: Path,
        target_path: Callable[[], Path | None],
        providers: Callable[[], list[str] | None],
        detection_size: Callable[[], int],
        current_frame: Callable[[], int],
        sections: Callable[[], Any] | None = None,
        detector: Callable[[], Any] | None = None,
        landmark_refine: Callable[[], bool] | None = None,
        landmark_min_score: Callable[[], float] | None = None,
        show_preview: Callable[[Any], None] | None = None,
        set_position: Callable[[int], None] | None = None,
        navigate: Callable[[int], None] | None = None,
        status: Callable[[str, int], None] | None = None,
        extract_target_thumb: TargetThumbFn | None = None,
        load_source_thumb: SourceThumbFn | None = None,
        job: FaceMapAnalysisJob | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel = panel
        self._player = player
        self._sink = detection_sink
        self._store_dir = store_dir
        self._target_path = target_path
        self._providers = providers
        self._detection_size = detection_size
        self._current_frame = current_frame
        self._sections = sections or (lambda: None)
        self._detector_choice = detector or (lambda: None)
        self._landmark_refine = landmark_refine or (lambda: False)
        self._landmark_min_score = landmark_min_score or (lambda: 0.5)
        self._show_preview = show_preview
        self._set_position = set_position
        self._navigate = navigate
        self._status = status or (lambda _m, _ms=0: None)
        self._extract_target = extract_target_thumb or _default_target_thumb
        self._load_source = load_source_thumb or _default_source_thumb
        self._resuming = False
        self._signature = ""
        # Authoritative catalog (held here, not on the player) + the precomputed
        # geometry + whether face-mapping MODE is on. The catalog routing AND the
        # geometry are pushed to the live swapper ONLY while mode is active; mode
        # off clears them so the swapper returns to the single global source
        # (otherwise an analyzed-but-unassigned map would jam it into multi mode).
        self._catalog: FaceMap = FaceMap.empty()
        self._geometry: FrameGeometry | None = None
        self._mode_active = False

        self._job = job or FaceMapAnalysisJob()
        self._thread = QThread(self)
        self._job.moveToThread(self._thread)
        self._thread.start()
        self._requestAnalysis.connect(
            self._job.run, Qt.ConnectionType.QueuedConnection
        )
        self._job.progress.connect(
            self._panel.set_progress, Qt.ConnectionType.QueuedConnection
        )
        self._job.finished.connect(
            self._on_analysis_finished, Qt.ConnectionType.QueuedConnection
        )
        self._job.geometryStarted.connect(
            self._on_geometry_started, Qt.ConnectionType.QueuedConnection
        )
        self._job.failed.connect(
            self._on_analysis_failed, Qt.ConnectionType.QueuedConnection
        )
        self._job.preview.connect(
            self._on_preview, Qt.ConnectionType.QueuedConnection
        )
        self._job.position.connect(
            self._on_position, Qt.ConnectionType.QueuedConnection
        )

        panel.analyzeRequested.connect(self._on_analyze_requested)
        # DIRECT connection: cancel() only sets a thread-safe Event, and the job
        # thread is busy in the scan loop (its event queue can't run a queued
        # slot until run() returns) — a queued connection would never cancel.
        panel.cancelRequested.connect(
            self._job.cancel, Qt.ConnectionType.DirectConnection
        )
        panel.deleteIdentitiesRequested.connect(self._on_delete_identities)
        panel.mergeIdentitiesRequested.connect(self._on_merge_identities)
        panel.navigateRequested.connect(self._on_navigate)

    # ---- Analysis ----

    def _scan_signature(self, stride: int, sections: Any) -> str:
        """A fingerprint of everything that affects clustering, so a resume only
        continues a scan of the SAME shape. Crucially includes the detector /
        detection size / demographics(full-pack) / threshold — resuming after
        changing any of those would seed the old catalog with embeddings from a
        different pipeline and silently corrupt the identities."""
        pairs = (
            sections.to_pairs()
            if sections is not None and not sections.is_empty() else []
        )
        detector = self._detector_choice()
        det_tok = getattr(detector, "value", detector)
        fast = not self._panel.detect_demographics()
        return (
            f"{int(stride)}|{det_tok}|{int(self._detection_size())}|"
            f"{int(fast)}|{float(self._catalog.threshold):.4f}|{pairs}"
        )

    def _on_analyze_requested(self, stride: int) -> None:
        target = self._target_path()
        if target is None:
            self._status("Load a target first.", 3000)
            self._panel.set_analyzing(False)
            return
        sections = self._sections()
        self._signature = self._scan_signature(stride, sections)
        # Resume an interrupted scan of the SAME shape (stride + sections) from
        # where it stopped, seeding with the current catalog so nothing's lost.
        prog = load_progress(progress_path(target, self._store_dir))
        self._resuming = bool(
            prog
            and prog.get("signature") == self._signature
            and 0 < int(prog.get("scanned", 0)) < int(prog.get("total", 0))
        )
        start_index = int(prog["scanned"]) if (self._resuming and prog) else 0
        if self._resuming:
            self._status(f"Resuming scan from {start_index}…", 3000)

        self._panel.set_analyzing(True)
        self.analyzingChanged.emit(True)
        self._requestAnalysis.emit(AnalysisRequest(
            target_path=str(target),
            stride=int(stride),
            threshold=self._catalog.threshold,
            providers=self._providers(),
            detection_size=self._detection_size(),
            detector=self._detector_choice(),
            sections=sections,
            preview=bool(
                self._show_preview is not None and self._panel.preview_enabled()
            ),
            workers=int(self._panel.workers()),
            fast=bool(not self._panel.detect_demographics()),
            start_index=start_index,
            initial=self._catalog if self._resuming else None,
            compute_geometry=bool(self._panel.precompute_geometry()),
            landmark_refine=bool(self._landmark_refine()),
            landmark_min_score=float(self._landmark_min_score()),
            bake_angle=bool(self._panel.bake_angle()),
        ))

    def _on_geometry_started(self) -> None:
        self._status("Building per-frame face map…", 0)

    def _on_analysis_finished(
        self, catalog: FaceMap, geometry: FrameGeometry | None,
        scanned: int, total: int,
    ) -> None:
        self._panel.set_analyzing(False)
        self.analyzingChanged.emit(False)
        # A resume seeded the existing catalog, so its assignments are already
        # in the result. A fresh scan rebuilt it — carry assignments over by
        # matching the new centroids against the old.
        result = (
            catalog if self._resuming
            else _carry_over_assignments(self._catalog, catalog)
        )
        self._apply(result, persist=True)
        self._set_geometry(geometry, persist=True)
        target = self._target_path()
        if target is not None:
            save_progress(
                progress_path(target, self._store_dir),
                self._signature, scanned, total,
            )
        # Geometry summary goes to the STATUS BAR only (no persistent panel
        # caption): did Precompute build a usable detection-free table?
        if geometry is None:
            geo = " · no per-frame map (Precompute off)"
        elif geometry.is_empty():
            geo = " · per-frame map: 0 faces (live detection)"
        else:
            kind = "refined" if geometry.refined else "raw kps"
            geo = (
                f" · per-frame map: {geometry.face_count()} faces / "
                f"{len(geometry.faces)} frames ({kind})"
            )
        tail = " (cancelled — Analyze again to resume)" if scanned < total else ""
        self._status(f"Found {len(result.identities)} face(s){geo}{tail}.", 8000)
        # A fresh scan that found anyone turns routing ON (the "Use face map"
        # switch) so the map drives playback immediately — emitted LAST, after the
        # catalog + geometry are applied, so the resulting set_mode_active cascade
        # (set_face_map / set_geometry) reprocesses the current frame with the new
        # map in place. Target-restore takes the useForPlaybackRestored path
        # instead, so this only fires on a real analysis.
        self.analysisProducedMap.emit(not result.is_empty())

    def _on_analysis_failed(self, message: str) -> None:
        self._panel.set_analyzing(False)
        self.analyzingChanged.emit(False)
        self._status(f"Face analysis failed: {message}", 6000)

    def _on_preview(self, frame: Any) -> None:
        if self._show_preview is not None:
            self._show_preview(frame)

    def _on_position(self, frame_idx: int) -> None:
        if self._set_position is not None:
            self._set_position(int(frame_idx))

    def _on_navigate(self, frame_idx: int) -> None:
        if self._navigate is not None and frame_idx >= 0:
            self._navigate(int(frame_idx))

    # ---- Edits (called by the panel / main window) ----

    def _on_delete_identities(self, identity_ids: list) -> None:
        face_map = self._catalog
        for ident_id in identity_ids:
            face_map = face_map.without_identity(str(ident_id))
        emptied = face_map.is_empty() and not self._catalog.is_empty()
        self._apply(face_map, persist=True)
        if emptied and self._mode_active:
            # The last face is gone while routing was ON, so the map can no
            # longer route: the GUI forces single-source and unlocks the source
            # picker. Say why — the editor stays open, so a silent flip looks
            # like a glitch. (No-op when routing was already off.)
            self._status(
                "Last face removed — face-map routing off; using the single source.",
                5000,
            )

    def _on_merge_identities(self, identity_ids: list) -> None:
        """Fold the selected identities into one (the fragmentation fix). A pure
        catalog edit — the baked geometry re-routes the absorbed fragments to the
        survivor by embedding, so no re-precompute is needed."""
        ids = [str(i) for i in identity_ids]
        merged = self._catalog.merge(ids)
        if merged == self._catalog:
            return  # <2 valid → nothing to do
        self._apply(merged, persist=True)
        self._panel.select_identity(ids[0])  # keep the survivor selected
        self._status(f"Merged {len(ids)} faces into one.", 4000)

    def assign_source(self, identity_ids: list, source_path: Path) -> bool:
        """Assign ``source_path`` to each id in ``identity_ids`` (one source for
        many). FAST path: only the changed Source cells repaint — NOT a full
        _apply, whose _load_thumbnails re-extracts a video frame per identity and
        froze the UI for seconds on every click. False when the list is empty."""
        ids = [str(i) for i in identity_ids]
        if not ids:
            return False
        face_map = self._catalog
        for ident_id in ids:
            face_map = face_map.assign_source(ident_id, str(source_path))
        self._catalog = face_map
        # Update the one source thumbnail + the changed cells first so the table
        # reacts instantly; routing (queued, non-blocking) + persist follow.
        self._panel.note_face_map(face_map)
        thumb = self._load_source(str(source_path))
        name = Path(source_path).name
        for ident_id in ids:
            self._panel.set_source_thumbnail(ident_id, thumb, name)
        self._sync_face_map()
        target = self._target_path()
        if target is not None:
            self._save(target, face_map)
        self._status(f"Source assigned to {len(ids)} face(s).", 2000)
        return True

    def reset_catalog(self) -> None:
        """Clear the catalog, the per-frame geometry, AND the saved scan progress
        so the next Analyze starts fresh from frame 0 (live routing clears too)."""
        self._apply(FaceMap.empty(), persist=True)
        self._set_geometry(None, persist=True)
        target = self._target_path()
        if target is not None:
            progress_path(target, self._store_dir).unlink(missing_ok=True)
            save_use_map(use_map_path(target, self._store_dir), False)  # clear pref
        self._resuming = False
        self._status("Catalog cleared — Analyze starts fresh.", 3000)

    def _set_geometry(self, geometry: FrameGeometry | None, *, persist: bool) -> None:
        """Hold the per-frame geometry, persist its NPZ sidecar (or delete it when
        empty), and apply it to the live swapper iff mapping mode is active."""
        self._geometry = geometry
        if persist:
            target = self._target_path()
            if target is not None:
                path = geometry_path(target, self._store_dir)
                if geometry is None or geometry.is_empty():
                    delete_geometry(path)
                else:
                    try:
                        save_geometry(path, geometry)
                    except OSError as exc:
                        self._status(f"could not save geometry: {exc}", 4000)
        if self._mode_active:
            self._player.set_geometry(geometry)

    def set_mode_active(self, on: bool) -> None:
        """Face-mapping MODE on/off (the Faces toggle). On → push the catalog
        routing + the precomputed geometry to the live swapper. Off → clear BOTH
        so the swapper returns to the single global source + live detection (an
        analyzed map must not jam normal swapping once you leave the mode)."""
        self._mode_active = bool(on)
        self._sync_face_map()
        self._player.set_geometry(self._geometry if on else None)

    def on_face_clicked(
        self,
        bbox: tuple[float, float, float, float],
        expected_frame: int | None = None,
    ) -> None:
        """A face box was clicked on the preview: select its catalogued identity,
        or capture a new one (a face the scan missed) at the current frame.

        ``expected_frame`` is the frame whose boxes the user is looking at (what
        the overlay last drew). If the sink has since advanced to a DIFFERENT
        frame, the click is rejected as stale — capturing then would grab a face
        from a frame that isn't on screen (the click vs sink cross-clock)."""
        latest = self._sink.latest_raw()
        if latest is None:
            return
        faces, _w, _h, frame_index = latest
        if (
            expected_frame is not None
            and frame_index is not None
            and frame_index != expected_frame
        ):
            return  # stale snapshot — the sink advanced past the displayed boxes
        face = self._nearest_face(bbox, faces)
        if face is None:
            return
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            self._status("That detector has no embeddings (use buffalo_l).", 4000)
            return
        face_map = self._catalog
        match = face_map.best_match(embedding)
        if match is not None:
            self._panel.select_identity(match.id)
            return
        new = Identity.new(embedding)
        face_map = face_map.with_identity(new).with_reference(
            new.id, self._current_frame(), tuple(float(v) for v in bbox)  # type: ignore[arg-type]
        )
        self._apply(face_map, persist=True)
        self._panel.select_identity(new.id)
        self._status("New face captured — assign a source.", 3000)

    # ---- Target lifecycle ----

    def restore_for_target(self, target: Path) -> None:
        """Load the sidecar catalog for ``target`` (empty if none) and apply it
        without re-saving."""
        loaded = load_face_map(face_map_path(target, self._store_dir))
        self._apply(loaded or FaceMap.empty(), persist=False)
        self._set_geometry(
            load_geometry(geometry_path(target, self._store_dir)), persist=False
        )
        # Restore the per-target "use the map for playback" preference (only
        # meaningful when a catalog exists).
        use = bool(loaded) and load_use_map(use_map_path(target, self._store_dir))
        self.useForPlaybackRestored.emit(use)

    # ---- Internals ----

    def _apply(self, face_map: FaceMap, *, persist: bool) -> None:
        self._catalog = face_map
        self._sync_face_map()
        self._panel.set_face_map(face_map)
        self._load_thumbnails(face_map)
        if persist:
            target = self._target_path()
            if target is not None:
                self._save(target, face_map)
        # A usable catalog exists once it has any identity → the "Use face map"
        # control can offer to route playback through it.
        self.mapAvailabilityChanged.emit(not face_map.is_empty())

    def set_use_for_playback(self, on: bool) -> None:
        """Persist the per-target 'route playback through this map' preference
        (independent of the editor panel). The caller drives routing; this only
        remembers the choice so reopening the target restores it."""
        target = self._target_path()
        if target is not None:
            try:
                save_use_map(use_map_path(target, self._store_dir), bool(on))
            except OSError as exc:
                self._status(f"could not save map preference: {exc}", 4000)

    def _sync_face_map(self) -> None:
        """Push the catalog routing to the live swapper only while mode is on;
        off → the empty map, so the swapper swaps with the single global source.

        While on, the pushed copy is ARMED so routing engages even before the
        first assignment — unmapped faces then show the original instead of the
        global source. ``self._catalog`` itself stays unarmed (clean to persist
        and carry over)."""
        self._player.set_face_map(
            self._catalog.with_armed(True) if self._mode_active else FaceMap.empty()
        )

    def _load_thumbnails(self, face_map: FaceMap) -> None:
        target = self._target_path()
        for ident in face_map.identities:
            if target is not None and ident.ref_bbox is not None:
                thumb = self._extract_target(target, ident)
                if thumb is not None:
                    self._panel.set_target_thumbnail(ident.id, thumb)
            if ident.source_path:
                src = self._load_source(ident.source_path)
                self._panel.set_source_thumbnail(
                    ident.id, src, Path(ident.source_path).name
                )

    def _save(self, target: Path, face_map: FaceMap) -> None:
        path = face_map_path(target, self._store_dir)
        if face_map.is_empty():
            from sinner2.pipeline.face_map_store import delete_face_map

            delete_face_map(path)
        else:
            try:
                save_face_map(path, face_map)
            except OSError as exc:
                self._status(f"could not save face map: {exc}", 4000)

    def selected_face_bbox(self) -> tuple[float, float, float, float] | None:
        """The box of the SINGLE selected identity to highlight on the overlay —
        taken from the SAME published detections the overlay drew its boxes from
        (the detection sink), matched cosine-nearest to the identity's centroid.
        The highlight then always lands on a real, drawn box. Matching the
        precomputed geometry at the CURRENT frame instead floated the highlight
        off the boxes whenever the displayed/published frame lagged the playhead
        (load / seek-settle) — the sink is the published frame, geometry@current
        is a different clock. None when 0/many rows are selected or the identity
        isn't in this frame."""
        ids = self._panel.selected_identities()
        if len(ids) != 1:
            return None
        face_map = self._catalog
        ident = next((i for i in face_map.identities if i.id == ids[0]), None)
        if ident is None:
            return None
        latest = self._sink.latest_raw()  # raw faces (with embeddings) + index
        if latest is None:
            return None
        faces, _w, _h, _idx = latest
        best: Any = None
        best_sim = -1.0
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            sim = cosine(ident.centroid, emb)
            if sim > best_sim:
                best_sim, best = sim, face
        if best is None or best_sim < face_map.threshold:
            return None
        bbox = getattr(best, "bbox", None)
        if bbox is None:
            return None
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    def _nearest_face(
        self, bbox: tuple[float, float, float, float], faces: list[Any]
    ) -> Any:
        """The ``faces`` entry whose box centre is nearest the clicked bbox centre
        (the raw faces carry embeddings; the overlay only sends the drawable
        bbox). Takes the faces snapshot so the caller validates freshness once."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        best: Any = None
        best_d = float("inf")
        for face in faces:
            fb = getattr(face, "bbox", None)
            if fb is None:
                continue
            fcx = (float(fb[0]) + float(fb[2])) / 2.0
            fcy = (float(fb[1]) + float(fb[3])) / 2.0
            d = (fcx - cx) ** 2 + (fcy - cy) ** 2
            if d < best_d:
                best_d, best = d, face
        return best

    def shutdown(self) -> None:
        self._job.cancel()
        self._thread.quit()
        self._thread.wait(2000)


def _carry_over_assignments(old: FaceMap, fresh: FaceMap) -> FaceMap:
    """Re-attach source assignments from a previous catalog onto a freshly
    analysed one by matching centroids (a re-scan shouldn't lose the user's
    work). Each fresh identity adopts the source of the old identity it best
    matches above the fresh threshold."""
    if old.is_empty():
        return fresh
    result = fresh
    for ident in fresh.identities:
        match = old.best_match(ident.centroid)
        if match is not None and match.source_path:
            result = result.assign_source(ident.id, match.source_path)
    return result


def _default_target_thumb(target: Path, identity: Identity) -> Any:
    """Read the identity's representative frame and crop its bbox to a QPixmap.
    None on any failure (a debug thumbnail must never break the panel)."""
    if identity.ref_frame is None or identity.ref_bbox is None:
        return None
    try:
        from sinner2.config.target import Target, TargetKind
        from sinner2.io.cv2_video_target_reader import CV2VideoTargetReader
        from sinner2.io.target_reader import ImageTargetReader

        tgt = Target(path=target)
        reader: Any
        if tgt.kind is TargetKind.IMAGE:
            reader = ImageTargetReader(tgt)
        elif tgt.kind is TargetKind.VIDEO:
            reader = CV2VideoTargetReader(tgt)
        else:
            return None
        try:
            frame = reader.read(identity.ref_frame)
        finally:
            reader.release()
        if frame is None:
            return None
        return _crop_to_pixmap(frame, identity.ref_bbox)
    except Exception:  # noqa: BLE001
        return None


def _default_source_thumb(source_path: str) -> Any:
    try:
        from sinner2.io.cv2_unicode import imread_unicode

        img = imread_unicode(Path(source_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return _crop_to_pixmap(img, (0.0, 0.0, float(w), float(h)))
    except Exception:  # noqa: BLE001
        return None


def _crop_to_pixmap(frame: Any, bbox: tuple[float, float, float, float]) -> Any:
    import numpy as np
    from PySide6.QtGui import QImage, QPixmap

    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
    ch, cw = crop.shape[:2]
    image = QImage(crop.data, cw, ch, cw * 3, QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(image)
