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

from sinner2.gui.face_map_job import FaceMapAnalysisJob
from sinner2.pipeline.face_map import FaceMap, Identity
from sinner2.pipeline.face_map_store import (
    face_map_path,
    load_face_map,
    save_face_map,
)

# (identity_id) -> a target thumbnail object (a QPixmap in the app; anything in
# tests). Returns None when the crop can't be extracted.
TargetThumbFn = Callable[[Path, Identity], Any]
SourceThumbFn = Callable[[str], Any]


class FaceMapController(QObject):
    """Owns the analysis job + thread and the face-map editing flow."""

    _requestAnalysis = Signal(str, int, float, object, int)

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
        self._status = status or (lambda _m, _ms=0: None)
        self._extract_target = extract_target_thumb or _default_target_thumb
        self._load_source = load_source_thumb or _default_source_thumb

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
        self._job.failed.connect(
            self._on_analysis_failed, Qt.ConnectionType.QueuedConnection
        )

        panel.analyzeRequested.connect(self._on_analyze_requested)
        panel.cancelRequested.connect(self._job.cancel)
        panel.deleteIdentityRequested.connect(self._on_delete_identity)

    # ---- Analysis ----

    def _on_analyze_requested(self, stride: int) -> None:
        target = self._target_path()
        if target is None:
            self._status("Load a target first.", 3000)
            self._panel.set_analyzing(False)
            return
        self._panel.set_analyzing(True)
        self._requestAnalysis.emit(
            str(target),
            int(stride),
            self._player.face_map().threshold,
            self._providers(),
            self._detection_size(),
        )

    def _on_analysis_finished(self, face_map: FaceMap) -> None:
        self._panel.set_analyzing(False)
        # Keep any source assignments the user already made for surviving
        # identities — but a fresh scan rebuilds the catalog, so carry over
        # assignments by matching the new centroids against the old ones.
        merged = _carry_over_assignments(self._player.face_map(), face_map)
        self._apply(merged, persist=True)
        self._status(f"Found {len(merged.identities)} face(s).", 4000)

    def _on_analysis_failed(self, message: str) -> None:
        self._panel.set_analyzing(False)
        self._status(f"Face analysis failed: {message}", 6000)

    # ---- Edits (called by the panel / main window) ----

    def _on_delete_identity(self, identity_id: str) -> None:
        self._apply(self._player.face_map().without_identity(identity_id), persist=True)

    def assign_source_to_selected(self, source_path: Path) -> bool:
        """Assign ``source_path`` to the currently-selected identity. Returns
        False (a no-op) when nothing is selected — the main window only routes a
        library click here while the Faces tab is active."""
        selected = self._panel.selected_identity()
        if selected is None:
            return False
        self._apply(
            self._player.face_map().assign_source(selected, str(source_path)),
            persist=True,
        )
        self._status("Source assigned.", 2000)
        return True

    def on_face_clicked(self, bbox: tuple[float, float, float, float]) -> None:
        """A face box was clicked on the preview: select its catalogued identity,
        or capture a new one (a face the scan missed) at the current frame."""
        face = self._raw_face_at(bbox)
        if face is None:
            return
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            self._status("That detector has no embeddings (use buffalo_l).", 4000)
            return
        face_map = self._player.face_map()
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

    # ---- Internals ----

    def _apply(self, face_map: FaceMap, *, persist: bool) -> None:
        self._player.set_face_map(face_map)
        self._panel.set_face_map(face_map)
        self._load_thumbnails(face_map)
        if persist:
            target = self._target_path()
            if target is not None:
                self._save(target, face_map)

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

    def _raw_face_at(self, bbox: tuple[float, float, float, float]) -> Any:
        """The detection-sink face whose box centre is nearest the clicked bbox
        centre (the sink holds the raw faces WITH embeddings; the overlay only
        carries the drawable bbox)."""
        latest = self._sink._latest  # noqa: SLF001 — raw faces + size
        if latest is None:
            return None
        faces, _w, _h = latest
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
