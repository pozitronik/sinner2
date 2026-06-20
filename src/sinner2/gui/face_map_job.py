"""Background job that builds a target's face-map catalog off the GUI thread.

Wraps the headless `analyze_target` scan in a QObject that lives on its own
QThread (like FaceDetectionProbe): the GUI kicks it via a queued `run` carrying
an `AnalysisRequest`, and gets `progress` / `preview` / `position` / `finished`
/ `failed` back. The reader + detector builders are injectable so tests need no
media or models.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from sinner2.config.target import Target, TargetKind
from sinner2.io.cv2_video_target_reader import CV2VideoTargetReader
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.pipeline.face_analyser import pin_shared_face_analysis
from sinner2.pipeline.face_map_analyzer import (
    DetectFn,
    analyze_target,
    precompute_geometry,
)

ReaderFactory = Callable[[str], TargetReader]
DetectFactory = Callable[[list[str] | None, int, bool, Any], DetectFn]
LandmarkerFactory = Callable[[list[str] | None], Any]


@dataclass
class AnalysisRequest:
    """Everything one scan needs. Carried verbatim across the queued `run`
    signal so the parameter list never has to grow into the signal signature."""

    target_path: str
    stride: int = 15
    threshold: float = 0.5
    providers: list[str] | None = None
    detection_size: int = 640
    # The face DETECTOR for the scan (None = buffalo_l). A non-buffalo detector
    # finds faces and ArcFace still adds the embedding for clustering; it can't
    # produce gender/age, so `fast` is forced (no genderage pack) for it.
    detector: Any = None  # DetectorModel | None
    sections: Any = None              # SectionSet | None
    preview: bool = False
    workers: int = 1
    fast: bool = True
    start_index: int = 0              # resume: skip this many sampled positions
    initial: Any = field(default=None)  # FaceMap | None — seed for a resume
    # After the catalog scan, also build the per-frame geometry table (the
    # detection-free runtime artifact). landmark_refine BAKES 2dfan4-refined
    # keypoints into it (the runtime then uses them as-is).
    compute_geometry: bool = True
    landmark_refine: bool = False
    landmark_min_score: float = 0.5
    # Bake a steady per-face roll (2dfan4) into the geometry so rotation
    # compensation works in detection-free playback (POSE/Landmark-68 would
    # otherwise fall back to the noisier keypoint angle there).
    bake_angle: bool = True
    # Recognise faces in cross-frame batches (one ArcFace call per ~32 faces
    # instead of per face). Only the det+rec scan path is batched; the full
    # demographics pack embeds inline. Output-identical; ~22% faster scan.
    batch_recognition: bool = True


def _default_reader(target_path: str) -> TargetReader:
    """Open the target for analysis. Video uses cv2 (no ffmpeg dependency — the
    catalog scan must work regardless of the chosen video backend)."""
    target = Target(path=Path(target_path))
    if target.kind is TargetKind.IMAGE:
        return ImageTargetReader(target)
    if target.kind is TargetKind.VIDEO:
        return CV2VideoTargetReader(target)
    raise ValueError(f"unsupported target kind: {target.kind}")


def _default_detect(
    providers: list[str] | None, detection_size: int, fast: bool,
    detector: Any = None,
) -> DetectFn:
    """A detector closure over a fresh analyser. ``detector`` (None = buffalo_l)
    finds the faces; ArcFace adds the embedding for clustering. ``fast`` runs
    detection + recognition only (no age/sex/landmark — much quicker); the full
    pack (demographics) is buffalo_l-only, so a custom detector always uses the
    fast det+rec path regardless of ``fast``."""
    from sinner2.pipeline.detectors import DetectorModel
    from sinner2.pipeline.face_analyser import FaceAnalyser

    analyser = FaceAnalyser(
        providers=providers, detection_size=detection_size, detector=detector
    )
    custom = detector is not None and detector != DetectorModel.BUFFALO_L
    return _AnalyserDetect(analyser, det_rec=(custom or fast))


class _AnalyserDetect:
    """A ``DetectFn`` that OWNS its FaceAnalyser so the scan can release it when
    done — a standalone yoloface/scrfd detector's ONNX session would otherwise
    leak (a fresh analyser is built per scan). buffalo_l's release is a no-op
    (shared singleton)."""

    def __init__(self, analyser: Any, *, det_rec: bool) -> None:
        self._analyser = analyser
        # Exposed so the job can switch to cross-frame batched recognition: only
        # the det+rec path is batchable (the full pack embeds inline via .get()).
        self.analyser = analyser
        self.det_rec = det_rec
        self._fn = (
            analyser.analyse_det_rec if det_rec else analyser.analyse_uncached
        )

    def __call__(self, frame: Any) -> list:
        return self._fn(frame)

    def release(self) -> None:
        self._analyser.release()


def _release_detect(detect: Any) -> None:
    """Release a detect fn's analyser if it owns one (test stubs don't)."""
    rel = getattr(detect, "release", None)
    if callable(rel):
        rel()


def _batch_wiring(detect: Any, enabled: bool) -> tuple[Any, Any]:
    """Wire cross-frame batched recognition for the catalog scan.

    When enabled AND ``detect`` is a batchable det+rec analyser-detect, returns
    (a detect-only-with-crops closure, the analyser's recognise fn) so the scan
    detects per frame but recognises crops in batches. Otherwise returns
    ``(detect, None)`` — the unchanged per-frame path (test stubs / full pack)."""
    analyser = getattr(detect, "analyser", None)
    if not (enabled and analyser is not None and getattr(detect, "det_rec", False)):
        return detect, None
    return (
        (lambda frame, a=analyser: a.detect_with_crops(frame)),
        analyser.recognize_crops,
    )


def _default_landmarker(providers: list[str] | None) -> Any:
    """A set-up 2dfan4 landmarker for baking refined keypoints into geometry."""
    from sinner2.pipeline.processors.landmarker import FaceLandmarker

    lm = FaceLandmarker(providers=providers)
    lm.setup()
    return lm


class FaceMapAnalysisJob(QObject):
    """Runs one catalog scan. Reused across runs; `cancel()` stops the active
    one (thread-safe). Emits from whatever thread `run` executes on, so the GUI
    connects the signals with a queued connection."""

    progress = Signal(int, int)        # positions done, total (current phase)
    geometryStarted = Signal()         # phase 2: the per-frame geometry pass began
    finished = Signal(object, object, int, int)  # catalog, geometry|None, scanned, total
    failed = Signal(str)
    preview = Signal(object)           # a frame being scanned (when preview is on)
    position = Signal(int)             # the frame index currently being scanned

    def __init__(
        self,
        *,
        reader_factory: ReaderFactory | None = None,
        detect_factory: DetectFactory | None = None,
        landmarker_factory: LandmarkerFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._reader_factory = reader_factory or _default_reader
        self._detect_factory = detect_factory or _default_detect
        self._landmarker_factory = landmarker_factory or _default_landmarker
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @Slot(object)
    def run(self, request: AnalysisRequest) -> None:
        self._cancel.clear()
        try:
            reader = self._reader_factory(request.target_path)
        except Exception as exc:  # noqa: BLE001 — surfaced to the GUI
            self.failed.emit(f"cannot open target: {exc}")
            return
        # Copy the previewed frame: the reader may reuse its decode buffer, and
        # the frame crosses to the GUI thread via a queued signal.
        on_preview = (
            (lambda frame: self.preview.emit(frame.copy()))
            if request.preview else None
        )
        providers = list(request.providers) if request.providers else None
        geometry: Any = None
        detect: Any = None
        try:
            detect = self._detect_factory(
                providers, request.detection_size, request.fast, request.detector,
            )
            # Cross-frame batched recognition (when enabled + batchable): scan
            # with a detect-only-with-crops closure and recognise in chunks.
            scan_detect, recognize_batch = _batch_wiring(
                detect, request.batch_recognition
            )
            # Pin the shared buffalo_l pack across BOTH phases: a concurrent
            # providers/det-size change must not null + finalize its ORT sessions
            # under these scan workers (the teardown defers to the scan's end).
            with pin_shared_face_analysis():
                catalog, scanned, total = analyze_target(
                    reader, scan_detect,
                    stride=request.stride, threshold=request.threshold,
                    sections=request.sections, workers=request.workers,
                    start_index=request.start_index, initial=request.initial,
                    recognize_batch=recognize_batch,
                    cancel_event=self._cancel,
                    on_progress=lambda done, tot: self.progress.emit(done, tot),
                    on_preview=on_preview,
                    on_position=lambda idx: self.position.emit(idx),
                )
                # Phase 2 — the detection-free runtime artifact. Reuses the open
                # reader; matches every frame's faces to the catalog just built.
                if request.compute_geometry and not self._cancel.is_set():
                    geometry = self._build_geometry(
                        reader, catalog, providers, request
                    )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        finally:
            _release_detect(detect)  # free the scan detector's ONNX session
            try:
                reader.release()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        self.finished.emit(catalog, geometry, scanned, total)

    def _build_geometry(
        self, reader: TargetReader, catalog: Any,
        providers: list[str] | None, request: AnalysisRequest,
    ) -> Any:
        """Run the full-frame geometry pass (phase 2), baking 2dfan4-refined
        keypoints when landmark_refine is on. Throttles its per-frame progress so
        a long video doesn't flood the GUI signal queue."""
        # The 2dfan4 landmarker is needed to REFINE keypoints and/or to BAKE a
        # steady roll angle — build it when either is on (they share the pass).
        landmarker = (
            self._landmarker_factory(providers)
            if (request.landmark_refine or request.bake_angle) else None
        )
        # Geometry needs detection + recognition (box/keypoints/embedding) only —
        # NOT the genderage pack the catalog scan may run. Build the fast det+rec
        # detector regardless of `fast`; it reuses the cached buffalo_l models, so
        # it's not a second load — just skips the per-frame age/sex inference.
        geo_base = self._detect_factory(
            providers, request.detection_size, True, request.detector
        )
        # Cross-frame batched recognition for the geometry pass too (it's the
        # longer, stride-1 pass): detect ONLY per frame, attach each face's crop
        # on its ORIGINAL keypoints (so embeddings match the non-batch path, which
        # recognises before refinement), and recognise in chunks.
        geo_analyser = getattr(geo_base, "analyser", None)
        geo_batch = (
            request.batch_recognition and geo_analyser is not None
            and getattr(geo_base, "det_rec", False)
        )
        geo_detect = self._geometry_detect(
            (lambda f, a=geo_analyser: a.detect_faces(f)) if geo_batch else geo_base,
            landmarker, request.landmark_min_score,
            refine=request.landmark_refine, bake_angle=request.bake_angle,
            attach_crops=(
                geo_analyser.attach_recognition_crops if geo_batch else None
            ),
        )
        recognize_batch = geo_analyser.recognize_crops if geo_batch else None
        self.geometryStarted.emit()
        try:
            geometry, _scanned, _total = precompute_geometry(
                reader, geo_detect, catalog,
                sections=request.sections, workers=request.workers,
                refined=request.landmark_refine,
                recognize_batch=recognize_batch,
                cancel_event=self._cancel,
                on_progress=lambda done, tot: (
                    self.progress.emit(done, tot)
                    if (done % 5 == 0 or done == tot) else None
                ),
            )
            return geometry
        finally:
            _release_detect(geo_base)  # free the geometry detector's ONNX session
            if landmarker is not None:
                try:
                    landmarker.release()
                except Exception:  # noqa: BLE001 — best-effort
                    pass

    @staticmethod
    def _geometry_detect(
        base_detect: DetectFn, landmarker: Any, min_score: float,
        *, refine: bool, bake_angle: bool, attach_crops: Any = None,
    ) -> DetectFn:
        """The geometry detector: the catalog's det+rec, plus (when a landmarker
        is given) per face — 2dfan4 keypoint refinement (``refine``) and/or a
        baked in-plane roll angle (``bake_angle``) for detection-free rotation
        compensation. The roll uses the 2dfan4 eye-line when 2dfan4 was confident,
        else the detector's 5-keypoint eye-line.

        ``attach_crops`` (cross-frame batched recognition): a ``(frame, faces)``
        callable that stashes each face's ArcFace crop — applied BEFORE keypoint
        refinement so the embedding aligns on the original detector keypoints,
        exactly as the non-batched path's inline recognition would."""
        if landmarker is None and attach_crops is None:
            return base_detect
        if landmarker is not None:
            from sinner2.pipeline.processors.face_swapper_types import (
                RotationAngleSource,
            )
            from sinner2.pipeline.processors.landmarker import landmark_68_to_5
            from sinner2.pipeline.processors.rotation_compensation import (
                compute_roll,
            )

        def detect(frame: Any) -> list:
            faces = base_detect(frame)
            # Recognition crop on the ORIGINAL keypoints (before any refine).
            if attach_crops is not None:
                attach_crops(frame, faces)
            if landmarker is None:
                return faces
            for face in faces:
                lm68: Any = None
                try:
                    lm68, score = landmarker.detect_68(frame, face.bbox)
                except Exception:  # noqa: BLE001 — best-effort
                    lm68, score = None, 0.0
                good = lm68 is not None and score >= min_score
                if good and refine:
                    try:
                        face.kps = landmark_68_to_5(lm68)
                    except Exception:  # noqa: BLE001
                        pass
                if bake_angle:
                    # 2dfan4 eye-line when confident (steadiest on tilt), else the
                    # detector keypoints — better than the runtime kps fallback a
                    # pose-less rebuilt face would otherwise hit.
                    src = (
                        RotationAngleSource.LANDMARK_68 if good
                        else RotationAngleSource.KEYPOINTS
                    )
                    try:
                        face.baked_roll = compute_roll(
                            face, src, lm68 if good else None
                        )
                    except Exception:  # noqa: BLE001
                        pass
            return faces

        return detect
