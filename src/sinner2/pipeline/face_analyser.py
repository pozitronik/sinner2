import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from sinner2.types import Frame

_FACE_MODEL_NAME = "buffalo_l"
# Optional progress notifier for the (insightface-internal) buffalo_l pack
# download. insightface has no progress hook, so we just flag the START
# (non-empty message) and END (empty message) so the GUI can show a busy
# indicator instead of a silent multi-minute hang on the first run. Set once
# at GUI startup; None = no listener (headless / tests).
_load_notifier: Callable[[str], None] | None = None


def set_load_notifier(notifier: Callable[[str], None] | None) -> None:
    """Install (or clear) the model-load progress notifier — called with a
    message when the buffalo_l download starts and "" when it finishes."""
    global _load_notifier
    _load_notifier = notifier


def _notify_load(message: str) -> None:
    notifier = _load_notifier
    if notifier is not None:
        try:
            notifier(message)
        except Exception:  # noqa: BLE001 — a UI hint must never break a load
            pass
_DEFAULT_DET_SIZE = 640
# SCRFD (the buffalo_l detector) downsamples by strides 8/16/32, so its input
# must be a multiple of 32. Align any requested det_size down to the nearest
# multiple and never below one stride tile.
_DET_SIZE_ALIGN = 32


def _normalize_det_size(size: int) -> tuple[int, int]:
    aligned = max(_DET_SIZE_ALIGN, (int(size) // _DET_SIZE_ALIGN) * _DET_SIZE_ALIGN)
    return (aligned, aligned)


def _buffalo_root_and_pack(models_dir: Any) -> tuple[Any, Any]:
    """The insightface ``root`` to pass and the resulting pack directory.

    insightface stores at ``<root>/models/<name>``. When the models dir is
    named "models" (the default), root = its parent → the pack lands at
    ``<models_dir>/buffalo_l``. Otherwise root = the models dir itself → the
    pack nests at ``<models_dir>/models/buffalo_l`` (kept inside the chosen
    folder; the parent could place it outside)."""
    if models_dir.name == "models":
        return models_dir.parent, models_dir / _FACE_MODEL_NAME
    return models_dir, models_dir / "models" / _FACE_MODEL_NAME


_shared_app: Any = None
_shared_lock = threading.RLock()
# A catalog scan pins the shared pack while it infers (pin_shared_face_analysis):
# a providers/det-size teardown landing mid-scan is DEFERRED to the last unpin so
# it can't null + finalize an ORT session under a running scan worker.
_shared_pins = 0
_reset_pending = False


def _get_shared_face_analysis(
    providers: list[str] | None = None, det_size: int = _DEFAULT_DET_SIZE
) -> Any:
    """Lazily load and cache the insightface FaceAnalysis singleton.

    The insightface model itself is expensive — load once and share across
    every FaceAnalyser instance in the process. Per-stream detection state
    lives on FaceAnalyser; the underlying model has no per-stream state.

    Providers are passed in by the caller (FaceAnalyser, from its owning
    processor's execution profile); None falls back to the platform-default
    EP order. The model is a process-wide singleton, so it picks up the
    FIRST caller's providers AND det_size — changing either requires calling
    `reset_shared_face_analysis()` so the next call rebuilds (insightface
    picks providers + prepares det_size at construction time).
    """
    global _shared_app
    with _shared_lock:
        if _shared_app is None:
            from insightface.app import FaceAnalysis

            from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
            from sinner2.pipeline.memory_probe import measure_model_load
            from sinner2.pipeline.model_cache import (
                build_provider_options,
                detector_providers,
                get_models_dir,
            )

            # None → platform default; an explicit [] stays empty (the user chose
            # no providers → ORT runs on CPU). Only unspecified falls back.
            eps = list(DEFAULT_ONNX_PROVIDERS) if providers is None else list(providers)
            # Detectors run on CUDA, not TensorRT, by default — the buffalo_l pack
            # is 5 small fixed-shape models, each of which would compile its OWN
            # engine (minutes of first-run build) for little gain. Configurable via
            # SINNER2_TENSORRT_DETECTOR; detector_providers() logs the downgrade so
            # it isn't silent. FaceAnalysis still forwards provider_options, so the
            # detector keeps the CUDA cuDNN/arena tuning.
            eps = detector_providers(eps)
            # insightface ALWAYS stores the pack at <root>/models/<name> (the
            # "models" segment is hardcoded). The models dir is normally named
            # "models", so passing its PARENT as root lands the pack cleanly at
            # <models_dir>/buffalo_l instead of the doubled
            # <models_dir>/models/buffalo_l. For a custom models dir NOT named
            # "models" we keep the dir itself as root (nested, but still inside
            # the chosen folder — the parent could place it outside).
            root, pack_dir = _buffalo_root_and_pack(get_models_dir())
            # First run downloads the pack (~300MB) from inside insightface with
            # no progress hook. Flag start/end so the GUI can show a busy
            # indicator; the empty message in `finally` clears it on success or
            # failure alike.
            pack_present = pack_dir.is_dir()
            if not pack_present:
                _notify_load("Downloading face-analysis models (~300 MB)…")
            try:
                with measure_model_load("buffalo_l (face pack)"):
                    app = FaceAnalysis(
                        name=_FACE_MODEL_NAME,
                        root=str(root),
                        providers=eps,
                        provider_options=build_provider_options(eps),
                    )
                    app.prepare(
                        ctx_id=0, det_size=_normalize_det_size(det_size)
                    )
            finally:
                if not pack_present:
                    _notify_load("")
            _shared_app = app
        return _shared_app


def reset_shared_face_analysis() -> None:
    """Drop the cached insightface model so the next call rebuilds it — used on a
    providers / det-size change (player_controller) and by tests.

    If a catalog scan is PINNING the pack (mid-inference on its ORT sessions via
    ``pin_shared_face_analysis``), the drop is DEFERRED until the last pin
    releases: nulling + finalizing an ORT session under a running scan worker is
    a use-after-free. Quiescent (the live chain / detection probe path) → drops
    immediately, exactly as before."""
    global _reset_pending
    with _shared_lock:
        if _shared_pins > 0:
            _reset_pending = True
            return
        _drop_shared_app_locked()


def _drop_shared_app_locked() -> None:
    """Null the singleton and clear any deferred-drop flag. Caller holds the lock
    (RLock, so reentry from a pin release is fine)."""
    global _shared_app, _reset_pending
    _shared_app = None
    _reset_pending = False


@contextmanager
def pin_shared_face_analysis() -> Iterator[None]:
    """Pin the shared buffalo_l pack for the duration of a catalog scan so a
    concurrent providers/det-size change can't null + finalize its ORT sessions
    under the scan's worker threads. A teardown requested while pinned is
    deferred and applied when the last pin releases — which also keeps the scan
    on ONE consistent detector (the premise the resume signature relies on). The
    live chain and detection probe do NOT pin: they re-fetch the singleton per
    frame and want a providers change to take effect at once."""
    global _shared_pins
    with _shared_lock:
        _shared_pins += 1
    try:
        yield
    finally:
        with _shared_lock:
            _shared_pins -= 1
            if _shared_pins == 0 and _reset_pending:
                _drop_shared_app_locked()


def _recognition_batch_capable(rec: Any) -> bool:
    """Whether the ArcFace ONNX export accepts a batch of N crops in one call.

    insightface stores the session's declared input shape on ``rec.input_shape``
    (e.g. ``['None', 3, 112, 112]``). A symbolic batch dim (a string like
    ``'None'`` / ``'batch'``) or a non-positive int is dynamic → stack-able; a
    fixed positive int (a batch locked to that size) is NOT → recognise per-face.
    """
    shape = getattr(rec, "input_shape", None)
    if not shape:
        return False
    batch = shape[0]
    if isinstance(batch, int):
        return batch <= 0  # 0 / -1 = dynamic; >=1 = fixed
    return True  # 'None' / 'batch' / other symbolic → dynamic


def _batch_recognize(frame: Frame, rec: Any, faces: list[Any]) -> None:
    """Set ``face.embedding`` for every face carrying keypoints, batching them
    through ArcFace in ONE ``get_feat`` call when the export allows it.

    Equivalent to ``rec.get(frame, face)`` per face — same alignment
    (``norm_crop`` on the face keypoints) and the same per-row embeddings (the
    batched ``get_feat`` is bit-identical to the per-image calls it replaces),
    just fewer ONNX invocations. Faces without keypoints are left embedding-less,
    exactly as before. A fixed-batch export falls back to the per-face path."""
    if rec is None:
        return
    targets = [f for f in faces if getattr(f, "kps", None) is not None]
    if not targets:
        return
    if not _recognition_batch_capable(rec):
        for face in targets:
            rec.get(frame, face)  # → face.embedding / normed_embedding
        return
    from insightface.utils import face_align

    size = rec.input_size[0]
    crops = [
        face_align.norm_crop(frame, landmark=f.kps, image_size=size)
        for f in targets
    ]
    feats = rec.get_feat(crops)  # (N, D) — one ONNX call for all N crops
    for face, feat in zip(targets, feats):
        face.embedding = feat.flatten()


class FaceAnalyser:
    """Per-stream face detection with optional caching by interval.

    `detection_interval=1` runs detection on every frame. Higher values reuse
    the previous detection result on intermediate frames — the major perf win
    on stable scenes where faces don't move much between frames. The cache
    assumption holds only for sequential frames; with multi-worker executors
    processing in parallel, prefer `detection_interval=1`.
    """

    def __init__(
        self,
        detection_interval: int = 1,
        providers: list[str] | None = None,
        detection_size: int = _DEFAULT_DET_SIZE,
        detector: Any = None,
        detection_only: bool = False,
    ) -> None:
        if detection_interval < 1:
            raise ValueError(f"detection_interval must be >= 1; got {detection_interval}")
        self._detection_interval = detection_interval
        self._detection_size = detection_size
        # detection_only drives the shared buffalo_l pack's DET model alone:
        # `.get()` runs four aux models (two landmark nets, genderage,
        # recognition) per face, but consumers that only ALIGN BY KEYPOINTS
        # (the ONNX restorer backends) need none of them — at FullHD the aux
        # passes roughly doubled detection cost (scripts/enhancer_bench.py).
        # Same detector, same boxes/kps; faces are FaceLite (no sex/pose).
        # Ignored when a standalone detector is set (already detection-only).
        self._detection_only = bool(detection_only)
        # Optional standalone TARGET detector (yoloface / scrfd). None = the full
        # buffalo_l pack. Built + loaded eagerly here (single-threaded
        # construction, so the N-worker pool that later shares this analyser
        # never races on first-frame setup). The SOURCE face still uses
        # buffalo_l (analyse_uncached) for its ArcFace embedding.
        from sinner2.pipeline.detectors import DetectorModel, build_detector

        det_model = detector if detector is not None else DetectorModel.BUFFALO_L
        self._detector = build_detector(
            det_model,
            providers if providers is not None else None,
            size=detection_size,
        )
        if self._detector is not None:
            self._detector.setup()
        # Preserve an explicit empty list (user selected no providers) — only
        # None means "unspecified" (→ platform default in _get_shared_face_analysis).
        self._providers = list(providers) if providers is not None else None
        self._frame_counter = 0
        self._cached_faces: list[Any] | None = None
        self._lock = threading.RLock()

    def analyse(self, frame: Frame) -> list[Any]:
        # The expensive insightface .get() call MUST happen outside the lock —
        # otherwise N worker threads hammering the shared FaceAnalyser would
        # serialize on detection and lose all parallelism. The race here is
        # benign: two concurrent cache misses both detect, the second write
        # overwrites the first, no incorrectness — just one wasted detection.
        with self._lock:
            cache_miss = (
                self._cached_faces is None
                or self._frame_counter % self._detection_interval == 0
            )
            self._frame_counter += 1
            cached = self._cached_faces
        if not cache_miss:
            return list(cached or [])
        if self._detector is not None:
            faces = self._detector.detect(frame)
        elif self._detection_only:
            faces = self.detect_only(frame)
        else:
            faces = _get_shared_face_analysis(
                self._providers, self._detection_size
            ).get(frame)
        with self._lock:
            self._cached_faces = faces
        return list(faces or [])

    def detect_only(self, frame: Frame) -> list[Any]:
        """Run ONLY the shared pack's detection model (det_10g) — the exact
        call `.get()` starts with — and wrap results as FaceLite (box + kps +
        det_score), skipping the per-face aux models (2 landmark nets, genderage,
        recognition). Thread-safety matches `.get()` (same det call). Use this
        when only the box/keypoints are needed (e.g. rotation re-detect)."""
        from sinner2.pipeline.detectors import FaceLite

        app = _get_shared_face_analysis(self._providers, self._detection_size)
        bboxes, kpss = app.det_model.detect(frame, max_num=0, metric="default")
        faces: list[Any] = []
        for i in range(len(bboxes)):
            faces.append(
                FaceLite(
                    bbox=np.asarray(bboxes[i][:4], np.float32),
                    kps=np.asarray(kpss[i], np.float32),
                    det_score=float(bboxes[i][4]),
                )
            )
        return faces

    def analyse_uncached(self, frame: Frame) -> list[Any]:
        # Always the full buffalo_l pack — this is the path the SOURCE face uses,
        # and the source needs the ArcFace embedding a standalone detector lacks.
        return list(
            _get_shared_face_analysis(self._providers, self._detection_size).get(frame)
        )

    def detect_faces(self, frame: Frame) -> list[Any]:
        """Detection ONLY → insightface Face objects (bbox/kps/det_score), no
        recognition. A standalone detector (yoloface / scrfd) finds the faces
        when set; otherwise buffalo_l's det_model does. The cross-frame scan
        detects with this and recognises crops in batches later; ``analyse_det_rec``
        adds per-frame recognition on top."""
        from insightface.app.common import Face

        app = _get_shared_face_analysis(self._providers, self._detection_size)
        if self._detector is not None:
            faces: list[Any] = []
            for d in self._detector.detect(frame):
                kps = getattr(d, "kps", None)
                faces.append(Face(
                    bbox=np.asarray(d.bbox, np.float32),
                    kps=None if kps is None else np.asarray(kps, np.float32),
                    det_score=float(getattr(d, "det_score", 1.0)),
                ))
            return faces
        bboxes, kpss = app.det_model.detect(frame, max_num=0, metric="default")
        return [
            Face(
                bbox=bboxes[i][0:4],
                kps=kpss[i] if kpss is not None else None,
                det_score=bboxes[i][4],
            )
            for i in range(len(bboxes))
        ]

    def analyse_det_rec(self, frame: Frame) -> list[Any]:
        """Detection + RECOGNITION only — skip the genderage + two landmark nets
        the full `.get()` runs per face. The ArcFace embedding is all the
        face-map clustering needs, so this roughly halves the per-frame cost on
        a multi-face frame. The returned faces carry bbox/kps/det_score/embedding
        but NO sex/age/pose (use the full pack when you want those).

        Same shared, thread-safe ORT sessions as `.get()`, so it parallelizes.

        When a standalone detector is set (yoloface / scrfd) it FINDS the faces
        and ArcFace (from the shared pack) adds the embedding per face — so a
        faster detection-only detector can still drive identity clustering. The
        detector's 5 keypoints align the ArcFace crop, same as buffalo_l's."""
        app = _get_shared_face_analysis(self._providers, self._detection_size)
        rec = app.models.get("recognition")
        faces = self.detect_faces(frame)
        # Recognise every face in ONE ArcFace call (per-frame batch) rather than
        # one call per face — same embeddings (verified bit-identical), fewer
        # kernel launches. Falls back to per-face for a fixed-batch export.
        _batch_recognize(frame, rec, faces)
        return faces

    def attach_recognition_crops(self, frame: Frame, faces: list[Any]) -> None:
        """Align + stash each face's ArcFace crop on ``face._batch_crop`` (for
        deferred, cross-frame batched recognition). Uses the face's CURRENT
        keypoints, so call it AFTER any keypoint refinement. Faces without
        keypoints are left without a crop (they get no embedding later)."""
        app = _get_shared_face_analysis(self._providers, self._detection_size)
        rec = app.models.get("recognition")
        if rec is None:
            return
        from insightface.utils import face_align

        size = rec.input_size[0]
        for face in faces:
            if getattr(face, "kps", None) is not None:
                face._batch_crop = face_align.norm_crop(
                    frame, landmark=face.kps, image_size=size
                )

    def detect_with_crops(self, frame: Frame) -> list[Any]:
        """``detect_faces`` plus each face's aligned ArcFace crop stashed for
        later batched recognition — the cross-frame scan's per-frame step."""
        faces = self.detect_faces(frame)
        self.attach_recognition_crops(frame, faces)
        return faces

    def recognize_crops(self, crops: list[Any]) -> np.ndarray:
        """Embed N aligned ArcFace crops in ONE call (the cross-frame batch),
        returning an (N, D) array in input order. Falls back to per-crop for a
        fixed-batch export."""
        app = _get_shared_face_analysis(self._providers, self._detection_size)
        rec = app.models.get("recognition")
        if rec is None or not crops:
            return np.empty((0, 512), np.float32)
        if _recognition_batch_capable(rec):
            return np.asarray(rec.get_feat(crops))
        return np.stack([np.asarray(rec.get_feat(c)).flatten() for c in crops])

    def provides_gender(self) -> bool:
        """Whether detected faces carry insightface's `.sex` (only the full
        buffalo_l pack does — standalone detectors and detection_only mode are
        box+keypoints only). The swapper gates its gender filter on this."""
        return self._detector is None and not self._detection_only

    def provides_embeddings(self) -> bool:
        """Whether ``analyse()`` yields a `.normed_embedding` per face — only the
        full buffalo_l pack does (same condition as the aux models). Per-identity
        face-mapping needs this; when False the swapper switches to
        ``analyse_det_rec`` so routing has embeddings to match."""
        return self._detector is None and not self._detection_only

    def reset_cache(self) -> None:
        with self._lock:
            self._cached_faces = None
            self._frame_counter = 0

    def release(self) -> None:
        """Release the STANDALONE detector's ONNX session (yoloface / scrfd) —
        each scan builds a fresh analyser, so without this its CUDA session
        leaks. The shared buffalo_l pack is a process-wide singleton (not owned
        here, ``_detector`` is None for it), so it's left alone."""
        if self._detector is not None:
            self._detector.release()
            self._detector = None
