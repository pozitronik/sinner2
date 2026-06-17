import threading
from collections.abc import Callable
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
            from sinner2.pipeline.model_cache import (
                build_provider_options,
                get_models_dir,
            )

            # None → platform default; an explicit [] stays empty (the user chose
            # no providers → ORT runs on CPU). Only unspecified falls back.
            eps = list(DEFAULT_ONNX_PROVIDERS) if providers is None else list(providers)
            # The detector pack (buffalo_l = 5 small fixed-shape models) does NOT
            # go through TensorRT: each sub-model would compile its OWN engine
            # (minutes of first-run build) for little gain, and some hit the same
            # fp16 issues as the swapper. Strip TRT here so the detector runs on
            # CUDA(+CPU) even when the swapper uses TRT — only the (single, heavy)
            # inswapper model is worth a TRT engine. FaceAnalysis still forwards
            # provider_options, so the detector keeps the CUDA cuDNN/arena tuning.
            stripped = [p for p in eps if p != "TensorrtExecutionProvider"]
            if eps and not stripped:
                # User picked ONLY TensorRT — the detector can't use it; fall back
                # to the GPU default rather than nothing. An already-empty list
                # (no providers selected) stays empty.
                stripped = list(DEFAULT_ONNX_PROVIDERS)
            eps = stripped
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
                app = FaceAnalysis(
                    name=_FACE_MODEL_NAME,
                    root=str(root),
                    providers=eps,
                    provider_options=build_provider_options(eps),
                )
                app.prepare(ctx_id=0, det_size=_normalize_det_size(det_size))
            finally:
                if not pack_present:
                    _notify_load("")
            _shared_app = app
        return _shared_app


def reset_shared_face_analysis() -> None:
    """Test-only — drop the cached insightface model."""
    global _shared_app
    with _shared_lock:
        _shared_app = None


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
            faces = self._detect_kps_only(frame)
        else:
            faces = _get_shared_face_analysis(
                self._providers, self._detection_size
            ).get(frame)
        with self._lock:
            self._cached_faces = faces
        return list(faces or [])

    def _detect_kps_only(self, frame: Frame) -> list[Any]:
        """Run ONLY the shared pack's detection model (det_10g) — the exact
        call `.get()` starts with — and wrap results as FaceLite, skipping the
        per-face aux models. Thread-safety matches `.get()` (same det call)."""
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
        from insightface.app.common import Face

        app = _get_shared_face_analysis(self._providers, self._detection_size)
        rec = app.models.get("recognition")
        if self._detector is not None:
            faces: list[Any] = []
            for d in self._detector.detect(frame):
                kps = getattr(d, "kps", None)
                face = Face(
                    bbox=np.asarray(d.bbox, np.float32),
                    kps=None if kps is None else np.asarray(kps, np.float32),
                    det_score=float(getattr(d, "det_score", 1.0)),
                )
                if rec is not None and face.kps is not None:
                    rec.get(frame, face)  # → face.embedding / normed_embedding
                faces.append(face)
            return faces
        bboxes, kpss = app.det_model.detect(frame, max_num=0, metric="default")
        faces = []
        for i in range(len(bboxes)):
            face = Face(
                bbox=bboxes[i][0:4],
                kps=kpss[i] if kpss is not None else None,
                det_score=bboxes[i][4],
            )
            if rec is not None:
                rec.get(frame, face)  # sets face.embedding (→ normed_embedding)
            faces.append(face)
        return faces

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
