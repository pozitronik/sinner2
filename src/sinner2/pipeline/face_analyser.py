import threading
from typing import Any

from sinner2.types import Frame

_FACE_MODEL_NAME = "buffalo_l"
_DEFAULT_DET_SIZE = 640
# SCRFD (the buffalo_l detector) downsamples by strides 8/16/32, so its input
# must be a multiple of 32. Align any requested det_size down to the nearest
# multiple and never below one stride tile.
_DET_SIZE_ALIGN = 32


def _normalize_det_size(size: int) -> tuple[int, int]:
    aligned = max(_DET_SIZE_ALIGN, (int(size) // _DET_SIZE_ALIGN) * _DET_SIZE_ALIGN)
    return (aligned, aligned)


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
            # Pin the download/cache root to the project models dir — otherwise
            # insightface defaults to ~/.insightface and the buffalo_l pack lands
            # outside the chosen models folder (where every other model lives).
            # insightface forces a "models" subdir under root, so the pack ends up
            # at <models_dir>/models/buffalo_l; passing get_models_dir() (not its
            # parent) keeps it inside the chosen dir even under SINNER2_MODELS_DIR.
            app = FaceAnalysis(
                name=_FACE_MODEL_NAME,
                root=str(get_models_dir()),
                providers=eps,
                provider_options=build_provider_options(eps),
            )
            app.prepare(ctx_id=0, det_size=_normalize_det_size(det_size))
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
    ) -> None:
        if detection_interval < 1:
            raise ValueError(f"detection_interval must be >= 1; got {detection_interval}")
        self._detection_interval = detection_interval
        self._detection_size = detection_size
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
        else:
            faces = _get_shared_face_analysis(
                self._providers, self._detection_size
            ).get(frame)
        with self._lock:
            self._cached_faces = faces
        return list(faces or [])

    def analyse_uncached(self, frame: Frame) -> list[Any]:
        # Always the full buffalo_l pack — this is the path the SOURCE face uses,
        # and the source needs the ArcFace embedding a standalone detector lacks.
        return list(
            _get_shared_face_analysis(self._providers, self._detection_size).get(frame)
        )

    def provides_gender(self) -> bool:
        """Whether detected faces carry insightface's `.sex` (only the full
        buffalo_l pack does — standalone detectors are box+keypoints only). The
        swapper gates its gender filter on this."""
        return self._detector is None

    def reset_cache(self) -> None:
        with self._lock:
            self._cached_faces = None
            self._frame_counter = 0
