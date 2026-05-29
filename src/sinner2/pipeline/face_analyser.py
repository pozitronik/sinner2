import threading
from typing import Any

from sinner2.types import Frame

_FACE_MODEL_NAME = "buffalo_l"
_DET_SIZE = (640, 640)

_shared_app: Any = None
_shared_lock = threading.RLock()


def _get_shared_face_analysis(providers: list[str] | None = None) -> Any:
    """Lazily load and cache the insightface FaceAnalysis singleton.

    The insightface model itself is expensive — load once and share across
    every FaceAnalyser instance in the process. Per-stream detection state
    lives on FaceAnalyser; the underlying model has no per-stream state.

    Providers are passed in by the caller (FaceAnalyser, from its owning
    processor's execution profile); None falls back to the platform-default
    EP order. The model is a process-wide singleton, so it picks up the
    FIRST caller's providers — changing providers requires calling
    `reset_shared_face_analysis()` so the next call rebuilds with the new
    list (insightface picks providers at construction time).
    """
    global _shared_app
    with _shared_lock:
        if _shared_app is None:
            from insightface.app import FaceAnalysis

            from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS

            eps = list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
            app = FaceAnalysis(name=_FACE_MODEL_NAME, providers=eps)
            app.prepare(ctx_id=0, det_size=_DET_SIZE)
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
        self, detection_interval: int = 1, providers: list[str] | None = None
    ) -> None:
        if detection_interval < 1:
            raise ValueError(f"detection_interval must be >= 1; got {detection_interval}")
        self._detection_interval = detection_interval
        self._providers = list(providers) if providers else None
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
        faces = _get_shared_face_analysis(self._providers).get(frame)
        with self._lock:
            self._cached_faces = faces
        return list(faces or [])

    def analyse_uncached(self, frame: Frame) -> list[Any]:
        return list(_get_shared_face_analysis(self._providers).get(frame))

    def reset_cache(self) -> None:
        with self._lock:
            self._cached_faces = None
            self._frame_counter = 0
