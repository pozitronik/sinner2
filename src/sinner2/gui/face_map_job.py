"""Background job that builds a target's face-map catalog off the GUI thread.

Wraps the headless `analyze_target` scan in a QObject that lives on its own
QThread (like FaceDetectionProbe): the GUI kicks it via a queued `run` and gets
`progress` / `finished` / `failed` back. The reader + detector builders are
injectable so tests need no media or models.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from sinner2.config.target import Target, TargetKind
from sinner2.io.cv2_video_target_reader import CV2VideoTargetReader
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.pipeline.face_map_analyzer import DetectFn, analyze_target

ReaderFactory = Callable[[str], TargetReader]
DetectFactory = Callable[[list[str] | None, int], DetectFn]


def _default_reader(target_path: str) -> TargetReader:
    """Open the target for analysis. Video uses cv2 (no ffmpeg dependency — the
    catalog scan must work regardless of the chosen video backend)."""
    target = Target(path=Path(target_path))
    if target.kind is TargetKind.IMAGE:
        return ImageTargetReader(target)
    if target.kind is TargetKind.VIDEO:
        return CV2VideoTargetReader(target)
    raise ValueError(f"unsupported target kind: {target.kind}")


def _default_detect(providers: list[str] | None, detection_size: int) -> DetectFn:
    """A detector closure over a fresh buffalo_l analyser (the full pack — its
    ArcFace embeddings are what make the clustering identity-stable)."""
    from sinner2.pipeline.face_analyser import FaceAnalyser

    analyser = FaceAnalyser(providers=providers, detection_size=detection_size)
    return lambda frame: analyser.analyse_uncached(frame)


class FaceMapAnalysisJob(QObject):
    """Runs one catalog scan. Reused across runs; `cancel()` stops the active
    one (thread-safe). Emits from whatever thread `run` executes on, so the GUI
    connects the signals with a queued connection."""

    progress = Signal(int, int)   # frames scanned, frames to scan
    finished = Signal(object)     # the built FaceMap
    failed = Signal(str)

    def __init__(
        self,
        *,
        reader_factory: ReaderFactory | None = None,
        detect_factory: DetectFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._reader_factory = reader_factory or _default_reader
        self._detect_factory = detect_factory or _default_detect
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @Slot(str, int, float, object, int)
    def run(
        self,
        target_path: str,
        stride: int,
        threshold: float,
        providers: Any,
        detection_size: int,
    ) -> None:
        self._cancel.clear()
        try:
            reader = self._reader_factory(target_path)
        except Exception as exc:  # noqa: BLE001 — surfaced to the GUI
            self.failed.emit(f"cannot open target: {exc}")
            return
        try:
            detect = self._detect_factory(
                list(providers) if providers else None, detection_size
            )
            face_map = analyze_target(
                reader, detect,
                stride=stride, threshold=threshold,
                cancel_event=self._cancel,
                on_progress=lambda done, total: self.progress.emit(done, total),
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        finally:
            try:
                reader.release()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        self.finished.emit(face_map)
