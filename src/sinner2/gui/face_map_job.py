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
from sinner2.pipeline.face_map_analyzer import DetectFn, analyze_target

ReaderFactory = Callable[[str], TargetReader]
DetectFactory = Callable[[list[str] | None, int, bool], DetectFn]


@dataclass
class AnalysisRequest:
    """Everything one scan needs. Carried verbatim across the queued `run`
    signal so the parameter list never has to grow into the signal signature."""

    target_path: str
    stride: int = 15
    threshold: float = 0.5
    providers: list[str] | None = None
    detection_size: int = 640
    sections: Any = None              # SectionSet | None
    preview: bool = False
    workers: int = 1
    fast: bool = True
    start_index: int = 0              # resume: skip this many sampled positions
    initial: Any = field(default=None)  # FaceMap | None — seed for a resume


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
    providers: list[str] | None, detection_size: int, fast: bool
) -> DetectFn:
    """A detector closure over a fresh buffalo_l analyser. ``fast`` runs
    detection + recognition only (no age/sex/landmark — much quicker); otherwise
    the full pack, which also yields the demographics for the cards."""
    from sinner2.pipeline.face_analyser import FaceAnalyser

    analyser = FaceAnalyser(providers=providers, detection_size=detection_size)
    if fast:
        return lambda frame: analyser.analyse_det_rec(frame)
    return lambda frame: analyser.analyse_uncached(frame)


class FaceMapAnalysisJob(QObject):
    """Runs one catalog scan. Reused across runs; `cancel()` stops the active
    one (thread-safe). Emits from whatever thread `run` executes on, so the GUI
    connects the signals with a queued connection."""

    progress = Signal(int, int)        # sampled positions done, total
    finished = Signal(object, int, int)  # catalog, scanned positions, total
    failed = Signal(str)
    preview = Signal(object)           # a frame being scanned (when preview is on)
    position = Signal(int)             # the frame index currently being scanned

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
        try:
            detect = self._detect_factory(
                list(request.providers) if request.providers else None,
                request.detection_size, request.fast,
            )
            catalog, scanned, total = analyze_target(
                reader, detect,
                stride=request.stride, threshold=request.threshold,
                sections=request.sections, workers=request.workers,
                start_index=request.start_index, initial=request.initial,
                cancel_event=self._cancel,
                on_progress=lambda done, tot: self.progress.emit(done, tot),
                on_preview=on_preview,
                on_position=lambda idx: self.position.emit(idx),
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        finally:
            try:
                reader.release()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        self.finished.emit(catalog, scanned, total)
