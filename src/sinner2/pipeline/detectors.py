"""Alternative target-face detectors (detection-only).

buffalo_l (the default) is insightface's full pack — its `.get()` runs five
models per face every frame (detect + two landmark nets + gender + recognition).
The swapper only needs box + 5 keypoints for TARGET faces, so a detection-only
model skips four of those per frame. These detectors are used for per-frame
target detection ONLY; the SOURCE face still goes through buffalo_l (it needs
the ArcFace embedding), and gender/3D-pose are buffalo_l-only (the swapper
degrades gracefully — gender filter off, rotation falls back to keypoints).

Two backends:
  - YoloFaceDetector — yoloface_8n ONNX. Parsing verified numerically against
    buffalo_l on real frames (box IoU 0.93+, kps err <7px): letterbox-fit to
    640, pad zeros, BGR /255, NCHW; output (1,20,8400) → squeeze.T → split at
    [4,5] (box / score / 5×(x,y,conf) kps); scores thresholded; boxes+kps
    scaled back by the fit ratio; NMS.
  - ScrfdDetector — scrfd_2.5g via insightface's own SCRFD model class (no
    hand-parsing; that class is the reference SCRFD implementation).
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import cv2
import numpy as np

from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.pipeline.model_cache import (
    get_model_path,
    get_onnx_session_io,
    release_onnx_session,
)
from sinner2.types import Frame

_DET_SIZE = 640
_DEFAULT_SCORE = 0.5
_DEFAULT_NMS = 0.4


class DetectorModel(str, Enum):
    """Target-face detector. buffalo_l = the full insightface pack (default,
    only one that yields gender/pose); the others are detection-only."""

    BUFFALO_L = "buffalo_l"
    YOLOFACE = "yoloface"
    SCRFD_2_5G = "scrfd_2.5g"


# Model files for the standalone detectors.
DETECTOR_MODEL_FILES: dict[DetectorModel, str] = {
    DetectorModel.YOLOFACE: "yoloface_8n.onnx",
    DetectorModel.SCRFD_2_5G: "scrfd_2.5g.onnx",
}


class FaceLite:
    """Minimal detected face: bounding box + 5 keypoints + score. Deliberately
    carries NO `sex`/`pose`/`embedding`, so the swapper's getattr-guarded paths
    degrade (gender filter skipped, rotation uses keypoints)."""

    __slots__ = ("bbox", "kps", "det_score")

    def __init__(self, bbox: np.ndarray, kps: np.ndarray, det_score: float) -> None:
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score


def _yolo_detect(
    session: Any,
    frame: Frame,
    in_name: str,
    out_name: str,
    score_threshold: float,
    nms_threshold: float,
    size: int,
) -> list[FaceLite]:
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return []  # degenerate (empty) frame → no faces; avoids the /scale div-by-0
    scale = min(size / h, size / w)
    rh, rw = int(round(h * scale)), int(round(w * scale))
    canvas = np.zeros((size, size, 3), np.float32)
    canvas[:rh, :rw] = cv2.resize(frame, (rw, rh))
    blob = np.expand_dims((canvas / 255.0).transpose(2, 0, 1), 0).astype(np.float32)
    det = np.squeeze(session.run([out_name], {in_name: blob})[0]).T  # (8400, 20)
    box_raw, score_raw, kps_raw = det[:, :4], det[:, 4], det[:, 5:]
    keep = score_raw > score_threshold
    if not np.any(keep):
        return []
    box_raw, score_raw, kps_raw = box_raw[keep], score_raw[keep], kps_raw[keep]
    # (cx,cy,w,h) → (x1,y1,x2,y2), back to original resolution.
    xyxy = np.stack(
        [
            (box_raw[:, 0] - box_raw[:, 2] / 2) / scale,
            (box_raw[:, 1] - box_raw[:, 3] / 2) / scale,
            (box_raw[:, 0] + box_raw[:, 2] / 2) / scale,
            (box_raw[:, 1] + box_raw[:, 3] / 2) / scale,
        ],
        axis=1,
    )
    kps = kps_raw.reshape(len(kps_raw), -1, 3)[:, :, :2] / scale
    boxes_wh = np.stack(
        [xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1
    )
    indices = cv2.dnn.NMSBoxes(
        boxes_wh.tolist(), score_raw.tolist(), score_threshold, nms_threshold
    )
    if len(indices) == 0:
        return []
    faces: list[FaceLite] = []
    for i in np.array(indices).flatten():
        faces.append(
            FaceLite(
                bbox=xyxy[i].astype(np.float32),
                kps=kps[i].astype(np.float32),
                det_score=float(score_raw[i]),
            )
        )
    return faces


class YoloFaceDetector:
    """yoloface_8n ONNX detector → FaceLite list. Shared (thread-safe) session."""

    def __init__(
        self,
        model_file: str = "yoloface_8n.onnx",
        providers: list[str] | None = None,
        score_threshold: float = _DEFAULT_SCORE,
        nms_threshold: float = _DEFAULT_NMS,
        size: int = _DET_SIZE,
    ) -> None:
        self._model_file = model_file
        self._providers = list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
        self._score = score_threshold
        self._nms = nms_threshold
        self._size = size
        self._session: Any = None
        self._in_name = "input"
        self._out_name = "output"

    def setup(self) -> None:
        self._session, self._in_name, self._out_name = get_onnx_session_io(
            self._model_file, providers=self._providers
        )
        # yoloface_8n exports a FIXED square input (e.g. [1,3,640,640]); the
        # network can't accept any other size, so pin our letterbox target to the
        # model's static H/W. A requested det_size only takes effect if the model
        # were exported with a dynamic input.
        shape = self._session.get_inputs()[0].shape
        if len(shape) == 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h == w > 0:
                self._size = h

    def detect(self, frame: Frame) -> list[FaceLite]:
        if self._session is None:
            self.setup()
        return _yolo_detect(
            self._session, frame, self._in_name, self._out_name,
            self._score, self._nms, self._size,
        )

    def release(self) -> None:
        """Drop the shared ONNX session refcount so the cache can evict it — the
        scan builds a detector per run, so without this the CUDA session leaks."""
        if self._session is not None:
            release_onnx_session(self._model_file, self._providers)
            self._session = None


class ScrfdDetector:
    """scrfd_2.5g via insightface's reference SCRFD model class → FaceLite list."""

    def __init__(
        self,
        model_file: str = "scrfd_2.5g.onnx",
        providers: list[str] | None = None,
        size: int = _DET_SIZE,
    ) -> None:
        self._model_file = model_file
        self._providers = list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
        self._size = size
        self._det: Any = None

    def setup(self) -> None:
        from insightface.model_zoo import get_model

        # insightface's get_model does `name.endswith('.onnx')`, so it needs a
        # str — get_model_path returns a Path (would raise AttributeError).
        self._det = get_model(
            str(get_model_path(self._model_file)), providers=self._providers
        )
        # SCRFD downsamples by strides 8/16/32, so its input must be a multiple
        # of 32 (mirrors the buffalo_l det_size alignment in face_analyser).
        aligned = max(32, (self._size // 32) * 32)
        self._det.prepare(ctx_id=0, input_size=(aligned, aligned))

    def detect(self, frame: Frame) -> list[FaceLite]:
        if self._det is None:
            self.setup()
        bboxes, kpss = self._det.detect(frame, max_num=0, metric="default")
        faces: list[FaceLite] = []
        for i in range(len(bboxes)):
            b = bboxes[i]
            faces.append(
                FaceLite(
                    bbox=np.asarray(b[:4], np.float32),
                    kps=np.asarray(kpss[i], np.float32),
                    det_score=float(b[4]),
                )
            )
        return faces

    def release(self) -> None:
        """Drop the insightface SCRFD model so its ORT session can be reclaimed
        (insightface manages the session itself; we just release our reference)."""
        self._det = None


def build_detector(
    model: DetectorModel, providers: list[str] | None = None, size: int = _DET_SIZE
) -> YoloFaceDetector | ScrfdDetector | None:
    """The standalone detector for a model, or None for buffalo_l (which the
    FaceAnalyser drives through the full insightface pack itself)."""
    if model is DetectorModel.YOLOFACE:
        return YoloFaceDetector(
            DETECTOR_MODEL_FILES[model], providers=providers, size=size
        )
    if model is DetectorModel.SCRFD_2_5G:
        return ScrfdDetector(DETECTOR_MODEL_FILES[model], providers=providers, size=size)
    return None
