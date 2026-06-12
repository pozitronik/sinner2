"""68-point face landmarking (2dfan4) for refined alignment.

The detectors (buffalo_l / yoloface / scrfd) emit 5 keypoints; on tilted or
hard-pose faces those keypoints degrade, so the swap's keypoint-driven
alignment is off BEFORE any restoration can help — the root reason rotation
compensation exists. 2dfan4 (FAN) is a dedicated landmark net, robust across
pose; running it on each detected box and deriving the 5 ArcFace points from
its 68 landmarks gives a more accurate crop. With detect-once the refined
geometry flows to the swap, occlusion mask, and enhancer via the shared
ChainContext — everything aligns better, once per frame.

Pipeline replicated VERBATIM from facefusion's face_landmarker.detect_with_2dfan4
(verified against master): scale the box so its larger side is ~195px,
translate-center it into a 256 crop, optional CLAHE on dark faces, then BGR /255
NCHW (facefusion feeds the BGR crop through a channel-order-preserving Lab
round-trip and never swaps to RGB, so the model is fed BGR — we match that). The
model returns 68 points in 64-heatmap space which scale to 256 and invert back
to frame coordinates; the score is the mean heatmap peak. The 68->5 mapping
(eye centers / nose tip / mouth corners) is facefusion's convert_to_face_landmark_5.

Out-of-plane / upside-down handling (facefusion's face_angle pre-rotation) is
deliberately omitted: the detector boxes are axis-aligned and the common case
is angle 0; in-plane roll is handled downstream by rotation compensation.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.types import Frame

MODEL_FILE = "2dfan4.onnx"
_MODEL_SIZE = 256
_BOX_TARGET = 195.0  # facefusion scales the box's larger side to ~195px
_HEATMAP_SIZE = 64
# 68-landmark indices (ibug/300-W ordering) for the 5 ArcFace points.
_LEFT_EYE = slice(36, 42)
_RIGHT_EYE = slice(42, 48)
_NOSE_TIP = 30
_MOUTH_LEFT = 48
_MOUTH_RIGHT = 54


def landmark_68_to_5(landmark_68: np.ndarray) -> np.ndarray:
    """The 5 ArcFace keypoints derived from 68 landmarks (facefusion's
    convert_to_face_landmark_5): left-eye centre, right-eye centre, nose tip,
    left mouth corner, right mouth corner."""
    return np.array(
        [
            landmark_68[_LEFT_EYE].mean(axis=0),
            landmark_68[_RIGHT_EYE].mean(axis=0),
            landmark_68[_NOSE_TIP],
            landmark_68[_MOUTH_LEFT],
            landmark_68[_MOUTH_RIGHT],
        ],
        dtype=np.float32,
    )


def roll_from_landmark_68(landmark_68: np.ndarray) -> float:
    """In-plane roll (degrees, 0 = upright) from the eye-centre line of the 68
    landmarks — steadier than the detector's two eye keypoints on tilted faces."""
    left = landmark_68[_LEFT_EYE].mean(axis=0)
    right = landmark_68[_RIGHT_EYE].mean(axis=0)
    return float(
        np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
    )


def _optimize_contrast(crop: Frame) -> Frame:
    """facefusion's conditional CLAHE: lift the L channel on very dark crops so
    the landmarker sees the face. The Lab labels say RGB but operate on the
    BGR crop — channel-order-preserving (L is luminance), so the round-trip
    returns the same BGR data when no CLAHE is applied."""
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    if float(np.mean(lab[:, :, 0])) < 30:
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


class FaceLandmarker:
    """2dfan4 ONNX landmarker → 68 points + a confidence score. Shared,
    thread-safe ORT session (cached by path), so N workers share one model."""

    thread_safe = True

    def __init__(self, providers: list[str] | None = None) -> None:
        self._providers = (
            list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
        )
        self._session: Any = None
        self._in_name = "input"
        self._out_names: list[str] = []

    def setup(self) -> None:
        from sinner2.pipeline.model_cache import get_onnx_session

        self._session = get_onnx_session(MODEL_FILE, providers=self._providers)
        self._in_name = self._session.get_inputs()[0].name
        self._out_names = [o.name for o in self._session.get_outputs()]

    def detect_68(self, frame: Frame, bbox: Any) -> tuple[np.ndarray, float]:
        """68 landmarks (image coords) + score in [0,1] for the face in `bbox`
        ([x1,y1,x2,y2]). Higher score = more confident; callers gate on it."""
        if self._session is None:
            raise RuntimeError("FaceLandmarker.detect_68 called before setup()")
        box = np.asarray(bbox, np.float32)[:4]
        side = float(max(box[2] - box[0], box[3] - box[1]))
        scale = _BOX_TARGET / max(side, 1.0)
        translation = (_MODEL_SIZE - (box[2:] + box[:2]) * scale) * 0.5
        affine = np.array(
            [[scale, 0.0, translation[0]], [0.0, scale, translation[1]]],
            np.float32,
        )
        crop = cv2.warpAffine(frame, affine, (_MODEL_SIZE, _MODEL_SIZE))
        crop = _optimize_contrast(crop.astype(np.uint8))
        blob = np.ascontiguousarray(
            crop.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        landmark, heatmap = self._session.run(self._out_names, {self._in_name: blob})
        lm = landmark[:, :, :2][0] / _HEATMAP_SIZE * _MODEL_SIZE
        lm = _transform_points(lm, cv2.invertAffineTransform(affine))
        peak = float(np.mean(np.amax(heatmap, axis=(2, 3))))
        score = float(np.interp(peak, [0.0, 0.9], [0.0, 1.0]))
        return lm.astype(np.float32), score

    def refine_kps(self, frame: Frame, bbox: Any) -> tuple[np.ndarray, float]:
        """Convenience: the refined 5 ArcFace keypoints + score for a box."""
        lm68, score = self.detect_68(frame, bbox)
        return landmark_68_to_5(lm68), score

    def release(self) -> None:
        from sinner2.pipeline.model_cache import release_onnx_session

        self._session = None
        release_onnx_session(MODEL_FILE, self._providers)
