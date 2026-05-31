"""Rotation compensation for the face swapper (experimental).

When a target face is tilted in-plane past a threshold, the detector's
keypoints get sloppy and inswapper's keypoint-driven alignment suffers. This
module uprights a crop around the face, gets clean keypoints there (by
re-detecting, or by rotating the existing ones in), runs the swap on the
upright crop, and composites the swapped face back at its original angle with
a feathered mask.

Note this only addresses IN-PLANE roll. Out-of-plane yaw (a profile turn) is a
different, harder problem that uprighting can't fix.

Everything is best-effort: any failure falls back to a plain in-place swap, so
turning the feature on can never crash a render.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from sinner2.pipeline.processors.face_swapper_types import RotationAngleSource
from sinner2.types import Frame

# Crop side as a multiple of the face's larger bbox dimension — generous so the
# rotated face never clips, with a floor so re-detect/swap has enough pixels.
_CROP_SCALE = 1.6
_CROP_MIN = 128
# Pixel-difference threshold for the "what did the swap change" mask.
_DIFF_THRESHOLD = 8


def compute_roll(face: Any, source: RotationAngleSource) -> float:
    """In-plane roll of a face in degrees (0 = upright).

    POSE reads insightface's 3D estimate (face.pose[2]); KEYPOINTS measures the
    eye-line angle. POSE falls back to keypoints when the pose model isn't in
    the pack. Returns 0.0 when neither is available (→ no compensation)."""
    if source is RotationAngleSource.POSE:
        pose = getattr(face, "pose", None)
        if pose is not None and len(pose) >= 3:
            return float(pose[2])
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 2:
        return 0.0
    (lx, ly), (rx, ry) = kps[0][:2], kps[1][:2]
    return math.degrees(math.atan2(float(ry) - float(ly), float(rx) - float(lx)))


def _upright_matrix(cx: float, cy: float, roll_deg: float, size: int) -> np.ndarray:
    """Affine that rotates the frame about (cx, cy) by `roll_deg` (uprighting
    the eye-line) and recenters the face in a `size`×`size` crop."""
    m = cv2.getRotationMatrix2D((cx, cy), roll_deg, 1.0)
    m[0, 2] += size / 2.0 - cx
    m[1, 2] += size / 2.0 - cy
    return m


def _rotated_face(face: Any, m: np.ndarray) -> SimpleNamespace:
    """The original face with its keypoints mapped into upright-crop space —
    the fallback target when re-detection is off or finds nothing."""
    kps = np.asarray(face.kps, dtype=np.float32).reshape(1, -1, 2)
    kps_crop = cv2.transform(kps, m).reshape(-1, 2)
    return SimpleNamespace(kps=kps_crop)


def _central_face(faces: list, size: int) -> Any | None:
    """The detected face nearest the crop centre (the face we uprighted),
    or None if nothing was found."""
    if not faces:
        return None
    centre = np.array([size / 2.0, size / 2.0])

    def dist(f: Any) -> float:
        x1, y1, x2, y2 = (float(v) for v in f.bbox[:4])
        return float(np.linalg.norm(np.array([(x1 + x2) / 2, (y1 + y2) / 2]) - centre))

    return min(faces, key=dist)


def swap_with_uprighting(
    result: Frame,
    face: Any,
    source_face: Any,
    swapper: Any,
    analyser: Any,
    *,
    angle_deg: float,
    redetect: bool,
) -> Frame:
    """Swap one tilted face by uprighting it first, compositing back at angle.

    `result` is the frame being built up (earlier faces already swapped into
    it); returns a new frame with this face swapped. Falls back to a plain
    in-place swap on any error."""
    try:
        h, w = result.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        size = max(_CROP_MIN, int(_CROP_SCALE * max(x2 - x1, y2 - y1)))

        m = _upright_matrix(cx, cy, angle_deg, size)
        upright = cv2.warpAffine(result, m, (size, size))

        target = None
        if redetect:
            target = _central_face(analyser.analyse_uncached(upright), size)
        if target is None:
            target = _rotated_face(face, m)

        swapped = swapper.get(upright, target, source_face, paste_back=True)

        # Mask only what the swap changed (inswapper pastes the new face and
        # leaves the rest of the crop equal to `upright`), feather it, then warp
        # both the swapped crop and the mask back to frame space and alpha-blend
        # — so only the swapped-face pixels are composited, at the right angle.
        diff = cv2.absdiff(swapped, upright).max(axis=2)
        mask = (diff > _DIFF_THRESHOLD).astype(np.float32)
        if mask.max() <= 0:
            return swapper.get(result, face, source_face, paste_back=True)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)

        m_inv = cv2.invertAffineTransform(m)
        back = cv2.warpAffine(swapped, m_inv, (w, h)).astype(np.float32)
        alpha = cv2.warpAffine(mask, m_inv, (w, h))[..., None]
        blended = result.astype(np.float32) * (1.0 - alpha) + back * alpha
        return blended.astype(np.uint8)
    except Exception:
        return swapper.get(result, face, source_face, paste_back=True)
