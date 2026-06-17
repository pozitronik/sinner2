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
import numbers
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from sinner2.pipeline.processors.face_swapper_types import RotationAngleSource
from sinner2.types import Frame

# Crop side as a multiple of the face's larger bbox dimension — generous so the
# rotated face never clips and stays well clear of the crop border (where the
# rotate-back warp samples black), with a floor so re-detect/swap/enhance has
# enough pixels.
_CROP_SCALE = 2.0
_CROP_MIN = 128
# Pixel-difference threshold for the "what did the operation change" mask.
_DIFF_THRESHOLD = 8
# Feather (px) on the composite mask — softens the blend back into the frame.
_FEATHER_SIGMA = 3.0


def compute_roll(
    face: Any, source: RotationAngleSource, landmark_68: Any = None
) -> float:
    """In-plane roll of a face in degrees (0 = upright).

    LANDMARK_68 measures the eye-centre line of the 2dfan4 68 landmarks (the
    steadiest on tilted faces) when they're supplied; POSE reads insightface's
    3D estimate (face.pose[2]); KEYPOINTS measures the detector eye-line. Each
    falls back to the detector eye keypoints when its source is unavailable.
    Returns 0.0 when nothing usable is present (→ no compensation).

    A face carrying a precomputed ``baked_roll`` (a detection-free geometry face)
    uses it directly, regardless of the source — there's no live pose estimate to
    read, so the angle baked during the scan is the steadiest available."""
    baked = getattr(face, "baked_roll", None)
    if isinstance(baked, numbers.Real) and math.isfinite(float(baked)):
        return float(baked)
    if source is RotationAngleSource.LANDMARK_68 and landmark_68 is not None:
        lm = np.asarray(landmark_68, np.float32)
        if len(lm) >= 48:
            left = lm[36:42].mean(axis=0)
            right = lm[42:48].mean(axis=0)
            return math.degrees(
                math.atan2(float(right[1] - left[1]), float(right[0] - left[0]))
            )
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


def _crop_geometry(face: Any, angle_deg: float) -> tuple[np.ndarray, int]:
    """Upright affine + crop side for a face's bbox."""
    x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(_CROP_MIN, int(_CROP_SCALE * max(x2 - x1, y2 - y1)))
    return _upright_matrix(cx, cy, angle_deg, size), size


def _composite_back(
    target: Frame, upright: np.ndarray, processed: np.ndarray, m: np.ndarray
) -> Frame | None:
    """Blend `processed` (an upright crop after swap/enhance) back into `target`
    at the original angle, over only the pixels the operation changed.

    Returns None when nothing changed. The alpha is clamped to the warp's valid
    region (eroded) so the rotate-back never blends in the black border the
    inverse warp samples outside the crop — the source of the square halos."""
    h, w = target.shape[:2]
    diff = cv2.absdiff(processed, upright).max(axis=2)
    mask = (diff > _DIFF_THRESHOLD).astype(np.float32)
    if mask.max() <= 0:
        return None
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=_FEATHER_SIGMA)

    m_inv = cv2.invertAffineTransform(m)
    back = cv2.warpAffine(processed, m_inv, (w, h)).astype(np.float32)
    alpha = cv2.warpAffine(mask, m_inv, (w, h))
    # Only composite where the inverse warp had real source pixels (inside the
    # crop). Erode to drop the 1-px interpolation fringe at the crop edge.
    valid = cv2.warpAffine(
        np.ones(upright.shape[:2], np.float32), m_inv, (w, h)
    )
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8), iterations=2)
    alpha = (alpha * valid)[..., None]
    blended = target.astype(np.float32) * (1.0 - alpha) + back * alpha
    return blended.astype(np.uint8)


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
        m, size = _crop_geometry(face, angle_deg)
        # Replicate edge pixels for off-frame areas instead of filling black —
        # otherwise inswapper blends the swapped face against that black and
        # leaves a dark halo on the face (which the valid-region clamp, being
        # outside the crop, can't catch).
        upright = cv2.warpAffine(
            result, m, (size, size), borderMode=cv2.BORDER_REPLICATE
        )

        target = None
        if redetect:
            target = _central_face(analyser.analyse_uncached(upright), size)
        if target is None:
            target = _rotated_face(face, m)

        swapped = swapper.get(upright, target, source_face, paste_back=True)
        out = _composite_back(result, upright, swapped, m)
        if out is None:
            return swapper.get(result, face, source_face, paste_back=True)
        return out
    except Exception:
        return swapper.get(result, face, source_face, paste_back=True)


def enhance_with_uprighting(
    result: Frame,
    original: Frame,
    face: Any,
    enhance_crop: Any,
    *,
    angle_deg: float,
) -> Frame:
    """Re-enhance one tilted face uprighted, compositing over `result`.

    The crop is taken from `original` (pre-enhance pixels) so the enhancer sees
    the raw face, not the cursed whole-frame enhancement already in `result`;
    `enhance_crop(upright_crop)` returns the enhanced crop. Leaves `result`
    untouched for this face on any error (it keeps the whole-frame enhance)."""
    try:
        m, size = _crop_geometry(face, angle_deg)
        upright = cv2.warpAffine(
            original, m, (size, size), borderMode=cv2.BORDER_REPLICATE
        )
        enhanced = enhance_crop(upright)
        out = _composite_back(result, upright, enhanced, m)
        return result if out is None else out
    except Exception:
        return result
