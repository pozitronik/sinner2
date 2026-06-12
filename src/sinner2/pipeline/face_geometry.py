"""Shared face-geometry primitives for the ONNX face restorers.

The align → restore → warp-back → feathered-blend shape is common to the ONNX
restorers (CodeFormer and the plain BFR family — GPEN / RestoreFormer++). Two
pieces of that shape were byte-identical across them; this module is their single
home. The per-model ALIGN step (template + solver) deliberately stays in each
backend — those genuinely differ, and changing them changes output pixels.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from sinner2.types import Frame


@lru_cache(maxsize=8)
def feather_mask(size: int, pad_frac: float = 0.08) -> np.ndarray:
    """A square [0,1] alignment mask: a 1.0 interior inset by ``pad_frac`` on
    every side, Gaussian-blurred so the paste-back blend feathers at the edge
    instead of leaving a hard seam. Cached per ``size`` (one per resolution, so
    the 512 / 1024 / 2048 restorers each reuse theirs); the masks are read-only —
    callers warp them, never mutate."""
    m = np.zeros((size, size), np.float32)
    pad = int(size * pad_frac)
    m[pad:size - pad, pad:size - pad] = 1.0
    return cv2.GaussianBlur(m, (0, 0), sigmaX=size * 0.02)


def paste_back(
    frame: Frame,
    patch: Frame,
    matrix: np.ndarray,
    mask: np.ndarray,
    *,
    border_replicate: bool = False,
    clip_mask: bool = False,
) -> Frame:
    """Composite an aligned ``patch`` back into ``frame``.

    ``matrix`` is the 2x3 affine that mapped ``frame`` into the aligned space the
    ``patch`` lives in; this inverts it, warps both the patch and its ``mask``
    back to frame coordinates, and alpha-blends the patch in where the mask is
    set (feathered at the edges).

    ``border_replicate`` edge-extends the patch under warp (BORDER_REPLICATE)
    instead of black-filling — the swapper uses this to avoid dark halos around
    the pasted crop. ``clip_mask`` clamps the warped alpha to [0,1] before the
    blend (the swapper's mask can over/undershoot under interpolation). Both
    default off, reproducing the restorers' original paste byte-for-byte.

    The warp + float blend run only inside the bounding box of the warped
    patch (``mask`` must live in the same aligned space as ``patch``). Outside
    that box the warped alpha is identically 0, so a full-frame blend changes
    nothing there while costing ~8 full-frame float passes — at FullHD that was
    over HALF the per-frame cost of every restorer (scripts/enhancer_bench.py).
    The box is padded 2px for interpolation bleed at the quad edge.

    Pixel contract: byte-identical to the full-frame blend when the box covers
    the frame (the matrix is then untouched — pinned by the legacy-formula
    tests). For a translated box, cv2's fixed-point warp arithmetic is
    association-sensitive, so isolated pixels (~0.05%, measured) may shift one
    interpolation tap (a few LSB on noise, ±1 on natural images) — orders of
    magnitude below codec noise, pinned by tolerance tests."""
    h, w = frame.shape[:2]
    m_inv = cv2.invertAffineTransform(matrix)
    ph, pw = patch.shape[:2]
    corners = np.array(
        [[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32
    ).reshape(1, 4, 2)
    warped = cv2.transform(corners, m_inv)[0]
    x0 = max(int(np.floor(float(warped[:, 0].min()))) - 2, 0)
    y0 = max(int(np.floor(float(warped[:, 1].min()))) - 2, 0)
    x1 = min(int(np.ceil(float(warped[:, 0].max()))) + 2, w)
    y1 = min(int(np.ceil(float(warped[:, 1].max()))) + 2, h)
    if x0 >= x1 or y0 >= y1:
        return frame.copy()  # patch lands entirely outside the frame
    # Shift the inverse transform into ROI coordinates: warp sampling for a
    # destination pixel depends only on its absolute coordinates, so an
    # integer-translated matrix over a cropped destination samples the exact
    # same source positions — identical pixels, ~1/15th the area for a face.
    m_roi = m_inv.copy()
    m_roi[0, 2] -= x0
    m_roi[1, 2] -= y0
    rw, rh = x1 - x0, y1 - y0
    border = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
    back = cv2.warpAffine(patch, m_roi, (rw, rh), borderMode=border).astype(np.float32)
    alpha = cv2.warpAffine(mask, m_roi, (rw, rh))
    if clip_mask:
        alpha = alpha.clip(0.0, 1.0)
    alpha = alpha[..., None]
    out = frame.copy()
    roi = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (
        roi.astype(np.float32) * (1.0 - alpha) + back * alpha
    ).astype(np.uint8)
    return out
