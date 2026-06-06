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
    default off, reproducing the restorers' original paste byte-for-byte."""
    h, w = frame.shape[:2]
    m_inv = cv2.invertAffineTransform(matrix)
    border = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
    back = cv2.warpAffine(patch, m_inv, (w, h), borderMode=border).astype(np.float32)
    alpha = cv2.warpAffine(mask, m_inv, (w, h))
    if clip_mask:
        alpha = alpha.clip(0.0, 1.0)
    alpha = alpha[..., None]
    return (frame.astype(np.float32) * (1.0 - alpha) + back * alpha).astype(np.uint8)
