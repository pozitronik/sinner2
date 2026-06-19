"""Shared tensor I/O for the blind-face-restoration (BFR) ONNX backends.

GPEN-BFR / RestoreFormer++ (`PlainBfrBackend`) and CodeFormer share one restore
contract: a 512 aligned BGR face → RGB, normalized to [-1,1], NCHW in; a [-1,1]
output (clipped) → BGR uint8. The per-model `session.run` differs (CodeFormer
adds a fidelity `weight` input), but the normalize/denormalize wrapping it is
identical — defined once here so the convention has a single home.
"""
from __future__ import annotations

import cv2
import numpy as np

from sinner2.types import Frame


def normalize_aligned_face(aligned_bgr: np.ndarray) -> np.ndarray:
    """512 aligned BGR face → RGB [-1,1] NCHW float32 tensor, shape (1,3,H,W)."""
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None], np.float32)


def denormalize_restored_face(out: np.ndarray) -> Frame:
    """Raw BFR model output (1,3,H,W) in [-1,1] → restored BGR uint8 face."""
    img = (np.clip(out[0], -1.0, 1.0) + 1.0) / 2.0
    img = (img.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
