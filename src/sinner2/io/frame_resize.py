"""Frame downscale helpers shared by the readers and the GUI readout.

"Processing scale" trades output resolution for speed: smaller frames decode,
detect, swap, enhance, and encode faster. The scale is applied once, at the
reader, so everything downstream operates in the reduced space — no coordinate
or paste-back fixups, and the output is simply the smaller resolution.

The scaled dimensions are forced EVEN because the batch encoder targets H.264
/ yuv420p, which requires even width and height. A scale of >= 1.0 is a no-op
fast path: native dimensions pass through untouched (preserving the existing
odd-dimension behaviour for full-resolution runs).
"""
from __future__ import annotations

import cv2

from sinner2.types import Frame


def scaled_dims(width: int, height: int, scale: float) -> tuple[int, int]:
    """Target (width, height) for a downscale by `scale` in (0, 1].

    >= 1.0 returns the native size unchanged (no-op). Otherwise both axes are
    scaled, floored to even, and clamped to a >= 2 minimum so a tiny scale on
    a small source can't produce a 0- or 1-pixel (or odd) frame.
    """
    if scale >= 1.0:
        return width, height
    w = max(2, round(width * scale))
    h = max(2, round(height * scale))
    return w - (w % 2), h - (h % 2)


def resize_frame(frame: Frame, width: int, height: int) -> Frame:
    """Resize a BGR frame to (width, height), no-op if already that size.

    INTER_AREA is the right interpolation for downscaling (avoids the moiré
    INTER_LINEAR introduces when shrinking)."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
