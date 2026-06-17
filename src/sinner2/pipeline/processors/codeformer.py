"""CodeFormer face restoration backend (ONNX).

A second enhancer option alongside GFPGAN — often better identity preservation
on degraded faces, with a fidelity knob `w` (0 = max restoration/quality,
1 = max fidelity to the input). Runs via onnxruntime (no arch to vendor); the
model exposes `w` as a scalar input, so the knob is fully controllable.

Unlike GFPGAN (which detects + restores + pastes internally), the ONNX model is
just the restore net, so this detects faces (shared insightface), aligns each
to 512, restores, and pastes back with a feathered mask — the same align →
process → warp-back → blend shape as the upscaler / occlusion passes.

I/O contract (verified against the model): input (1,3,512,512) RGB normalized
to [-1,1]; scalar double `weight`; output (1,3,512,512) in [-1,1] (clip).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.face_geometry import feather_mask, paste_back
from sinner2.pipeline.model_cache import get_onnx_session, release_onnx_session
from sinner2.types import Frame

MODEL_FILE = "codeformer.onnx"
_ALIGN_SIZE = 512
_FEATHER_MASK = feather_mask(_ALIGN_SIZE)


def _restore_aligned(session: Any, aligned_bgr: np.ndarray, fidelity: float) -> Frame:
    """Run CodeFormer on a 512 aligned BGR face → restored 512 BGR face."""
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None], np.float32)
    out = session.run(
        ["output"], {"input": chw, "weight": np.array(fidelity, np.float64)}
    )[0]
    img = (np.clip(out[0], -1.0, 1.0) + 1.0) / 2.0
    img = (img.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class CodeFormerBackend:
    """Detect faces, restore each with CodeFormer, paste back. ONNX session is
    thread-safe + shared (cached by path), so no per-worker model copy."""

    def __init__(
        self, fidelity: float = 0.7, providers: list[str] | None = None
    ) -> None:
        self._fidelity = float(fidelity)
        # None → platform default; an explicit [] is PRESERVED (CPU last-resort),
        # matching the swapper so the global ONNX provider list is uniform.
        self._providers = (
            list(providers) if providers is not None else list(DEFAULT_ONNX_PROVIDERS)
        )
        self._session: Any = None
        self._analyser: FaceAnalyser | None = None

    def setup(self) -> None:
        # get_onnx_session raises if the model is missing — the GUI ensures it
        # (with a download confirmation) before enabling CodeFormer.
        self._session = get_onnx_session(MODEL_FILE, providers=self._providers)
        # detection_only: alignment needs box+kps only — skips the four aux
        # models buffalo_l's .get() runs per face (≈half the detect cost).
        self._analyser = FaceAnalyser(providers=self._providers, detection_only=True)

    def enhance(self, img: Frame, faces: list | None = None) -> Frame:
        """Restore every detected face in `img`. Best-effort per face — a
        failure leaves that face untouched rather than breaking the frame.

        ``faces``: upstream detections (the swapper's, via ChainContext) to
        align with instead of re-detecting; None → self-detect; an empty
        list is trusted (no faces on this frame)."""
        if self._session is None or self._analyser is None:
            raise RuntimeError("CodeFormerBackend.enhance called before setup()")
        result = img
        from insightface.utils import face_align

        detected = faces if faces is not None else self._analyser.analyse(img)
        for face in detected:
            try:
                m = face_align.estimate_norm(
                    np.asarray(face.kps, np.float32), _ALIGN_SIZE
                )
                aligned = cv2.warpAffine(result, m, (_ALIGN_SIZE, _ALIGN_SIZE))
                restored = _restore_aligned(self._session, aligned, self._fidelity)
                result = paste_back(result, restored, m, _FEATHER_MASK)
            except Exception:
                continue
        return result

    def release(self) -> None:
        """Drop the session + detector refs and evict the (exclusive) CodeFormer
        ONNX session from the shared cache so disabling the enhancer frees its
        ~377 MB of VRAM instead of leaving it resident."""
        self._session = None
        self._analyser = None
        release_onnx_session(MODEL_FILE, self._providers)
