"""Generic plain blind-face-restoration (BFR) ONNX backend.

Drives the no-knob ONNX restorers — GPEN-BFR and RestoreFormer++ — that share
one I/O contract: a single image input (1,3,512,512) RGB normalized to [-1,1]
and a single output (1,3,512,512) in [-1,1]. This is CodeFormer's restore path
minus the fidelity (`weight`) scalar, so it follows the same align → restore →
warp-back → feathered-blend shape as the CodeFormer / upscaler / occlusion
passes.

Input/output tensor names are read from the session at setup() (GPEN's export
lists every weight as a graph input, but onnxruntime still reports exactly one
real data input/output), so the backend adapts to whatever a given model names
them. Best-effort per face: a failure leaves that face untouched.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.model_cache import get_onnx_session, release_onnx_session
from sinner2.types import Frame

_ALIGN_SIZE = 512


def _make_feather_mask(size: int = _ALIGN_SIZE, pad_frac: float = 0.08) -> np.ndarray:
    m = np.zeros((size, size), np.float32)
    pad = int(size * pad_frac)
    m[pad:size - pad, pad:size - pad] = 1.0
    return cv2.GaussianBlur(m, (0, 0), sigmaX=size * 0.02)


_FEATHER_MASK = _make_feather_mask()


def _restore_aligned(
    session: Any,
    aligned_bgr: np.ndarray,
    in_name: str = "input",
    out_name: str = "output",
) -> Frame:
    """Run a plain BFR model on a 512 aligned BGR face → restored 512 BGR face.

    RGB, normalized to [-1,1], NCHW in; [-1,1] out (clipped). No fidelity input.
    """
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None], np.float32)
    out = session.run([out_name], {in_name: chw})[0]
    img = (np.clip(out[0], -1.0, 1.0) + 1.0) / 2.0
    img = (img.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _paste_face(frame: Frame, restored: Frame, m: np.ndarray) -> Frame:
    """Warp a restored aligned face back by the inverse of `m` and blend it in
    with the feathered face mask."""
    h, w = frame.shape[:2]
    m_inv = cv2.invertAffineTransform(m)
    back = cv2.warpAffine(restored, m_inv, (w, h)).astype(np.float32)
    alpha = cv2.warpAffine(_FEATHER_MASK, m_inv, (w, h))[..., None]
    return (frame.astype(np.float32) * (1.0 - alpha) + back * alpha).astype(np.uint8)


class PlainBfrBackend:
    """Detect faces, restore each with a plain BFR ONNX model, paste back.
    The ONNX session is thread-safe + shared (cached by path), so there's no
    per-worker model copy."""

    def __init__(self, model_file: str, providers: list[str] | None = None) -> None:
        self._model_file = model_file
        self._providers = list(providers) if providers else list(DEFAULT_ONNX_PROVIDERS)
        self._session: Any = None
        self._analyser: FaceAnalyser | None = None
        # Overwritten from the real session at setup(); the defaults match
        # GPEN's verified names and keep the unit tests' stub session simple.
        self._in_name = "input"
        self._out_name = "output"

    def setup(self) -> None:
        # get_onnx_session raises if the model is missing — the GUI ensures it
        # (with a download confirmation) before enabling the enhancer.
        self._session = get_onnx_session(self._model_file, providers=self._providers)
        self._in_name = self._session.get_inputs()[0].name
        self._out_name = self._session.get_outputs()[0].name
        self._analyser = FaceAnalyser(providers=self._providers)

    def enhance(self, img: Frame) -> Frame:
        """Restore every detected face in `img`. Best-effort per face — a
        failure leaves that face untouched rather than breaking the frame."""
        if self._session is None or self._analyser is None:
            raise RuntimeError("PlainBfrBackend.enhance called before setup()")
        result = img
        from insightface.utils import face_align

        for face in self._analyser.analyse(img):
            try:
                m = face_align.estimate_norm(
                    np.asarray(face.kps, np.float32), _ALIGN_SIZE
                )
                aligned = cv2.warpAffine(result, m, (_ALIGN_SIZE, _ALIGN_SIZE))
                restored = _restore_aligned(
                    self._session, aligned, self._in_name, self._out_name
                )
                result = _paste_face(result, restored, m)
            except Exception:
                continue
        return result

    def release(self) -> None:
        """Drop the session + detector refs and evict the (exclusive) ONNX
        session from the shared cache so disabling the enhancer frees its VRAM
        instead of leaving it resident."""
        self._session = None
        self._analyser = None
        release_onnx_session(self._model_file)
