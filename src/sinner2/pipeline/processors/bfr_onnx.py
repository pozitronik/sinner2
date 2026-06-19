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
from sinner2.pipeline.face_geometry import feather_mask, paste_back
from sinner2.pipeline.model_cache import get_onnx_session, release_onnx_session
from sinner2.pipeline.processors.bfr_common import (
    denormalize_restored_face,
    normalize_aligned_face,
)
from sinner2.types import Frame

_ALIGN_SIZE = 512


def _derive_align_size(input_shape: Any, default: int = _ALIGN_SIZE) -> int:
    """Read the model's expected square input size from its declared input shape
    (e.g. [1, 3, 512, 512] → 512). GPEN/RestoreFormer export static shapes, so
    the model self-describes its resolution. Falls back to `default` for dynamic
    / non-square / malformed shapes."""
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
        return default
    h, w = input_shape[2], input_shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and h == w:
        return h
    return default


def _restore_aligned(
    session: Any,
    aligned_bgr: np.ndarray,
    in_name: str = "input",
    out_name: str = "output",
) -> Frame:
    """Run a plain BFR model on a 512 aligned BGR face → restored 512 BGR face.

    RGB, normalized to [-1,1], NCHW in; [-1,1] out (clipped). No fidelity input.
    """
    chw = normalize_aligned_face(aligned_bgr)
    out = session.run([out_name], {in_name: chw})[0]
    return denormalize_restored_face(out)


class PlainBfrBackend:
    """Detect faces, restore each with a plain BFR ONNX model, paste back.
    The ONNX session is thread-safe + shared (cached by path), so there's no
    per-worker model copy."""

    def __init__(self, model_file: str, providers: list[str] | None = None) -> None:
        self._model_file = model_file
        # None → platform default; an explicit [] is PRESERVED (user picked no
        # providers → ORT's CPU last-resort), matching the swapper/analyser so the
        # global ONNX provider list behaves the same everywhere.
        self._providers = (
            list(providers) if providers is not None else list(DEFAULT_ONNX_PROVIDERS)
        )
        self._session: Any = None
        self._analyser: FaceAnalyser | None = None
        # Overwritten from the real session at setup(); the defaults match
        # GPEN's verified names and keep the unit tests' stub session simple.
        self._in_name = "input"
        self._out_name = "output"
        # Alignment resolution — derived from the model's declared input shape at
        # setup() (512 / 1024 / 2048 across the GPEN family). Default + mask kept
        # so stub-session tests that bypass setup() still work.
        self._align_size = _ALIGN_SIZE
        self._mask = feather_mask(_ALIGN_SIZE)

    def setup(self) -> None:
        # get_onnx_session raises if the model is missing — the GUI ensures it
        # (with a download confirmation) before enabling the enhancer.
        self._session = get_onnx_session(self._model_file, providers=self._providers)
        inp = self._session.get_inputs()[0]
        self._in_name = inp.name
        self._out_name = self._session.get_outputs()[0].name
        self._align_size = _derive_align_size(inp.shape)
        self._mask = feather_mask(self._align_size)
        # detection_only: alignment needs box+kps only — skips the four aux
        # models buffalo_l's .get() runs per face (≈half the detect cost).
        self._analyser = FaceAnalyser(providers=self._providers, detection_only=True)

    def enhance(self, img: Frame, faces: list | None = None) -> Frame:
        """Restore every detected face in `img`. Best-effort per face — a
        failure leaves that face untouched rather than breaking the frame.

        ``faces``: upstream detections (the swapper's, via ChainContext) to
        align with instead of re-detecting — only ``.kps`` is read, so both
        insightface faces and FaceLite work. None → self-detect; an empty
        list is trusted (no faces on this frame)."""
        if self._session is None or self._analyser is None:
            raise RuntimeError("PlainBfrBackend.enhance called before setup()")
        result = img
        from insightface.utils import face_align

        size = self._align_size
        detected = faces if faces is not None else self._analyser.analyse(img)
        for face in detected:
            try:
                m = face_align.estimate_norm(np.asarray(face.kps, np.float32), size)
                aligned = cv2.warpAffine(result, m, (size, size))
                restored = _restore_aligned(
                    self._session, aligned, self._in_name, self._out_name
                )
                result = paste_back(result, restored, m, self._mask)
            except Exception:
                continue
        return result

    def release(self) -> None:
        """Drop the session + detector refs and evict the (exclusive) ONNX
        session from the shared cache so disabling the enhancer frees its VRAM
        instead of leaving it resident."""
        self._session = None
        self._analyser = None
        release_onnx_session(self._model_file, self._providers)
