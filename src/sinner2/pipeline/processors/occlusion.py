"""Occlusion-aware face masking for the swapper.

inswapper pastes a roughly-rectangular aligned region over the target, which
overwrites whatever crosses the face — hair on the forehead, glasses, a hat,
the neck/jaw boundary, background in the corners. This masks the swap to the
actual facial-skin region (a BiSeNet face-parse), so those keep the original.

The parser is facexlib's BiSeNet — already installed (a GFPGAN dependency) and
its weights download from a stable facexlib release through our confirmed
download flow (no silent fetch). The composite is the same shape as the
rotation pass: align → mask → warp back → blend. Best-effort: any failure
falls back to the plain swap, so enabling it can never break a render.

Note: handles hair/glasses/hat/neck/boundary well (those are distinct parse
classes); arbitrary occluders like a hand over the cheek (parsed as skin) are a
follow-up needing a dedicated occluder model.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from sinner2.pipeline.model_cache import get_models_dir, model_present
from sinner2.types import Frame


class FaceParser(str, Enum):
    """Which facexlib face-parser drives occlusion masking."""

    BISENET = "bisenet"    # accurate boundaries, heavier
    PARSENET = "parsenet"  # lighter/faster (GFPGAN's default parser)


@dataclass(frozen=True)
class _ParserSpec:
    filename: str
    # Per-parser input normalization (they differ): bisenet expects ImageNet
    # stats; parsenet expects (0.5, 0.5, 0.5) → [-1, 1].
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


_PARSER_SPECS: dict[FaceParser, _ParserSpec] = {
    FaceParser.BISENET: _ParserSpec(
        "parsing_bisenet.pth", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ),
    FaceParser.PARSENET: _ParserSpec(
        "parsing_parsenet.pth", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    ),
}


def parser_model_file(parser: FaceParser) -> str:
    """The weights filename for a parser (so the GUI can confirm + download it
    before enabling occlusion)."""
    return _PARSER_SPECS[parser].filename


_ALIGN_SIZE = 512
_FEATHER_SIGMA = 5.0
# BiSeNet (face-parsing.PyTorch) class ids that ARE the face to swap: skin,
# brows, eyes, nose, mouth, lips. Everything else — hair(17), hat(18),
# glasses(6), ears(7-9), neck(14-15), cloth(16), background(0) — keeps the
# original, which is exactly the occlusion behaviour we want.
_FACE_CLASSES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 10, 11, 12, 13})

# ArcFace 5-point template (112 space), used to align faces to _ALIGN_SIZE.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def _align_matrix(kps: np.ndarray) -> np.ndarray:
    template = _ARCFACE_DST * (_ALIGN_SIZE / 112.0)
    m, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, np.float32), template, method=cv2.LMEDS
    )
    return m


class OcclusionMasker:
    """facexlib face-parse → a facial-region mask. torch model (not thread-safe;
    the swapper serializes calls)."""

    def __init__(
        self, parser: FaceParser = FaceParser.BISENET, device: str = "auto"
    ) -> None:
        self._spec = _PARSER_SPECS[parser]
        self._parser = parser
        self._device_str = device
        self._device: Any = None
        self._model: Any = None

    def setup(self) -> None:
        from facexlib.parsing import init_parsing_model

        from sinner2.config.execution import resolve_torch_device

        if not model_present(self._spec.filename):
            # The GUI ensures the model (with a download confirmation) before
            # enabling occlusion — never let facexlib fetch it silently.
            raise FileNotFoundError(
                f"face-parsing model not found: "
                f"{get_models_dir() / self._spec.filename}"
            )
        self._device = resolve_torch_device(self._device_str)
        self._model = init_parsing_model(
            model_name=self._parser.value,
            device=self._device,
            model_rootpath=str(get_models_dir()),
        )
        self._model.eval()

    def face_mask(self, aligned_bgr: Frame) -> np.ndarray:
        """512×512 float mask (1 = facial region) for an aligned BGR crop."""
        import torch

        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(self._spec.mean, np.float32)
        std = np.array(self._spec.std, np.float32)
        chw = np.ascontiguousarray(
            ((rgb - mean) / std).transpose(2, 0, 1), dtype=np.float32
        )
        t = torch.from_numpy(chw)[None].to(self._device)
        with torch.no_grad():
            out = self._model(t)[0]
        classes = out.argmax(dim=1).squeeze().cpu().numpy()
        return np.isin(classes, list(_FACE_CLASSES)).astype(np.float32)


def apply_occlusion(
    before: Frame, swapped: Frame, face: Any, masker: Any
) -> Frame:
    """Keep only the facial-region pixels from `swapped`; the rest reverts to
    `before` (the pre-swap frame) — so occluders stay original. Best-effort:
    returns `swapped` unchanged on any error."""
    try:
        m = _align_matrix(face.kps)
        aligned = cv2.warpAffine(before, m, (_ALIGN_SIZE, _ALIGN_SIZE))
        mask = masker.face_mask(aligned)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=_FEATHER_SIGMA)
        m_inv = cv2.invertAffineTransform(m)
        h, w = before.shape[:2]
        alpha = cv2.warpAffine(mask, m_inv, (w, h))[..., None]
        blended = (
            swapped.astype(np.float32) * alpha + before.astype(np.float32) * (1.0 - alpha)
        )
        return blended.astype(np.uint8)
    except Exception:
        return swapped
