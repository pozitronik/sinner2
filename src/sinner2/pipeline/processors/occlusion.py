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
        # Whether the resolved device is CUDA, so release() knows to hand the
        # parser's GPU memory back to the driver.
        self._device_is_cuda = False

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
        self._device_is_cuda = self._device.type == "cuda"
        self._model = init_parsing_model(
            model_name=self._parser.value,
            device=self._device,
            model_rootpath=str(get_models_dir()),
        )
        self._model.eval()

    def release(self) -> None:
        """Drop the parser model and hand its VRAM back to the driver — torch's
        caching allocator keeps the freed blocks reserved otherwise, so a chain
        rebuild with occlusion on would stack a fresh parser's GPU memory on top
        of the old one's until process exit (mirrors FaceEnhancer.release)."""
        self._model = None
        if self._device_is_cuda:
            import torch

            torch.cuda.empty_cache()

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
    returns `swapped` unchanged on any error.

    The mask warp + float blend run only inside the bounding box of the warped
    aligned square (+2px interpolation bleed) — outside it the warped alpha is
    identically 0, so the result is `before` there by definition, and the full-
    frame float blend this replaces was ~60-80ms of CPU per face at FullHD
    (the same pattern paste_back fixed; see scripts/enhancer_bench.py)."""
    try:
        m = _align_matrix(face.kps)
        aligned = cv2.warpAffine(before, m, (_ALIGN_SIZE, _ALIGN_SIZE))
        mask = masker.face_mask(aligned)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=_FEATHER_SIGMA)
        m_inv = cv2.invertAffineTransform(m)
        h, w = before.shape[:2]
        corners = np.array(
            [[0, 0], [_ALIGN_SIZE, 0], [_ALIGN_SIZE, _ALIGN_SIZE], [0, _ALIGN_SIZE]],
            np.float32,
        ).reshape(1, 4, 2)
        warped = cv2.transform(corners, m_inv)[0]
        x0 = max(int(np.floor(float(warped[:, 0].min()))) - 2, 0)
        y0 = max(int(np.floor(float(warped[:, 1].min()))) - 2, 0)
        x1 = min(int(np.ceil(float(warped[:, 0].max()))) + 2, w)
        y1 = min(int(np.ceil(float(warped[:, 1].max()))) + 2, h)
        if x0 >= x1 or y0 >= y1:
            return before.copy()  # aligned square fully off-frame → all reverts
        m_roi = m_inv.copy()
        m_roi[0, 2] -= x0
        m_roi[1, 2] -= y0
        alpha = cv2.warpAffine(mask, m_roi, (x1 - x0, y1 - y0))[..., None]
        # Outside the warped square alpha is 0 → the blend yields `before`
        # everywhere outside the ROI, so start from a copy of `before`.
        out = before.copy()
        roi_sw = swapped[y0:y1, x0:x1].astype(np.float32)
        roi_bf = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (
            roi_sw * alpha + roi_bf * (1.0 - alpha)
        ).astype(np.uint8)
        return out
    except Exception:
        return swapped
