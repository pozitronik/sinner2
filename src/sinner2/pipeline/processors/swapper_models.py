"""Face-swapper model registry + a generic ONNX swap backend.

sinner2 ships inswapper_128 as the default swapper, but several openly-sourced
alternatives exist. Two integration tiers:

* **insightface-compatible** (inswapper_128, ReSwapper) — loaded through
  ``insightface.model_zoo.get_model`` exactly like the original; the I/O is
  fully encapsulated by insightface's ``INSwapper`` wrapper.
* **facefusion-style** (Ghost, SimSwap, UniFace) — a single ONNX restore net
  with the alignment / normalization / source-embedding handling living in the
  caller. ``GenericOnnxSwapper`` reimplements facefusion's verified pipeline so
  these slot into ``FaceSwapper`` behind the SAME ``.get()`` signature insightface
  exposes (so rotation compensation + the plain path are untouched).

The per-model contract (template, crop size, input mean/std, how the source
identity vector is built, whether the output is de-normalized) was taken
verbatim from facefusion's source — see the spec table below. Wrong values here
mean a misaligned or discolored swap, so they are not guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from sinner2.config.execution import DEFAULT_ONNX_PROVIDERS
from sinner2.pipeline.model_cache import (
    get_onnx_session,
    record_actual_providers,
    release_onnx_session,
)
from sinner2.types import Frame


class SwapperModel(str, Enum):
    """Selectable face-swap model. Values are stable tokens (settings + cache
    key); the first two load via insightface, the rest via GenericOnnxSwapper."""

    INSWAPPER_128 = "inswapper_128"   # default; InsightFace (128px)
    RESWAPPER_128 = "reswapper_128"   # clean-room inswapper reproduction (128px)
    GHOST_1_256 = "ghost_1_256"       # ai-forever Ghost (256px)
    GHOST_2_256 = "ghost_2_256"
    GHOST_3_256 = "ghost_3_256"
    SIMSWAP_256 = "simswap_256"       # SimSwap (256px) — CC-BY-NC
    UNIFACE_256 = "uniface_256"       # UniFace (256px, pose-aware)


# Warp templates — normalized 5-point destination landmarks (left eye, right
# eye, nose, left mouth, right mouth), VERBATIM from facefusion's
# WARP_TEMPLATE_SET. Scaled by the crop size at warp time.
_TEMPLATES: dict[str, np.ndarray] = {
    "arcface_112_v1": np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500],
        ],
        np.float32,
    ),
    "arcface_128": np.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453],
        ],
        np.float32,
    ),
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ],
        np.float32,
    ),
}


@dataclass(frozen=True)
class SwapperSpec:
    """Everything FaceSwapper needs to load and drive one model."""

    model_file: str
    insightface: bool = False
    # Generic-backend fields (ignored when insightface is True):
    template: str = ""
    size: int = 0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # How the ONNX 'source' input is built: "embedding" via a crossface
    # converter (ghost/simswap), or "frame" from an aligned source crop
    # (uniface). normalize_embedding only applies to the embedding modes.
    source_mode: str = ""              # "ghost" | "simswap" | "uniface"
    denorm_output: bool = False        # de-normalize (x*std+mean) before clip
    converter_file: str | None = None  # crossface_* ONNX (embedding modes)


# Per-model contract, verified against facefusion master. Ghost variants share
# everything but the weights file (and the crossface_ghost converter).
_SPECS: dict[SwapperModel, SwapperSpec] = {
    SwapperModel.INSWAPPER_128: SwapperSpec("inswapper_128.onnx", insightface=True),
    SwapperModel.RESWAPPER_128: SwapperSpec("reswapper_128.onnx", insightface=True),
    SwapperModel.GHOST_1_256: SwapperSpec(
        "ghost_1_256.onnx", template="arcface_112_v1", size=256,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        source_mode="ghost", denorm_output=True,
        converter_file="crossface_ghost.onnx",
    ),
    SwapperModel.GHOST_2_256: SwapperSpec(
        "ghost_2_256.onnx", template="arcface_112_v1", size=256,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        source_mode="ghost", denorm_output=True,
        converter_file="crossface_ghost.onnx",
    ),
    SwapperModel.GHOST_3_256: SwapperSpec(
        "ghost_3_256.onnx", template="arcface_112_v1", size=256,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        source_mode="ghost", denorm_output=True,
        converter_file="crossface_ghost.onnx",
    ),
    SwapperModel.SIMSWAP_256: SwapperSpec(
        "simswap_256.onnx", template="arcface_112_v1", size=256,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        source_mode="simswap", denorm_output=False,
        converter_file="crossface_simswap.onnx",
    ),
    SwapperModel.UNIFACE_256: SwapperSpec(
        "uniface_256.onnx", template="ffhq_512", size=256,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        source_mode="uniface", denorm_output=True,
        converter_file=None,
    ),
}


def get_spec(model: SwapperModel) -> SwapperSpec:
    return _SPECS[model]


def is_insightface_model(model: SwapperModel) -> bool:
    return _SPECS[model].insightface


def model_filename(model: SwapperModel) -> str:
    """The primary ONNX weight file for the model."""
    return _SPECS[model].model_file


def model_files(model: SwapperModel) -> list[str]:
    """Every model file the selection needs present before it can run —
    the weights plus the crossface embedding converter for ghost/simswap.
    Drives the GUI's download-confirmation."""
    spec = _SPECS[model]
    files = [spec.model_file]
    if spec.converter_file:
        files.append(spec.converter_file)
    return files


# ---- generic-pipeline helpers (verbatim port of facefusion's math) ----


def _warp_face(frame: Frame, kps: Any, template: np.ndarray, size: int) -> tuple[Frame, np.ndarray]:
    """Align a face to `size`×`size` against `template` (normalized landmarks
    scaled by the crop size). Returns the crop and the affine matrix used."""
    dst = template * size
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, np.float32), dst,
        method=cv2.RANSAC, ransacReprojThreshold=100,
    )
    crop = cv2.warpAffine(
        frame, matrix, (size, size),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA,
    )
    return crop, matrix


def _box_mask(size: int, blur: float = 0.3) -> np.ndarray:
    """facefusion's default box mask: a square of ones with feathered edges
    (no padding). `blur` is the fraction used to size the Gaussian feather."""
    blur_amount = int(size * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)
    mask: np.ndarray = np.ones((size, size), np.float32)
    mask[:blur_area, :] = 0
    mask[-blur_area:, :] = 0
    mask[:, :blur_area] = 0
    mask[:, -blur_area:] = 0
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
    return mask


def _paste_back(frame: Frame, crop: Frame, mask: np.ndarray, matrix: np.ndarray) -> Frame:
    """Warp the swapped crop back by the inverse affine and alpha-blend it in
    with the feathered mask (BORDER_REPLICATE to avoid black-fill halos)."""
    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(matrix)
    inv_mask = cv2.warpAffine(mask, inv, (w, h)).clip(0.0, 1.0)[..., None]
    inv_crop = cv2.warpAffine(crop, inv, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = frame.astype(np.float32) * (1.0 - inv_mask) + inv_crop.astype(np.float32) * inv_mask
    return out.astype(np.uint8)


class GenericOnnxSwapper:
    """facefusion-style ONNX swap backend for ghost / simswap / uniface.

    Exposes ``get(img, target_face, source_face=None, paste_back=True)`` so it's
    a drop-in for insightface's ``INSwapper`` inside FaceSwapper and the rotation
    path. The source identity is computed ONCE via ``prepare_source`` and reused
    (the caller always passes the same source face); the runtime ``source_face``
    arg is accepted for signature parity but ignored.
    """

    def __init__(self, spec: SwapperSpec, providers: list[str] | None = None) -> None:
        self._spec = spec
        # None → platform default; an explicit [] stays empty (user picked no
        # providers → ORT runs on CPU), not a substituted GPU default.
        self._providers = (
            list(DEFAULT_ONNX_PROVIDERS) if providers is None else list(providers)
        )
        self._template = _TEMPLATES[spec.template]
        self._mean = np.array(spec.mean, np.float32)
        self._std = np.array(spec.std, np.float32)
        self._session: Any = None
        self._converter: Any = None
        self._mask = _box_mask(spec.size)
        self._source_input: np.ndarray | None = None

    def setup(self) -> None:
        self._session = get_onnx_session(self._spec.model_file, providers=self._providers)
        try:
            record_actual_providers(self._session.get_providers())
        except Exception:
            pass
        if self._spec.converter_file:
            self._converter = get_onnx_session(
                self._spec.converter_file, providers=self._providers
            )

    def prepare_source(self, source_img: Frame, source_face: Any) -> None:
        """Build the ONNX 'source' input from the source face — a converted
        ArcFace embedding (ghost/simswap) or an aligned source crop (uniface)."""
        mode = self._spec.source_mode
        if mode == "uniface":
            crop, _ = _warp_face(
                source_img, source_face.kps, _TEMPLATES["ffhq_512"], self._spec.size
            )
            rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
            self._source_input = np.ascontiguousarray(
                rgb.transpose(2, 0, 1)[None], np.float32
            )
            return
        if self._converter is None:
            raise RuntimeError(f"converter not loaded for source mode {mode!r}")
        embedding = np.asarray(source_face.embedding, np.float32).reshape(1, 512)
        converted = self._converter.run(None, {"input": embedding})[0].ravel()
        if mode == "simswap":  # simswap uses the L2-normalized converted vector
            converted = converted / np.linalg.norm(converted)
        # ghost uses the raw converted vector (no normalization).
        self._source_input = converted.reshape(1, -1).astype(np.float32)

    def _prepare_crop(self, crop: Frame) -> np.ndarray:
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], np.float32)

    def _normalize_output(self, out: np.ndarray) -> Frame:
        crop = out[0].transpose(1, 2, 0)  # strip batch, CHW -> HWC
        if self._spec.denorm_output:
            crop = crop * self._std + self._mean
        crop = np.clip(crop, 0.0, 1.0)
        crop = crop[:, :, ::-1] * 255.0  # RGB -> BGR
        return crop.astype(np.uint8)

    def get(
        self, img: Frame, target_face: Any, source_face: Any = None, paste_back: bool = True
    ) -> Frame:
        if self._session is None or self._source_input is None:
            raise RuntimeError("GenericOnnxSwapper.get called before setup()/prepare_source()")
        crop, matrix = _warp_face(img, target_face.kps, self._template, self._spec.size)
        blob = self._prepare_crop(crop)
        out = self._session.run(
            None, {"source": self._source_input, "target": blob}
        )[0]
        swapped = self._normalize_output(out)
        if not paste_back:
            return swapped
        return _paste_back(img, swapped, self._mask, matrix)

    def release(self) -> None:
        """Drop session/converter refs and evict the (exclusive) model + its
        crossface converter from the shared ONNX cache, so disabling/switching
        the swapper frees their VRAM instead of leaving them resident."""
        self._session = None
        self._converter = None
        self._source_input = None
        release_onnx_session(self._spec.model_file)
        if self._spec.converter_file:
            release_onnx_session(self._spec.converter_file)
