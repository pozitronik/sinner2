import contextlib
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from sinner2.config.base import SinnerBaseModel
from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.model_cache import get_model_path
from sinner2.pipeline.processors.bfr_onnx import PlainBfrBackend
from sinner2.pipeline.processors.codeformer import CodeFormerBackend
from sinner2.pipeline.processors.codeformer import MODEL_FILE as _CODEFORMER_FILE
from sinner2.pipeline.processors.face_swapper_types import RotationAngleSource
from sinner2.pipeline.processors.rotation_compensation import (
    compute_roll,
    enhance_with_uprighting,
)
from sinner2.types import Frame


class EnhancerModel(str, Enum):
    """Face-restoration backend."""

    GFPGAN = "gfpgan"          # whole-frame restore; an Upscale knob
    CODEFORMER = "codeformer"  # ONNX; a fidelity (w) knob, no upscale
    GPEN_512 = "gpen_512"      # ONNX; plain BFR-512, no knobs (more detail)
    GPEN_1024 = "gpen_1024"    # ONNX; plain BFR-1024 (higher-res restore)
    GPEN_2048 = "gpen_2048"    # ONNX; plain BFR-2048 (highest-res; heavy)
    RESTOREFORMER_PP = "restoreformer_pp"  # ONNX; transformer restorer, no knobs


# Plain BFR ONNX restorers (no fidelity knob) → their model filenames. Driven by
# the shared PlainBfrBackend, which derives the alignment resolution from each
# model's own declared input shape (512 / 1024 / 2048).
_BFR_MODEL_FILES: dict[EnhancerModel, str] = {
    EnhancerModel.GPEN_512: "gpen_bfr_512.onnx",
    EnhancerModel.GPEN_1024: "gpen_bfr_1024.onnx",
    EnhancerModel.GPEN_2048: "gpen_bfr_2048.onnx",
    EnhancerModel.RESTOREFORMER_PP: "restoreformer_plus_plus.onnx",
}


def enhancer_onnx_model_file(model: EnhancerModel) -> str | None:
    """The ONNX weight file an enhancer needs, or None for GFPGAN (a .pth that
    ships with the required-models set). Used by the GUI to confirm/download the
    weight before the ONNX enhancer is enabled."""
    if model is EnhancerModel.CODEFORMER:
        return _CODEFORMER_FILE
    return _BFR_MODEL_FILES.get(model)


class FaceEnhancerParams(SinnerBaseModel):
    model: EnhancerModel = Field(default=EnhancerModel.GFPGAN)
    upscale: int = Field(default=1, ge=1, le=4, description="Output upscale factor (GFPGAN)")
    only_center_face: bool = Field(
        default=False, description="Enhance only the center face"
    )
    fp16: bool = Field(
        default=True,
        description=(
            "GFPGAN half precision: halves the generator's VRAM and uses "
            "tensor-core convs (faster). CUDA only; ignored by CodeFormer (ONNX)."
        ),
    )
    codeformer_fidelity: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="CodeFormer fidelity w: 0 = max restore, 1 = max fidelity",
    )
    # Rotation compensation — shared config with the swapper (same UI controls).
    # GFPGAN has no rotation handling of its own and mangles tilted faces;
    # uprighting a crop and re-enhancing it fixes them. Output-affecting.
    rotation_compensation: bool = Field(default=True)
    rotation_threshold_deg: int = Field(default=15, ge=0, le=90)
    rotation_redetect: bool = Field(default=True)  # unused here; kept for parity
    rotation_angle_source: RotationAngleSource = Field(
        default=RotationAngleSource.POSE
    )


_MODEL_FILE = "GFPGANv1.4.pth"


def _load_restorer(path: Path, upscale: int, device: Any, fp16: bool = False) -> Any:
    """Loader indirection so tests can stub the gfpgan call cheaply.
    ``device`` is a torch.device chosen by the caller.

    ``fp16`` halves ONLY the GFPGAN generator's weights — the bulk of each
    worker's VRAM — for less memory + faster tensor-core convs. The face
    helper's detector/parser stay fp32 for stability; the fp32 input crop is
    reconciled with the half-precision weights by the torch.autocast wrapper in
    process(). The "clean" arch is pure PyTorch (no custom fp32-only CUDA ops),
    so it half()s safely; any build that doesn't expose `.gfpgan` falls back to
    fp32 rather than failing the load."""
    from gfpgan import GFPGANer

    restorer = GFPGANer(
        model_path=str(path),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        device=device,
    )
    if fp16:
        try:
            restorer.gfpgan.half()
        except Exception:
            pass
    return restorer


class FaceEnhancer:
    name = "FaceEnhancer"
    thread_safe = False  # GFPGAN mutates torch state — each worker needs its own

    def __init__(
        self,
        params: FaceEnhancerParams | None = None,
        device: str = "auto",
    ) -> None:
        self._params = params or FaceEnhancerParams()
        # Torch device from the enhancer's TorchExecution profile
        # ("auto"/"cpu"/"cuda"/"cuda:N"); resolved at setup().
        self._device = device
        self._restorer: Any = None
        # CodeFormer backend (ONNX) when that model is selected; None for GFPGAN.
        self._codeformer: CodeFormerBackend | None = None
        # Plain BFR backend (ONNX) for GPEN / RestoreFormer++; None otherwise.
        self._bfr: PlainBfrBackend | None = None
        # Detector for rotation compensation (finds tilted faces to upright).
        # Built in setup(); reuses the shared insightface model.
        self._analyser: FaceAnalyser | None = None
        # Set at setup(): whether the resolved device is CUDA, so release()
        # knows whether to hand the model's VRAM back to the driver.
        self._device_is_cuda = False
        # Set at setup(): whether GFPGAN runs in half precision. Only true when
        # the user enabled fp16 AND the device is CUDA (fp16 is a no-op / slow
        # on CPU). Gates the autocast wrapper around restorer.enhance().
        self._fp16 = False
        # GFPGAN's enhance() mutates torch state, so it isn't thread-safe.
        # Both run modes now give each worker its OWN instance (batch via the
        # stage runner's per-worker pool, realtime via PerWorkerProcessor), so
        # this lock is effectively never contended — kept as a safety net for
        # any path that might share an instance.
        self._enhance_lock = threading.Lock()

    def setup(self) -> None:
        from sinner2.config.execution import resolve_torch_device

        if self._params.model is EnhancerModel.CODEFORMER:
            # ONNX restorer — its own (shared, thread-safe) session + detector.
            self._codeformer = CodeFormerBackend(
                fidelity=self._params.codeformer_fidelity
            )
            self._codeformer.setup()
        elif self._params.model in _BFR_MODEL_FILES:
            # GPEN / RestoreFormer++ — plain BFR ONNX (no knobs), shared session.
            self._bfr = PlainBfrBackend(_BFR_MODEL_FILES[self._params.model])
            self._bfr.setup()
        else:
            # GFPGAN is PyTorch, so its device is torch's CUDA — independent of
            # the swapper's ONNX providers. A CPU fallback is far slower, so
            # surface it loudly instead of letting it look like a "slow GPU".
            device = resolve_torch_device(self._device)
            self._device_is_cuda = device.type == "cuda"
            # fp16 only helps on CUDA (tensor cores); on CPU it's slow/unsupported.
            self._fp16 = self._params.fp16 and self._device_is_cuda
            if device.type == "cuda":
                print(
                    f"[sinner2] FaceEnhancer (GFPGAN) device: {device} "
                    f"(fp16={self._fp16})",
                    file=sys.stderr,
                )
            else:
                print(
                    "[sinner2] WARNING: FaceEnhancer (GFPGAN) running on CPU "
                    f"(requested device={self._device!r}).",
                    file=sys.stderr,
                )
            self._restorer = _load_restorer(
                get_model_path(_MODEL_FILE), self._params.upscale, device,
                fp16=self._fp16,
            )
        if self._params.rotation_compensation:
            self._analyser = FaceAnalyser()

    def _gfpgan_autocast(self) -> Any:
        """fp16 autocast context for the GFPGAN forward, or a no-op context when
        fp16 is off. Casts the fp32 input crop to half to match the half-
        precision generator weights and routes the convs through tensor cores."""
        if not self._fp16:
            return contextlib.nullcontext()
        import torch

        return torch.autocast("cuda", dtype=torch.float16)

    def process(self, frame: Frame) -> Frame:
        # Local snapshots — release() can null these concurrently; holding a
        # local ref keeps the active backend alive for this call.
        restorer = self._restorer
        codeformer = self._codeformer
        bfr = self._bfr
        if restorer is None and codeformer is None and bfr is None:
            raise RuntimeError("FaceEnhancer.process called before setup()")

        def enhance_image(img: Frame, only_center: bool) -> Frame:
            if codeformer is not None:
                return codeformer.enhance(img)  # ONNX session is thread-safe
            if bfr is not None:
                return bfr.enhance(img)  # ONNX session is thread-safe
            with self._enhance_lock, self._gfpgan_autocast():
                _, _, out = restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=only_center,
                    paste_back=True,
                )
            return out if out is not None else img

        result = enhance_image(frame, self._params.only_center_face)
        # Only GFPGAN needs the uprighting pass. The ONNX restorers (CodeFormer /
        # GPEN / RestoreFormer) already remove in-plane roll via their per-face
        # estimate_norm alignment, so re-enhancing an uprighted crop is wasted
        # work — a full extra detect + restore + blend (the heaviest op) per
        # tilted face, on by default.
        if (
            restorer is None
            or not self._params.rotation_compensation
            or self._analyser is None
        ):
            return result
        # GFPGAN just mangled any tilted faces (no rotation handling). For each
        # face rolled past the threshold, re-enhance an uprighted crop of the
        # ORIGINAL face and composite it over the cursed result.
        for face in self._analyser.analyse(frame):
            roll = compute_roll(face, self._params.rotation_angle_source)
            if abs(roll) >= self._params.rotation_threshold_deg:
                result = enhance_with_uprighting(
                    result,
                    frame,
                    face,
                    lambda crop: enhance_image(crop, only_center=True),
                    angle_deg=roll,
                )
        return result

    def release(self) -> None:
        self._restorer = None
        if self._codeformer is not None:
            # Evict the CodeFormer ONNX session from VRAM (its own release does
            # the cache eviction) rather than just dropping our reference.
            self._codeformer.release()
            self._codeformer = None
        if self._bfr is not None:
            self._bfr.release()
            self._bfr = None
        self._analyser = None
        # Hand the model's VRAM back to the driver. Torch's caching allocator
        # otherwise keeps the freed blocks reserved, so a realtime worker-count
        # DECREASE (or a chain rebuild) wouldn't visibly free GPU memory —
        # nvidia-smi would still show the per-worker models resident.
        if self._device_is_cuda:
            import torch

            torch.cuda.empty_cache()
