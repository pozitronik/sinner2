"""Human-facing metadata for every downloadable model.

`model_cache.MODEL_SOURCES` only maps filename → URL. This catalog adds the
category / display name / description / approximate download size / license that
the Models management tab needs. Sizes are hardcoded (offline + instant); they
were taken from the upstream Content-Length and may drift if a weight is
re-uploaded. The on-disk size of an installed model is read live from disk.

A test asserts the catalog and MODEL_SOURCES cover exactly the same files, so a
newly-registered model can't silently miss its metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelCategory(str, Enum):
    SWAPPER = "Face swapper"
    ENHANCER = "Face enhancer"
    DETECTOR = "Face detector"
    UPSCALER = "Frame upscaler"
    MASK_PARSER = "Mask parser"
    CONVERTER = "Embedding converter"


@dataclass(frozen=True)
class ModelInfo:
    filename: str
    display_name: str
    category: ModelCategory
    description: str
    size_mb: int           # approximate download size
    license: str = ""      # e.g. "CC-BY-NC (non-commercial)"


_C = ModelCategory


def _e(*args, **kw) -> ModelInfo:
    return ModelInfo(*args, **kw)


MODEL_CATALOG: dict[str, ModelInfo] = {
    # ---- Swappers ----
    "inswapper_128.onnx": _e(
        "inswapper_128.onnx", "inswapper_128", _C.SWAPPER,
        "Default face-swap model (InsightFace, 128px).", 529),
    "reswapper_128.onnx": _e(
        "reswapper_128.onnx", "ReSwapper 128", _C.SWAPPER,
        "Open reproduction of inswapper (128px).", 529),
    "ghost_1_256.onnx": _e(
        "ghost_1_256.onnx", "Ghost 1 (256)", _C.SWAPPER,
        "Ghost face-swap, 256px — variant 1.", 491),
    "ghost_2_256.onnx": _e(
        "ghost_2_256.onnx", "Ghost 2 (256)", _C.SWAPPER,
        "Ghost face-swap, 256px — variant 2.", 705),
    "ghost_3_256.onnx": _e(
        "ghost_3_256.onnx", "Ghost 3 (256)", _C.SWAPPER,
        "Ghost face-swap, 256px — variant 3 (heaviest).", 816),
    "simswap_256.onnx": _e(
        "simswap_256.onnx", "SimSwap (256)", _C.SWAPPER,
        "SimSwap face-swap, 256px.", 210, "CC-BY-NC (non-commercial)"),
    "uniface_256.onnx": _e(
        "uniface_256.onnx", "UniFace (256)", _C.SWAPPER,
        "UniFace pose-aware face-swap, 256px.", 388),
    # ---- Embedding converters (companions to ghost / simswap) ----
    "crossface_ghost.onnx": _e(
        "crossface_ghost.onnx", "CrossFace (Ghost)", _C.CONVERTER,
        "ArcFace→Ghost embedding converter — needed by the Ghost swappers.", 21),
    "crossface_simswap.onnx": _e(
        "crossface_simswap.onnx", "CrossFace (SimSwap)", _C.CONVERTER,
        "ArcFace→SimSwap embedding converter — needed by SimSwap.", 21),
    # ---- Enhancers ----
    "GFPGANv1.4.pth": _e(
        "GFPGANv1.4.pth", "GFPGAN v1.4", _C.ENHANCER,
        "Default face enhancer (whole-frame restore + upscale knob).", 332),
    "gfpgan_1.4.onnx": _e(
        "gfpgan_1.4.onnx", "GFPGAN v1.4 (ONNX)", _C.ENHANCER,
        "GFPGAN as ONNX — same restorer, much faster pipeline.", 325),
    "codeformer.onnx": _e(
        "codeformer.onnx", "CodeFormer", _C.ENHANCER,
        "Face restorer with a fidelity (w) knob.", 360),
    "gpen_bfr_512.onnx": _e(
        "gpen_bfr_512.onnx", "GPEN-512", _C.ENHANCER,
        "GPEN blind-face restorer, 512px.", 271),
    "gpen_bfr_1024.onnx": _e(
        "gpen_bfr_1024.onnx", "GPEN-1024", _C.ENHANCER,
        "GPEN blind-face restorer, 1024px (higher detail).", 272),
    "gpen_bfr_2048.onnx": _e(
        "gpen_bfr_2048.onnx", "GPEN-2048", _C.ENHANCER,
        "GPEN blind-face restorer, 2048px (heaviest).", 272),
    "restoreformer_plus_plus.onnx": _e(
        "restoreformer_plus_plus.onnx", "RestoreFormer++", _C.ENHANCER,
        "Transformer face restorer.", 281),
    # ---- Detectors ----
    "yoloface_8n.onnx": _e(
        "yoloface_8n.onnx", "YOLOFace 8n", _C.DETECTOR,
        "Fast detection-only face detector.", 12),
    "scrfd_2.5g.onnx": _e(
        "scrfd_2.5g.onnx", "SCRFD 2.5g", _C.DETECTOR,
        "Fast detection-only face detector (SCRFD).", 3),
    # ---- Upscalers ----
    "realesr-general-x4v3.pth": _e(
        "realesr-general-x4v3.pth", "Real-ESRGAN general x4 v3", _C.UPSCALER,
        "General super-resolution x4 (small, fast).", 5),
    "RealESRGAN_x4plus.pth": _e(
        "RealESRGAN_x4plus.pth", "Real-ESRGAN x4plus", _C.UPSCALER,
        "Higher-quality super-resolution x4.", 64),
    "RealESRGAN_x2plus.pth": _e(
        "RealESRGAN_x2plus.pth", "Real-ESRGAN x2plus", _C.UPSCALER,
        "Super-resolution x2.", 64),
    "swinir_realsr_m_x4.pth": _e(
        "swinir_realsr_m_x4.pth", "SwinIR x4", _C.UPSCALER,
        "Transformer super-resolution x4 (sharp, slow — batch).", 64),
    "real_hatgan_x4.onnx": _e(
        "real_hatgan_x4.onnx", "HAT x4", _C.UPSCALER,
        "HAT super-resolution x4 — ~24s/frame at FullHD; stills only.", 153),
    "ultra_sharp_x4.onnx": _e(
        "ultra_sharp_x4.onnx", "4x-UltraSharp", _C.UPSCALER,
        "Community ESRGAN-arch super-resolution x4.", 64),
    "span_kendata_x4.onnx": _e(
        "span_kendata_x4.onnx", "SPAN x4", _C.UPSCALER,
        "SPAN super-resolution x4 (tiny + fast).", 2),
    # ---- Mask parsers (occlusion-aware masking) ----
    "parsing_bisenet.pth": _e(
        "parsing_bisenet.pth", "BiSeNet parser", _C.MASK_PARSER,
        "Face parser for occlusion masking (sharper boundaries).", 51),
    "parsing_parsenet.pth": _e(
        "parsing_parsenet.pth", "ParseNet parser", _C.MASK_PARSER,
        "Face parser for occlusion masking (lighter/faster).", 81),
    "bisenet_resnet_34.onnx": _e(
        "bisenet_resnet_34.onnx", "BiSeNet-34 (ONNX)", _C.MASK_PARSER,
        "Face parser as ONNX — workers mask in parallel (no lock).", 89),
    "bisenet_resnet_18.onnx": _e(
        "bisenet_resnet_18.onnx", "BiSeNet-18 (ONNX)", _C.MASK_PARSER,
        "Lighter ONNX face parser — parallel masking, faster.", 50),
    "xseg_1.onnx": _e(
        "xseg_1.onnx", "XSeg occluder 1", _C.MASK_PARSER,
        "Occluder segmentation — masks hands/objects over the face.", 67),
    "xseg_2.onnx": _e(
        "xseg_2.onnx", "XSeg occluder 2", _C.MASK_PARSER,
        "Occluder segmentation — alternative XSeg training.", 67),
    "xseg_3.onnx": _e(
        "xseg_3.onnx", "XSeg occluder 3", _C.MASK_PARSER,
        "Occluder segmentation — alternative XSeg training.", 67),
    "depth_anything_v2_small.onnx": _e(
        "depth_anything_v2_small.onnx", "Depth occluder (exp.)", _C.MASK_PARSER,
        "Depth-based occluder — masks anything closer than the face.", 95),
}


def model_info(filename: str) -> ModelInfo | None:
    return MODEL_CATALOG.get(filename)


def catalog_entries() -> list[ModelInfo]:
    """All entries, ordered by category then display name (the tab's order)."""
    cats = list(ModelCategory)
    return sorted(
        MODEL_CATALOG.values(),
        key=lambda m: (cats.index(m.category), m.display_name.lower()),
    )
