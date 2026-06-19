"""Canonical (token, label) catalogs for the processor combo boxes.

Single source of truth shared by the LIVE settings panel (`QProcessorControls`)
and the BATCH task form (`batch_task_dialog`) so the two can't drift. Each pair
binds the enum's `.value` — the token persisted in settings.json / a BatchTask —
to its display label, defined ONCE here. A rename of an enum value updates both
forms automatically instead of silently leaving the batch form writing a stale
token into a saved task.

List order IS the combo order. The model catalogs are `(token, label)`; the
rotation list stays `(label, token)` to match its historical combo order.
"""
from __future__ import annotations

from sinner2.pipeline.detectors import DetectorModel
from sinner2.pipeline.processors.face_enhancer import EnhancerModel
from sinner2.pipeline.processors.face_swapper import RotationAngleSource, SwapperModel
from sinner2.pipeline.processors.occlusion import (
    FaceParser,
    OccluderModel,
    OcclusionMaskMode,
)
from sinner2.pipeline.processors.upscaler import UpscalerModel

SWAPPER_MODELS: list[tuple[str, str]] = [
    (SwapperModel.INSWAPPER_128.value, "inswapper_128 (default, InsightFace)"),
    (SwapperModel.RESWAPPER_128.value, "ReSwapper 128 (open reproduction)"),
    (SwapperModel.GHOST_1_256.value, "Ghost 1 (256)"),
    (SwapperModel.GHOST_2_256.value, "Ghost 2 (256)"),
    (SwapperModel.GHOST_3_256.value, "Ghost 3 (256, heaviest)"),
    (SwapperModel.SIMSWAP_256.value, "SimSwap (256, non-commercial)"),
    (SwapperModel.UNIFACE_256.value, "UniFace (256, pose-aware)"),
    (SwapperModel.HYPERSWAP_1A_256.value, "Hyperswap 1a (256, newest)"),
    (SwapperModel.HYPERSWAP_1B_256.value, "Hyperswap 1b (256, newest)"),
]

DETECTOR_MODELS: list[tuple[str, str]] = [
    (DetectorModel.BUFFALO_L.value, "buffalo_l (full pack, gender + pose)"),
    (DetectorModel.YOLOFACE.value, "YOLOFace 8n (fast, detection-only)"),
    (DetectorModel.SCRFD_2_5G.value, "SCRFD 2.5g (fast, detection-only)"),
]

ENHANCER_MODELS: list[tuple[str, str]] = [
    (EnhancerModel.GFPGAN.value, "GFPGAN (whole-frame, Upscale knob)"),
    (EnhancerModel.GFPGAN_ONNX.value, "GFPGAN (ONNX, much faster)"),
    (EnhancerModel.CODEFORMER.value, "CodeFormer (ONNX, fidelity knob)"),
    (EnhancerModel.GPEN_512.value, "GPEN-512 (ONNX, more detail)"),
    (EnhancerModel.GPEN_1024.value, "GPEN-1024 (ONNX, higher-res)"),
    (EnhancerModel.GPEN_2048.value, "GPEN-2048 (ONNX, highest-res; heavy)"),
    (EnhancerModel.RESTOREFORMER_PP.value, "RestoreFormer++ (ONNX)"),
]

UPSCALER_MODELS: list[tuple[str, str]] = [
    (UpscalerModel.GENERAL_X4V3.value, "Real-ESRGAN general x4 v3 (fast, small)"),
    (UpscalerModel.X4PLUS.value, "Real-ESRGAN x4plus (higher quality, heavy)"),
    (UpscalerModel.X2PLUS.value, "Real-ESRGAN x2plus"),
    (UpscalerModel.SWINIR_M.value, "SwinIR x4 (transformer, sharp, slow)"),
    (UpscalerModel.HAT_X4.value, "HAT x4 (ONNX, very slow — stills only)"),
    (UpscalerModel.ULTRASHARP_X4.value, "4x-UltraSharp (ONNX)"),
    (UpscalerModel.SPAN_X4.value, "SPAN x4 (ONNX, fast)"),
    (UpscalerModel.REAL_ESRGAN_X4_FP16.value, "Real-ESRGAN x4 fp16 (ONNX)"),
    (UpscalerModel.REAL_ESRGAN_X2_FP16.value, "Real-ESRGAN x2 fp16 (ONNX)"),
]

OCCLUSION_PARSERS: list[tuple[str, str]] = [
    (FaceParser.BISENET.value, "BiSeNet (torch, sharper)"),
    (FaceParser.PARSENET.value, "ParseNet (torch, GFPGAN default)"),
    (FaceParser.BISENET_ONNX_34.value, "BiSeNet-34 (ONNX, parallel workers)"),
    (FaceParser.BISENET_ONNX_18.value, "BiSeNet-18 (ONNX, parallel + faster)"),
]

OCCLUSION_MODES: list[tuple[str, str]] = [
    (OcclusionMaskMode.REGION.value, "Region (face parser)"),
    (OcclusionMaskMode.OCCLUDER.value, "Occluder (XSeg — sees hands/objects)"),
    (OcclusionMaskMode.BOTH.value, "Both (strictest)"),
]

OCCLUDER_MODELS: list[tuple[str, str]] = [
    (OccluderModel.XSEG_1.value, "XSeg 1"),
    (OccluderModel.XSEG_2.value, "XSeg 2"),
    (OccluderModel.XSEG_3.value, "XSeg 3"),
    (OccluderModel.XSEG_MANY.value, "XSeg all three (strictest, 3x cost)"),
    (OccluderModel.DEPTH.value, "Depth (experimental, closer-than-face)"),
]

ROTATION_SOURCES: list[tuple[str, str]] = [
    ("Eye keypoints", RotationAngleSource.KEYPOINTS.value),
    ("3D pose estimate", RotationAngleSource.POSE.value),
    ("2dfan4 landmarks", RotationAngleSource.LANDMARK_68.value),
]
