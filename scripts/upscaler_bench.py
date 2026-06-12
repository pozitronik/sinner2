"""Standalone Upscaler (frame super-resolution) performance diagnostic.

The last unexamined processor from the original slowdown report ("frame
enhancer drops FPS to ~1"). At x4 a FullHD frame becomes a 7680x4320 output —
a ~400MB float32 intermediate — so genuine model cost is high; this measures
how the frame budget splits between:

    pre     cvtColor + [0,1] normalize + NCHW (CPU)
    forward the model inference itself (incl. the tile loop where used)
    post    clip + transpose + *255 + uint8 cast + cvtColor (CPU, on the
            x16-area output)

per model, plus the end-to-end Upscaler.process() wall time. HAT has a fixed
256px input -> a FullHD frame is ~40 sequential tile inferences + stitching.

Run on Windows from the project root (GPU required):

    .venv\\Scripts\\python.exe scripts\\upscaler_bench.py <video_or_image> [frames] [models]

Defaults: frames=5 (x4 SR is heavy), models=hat-x4,span-x4,general-x4v3
(the user's selection, the fast ONNX alternative, the torch default).
Paste the whole output back.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sinner2.pipeline.processors import upscaler as up_mod
from sinner2.pipeline.processors.upscaler import (
    Upscaler,
    UpscalerModel,
    UpscalerParams,
    _MODEL_SPECS,
)


def load_frame(media: Path) -> np.ndarray:
    img = cv2.imread(str(media))
    if img is not None:
        return img
    cap = cv2.VideoCapture(str(media))
    if not cap.isOpened():
        raise SystemExit(f"cannot open: {media}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 4:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 4)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"cannot read a frame from: {media}")
    return frame


def _cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def report(label: str, seconds: list[float]) -> float:
    mean = sum(seconds) / len(seconds)
    print(f"    {label:10s} mean {mean * 1000:7.0f}ms")
    return mean


def bench_model(model: UpscalerModel, frame: np.ndarray, n: int) -> None:
    spec = _MODEL_SPECS[model]
    print(f"\n=== {model.value} ({spec.runtime}, x{spec.scale}) ===")
    # Mirror the user's settings: tile=0 (whole frame), fp16 off.
    up = Upscaler(params=UpscalerParams(model=model, tile=0, fp16=False))
    t0 = time.perf_counter()
    up.setup()
    print(f"  setup: {time.perf_counter() - t0:.2f}s")

    # Warmup (CUDA context / cudnn search / ORT first-run) — not counted.
    up.process(frame.copy())

    pre_t: list[float] = []
    fwd_t: list[float] = []
    post_t: list[float] = []
    if spec.runtime == "onnx":
        session = up._session                      # noqa: SLF001
        in_name, out_name = up._in_name, up._out_name  # noqa: SLF001
        fixed = up._onnx_fixed_size                # noqa: SLF001
        tiles = 0
        for _ in range(n):
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
            pre_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            if fixed:
                out = up_mod._onnx_fixed_tile(     # noqa: SLF001
                    session, chw, spec.scale, fixed, in_name, out_name
                )
                h, w = chw.shape[2], chw.shape[3]
                tiles = -(-h // fixed) * -(-w // fixed)
            else:
                out = up_mod._onnx_run(session, chw, in_name, out_name)  # noqa: SLF001
            fwd_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            result = up_mod._finalize_bgr(out[0])  # noqa: SLF001
            post_t.append(time.perf_counter() - t0)
        if tiles:
            print(f"  fixed {fixed}px input -> {tiles} tile inferences/frame")
    else:
        import torch

        model_net = up._model                      # noqa: SLF001
        device = up._device                        # noqa: SLF001
        for _ in range(n):
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = (
                torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                .float().div(255.0).to(device)
            )
            _cuda_sync()
            pre_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = up_mod._run_aligned(         # noqa: SLF001
                    model_net, t, spec.scale, spec.align
                )
            _cuda_sync()
            fwd_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            arr = (
                out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0)
                .mul(255.0).round().to(torch.uint8).cpu().numpy()
            )
            result = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            post_t.append(time.perf_counter() - t0)

    p = report("pre", pre_t)
    f = report("forward", fwd_t)
    q = report("post", post_t)
    total = p + f + q
    print(f"    {'sum':10s} mean {total * 1000:7.0f}ms -> {1 / total:.2f} FPS  "
          f"(pre {p / total * 100:.0f}% / fwd {f / total * 100:.0f}% / "
          f"post {q / total * 100:.0f}%)")
    print(f"  output: {result.shape[1]}x{result.shape[0]}")

    proc_t: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        up.process(frame.copy())
        proc_t.append(time.perf_counter() - t0)
    report("process()", proc_t)
    up.release()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/upscaler_bench.py <video_or_image> "
              "[frames] [models-csv]")
        return 2
    media = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    models = (
        sys.argv[3].split(",") if len(sys.argv) > 3
        else ["hat-x4", "span-x4", "general-x4v3"]
    )
    import torch

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    frame = load_frame(media)
    print(f"frame: {frame.shape[1]}x{frame.shape[0]} from {media.name}; "
          f"{n} frames per model")
    for name in models:
        try:
            bench_model(UpscalerModel(name), frame, n)
        except Exception as exc:  # noqa: BLE001 — report and move to the next model
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    print("\nDone. Paste this whole output back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
