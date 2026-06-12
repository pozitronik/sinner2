"""Standalone FaceEnhancer (GFPGAN) performance diagnostic.

Symptom: batch enhancer stage runs ~4 FPS on 4 workers with the GPU mostly
idle and the CPU pegged, on an RTX 5090 with healthy torch CUDA (verified).
This script isolates FaceEnhancer from ALL batch plumbing (no PNG IO, no
queue, no GUI) and answers two questions with numbers:

  1. Where does a single enhance() spend its time?  (detection / GFPGAN
     forward / paste-back, timed with torch.cuda.synchronize at boundaries)
  2. Does it scale across workers the way the batch stage assumes?
     (W instances, W threads, same division of labour as _ProcessorPool)

Run on Windows from the project root:

    .venv\\Scripts\\python.exe scripts\\enhancer_bench.py <video_or_image> [frames] [workers]

Defaults: frames=20, workers=4. Paste the whole output back.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    FaceEnhancer,
    FaceEnhancerParams,
)


def load_frame(media: Path) -> np.ndarray:
    """A frame from the media: image as-is; video at 25% in (past intros)."""
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


MODEL = EnhancerModel.GFPGAN  # set from argv in main()


def make_enhancer() -> FaceEnhancer:
    """Mirror the batch driver's factory for the user's settings: fp16,
    upscale=1, no rotation compensation."""
    params = FaceEnhancerParams(
        model=MODEL,
        upscale=1,
        only_center_face=False,
        fp16=True,
        rotation_compensation=False,
    )
    return FaceEnhancer(params=params, device="auto")


class _Timed:
    """Wraps a callable; accumulates wall time (post-CUDA-sync) per call into
    a shared bucket. Sync makes the attribution honest — without it, async
    kernel launches bill the NEXT phase."""

    def __init__(self, target, bucket: dict, name: str) -> None:
        self._target = target
        self._bucket = bucket
        self._name = name

    def __call__(self, *args, **kwargs):
        import torch

        t0 = time.perf_counter()
        try:
            return self._target(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._bucket[self._name] += time.perf_counter() - t0
            self._bucket[f"n:{self._name}"] += 1


def instrument(enh: FaceEnhancer, bucket: dict) -> None:
    """Hook the heavyweight phases of whichever backend is active."""
    if enh._restorer is not None:  # noqa: SLF001 — GFPGAN
        restorer = enh._restorer  # noqa: SLF001
        fh = restorer.face_helper
        fh.get_face_landmarks_5 = _Timed(fh.get_face_landmarks_5, bucket, "detect")
        fh.paste_faces_to_input_image = _Timed(
            fh.paste_faces_to_input_image, bucket, "paste_back"
        )
        restorer.gfpgan = _Timed(restorer.gfpgan, bucket, "gfpgan_forward")
        return
    # PlainBfrBackend (GPEN / RestoreFormer): patch its module-level phases +
    # the analyser instance. Module patches are process-global — instrumented
    # single-worker runs only.
    from sinner2.pipeline.processors import bfr_onnx

    bfr = enh._bfr  # noqa: SLF001
    bfr._analyser.analyse = _Timed(  # noqa: SLF001
        bfr._analyser.analyse, bucket, "detect"  # noqa: SLF001
    )
    if not isinstance(bfr_onnx._restore_aligned, _Timed):  # noqa: SLF001
        bfr_onnx._restore_aligned = _Timed(  # noqa: SLF001
            bfr_onnx._restore_aligned, bucket, "onnx_restore"  # noqa: SLF001
        )
        bfr_onnx.paste_back = _Timed(bfr_onnx.paste_back, bucket, "paste_back")
    else:  # rebind buckets on a fresh instrumented run
        bfr_onnx._restore_aligned._bucket = bucket  # noqa: SLF001
        bfr_onnx.paste_back._bucket = bucket


def bench_single(frame: np.ndarray, n: int) -> None:
    print("\n=== single worker, instrumented ===")
    enh = make_enhancer()
    t0 = time.perf_counter()
    enh.setup()
    print(f"  setup: {time.perf_counter() - t0:.2f}s")
    bucket: dict = defaultdict(float)
    instrument(enh, bucket)
    # Warmup (CUDA context, lazy allocs, first-call inits) — not counted.
    enh.process(frame.copy())
    if enh._restorer is not None:  # noqa: SLF001
        faces = len(enh._restorer.face_helper.cropped_faces)  # noqa: SLF001
    else:
        faces = len(enh._bfr._analyser.analyse(frame))  # noqa: SLF001
    print(f"  faces detected in frame: {faces}")
    if faces == 0:
        print("  WARNING: no faces — paste/forward phases are trivially fast; "
              "pick media with a visible face.")
    bucket.clear()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        enh.process(frame.copy())
        times.append(time.perf_counter() - t0)
    total = sum(times)
    times.sort()
    print(f"  {n} frames: total {total:.2f}s -> {n / total:.2f} FPS")
    print(f"  per-frame: mean {total / n * 1000:.0f}ms  "
          f"p50 {times[n // 2] * 1000:.0f}ms  p95 {times[int(n * 0.95)] * 1000:.0f}ms")
    accounted = 0.0
    phases = [p for p in ("detect", "gfpgan_forward", "onnx_restore", "paste_back")
              if bucket.get(f"n:{p}")]
    for phase in phases:
        t = bucket[phase]
        calls = int(bucket[f"n:{phase}"])
        accounted += t
        print(f"    {phase:15s} {t:7.2f}s total  "
              f"{t / max(1, calls) * 1000:6.0f}ms/call  ({calls} calls)  "
              f"{t / total * 100:5.1f}%")
    print(f"    {'other (glue)':15s} {total - accounted:7.2f}s total"
          f"{'':24s}{(total - accounted) / total * 100:5.1f}%")
    enh.release()


def bench_workers(frame: np.ndarray, n_per_worker: int, workers: int) -> None:
    print(f"\n=== {workers} workers x {n_per_worker} frames "
          f"(mirrors the batch _ProcessorPool) ===")
    instances = []
    t0 = time.perf_counter()
    for _ in range(workers):
        e = make_enhancer()
        e.setup()
        instances.append(e)
    print(f"  setup x{workers}: {time.perf_counter() - t0:.2f}s")
    # Warmup each instance once.
    for e in instances:
        e.process(frame.copy())
    barrier = threading.Barrier(workers)

    def run(e: FaceEnhancer) -> float:
        barrier.wait()
        t = time.perf_counter()
        for _ in range(n_per_worker):
            e.process(frame.copy())
        return time.perf_counter() - t

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        per_worker = list(ex.map(run, instances))
    wall = time.perf_counter() - t0
    frames = n_per_worker * workers
    print(f"  {frames} frames in {wall:.2f}s -> aggregate {frames / wall:.2f} FPS")
    for i, t in enumerate(per_worker):
        print(f"    worker {i}: {t:.2f}s ({n_per_worker / t:.2f} FPS alone)")
    for e in instances:
        e.release()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/enhancer_bench.py <video_or_image> "
              "[frames] [workers]")
        return 2
    media = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if len(sys.argv) > 4:
        global MODEL
        MODEL = EnhancerModel(sys.argv[4])
    print(f"enhancer model: {MODEL.value}")
    import torch

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    print(f"cv2 threads={cv2.getNumThreads()}  torch threads={torch.get_num_threads()}")
    frame = load_frame(media)
    print(f"frame: {frame.shape[1]}x{frame.shape[0]} from {media.name}")
    bench_single(frame, n)
    bench_workers(frame, n, 2)
    bench_workers(frame, n, workers)
    print("\nDone. Paste this whole output back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
