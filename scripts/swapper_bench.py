"""Standalone FaceSwapper (inswapper) performance diagnostic.

Follow-up to scripts/enhancer_bench.py: every full-frame paste we measured was
the dominant CPU cost of its processor, and the default swapper's paste runs
INSIDE insightface's INSwapper.get(paste_back=True) (warp + full-res float
blend + fake_diff masking + full-res GaussianBlur) where the ROI optimization
can't reach it. This isolates the swapper and answers:

  1. How does a swap frame split between detection / model forward / paste?
     (paste cost = get(paste_back=True) minus get(paste_back=False))
  2. What would the candidate fix cost instead? (get(paste_back=False) + our
     ROI paste_back with a feathered box mask — the facefusion-style blend)
  3. How does the shared-session swapper scale across workers?

It also writes side-by-side quality comparisons next to this script:
  swapper_paste_insightface.png  (current: fake_diff blend)
  swapper_paste_roi.png          (candidate: feathered ROI blend)

Run on Windows from the project root (GPU required):

    .venv\\Scripts\\python.exe scripts\\swapper_bench.py <target_video> <source_face_img> [frames] [workers]

Defaults: frames=20, workers=8 (the realtime pool size in use). Paste the
whole output back.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sinner2.config.source import Source
from sinner2.pipeline.face_geometry import feather_mask, paste_back
from sinner2.pipeline.processors.face_swapper import FaceSwapper, FaceSwapperParams


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


def make_swapper(source_img: Path) -> FaceSwapper:
    """Default model (inswapper_128), buffalo_l detection, rotation off so the
    measurement is the plain swap path, no occlusion/sex filter."""
    params = FaceSwapperParams(rotation_compensation=False)
    sw = FaceSwapper(source=Source(path=source_img), params=params)
    sw.setup()
    return sw


def timed(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def report(label: str, times: list[float]) -> float:
    times = sorted(times)
    n = len(times)
    mean = sum(times) / n
    print(f"    {label:34s} mean {mean * 1000:6.1f}ms  "
          f"p50 {times[n // 2] * 1000:6.1f}ms  p95 {times[int(n * 0.95)] * 1000:6.1f}ms")
    return mean


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python scripts/swapper_bench.py <target_video> "
              "<source_face_img> [frames] [workers]")
        return 2
    media, source = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    frame = load_frame(media)
    print(f"frame: {frame.shape[1]}x{frame.shape[0]} from {media.name}; "
          f"source: {source.name}")
    t0 = time.perf_counter()
    sw = make_swapper(source)
    print(f"setup: {time.perf_counter() - t0:.2f}s")
    analyser = sw._analyser          # noqa: SLF001
    swapper = sw._swapper            # noqa: SLF001
    source_face = sw._source_face    # noqa: SLF001

    faces = analyser.analyse(frame)
    print(f"faces detected in frame: {len(faces)}")
    if not faces:
        print("no faces — pick media with a visible face")
        return 1
    face = faces[0]

    # Warmup (CUDA context, cudnn EXHAUSTIVE search) — not counted.
    for _ in range(3):
        sw.process(frame.copy())

    print(f"\n=== single worker, phase isolation ({n} frames) ===")
    analyser.reset_cache()
    t_det = report("analyse (buffalo_l full pack)",
                   timed(lambda: (analyser.reset_cache(),
                                  analyser.analyse(frame))[1] and None, n))
    t_fwd = report("get(paste_back=False) = fwd only",
                   timed(lambda: swapper.get(frame.copy(), face, source_face,
                                             paste_back=False), n))
    t_full = report("get(paste_back=True) = fwd+paste",
                    timed(lambda: swapper.get(frame.copy(), face, source_face,
                                              paste_back=True), n))
    print(f"    {'-> insightface internal paste':34s} ~{(t_full - t_fwd) * 1000:6.1f}ms "
          f"({(t_full - t_fwd) / (t_det + t_full) * 100:.0f}% of the frame)")

    # Candidate: forward only + OUR ROI paste with a feathered box mask
    # (facefusion-style static blend). INSwapper.get(paste_back=False) returns
    # (aligned swapped crop, the norm_crop2 matrix) — reuse that exact matrix.
    crop, _m = swapper.get(frame.copy(), face, source_face, paste_back=False)
    mask = feather_mask(crop.shape[0])

    def candidate() -> np.ndarray:
        c, m = swapper.get(frame.copy(), face, source_face, paste_back=False)
        return paste_back(frame.copy(), c, m, mask,
                          border_replicate=True, clip_mask=True)

    t_cand = report("candidate: fwd + ROI feather paste", timed(candidate, n))
    t_proc = report("FaceSwapper.process() end-to-end", timed(
        lambda: sw.process(frame.copy()), n))
    print(f"    candidate end-to-end estimate     ~{(t_det + t_cand) * 1000:6.1f}ms "
          f"vs current {t_proc * 1000:6.1f}ms")

    # Quality comparison images.
    out_dir = Path(__file__).resolve().parent
    cv2.imwrite(str(out_dir / "swapper_paste_insightface.png"),
                swapper.get(frame.copy(), face, source_face, paste_back=True))
    cv2.imwrite(str(out_dir / "swapper_paste_roi.png"), candidate())
    print(f"\n  quality pair written to {out_dir}\\swapper_paste_*.png — "
          "compare the face boundary closely.")

    print(f"\n=== {workers} workers x {n} frames (shared session, "
          "like the realtime pool) ===")
    def run_one() -> None:
        sw.process(frame.copy())

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(lambda: [run_one() for _ in range(n)])
                  for _ in range(workers)]:
            f.result()
    wall = time.perf_counter() - t0
    frames = n * workers
    print(f"  {frames} frames in {wall:.2f}s -> aggregate {frames / wall:.2f} FPS")

    sw.release()
    print("\nDone. Paste this whole output back and eyeball the two PNGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
