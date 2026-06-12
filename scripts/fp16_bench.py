"""fp16 vs fp32 diagnostic for the hot ONNX models.

The CPU-side fixes landed; the remaining chain cost is the ONNX forwards
themselves (gfpgan_1.4 ~32ms, gpen_512 ~55ms, SPAN ~248ms at FullHD). fp16
could roughly halve them on tensor cores — but the inswapper precedent says
fp16 can CORRUPT output while looking fast, so this measures BOTH:

  * speed: fp32 vs fp16 session (CUDA EP), same real input, N runs each
  * output: max abs diff + % differing pixels on the final uint8 image,
    plus side-by-side PNGs (scripts/fp16_<model>_fp32.png / _fp16.png)
    for the eyeball check

Conversion: onnxconverter_common.convert_float_to_float16(keep_io_types=True)
— I/O stays float32, internals run half. Converted models are cached next to
the originals as <name>.fp16.onnx (delete to reconvert).

Run on Windows from the project root (GPU required):

    .venv\\Scripts\\python.exe scripts\\fp16_bench.py <video_or_image> [runs]

Paste the whole output back and eyeball the PNG pairs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sinner2.pipeline.face_analyser import FaceAnalyser
from sinner2.pipeline.model_cache import get_model_path, get_models_dir


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


def aligned_face(frame: np.ndarray, size: int = 512) -> np.ndarray:
    """A real ArcFace-aligned crop — the actual input distribution the
    restorers see (detection-only buffalo, same as the backends)."""
    from insightface.utils import face_align

    analyser = FaceAnalyser(detection_only=True)
    faces = analyser.analyse(frame)
    if not faces:
        raise SystemExit("no face in the bench frame — pick other media")
    m = face_align.estimate_norm(np.asarray(faces[0].kps, np.float32), size)
    return cv2.warpAffine(frame, m, (size, size))


def fp16_path(name: str) -> Path:
    return get_models_dir() / (Path(name).stem + ".fp16.onnx")


def convert_fp16(name: str) -> Path:
    out = fp16_path(name)
    if out.is_file():
        return out
    import onnx
    from onnxconverter_common import float16

    print(f"  converting {name} -> {out.name} ...")
    model = onnx.load(str(get_model_path(name)))
    # GPEN-style exports list every WEIGHT as a graph input; keep_io_types then
    # pins them fp32 while the initializers go fp16 → unloadable model. Strip
    # initializer-named entries from graph.input first (they're weights, not
    # real inputs — same cleanup as onnxruntime's remove_initializer_from_input).
    init_names = {i.name for i in model.graph.initializer}
    real_inputs = [i for i in model.graph.input if i.name not in init_names]
    if len(real_inputs) != len(model.graph.input):
        del model.graph.input[:]
        model.graph.input.extend(real_inputs)
    model16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(
        model16, str(out),
        save_as_external_data=model16.ByteSize() > (2 << 30) - (64 << 20),
    )
    return out


def make_session(path: Path):
    import onnxruntime as ort

    from sinner2.pipeline.model_cache import (
        build_provider_options,
        build_session_options,
    )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(
        str(path), sess_options=build_session_options(), providers=providers,
        provider_options=build_provider_options(providers),
    )


# ---- per-model input/run/post (the backends' verified contracts) ----

def bfr_feeds(session, crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None], np.float32)
    in_name = session.get_inputs()[0].name
    return {in_name: chw}


def codeformer_feeds(session, crop_bgr):
    feeds = bfr_feeds(session, crop_bgr)
    feeds["weight"] = np.array(0.7, np.float64)
    return feeds


def bfr_post(out) -> np.ndarray:
    img = (np.clip(out[0], -1.0, 1.0) + 1.0) / 2.0
    img = (img.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def span_feeds(session, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    return {session.get_inputs()[0].name: chw}


def span_post(out) -> np.ndarray:
    from sinner2.pipeline.processors.upscaler import _finalize_bgr

    return _finalize_bgr(out[0])


MODELS = [
    # (model file, input source, feeds fn, post fn)
    ("gfpgan_1.4.onnx", "face", bfr_feeds, bfr_post),
    ("gpen_bfr_512.onnx", "face", bfr_feeds, bfr_post),
    ("codeformer.onnx", "face", codeformer_feeds, bfr_post),
    ("span_kendata_x4.onnx", "frame", span_feeds, span_post),
]


def bench(session, feeds, post, runs: int):
    out_name = session.get_outputs()[0].name
    session.run([out_name], feeds)  # warmup (cudnn search / engine init)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = session.run([out_name], feeds)[0]
        times.append(time.perf_counter() - t0)
    return post(out), sum(times) / len(times)  # post fns strip the batch dim


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/fp16_bench.py <video_or_image> [runs]")
        return 2
    media = Path(sys.argv[1])
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    frame = load_frame(media)
    face = aligned_face(frame)
    print(f"frame {frame.shape[1]}x{frame.shape[0]}, face crop 512x512, "
          f"{runs} runs per variant")
    out_dir = Path(__file__).resolve().parent

    for name, source, feeds_fn, post_fn in MODELS:
        print(f"\n=== {name} ===")
        inp = face if source == "face" else frame
        try:
            p16 = convert_fp16(name)
            s32 = make_session(get_model_path(name))
            img32, t32 = bench(s32, feeds_fn(s32, inp), post_fn, runs)
            del s32
            s16 = make_session(p16)
            img16, t16 = bench(s16, feeds_fn(s16, inp), post_fn, runs)
            del s16
        except Exception as exc:  # noqa: BLE001 — keep measuring the rest
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        diff = np.abs(img32.astype(np.int16) - img16.astype(np.int16))
        stem = Path(name).stem
        cv2.imwrite(str(out_dir / f"fp16_{stem}_fp32.png"), img32)
        cv2.imwrite(str(out_dir / f"fp16_{stem}_fp16.png"), img16)
        print(f"  fp32: {t32 * 1000:7.1f}ms   fp16: {t16 * 1000:7.1f}ms   "
              f"speedup x{t32 / t16:.2f}")
        print(f"  output diff: max {diff.max()}  mean {diff.mean():.3f}  "
              f">2 LSB on {(diff > 2).mean() * 100:.2f}% of subpixels")

    print("\nDone. Paste this whole output back and eyeball the "
          "scripts/fp16_*_fp32.png vs _fp16.png pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
