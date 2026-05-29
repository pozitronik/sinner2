"""Disk write capacity benchmark for sinner2's frame cache.

Run from the project root with the Windows venv:
    .venv\\Scripts\\python.exe scripts\\bench_write.py

Measures encoding + write throughput for combinations of:
- Frame size (FHD, 4K)
- Format / quality (PNG-1, PNG-6, JPEG-95, JPEG-85, WebP-95 if available)
- Single-threaded vs concurrent (mimics the 8-worker production pool)

Reports: fps, MB/s, encoded size, encode latency p50/p95, write latency p50/p95.
Writes to a temp_bench/ subdirectory and cleans it up at the end.
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Random frames are uncompressible — a conservative stand-in for real content
# (which compresses better, especially for face-region backgrounds).
def gen_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@dataclass
class Scenario:
    label: str
    ext: str
    params: list[int]


def scenarios() -> list[Scenario]:
    out: list[Scenario] = [
        Scenario("PNG c=1", ".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]),
        Scenario("PNG c=6", ".png", [cv2.IMWRITE_PNG_COMPRESSION, 6]),
        Scenario("JPEG q=95", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
        Scenario("JPEG q=85", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]),
    ]
    # WebP support is build-dependent; probe with a 1-pixel write.
    try:
        ok, _ = cv2.imencode(".webp", np.zeros((1, 1, 3), np.uint8), [cv2.IMWRITE_WEBP_QUALITY, 95])
        if ok:
            out.append(Scenario("WebP q=95", ".webp", [cv2.IMWRITE_WEBP_QUALITY, 95]))
    except Exception:
        pass
    return out


@dataclass
class Result:
    label: str
    count: int
    elapsed_s: float
    fps: float
    throughput_mb_s: float
    encoded_size_mb_avg: float
    encode_ms_p50: float
    encode_ms_p95: float
    write_ms_p50: float
    write_ms_p95: float


def _pct(values: list[float], p: float) -> float:
    """Percentile in ms — values are seconds, multiply at output."""
    if not values:
        return 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def bench_single_threaded(
    out_dir: Path, frame: np.ndarray, scenario: Scenario, count: int
) -> Result:
    out_dir.mkdir(parents=True, exist_ok=True)
    encode_times: list[float] = []
    write_times: list[float] = []
    total_bytes = 0

    t0 = time.perf_counter()
    for i in range(count):
        e_start = time.perf_counter()
        ok, buf = cv2.imencode(scenario.ext, frame, scenario.params)
        e_end = time.perf_counter()
        if not ok:
            raise RuntimeError(f"imencode failed for {scenario.ext}")

        path = out_dir / f"frame_{i:06d}{scenario.ext}"
        w_start = time.perf_counter()
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        w_end = time.perf_counter()

        encode_times.append(e_end - e_start)
        write_times.append(w_end - w_start)
        total_bytes += len(buf)

    elapsed = time.perf_counter() - t0
    return Result(
        label=scenario.label,
        count=count,
        elapsed_s=elapsed,
        fps=count / elapsed,
        throughput_mb_s=(total_bytes / 1024 / 1024) / elapsed,
        encoded_size_mb_avg=(total_bytes / count) / 1024 / 1024,
        encode_ms_p50=_pct(encode_times, 50) * 1000,
        encode_ms_p95=_pct(encode_times, 95) * 1000,
        write_ms_p50=_pct(write_times, 50) * 1000,
        write_ms_p95=_pct(write_times, 95) * 1000,
    )


def bench_concurrent(
    out_dir: Path,
    frame: np.ndarray,
    scenario: Scenario,
    count: int,
    workers: int,
) -> Result:
    """N writer threads racing on the same directory.

    Mimics production: each producer worker submits encode+write tasks to
    a shared ThreadPoolExecutor. Total throughput is what we want, not
    per-task latency — but we still record latency to expose tail behaviour
    under contention.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    encode_times: list[float] = []
    write_times: list[float] = []
    total_bytes_list: list[int] = []

    def write_one(i: int) -> tuple[float, float, int]:
        e_start = time.perf_counter()
        ok, buf = cv2.imencode(scenario.ext, frame, scenario.params)
        e_end = time.perf_counter()
        if not ok:
            raise RuntimeError(f"imencode failed for {scenario.ext}")
        path = out_dir / f"frame_{i:06d}{scenario.ext}"
        w_start = time.perf_counter()
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        w_end = time.perf_counter()
        return e_end - e_start, w_end - w_start, len(buf)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(write_one, i) for i in range(count)]
        for f in as_completed(futures):
            e, w, b = f.result()
            encode_times.append(e)
            write_times.append(w)
            total_bytes_list.append(b)
    elapsed = time.perf_counter() - t0

    total_bytes = sum(total_bytes_list)
    return Result(
        label=f"{scenario.label} x{workers}",
        count=count,
        elapsed_s=elapsed,
        fps=count / elapsed,
        throughput_mb_s=(total_bytes / 1024 / 1024) / elapsed,
        encoded_size_mb_avg=(total_bytes / count) / 1024 / 1024,
        encode_ms_p50=_pct(encode_times, 50) * 1000,
        encode_ms_p95=_pct(encode_times, 95) * 1000,
        write_ms_p50=_pct(write_times, 50) * 1000,
        write_ms_p95=_pct(write_times, 95) * 1000,
    )


def fmt_result(r: Result) -> str:
    return (
        f"  {r.label:18s}  "
        f"{r.fps:7.1f} fps  "
        f"{r.throughput_mb_s:7.1f} MB/s  "
        f"size {r.encoded_size_mb_avg:6.3f} MB  "
        f"enc {r.encode_ms_p50:6.2f}/{r.encode_ms_p95:6.2f} ms  "
        f"wr  {r.write_ms_p50:6.2f}/{r.write_ms_p95:6.2f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="temp_bench",
        help="Where to write benchmark files (relative or absolute).",
    )
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Concurrent writer counts to test.",
    )
    args = parser.parse_args(argv)

    out_root = Path(args.out_dir).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)

    print(f"OUT:    {out_root}")
    print(f"COUNT:  {args.count} frames per scenario")
    print(f"FMT:    fps | MB/s | encoded-size | encode p50/p95 ms | write p50/p95 ms")
    print()

    sizes = [
        ("FHD 1920x1080", 1920, 1080),
        ("4K  3840x2160", 3840, 2160),
    ]
    scns = scenarios()

    for size_label, w, h in sizes:
        raw_mb = w * h * 3 / 1024 / 1024
        print(f"=== {size_label}  (raw {raw_mb:.2f} MB/frame) ===")
        frame = gen_frame(w, h)

        print("[single-threaded baseline]")
        for s in scns:
            d = out_root / size_label.split()[0] / "st" / s.label.replace(" ", "_")
            r = bench_single_threaded(d, frame, s, args.count)
            print(fmt_result(r))

        for nw in args.workers:
            if nw == 1:
                continue  # already covered by single-threaded
            print(f"[concurrent x{nw}]")
            for s in scns:
                d = out_root / size_label.split()[0] / f"c{nw}" / s.label.replace(" ", "_")
                r = bench_concurrent(d, frame, s, args.count, nw)
                print(fmt_result(r))
        print()

    shutil.rmtree(out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
