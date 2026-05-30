import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

from sinner2.config.source import Source
from sinner2.config.target import Target
from sinner2.io.cv2_unicode import imwrite_unicode
from sinner2.io.reader_pool import ReaderPool
from sinner2.io.target_reader import ImageTargetReader
from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import DiskFrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_enhancer import FaceEnhancer
from sinner2.pipeline.processors.face_swapper import FaceSwapper
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import BestEffortStrategy
from sinner2.types import Frame, FrameIndex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="sinner2 smoke test: image-source × image-target end-to-end"
    )
    p.add_argument("--source", type=Path, required=True, help="Source face image")
    p.add_argument("--target", type=Path, required=True, help="Target image to process")
    p.add_argument("--output", type=Path, required=True, help="Output path")
    p.add_argument("--no-enhance", action="store_true", help="Skip the FaceEnhancer step")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/sinner2-smoke"),
        help="Scratch directory for the disk frame store",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    source = Source(path=args.source)
    target = Target(path=args.target)

    chain: list[Processor] = [FaceSwapper(source=source)]
    if not args.no_enhance:
        chain.append(FaceEnhancer())

    timeline = Timeline(fps=1.0)
    # Clear the frame store so leftovers from a prior run don't get surfaced
    # by the playback thread before this run's worker completes.
    frames_dir = args.work_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    store = DiskFrameStore(frames_dir)
    cache = MemoryFrameCache(max_bytes=128 * 1024 * 1024)
    write_executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
    buffer = FrameBuffer(store, cache, timeline, write_executor)
    reader_pool = ReaderPool(lambda: ImageTargetReader(target), args.workers)

    output_lock = threading.Lock()
    output_frame: list[Frame | None] = [None]
    delivered = threading.Event()

    def on_frame(frame: Frame, _index: FrameIndex) -> None:
        with output_lock:
            output_frame[0] = frame
        delivered.set()

    executor = RealtimeExecutor(
        reader_pool=reader_pool,
        buffer=buffer,
        timeline=timeline,
        chain=chain,
        strategy=BestEffortStrategy(),
        worker_count=args.workers,
    )
    executor.on_frame_ready(on_frame)

    print(f"loading models, running chain: {[p.name for p in chain]}")
    try:
        executor.start()
    except Exception as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        write_executor.shutdown(wait=False)
        return 3

    executor.play()

    timeout_s = 300.0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        end_of_target = executor.status.get() == "end of target"
        if end_of_target and delivered.is_set():
            break
        time.sleep(0.1)
    else:
        print(f"timeout after {timeout_s}s waiting for output", file=sys.stderr)

    executor.stop()
    write_executor.shutdown(wait=True)

    with output_lock:
        frame = output_frame[0]
    if frame is None:
        print("no output frame was delivered", file=sys.stderr)
        return 1
    if not imwrite_unicode(args.output, frame):
        print(f"imwrite failed for {args.output}", file=sys.stderr)
        return 2
    print(f"wrote {args.output} (shape={frame.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
