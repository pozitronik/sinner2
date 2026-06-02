"""Parallel source-frame reader pool.

Decouples source I/O from the realtime dispatcher. The dispatcher used
to call `target_reader.read(idx)` synchronously inside its tick, which
serialised every read through one thread — fine for sequential reads
(cv2/ffmpeg both keep a sequential buffer) but catastrophic for any
non-sequential access on slow storage (network share, HDD). One py-spy
on an SMB target showed the dispatcher stuck inside `cv2.VideoCapture.read()`
while all 8 worker threads sat idle waiting for the queue.

ReaderPool fixes that by holding N reader instances, each owned by its
own thread, fed from a shared FIFO request queue. The caller submits
`read_async(idx)` (non-blocking) and gets back a `concurrent.futures.Future`
that resolves on whichever reader thread services the request. Workers
in the executor await the future before calling the chain.

Key properties:
  * N parallel reads possible — SMB pipelines them well, scales nearly
    linearly until the network saturates.
  * Each reader stays sequential within its own request stream; whether
    that's efficient depends on backend + access pattern (see trade-off
    below).
  * Pool size is independent of worker count. A typical config:
    workers=8 + pool=1 on local SSD (one buffered reader feeds many
    workers); workers=8 + pool=8 on SMB (saturate the network with
    parallel seeks).

Trade-off: with `size > 1` and a sequential access pattern, requests
distribute round-robin across N readers, so each reader sees a stride-N
pattern (frames 0, N, 2N, ...). With the **ffmpeg subprocess backend**
this forces a decoder restart on every read — a regression vs. the
single-reader case. The CV2 backend's in-place seek absorbs this with
negligible cost. The processor-controls tooltip names the case.

A future iteration could add affinity-aware routing (route the next
sequential index to the reader that just produced index-1) to eliminate
this regression without losing parallelism. Out of scope for v1.

Threading: the pool is meant to be constructed and shut down from a
single owner (RealtimeExecutor). `read_async` is safe to call from any
thread once construction completes. Reader instances inside the pool
are never accessed from outside their owning thread.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from queue import Empty, Queue

from sinner2.io.target_reader import TargetReader
from sinner2.types import Frame, FrameIndex

_POLL_S = 0.5  # how long a reader thread waits for a request before re-checking _stop_event
# Rolling time window for the reads_per_second metric. Same shape as the
# executor's processing_fps: time-windowed so a brief idle period
# decays cleanly to 0 instead of holding a stale historical average.
_READ_RATE_WINDOW_S = 3.0

# Sentinel pushed onto the request queue at shutdown to unblock the
# reader-thread `queue.get`. A None future also works as the sentinel —
# explicit type makes the worker loop's check clear.
_ShutdownSentinel = object
_SHUTDOWN: _ShutdownSentinel = object()


class ReaderPool:
    def __init__(
        self,
        factory: Callable[[], TargetReader],
        size: int,
        *,
        name: str = "reader",
    ) -> None:
        if size < 1:
            raise ValueError(f"pool size must be >= 1; got {size}")
        # The first instance becomes the metadata probe AND the first
        # pool worker — we never throw away a built reader. If factory
        # raises here, propagate to the caller (session setup will
        # surface it as today's reader construction does).
        probe = factory()
        self._fps = float(probe.fps)
        self._frame_count = int(probe.frame_count)
        # Native (pre-scale) source dimensions, surfaced so the GUI can show
        # the resulting size for any processing scale.
        self._native_width = int(probe.native_width)
        self._native_height = int(probe.native_height)
        # Build the remaining N-1 instances. If any later instance
        # raises, release the ones built so far before propagating.
        readers: list[TargetReader] = [probe]
        try:
            for _ in range(size - 1):
                readers.append(factory())
        except Exception:
            for r in readers:
                try:
                    r.release()
                except Exception:
                    pass
            raise

        self._readers = readers
        self._size = size
        self._stop_event = threading.Event()
        self._request_queue: Queue[tuple[FrameIndex, Future[Frame | None]] | object] = Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shut_down = False
        # Time-windowed rate of successful reads across the whole pool.
        # Used by the metrics overlay to display read throughput
        # distinct from worker-completion throughput.
        self._rate_lock = threading.Lock()
        self._read_times: deque[float] = deque()
        # (timestamp, duration_ms) of recent successful reads. The median over
        # the window is the "how expensive are reads right now" signal the skip
        # strategy uses to tell an I/O-bound source (slow random reads) from a
        # compute-bound one (cheap reads, slow GPU).
        self._read_durations_ms: deque[tuple[float, float]] = deque()

        for i, reader in enumerate(readers):
            t = threading.Thread(
                target=self._reader_loop,
                args=(reader,),
                name=f"sinner2-{name}-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    @property
    def size(self) -> int:
        return self._size

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def native_width(self) -> int:
        return self._native_width

    @property
    def native_height(self) -> int:
        return self._native_height

    def reads_per_second(self) -> float:
        """Successful reads per wall-clock second over the last few
        seconds. Trims old entries on each call so the value decays to
        0 when the pool is idle (rather than holding a stale historical
        average that would mislead the metrics overlay)."""
        now = time.monotonic()
        cutoff = now - _READ_RATE_WINDOW_S
        with self._rate_lock:
            while self._read_times and self._read_times[0] < cutoff:
                self._read_times.popleft()
            count = len(self._read_times)
            if count < 2:
                return 0.0
            span = self._read_times[-1] - self._read_times[0]
        if span <= 0:
            return 0.0
        return (count - 1) / span

    def recent_read_latency_ms(self) -> float:
        """Median duration of recent successful reads (ms), over the same window
        as reads_per_second. A read slower than a frame budget means the SOURCE
        is the bottleneck; the skip strategy uses this to decide whether
        random-access skipping is affordable. 0.0 when no reads in the window
        (idle / startup) — read as 'no evidence of an I/O bottleneck'."""
        now = time.monotonic()
        cutoff = now - _READ_RATE_WINDOW_S
        with self._rate_lock:
            while self._read_durations_ms and self._read_durations_ms[0][0] < cutoff:
                self._read_durations_ms.popleft()
            durations = sorted(d for _, d in self._read_durations_ms)
        if not durations:
            return 0.0
        n = len(durations)
        mid = n // 2
        if n % 2:
            return durations[mid]
        return (durations[mid - 1] + durations[mid]) / 2.0

    def read_async(self, index: FrameIndex) -> Future[Frame | None]:
        """Submit a read request. Returns a Future that resolves on the
        reader thread that services it. If the pool is shutting down,
        the returned future is cancelled immediately so callers always
        get a defined future back."""
        future: Future[Frame | None] = Future()
        if self._stop_event.is_set():
            future.cancel()
            return future
        self._request_queue.put((index, future))
        return future

    def _reader_loop(self, reader: TargetReader) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._request_queue.get(timeout=_POLL_S)
                except Empty:
                    continue
                if item is _SHUTDOWN:
                    break
                index, future = item  # type: ignore[misc]
                # set_running_or_notify_cancel returns False if the
                # future was cancelled before we picked it up — skip the
                # read entirely in that case.
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    t0 = time.monotonic()
                    frame = reader.read(index)
                    duration_ms = (time.monotonic() - t0) * 1000.0
                    future.set_result(frame)
                    # Record successful reads only — exceptions and Nones
                    # would inflate the rate with non-useful work.
                    if frame is not None:
                        now = time.monotonic()
                        with self._rate_lock:
                            self._read_times.append(now)
                            self._read_durations_ms.append((now, duration_ms))
                except Exception as exc:
                    future.set_exception(exc)
        finally:
            # Release this thread's reader on the way out — the executor
            # is single-owner of the pool, so this is the right place
            # for per-reader cleanup (vs. shutdown() iterating).
            try:
                reader.release()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Stop reader threads and release every reader. Idempotent.

        Any read requests still queued but not yet picked up have their
        futures cancelled so callers waiting on `.result()` exit cleanly.
        """
        with self._lock:
            if self._shut_down:
                return
            self._shut_down = True
            self._stop_event.set()

        # Cancel any not-yet-claimed requests so workers don't block
        # forever on a future that will never resolve. We can't easily
        # drain through the Queue's internal lock, so just race: pull
        # what's there and cancel.
        drained: list[tuple[FrameIndex, Future[Frame | None]]] = []
        while True:
            try:
                item = self._request_queue.get_nowait()
            except Empty:
                break
            if item is _SHUTDOWN:
                continue
            drained.append(item)  # type: ignore[arg-type]
        for _, future in drained:
            future.cancel()

        # Wake every thread out of its queue.get even if poll timeout
        # hasn't elapsed.
        for _ in self._threads:
            self._request_queue.put(_SHUTDOWN)

        for t in self._threads:
            t.join(timeout=2.0)
