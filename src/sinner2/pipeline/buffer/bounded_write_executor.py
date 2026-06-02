"""Thread pool + bounded outstanding-task counter.

Wraps ThreadPoolExecutor with two extra properties:

1. Outstanding-task cap. submit() returns False instead of blocking when
   the cap is reached. The caller (FrameBuffer.put) silently skips the
   write — the frame stays in the memory cache until LRU eviction. This
   is the backpressure that keeps the cache + executor combined memory
   usage bounded on slow disks.

2. Lightweight latency and throughput metrics: outstanding depth, dropped
   count, completed count, and a rolling latency window (p50/p95). All
   read via metrics_snapshot() under a single lock.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

_LATENCY_WINDOW = 200  # last N completion latencies kept for p50/p95


@dataclass(frozen=True)
class WriteExecutorMetrics:
    outstanding: int
    max_outstanding: int
    submitted: int
    completed: int
    dropped: int
    latency_p50_ms: float
    latency_p95_ms: float
    failed: int = 0  # writes that raised (disk full / permission / bad path)


class BoundedWriteExecutor:
    def __init__(self, max_workers: int, max_outstanding: int) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1; got {max_workers}")
        if max_outstanding < 1:
            raise ValueError(f"max_outstanding must be >= 1; got {max_outstanding}")
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sinner2-write"
        )
        self._max_outstanding = max_outstanding
        self._lock = threading.Lock()
        self._outstanding = 0
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        self._latencies_ms: deque[float] = deque(maxlen=_LATENCY_WINDOW)

    def submit(self, fn: Callable[..., None], *args: object, **kwargs: object) -> bool:
        """Schedule fn(*args, **kwargs) for execution. Returns False (and
        bumps the drop counter) when the outstanding cap is reached."""
        with self._lock:
            if self._outstanding >= self._max_outstanding:
                self._dropped += 1
                return False
            self._outstanding += 1
            self._submitted += 1

        def wrapped() -> None:
            t0 = time.perf_counter()
            ok = True
            try:
                fn(*args, **kwargs)
            except Exception:
                # A raised write (disk full / permission / bad path) must NOT be
                # counted as completed — that would hide a persistent disk
                # failure (the frame is silently lost once LRU evicts the cached
                # copy) behind healthy-looking metrics. Count it as failed and
                # swallow here (the Future is discarded — nobody checks it — so
                # re-raising would just be a lost-exception); _failed surfaces it.
                ok = False
            latency_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._outstanding -= 1
                if ok:
                    self._completed += 1
                    self._latencies_ms.append(latency_ms)
                else:
                    self._failed += 1

        try:
            self._pool.submit(wrapped)
            return True
        except RuntimeError:
            # Pool is shutting down — refund the slot and drop.
            with self._lock:
                self._outstanding -= 1
                self._submitted -= 1
                self._dropped += 1
            return False

    def metrics_snapshot(self) -> WriteExecutorMetrics:
        with self._lock:
            return WriteExecutorMetrics(
                outstanding=self._outstanding,
                max_outstanding=self._max_outstanding,
                submitted=self._submitted,
                completed=self._completed,
                dropped=self._dropped,
                failed=self._failed,
                latency_p50_ms=_percentile(self._latencies_ms, 50),
                latency_p95_ms=_percentile(self._latencies_ms, 95),
            )

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


def _percentile(values: deque[float], p: float) -> float:
    if len(values) < 2:
        return 0.0
    # quantiles needs n>=2 for inclusive percentile.
    return statistics.quantiles(list(values), n=100, method="inclusive")[int(p) - 1]
