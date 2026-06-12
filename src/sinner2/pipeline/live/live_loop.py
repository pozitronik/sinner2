"""The live-camera processing loop: capture -> chain -> sinks + preview.

A single FEEDER thread grabs the latest captured frame, stamps it with a
monotonic sequence number, and hands it to a pool of N WORKER threads that each
run the frame through the processor chain. Results are published by a
generation-gated, latest-completed-wins EMITTER: a finished frame is shown only
if it is newer than the last one already shown (stragglers are dropped), so the
feed never stutters backwards. Latency-first by construction: a bounded
drop-oldest work queue keeps only the freshest frames in flight.

Concurrency model (a lean mirror of RealtimeExecutor, without the timeline /
buffer / seek / cache machinery a finite seekable target needs):
  * The chain is SHARED across workers; thread-unsafe processors are wrapped in
    PerWorkerProcessor upstream (build_chain), so each worker lazily builds its
    own instance.
  * A hot chain swap (set_chain) loads the new chain's models on a side thread,
    installs it atomically under a lock (bumping a generation), then drains the
    OLD generation's in-flight workers before releasing the old chain -- so
    setup()/release() never overlap process() on the same instance (per the
    Processor contract, save the documented timed-out tail).
  * Worker count is adjustable at runtime (set_worker_count); added workers cost
    no chain reload, removed workers release their own per-worker instances.

`on_frame` runs on a worker thread; a GUI caller must marshal it to the GUI
thread (e.g. via a queued Qt signal).
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from queue import Empty, Full, Queue
from typing import Protocol, runtime_checkable

from sinner2.pipeline.live.sink import FrameSink
from sinner2.pipeline.processor import ChainContext, Processor
from sinner2.types import Frame

MAX_LIVE_WORKERS = 16
_POLL_S = 0.2       # worker queue-get timeout (bounds shutdown latency)
_FPS_WINDOW_S = 3.0  # rolling window for the measured-fps readout


@runtime_checkable
class LiveSource(Protocol):
    """Minimal live capture surface (CameraSource satisfies it)."""

    def start(self) -> None:
        ...

    def read(self) -> Frame | None:
        ...

    def stop(self) -> None:
        ...


class _Sentinel:
    """Queue marker that tells a worker to exit promptly."""


_SENTINEL = _Sentinel()

# "No source-face has been requested yet" — distinct from a real source value
# (which may be any object) so the initial chain install knows to skip the
# source reconcile rather than clobbering the swapper with a bogus source.
_UNSET: object = object()

_WorkItem = tuple[int, Frame, int]  # (seq, frame, generation)


class _Worker:
    def __init__(self, thread: threading.Thread, exit_event: threading.Event) -> None:
        self.thread = thread
        self.exit_event = exit_event


class LiveLoop:
    """Owns the run lifecycle of a live session: starts the source + sinks +
    feeder + worker pool on `start()`, tears them all down on `stop()`."""

    def __init__(
        self,
        source: LiveSource,
        chain: list[Processor],
        sinks: list[FrameSink],
        on_frame: Callable[[Frame], None] | None = None,
        fps: int = 30,
        workers: int = 1,
    ) -> None:
        self._source = source
        self._sinks = sinks
        self._on_frame = on_frame
        self._interval = 1.0 / max(1, fps)
        self._initial_chain = chain
        self._initial_workers = max(1, min(workers, MAX_LIVE_WORKERS))
        # Chain hot-swap state. `_active` is what workers process with now
        # (empty == raw passthrough until the first chain is installed).
        self._active: list[Processor] = []
        self._generation = 0
        # Last source-face requested via set_source(), re-applied to every newly
        # installed chain so a source change survives a chain hot-swap (enhancer
        # etc.) and is never lost to a set_source/set_chain install race. Read +
        # written only under _lock.
        self._current_source: object = _UNSET
        self._loop_done = False
        self._lock = threading.Lock()        # guards _active/_generation/_last_emitted_seq/_loop_done + the emit
        self._setup_lock = threading.Lock()  # serialize model loads
        self._last_emitted_seq = -1
        self._delivered = False
        # In-flight tracking, per generation, so a swap waits only for the OLD
        # chain's workers to drain (new-gen work keeps the total non-zero).
        self._inflight_cv = threading.Condition()
        self._inflight_by_gen: dict[int, int] = {}
        # Worker pool.
        self._workers_lock = threading.Lock()
        self._workers: list[_Worker] = []
        self._work_q: Queue[_WorkItem | _Sentinel] = Queue(maxsize=MAX_LIVE_WORKERS + 1)
        self._stopping = threading.Event()
        self._feeder: threading.Thread | None = None
        self._next_seq = 0  # feeder-only
        # FPS measurement (emitted-frame timestamps, trimmed on read).
        self._fps_lock = threading.Lock()
        self._emit_times: deque[float] = deque()
        # Lightweight counters for tests / the live perf readout.
        self.frames_processed = 0  # emitted frames
        self.errors = 0

    # ---- lifecycle ----
    def start(self) -> None:
        for sink in self._sinks:
            sink.start()
        self._source.start()
        with self._workers_lock:
            for _ in range(self._initial_workers):
                self._spawn_worker_locked()
        self._feeder = threading.Thread(
            target=self._feed, name="sinner2-live-feeder", daemon=True
        )
        self._feeder.start()
        # Prepare + install the first chain (raw frames show until it's ready).
        self.set_chain(self._initial_chain)

    def stop(self) -> None:
        self._stopping.set()
        if self._feeder is not None:
            self._feeder.join(timeout=2.0)
        self._drain_queue()
        with self._workers_lock:
            handles = list(self._workers)
        for w in handles:
            w.exit_event.set()
        for _ in handles:  # wake any worker parked on get()
            try:
                self._work_q.put_nowait(_SENTINEL)
            except Full:
                break
        with self._inflight_cv:
            self._inflight_cv.notify_all()
        for w in handles:
            w.thread.join(timeout=30.0)
        with self._lock:
            active, self._active = self._active, []
            self._loop_done = True
        self._release_chain(active)
        self._source.stop()
        for sink in self._sinks:
            sink.stop()

    # ---- chain hot-swap ----
    def set_chain(self, chain: list[Processor]) -> None:
        """Hot-swap the processing chain. Loads the new chain's models on a side
        thread (workers keep running the current chain meanwhile), installs it
        atomically, then drains the outgoing generation before releasing it."""
        def _prepare() -> None:
            if self._stopping.is_set():
                return  # already tearing down; skip the (wasted) model load
            with self._setup_lock:  # one model load at a time
                self._setup_chain(chain)
                with self._lock:
                    if self._loop_done:
                        # Loop gone before we installed -> release our own chain.
                        old_gen: int | None = None
                        to_release: list[Processor] = chain
                        source: object = _UNSET
                    else:
                        old_gen = self._generation
                        to_release, self._active = self._active, chain
                        self._generation += 1
                        self._last_emitted_seq = -1  # new chain's first frame always shows
                        # Snapshot the current source UNDER the same lock that
                        # set_source writes it: whichever critical section runs
                        # second sees the other's effect, so the source can't be
                        # lost to the install race (and survives this hot-swap).
                        source = self._current_source
            if source is not _UNSET:
                self._apply_source_to(chain, source)
            if to_release:
                if old_gen is not None:
                    self._wait_for_gen_drain(old_gen, 5.0)
                self._release_chain(to_release)

        threading.Thread(
            target=_prepare, name="sinner2-live-setup", daemon=True
        ).start()

    def set_source(self, source: object) -> None:
        """Fast source-face change: re-point the active chain's swapper at a new
        source WITHOUT rebuilding the chain, so the enhancer/upscaler per-worker
        instances are NOT torn down + reloaded. Runs on a side thread (the source
        re-analysis is off the caller's thread); no-op if no processor accepts a
        source. The swapper's set_source is internally thread-safe vs workers.

        The source is also persisted (under _lock) so it's re-applied to any chain
        installed later (a hot-swap) and survives a concurrent set_chain install
        without being lost to the read-active/install race."""
        def _apply() -> None:
            with self._lock:
                self._current_source = source
                chain = self._active
            self._apply_source_to(chain, source)

        threading.Thread(
            target=_apply, name="sinner2-live-source", daemon=True
        ).start()

    def _apply_source_to(self, chain: list[Processor], source: object) -> None:
        """Push `source` to every processor in `chain` that accepts one. Called
        off the caller's thread; the swapper's set_source is internally thread-safe
        vs concurrent process() on worker threads."""
        for processor in chain:
            setter = getattr(processor, "set_source", None)
            if not callable(setter):
                continue
            try:
                setter(source)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] set_source failed for "
                      f"{getattr(processor, 'name', '?')}: {exc}",
                      file=sys.stderr)

    def _wait_for_gen_drain(self, gen: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._inflight_cv:
            while self._inflight_by_gen.get(gen, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stopping.is_set():
                    return
                self._inflight_cv.wait(remaining)

    # ---- worker count ----
    def set_worker_count(self, n: int) -> None:
        n = max(1, min(int(n), MAX_LIVE_WORKERS))
        with self._workers_lock:
            self._workers = [w for w in self._workers if w.thread.is_alive()]
            active = [w for w in self._workers if not w.exit_event.is_set()]
            cur = len(active)
            if n > cur:
                for _ in range(n - cur):
                    self._spawn_worker_locked()
            elif n < cur:
                for w in active[n:]:  # surplus workers exit + free their instances
                    w.exit_event.set()

    def _spawn_worker_locked(self) -> None:
        exit_event = threading.Event()
        thread = threading.Thread(
            target=self._worker, args=(exit_event,),
            name="sinner2-live-worker", daemon=True,
        )
        self._workers.append(_Worker(thread, exit_event))
        thread.start()

    def _active_worker_count(self) -> int:
        with self._workers_lock:
            return sum(
                1 for w in self._workers
                if w.thread.is_alive() and not w.exit_event.is_set()
            )

    # ---- feeder (single reader; assigns global seq; bounded drop-oldest) ----
    def _feed(self) -> None:
        while not self._stopping.is_set():
            t0 = time.perf_counter()
            frame = self._source.read()
            if frame is None:
                time.sleep(0.005)  # nothing captured yet
                continue
            with self._lock:
                gen = self._generation
            cap = self._active_worker_count() + 1  # keep ~one frame per worker
            while self._work_q.qsize() >= cap:
                try:
                    self._work_q.get_nowait()  # evict the OLDEST -> stay fresh
                except Empty:
                    break
            seq = self._next_seq
            self._next_seq += 1
            try:
                self._work_q.put_nowait((seq, frame, gen))
            except Full:
                pass
            self._pace(t0)

    # ---- worker ----
    def _worker(self, exit_event: threading.Event) -> None:
        while not (exit_event.is_set() or self._stopping.is_set()):
            try:
                item = self._work_q.get(timeout=_POLL_S)
            except Empty:
                continue
            if isinstance(item, _Sentinel):
                break
            seq, frame, gen = item
            with self._lock:
                if gen != self._generation:
                    continue  # stale-world frame queued before a swap
                chain = self._active
            out = frame
            if chain:
                with self._inflight_cv:
                    self._inflight_by_gen[gen] = self._inflight_by_gen.get(gen, 0) + 1
                try:
                    # One ChainContext per frame: detect-once-share-faces,
                    # mirroring RealtimeExecutor._apply_chain.
                    ctx = ChainContext()
                    for processor in chain:
                        if getattr(processor, "accepts_context", False):
                            out = processor.process(out, ctx)  # type: ignore[call-arg]
                        else:
                            out = processor.process(out)
                except Exception as exc:  # noqa: BLE001 — show raw, don't freeze
                    self.errors += 1
                    if self.errors <= 3:  # log the first few, then stay quiet
                        print(f"[live] chain error (showing raw frame): {exc}",
                              file=sys.stderr)
                    out = frame
                finally:
                    with self._inflight_cv:
                        self._inflight_by_gen[gen] -= 1
                        if self._inflight_by_gen[gen] == 0:
                            del self._inflight_by_gen[gen]
                        self._inflight_cv.notify_all()
            self._maybe_emit(seq, gen, out)
        # On exit, free this worker's per-worker instances from the live chain.
        with self._lock:
            final_chain = self._active
        self._release_thread_local(final_chain)

    def _maybe_emit(self, seq: int, gen: int, out: Frame) -> None:
        with self._lock:
            if gen != self._generation or seq <= self._last_emitted_seq:
                return  # stale world, or a straggler older than one already shown
            self._last_emitted_seq = seq
            for sink in self._sinks:
                sink.push(out)
            if self._on_frame is not None:
                self._on_frame(out)
            self.frames_processed += 1
            if not self._delivered:
                self._delivered = True
                print(f"[live] first frame delivered to {len(self._sinks)} "
                      "sink(s) + preview", file=sys.stderr)
        with self._fps_lock:
            self._emit_times.append(time.monotonic())

    # ---- fps (lazy, trim-on-read) ----
    def measured_fps(self) -> float:
        cutoff = time.monotonic() - _FPS_WINDOW_S
        with self._fps_lock:
            while self._emit_times and self._emit_times[0] < cutoff:
                self._emit_times.popleft()
            n = len(self._emit_times)
            if n < 2:
                return 0.0
            span = self._emit_times[-1] - self._emit_times[0]
            return (n - 1) / span if span > 0 else 0.0

    # ---- chain setup / teardown ----
    def _setup_chain(self, chain: list[Processor]) -> None:
        if chain:
            print("[live] loading chain (models)...", file=sys.stderr)
        for processor in chain:
            try:
                processor.setup()
            except Exception as exc:  # noqa: BLE001
                print(f"[live] setup failed for "
                      f"{getattr(processor, 'name', '?')}: {exc}", file=sys.stderr)
        if chain:
            print("[live] chain ready", file=sys.stderr)

    def _release_chain(self, chain: list[Processor]) -> None:
        for processor in chain:
            try:
                processor.release()
            except Exception:  # noqa: BLE001
                pass

    def _release_thread_local(self, chain: list[Processor]) -> None:
        for processor in chain:
            rel = getattr(processor, "release_thread_local", None)
            if callable(rel):
                try:
                    rel()
                except Exception:  # noqa: BLE001
                    pass

    def _drain_queue(self) -> None:
        while True:
            try:
                self._work_q.get_nowait()
            except Empty:
                break

    def _pace(self, t0: float) -> None:
        """Cap the feeder at the fps target; no-op when already running behind."""
        remaining = self._interval - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
