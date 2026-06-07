"""The live-camera processing loop: capture -> chain -> sinks + preview.

A single processing thread grabs the LATEST captured frame, runs it through the
processor chain, and pushes the result to every sink plus an optional preview
callback. Latency-first by construction: it always reads the freshest frame, so
when processing can't keep up, intermediate frames are simply dropped (the
camera source overwrites its latest) rather than queued into growing lag.

One processing thread (not a pool) is deliberate for the MVP: on a single GPU
the chain serializes anyway, and a pool would reorder frames (unacceptable for a
live feed). A heavy chain just lowers the achievable fps. Because it's
single-threaded, thread-unsafe processors are safe here without per-worker copies.

`on_frame` runs on the loop thread; a GUI caller must marshal it to the GUI
thread (e.g. via a queued Qt signal).
"""
from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sinner2.pipeline.live.sink import FrameSink
from sinner2.pipeline.processor import Processor
from sinner2.types import Frame


@runtime_checkable
class LiveSource(Protocol):
    """Minimal live capture surface (CameraSource satisfies it)."""

    def start(self) -> None:
        ...

    def read(self) -> Frame | None:
        ...

    def stop(self) -> None:
        ...


class LiveLoop:
    """Owns the run lifecycle of a live session: starts the source + sinks +
    processing thread on `start()`, tears them all down on `stop()`."""

    def __init__(
        self,
        source: LiveSource,
        chain: list[Processor],
        sinks: list[FrameSink],
        on_frame: Callable[[Frame], None] | None = None,
        fps: int = 30,
    ) -> None:
        self._source = source
        self._sinks = sinks
        self._on_frame = on_frame
        self._interval = 1.0 / max(1, fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Chain hot-swap state. `_active` is what the loop processes with right
        # now (empty == raw passthrough until the first chain is installed);
        # `_pending` is a fully set-up chain waiting to be installed at a frame
        # boundary. `set_chain()` prepares into `_pending`; the loop installs it.
        self._initial_chain = chain
        self._active: list[Processor] = []
        self._pending: list[Processor] | None = None
        self._loop_done = False  # set under _lock when the loop has torn down
        self._lock = threading.Lock()
        self._setup_lock = threading.Lock()  # serialize model loads
        # Lightweight counters for tests / a future live perf readout.
        self.frames_processed = 0
        self.errors = 0

    def start(self) -> None:
        for sink in self._sinks:
            sink.start()
        self._source.start()
        self._thread = threading.Thread(
            target=self._run, name="sinner2-live", daemon=True
        )
        self._thread.start()
        # Prepare + install the first chain (raw frames show until it's ready).
        self.set_chain(self._initial_chain)

    def set_chain(self, chain: list[Processor]) -> None:
        """Hot-swap the processing chain. Loads the new chain's models on a side
        thread (the loop keeps running the current chain meanwhile), then queues
        it; the loop installs it + releases the old chain at a frame boundary."""
        def _prepare() -> None:
            if self._stop.is_set():
                return  # already tearing down; skip the (wasted) model load
            with self._setup_lock:  # one model load at a time
                self._setup_chain(chain)
                with self._lock:
                    if self._loop_done:
                        # The loop finished + drained pending before we got here,
                        # so it will never install/release this chain — we must.
                        to_release: list[Processor] | None = chain
                    else:
                        to_release, self._pending = self._pending, chain
            if to_release is not None:  # superseded un-installed chain (or our own)
                self._release_chain(to_release)

        threading.Thread(
            target=_prepare, name="sinner2-live-setup", daemon=True
        ).start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._source.stop()
        for sink in self._sinks:
            sink.stop()

    def _run(self) -> None:
        delivered = False
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                self._install_pending()  # swap in a freshly-built chain, if any
                frame = self._source.read()
                if frame is None:
                    time.sleep(0.005)  # nothing captured yet
                    continue
                out = frame
                chain = self._active
                if chain:
                    try:
                        for processor in chain:
                            out = processor.process(out)
                    except Exception as exc:  # noqa: BLE001 — show raw, don't freeze
                        self.errors += 1
                        if self.errors <= 3:  # log the first few, then stay quiet
                            print(f"[live] chain error (showing raw frame): {exc}",
                                  file=sys.stderr)
                        out = frame
                # else: raw passthrough until the first chain is installed.
                for sink in self._sinks:
                    sink.push(out)
                if self._on_frame is not None:
                    self._on_frame(out)
                if not delivered:
                    delivered = True
                    print(f"[live] first frame delivered to {len(self._sinks)} "
                          "sink(s) + preview", file=sys.stderr)
                self.frames_processed += 1
                self._pace(t0)
        finally:
            self._release_chain(self._active)
            with self._lock:
                pending, self._pending = self._pending, None
                self._loop_done = True  # any later set_chain releases its own chain
            if pending is not None:
                self._release_chain(pending)

    def _install_pending(self) -> None:
        """On the loop thread, between frames: swap in the pending chain and
        release the outgoing one (never overlaps process() — same thread)."""
        with self._lock:
            if self._pending is None:
                return
            new_chain, self._pending = self._pending, None
        old = self._active
        self._active = new_chain
        self._release_chain(old)

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

    def _pace(self, t0: float) -> None:
        """Cap the loop at the fps target; no-op when already running behind."""
        remaining = self._interval - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
