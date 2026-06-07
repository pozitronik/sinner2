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
        self._chain = chain
        self._sinks = sinks
        self._on_frame = on_frame
        self._interval = 1.0 / max(1, fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
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

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._source.stop()
        for sink in self._sinks:
            sink.stop()

    def _run(self) -> None:
        # Load the chain's models on a side thread so the RAW camera shows
        # immediately; the chain is applied only once setup completes. Per the
        # Processor contract, setup() must not run concurrently with process() —
        # the loop only processes after `ready` is set (i.e. setup has finished).
        ready = threading.Event()

        def _setup() -> None:
            self._setup_chain()
            ready.set()

        threading.Thread(
            target=_setup, name="sinner2-live-setup", daemon=True
        ).start()
        delivered = False
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                frame = self._source.read()
                if frame is None:
                    time.sleep(0.005)  # nothing captured yet
                    continue
                out = frame
                if ready.is_set():
                    try:
                        for processor in self._chain:
                            out = processor.process(out)
                    except Exception as exc:  # noqa: BLE001 — show raw, don't freeze
                        self.errors += 1
                        if self.errors <= 3:  # log the first few, then stay quiet
                            print(f"[live] chain error (showing raw frame): {exc}",
                                  file=sys.stderr)
                        out = frame
                # else: raw passthrough while the models are still loading.
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
            # Let setup finish before releasing so the two never overlap.
            ready.wait(timeout=15.0)
            self._release_chain()

    def _setup_chain(self) -> None:
        if self._chain:
            print("[live] loading chain (models)...", file=sys.stderr)
        for processor in self._chain:
            try:
                processor.setup()
            except Exception as exc:  # noqa: BLE001
                print(f"[live] setup failed for "
                      f"{getattr(processor, 'name', '?')}: {exc}", file=sys.stderr)
        if self._chain:
            print("[live] chain ready", file=sys.stderr)

    def _release_chain(self) -> None:
        for processor in self._chain:
            try:
                processor.release()
            except Exception:  # noqa: BLE001
                pass

    def _pace(self, t0: float) -> None:
        """Cap the loop at the fps target; no-op when already running behind."""
        remaining = self._interval - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
