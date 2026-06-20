"""Per-frame chain execution for RealtimeExecutor.

Runs one source frame through the processor chain and reports per-processor
wall-clock to the telemetry collector, plus the per-thread cleanup a shrinking
worker pool needs. Stateless w.r.t. the chain: the executor re-reads its shared
chain every worker iteration so a ``set_chain`` swap is picked up without
restarting the worker, and passes that snapshot in per call. Holds only the
timing sink and a status sink (for best-effort thread-local release errors).
"""
from __future__ import annotations

import time
from collections.abc import Callable

from sinner2.pipeline.processor import ChainContext, Processor
from sinner2.pipeline.realtime.telemetry import TelemetryCollector
from sinner2.types import Frame


class ChainRunner:
    def __init__(
        self,
        telemetry: TelemetryCollector,
        on_error: Callable[[str], None],
    ) -> None:
        self._telemetry = telemetry
        self._on_error = on_error

    def apply(
        self,
        frame: Frame,
        chain: tuple[Processor, ...],
        frame_index: int | None = None,
    ) -> tuple[Frame, bool, bool]:
        """Run the chain over one frame; return (result, had_faces, detection_ran).

        One ChainContext per frame: the swapper publishes its detections,
        downstream context-aware processors (enhancer ONNX backends) reuse them
        instead of re-detecting — one detection pass per frame. The frame index
        lets face-mapping read its precomputed per-frame geometry. Each
        process() call is wrapped with perf_counter so the metrics overlay can
        attribute wall-clock per processor — strictly the .process() call, not
        the chain-iteration overhead or the buffer.put downstream."""
        ctx = ChainContext(frame_index=frame_index)
        for p in chain:
            t0 = time.perf_counter_ns()
            if getattr(p, "accepts_context", False):
                frame = p.process(frame, ctx)  # type: ignore[call-arg]
            else:
                frame = p.process(frame)
            elapsed_ns = time.perf_counter_ns() - t0
            self._telemetry.record_processor_timing(p.name, elapsed_ns)
        # ctx.faces: None = no detection ran, [] = no faces, [..] = faces. The
        # preprocessing head-start sizes off how fast FACE frames render (the
        # expensive ones), so report whether this frame carried any AND whether
        # detection actually ran (so the visualiser can tell "no face found"
        # apart from "didn't look" — an enhancer-only or detection-skip frame).
        return frame, bool(ctx.faces), ctx.faces is not None

    def release_thread_local(self, chain: tuple[Processor, ...]) -> None:
        """Release any per-thread processor instances the calling (exiting)
        worker built — e.g. a PerWorkerProcessor's own GFPGAN. Plain shared
        processors don't expose release_thread_local() and are skipped. On a
        full stop() the chain is released wholesale anyway; this matters for the
        live worker-count-DECREASE path, where the surplus worker's model would
        otherwise linger until the next chain swap."""
        for p in chain:
            release = getattr(p, "release_thread_local", None)
            if release is None:
                continue
            try:
                release()
            except Exception as e:  # noqa: BLE001
                self._on_error(f"thread-local release error: {e}")
