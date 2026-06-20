"""Unit tests for ChainRunner — per-frame chain execution + per-thread cleanup.

The ChainContext-flow / had-faces / fresh-context cases and the thread-local
release cases were migrated verbatim (same assertions) from test_executor.py
when the logic moved here, now driven against ChainRunner directly with a real
TelemetryCollector as the timing sink.
"""
from __future__ import annotations

import numpy as np

from sinner2.pipeline.realtime.chain_runner import ChainRunner
from sinner2.pipeline.realtime.telemetry import TelemetryCollector


def _runner():
    """A ChainRunner with a real telemetry sink and a list-backed error sink."""
    errors: list[str] = []
    return ChainRunner(TelemetryCollector(), on_error=errors.append), errors


class TestApplyContext:
    """apply() feeds ONE ChainContext per frame to context-aware processors
    (detect-once-share-faces); plain processors keep the one-argument call."""

    def test_context_flows_between_context_aware_processors(self):
        class _Producer:
            name = "producer"
            accepts_context = True

            def process(self, frame, ctx=None):
                ctx.faces = ["FACE"]
                return frame

        class _Plain:
            name = "plain"

            def process(self, frame):
                return frame

        class _Consumer:
            name = "consumer"
            accepts_context = True

            def __init__(self):
                self.seen = "unset"

            def process(self, frame, ctx=None):
                self.seen = ctx.faces
                return frame

        runner, _ = _runner()
        consumer = _Consumer()
        frame = np.zeros((4, 4, 3), np.uint8)
        runner.apply(frame, (_Producer(), _Plain(), consumer))
        assert consumer.seen == ["FACE"]

    def test_fresh_context_per_frame(self):
        class _Consumer:
            name = "consumer"
            accepts_context = True

            def __init__(self):
                self.seen: list = []

            def process(self, frame, ctx=None):
                self.seen.append(ctx.faces)
                ctx.faces = ["stale"]
                return frame

        runner, _ = _runner()
        consumer = _Consumer()
        frame = np.zeros((4, 4, 3), np.uint8)
        runner.apply(frame, (consumer,))
        runner.apply(frame, (consumer,))
        # The second frame's context starts clean — nothing leaks across frames.
        assert consumer.seen == [None, None]

    def test_reports_had_faces_when_swapper_detected_some(self):
        class _Producer:
            name = "p"
            accepts_context = True

            def process(self, frame, ctx=None):
                ctx.faces = ["FACE"]
                return frame

        runner, _ = _runner()
        frame = np.zeros((4, 4, 3), np.uint8)
        _result, had_faces, detection_ran = runner.apply(frame, (_Producer(),))
        assert had_faces is True
        assert detection_ran is True

    def test_reports_no_faces_for_empty_detection(self):
        class _Empty:
            name = "p"
            accepts_context = True

            def process(self, frame, ctx=None):
                ctx.faces = []  # detection ran, found nothing
                return frame

        runner, _ = _runner()
        frame = np.zeros((4, 4, 3), np.uint8)
        _result, had_faces, detection_ran = runner.apply(frame, (_Empty(),))
        assert had_faces is False
        assert detection_ran is True  # ran, found nothing → a problem frame

    def test_reports_no_faces_when_no_detection_ran(self):
        class _Plain:
            name = "p"

            def process(self, frame):
                return frame

        runner, _ = _runner()
        frame = np.zeros((4, 4, 3), np.uint8)
        _result, had_faces, detection_ran = runner.apply(frame, (_Plain(),))
        assert had_faces is False  # ctx.faces stays None
        assert detection_ran is False  # didn't look → NOT a problem frame

    def test_records_one_timing_sample_per_process_call(self):
        telemetry = TelemetryCollector()
        runner = ChainRunner(telemetry, on_error=lambda _m: None)

        class _P:
            name = "P"

            def process(self, frame):
                return frame

        frame = np.zeros((4, 4, 3), np.uint8)
        runner.apply(frame, (_P(), _P()))
        with telemetry._timings_lock:  # noqa: SLF001
            names = [n for (_ts, n, _ns) in telemetry._processor_timings]  # noqa: SLF001
        assert names == ["P", "P"]


class TestReleaseThreadLocal:
    """On worker exit, release per-thread processor instances
    (PerWorkerProcessor's GFPGAN); plain shared processors are skipped, and
    errors are swallowed into the status sink."""

    def test_invokes_release_thread_local_on_wrappers_only(self):
        calls: list[str] = []

        class _Wrapper:
            name = "enh"

            def release_thread_local(self) -> None:
                calls.append("released")

        class _Plain:
            name = "swap"  # no release_thread_local → skipped

        runner, _ = _runner()
        runner.release_thread_local((_Plain(), _Wrapper()))
        assert calls == ["released"]

    def test_swallows_release_errors_into_status(self):
        class _Boom:
            name = "enh"

            def release_thread_local(self) -> None:
                raise RuntimeError("kaboom")

        runner, errors = _runner()
        runner.release_thread_local((_Boom(),))  # must not raise
        assert any("kaboom" in e for e in errors)

    def test_nothing_to_release_is_safe(self):
        class _Plain:
            name = "swap"

        runner, errors = _runner()
        runner.release_thread_local((_Plain(),))  # no-op, no errors
        assert errors == []
