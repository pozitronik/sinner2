"""Tests for the optional A/V sync diagnostic tracer.

The tracer is pure instrumentation: dormant unless SINNER2_SYNC_TRACE is set,
and read-only (it never feeds back into playback). These tests pin the env
gate, the start/stop timer behaviour, and the log line format — without
spinning a real event loop (we invoke _tick directly)."""
from __future__ import annotations

import logging

import pytest

from sinner2.gui import sync_tracer as st
from sinner2.gui.sync_tracer import SyncSample, SyncTracer, sync_trace_enabled


@pytest.fixture(autouse=True)
def _isolate_sync_logger(tmp_path, monkeypatch):
    """Keep each test's logger mutations local: point the trace file at a tmp
    path (no cwd litter) and strip any handlers a test wires onto the shared
    module logger, restoring its level/propagation afterwards."""
    monkeypatch.setattr(st, "_LOG_FILENAME", str(tmp_path / "trace.log"))
    lg = st.logger
    saved_handlers = list(lg.handlers)
    saved_level, saved_propagate = lg.level, lg.propagate
    yield
    for h in list(lg.handlers):
        if h not in saved_handlers:
            lg.removeHandler(h)
            h.close()
    lg.setLevel(saved_level)
    lg.propagate = saved_propagate


class TestEnvGate:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SINNER2_SYNC_TRACE", raising=False)
        assert sync_trace_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "YES", "on", " On "])
    def test_truthy_values_enable(self, monkeypatch, val):
        monkeypatch.setenv("SINNER2_SYNC_TRACE", val)
        assert sync_trace_enabled() is True

    def test_falsey_value_stays_disabled(self, monkeypatch):
        monkeypatch.setenv("SINNER2_SYNC_TRACE", "0")
        assert sync_trace_enabled() is False


class TestStartRespectsEnv:
    def test_start_is_noop_when_disabled(self, qtbot, monkeypatch):
        monkeypatch.delenv("SINNER2_SYNC_TRACE", raising=False)
        tracer = SyncTracer(lambda: None)
        tracer.start()
        assert tracer._timer.isActive() is False  # noqa: SLF001

    def test_start_activates_when_enabled(self, qtbot, monkeypatch):
        monkeypatch.setenv("SINNER2_SYNC_TRACE", "1")
        tracer = SyncTracer(lambda: None)
        tracer.start()
        assert tracer._timer.isActive() is True  # noqa: SLF001
        tracer.stop()
        assert tracer._timer.isActive() is False  # noqa: SLF001


class TestTickLogging:
    def _sample(self) -> SyncSample:
        # Picture trails the playhead: target frame 126, only frame 63 on screen.
        return SyncSample(
            frame=126,
            shown_frame=63,
            video_seconds=2.10,
            audio_seconds=2.38,
            playing=True,
            strategy_mode="synced",
        )

    def test_tick_logs_offset_and_fields(self, qtbot, caplog):
        tracer = SyncTracer(lambda: self._sample())
        with caplog.at_level(logging.INFO, logger="sinner2.sync_trace"):
            tracer._tick()  # noqa: SLF001
        text = caplog.text
        assert "target=126" in text
        assert "shown=63" in text
        assert "offset=+0.280s" in text  # audio - video = 2.38 - 2.10
        assert "lag=63f" in text  # target - shown = 126 - 63
        assert "mode=synced" in text
        assert "playing=1" in text

    def test_tick_skips_when_sample_is_none(self, qtbot, caplog):
        tracer = SyncTracer(lambda: None)
        with caplog.at_level(logging.INFO, logger="sinner2.sync_trace"):
            tracer._tick()  # noqa: SLF001
        assert caplog.text == ""

    def test_negative_offset_when_audio_behind(self, qtbot, caplog):
        sample = SyncSample(
            frame=30,
            shown_frame=30,
            video_seconds=1.0,
            audio_seconds=0.6,
            playing=True,
            strategy_mode="best-effort",
        )
        tracer = SyncTracer(lambda: sample)
        with caplog.at_level(logging.INFO, logger="sinner2.sync_trace"):
            tracer._tick()  # noqa: SLF001
        assert "offset=-0.400s" in caplog.text
        assert "lag=0f" in caplog.text  # target == shown, picture caught up


class TestProductionOutput:
    """The app configures no logging, so without wiring the tracer is mute even
    when enabled. start() must lift the logger to INFO and attach a handler."""

    def test_start_wires_logger_to_emit(self, qtbot, monkeypatch):
        monkeypatch.setenv("SINNER2_SYNC_TRACE", "1")
        tracer = SyncTracer(lambda: None)
        tracer.start()
        try:
            assert st.logger.level == logging.INFO
            assert any(
                getattr(h, st._HANDLER_MARK, False)  # noqa: SLF001
                for h in st.logger.handlers
            )
        finally:
            tracer.stop()

    def test_wiring_is_idempotent(self, qtbot, monkeypatch):
        monkeypatch.setenv("SINNER2_SYNC_TRACE", "1")
        st._ensure_log_output()  # noqa: SLF001
        st._ensure_log_output()  # noqa: SLF001
        marked = [
            h for h in st.logger.handlers
            if getattr(h, st._HANDLER_MARK, False)  # noqa: SLF001
        ]
        # One stderr + one file handler, added once — not stacked per call.
        assert len(marked) == 2

    def test_disabled_start_does_not_wire(self, qtbot, monkeypatch):
        monkeypatch.delenv("SINNER2_SYNC_TRACE", raising=False)
        before = list(st.logger.handlers)
        SyncTracer(lambda: None).start()
        assert list(st.logger.handlers) == before
