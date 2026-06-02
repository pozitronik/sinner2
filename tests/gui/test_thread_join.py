"""Tests for the bounded QThread-join helper used at window close.

Regression: closeEvent did quit() + wait(2000) and discarded the bool. The
detection probe's first detection lazily builds the buffalo_l pack (5 ONNX
models + prepare(), can exceed 2s, may download), so a close during that load
timed out and destroyed a still-running QThread ('QThread: Destroyed while
thread is still running'). The helper keeps waiting in bounded increments.
"""
from __future__ import annotations

from sinner2.gui.main_window import _join_qthread


class _FakeThread:
    """Finishes (isRunning -> False) only after `waits_needed` wait() calls."""

    def __init__(self, waits_needed: int) -> None:
        self._waits_needed = waits_needed
        self._waits = 0
        self.quit_called = 0

    def quit(self) -> None:
        self.quit_called += 1

    def isRunning(self) -> bool:
        return self._waits < self._waits_needed

    def wait(self, msecs: int) -> bool:
        self._waits += 1
        return not self.isRunning()


class TestJoinQThread:
    def test_returns_true_when_thread_stops_within_budget(self) -> None:
        t = _FakeThread(waits_needed=3)
        ok = _join_qthread(t, per_wait_ms=2000, max_waits=5)
        assert ok is True
        assert t.quit_called == 1
        assert t._waits == 3  # stopped as soon as it finished, no extra waits

    def test_returns_false_when_thread_never_stops(self) -> None:
        t = _FakeThread(waits_needed=99)
        ok = _join_qthread(t, per_wait_ms=2000, max_waits=3)
        assert ok is False
        assert t._waits == 3  # bounded — does not loop forever

    def test_stops_immediately_if_already_finished(self) -> None:
        t = _FakeThread(waits_needed=0)
        ok = _join_qthread(t, per_wait_ms=2000, max_waits=5)
        assert ok is True
        assert t._waits == 0  # already stopped — no waits at all
