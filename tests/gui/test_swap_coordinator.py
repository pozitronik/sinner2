"""Tests for SwapCoordinator — the async source/target swap sequencer.

The coordinator owns only the *when/how* of a live swap: it coalesces requests
arriving mid-swap (latest-wins), spawns the worker, hops the result back to the
GUI thread via a Qt signal, and tracks the swapping/pending state. The actual
*work* is injected as callbacks (``run_job`` on the worker, ``on_complete`` /
``on_begin`` / ``on_switching`` on the GUI), so the whole flow is exercisable
with recording stubs and a manual spawn.

The real ``spawn`` fires a daemon thread; here a ``_ManualSpawn`` captures each
job instead and runs it on demand, so the worker→GUI ``_ready.emit`` happens as
a same-thread direct connection — fully synchronous and deterministic. ``qtbot``
is requested only to guarantee a QApplication exists for the signal machinery.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.gui.swap_coordinator import SwapCoordinator, _SwapOutcome

SRC_A, TGT_A = Path("a/source.png"), Path("a/target.mp4")
SRC_B, TGT_B = Path("b/source.png"), Path("b/target.mp4")
SRC_C, TGT_C = Path("c/source.png"), Path("c/target.mp4")


class _FakeThread:
    """Stand-in for the worker thread; records join() calls and timeouts."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        self._alive = False


class _ManualSpawn:
    """Captures each swap job instead of threading it, so the async hop runs on
    demand. ``__call__`` (the spawn hook) stores the job and returns a fake
    thread; ``run_next`` runs the oldest captured job, which fires
    ``_ready.emit`` → ``_on_ready`` synchronously (same-thread direct connection)."""

    def __init__(self) -> None:
        self.jobs: list = []
        self.threads: list[_FakeThread] = []

    def __call__(self, fn) -> _FakeThread:
        self.jobs.append(fn)
        thread = _FakeThread()
        self.threads.append(thread)
        return thread

    def pending(self) -> int:
        return len(self.jobs)

    def run_next(self) -> None:
        self.jobs.pop(0)()


class _Recorder:
    """Records every coordinator callback in one ordered log so both counts and
    relative ordering are assertable. ``run_job`` returns a preset outcome."""

    def __init__(self, outcome: _SwapOutcome) -> None:
        self.outcome = outcome
        self.events: list[tuple] = []
        self.run_job_args: list[tuple[Path, Path]] = []
        self.completed: list[_SwapOutcome] = []

    def run_job(self, source: Path, target: Path) -> _SwapOutcome:
        self.run_job_args.append((source, target))
        self.events.append(("job", (source, target)))
        return self.outcome

    def on_complete(self, outcome: _SwapOutcome) -> None:
        self.completed.append(outcome)
        self.events.append(("complete", outcome))

    def on_begin(self) -> None:
        self.events.append(("begin", None))

    def on_switching(self, flag: bool) -> None:
        self.events.append(("switching", flag))

    def begin_count(self) -> int:
        return sum(1 for kind, _ in self.events if kind == "begin")


@pytest.fixture
def make_harness(qtbot):
    """Factory: build a coordinator wired to a fresh recorder + manual spawn.
    Pass an explicit outcome to exercise the error path."""

    def _make(outcome: _SwapOutcome | None = None):
        rec = _Recorder(outcome if outcome is not None else _SwapOutcome(bundle=object()))
        spawn = _ManualSpawn()
        coord = SwapCoordinator(
            run_job=rec.run_job,
            on_complete=rec.on_complete,
            on_begin=rec.on_begin,
            on_switching=rec.on_switching,
            spawn=spawn,
        )
        return coord, rec, spawn

    return _make


class TestRequestBegin:
    def test_idle_request_begins_and_defers_work(self, make_harness):
        """A request from idle flips to swapping, signals switching + begin, and
        captures (but does not yet run) the worker job."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        assert coord.swapping is True
        assert spawn.pending() == 1
        assert rec.run_job_args == []  # worker hasn't run
        assert rec.events == [("switching", True), ("begin", None)]

    def test_completion_resets_state_and_runs_work(self, make_harness):
        """Running the captured job completes the swap: state resets and the
        worker + completion callbacks fire exactly once."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        spawn.run_next()
        assert coord.swapping is False
        assert rec.run_job_args == [(SRC_A, TGT_A)]
        assert rec.completed == [rec.outcome]

    def test_request_after_completion_begins_fresh(self, make_harness):
        """A request after a finished swap is a brand-new swap, not a coalesce."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        spawn.run_next()
        coord.request(SRC_B, TGT_B)
        assert coord.swapping is True
        assert spawn.pending() == 1
        assert rec.begin_count() == 2
        spawn.run_next()
        assert rec.run_job_args == [(SRC_A, TGT_A), (SRC_B, TGT_B)]

    def test_swapping_reflects_lifecycle(self, make_harness):
        coord, _rec, spawn = make_harness()
        assert coord.swapping is False
        coord.request(SRC_A, TGT_A)
        assert coord.swapping is True
        spawn.run_next()
        assert coord.swapping is False


class TestEventOrdering:
    def test_happy_path_event_sequence(self, make_harness):
        """The bug-sensitive order: switching(True) → begin → job → complete →
        switching(False). on_complete must precede switching(False) (the new
        world is adopted before the swap is declared done)."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        spawn.run_next()
        assert rec.events == [
            ("switching", True),
            ("begin", None),
            ("job", (SRC_A, TGT_A)),
            ("complete", rec.outcome),
            ("switching", False),
        ]


class TestCoalescing:
    def test_request_during_swap_coalesces_not_spawns(self, make_harness):
        """A request mid-swap is stashed, not started: no second worker, still
        one swap, on_begin fired only once."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        coord.request(SRC_B, TGT_B)
        assert spawn.pending() == 1
        assert coord.swapping is True
        assert rec.begin_count() == 1
        assert rec.run_job_args == []

    def test_pending_runs_after_completion(self, make_harness):
        """Completing the in-flight swap automatically begins the coalesced one."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        coord.request(SRC_B, TGT_B)
        spawn.run_next()  # A completes → B auto-begins
        assert coord.swapping is True
        assert spawn.pending() == 1
        spawn.run_next()  # B completes
        assert coord.swapping is False
        assert rec.run_job_args == [(SRC_A, TGT_A), (SRC_B, TGT_B)]

    def test_latest_pending_wins(self, make_harness):
        """Several requests during one swap collapse to the last — B is dropped,
        C runs (callers update their restore state before each request)."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        coord.request(SRC_B, TGT_B)
        coord.request(SRC_C, TGT_C)
        spawn.run_next()  # A done → C begins, B discarded
        spawn.run_next()  # C done
        assert rec.run_job_args == [(SRC_A, TGT_A), (SRC_C, TGT_C)]

    def test_on_begin_per_actual_swap_not_per_request(self, make_harness):
        """Three requests, two real swaps (A then C) → begin fires twice."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        coord.request(SRC_B, TGT_B)
        coord.request(SRC_C, TGT_C)
        spawn.run_next()
        spawn.run_next()
        assert rec.begin_count() == 2


class TestErrorOutcome:
    def test_error_outcome_still_completes_state_machine(self, make_harness):
        """A worker that returns an error still drives the full completion: the
        outcome reaches on_complete and the coordinator returns to idle."""
        coord, rec, spawn = make_harness(_SwapOutcome(error="boom"))
        coord.request(SRC_A, TGT_A)
        spawn.run_next()
        assert coord.swapping is False
        assert rec.completed == [rec.outcome]
        assert rec.completed[0].error == "boom"
        assert rec.completed[0].bundle is None
        assert rec.events[-1] == ("switching", False)


class TestSpawnProperty:
    def test_getter_returns_injected_spawn(self, make_harness):
        coord, _rec, spawn = make_harness()
        assert coord.spawn is spawn

    def test_setter_replaces_spawn(self, make_harness):
        coord, _rec, _spawn = make_harness()
        replacement = _ManualSpawn()
        coord.spawn = replacement
        assert coord.spawn is replacement
        coord.request(SRC_A, TGT_A)
        assert replacement.pending() == 1  # the new spawn was used


class TestCancelPendingAndJoin:
    def test_clears_pending_so_no_followup_swap(self, make_harness):
        """Cancelling drops the coalesced request: completing the in-flight swap
        does not start the dropped one."""
        coord, rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        coord.request(SRC_B, TGT_B)
        coord.cancel_pending_and_join(0.1)
        spawn.run_next()  # A completes; B was cancelled
        assert coord.swapping is False
        assert spawn.pending() == 0
        assert rec.run_job_args == [(SRC_A, TGT_A)]

    def test_joins_in_flight_thread_with_timeout(self, make_harness):
        """An alive worker thread is joined with the given timeout so teardown
        doesn't race a session still being built."""
        coord, _rec, spawn = make_harness()
        coord.request(SRC_A, TGT_A)
        coord.cancel_pending_and_join(2.5)
        assert spawn.threads[0].join_calls == [2.5]

    def test_noop_when_no_thread(self, make_harness):
        """No in-flight worker → nothing to join, no error."""
        coord, _rec, spawn = make_harness()
        coord.cancel_pending_and_join(1.0)
        assert spawn.pending() == 0
        assert spawn.threads == []

    def test_does_not_join_dead_thread(self, make_harness):
        """A finished-but-unreferenced worker (not alive) is not joined."""
        coord, _rec, _spawn = make_harness()
        dead = _FakeThread(alive=False)
        coord._thread = dead  # noqa: SLF001 — simulate a completed worker
        coord.cancel_pending_and_join(1.0)
        assert dead.join_calls == []
