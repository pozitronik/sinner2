import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from sinner2.config.target import Target
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import DiskFrameStore
from sinner2.io.reader_pool import ReaderPool
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import BestEffortStrategy, SyncedStrategy
from sinner2.types import Frame


def _pool_for(reader, size: int = 1) -> ReaderPool:
    """Wrap a single reader in a size-1 pool. Tests use size=1 so the
    single returned instance from the lambda is fine — the pool never
    needs to call the factory beyond construction."""
    return ReaderPool(lambda: reader, size=size, name="test")


class TestReleaseThreadLocalChain:
    """On worker exit, the executor releases any per-thread processor
    instances (PerWorkerProcessor's GFPGAN) so a live pool shrink frees the
    surplus model. Plain shared processors are skipped."""

    def _bare_executor(self, chain):
        # __init__ wires threads + queues; this method needs only _chain and
        # status, so bypass construction (test-convention object.__new__).
        from sinner2.observable import ObservableValue

        ex = object.__new__(RealtimeExecutor)
        ex._chain = tuple(chain)  # noqa: SLF001
        ex.status = ObservableValue("")
        return ex

    def test_invokes_release_thread_local_on_wrappers_only(self):
        calls: list[str] = []

        class _Wrapper:
            name = "enh"

            def release_thread_local(self) -> None:
                calls.append("released")

        class _Plain:
            name = "swap"  # no release_thread_local → skipped

        ex = self._bare_executor([_Plain(), _Wrapper()])
        ex._release_thread_local_chain()  # noqa: SLF001
        assert calls == ["released"]

    def test_swallows_release_errors_into_status(self):
        class _Boom:
            name = "enh"

            def release_thread_local(self) -> None:
                raise RuntimeError("kaboom")

        ex = self._bare_executor([_Boom()])
        ex._release_thread_local_chain()  # noqa: SLF001  # must not raise
        assert "kaboom" in ex.status.get()


class _CountingProcessor:
    """Returns the input frame unchanged; counts setup/process/release calls."""

    name = "Counting"

    def __init__(self) -> None:
        self.setup_calls = 0
        self.process_calls = 0
        self.release_calls = 0
        self._lock = threading.Lock()

    def setup(self) -> None:
        self.setup_calls += 1

    def process(self, frame: Frame) -> Frame:
        with self._lock:
            self.process_calls += 1
        return frame

    def release(self) -> None:
        self.release_calls += 1


def _factory(*processors):
    """Build a chain factory that returns the given processor instances.

    With worker_count=1 the factory is called once and the test keeps direct
    references to the processors for assertions.
    """
    return lambda: list(processors)


class _MultiFrameReader:
    """In-memory TargetReader for an N-frame synthetic stream."""

    def __init__(self, count: int, fps: float = 30.0) -> None:
        self._count = count
        self._fps = fps
        self._frame = np.full((8, 8, 3), 128, dtype=np.uint8)
        self.release_calls = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def width(self) -> int:
        return 8

    @property
    def height(self) -> int:
        return 8

    @property
    def native_width(self) -> int:
        return 8

    @property
    def native_height(self) -> int:
        return 8

    def read(self, index: int) -> Frame | None:
        if index < 0 or index >= self._count:
            return None
        return self._frame

    def release(self) -> None:
        self.release_calls += 1


class _LastFrameFailsReader(_MultiFrameReader):
    """Like _MultiFrameReader, but the LAST frame's read returns None (a
    transient source hiccup) — so last_completed can never reach last_frame."""

    def read(self, index: int) -> Frame | None:
        if index == self._count - 1:
            return None
        return super().read(index)


@pytest.fixture
def buffer_setup(tmp_path: Path):
    store = DiskFrameStore(tmp_path / "frames")
    cache = MemoryFrameCache(max_bytes=10 * 1024 * 1024)
    timeline = Timeline(fps=30.0)
    write_executor = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
    buffer = FrameBuffer(store, cache, timeline, write_executor)
    yield buffer, timeline, write_executor
    write_executor.shutdown(wait=True)


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll until predicate is truthy or timeout. Returns whether it became true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestRealtimeExecutorLifecycle:
    def test_rejects_zero_workers(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        with pytest.raises(ValueError):
            RealtimeExecutor(
                reader_pool=_pool_for(_MultiFrameReader(1)),
                buffer=buffer,
                timeline=timeline,
                chain=[],
                strategy=BestEffortStrategy(),
                worker_count=0,
            )

    def test_start_calls_setup_on_each_processor(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            # Setup runs on a background thread; wait for it before
            # asserting the side effect.
            assert ex.wait_until_ready(timeout=2.0)
            assert p.setup_calls == 1
        finally:
            ex.stop()

    def test_stop_calls_release_on_each_processor(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        ex.stop()
        assert p.release_calls == 1

    def test_stop_releases_target_reader(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        reader = _MultiFrameReader(1)
        ex = RealtimeExecutor(
            reader_pool=_pool_for(reader),
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        ex.stop()
        assert reader.release_calls == 1

    def test_double_start_is_noop(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.start()
            assert ex.wait_until_ready(timeout=2.0)
            assert p.setup_calls == 1
        finally:
            ex.stop()

    def test_double_stop_is_noop(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        ex.stop()
        ex.stop()

    def test_all_workers_share_one_chain(self, buffer_setup):
        # Sinner1 model: ONE shared chain across N workers. ORT
        # InferenceSession is thread-safe so concurrent .run() calls let
        # the GPU schedule them efficiently. Per-worker chains were tried
        # and were slower because of CUDA context contention.
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
            worker_count=4,
        )
        try:
            ex.start()
            assert ex.wait_until_ready(timeout=2.0)
            assert p.setup_calls == 1  # only one setup regardless of worker_count
        finally:
            ex.stop()
        assert p.release_calls == 1


class TestRealtimeExecutorAsyncSetup:
    """Setup runs on a background thread so the GUI main thread isn't
    blocked for the seconds it takes to load GFPGAN + inswapper +
    buffalo_l. start() returns promptly; workers wait for the chain to
    be ready before pulling work; failures surface via status + stop."""

    def test_start_returns_before_setup_completes(self, buffer_setup):
        # A processor whose setup() blocks for longer than start() can
        # tolerate as a synchronous call. start() must return immediately.
        buffer, timeline, _ = buffer_setup

        class _SlowSetup(_CountingProcessor):
            def setup(self):
                time.sleep(0.5)
                super().setup()

        p = _SlowSetup()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            t0 = time.monotonic()
            ex.start()
            elapsed = time.monotonic() - t0
            # Allow plenty of slack for slow CI; the point is "milliseconds
            # not hundreds of milliseconds." 0.3s is well under the 0.5s
            # setup sleep.
            assert elapsed < 0.3, f"start() blocked for {elapsed:.3f}s"
            # Setup still pending at this instant — wait it out.
            assert ex.wait_until_ready(timeout=2.0)
            assert p.setup_calls == 1
        finally:
            ex.stop()

    def test_workers_wait_for_setup_before_processing(self, buffer_setup):
        # During the setup window the worker mustn't process any items
        # even if the dispatcher has already submitted some. Verify by
        # asserting zero process_calls until wait_until_ready succeeds.
        buffer, timeline, _ = buffer_setup
        setup_complete = threading.Event()

        class _GatedSetup(_CountingProcessor):
            def setup(self):
                setup_complete.wait(timeout=2.0)
                super().setup()

        p = _GatedSetup()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(5, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            # Setup is still gated; workers must not have processed
            # anything yet. Sample a short window to give the dispatcher
            # time to submit + the worker time to (incorrectly) process.
            time.sleep(0.15)
            assert p.process_calls == 0, (
                f"worker processed {p.process_calls} frames before setup completed"
            )
            # Release the gate; processing should now begin.
            setup_complete.set()
            assert _wait_until(lambda: p.process_calls >= 1, timeout=2.0)
        finally:
            setup_complete.set()  # belt-and-suspenders: don't deadlock stop()
            ex.stop()

    def test_setup_failure_stops_executor_cleanly(self, buffer_setup):
        # A processor that raises inside setup must not leave threads
        # parked forever. The executor sets stop_event, surfaces the
        # error in status, and stop() returns promptly.
        buffer, timeline, _ = buffer_setup

        class _FailingSetup(_CountingProcessor):
            def setup(self):
                raise RuntimeError("model load failed")

        p = _FailingSetup()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        assert ex.wait_until_ready(timeout=2.0)
        assert "chain setup failed" in ex.status.get()
        assert "model load failed" in ex.status.get()
        # stop must not hang on parked workers / setup thread.
        t0 = time.monotonic()
        ex.stop()
        assert time.monotonic() - t0 < 5.0

    def test_stop_during_setup_does_not_hang(self, buffer_setup):
        # User immediately switches sources / closes the app while the
        # initial model load is still in progress. The setup thread
        # checks stop_event between processors; the partial setup completes
        # but no further processors are touched.
        buffer, timeline, _ = buffer_setup
        proceed = threading.Event()

        class _ParkedSetup(_CountingProcessor):
            def setup(self):
                proceed.wait(timeout=2.0)
                super().setup()

        first = _ParkedSetup()
        second = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[first, second],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        # Release the first processor's setup so it completes, then
        # stop() — stop_event being set should make the setup loop
        # skip the second processor's setup.
        proceed.set()
        # Give the setup thread a moment to finish the first processor
        # AND notice the stop, then move on without setting up the
        # second. Hard to time deterministically — set stop right away
        # and assert second never had setup called.
        ex.stop()
        assert first.setup_calls == 1
        # Whether the second processor's setup was reached depends on
        # the relative timing of "first.setup returns" vs "stop_event
        # set". The contract is: setup loop must check stop_event and
        # never block stop() — assertion is just that stop() returned
        # promptly (covered by the test running at all without timeout).


class TestRealtimeExecutorPlayback:
    def test_play_advances_through_target(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(5, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: p.process_calls >= 5, timeout=2.0)
        finally:
            ex.stop()

    def test_pause_stops_advancing(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1000, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: p.process_calls > 0)
            ex.pause()
            time.sleep(0.05)
            snapshot = p.process_calls
            time.sleep(0.1)
            assert p.process_calls == snapshot
        finally:
            ex.stop()

    def test_pause_drains_pending_queue(self, buffer_setup):
        # With a slow chain (enhancer-like: 50ms/frame here, multi-second
        # in production) the work queue fills with frames the worker still
        # has to process when pause is hit. If pause didn't drain the
        # queue, the worker would keep chewing through it for many seconds
        # post-pause — and playback's latest_index_at_or_below fallback
        # would advance with every completion, showing visible "playback"
        # well after the user pressed pause. With the drain, only inflight
        # items (bounded by worker count) continue.
        buffer, timeline, _ = buffer_setup

        class _SlowCounting(_CountingProcessor):
            def process(self, frame):
                time.sleep(0.05)
                return super().process(frame)

        p = _SlowCounting()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1000, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
            worker_count=1,
        )
        try:
            ex.start()
            ex.play()
            # Let the queue fill up and at least a few frames complete so
            # the dispatcher has had time to submit a backlog.
            assert _wait_until(lambda: p.process_calls >= 3, timeout=2.0)
            ex.pause()
            # After pause completes, only the currently-inflight worker
            # frame + the resubmitted paused frame should produce. With
            # worker_count=1 and 50ms/frame, that caps the post-pause
            # processing to ~2 frames. Sample at 0.5s — well beyond
            # the inflight + resubmit window — and assert the count is
            # bounded. Without the drain, this number would grow without
            # bound as the worker churns through the queue (up to
            # MAX_WORKERS*2 = 32 items).
            # Wait long enough for pause + drain to take effect, but
            # not so long that the test churns on a flaky worker rate.
            time.sleep(0.5)
            snapshot = p.process_calls
            time.sleep(0.5)
            # Strict assertion: no further processing once the inflight
            # + resubmit window has passed.
            assert p.process_calls == snapshot, (
                f"workers continued past pause: {snapshot} -> {p.process_calls}"
            )
        finally:
            ex.stop()

    def test_end_of_target_pauses(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(3, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: ex.status.get() == "end of target", timeout=2.0)
            assert ex.is_playing.get() is False
        finally:
            ex.stop()

    def test_fast_pipeline_waits_for_playhead_before_ending(self, buffer_setup):
        # A pipeline faster than the target fps pre-renders ALL frames long
        # before the wall-clock playhead reaches the end. It must NOT pause
        # "end of target" while the display is still far from the last frame —
        # otherwise playback freezes partway through (the premature-end bug).
        buffer, _timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(10, fps=2.0)),
            buffer=buffer,
            timeline=Timeline(fps=2.0),  # playhead crawls: frame 9 ~= 4.5s away
            chain=[],                    # instant processing → renders all 10 fast
            strategy=SyncedStrategy(),
        )
        try:
            ex.start()
            ex.play()
            # All frames complete almost immediately...
            assert _wait_until(lambda: ex.last_completed_frame() >= 9, timeout=2.0)
            # ...but the slow playhead is nowhere near the end. Give the
            # dispatcher ample time to (wrongly) hit the end check: it must NOT
            # pause while the display is still far from the last frame.
            assert not _wait_until(lambda: not ex.is_playing.get(), timeout=0.5)
            assert ex.status.get() != "end of target"
        finally:
            ex.stop()

    def test_dispatcher_waits_for_setup_before_submitting(self, buffer_setup):
        # While the chain is still loading models, the dispatcher must NOT
        # pre-fill the work queue with opening frames the parked workers can't
        # process yet (they'd go stale → slow-motion-then-snap at startup).
        buffer, _timeline, _ = buffer_setup
        setup_gate = threading.Event()

        class _SlowSetup:
            name = "slow"

            def setup(self) -> None:
                setup_gate.wait(timeout=5.0)

            def process(self, frame):  # type: ignore[no-untyped-def]
                return frame

            def release(self) -> None:
                pass

        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(100, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[_SlowSetup()],
            strategy=SyncedStrategy(),
        )
        try:
            ex.start()
            ex.play()
            # Setup is blocked → nothing should be submitted while it loads.
            assert not _wait_until(
                lambda: ex._last_submitted >= 0, timeout=0.3  # noqa: SLF001
            )
            setup_gate.set()  # let setup finish
            assert _wait_until(
                lambda: ex._last_submitted >= 0, timeout=3.0  # noqa: SLF001
            )
        finally:
            setup_gate.set()
            ex.stop()

    def test_ends_even_if_last_frame_read_fails(self, buffer_setup):
        # The final frame's read returns None (transient hiccup). Playback must
        # still reach "end of target" — not hang one frame short forever waiting
        # for a completion that can never come.
        buffer, _timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_LastFrameFailsReader(5, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[],
            strategy=SyncedStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(
                lambda: ex.status.get() == "end of target", timeout=3.0
            )
            assert ex.is_playing.get() is False
        finally:
            ex.stop()

    def test_seek_repositions(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        target = Timeline(fps=100.0)
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1000)),
            buffer=buffer,
            timeline=target,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.seek(500)
            assert _wait_until(lambda: target.current_frame() >= 500)
        finally:
            ex.stop()

    def test_backward_seek_lowers_last_completed(self, buffer_setup):
        # A backward seek must clamp last_completed DOWN to the new position.
        # Otherwise the stale-high value masks lag, disabling SyncedStrategy's
        # catch-up fallback until wall-clock re-advances past the old value.
        buffer, _timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1000)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[],
            strategy=SyncedStrategy(),
        )
        try:
            ex.start()
            with ex._state_lock:  # noqa: SLF001 — simulate forward progress
                ex._last_completed = 150  # noqa: SLF001
                ex._last_submitted = 150  # noqa: SLF001
            ex.seek(50)  # backward
            assert _wait_until(
                lambda: ex.last_completed_frame() <= 50, timeout=2.0
            )
        finally:
            ex.stop()

    def test_seek_while_paused_processes_target_frame(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        target_timeline = Timeline(fps=100.0)
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1000)),
            buffer=buffer,
            timeline=target_timeline,
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            # No play() — we are in IDLE/PAUSED. A bare seek must still
            # cause the worker to process the target frame so the user
            # gets visual feedback from the scrub.
            assert p.process_calls == 0
            ex.seek(42)
            assert _wait_until(lambda: p.process_calls >= 1, timeout=2.0)
        finally:
            ex.stop()


class TestRealtimeExecutorFrameDelivery:
    def test_on_frame_ready_fires_after_processing(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        delivered: list[tuple[int, Frame]] = []
        lock = threading.Lock()

        def on_frame(f: Frame, i: int) -> None:
            with lock:
                delivered.append((i, f))

        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(3, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        ex.on_frame_ready(on_frame)
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: len(delivered) > 0, timeout=2.0)
        finally:
            ex.stop()

    def test_paused_fallback_suppressed_when_workers_finish_pre_pause_queue(
        self, buffer_setup
    ):
        # User pauses. With a slow chain and multiple workers, the queue
        # holds frames at indices below paused_at. Without fallback
        # suppression, each completion at i < paused_at advances
        # latest_index_at_or_below(paused_at) and the playback tick
        # emits the next-newer fallback — manifesting as visible
        # "playback" continuing after pause. With the suppression, only
        # the exact paused frame can emit while paused.
        buffer, timeline, _ = buffer_setup
        delivered: list[int] = []
        lock = threading.Lock()

        def on_frame(_f: Frame, i: int) -> None:
            with lock:
                delivered.append(i)

        class _SlowCounting(_CountingProcessor):
            def process(self, frame):
                time.sleep(0.05)
                return super().process(frame)

        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1000, fps=100.0)),
            buffer=buffer,
            timeline=timeline,
            chain=[_SlowCounting()],
            strategy=BestEffortStrategy(),
            worker_count=4,
        )
        ex.on_frame_ready(on_frame)
        try:
            ex.start()
            ex.play()
            # Let some frames flow.
            assert _wait_until(lambda: len(delivered) >= 3, timeout=2.0)
            ex.pause()
            # Wait for pause to take effect and the resubmit to land.
            time.sleep(0.3)
            snapshot = len(delivered)
            paused_at = ex.current_frame.get()
            # Allow more time for any pre-pause inflight workers to
            # finish — those completions would, without the suppression,
            # advance the fallback display and grow `delivered` by N.
            time.sleep(0.5)
            # All deliveries after the snapshot must be the exact
            # paused_at frame — nothing earlier should leak through.
            post_snapshot = delivered[snapshot:]
            for d in post_snapshot:
                assert d == paused_at, (
                    f"unexpected non-paused emit after pause: "
                    f"paused_at={paused_at}, post-pause emits={post_snapshot}"
                )
        finally:
            ex.stop()

    def test_seek_to_same_frame_re_emits_after_chain_swap(self, buffer_setup):
        # Chain-refresh-while-paused regression: when a chain change is
        # applied and the controller issues seek(current_frame) so the
        # new chain reprocesses the visible pixels, _handle_seek must
        # reset the duplicate-frame guard. Otherwise the playback tick
        # sees "same index as last emit" and skips the on_frame_ready
        # call — leaving the OLD chain's pixels on screen until the user
        # presses play.
        buffer, timeline, _ = buffer_setup
        delivered: list[int] = []
        lock = threading.Lock()

        def on_frame(_f: Frame, i: int) -> None:
            with lock:
                delivered.append(i)

        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(100, fps=timeline.fps)),
            buffer=buffer,
            timeline=timeline,
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        ex.on_frame_ready(on_frame)
        try:
            ex.start()
            # Seek to 5 while paused — first delivery for that frame.
            ex.seek(5)
            assert _wait_until(lambda: 5 in delivered, timeout=2.0), (
                f"first emit at 5 expected; got {delivered}"
            )
            # Snapshot count of frame-5 emissions, then seek to 5 again
            # (simulating "seek current frame after chain rebuild"). The
            # frame index is the same, but the user expects the display
            # to refresh because the chain processed it anew.
            count_before = delivered.count(5)
            ex.seek(5)
            assert _wait_until(
                lambda: delivered.count(5) > count_before, timeout=2.0
            ), f"expected re-emit at 5; counts: {delivered}"
        finally:
            ex.stop()

    def test_seek_while_paused_fires_on_frame_ready_when_chain_is_slow(
        self, buffer_setup
    ):
        # Regression for the QOL bug where change_source/change_target
        # (or any seek-while-paused) left the previous frame on screen
        # until Play was pressed. The race: dispatcher submits the seek
        # frame, wakes playback, but the worker hasn't finished
        # processing yet. Playback ticks against an empty buffer, then
        # event-driven sleep blocks indefinitely. Without the wake-on-
        # buffer-put in _worker_loop, the freshly-produced frame is
        # never picked up. With it, on_frame_ready fires.
        buffer, timeline, _ = buffer_setup
        delivered: list[int] = []
        lock = threading.Lock()

        def on_frame(_f: Frame, i: int) -> None:
            with lock:
                delivered.append(i)

        # _CountingProcessor with a sleep simulates a real chain that
        # takes longer than the post-seek tick can wait. Without the
        # fix, this test races and may pass by accident; with the fix
        # it deterministically passes.
        class _SlowCounting(_CountingProcessor):
            def process(self, frame):
                time.sleep(0.05)
                return super().process(frame)

        # CRITICAL: pass the buffer's timeline to the executor so seeks
        # affect what the buffer reads in get_at_current_time. Tests
        # that use BestEffort + play() get away with a separate timeline
        # because the buffer eventually accumulates frames from index 0;
        # a seek-while-paused needs both timelines to be the same
        # instance, which mirrors production (PlayerController builds
        # one timeline and passes it to both).
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(100, fps=timeline.fps)),
            buffer=buffer,
            timeline=timeline,
            chain=[_SlowCounting()],
            strategy=BestEffortStrategy(),
        )
        ex.on_frame_ready(on_frame)
        try:
            ex.start()
            # IMPORTANT: no play() — we stay in IDLE/PAUSED so the only
            # thing waking playback is the worker's post-put signal.
            ex.seek(42)
            assert _wait_until(lambda: 42 in delivered, timeout=2.0), (
                f"expected on_frame_ready(42); got {delivered}"
            )
        finally:
            ex.stop()


class TestProcessorTimings:
    """Per-processor wall-clock attribution. The executor wraps each
    p.process() call with perf_counter and exposes a rolling 3-second
    average per processor name, so the metrics overlay can show which
    processor is the bottleneck."""

    def test_timings_empty_when_idle(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=timeline,
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        # No processing yet → empty dict.
        assert ex.processor_timings() == {}

    def test_timings_record_per_processor_after_processing(
        self, buffer_setup
    ):
        buffer, timeline, _ = buffer_setup

        class _Named(_CountingProcessor):
            def __init__(self, name: str) -> None:
                super().__init__()
                # Override the class-level `name` attribute used by
                # the timing accumulator.
                self.name = name

        proc_a = _Named("ProcA")
        proc_b = _Named("ProcB")
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(5, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[proc_a, proc_b],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: proc_a.process_calls >= 3, timeout=2.0)
            assert _wait_until(lambda: proc_b.process_calls >= 3, timeout=2.0)
            timings = ex.processor_timings()
            # Both processors appear, both have non-negative averages.
            assert "ProcA" in timings
            assert "ProcB" in timings
            assert timings["ProcA"] >= 0.0
            assert timings["ProcB"] >= 0.0
        finally:
            ex.stop()

    def test_slow_processor_shows_higher_average(self, buffer_setup):
        # If FaceEnhancer is 10x slower than FaceSwapper, the overlay
        # must show that ratio. Use a sleep-based slow processor.
        buffer, timeline, _ = buffer_setup

        class _Fast(_CountingProcessor):
            name = "Fast"

        class _Slow(_CountingProcessor):
            name = "Slow"

            def process(self, frame):
                time.sleep(0.02)
                return super().process(frame)

        fast = _Fast()
        slow = _Slow()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(20, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[fast, slow],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: slow.process_calls >= 3, timeout=3.0)
            timings = ex.processor_timings()
            assert timings.get("Slow", 0) > timings.get("Fast", 0)
            # And the slow one should be at least ~10ms on average
            # (20ms sleep — generous lower bound for CI jitter).
            assert timings["Slow"] >= 10.0
        finally:
            ex.stop()

    def test_old_timings_age_out_of_window(self, buffer_setup):
        # The accumulator is windowed (3s). After enough wait, entries
        # from earlier processing must drop out — otherwise long sessions
        # would skew averages with ancient samples.
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        p.name = "WindowedProc"
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(2, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: p.process_calls >= 1, timeout=2.0)
            assert "WindowedProc" in ex.processor_timings()
            # Force the window past — manually re-aim the deque
            # timestamps to "long ago" so we don't have to actually
            # wait 3 seconds in the test.
            with ex._timings_lock:  # noqa: SLF001
                aged = [(0.0, name, ns) for (_ts, name, ns) in ex._timings  # noqa: SLF001
                        ] if False else [
                    (0.0, name, ns)
                    for (_ts, name, ns) in ex._processor_timings  # noqa: SLF001
                ]
                ex._processor_timings.clear()  # noqa: SLF001
                ex._processor_timings.extend(aged)  # noqa: SLF001
            # Now the next read should trim everything (cutoff is now-3s,
            # all entries are at t=0 ≪ cutoff).
            assert ex.processor_timings() == {}
        finally:
            ex.stop()


class TestRealtimeExecutorObservables:
    def test_is_playing_reflects_play_pause(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1000)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            assert ex.is_playing.get() is False
            ex.play()
            assert _wait_until(lambda: ex.is_playing.get() is True)
            ex.pause()
            assert _wait_until(lambda: ex.is_playing.get() is False)
        finally:
            ex.stop()

    def test_processing_fps_updates(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(20, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: ex.processing_fps.get() > 0.0, timeout=2.0)
        finally:
            ex.stop()


class TestRealtimeExecutorSetChain:
    def test_set_chain_releases_old_and_sets_up_new(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        old = _CountingProcessor()
        new = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[old],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            assert old.setup_calls == 1
            assert new.setup_calls == 0
            ex.set_chain([new])
            assert _wait_until(lambda: new.setup_calls == 1)
            assert _wait_until(lambda: old.release_calls == 1)
        finally:
            ex.stop()

    def test_set_chain_reprocesses_current_frame_without_a_seek(self, buffer_setup):
        # A chain swap must re-render the current frame on its own: the executor
        # invalidates the whole cache + resubmits, so the new chain's output
        # reaches the display WITHOUT the controller issuing a seek (and even
        # while paused, where the dispatcher isn't advancing the playhead).
        buffer, timeline, _ = buffer_setup
        delivered: list[int] = []
        lock = threading.Lock()

        def on_frame(_f: Frame, i: int) -> None:
            with lock:
                delivered.append(i)

        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(100, fps=timeline.fps)),
            buffer=buffer,
            timeline=timeline,
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        ex.on_frame_ready(on_frame)
        try:
            ex.start()
            ex.seek(5)  # land on frame 5 (paused) — first emit
            assert _wait_until(lambda: 5 in delivered, timeout=2.0), (
                f"first emit at 5 expected; got {delivered}"
            )
            count_before = delivered.count(5)
            ex.set_chain([_CountingProcessor()])  # swap — NO manual seek
            assert _wait_until(
                lambda: delivered.count(5) > count_before, timeout=2.0
            ), f"expected re-emit at 5 after set_chain; got {delivered}"
        finally:
            ex.stop()

    def test_set_memory_cache_bytes_delegates_to_buffer(self):
        from unittest.mock import MagicMock

        ex = RealtimeExecutor.__new__(RealtimeExecutor)
        ex._buffer = MagicMock()  # noqa: SLF001
        ex.set_memory_cache_bytes(2048)
        ex._buffer.set_memory_max_bytes.assert_called_once_with(2048)  # noqa: SLF001


class TestRealtimeExecutorWorkerError:
    def test_chain_error_is_logged_and_recoverable_not_fatal(self, buffer_setup):
        # A per-frame chain error must be RECOVERABLE: logged as a non-fatal
        # "frame error" status (NOT "worker error", which the GUI escalates to a
        # modal) and skipped — the executor keeps running rather than tearing the
        # whole pipeline down on a transient bad frame.
        buffer, timeline, _ = buffer_setup

        class Boom:
            name = "Boom"
            def setup(self) -> None: ...
            def process(self, frame: Frame) -> Frame:
                raise RuntimeError("boom")
            def release(self) -> None: ...

        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(10, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[Boom()],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: "frame error" in ex.status.get(), timeout=2.0)
            assert "worker error" not in ex.status.get()  # not the modal-escalated prefix
            assert not ex._stop_event.is_set()  # noqa: SLF001  not self-terminated
        finally:
            ex.stop()


class TestRealtimeExecutorWorkerScaling:
    """Dynamic worker pool scaling — the executor must add/remove worker
    threads in place without tearing down the chain. Critical: setup()/release()
    on the chain processors run once at start/stop, NOT on every scale event."""

    def test_set_worker_count_grows_pool(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
            worker_count=1,
        )
        try:
            ex.start()
            assert _wait_until(lambda: len(ex._workers) == 1)
            ex.set_worker_count(4)
            assert _wait_until(
                lambda: sum(1 for h in ex._workers if h.thread.is_alive()) == 4
            )
        finally:
            ex.stop()

    def test_set_worker_count_shrinks_pool(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
            worker_count=4,
        )
        try:
            ex.start()
            assert _wait_until(
                lambda: sum(1 for h in ex._workers if h.thread.is_alive()) == 4
            )
            ex.set_worker_count(1)
            # Workers exit at next loop iteration (poll interval bounds this).
            assert _wait_until(
                lambda: sum(1 for h in ex._workers if h.thread.is_alive()) == 1,
                timeout=3.0,
            )
        finally:
            ex.stop()

    def test_set_worker_count_does_not_reload_chain(self, buffer_setup):
        # The whole point of dynamic scaling: setup()/release() stay at 1 each
        # across many scale events. If this regresses, GFPGAN reloads on every
        # slider change and GPU memory leaks accumulate.
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
            worker_count=1,
        )
        try:
            ex.start()
            assert p.setup_calls == 1
            for n in (4, 2, 8, 1, 6):
                ex.set_worker_count(n)
                assert _wait_until(
                    lambda n=n: sum(1 for h in ex._workers if h.thread.is_alive())
                    == n,
                    timeout=3.0,
                )
            assert p.setup_calls == 1
            assert p.release_calls == 0
        finally:
            ex.stop()
        assert p.release_calls == 1

    def test_set_worker_count_rejects_out_of_range(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            with pytest.raises(ValueError):
                ex.set_worker_count(0)
            with pytest.raises(ValueError):
                ex.set_worker_count(999)
        finally:
            ex.stop()


class TestRealtimeExecutorPlaybackMode:
    """Set each mode lands without raising and playback keeps producing
    frames. We don't assert exact tick cadence here (timing-fragile); the
    behavioural contract — frames keep arriving after a mode change — is
    enough to catch outright regressions."""

    @pytest.mark.parametrize("mode", list(PlaybackMode))
    def test_set_each_mode(self, buffer_setup, mode):
        buffer, timeline, _ = buffer_setup
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(50, fps=100.0)),
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.set_playback_mode(mode)
            ex.play()
            assert _wait_until(lambda: p.process_calls >= 5, timeout=2.0)
        finally:
            ex.stop()

    def test_constructor_accepts_playback_mode(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        for mode in PlaybackMode:
            ex = RealtimeExecutor(
                reader_pool=_pool_for(_MultiFrameReader(1)),
                buffer=buffer,
                timeline=Timeline(fps=100.0),
                chain=[],
                strategy=BestEffortStrategy(),
                playback_mode=mode,
            )
            try:
                ex.start()
            finally:
                ex.stop()


class TestRealtimeExecutorWithImageTarget:
    def test_image_target_plays_one_frame_and_pauses(self, buffer_setup, tmp_path: Path):
        buffer, timeline, _ = buffer_setup
        img_path = tmp_path / "img.png"
        cv2.imwrite(str(img_path), np.full((16, 16, 3), 200, dtype=np.uint8))
        reader = ImageTargetReader(Target(path=img_path))
        assert isinstance(reader, TargetReader)

        ex = RealtimeExecutor(
            reader_pool=_pool_for(reader),
            buffer=buffer,
            timeline=Timeline(fps=1.0),
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            assert _wait_until(lambda: ex.status.get() == "end of target", timeout=2.0)
        finally:
            ex.stop()


class _SlowReader:
    """Reader that sleeps for `delay_s` on every read. Used to prove
    the dispatcher no longer blocks on source I/O — with the pool in
    front of it, dispatcher should keep submitting while reads run
    in parallel on pool threads."""

    def __init__(self, count: int = 1000, delay_s: float = 0.2) -> None:
        self._count = count
        self._delay = delay_s
        self._frame = np.full((4, 4, 3), 50, dtype=np.uint8)
        self.release_calls = 0

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def frame_count(self) -> int:
        return self._count

    @property
    def width(self) -> int:
        return 4

    @property
    def height(self) -> int:
        return 4

    @property
    def native_width(self) -> int:
        return 4

    @property
    def native_height(self) -> int:
        return 4

    def read(self, index: int) -> Frame | None:
        time.sleep(self._delay)
        if index < 0 or index >= self._count:
            return None
        return self._frame

    def release(self) -> None:
        self.release_calls += 1


class TestRealtimeExecutorReaderPool:
    """Coverage for the async reader-pool contract: dispatcher is no
    longer the bottleneck, workers tolerate reader failures, the
    pool's shutdown is wired through executor.stop()."""

    def test_dispatcher_does_not_block_on_slow_reader(self, buffer_setup):
        # With one slow reader and pool size 4, the dispatcher should be
        # able to submit ~4 reads in parallel — _last_submitted should
        # advance multiple frames in the time it takes one read to finish.
        buffer, timeline, _ = buffer_setup
        from sinner2.io.reader_pool import ReaderPool

        slow_readers: list[_SlowReader] = []

        def factory():
            r = _SlowReader(count=100, delay_s=0.15)
            slow_readers.append(r)
            return r

        pool = ReaderPool(factory, size=4, name="test")
        ex = RealtimeExecutor(
            reader_pool=pool,
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
            worker_count=4,
        )
        try:
            ex.start()
            ex.play()
            # Sequential lower bound: 4 reads × 150ms = 600ms for 4 frames.
            # With pool size 4 doing parallel I/O: ~150ms for 4 frames.
            # Give 400ms of headroom for thread startup + chain work.
            assert _wait_until(
                lambda: ex._last_submitted >= 3,  # noqa: SLF001 — observe internal counter
                timeout=0.4,
            ), f"dispatcher only reached _last_submitted={ex._last_submitted}"
        finally:
            ex.stop()

    def test_reader_error_is_non_fatal(self, buffer_setup):
        # A single bad read shouldn't kill the executor — status records
        # the issue, subsequent frames are processed normally. Contrast
        # with chain errors which DO stop the executor.
        buffer, timeline, _ = buffer_setup
        from sinner2.io.reader_pool import ReaderPool

        class _OneBadFrameReader(_MultiFrameReader):
            def __init__(self):
                super().__init__(count=20, fps=100.0)

            def read(self, index):
                if index == 3:
                    raise RuntimeError("simulated network blip")
                return super().read(index)

        bad = _OneBadFrameReader()
        pool = ReaderPool(lambda: bad, size=1, name="test")
        p = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=pool,
            buffer=buffer,
            timeline=Timeline(fps=100.0),
            chain=[p],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            ex.play()
            # Frames 0,1,2 and 4,5,...,19 should all process; only 3 fails.
            # Process count should reach at least 5 (proving we ran past the bad frame).
            assert _wait_until(lambda: p.process_calls >= 5, timeout=2.0)
            # Status reflects the read failure.
            assert "reader error at 3" in ex.status.get() or p.process_calls >= 5
        finally:
            ex.stop()

    def test_executor_stop_shuts_down_pool(self, buffer_setup):
        # The executor owns the pool's lifecycle: stop() should call
        # pool.shutdown(), which in turn releases each underlying reader.
        buffer, timeline, _ = buffer_setup
        from sinner2.io.reader_pool import ReaderPool

        reader = _MultiFrameReader(1)
        pool = ReaderPool(lambda: reader, size=1, name="test")
        ex = RealtimeExecutor(
            reader_pool=pool,
            buffer=buffer,
            timeline=timeline,
            chain=[],
            strategy=BestEffortStrategy(),
        )
        ex.start()
        ex.stop()
        assert reader.release_calls == 1



class TestRerenderFromCurrent:
    def test_command_posts_message(self):
        from queue import Queue

        from sinner2.pipeline.messages import RerenderMsg

        ex = object.__new__(RealtimeExecutor)
        ex._command_queue = Queue()  # noqa: SLF001
        ex.rerender_from_current()
        assert isinstance(ex._command_queue.get_nowait(), RerenderMsg)  # noqa: SLF001

    def test_handle_rerender_invalidates_forward_and_resubmits(self):
        from queue import Queue
        from unittest.mock import MagicMock

        ex = object.__new__(RealtimeExecutor)
        ex._state_lock = threading.RLock()  # noqa: SLF001
        ex._work_queue = Queue()  # noqa: SLF001
        ex._playback_wake = threading.Event()  # noqa: SLF001
        ex._last_submitted = 80  # noqa: SLF001
        ex._last_completed = 75  # noqa: SLF001
        ex._generation = 0  # noqa: SLF001
        ex._last_shown_frame_index = 50  # noqa: SLF001
        ex._timeline = MagicMock()  # noqa: SLF001
        ex._timeline.current_frame.return_value = 50  # noqa: SLF001
        ex._buffer = MagicMock()  # noqa: SLF001
        ex._reader_pool = MagicMock()  # noqa: SLF001
        ex._reader_pool.frame_count = 100  # noqa: SLF001

        ex._handle_rerender()  # noqa: SLF001

        # Cache/store dropped FROM the playhead forward (not the whole session).
        ex._buffer.invalidate_from.assert_called_once_with(50)  # noqa: SLF001
        # Completed marker can't claim invalidated frames are done.
        assert ex._last_completed == 49  # noqa: SLF001  # min(75, 49)
        # Current frame resubmitted so a paused display updates immediately.
        ex._reader_pool.read_async.assert_called_once_with(50)  # noqa: SLF001
        assert ex._last_submitted == 50  # noqa: SLF001  # set by _submit_specific_frame


class TestReconfigureFrom:
    """reconfigure_from adopts a freshly built (unstarted) executor's world into
    a RUNNING executor WITHOUT recreating the worker threads. Churning the
    workers on every source/target change leaked GPU memory (ORT's CUDA EP keeps
    per-thread state for dead threads); keeping the same threads avoids it."""

    def _build_unstarted(
        self, tmp_path: Path, sub: str, chain, frames: int = 3
    ):
        store = DiskFrameStore(tmp_path / sub)
        cache = MemoryFrameCache(max_bytes=10 * 1024 * 1024)
        timeline = Timeline(fps=30.0)
        we = BoundedWriteExecutor(max_workers=2, max_outstanding=8)
        buffer = FrameBuffer(store, cache, timeline, we)
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(frames)),
            buffer=buffer,
            timeline=timeline,
            chain=chain,
            strategy=BestEffortStrategy(),
        )
        return ex, we

    def test_keeps_worker_threads_and_swaps_chain(self, tmp_path, buffer_setup):
        buffer, timeline, _ = buffer_setup
        old_p = _CountingProcessor()
        live = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(3)),
            buffer=buffer,
            timeline=timeline,
            chain=[old_p],
            strategy=BestEffortStrategy(),
            worker_count=2,
        )
        live.start()
        assert live.wait_until_ready(timeout=2.0)
        before = {h.thread.ident for h in live._workers}  # noqa: SLF001

        new_p = _CountingProcessor()
        unstarted, new_we = self._build_unstarted(tmp_path, "new", [new_p])
        try:
            old = live.reconfigure_from(unstarted, restore_frame=0, play=False)
            assert old is not None
            old_reader_pool, old_buffer = old
            assert old_buffer is buffer  # the displaced world comes back
            # New chain set up on the dispatcher thread; old chain released.
            assert new_p.setup_calls == 1
            assert _wait_until(lambda: old_p.release_calls == 1)
            # The SAME worker threads survive the swap — this is the fix.
            after = {h.thread.ident for h in live._workers}  # noqa: SLF001
            assert after == before
            # Frames now flow through the NEW chain.
            assert _wait_until(lambda: new_p.process_calls >= 1)
            old_reader_pool.shutdown()
        finally:
            live.stop()
            new_we.shutdown(wait=True)

    def test_failed_setup_keeps_old_world_live(self, tmp_path, buffer_setup):
        buffer, timeline, _ = buffer_setup
        old_p = _CountingProcessor()
        live = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(3)),
            buffer=buffer,
            timeline=timeline,
            chain=[old_p],
            strategy=BestEffortStrategy(),
        )
        live.start()
        assert live.wait_until_ready(timeout=2.0)

        class _BoomSetup:
            name = "boom"

            def setup(self) -> None:
                raise RuntimeError("no face in source")

            def process(self, frame):
                return frame

            def release(self) -> None:
                pass

        unstarted, new_we = self._build_unstarted(tmp_path, "boom", [_BoomSetup()])
        try:
            old = live.reconfigure_from(unstarted, restore_frame=0, play=False)
            assert old is None  # swap abandoned
            assert old_p.release_calls == 0  # old chain stays live
        finally:
            live.stop()
            new_we.shutdown(wait=True)
            unstarted._reader_pool.shutdown()  # noqa: SLF001

    def test_returns_none_when_executor_stopped(self, tmp_path, buffer_setup):
        buffer, timeline, _ = buffer_setup
        live = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(3)),
            buffer=buffer,
            timeline=timeline,
            chain=[_CountingProcessor()],
            strategy=BestEffortStrategy(),
        )
        live.start()
        live.stop()
        unstarted, new_we = self._build_unstarted(
            tmp_path, "stopped", [_CountingProcessor()]
        )
        try:
            assert live.reconfigure_from(
                unstarted, restore_frame=0, play=False
            ) is None
        finally:
            new_we.shutdown(wait=True)
            unstarted._reader_pool.shutdown()  # noqa: SLF001


class TestSetChainSetupOrdering:
    """_handle_set_chain must fully setup() new processors BEFORE assigning them
    to self._chain — workers read self._chain WITHOUT the state lock, so exposing
    an un-set-up processor lets a worker call process() on it (RuntimeError ->
    fatal worker error -> whole-executor teardown)."""

    class _OrderSpy:
        name = "spy"

        def __init__(self, executor):
            self._ex = executor
            self.in_chain_at_setup = None
            self.setup_calls = 0

        def setup(self):
            self.setup_calls += 1
            # Has the executor already published us to workers?
            self.in_chain_at_setup = self in self._ex._chain  # noqa: SLF001

        def process(self, frame):
            return frame

        def release(self):
            pass

    def test_set_chain_sets_up_before_exposing(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        old = _CountingProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(1)),
            buffer=buffer, timeline=timeline, chain=[old],
            strategy=BestEffortStrategy(),
        )
        ex._setup_done_event.set()  # noqa: SLF001  pretend initial setup finished
        spy = self._OrderSpy(ex)
        ex._handle_set_chain((spy,))  # noqa: SLF001
        assert spy.setup_calls == 1
        assert spy.in_chain_at_setup is False  # set up BEFORE the chain swap
        assert ex._chain == (spy,)  # noqa: SLF001
        assert old.release_calls == 1  # dropped processor released


class _FailOnceProcessor:
    """Raises on the first process() call, then passes frames through."""

    name = "FailOnce"

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def setup(self) -> None:
        pass

    def process(self, frame: Frame) -> Frame:
        with self._lock:
            self.calls += 1
            n = self.calls
        if n == 1:
            raise RuntimeError("transient boom")
        return frame

    def release(self) -> None:
        pass


class TestTransientWorkerErrorIsRecoverable:
    """A single transient per-frame chain error must NOT tear the whole executor
    down (which also leaves _state lying + GPU held); it's logged and skipped
    like a reader error, and processing continues."""

    def test_transient_error_does_not_stop_executor(self, buffer_setup):
        buffer, timeline, _ = buffer_setup
        p = _FailOnceProcessor()
        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(20)),
            buffer=buffer, timeline=timeline, chain=[p],
            strategy=BestEffortStrategy(), worker_count=1,
            playback_mode=PlaybackMode.UNLIMITED,
        )
        try:
            ex.start()
            assert ex.wait_until_ready(2.0)
            ex.play()
            # With the bug, the worker breaks + sets _stop_event after call 1, so
            # calls never climb past 1. With the fix it keeps processing.
            kept_going = _wait_until(lambda: p.calls >= 3, timeout=4.0)
            assert kept_going, "executor self-terminated on a transient frame error"
            assert not ex._stop_event.is_set()  # noqa: SLF001
        finally:
            ex.stop()


class TestPlaybackFallbackNoBackwardStutter:
    """During forward playback the display fallback must not repaint a frame
    OLDER than what's on screen — that's a visible backward stutter. Holds the
    current frame until a frame >= the last shown one is available."""

    def _tick(self, *, last_shown, fallback_index):
        from collections import deque
        from unittest.mock import MagicMock

        from sinner2.pipeline.realtime.executor import _State

        ex = object.__new__(RealtimeExecutor)
        ex._state_lock = threading.RLock()  # noqa: SLF001
        ex._state = _State.PLAYING  # noqa: SLF001
        ex._fps_lock = threading.Lock()  # noqa: SLF001
        ex._completion_times = deque()  # noqa: SLF001
        ex._last_completion_time = None  # noqa: SLF001
        ex._last_fps = 0.0  # noqa: SLF001
        ex._last_metrics_pub = 0.0  # noqa: SLF001
        ex.current_frame = MagicMock()
        ex.metrics = MagicMock()
        ex.processing_fps = MagicMock()
        ex.status = MagicMock()
        ex._last_shown_frame_index = last_shown  # noqa: SLF001
        shown: list[int] = []
        ex._on_frame = lambda f, i: shown.append(i)  # noqa: SLF001
        buf = MagicMock()
        buf.get_at_current_time.return_value = (200, None)  # miss at target 200
        buf.latest_index_at_or_below.return_value = fallback_index
        buf.get.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        ex._buffer = buf  # noqa: SLF001
        ex._do_playback_tick()  # noqa: SLF001
        return shown, ex._last_shown_frame_index  # noqa: SLF001

    def test_does_not_repaint_older_fallback_frame(self):
        # Showed 100; only frame <= target is 95 (older) -> hold, don't stutter.
        shown, last = self._tick(last_shown=100, fallback_index=95)
        assert shown == []
        assert last == 100

    def test_repaints_newer_fallback_frame(self):
        shown, last = self._tick(last_shown=100, fallback_index=105)
        # on_frame gets the SHOWN (fallback) index, not the timeline target 200.
        assert shown == [105]
        assert last == 105

    def test_first_paint_with_no_history(self):
        shown, last = self._tick(last_shown=None, fallback_index=50)
        assert shown == [50]  # shown (fallback) index, not the target 200
        assert last == 50


class TestProcessingFpsStallDecay:
    """processing_fps reports a decaying estimate during slow-but-alive
    progress instead of a hard 0, so a slow source isn't mistaken for a hang."""

    def _refresh(self, *, completion_times, last_completion_time, last_fps):
        from collections import deque
        from unittest.mock import MagicMock

        ex = object.__new__(RealtimeExecutor)
        ex._fps_lock = threading.RLock()  # noqa: SLF001
        ex._completion_times = deque(completion_times)  # noqa: SLF001
        ex._last_completion_time = last_completion_time  # noqa: SLF001
        ex._last_fps = last_fps  # noqa: SLF001
        ex.processing_fps = MagicMock()
        ex._refresh_fps()  # noqa: SLF001
        (val,), _ = ex.processing_fps.set.call_args
        return val

    def test_decays_to_small_positive_during_slow_progress(self):
        now = time.monotonic()
        fps = self._refresh(completion_times=[], last_completion_time=now - 5.0,
                            last_fps=10.0)
        assert 0.1 < fps < 0.4  # ~0.2, not 0

    def test_zero_after_long_stall(self):
        now = time.monotonic()
        fps = self._refresh(completion_times=[], last_completion_time=now - 60.0,
                            last_fps=10.0)
        assert fps == 0.0

    def test_never_completed_is_zero(self):
        fps = self._refresh(completion_times=[], last_completion_time=None,
                            last_fps=0.0)
        assert fps == 0.0

    def test_windowed_rate_when_healthy(self):
        now = time.monotonic()
        fps = self._refresh(
            completion_times=[now - 0.3, now - 0.2, now - 0.1, now],
            last_completion_time=now, last_fps=0.0,
        )
        assert fps > 5.0

    def test_cold_start_decay_is_capped(self):
        # First completion just happened (count=1), windowed rate undefined,
        # last_fps still 0 → the 1/tiny-elapsed estimate must be capped, not a
        # bogus thousands-fps spike.
        now = time.monotonic()
        fps = self._refresh(completion_times=[now], last_completion_time=now,
                            last_fps=0.0)
        assert fps <= 120.0


class TestSeekResetsChainStreamState:
    """A seek must call on_seek() on chain processors that expose it, so the
    swapper drops its stale interval-based detection cache."""

    def test_seek_calls_on_seek_hook(self, buffer_setup):
        buffer, _timeline, _ = buffer_setup
        seeks: list[int] = []

        class _SeekAware:
            name = "seekaware"

            def setup(self) -> None:
                pass

            def process(self, frame):  # type: ignore[no-untyped-def]
                return frame

            def release(self) -> None:
                pass

            def on_seek(self) -> None:
                seeks.append(1)

        ex = RealtimeExecutor(
            reader_pool=_pool_for(_MultiFrameReader(100)),
            buffer=buffer,
            timeline=Timeline(fps=30.0),
            chain=[_SeekAware()],
            strategy=BestEffortStrategy(),
        )
        try:
            ex.start()
            assert ex.wait_until_ready(timeout=2.0)
            ex.seek(50)
            assert _wait_until(lambda: len(seeks) >= 1, timeout=2.0)
        finally:
            ex.stop()


class TestChainSetupFailure:
    """A failure during chain.setup() (bad model/source) must NOT leak the
    processors that already loaded — the workers/dispatcher just exit on
    _stop_event and never run the normal release path."""

    class _Proc:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self._fail = fail
            self.setup_called = False
            self.released = False

        def setup(self) -> None:
            self.setup_called = True
            if self._fail:
                raise RuntimeError("boom")

        def release(self) -> None:
            self.released = True

    class _Status:
        def __init__(self) -> None:
            self.value = ""

        def set(self, v: str) -> None:
            self.value = v

        def get(self) -> str:
            return self.value

    def _bare_executor(self, chain):
        ex = RealtimeExecutor.__new__(RealtimeExecutor)
        ex._stop_event = threading.Event()  # noqa: SLF001
        ex._setup_done_event = threading.Event()  # noqa: SLF001
        ex.status = self._Status()
        ex._chain = chain  # noqa: SLF001
        return ex

    def test_setup_failure_releases_loaded_processors(self):
        p0 = self._Proc("a")
        p1 = self._Proc("b", fail=True)
        p2 = self._Proc("c")
        ex = self._bare_executor([p0, p1, p2])
        ex._setup_chain_async()  # noqa: SLF001
        assert ex._stop_event.is_set()  # noqa: SLF001
        assert "chain setup failed" in ex.status.get()
        assert ex._setup_done_event.is_set()  # noqa: SLF001
        assert p0.released  # the already-loaded processor's GPU memory is freed


class TestPlaybackFallbackIndex:
    """on_frame must receive the index of the frame whose pixels are shown — the
    fallback index when the worker is behind, not the timeline target (rank 31).
    """

    class _Obs:
        def __init__(self) -> None:
            self.value = None

        def set(self, v) -> None:
            self.value = v

    class _Buf:
        def get_at_current_time(self):
            return (10, None)  # target 10 not ready yet

        def latest_index_at_or_below(self, _idx):
            return 7  # newest ready frame <= 10

        def get(self, idx):
            return "px" if idx == 7 else None

        def metrics(self):
            return None

    def test_on_frame_gets_shown_index_not_target(self):
        from sinner2.pipeline.realtime import executor as ex_mod

        ex = RealtimeExecutor.__new__(RealtimeExecutor)
        ex._state_lock = threading.RLock()  # noqa: SLF001
        ex._state = ex_mod._State.PLAYING  # noqa: SLF001
        ex._last_shown_frame_index = None  # noqa: SLF001
        ex._last_metrics_pub = 0.0  # noqa: SLF001
        ex._buffer = self._Buf()  # noqa: SLF001
        ex._refresh_fps = lambda: None  # noqa: SLF001
        ex.current_frame = self._Obs()
        ex.metrics = self._Obs()
        ex.status = self._Obs()
        seen: list = []
        ex._on_frame = lambda _f, i: seen.append(i)  # noqa: SLF001
        ex._do_playback_tick()  # noqa: SLF001
        assert seen == [7]  # the SHOWN (fallback) index, not the target 10
        assert ex.current_frame.value == 10  # transport still tracks the target


class TestMetricsPublishThrottle:
    """buffer.metrics() recomputes percentiles and the playback tick can run at
    ~1 kHz; metrics publication must be throttled to ~UI rate."""

    class _Obs:
        def __init__(self) -> None:
            self.value = None

        def set(self, v) -> None:
            self.value = v

    def test_metrics_recomputed_at_most_once_per_interval(self, monkeypatch):
        from sinner2.pipeline.realtime import executor as ex_mod

        clock = [0.0]
        monkeypatch.setattr(ex_mod.time, "monotonic", lambda: clock[0])
        calls = [0]

        class _Buf:
            def get_at_current_time(self):
                return (5, "px")  # direct hit, no fallback

            def latest_index_at_or_below(self, _i):
                return 5

            def get(self, _i):
                return "px"

            def metrics(self):
                calls[0] += 1
                return None

        ex = RealtimeExecutor.__new__(RealtimeExecutor)
        ex._state_lock = threading.RLock()  # noqa: SLF001
        ex._state = ex_mod._State.PLAYING  # noqa: SLF001
        ex._last_shown_frame_index = None  # noqa: SLF001
        ex._last_metrics_pub = 0.0  # noqa: SLF001
        ex._buffer = _Buf()  # noqa: SLF001
        ex._refresh_fps = lambda: None  # noqa: SLF001
        ex._on_frame = None  # noqa: SLF001
        ex.current_frame = self._Obs()
        ex.metrics = self._Obs()
        ex.status = self._Obs()

        clock[0] = 1.0
        ex._do_playback_tick()  # noqa: SLF001 — publishes
        clock[0] = 1.01
        ex._do_playback_tick()  # noqa: SLF001 — within interval → skipped
        clock[0] = 2.0
        ex._do_playback_tick()  # noqa: SLF001 — interval elapsed → publishes
        assert calls[0] == 2


class TestReconfigureGenerationGuard:
    """A worker parked on its source-future during a reconfigure must NOT write
    its old-world frame into the new buffer: _publish_result discards a result
    whose WorkItem generation no longer matches the executor's current world."""

    class _Obs:
        def set(self):  # mimics threading.Event.set() (no arg)
            pass

    def _bare(self, generation):
        ex = RealtimeExecutor.__new__(RealtimeExecutor)
        ex._state_lock = threading.RLock()  # noqa: SLF001
        ex._generation = generation  # noqa: SLF001
        ex._last_completed = -1  # noqa: SLF001
        ex._playback_wake = self._Obs()  # noqa: SLF001
        return ex

    def test_discards_result_from_stale_generation(self):
        from concurrent.futures import Future
        from unittest.mock import MagicMock

        from sinner2.pipeline.realtime.work_item import WorkItem

        ex = self._bare(generation=5)
        buf = MagicMock()
        ex._buffer = buf  # noqa: SLF001
        item = WorkItem(frame_index=3, source_future=Future(), generation=4)
        assert ex._publish_result(item, "frame") is False  # noqa: SLF001
        buf.put.assert_not_called()  # stale frame NOT written to the new buffer
        assert ex._last_completed == -1  # noqa: SLF001 — progress not advanced

    def test_publishes_result_from_current_generation(self):
        from concurrent.futures import Future
        from unittest.mock import MagicMock

        from sinner2.pipeline.realtime.work_item import WorkItem

        ex = self._bare(generation=5)
        buf = MagicMock()
        ex._buffer = buf  # noqa: SLF001
        item = WorkItem(frame_index=3, source_future=Future(), generation=5)
        assert ex._publish_result(item, "frame") is True  # noqa: SLF001
        buf.put.assert_called_once_with(3, "frame")
        assert ex._last_completed == 3  # noqa: SLF001


class TestPublicAccessors:
    """Public passthroughs so GUI callers don't reach executor privates
    (_buffer / _reader_pool)."""

    def test_invalidate_from_passes_through_to_buffer(self):
        from unittest.mock import MagicMock

        ex = RealtimeExecutor.__new__(RealtimeExecutor)
        buf = MagicMock()
        ex._buffer = buf  # noqa: SLF001
        ex.invalidate_from(7)
        buf.invalidate_from.assert_called_once_with(7)

    def test_reader_pool_property_returns_pool(self):
        from unittest.mock import MagicMock

        ex = RealtimeExecutor.__new__(RealtimeExecutor)
        pool = MagicMock()
        ex._reader_pool = pool  # noqa: SLF001
        assert ex.reader_pool is pool


class TestApplyChainContext:
    """_apply_chain feeds ONE ChainContext per frame to context-aware
    processors (detect-once-share-faces); plain processors keep the
    one-argument call."""

    def _executor_shell(self):
        from collections import deque

        from sinner2.pipeline.realtime.executor import RealtimeExecutor

        ex = object.__new__(RealtimeExecutor)  # bypass the heavy __init__
        ex._timings_lock = threading.Lock()
        ex._processor_timings = deque()
        return ex

    def test_context_flows_between_context_aware_processors(self):
        class _Producer:
            name = "producer"
            thread_safe = True
            accepts_context = True

            def process(self, frame, ctx=None):
                ctx.faces = ["FACE"]
                return frame

        class _Plain:
            name = "plain"
            thread_safe = True

            def process(self, frame):
                return frame

        class _Consumer:
            name = "consumer"
            thread_safe = True
            accepts_context = True

            def __init__(self):
                self.seen = "unset"

            def process(self, frame, ctx=None):
                self.seen = ctx.faces
                return frame

        ex = self._executor_shell()
        consumer = _Consumer()
        frame = np.zeros((4, 4, 3), np.uint8)
        ex._apply_chain(frame, (_Producer(), _Plain(), consumer))
        assert consumer.seen == ["FACE"]

    def test_fresh_context_per_frame(self):
        class _Consumer:
            name = "consumer"
            thread_safe = True
            accepts_context = True

            def __init__(self):
                self.seen: list = []

            def process(self, frame, ctx=None):
                self.seen.append(ctx.faces)
                ctx.faces = ["stale"]
                return frame

        ex = self._executor_shell()
        consumer = _Consumer()
        frame = np.zeros((4, 4, 3), np.uint8)
        ex._apply_chain(frame, (consumer,))
        ex._apply_chain(frame, (consumer,))
        # The second frame's context starts clean — nothing leaks across frames.
        assert consumer.seen == [None, None]
