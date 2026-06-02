import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Full, Queue
from typing import Any

from sinner2.io.reader_pool import ReaderPool
from sinner2.observable import ObservableValue
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.messages import (
    Message,
    PauseMsg,
    PlayMsg,
    ReconfigureMsg,
    RerenderMsg,
    SeekMsg,
    SetChainMsg,
    SetParamsMsg,
    SetPlaybackModeMsg,
    SetSkipStrategyMsg,
    SetWorkerCountMsg,
    StopMsg,
)
from sinner2.pipeline.cache_mode import CacheMode
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.realtime.work_item import WorkItem
from sinner2.pipeline.skip_strategy import FrameSkipStrategy
from sinner2.types import Frame, FrameIndex


class _State(Enum):
    STOPPED = "stopped"
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"


_WORKER_SENTINEL: WorkItem | None = None
_DISPATCHER_TICK_S = 0.005
# Worker waits this long for a source frame before treating it as a
# stuck/dead read. Generous because SMB seeks can be slow; if the
# reader genuinely hangs longer than this, something else is wrong
# and surfacing the timeout is better than blocking the worker.
_WORKER_READ_TIMEOUT_S = 30.0
# Default fixed-rate tick when PlaybackMode.FIXED_30 is selected. Keep at
# 30 Hz: high enough for smooth perceived motion, low enough to be cheap.
_FIXED_PLAYBACK_TICK_S = 1.0 / 30
# Floor for UNLIMITED mode so we still yield to the OS scheduler rather
# than burning a core in a tight loop. The per-tick duplicate-frame guard
# means we don't actually emit more frames than the timeline produces, so
# this floor mostly just bounds wakeup frequency.
_UNLIMITED_PLAYBACK_TICK_S = 0.001
# Time-window for processing_fps. Time-based (not count-based) so the
# reading reflects current throughput rather than a sample-window average
# that can stay stale through pauses. 3s is short enough to update visibly
# after the user changes a setting and long enough to be smooth at low fps.
_FPS_WINDOW_S = 3.0
# When completions are too sparse for a windowed rate, report the rate implied
# by "time since the last completion" so a slow-but-progressing pipeline shows a
# small positive fps instead of a hard 0 (indistinguishable from a hang). After
# this long with no completion at all, fall through to 0 — genuinely stalled.
_FPS_STALL_HOLD_S = 30.0
# Ceiling for the stall decay estimate when there's no prior windowed rate to
# cap it (cold start): keeps the first-completion 1/tiny-elapsed reading from
# spiking to a bogus thousands-fps value before the windowed rate kicks in.
_FPS_DECAY_CAP = 120.0
# Time window for per-processor average-ms readout. Matches _FPS_WINDOW_S
# so the metrics-overlay row and the rates next to it cover the same
# wall-clock slice. Cap deque size so a fast no-op chain (1000+ fps)
# can't unbounded-grow the buffer between window trims.
_TIMING_WINDOW_S = 3.0
_TIMING_DEQUE_CAP = 4096
# Worker queue.get timeout: how long an idle worker waits before checking
# its exit_event / the global stop_event. Bounds shutdown latency for the
# rare case where no item or sentinel is available (e.g. user shrinks the
# pool to N=1 with the queue empty).
_WORKER_POLL_S = 0.5
# Hard upper bound for the worker pool. The work queue is sized at
# MAX_WORKERS * 2 at construction so set_worker_count never has to grow
# the queue. Coupled to QProcessorControls' worker_count spinbox range.
MAX_WORKERS = 16


@dataclass
class _WorkerHandle:
    """A spawned worker thread plus the event used to ask it to exit.

    The thread checks exit_event at the top of every loop iteration; when
    set, the worker finishes any in-flight frame and returns. Held inside
    RealtimeExecutor._workers; only the dispatcher thread mutates the list
    after start() completes.
    """

    thread: threading.Thread
    exit_event: threading.Event


class RealtimeExecutor:
    """Frame-major realtime conveyor.

    Threads owned: one dispatcher, N workers, one playback. The dispatcher
    drains the command queue, manages state, asks the skip strategy what
    frame to submit next, and decodes source frames before pushing WorkItems
    to the work queue. Workers apply the chain snapshot and put results
    into the buffer. The playback thread reads from the buffer at the
    timeline's pace and fires the on_frame_ready callback.

    All commands post messages; nothing blocks the caller. State is exposed
    through ObservableValues so a GUI can subscribe via the existing bridge
    pattern without reaching into private fields.
    """

    def __init__(
        self,
        reader_pool: ReaderPool,
        buffer: FrameBuffer,
        timeline: Timeline,
        chain: list[Processor],
        strategy: FrameSkipStrategy,
        worker_count: int = 1,
        playback_mode: PlaybackMode = PlaybackMode.FIXED_30,
    ) -> None:
        if worker_count < 1:
            raise ValueError(f"worker_count must be >= 1; got {worker_count}")
        if worker_count > MAX_WORKERS:
            raise ValueError(
                f"worker_count must be <= {MAX_WORKERS}; got {worker_count}"
            )
        self._reader_pool = reader_pool
        self._buffer = buffer
        self._timeline = timeline
        # Clamp the wall-clock playhead to the real last frame so it can't run
        # off the end of the media (see Timeline.set_max_frame).
        self._timeline.set_max_frame(max(0, reader_pool.frame_count - 1))
        self._strategy = strategy
        # Initial worker count; the live count is len(self._workers) and can
        # be changed via set_worker_count() at any point after start().
        self._initial_worker_count = worker_count
        self._playback_mode = playback_mode

        self._state_lock = threading.RLock()
        # ONE chain shared by all workers. ORT InferenceSession is thread-safe
        # for concurrent .run() calls — N workers calling the same swapper let
        # ORT schedule across the GPU efficiently. Building N independent
        # swappers (an earlier attempt) created N CUDA contexts and SLOWED
        # things down, so the thread-safe swapper stays shared. A processor
        # that ISN'T thread-safe (e.g. GFPGAN) is wrapped upstream in a
        # PerWorkerProcessor, which hands each worker thread its own instance —
        # parallel enhance without multiplying the swapper's context. From
        # here it still looks like one shared chain of Processors.
        self._chain: tuple[Processor, ...] = tuple(chain)
        self._state: _State = _State.STOPPED
        self._last_submitted: FrameIndex = -1
        self._last_completed: FrameIndex = -1

        # Wakes the playback thread early on any state change (play, pause,
        # seek, mode change, stop). When the executor is paused/idle the
        # playback thread blocks on this event indefinitely instead of
        # polling — that's how we get sinner1's idle-zero-CPU behaviour
        # without sinner1's spawn/join race on every play/pause.
        self._playback_wake = threading.Event()
        # Last frame index actually handed to on_frame. Duplicate guard:
        # the playback tick fires often, but we only call on_frame when
        # the displayed frame would change. Without this, UNLIMITED mode
        # would flood the GUI thread with redundant emits.
        self._last_shown_frame_index: FrameIndex | None = None

        self._command_queue: Queue[Message] = Queue()
        # Queue is sized for MAX_WORKERS, not the current worker_count, so we
        # never need to resize when set_worker_count grows the pool. The 2x
        # factor gives the dispatcher enough head-room to keep all workers
        # busy without becoming a sink for unbounded buffered work.
        self._work_queue: Queue[WorkItem | None] = Queue(maxsize=MAX_WORKERS * 2)
        self._stop_event = threading.Event()
        # Chain setup runs on a background thread so executor.start() doesn't
        # block the caller (typically the GUI main thread) for the seconds it
        # takes to load GFPGAN + inswapper + buffalo_l. Workers and
        # _handle_set_chain wait on this before processing or swapping;
        # dispatcher/playback can run immediately because they don't touch
        # chain.process(). Set on success AND failure (failure also sets
        # _stop_event) so anything blocked on it can wake and check stop.
        self._setup_done_event = threading.Event()
        self._setup_thread: threading.Thread | None = None
        # Track workers mid-_apply_chain so set_chain can wait for them to
        # finish before releasing the old chain's processors. Without this,
        # release() (e.g. FaceEnhancer dropping its GFPGAN ref) races with
        # workers calling .process() on the same instance.
        self._inflight_cv = threading.Condition()
        self._inflight_count = 0
        self._dispatcher_thread: threading.Thread | None = None
        # Active worker handles. Mutated only by start() (before dispatcher
        # exists) and by _handle_set_worker_count (on the dispatcher thread),
        # so no lock is needed for ordinary access from those paths. stop()
        # reads the list only after joining the dispatcher.
        self._workers: list[_WorkerHandle] = []
        self._worker_thread_counter = 0
        self._playback_thread: threading.Thread | None = None

        self.current_frame: ObservableValue[FrameIndex] = ObservableValue(0)
        self.is_playing: ObservableValue[bool] = ObservableValue(False)
        self.processing_fps: ObservableValue[float] = ObservableValue(0.0)
        self.metrics: ObservableValue[BufferMetrics] = ObservableValue(buffer.metrics())
        self.status: ObservableValue[str] = ObservableValue("")
        # Strategy's current mode label, surfaced to the status bar so
        # the user can see when an adaptive strategy shifts gears (e.g.
        # SyncedStrategy falling back to sequential on slow sources).
        # Updated after each dispatcher decide() call; equality
        # suppression in ObservableValue means no-op updates are free.
        self.strategy_mode: ObservableValue[str] = ObservableValue(strategy.current_mode())

        self._on_frame: Callable[[Frame, FrameIndex], None] | None = None

        self._fps_lock = threading.RLock()
        # Timestamps of frame completions in the last _FPS_WINDOW_S seconds.
        # Workers append (cheap, list append under lock). The playback
        # thread trims and publishes once per tick so workers never block
        # on Qt signal emission and the GUI gets updates at a sane rate
        # (~30 Hz) instead of per-completion (which scales with worker
        # count and serialises the whole pool on the observable's lock).
        self._completion_times: deque[float] = deque()
        # Most recent completion timestamp (never trimmed) + last windowed fps,
        # so a slow-but-alive pipeline reports a decaying estimate rather than 0.
        self._last_completion_time: float | None = None
        self._last_fps = 0.0
        # Per-processor timing: append (timestamp, processor_name, ns)
        # per process() call inside _apply_chain. Readers (overlay) get
        # a time-windowed dict via processor_timings(). bounded deque
        # so a fast no-op chain can't grow it without limit between
        # trim cycles — at 1000+ fps with no enhancer we'd otherwise
        # leak 3000+ entries between overlay ticks.
        self._timings_lock = threading.RLock()
        self._processor_timings: deque[tuple[float, str, int]] = deque(
            maxlen=_TIMING_DEQUE_CAP
        )

    # ---- Lifecycle ----

    def start(self) -> None:
        with self._state_lock:
            if self._state is not _State.STOPPED:
                return
            self._stop_event.clear()
            self._setup_done_event.clear()
            self._state = _State.IDLE
            # Surface a loading hint in the status observable so the GUI can
            # show something more informative than an idle status bar while
            # models load. Workers and _handle_set_chain block on
            # _setup_done_event until the background thread completes.
            self.status.set("loading models…")
            self._setup_thread = threading.Thread(
                target=self._setup_chain_async,
                name="sinner2-setup",
                daemon=True,
            )
            self._setup_thread.start()
            # Spawn workers BEFORE the dispatcher so a SetWorkerCountMsg that
            # somehow arrives while start() is still running can't race with
            # the initial pool construction. Workers immediately park on
            # _setup_done_event until the setup thread finishes.
            for _ in range(self._initial_worker_count):
                self._spawn_worker()
            self._dispatcher_thread = threading.Thread(
                target=self._dispatcher_loop, name="sinner2-dispatcher", daemon=True
            )
            self._dispatcher_thread.start()
            self._playback_thread = threading.Thread(
                target=self._playback_loop, name="sinner2-playback", daemon=True
            )
            self._playback_thread.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until chain.setup() has completed (or aborted via stop).

        Returns True if the setup-done event fired within the timeout,
        False on timeout. Useful for tests that need to observe setup-side
        effects deterministically, and for callers that want to surface a
        'ready' state after start() returns. NOT required for normal
        operation — workers and _handle_set_chain already park on the
        same event internally.
        """
        return self._setup_done_event.wait(timeout=timeout)

    def _setup_chain_async(self) -> None:
        """Run chain.setup() off the calling thread.

        Checks _stop_event between processors so a fast stop() during model
        load doesn't keep loading the remaining ones. Failures set the
        status and the stop event so the executor tears down cleanly
        instead of leaving workers parked forever.
        """
        try:
            for p in self._chain:
                if self._stop_event.is_set():
                    return
                p.setup()
        except Exception as e:
            self.status.set(f"chain setup failed: {e}")
            self._stop_event.set()
        else:
            # Clear the loading hint only on success — failures leave their
            # own message in place.
            self.status.set("")
        finally:
            # Always set on exit so workers / _handle_set_chain wake up,
            # whether to start processing (success) or to observe _stop_event
            # (failure / aborted-by-stop).
            self._setup_done_event.set()

    def _spawn_worker(self) -> None:
        """Create one worker thread with its own exit_event and append it.

        Called from start() (single-threaded init) and from the dispatcher
        thread inside _handle_set_worker_count. The counter is used purely
        for thread names so each spawn gets a unique identifier across the
        executor's lifetime — handy in py-spy dumps.
        """
        exit_event = threading.Event()
        name = f"sinner2-worker-{self._worker_thread_counter}"
        self._worker_thread_counter += 1
        thread = threading.Thread(
            target=self._worker_loop, args=(exit_event,), name=name, daemon=True
        )
        thread.start()
        self._workers.append(_WorkerHandle(thread=thread, exit_event=exit_event))

    def stop(self) -> None:
        with self._state_lock:
            if self._state is _State.STOPPED:
                return
            self._stop_event.set()

        # Wake anything blocked in _wait_for_inflight so the dispatcher (or
        # any caller of set_chain) can unblock and notice _stop_event.
        with self._inflight_cv:
            self._inflight_cv.notify_all()
        # Wake the playback thread out of its event-driven sleep so it can
        # observe _stop_event without waiting on its tick interval (which
        # may be indefinite when paused).
        self._playback_wake.set()
        # Wake workers and _handle_set_chain (if they're parked waiting
        # for initial setup to finish) so they can observe _stop_event
        # and exit instead of waiting for the setup thread to finish a
        # multi-second model load that we no longer care about.
        self._setup_done_event.set()

        # Let the dispatcher exit before touching _workers. This also lets
        # any in-progress _handle_set_chain finish its release() calls
        # before we start tearing down the chain ourselves.
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5.0)

        # Drain pending work so workers don't keep processing items they
        # don't need to. Then push one sentinel per still-alive worker for
        # an immediate wake; workers waiting on queue.get exit at once
        # instead of polling for up to _WORKER_POLL_S.
        self._drain_work_queue()
        live_workers = [h for h in self._workers if h.thread.is_alive()]
        for h in live_workers:
            h.exit_event.set()
        for _ in live_workers:
            try:
                self._work_queue.put_nowait(_WORKER_SENTINEL)
            except Full:
                # Queue was full of real items the workers will consume
                # first; the exit_event poll will catch them on the next
                # loop iteration even if no sentinel lands.
                break
        # Generous timeout: a worker might be mid-GFPGAN inference (1-3s
        # per frame on FHD) when stop fires; give it room to finish before
        # we leak the thread.
        for h in self._workers:
            h.thread.join(timeout=30.0)
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)
        # Join the setup thread before releasing chain processors — if
        # setup is mid-p.setup() and we release() the same processor
        # concurrently, internal state tears down under the setup call.
        # Timeout is generous because p.setup() can't be cancelled
        # cooperatively; we just have to wait for the model load to
        # finish or time out.
        if self._setup_thread is not None:
            self._setup_thread.join(timeout=30.0)

        with self._state_lock:
            for p in self._chain:
                p.release()
            self._state = _State.STOPPED
            self._dispatcher_thread = None
            self._workers = []
            self._playback_thread = None
            self._setup_thread = None
        # ReaderPool is shut down OUTSIDE the state_lock: shutdown blocks
        # while reader threads finish, and there's no need to hold the
        # state lock through that wait. Workers are already joined above,
        # so no future is awaiting a result from the pool.
        self._reader_pool.shutdown()

    # ---- Commands (non-blocking; post messages) ----

    def play(self) -> None:
        self._command_queue.put(PlayMsg())

    def pause(self) -> None:
        self._command_queue.put(PauseMsg())

    def seek(self, frame: FrameIndex) -> None:
        self._command_queue.put(SeekMsg(target_frame=frame))

    def rerender_from_current(self) -> None:
        """Reprocess from the playhead forward through the current chain
        (e.g. after a param change). Frames before the playhead keep their
        cached pixels."""
        self._command_queue.put(RerenderMsg())

    def set_params(self, processor_name: str, params: Mapping[str, Any]) -> None:
        self._command_queue.put(SetParamsMsg(processor_name=processor_name, params=params))

    def set_chain(self, chain: list[Processor]) -> None:
        self._command_queue.put(SetChainMsg(chain=tuple(chain)))

    def set_skip_strategy(self, strategy: FrameSkipStrategy) -> None:
        self._command_queue.put(SetSkipStrategyMsg(strategy=strategy))

    def set_worker_count(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"worker_count must be >= 1; got {n}")
        if n > MAX_WORKERS:
            raise ValueError(f"worker_count must be <= {MAX_WORKERS}; got {n}")
        self._command_queue.put(SetWorkerCountMsg(n=n))

    def set_playback_mode(self, mode: PlaybackMode) -> None:
        self._command_queue.put(SetPlaybackModeMsg(mode=mode))

    def set_cache_mode(self, mode: CacheMode) -> None:
        """Hot-swap the buffer's cache mode. Cheap — just toggles which
        I/O paths the buffer takes; no rebuild required."""
        self._buffer.set_cache_mode(mode)

    def reconfigure_from(
        self,
        other: "RealtimeExecutor",
        *,
        restore_frame: FrameIndex,
        play: bool,
        timeout_s: float = 30.0,
    ) -> tuple[ReaderPool, FrameBuffer] | None:
        """Adopt the world (reader pool, buffer, timeline, chain, strategy,
        playback mode) of an UNSTARTED executor into this RUNNING one, keeping
        this executor's own dispatcher / playback / worker threads alive.

        This is how a source/target change avoids leaking GPU memory: tearing
        the executor down and building a new one would destroy all worker
        threads and spawn fresh ones, and ORT's CUDA EP never frees the
        per-thread state of the dead threads. Reusing the threads sidesteps that
        completely.

        ``other`` must be a freshly built, NEVER-STARTED executor (so its chain
        is un-set-up and it owns no threads). The swap runs on this executor's
        dispatcher thread (so the new chain's setup() — which calls ORT for
        source-face detection — runs on a persistent thread, not a per-swap
        one). Returns this executor's PREVIOUS (reader_pool, buffer) for the
        caller to shut down off-thread, or None if the swap failed (chain setup
        raised, the executor is stopped, or it timed out) — in which case the
        old world stays live and ``other``'s resources should be discarded.
        """
        with self._state_lock:
            if self._state is _State.STOPPED:
                return None
        done = threading.Event()
        old_out: list[tuple[ReaderPool, FrameBuffer]] = []
        error_out: list[str] = []
        self._command_queue.put(
            ReconfigureMsg(
                reader_pool=other._reader_pool,
                buffer=other._buffer,
                timeline=other._timeline,
                chain=other._chain,
                strategy=other._strategy,
                playback_mode=other._playback_mode,
                restore_frame=restore_frame,
                play=play,
                done=done,
                old_out=old_out,
                error_out=error_out,
            )
        )
        if not done.wait(timeout=timeout_s):
            self.status.set("reconfigure timed out")
            return None
        if error_out:
            self.status.set(f"reconfigure failed: {error_out[0]}")
            return None
        return old_out[0] if old_out else None

    def reads_per_second(self) -> float:
        """Latest read-throughput rate from the underlying ReaderPool.
        Exposed so the metrics overlay can sample without needing
        direct access to the pool."""
        return self._reader_pool.reads_per_second()

    def frame_count(self) -> int:
        """Total frames in the target — for jump-to-end seeks etc."""
        return self._reader_pool.frame_count

    def last_completed_frame(self) -> int:
        """Highest frame index a worker has finished. -1 before any
        completion — useful as a 'loading' signal for the metrics
        overlay (system is up but no frame has produced yet)."""
        with self._state_lock:
            return self._last_completed

    def on_frame_ready(self, callback: Callable[[Frame, FrameIndex], None]) -> None:
        self._on_frame = callback

    # ---- Dispatcher ----

    def _dispatcher_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._command_queue.get(timeout=_DISPATCHER_TICK_S)
                self._handle_message(msg)
                continue
            except Empty:
                pass

            with self._state_lock:
                should_submit = self._state is _State.PLAYING
            # Don't submit until the chain is set up. During model load the
            # workers are parked on _setup_done_event, so submitting here just
            # pre-fills the bounded work queue with opening frames that go stale
            # before the first one can be processed — the user then sees the
            # opening crawl past in slow-motion before the display snaps to
            # wall-clock. Setup is typically the longest cold-start phase.
            if should_submit and self._setup_done_event.is_set():
                self._try_submit_next_frame()

    def _handle_message(self, msg: Message) -> None:
        match msg:
            case PlayMsg():
                self._handle_play()
            case PauseMsg():
                self._handle_pause()
            case StopMsg():
                self._stop_event.set()
            case SeekMsg(target_frame=target):
                self._handle_seek(target)
            case RerenderMsg():
                self._handle_rerender()
            case SetChainMsg(chain=chain):
                self._handle_set_chain(chain)
            case SetSkipStrategyMsg(strategy=strategy):
                self._handle_set_strategy(strategy)
            case SetWorkerCountMsg(n=n):
                self._handle_set_worker_count(n)
            case SetPlaybackModeMsg(mode=mode):
                self._handle_set_playback_mode(mode)
            case ReconfigureMsg():
                self._handle_reconfigure(msg)
            case SetParamsMsg():
                self.status.set(
                    "set_params is not implemented; rebuild processors and use set_chain"
                )

    def _handle_play(self) -> None:
        with self._state_lock:
            from_frame = self._timeline.current_frame()
            self._timeline.start(from_frame=from_frame)
            self._state = _State.PLAYING
        self.is_playing.set(True)
        self._playback_wake.set()

    def _handle_pause(self) -> None:
        with self._state_lock:
            self._timeline.pause()
            self._state = _State.PAUSED
            paused_at = self._timeline.current_frame()
        self.is_playing.set(False)
        # Drop pending work so workers stop processing the pre-pause
        # backlog. Without this drain, with a slow chain (enhancer:
        # 1-3s/frame) the queue holds many frames at indices ≤ paused_at
        # — as each completes, playback's latest_index_at_or_below
        # fallback advances and the display visibly "plays" for the
        # duration of the queued work (up to MAX_WORKERS*2 frames *
        # per-frame time = 10s-60s). Inflight workers still finish their
        # current frame; that's bounded by the worker pool size.
        self._drain_work_queue()
        # The paused frame may have been canceled by the drain. Resubmit
        # it so the display converges on the correct frame rather than
        # whatever the worker happened to finish last.
        if paused_at >= 0:
            self._submit_specific_frame(paused_at)
        self._playback_wake.set()

    def _handle_seek(self, target: FrameIndex) -> None:
        self._drain_work_queue()
        # A seek is a discontinuity: tell chain processors to drop per-stream
        # caches (notably the swapper's interval-based face-detection cache).
        # Otherwise the first frame at the new position reuses a face box
        # detected at the OLD position, so the new face is swapped in the wrong
        # place — it shows up unswapped until the next re-detection.
        self._reset_chain_stream_state()
        with self._state_lock:
            self._timeline.seek(target)
            self._last_submitted = target - 1
            # Reset progress to the seek point in BOTH directions. Only raising
            # it (the old behaviour) left a backward seek with a stale-high
            # last_completed, which makes SyncedStrategy read "caught up" and
            # never engage its catch-up fallback. Mirror _handle_rerender /
            # _handle_reconfigure, which both clamp to the new position.
            self._last_completed = target - 1
        # Invalidate the buffer entry for the target so the next read
        # there returns None instead of OLD cached/store data. Combined
        # with the duplicate-frame-guard reset below, this guarantees
        # the worker's reprocessed frame for the same index actually
        # reaches the display.
        self._buffer.invalidate(target)
        # Reset the playback duplicate-frame guard. A seek can legitimately
        # re-emit at the SAME index — e.g. controller seeks to current_frame
        # after a chain swap so the new chain's reprocessed pixels reach
        # the display. Without this reset, the guard would compare the
        # just-reprocessed frame's index to the prior identical index
        # and skip the emit.
        self._last_shown_frame_index = None
        # IMPORTANT: state_lock released BEFORE the slow read/put path. If we
        # held the lock through the put, a full queue would freeze workers
        # (they need the lock to mark frames complete, which is what drains
        # the queue → deadlock-like priority inversion that caps throughput
        # at ~9 fps with 1 worker).
        self._submit_specific_frame(target)
        # Wake playback so a seek-while-paused tick processes the new
        # target immediately rather than waiting on an indefinite block.
        self._playback_wake.set()

    def _reset_chain_stream_state(self) -> None:
        """Drop per-stream caches on chain processors that expose an on_seek()
        hook (e.g. FaceSwapper's interval-based detection cache). Called on a
        seek so the new position re-detects rather than reusing stale state."""
        with self._state_lock:
            chain = self._chain
        for p in chain:
            hook = getattr(p, "on_seek", None)
            if hook is None:
                continue
            try:
                hook()
            except Exception as e:  # noqa: BLE001
                self.status.set(f"on_seek error: {e}")

    def _handle_rerender(self) -> None:
        # Reprocess from the playhead forward through the current chain. Same
        # shape as a seek-in-place, except we drop the cache/store FROM the
        # playhead onward (invalidate_from) so frames already processed ahead
        # with stale params are redone; frames before the playhead are kept.
        self._drain_work_queue()
        with self._state_lock:
            current = self._timeline.current_frame()
            self._last_submitted = current - 1
            self._last_completed = min(self._last_completed, current - 1)
        self._buffer.invalidate_from(current)
        self._last_shown_frame_index = None
        # Resubmit the current frame now so a paused display updates at once;
        # when playing, the dispatcher resubmits the rest as the playhead moves.
        self._submit_specific_frame(current)
        self._playback_wake.set()

    def _submit_specific_frame(self, frame_index: FrameIndex) -> None:
        """Enqueue a specific frame regardless of state. Must NOT be called while holding state_lock."""
        if frame_index < 0 or frame_index >= self._reader_pool.frame_count:
            return
        # Submit the read non-blockingly — a reader thread will produce
        # the frame; the worker will await the future.
        future = self._reader_pool.read_async(frame_index)
        item = WorkItem(frame_index=frame_index, source_future=future)
        try:
            self._work_queue.put(item, timeout=0.1)
        except Full:
            # Queue is full — cancel so the pool can skip this read if
            # it hasn't started, and don't leak the future.
            future.cancel()
            return
        with self._state_lock:
            self._last_submitted = frame_index

    def _handle_set_chain(self, chain: tuple[Processor, ...]) -> None:
        # Don't race the initial setup: the setup thread is calling
        # p.setup() on the current chain. Swapping it here would call
        # setup() on the new chain and release() on the old in parallel
        # with the initial setup, double-initialising or tearing down
        # resources mid-load. After initial setup completes the event
        # stays set, so this is free on every subsequent call.
        self._setup_done_event.wait()
        if self._stop_event.is_set():
            return
        self._drain_work_queue()
        old_chain = self._chain
        # Set up the NEW processors BEFORE exposing the chain. Workers read
        # self._chain WITHOUT the state lock, so assigning it first would let a
        # worker call process() on a processor whose model/session is still None
        # (RuntimeError → fatal worker error → whole-executor teardown). Setup
        # also runs OUTSIDE the state lock so the slow model load doesn't block
        # workers marking frames complete. The old chain stays live during setup.
        for p in chain:
            if self._stop_event.is_set():
                return
            if p not in old_chain:
                p.setup()
        with self._state_lock:
            self._chain = chain
            to_release = [p for p in old_chain if p not in chain]
        # Wait for any worker mid-_apply_chain to finish before releasing
        # the dropped processors. Without this, release() can null internal
        # state while a worker is still calling .process() on the instance.
        if to_release:
            self._wait_for_inflight()
            for p in to_release:
                p.release()

    def _wait_for_inflight(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self._inflight_cv:
            while self._inflight_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop_event.is_set():
                    return
                self._inflight_cv.wait(timeout=remaining)

    def _handle_reconfigure(self, msg: ReconfigureMsg) -> None:
        """Swap reader pool / buffer / timeline / chain in place, on the
        dispatcher thread, without disturbing the worker pool. See
        reconfigure_from for why this avoids the per-thread CUDA leak."""
        # 1) Set up the NEW chain HERE (dispatcher thread) so the source-face
        #    detector's ORT call runs on a persistent thread, not a per-swap
        #    one. On failure, abandon the swap and leave the old world live.
        try:
            for p in msg.chain:
                if self._stop_event.is_set():
                    msg.error_out.append("stopped during reconfigure")
                    msg.done.set()
                    return
                p.setup()
        except Exception as e:  # noqa: BLE001 — surfaced to the caller via error_out
            msg.error_out.append(str(e))
            msg.done.set()
            return
        # 2) Quiesce the OLD world: cancel queued reads and wait for any worker
        #    mid-process to finish on the old chain/buffer before we swap +
        #    release it (same hazard set_chain guards against).
        self._drain_work_queue()
        self._wait_for_inflight()
        with self._state_lock:
            old_chain = self._chain
            old_reader_pool = self._reader_pool
            old_buffer = self._buffer
            self._reader_pool = msg.reader_pool
            self._buffer = msg.buffer
            self._timeline = msg.timeline
            self._timeline.set_max_frame(max(0, msg.reader_pool.frame_count - 1))
            self._chain = msg.chain
            self._strategy = msg.strategy
            self._playback_mode = msg.playback_mode
            # New world → reset progress trackers and the playback dup-guard so
            # the restored frame is actually re-emitted to the display.
            self._last_submitted = msg.restore_frame - 1
            self._last_completed = msg.restore_frame - 1
            self._last_shown_frame_index = None
            self._timeline.seek(msg.restore_frame)
            if msg.play:
                self._timeline.start(from_frame=msg.restore_frame)
                self._state = _State.PLAYING
            else:
                self._timeline.pause()
                self._state = _State.PAUSED
        # 3) Release the OLD chain's processors (drops refs; cached/shared models
        #    like the inswapper persist in model_cache). Off the state lock.
        for p in old_chain:
            try:
                p.release()
            except Exception as e:  # noqa: BLE001
                self.status.set(f"reconfigure release error: {e}")
        # 4) Publish the new state and submit the restore frame so the display
        #    reflects the new source/target immediately, then hand the old
        #    reader pool + buffer back for off-thread shutdown.
        self.is_playing.set(msg.play)
        self.strategy_mode.set(self._strategy.current_mode())
        self.metrics.set(self._buffer.metrics())
        self._submit_specific_frame(msg.restore_frame)
        self._playback_wake.set()
        msg.old_out.append((old_reader_pool, old_buffer))
        msg.done.set()

    def _handle_set_strategy(self, strategy: FrameSkipStrategy) -> None:
        self._drain_work_queue()
        with self._state_lock:
            self._strategy = strategy
        # Refresh the surfaced mode immediately so the status bar
        # reflects the new strategy's initial state without waiting
        # for the next decide() tick.
        self.strategy_mode.set(strategy.current_mode())

    def _handle_set_playback_mode(self, mode: PlaybackMode) -> None:
        with self._state_lock:
            self._playback_mode = mode
        # Wake immediately so the new sleep interval kicks in on the next
        # tick rather than after the old interval expires.
        self._playback_wake.set()

    def _handle_set_worker_count(self, n: int) -> None:
        """Scale the worker pool to size n. Runs on the dispatcher thread.

        Adding workers: spawn against the existing shared chain — no setup()
        re-runs and no GPU memory churn. New workers start consuming from
        the queue immediately.

        Removing workers: signal exit_event on the surplus handles. Those
        workers finish whatever frame they currently hold (so released
        processors don't see a torn-down chain) and exit at their next
        loop iteration. They stay in self._workers until pruned on a
        later call — is_alive() reflects their real state in the meantime.
        """
        n = max(1, min(n, MAX_WORKERS))
        self._workers = [h for h in self._workers if h.thread.is_alive()]
        current = len(self._workers)
        if n == current:
            return
        if n > current:
            for _ in range(n - current):
                self._spawn_worker()
        else:
            for h in self._workers[n:]:
                h.exit_event.set()

    def _drain_work_queue(self) -> None:
        while True:
            try:
                item = self._work_queue.get_nowait()
            except Empty:
                break
            # Cancel the source future on each drained item so any
            # reader-pool thread holding the request skips it. Sentinels
            # (None) don't carry futures.
            if item is not _WORKER_SENTINEL:
                item.source_future.cancel()

    def _try_submit_next_frame(self) -> None:
        """Decide what to submit (under lock), then submit (without lock).

        The `with state_lock` window is kept TINY because workers also need
        the lock to mark frames complete. The actual source-frame read is
        now off-thread (a ReaderPool thread services the future), so the
        dispatcher never blocks on I/O — earlier versions of this method
        held the lock through a synchronous reader.read() and serialised
        the whole pipeline to 1 fps on slow sources.
        """
        with self._state_lock:
            metrics = self._buffer.metrics()
            decision = self._strategy.decide(
                last_submitted=self._last_submitted,
                last_completed=self._last_completed,
                timeline=self._timeline,
                metrics=metrics,
                # How expensive reads are right now, so SyncedStrategy can tell
                # an I/O-bound source (skip → thrash) from a compute-bound one
                # (skip → free, stay synced).
                read_latency_ms=self._reader_pool.recent_read_latency_ms(),
            )
            # Read strategy mode right after decide so the value reflects
            # this exact tick's behaviour (SyncedStrategy sets _in_fallback
            # inside decide). Done inside the lock to keep the strategy
            # access ordered relative to other strategy mutations.
            mode = self._strategy.current_mode()
            if decision.next_frame is None:
                # Even when idle, surface the mode so a user toggling
                # play/pause sees the right label.
                self.strategy_mode.set(mode)
                return
            frame_index = decision.next_frame
            last_frame = self._reader_pool.frame_count - 1
            # End-of-playback is when the DISPLAY (wall-clock playhead) has
            # reached the last frame AND that frame is actually rendered — NOT
            # when submission runs off the end. A faster-than-target pipeline
            # submits the whole clip ahead of the playhead; keying the end on
            # the submission index froze playback partway through. The playhead
            # is clamped to last_frame (Timeline.set_max_frame), so it settles
            # there and this fires once the tail frame completes.
            # Robustness: if the last frame was submitted but can NEVER complete
            # (its read returned None / raised — a transient slow-source hiccup),
            # last_completed stays below last_frame forever and the dispatcher
            # won't resubmit it (idle branch below). End anyway once nothing is
            # left in flight, so playback can't hang one frame short of the end.
            reached_end = self._timeline.current_frame() >= last_frame
            last_done = self._last_completed >= last_frame
            stuck_at_end = (
                self._last_submitted >= last_frame
                and self._work_queue.empty()
                and self._inflight_count == 0
            )
            if reached_end and (last_done or stuck_at_end):
                self._timeline.pause()
                self._state = _State.PAUSED
                self.is_playing.set(False)
                self.status.set("end of target")
                self.strategy_mode.set(mode)
                return
            if frame_index > last_frame:
                # decide wants a frame past the end (everything up to last_frame
                # is already submitted). Nothing new to do this tick — idle and
                # let the playhead / workers advance toward the end condition.
                self.strategy_mode.set(mode)
                return

        # Lock RELEASED here. Publish mode outside the lock to avoid
        # serialising observable subscribers (the GUI bridge) on it.
        self.strategy_mode.set(mode)
        # Submit the read non-blockingly — a reader thread will produce
        # the frame; the worker awaits the future.
        future = self._reader_pool.read_async(frame_index)
        item = WorkItem(frame_index=frame_index, source_future=future)
        try:
            self._work_queue.put(item, timeout=0.1)
        except Full:
            # Queue is full — cancel the future so the pool can skip
            # this read if it hasn't started, and don't leak it.
            future.cancel()
            return
        with self._state_lock:
            if frame_index > self._last_submitted:
                self._last_submitted = frame_index

    # ---- Workers ----

    def _worker_loop(self, exit_event: threading.Event) -> None:
        """Pop items off the work queue and run the chain.

        Two ways to exit: this worker's own exit_event (set when the pool
        shrinks) or the global _stop_event (set on full executor stop). The
        poll timeout caps the worst-case shutdown latency at _WORKER_POLL_S
        when no sentinel is available (idle pool, no item to wake on).
        """
        # Park until the chain has been set up. Poll in short bursts so
        # exit_event / stop_event are still honoured during a multi-second
        # model load. _setup_done_event is set whether setup succeeded or
        # failed; the stop-event check after the wait covers the failure
        # path (setup raises → status set → stop_event set → exit).
        while not self._setup_done_event.is_set():
            if exit_event.is_set() or self._stop_event.is_set():
                return
            self._setup_done_event.wait(timeout=_WORKER_POLL_S)
        while not (exit_event.is_set() or self._stop_event.is_set()):
            try:
                item = self._work_queue.get(timeout=_WORKER_POLL_S)
            except Empty:
                continue
            if item is _WORKER_SENTINEL:
                # stop() pushes sentinels for a fast wake; honour them even
                # if neither event is set yet (the events typically are by
                # then, but don't depend on it).
                break
            # Await the source frame from the reader pool. Failures here
            # are non-fatal (one bad frame doesn't kill the executor):
            # cancellation, reader exception, and None all just skip
            # this item. Only chain errors below are fatal.
            try:
                source_frame = item.source_future.result(
                    timeout=_WORKER_READ_TIMEOUT_S
                )
            except CancelledError:
                continue
            except Exception as e:
                self.status.set(f"reader error at {item.frame_index}: {e}")
                continue
            if source_frame is None:
                self.status.set(f"target.read({item.frame_index}) returned None")
                continue
            # inflight is counted only around chain execution, NOT around
            # source-future await — _wait_for_inflight (used by set_chain)
            # shouldn't block on slow source I/O for a frame that hasn't
            # even entered the chain yet.
            with self._inflight_cv:
                self._inflight_count += 1
            try:
                # Re-read every iteration so set_chain's swap is picked up
                # without restarting the worker thread. ORT InferenceSession
                # is thread-safe for concurrent .run() calls, so multiple
                # workers calling the same chain's swapper.get() in parallel
                # is the intended fast path.
                chain = self._chain
                result = self._apply_chain(source_frame, chain)
                self._buffer.put(item.frame_index, result)
                # Wake playback so the freshly-written frame is picked
                # up on its next tick. Without this, a seek-while-paused
                # (or post-rebuild seek) is racy: playback wakes after
                # the seek's enqueue, ticks against an empty buffer, and
                # then sleeps forever waiting for an event. The worker
                # producing the frame is exactly the signal playback
                # needs. Set is O(1) and the dup-frame guard suppresses
                # redundant emits when the index hasn't changed.
                self._playback_wake.set()
                with self._state_lock:
                    if self._last_completed < item.frame_index:
                        self._last_completed = item.frame_index
                self._record_completion()
            except Exception as e:
                # A per-frame chain/buffer error is RECOVERABLE — log it and
                # skip this frame, exactly like a reader error above. Do NOT set
                # _stop_event: that tears down the whole executor (dispatcher +
                # all workers + playback) on a single transient bad frame WITHOUT
                # going through stop(), so processors are never release()d (GPU
                # held) and _state is left lying. The finally block restores
                # _inflight_count. Use a non-"worker error" prefix so the GUI
                # surfaces it in the status bar instead of popping a modal per
                # frame (the "worker error" prefix routes to errorOccurred).
                self.status.set(f"frame error at {item.frame_index}: {e}")
                continue
            finally:
                with self._inflight_cv:
                    self._inflight_count -= 1
                    if self._inflight_count == 0:
                        self._inflight_cv.notify_all()
        # Worker is exiting (pool shrank, or executor stopping): drop any
        # per-thread processor instances THIS worker built so a live shrink
        # frees the surplus model now, not at chain teardown.
        self._release_thread_local_chain()

    def _release_thread_local_chain(self) -> None:
        """Release any per-thread processor instances the calling (exiting)
        worker built — e.g. a PerWorkerProcessor's own GFPGAN. Plain shared
        processors don't expose release_thread_local() and are skipped. On a
        full stop() the chain is released wholesale anyway; this matters for
        the live worker-count-DECREASE path, where the surplus worker's model
        would otherwise linger until the next chain swap."""
        for p in self._chain:
            release = getattr(p, "release_thread_local", None)
            if release is None:
                continue
            try:
                release()
            except Exception as e:  # noqa: BLE001
                self.status.set(f"thread-local release error: {e}")

    def _apply_chain(self, frame: Frame, chain: tuple[Processor, ...]) -> Frame:
        # Wrap each processor with perf_counter so the metrics overlay
        # can attribute wall-clock to FaceSwapper vs FaceEnhancer. The
        # measurement excludes the chain-iteration overhead and the
        # buffer.put (those aren't processor work); strict measurement
        # of the .process() call only.
        for p in chain:
            t0 = time.perf_counter_ns()
            frame = p.process(frame)
            elapsed_ns = time.perf_counter_ns() - t0
            with self._timings_lock:
                self._processor_timings.append((time.monotonic(), p.name, elapsed_ns))
        return frame

    def processor_timings(self) -> dict[str, float]:
        """Average milliseconds per process() call over the last
        _TIMING_WINDOW_S seconds, per processor name.

        Used by the metrics overlay to surface where each frame's
        wall-clock is going. Aged-out entries are trimmed lazily on
        read so the deque doesn't bloat between ticks; callers should
        treat the result as the current rolling average, not cumulative.
        Empty dict when no frames have been processed in the window
        (paused or idle)."""
        cutoff = time.monotonic() - _TIMING_WINDOW_S
        sums: dict[str, list[int]] = {}
        with self._timings_lock:
            while self._processor_timings and self._processor_timings[0][0] < cutoff:
                self._processor_timings.popleft()
            for _ts, name, ns in self._processor_timings:
                sums.setdefault(name, []).append(ns)
        return {
            name: (sum(vals) / len(vals)) / 1_000_000.0
            for name, vals in sums.items()
        }

    def _record_completion(self) -> None:
        """Append a completion timestamp. Cheap by design — no calculation
        and no observable publish in the worker hot path. The playback
        thread reads these timestamps and publishes processing_fps."""
        with self._fps_lock:
            now = time.monotonic()
            self._completion_times.append(now)
            self._last_completion_time = now

    def _refresh_fps(self) -> None:
        """Trim timestamps older than _FPS_WINDOW_S and publish processing_fps.

        Called from the playback loop at _PLAYBACK_TICK_S cadence so the
        observable updates ~30 times/sec from a single thread regardless
        of worker count. FPS = (count-1)/span over the trimmed window;
        0.0 when fewer than two timestamps remain (idle or just started).
        """
        now = time.monotonic()
        cutoff = now - _FPS_WINDOW_S
        fps = 0.0
        with self._fps_lock:
            while self._completion_times and self._completion_times[0] < cutoff:
                self._completion_times.popleft()
            count = len(self._completion_times)
            last = self._last_completion_time
            if count >= 2:
                span = self._completion_times[-1] - self._completion_times[0]
                if span > 0:
                    fps = (count - 1) / span
                self._last_fps = fps
            elif last is not None and (now - last) <= _FPS_STALL_HOLD_S:
                # Sparse: report the rate implied by the time since the last
                # completion, decaying toward 0 the longer it's been. Cap with
                # the last windowed value so the instant after a completion
                # doesn't spike to a huge 1/tiny-elapsed reading.
                elapsed = now - last
                decayed = (1.0 / elapsed) if elapsed > 0 else self._last_fps
                cap = self._last_fps if self._last_fps > 0 else _FPS_DECAY_CAP
                fps = min(decayed, cap)
        self.processing_fps.set(fps)

    # ---- Playback ----

    def _playback_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._do_playback_tick()
            except Exception as e:
                self.status.set(f"playback error: {e}")
            sleep_s = self._compute_playback_sleep()
            # wait(None) blocks indefinitely; any wake source (play, pause,
            # seek, mode change, stop) calls _playback_wake.set() to unblock.
            self._playback_wake.wait(timeout=sleep_s)
            self._playback_wake.clear()

    def _do_playback_tick(self) -> None:
        with self._state_lock:
            is_paused = self._state is _State.PAUSED
        index, frame = self._buffer.get_at_current_time()
        # Track which actual frame's pixels we're about to display — the
        # timeline index when we have a direct hit, the fallback index when
        # the worker hasn't caught up. The duplicate guard compares this so
        # we suppress redundant on_frame calls even when the timeline
        # advances over still-unprocessed positions.
        shown_index: FrameIndex | None = index if frame is not None else None
        if frame is None and not is_paused:
            # Timeline is ahead of the worker — show the most recently
            # written frame ≤ current target so the display doesn't stay
            # blank. Clamp to ≤ target avoids the "stuck on a future frame"
            # effect after a backward seek.
            # SUPPRESSED WHEN PAUSED: workers in-flight from before pause
            # complete frames at indices ≤ paused_at as they drain. If we
            # fell back to them, each completion would advance the displayed
            # frame and the user would see seconds of "playback" continuing
            # after pressing pause. Instead, only emit on an exact match
            # at the paused frame — the worker's resubmit (from _handle_pause)
            # will produce that exact frame and the display converges.
            fallback_index = self._buffer.latest_index_at_or_below(index)
            # Don't repaint an OLDER frame than what's already on screen during
            # forward playback — that's a visible backward stutter (the newest
            # frame ≤ target can be lower than the last shown when skipped
            # frames complete out of order or an old one is evicted). Hold the
            # current frame instead. A seek resets _last_shown_frame_index to
            # None, so seeks (incl. backward) still repaint freely.
            if fallback_index is not None and (
                self._last_shown_frame_index is None
                or fallback_index >= self._last_shown_frame_index
            ):
                frame = self._buffer.get(fallback_index)
                shown_index = fallback_index
        if (
            frame is not None
            and self._on_frame is not None
            and shown_index != self._last_shown_frame_index
        ):
            try:
                self._on_frame(frame, index)
                self._last_shown_frame_index = shown_index
            except Exception as e:
                self.status.set(f"on_frame callback error: {e}")
        self.current_frame.set(index)
        self.metrics.set(self._buffer.metrics())
        self._refresh_fps()

    def _compute_playback_sleep(self) -> float | None:
        """How long to wait before the next playback tick.

        Returns None for "block until woken" — used when nothing is
        producing frame changes (paused or idle). Otherwise returns the
        per-mode tick interval. The wake event interrupts the wait early
        on any state change, so this is purely the upper bound between
        ticks during normal playback.
        """
        with self._state_lock:
            state = self._state
            mode = self._playback_mode
        if state is not _State.PLAYING:
            return None
        if mode is PlaybackMode.UNLIMITED:
            return _UNLIMITED_PLAYBACK_TICK_S
        if mode is PlaybackMode.SOURCE:
            return 1.0 / self._timeline.fps
        return _FIXED_PLAYBACK_TICK_S
