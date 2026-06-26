import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.types import FrameIndex


@dataclass(frozen=True)
class SkipDecision:
    """Strategy output. next_frame=None means 'idle — no work to submit now'."""

    next_frame: FrameIndex | None


@runtime_checkable
class FrameSkipStrategy(Protocol):
    """Decide what frame the dispatcher should submit next.

    Called once per dispatcher tick. The implementation gets enough state
    (last submitted, last completed, current timeline position, lag metrics)
    to choose between submitting the next sequential frame, skipping ahead
    to catch up with wall-clock, or returning None to wait.

    The last three arguments are optional context a predictive strategy needs
    and simpler ones ignore: ``process_fps`` (current pipeline throughput),
    ``worker_count`` (parallelism), and ``outstanding`` (queued + in-flight
    work) — enough to aim a frame at where the playhead WILL be when it
    finishes, and to keep the in-flight depth shallow.
    """

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
        read_latency_ms: float | None = None,
        process_fps: float | None = None,
        worker_count: int = 1,
        outstanding: int = 0,
    ) -> SkipDecision: ...

    def current_mode(self) -> str:
        """Human-readable label for the strategy's current behaviour.

        Surfaces in the status bar so the user can see when an adaptive
        strategy has shifted modes (e.g. SyncedStrategy falling back to
        sequential submission when reads can't keep up). Stable strategies
        like BestEffort just return a fixed label."""
        ...


class BestEffortStrategy:
    """Submit every frame in order. Let playback lag if processing is slow.

    Right for tuning workflows where the user wants to see the effect of every
    parameter change on every frame. Wrong for watching content where wall-
    clock sync matters — that's what SyncedStrategy is for.

    Callers should initialize last_submitted to -1 so the first call returns 0.
    """

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
        read_latency_ms: float | None = None,
        process_fps: float | None = None,
        worker_count: int = 1,
        outstanding: int = 0,
    ) -> SkipDecision:
        return SkipDecision(next_frame=last_submitted + 1)

    def current_mode(self) -> str:
        return "best effort"


class SyncedStrategy:
    """Skip ahead to the wall-clock target frame when processing falls behind.

    Right for viewing — the displayed frame stays as close to the timeline as
    throughput permits, by dropping intermediate frames from the work queue.
    Never goes backwards: if the timeline somehow reports a lower frame than
    we've already submitted (e.g. clock drift), the next submission still
    advances by one.

    Adaptive fallback. When processing falls catastrophically behind
    (`target - last_completed > max_lag_frames`), keeping up by jumping
    ahead is impossible — each jump is a random-access read, and on slow
    sources (network share, HDD) random reads cost a seek + fresh fetch
    that the underlying I/O can't sustain. Aggressive skipping in that
    regime starves the reader and produces 0 fps. We fall back to
    sequential submission (`last_submitted + 1`) so reads stay sequential,
    the reader's prefetch buffer absorbs them, and throughput recovers
    to whatever the pipeline can actually sustain. Display lags the
    timeline, but progress is made.

    Recovery is implicit: once `last_completed` catches up to within
    `recover_lag_frames` of `target`, the next call returns the target again.

    Self-correcting: if sequential ISN'T catching up — the lag keeps climbing past
    where it was when we fell back — the bottleneck is throughput (heavy compute
    or decode), not seek-thrash, so playing every frame in order is just slow-
    motion that drifts further behind. We then abandon the fallback and resume
    skipping (drop frames, stay synced with wall-clock), staying disarmed until
    skipping pulls the lag back under `recover_lag_frames` so the mode can't flap.
    """

    # 60 frames ≈ 2 seconds at 30 fps. Tuned to absorb brief slow-downs
    # without giving up on sync, while triggering quickly enough to
    # prevent the death-spiral on truly slow sources.
    _DEFAULT_MAX_LAG_FRAMES = 60

    # How far ahead of the wall-clock playhead the dispatcher may pre-render.
    # ~4 s at 30 fps: enough cushion to ride out brief slow-downs without a
    # faster-than-target pipeline rendering the whole rest of the clip ahead of
    # the display (wasted compute + memory, all discarded on a seek).
    _DEFAULT_LOOKAHEAD_FRAMES = 120

    # A single read slower than this (ms) means the SOURCE is the bottleneck:
    # skipping then forces random-access reads a slow source can't sustain
    # (network share, HDD, ffmpeg keyframe-seek restart ~100-200 ms), so the
    # sequential fallback is warranted. Below it the disk keeps up and skipping
    # is free — stay synced. ~1.5x a 30 fps frame budget; cheap SSD / cv2 in-
    # place seek reads (single-digit ms) never trip it.
    _DEFAULT_IO_BOUND_READ_MS = 50.0

    def __init__(
        self,
        max_lag_frames: int | None = None,
        recover_lag_frames: int | None = None,
        lookahead_frames: int | None = None,
        io_bound_read_ms: float | None = None,
    ) -> None:
        self._max_lag_frames = (
            max_lag_frames
            if max_lag_frames is not None
            else self._DEFAULT_MAX_LAG_FRAMES
        )
        self._lookahead_frames = (
            lookahead_frames
            if lookahead_frames is not None
            else self._DEFAULT_LOOKAHEAD_FRAMES
        )
        self._io_bound_read_ms = (
            io_bound_read_ms
            if io_bound_read_ms is not None
            else self._DEFAULT_IO_BOUND_READ_MS
        )
        # Hysteresis: enter fallback above max_lag, but don't LEAVE it until lag
        # drops below this lower bound. Without the gap, a lag parked near the
        # threshold flaps the mode (and the read pattern) on every frame. Default
        # to half the enter threshold; clamp so it's never above it.
        self._recover_lag_frames = (
            recover_lag_frames
            if recover_lag_frames is not None
            else self._max_lag_frames // 2
        )
        self._recover_lag_frames = min(self._recover_lag_frames, self._max_lag_frames)
        # Updated on each decide() call so current_mode() can report
        # whether we're keeping up or in fallback. The dispatcher reads
        # current_mode() right after decide(), so the value is fresh.
        self._in_fallback = False
        # Self-correcting fallback state: the lowest lag reached since entering
        # (if sequential is helping, lag falls toward recover_lag), and a latch
        # that keeps us OUT of the fallback once we've abandoned it as unhelpful
        # — until skipping pulls the lag back under recover_lag.
        self._fallback_best_lag = 0
        self._fallback_disarmed = False
        # Whether ANY frame has completed this session. Warm-up (cold start,
        # first frame loading models) is keyed on this, NOT on last_completed<0
        # alone — a mid-session seek to frame 0 also drives last_completed to -1
        # and must not be mistaken for a cold start (which would flood the
        # opening sequentially instead of tracking wall-clock).
        self._ever_completed = False

    @property
    def max_lag_frames(self) -> int:
        return self._max_lag_frames

    @property
    def recover_lag_frames(self) -> int:
        return self._recover_lag_frames

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
        read_latency_ms: float | None = None,
        process_fps: float | None = None,
        worker_count: int = 1,
        outstanding: int = 0,
    ) -> SkipDecision:
        target = timeline.current_frame()
        if last_completed >= 0:
            self._ever_completed = True
        # Warm-up: nothing has completed yet (cold start — the first frame is
        # still loading models / running its first inference, which can take
        # seconds). Don't chase the wall-clock target: it climbs the whole time,
        # so max(last_submitted+1, target) would fill the queue with sparse high
        # indices and SKIP the opening of the clip. Submit sequentially from
        # where we are so the opening frames are actually processed. Gated on
        # _ever_completed so a mid-session seek-to-0 (also last_completed<0)
        # isn't misread as a cold start.
        if last_completed < 0 and not self._ever_completed:
            self._in_fallback = False
            return SkipDecision(next_frame=last_submitted + 1)
        # Fall back to sequential only when we're catastrophically behind AND
        # READS are the bottleneck (I/O-bound): skipping then means random reads
        # a slow source can't sustain (→ ~0 fps death-spiral). When COMPUTE is
        # the bottleneck (fast disk, slow GPU), reads are cheap, so keep skipping
        # to stay synced — sequential there is just slow-motion drift. Hysteresis:
        # enter above max_lag, stay until lag drops below recover_lag, so a lag
        # parked at the boundary doesn't flap the mode/read pattern. No latency
        # signal → assume compute-bound (don't degrade to slow-motion blindly).
        lag = target - last_completed
        io_bound = (
            read_latency_ms is not None and read_latency_ms > self._io_bound_read_ms
        )
        if self._in_fallback:
            # If sequential is working, lag falls toward recover_lag; track the
            # best (lowest) reached so we can tell progress from a death spiral.
            self._fallback_best_lag = min(self._fallback_best_lag, lag)
            if lag <= self._recover_lag_frames:
                self._in_fallback = False
            elif lag > self._fallback_best_lag + self._max_lag_frames:
                # Sequential isn't catching up — the bottleneck is throughput
                # (heavy compute/decode), not seek-thrash, so playing every frame
                # in order is just slow-motion that falls further behind. Abandon
                # to skip-ahead (drop frames, stay synced); stay disarmed until
                # skipping pulls the lag back under recover_lag so we don't flap.
                self._in_fallback = False
                self._fallback_disarmed = True
        elif self._fallback_disarmed:
            if lag <= self._recover_lag_frames:
                self._fallback_disarmed = False
        elif lag > self._max_lag_frames and io_bound:
            self._in_fallback = True
            self._fallback_best_lag = lag
        if self._in_fallback:
            return SkipDecision(next_frame=last_submitted + 1)
        nxt = max(last_submitted + 1, target)
        if nxt > target + self._lookahead_frames:
            # Already rendered the full look-ahead window past the playhead —
            # idle rather than pre-render further. Bounds a faster-than-target
            # pipeline's render-ahead to a fixed cushion.
            return SkipDecision(next_frame=None)
        return SkipDecision(next_frame=nxt)

    def current_mode(self) -> str:
        return "synced (lagging)" if self._in_fallback else "synced"


class PredictiveStrategy:
    """Predict-ahead viewing strategy — real-time playback under a slow pipeline.

    SyncedStrategy submits the frame that is current *at submit time* and lets a
    deep work queue build up. Both hurt when processing can't keep up: the deep
    FIFO fills with sequential frames during warm-up, so later skip-to-target
    submissions land BEHIND that backlog and the worker crawls through stale
    frames in order — the display plays in slow-motion. This strategy fixes both:

    1. Shallow pipeline. Idle (next_frame=None) once outstanding work
       (queued + in-flight) reaches ``outstanding_factor * worker_count``. With
       only a couple of frames per worker in flight, a submitted frame starts
       processing almost immediately, so "what we submit" is "what shows next" —
       there is no backlog to crawl through.

    2. Predict-ahead lead. Aim at ``target + lead``, where ``lead`` is how many
       frames the wall-clock advances while this frame is processed. By Little's
       law the in-pipeline latency is ``outstanding_cap / processing_fps``
       seconds, so:

           lead = ceil(outstanding_cap * fps / processing_fps)

       A frame submitted now then finishes just as the playhead reaches it →
       near-zero display lag, showing every Nth frame in real time instead of
       every frame in slow-motion. By default ``lead`` is clamped only to
       ``[0, lookahead]`` — it compensates the FULL measured latency, so the
       residual lag is ~0 at any worker count (the I/O-bound fallback handles
       slow-seek sources, which never reach the lead). ``max_lead_seconds`` is an
       optional TIGHTER user cap; ``None`` (the default) means auto.

    Inherits SyncedStrategy's safeguards: cold-start warm-up (submit sequentially
    until the first completion so the opening isn't skipped) and the I/O-bound →
    sequential fallback with hysteresis (on a slow source, skipping forces random
    reads it can't sustain → fall back so reads stay sequential).
    """

    # Outstanding work cap = this * worker_count. 2 = one frame computing + one
    # queued per worker: keeps workers fed (no refill stall) while staying
    # shallow enough that submit-to-show latency is ~2 frame-computes, which the
    # lead compensates for exactly (Little's law uses this same cap).
    _DEFAULT_OUTSTANDING_FACTOR = 2
    # Catastrophic-lag + recovery thresholds and the I/O-bound read cutoff —
    # identical rationale to SyncedStrategy (see its docstring).
    _DEFAULT_MAX_LAG_FRAMES = 60
    _DEFAULT_LOOKAHEAD_FRAMES = 120
    _DEFAULT_IO_BOUND_READ_MS = 50.0

    def __init__(
        self,
        max_lead_seconds: float | None = None,
        outstanding_factor: int | None = None,
        max_lag_frames: int | None = None,
        recover_lag_frames: int | None = None,
        lookahead_frames: int | None = None,
        io_bound_read_ms: float | None = None,
    ) -> None:
        # None → auto: compensate the full measured in-flight latency (bounded by
        # lookahead_frames) so the residual lag is ~0 at any worker count. A set
        # value is an optional tighter cap on render-ahead for slow sources.
        self._max_lead_seconds = max_lead_seconds
        self._outstanding_factor = (
            outstanding_factor
            if outstanding_factor is not None
            else self._DEFAULT_OUTSTANDING_FACTOR
        )
        self._max_lag_frames = (
            max_lag_frames
            if max_lag_frames is not None
            else self._DEFAULT_MAX_LAG_FRAMES
        )
        self._lookahead_frames = (
            lookahead_frames
            if lookahead_frames is not None
            else self._DEFAULT_LOOKAHEAD_FRAMES
        )
        self._io_bound_read_ms = (
            io_bound_read_ms
            if io_bound_read_ms is not None
            else self._DEFAULT_IO_BOUND_READ_MS
        )
        self._recover_lag_frames = (
            recover_lag_frames
            if recover_lag_frames is not None
            else self._max_lag_frames // 2
        )
        self._recover_lag_frames = min(self._recover_lag_frames, self._max_lag_frames)
        # See SyncedStrategy: report keeping-up vs the sequential fallback, and
        # distinguish a cold start from a mid-session seek-to-0.
        self._in_fallback = False
        self._fallback_best_lag = 0
        self._fallback_disarmed = False
        self._ever_completed = False

    @property
    def max_lead_seconds(self) -> float | None:
        return self._max_lead_seconds

    @property
    def max_lag_frames(self) -> int:
        return self._max_lag_frames

    def decide(
        self,
        last_submitted: FrameIndex,
        last_completed: FrameIndex,
        timeline: Timeline,
        metrics: BufferMetrics,
        read_latency_ms: float | None = None,
        process_fps: float | None = None,
        worker_count: int = 1,
        outstanding: int = 0,
    ) -> SkipDecision:
        target = timeline.current_frame()
        if last_completed >= 0:
            self._ever_completed = True
        cap = max(1, worker_count) * self._outstanding_factor
        # (1) Shallow pipeline: keep only ~a couple of frames per worker in
        # flight. A deep FIFO is what turns a slow pipeline into slow-motion —
        # skip-ahead submissions queue behind a sequential backlog the worker
        # drains in order. Idle once enough is in flight to keep workers busy.
        if outstanding >= cap:
            return SkipDecision(next_frame=None)
        # Cold-start warm-up: nothing has completed yet (first frame still
        # loading models / running its first inference). Submit sequentially so
        # the opening is processed rather than skipped while the clock climbs.
        # Gated on _ever_completed so a mid-session seek-to-0 isn't misread.
        if last_completed < 0 and not self._ever_completed:
            self._in_fallback = False
            return SkipDecision(next_frame=last_submitted + 1)
        # I/O-bound catastrophic-lag fallback (mirrors SyncedStrategy): far
        # behind AND reads are the bottleneck → skipping forces random reads a
        # slow source can't sustain, so go sequential and let prefetch absorb
        # them. Hysteresis (enter above max_lag, leave below recover_lag) keeps
        # a lag parked at the boundary from flapping the mode.
        lag = target - last_completed
        io_bound = (
            read_latency_ms is not None and read_latency_ms > self._io_bound_read_ms
        )
        if self._in_fallback:
            # If sequential is working, lag falls toward recover_lag; track the
            # best (lowest) reached so we can tell progress from a death spiral.
            self._fallback_best_lag = min(self._fallback_best_lag, lag)
            if lag <= self._recover_lag_frames:
                self._in_fallback = False
            elif lag > self._fallback_best_lag + self._max_lag_frames:
                # Sequential isn't catching up — the bottleneck is throughput
                # (heavy compute/decode), not seek-thrash, so playing every frame
                # in order is just slow-motion that falls further behind. Abandon
                # to skip-ahead (drop frames, stay synced); stay disarmed until
                # skipping pulls the lag back under recover_lag so we don't flap.
                self._in_fallback = False
                self._fallback_disarmed = True
        elif self._fallback_disarmed:
            if lag <= self._recover_lag_frames:
                self._fallback_disarmed = False
        elif lag > self._max_lag_frames and io_bound:
            self._in_fallback = True
            self._fallback_best_lag = lag
        if self._in_fallback:
            return SkipDecision(next_frame=last_submitted + 1)
        # (2) Predict-ahead: aim where the playhead will be when this frame
        # finishes, so it lands on time instead of one pipeline-latency late.
        lead = self._compute_lead(cap, timeline.fps, process_fps)
        nxt = max(last_submitted + 1, target + lead)
        if nxt > target + self._lookahead_frames:
            # Already rendered the full look-ahead window — idle (a backstop;
            # the outstanding gate above usually bounds the depth first).
            return SkipDecision(next_frame=None)
        return SkipDecision(next_frame=nxt)

    def _compute_lead(
        self, cap: int, fps: float, process_fps: float | None
    ) -> int:
        """Frames the wall-clock advances during one frame's time in the
        pipeline — aim this far ahead so a submission lands ON the playhead
        instead of one in-flight latency late. 0 until throughput is known
        (warm-up) → aim at the present target (skip-to-now).

        By default (``max_lead_seconds is None``) it compensates the FULL
        measured latency, so the residual lag is ~0 at any worker count, bounded
        only by ``lookahead_frames`` (a sustained-low throughput reading then
        can't aim at the end of the clip; the I/O-bound fallback already protects
        slow-seek sources, which never reach this code). A set ``max_lead_seconds``
        is an optional tighter cap on render-ahead."""
        if not process_fps or process_fps <= 0:
            return 0
        lead = math.ceil(cap * fps / process_fps)
        bound = self._lookahead_frames
        if self._max_lead_seconds is not None:
            bound = min(bound, int(self._max_lead_seconds * fps))
        return max(0, min(lead, bound))

    def current_mode(self) -> str:
        return "predictive (lagging)" if self._in_fallback else "predictive"
