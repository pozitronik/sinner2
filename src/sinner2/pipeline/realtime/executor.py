import threading
import time
from collections.abc import Callable, Mapping
from enum import Enum
from queue import Empty, Full, Queue
from typing import Any

from sinner2.io.target_reader import TargetReader
from sinner2.observable import ObservableValue
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.metrics import BufferMetrics
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.messages import (
    Message,
    PauseMsg,
    PlayMsg,
    SeekMsg,
    SetChainMsg,
    SetParamsMsg,
    SetSkipStrategyMsg,
    StopMsg,
)
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
_PLAYBACK_TICK_S = 1.0 / 30
_FPS_WINDOW = 10


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
        target_reader: TargetReader,
        buffer: FrameBuffer,
        timeline: Timeline,
        chain: list[Processor],
        strategy: FrameSkipStrategy,
        worker_count: int = 1,
    ) -> None:
        if worker_count < 1:
            raise ValueError(f"worker_count must be >= 1; got {worker_count}")
        self._target_reader = target_reader
        self._buffer = buffer
        self._timeline = timeline
        self._strategy = strategy
        self._worker_count = worker_count

        self._state_lock = threading.RLock()
        self._chain: tuple[Processor, ...] = tuple(chain)
        self._state: _State = _State.STOPPED
        self._last_submitted: FrameIndex = -1
        self._last_completed: FrameIndex = -1

        self._command_queue: Queue[Message] = Queue()
        self._work_queue: Queue[WorkItem | None] = Queue(maxsize=worker_count * 2)
        self._stop_event = threading.Event()
        self._dispatcher_thread: threading.Thread | None = None
        self._worker_threads: list[threading.Thread] = []
        self._playback_thread: threading.Thread | None = None

        self.current_frame: ObservableValue[FrameIndex] = ObservableValue(0)
        self.is_playing: ObservableValue[bool] = ObservableValue(False)
        self.processing_fps: ObservableValue[float] = ObservableValue(0.0)
        self.metrics: ObservableValue[BufferMetrics] = ObservableValue(buffer.metrics())
        self.status: ObservableValue[str] = ObservableValue("")

        self._on_frame: Callable[[Frame, FrameIndex], None] | None = None

        self._fps_lock = threading.RLock()
        self._fps_window: list[float] = []

    # ---- Lifecycle ----

    def start(self) -> None:
        with self._state_lock:
            if self._state is not _State.STOPPED:
                return
            self._stop_event.clear()
            self._state = _State.IDLE
            for p in self._chain:
                p.setup()
            self._dispatcher_thread = threading.Thread(
                target=self._dispatcher_loop, name="sinner2-dispatcher", daemon=True
            )
            self._dispatcher_thread.start()
            for i in range(self._worker_count):
                t = threading.Thread(
                    target=self._worker_loop, name=f"sinner2-worker-{i}", daemon=True
                )
                t.start()
                self._worker_threads.append(t)
            self._playback_thread = threading.Thread(
                target=self._playback_loop, name="sinner2-playback", daemon=True
            )
            self._playback_thread.start()

    def stop(self) -> None:
        with self._state_lock:
            if self._state is _State.STOPPED:
                return
            self._stop_event.set()
            for _ in range(self._worker_count):
                self._work_queue.put(_WORKER_SENTINEL)

        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=2.0)
        for t in self._worker_threads:
            t.join(timeout=2.0)
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)

        with self._state_lock:
            for p in self._chain:
                p.release()
            self._target_reader.release()
            self._state = _State.STOPPED
            self._dispatcher_thread = None
            self._worker_threads = []
            self._playback_thread = None

    # ---- Commands (non-blocking; post messages) ----

    def play(self) -> None:
        self._command_queue.put(PlayMsg())

    def pause(self) -> None:
        self._command_queue.put(PauseMsg())

    def seek(self, frame: FrameIndex) -> None:
        self._command_queue.put(SeekMsg(target_frame=frame))

    def set_params(self, processor_name: str, params: Mapping[str, Any]) -> None:
        self._command_queue.put(SetParamsMsg(processor_name=processor_name, params=params))

    def set_chain(self, chain: list[Processor]) -> None:
        self._command_queue.put(SetChainMsg(chain=tuple(chain)))

    def set_skip_strategy(self, strategy: FrameSkipStrategy) -> None:
        self._command_queue.put(SetSkipStrategyMsg(strategy=strategy))

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
                if self._state is _State.PLAYING:
                    self._try_submit_next_frame_locked()

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
            case SetChainMsg(chain=chain):
                self._handle_set_chain(chain)
            case SetSkipStrategyMsg(strategy=strategy):
                self._handle_set_strategy(strategy)
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

    def _handle_pause(self) -> None:
        with self._state_lock:
            self._timeline.pause()
            self._state = _State.PAUSED
        self.is_playing.set(False)

    def _handle_seek(self, target: FrameIndex) -> None:
        self._drain_work_queue()
        with self._state_lock:
            self._timeline.seek(target)
            self._last_submitted = target - 1
            if self._last_completed < target - 1:
                self._last_completed = target - 1

    def _handle_set_chain(self, chain: tuple[Processor, ...]) -> None:
        self._drain_work_queue()
        with self._state_lock:
            old_chain = self._chain
            self._chain = chain
            for p in chain:
                if p not in old_chain:
                    p.setup()
            for p in old_chain:
                if p not in chain:
                    p.release()

    def _handle_set_strategy(self, strategy: FrameSkipStrategy) -> None:
        self._drain_work_queue()
        with self._state_lock:
            self._strategy = strategy

    def _drain_work_queue(self) -> None:
        while True:
            try:
                self._work_queue.get_nowait()
            except Empty:
                break

    def _try_submit_next_frame_locked(self) -> None:
        metrics = self._buffer.metrics()
        decision = self._strategy.decide(
            last_submitted=self._last_submitted,
            last_completed=self._last_completed,
            timeline=self._timeline,
            metrics=metrics,
        )
        if decision.next_frame is None:
            return

        frame_index = decision.next_frame
        if frame_index >= self._target_reader.frame_count:
            self._timeline.pause()
            self._state = _State.PAUSED
            self.is_playing.set(False)
            self.status.set("end of target")
            return

        source_frame = self._target_reader.read(frame_index)
        if source_frame is None:
            self.status.set(f"target.read({frame_index}) returned None")
            return

        item = WorkItem(
            frame_index=frame_index,
            source_frame=source_frame,
            chain_snapshot=self._chain,
        )
        try:
            self._work_queue.put(item, timeout=0.1)
            self._last_submitted = frame_index
        except Full:
            pass

    # ---- Workers ----

    def _worker_loop(self) -> None:
        while True:
            item = self._work_queue.get()
            if item is _WORKER_SENTINEL:
                break
            try:
                start = time.monotonic()
                result = self._apply_chain(item.source_frame, item.chain_snapshot)
                elapsed = time.monotonic() - start
                self._buffer.put(item.frame_index, result)
                with self._state_lock:
                    if self._last_completed < item.frame_index:
                        self._last_completed = item.frame_index
                self._record_processing_time(elapsed)
            except Exception as e:
                self.status.set(f"worker error: {e}")
                self._stop_event.set()
                break

    @staticmethod
    def _apply_chain(frame: Frame, chain: tuple[Processor, ...]) -> Frame:
        for p in chain:
            frame = p.process(frame)
        return frame

    def _record_processing_time(self, elapsed_s: float) -> None:
        with self._fps_lock:
            self._fps_window.append(elapsed_s)
            if len(self._fps_window) > _FPS_WINDOW:
                self._fps_window.pop(0)
            avg = sum(self._fps_window) / len(self._fps_window)
            fps = 1.0 / avg if avg > 0 else 0.0
        self.processing_fps.set(fps)

    # ---- Playback ----

    def _playback_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                index, frame = self._buffer.get_at_current_time()
                if frame is not None and self._on_frame is not None:
                    try:
                        self._on_frame(frame, index)
                    except Exception as e:
                        self.status.set(f"on_frame callback error: {e}")
                self.current_frame.set(index)
                self.metrics.set(self._buffer.metrics())
            except Exception as e:
                self.status.set(f"playback error: {e}")
            time.sleep(_PLAYBACK_TICK_S)
