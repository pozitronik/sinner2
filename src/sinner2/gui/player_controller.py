from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from sinner2.config.source import Source
from sinner2.config.target import Target, TargetKind
from sinner2.gui.bridges.observable_bridge import ObservableValueBridge
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.io.video_target_reader import VideoTargetReader
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import SessionFrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_enhancer import FaceEnhancer, FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapper, FaceSwapperParams
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import BestEffortStrategy, FrameSkipStrategy

SessionFactory = Callable[
    [TargetReader, list[Processor], FrameSkipStrategy, int],
    tuple[RealtimeExecutor, ThreadPoolExecutor, SessionFrameStore],
]


def _make_reader(target: Target) -> TargetReader:
    if target.kind == TargetKind.IMAGE:
        return ImageTargetReader(target)
    if target.kind == TargetKind.VIDEO:
        return VideoTargetReader(target)
    raise ValueError(f"unsupported target kind: {target.kind}")


def _default_session_factory(
    reader: TargetReader,
    chain: list[Processor],
    strategy: FrameSkipStrategy,
    worker_count: int,
) -> tuple[RealtimeExecutor, ThreadPoolExecutor, SessionFrameStore]:
    """Build a realtime executor + the SessionFrameStore that owns its scratch.

    Caller takes ownership of all three returned objects and is responsible
    for stop() → shutdown(wait=True) → close() in that order.
    """
    timeline = Timeline(fps=reader.fps)
    store = SessionFrameStore(prefix="sinner2-gui-")
    cache = MemoryFrameCache(max_bytes=128 * 1024 * 1024)
    write_executor = ThreadPoolExecutor(max_workers=2)
    buffer = FrameBuffer(store, cache, timeline, write_executor)
    executor = RealtimeExecutor(
        target_reader=reader,
        buffer=buffer,
        timeline=timeline,
        chain=chain,
        strategy=strategy,
        worker_count=worker_count,
    )
    return executor, write_executor, store


class PlayerController(QObject):
    """Owns the realtime executor lifecycle and wires widgets to it.

    Responsibilities:
      - Build / tear down the executor when source+target are both set
      - Bridge executor observables to widget setter slots
      - Forward widget signals (play/pause/seek) to executor commands
      - Surface setup / runtime errors via the errorOccurred signal
      - Clean up scratch directory on shutdown
    """

    errorOccurred = Signal(str)
    processingFpsChanged = Signal(object)  # carries float; declared `object` to match the bridge
    sessionScratchDirChanged = Signal(object)  # Path | None — emitted on session start/end

    def __init__(
        self,
        frame_display: QFrameDisplayWidget,
        transport: QTransportControls,
        session_factory: SessionFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._display = frame_display
        self._transport = transport
        self._session_factory = session_factory or _default_session_factory

        self._executor: RealtimeExecutor | None = None
        self._write_executor: ThreadPoolExecutor | None = None
        self._session_store: SessionFrameStore | None = None
        self._bridges: list[ObservableValueBridge] = []

        self._current_source: Source | None = None
        self._swapper_params = FaceSwapperParams()
        self._enhancer_params = FaceEnhancerParams()
        self._enhancer_enabled = True
        self._strategy: FrameSkipStrategy = BestEffortStrategy()
        self._worker_count = 1

        transport.playRequested.connect(self._on_play)
        transport.pauseRequested.connect(self._on_pause)
        transport.seekRequested.connect(self._on_seek)

    def set_source_and_target(self, source_path: Path | None, target_path: Path | None) -> None:
        if source_path is None or target_path is None:
            return
        self._teardown_session()
        try:
            source = Source(path=source_path)
            target = Target(path=target_path)
            reader = _make_reader(target)
            chain = self._build_chain(source)
            executor, write_executor, session_store = self._session_factory(
                reader, chain, self._strategy, self._worker_count
            )
        except Exception as exc:
            self.errorOccurred.emit(f"session setup failed: {exc}")
            return

        executor.on_frame_ready(self._display.show_frame)
        self._bind_observables(executor)
        self._current_source = source
        self._executor = executor
        self._write_executor = write_executor
        self._session_store = session_store
        self._transport.set_frame_count(reader.frame_count)
        self.sessionScratchDirChanged.emit(session_store.scratch_dir)

        try:
            executor.start()
        except Exception as exc:
            self.errorOccurred.emit(f"executor.start failed: {exc}")
            self._teardown_session()

    def apply_session_config(
        self,
        swapper_params: FaceSwapperParams,
        enhancer_params: FaceEnhancerParams,
        enhancer_enabled: bool,
        strategy: FrameSkipStrategy,
        worker_count: int,
    ) -> None:
        """Update stored params and strategy; hot-swap on the live executor.

        worker_count is stored but only takes effect at the next session start
        — changing it mid-session would require tearing down and rebuilding
        the executor, which we avoid here. No-op for the executor parts if
        no session is running yet — the next session start picks up the
        stored values.
        """
        chain_changed = (
            swapper_params != self._swapper_params
            or enhancer_params != self._enhancer_params
            or enhancer_enabled != self._enhancer_enabled
        )
        strategy_changed = type(strategy) is not type(self._strategy)

        self._swapper_params = swapper_params
        self._enhancer_params = enhancer_params
        self._enhancer_enabled = enhancer_enabled
        self._strategy = strategy
        self._worker_count = worker_count

        if self._executor is None or self._current_source is None:
            return
        if chain_changed:
            try:
                new_chain = self._build_chain(self._current_source)
            except Exception as exc:
                self.errorOccurred.emit(f"chain rebuild failed: {exc}")
                return
            self._executor.set_chain(new_chain)
        if strategy_changed:
            self._executor.set_skip_strategy(strategy)

    def _build_chain(self, source: Source) -> list[Processor]:
        chain: list[Processor] = [FaceSwapper(source=source, params=self._swapper_params)]
        if self._enhancer_enabled:
            chain.append(FaceEnhancer(params=self._enhancer_params))
        return chain

    def shutdown(self) -> None:
        self._teardown_session()

    def executor(self) -> RealtimeExecutor | None:
        return self._executor

    def _bind_observables(self, executor: RealtimeExecutor) -> None:
        current_bridge = ObservableValueBridge(executor.current_frame, self)
        current_bridge.valueChanged.connect(self._transport.set_current_frame)
        playing_bridge = ObservableValueBridge(executor.is_playing, self)
        playing_bridge.valueChanged.connect(self._transport.set_is_playing)
        status_bridge = ObservableValueBridge(executor.status, self)
        status_bridge.valueChanged.connect(self._on_status)
        fps_bridge = ObservableValueBridge(executor.processing_fps, self)
        fps_bridge.valueChanged.connect(self.processingFpsChanged)
        self._bridges = [current_bridge, playing_bridge, status_bridge, fps_bridge]

    def _on_status(self, message: object) -> None:
        text = str(message)
        if text and text.lower().startswith(("worker error", "executor.start", "session setup")):
            self.errorOccurred.emit(text)

    def _teardown_session(self) -> None:
        for bridge in self._bridges:
            bridge.shutdown()
        self._bridges = []
        if self._executor is not None:
            self._executor.stop()
            self._executor = None
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)
            self._write_executor = None
        if self._session_store is not None:
            self._session_store.close()
            self._session_store = None
            self.sessionScratchDirChanged.emit(None)
        self._current_source = None

    def _on_play(self) -> None:
        if self._executor is not None:
            self._executor.play()

    def _on_pause(self) -> None:
        if self._executor is not None:
            self._executor.pause()

    def _on_seek(self, frame: int) -> None:
        if self._executor is not None:
            self._executor.seek(frame)
