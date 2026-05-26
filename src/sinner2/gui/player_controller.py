import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from sinner2.config.source import Source
from sinner2.config.target import Target
from sinner2.gui.bridges.observable_bridge import ObservableValueBridge
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.io.target_reader import ImageTargetReader, TargetReader
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.cache import MemoryFrameCache
from sinner2.pipeline.buffer.store import DiskFrameStore
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.processors.face_swapper import FaceSwapper
from sinner2.pipeline.realtime.executor import RealtimeExecutor
from sinner2.pipeline.skip_strategy import BestEffortStrategy

SessionFactory = Callable[[Source, Target, Path], tuple[RealtimeExecutor, ThreadPoolExecutor]]


def _default_session_factory(
    source: Source, target: Target, scratch_dir: Path
) -> tuple[RealtimeExecutor, ThreadPoolExecutor]:
    """Build a realtime session from source/target. Owns all the wiring.

    Caller takes ownership of (executor, write_executor) and is responsible
    for stop()+shutdown() in that order. The scratch_dir is the disk store
    location — caller owns its lifecycle too.
    """
    reader: TargetReader = ImageTargetReader(target)
    chain: list[Processor] = [FaceSwapper(source=source)]
    timeline = Timeline(fps=reader.fps)
    frames_dir = scratch_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    store = DiskFrameStore(frames_dir)
    cache = MemoryFrameCache(max_bytes=128 * 1024 * 1024)
    write_executor = ThreadPoolExecutor(max_workers=2)
    buffer = FrameBuffer(store, cache, timeline, write_executor)
    executor = RealtimeExecutor(
        target_reader=reader,
        buffer=buffer,
        timeline=timeline,
        chain=chain,
        strategy=BestEffortStrategy(),
    )
    return executor, write_executor


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
        self._bridges: list[ObservableValueBridge] = []
        self._scratch_dir = Path(tempfile.mkdtemp(prefix="sinner2-gui-"))

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
            executor, write_executor = self._session_factory(source, target, self._scratch_dir)
        except Exception as exc:
            self.errorOccurred.emit(f"session setup failed: {exc}")
            return

        executor.on_frame_ready(self._display.show_frame)
        self._bind_observables(executor)
        self._executor = executor
        self._write_executor = write_executor
        self._transport.set_frame_count(executor._target_reader.frame_count)  # noqa: SLF001

        try:
            executor.start()
        except Exception as exc:
            self.errorOccurred.emit(f"executor.start failed: {exc}")
            self._teardown_session()

    def shutdown(self) -> None:
        self._teardown_session()
        if self._scratch_dir.exists():
            shutil.rmtree(self._scratch_dir, ignore_errors=True)

    def executor(self) -> RealtimeExecutor | None:
        return self._executor

    def _bind_observables(self, executor: RealtimeExecutor) -> None:
        current_bridge = ObservableValueBridge(executor.current_frame, self)
        current_bridge.valueChanged.connect(self._transport.set_current_frame)
        playing_bridge = ObservableValueBridge(executor.is_playing, self)
        playing_bridge.valueChanged.connect(self._transport.set_is_playing)
        status_bridge = ObservableValueBridge(executor.status, self)
        status_bridge.valueChanged.connect(self._on_status)
        self._bridges = [current_bridge, playing_bridge, status_bridge]

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

    def _on_play(self) -> None:
        if self._executor is not None:
            self._executor.play()

    def _on_pause(self) -> None:
        if self._executor is not None:
            self._executor.pause()

    def _on_seek(self, frame: int) -> None:
        if self._executor is not None:
            self._executor.seek(frame)
