from typing import Any

from PySide6.QtCore import QObject, Signal


class ThreadSafeCallback(QObject):
    """A callable that emits a Qt signal when invoked.

    Pass instances of this to non-Qt code (like the RealtimeExecutor) that
    expects a plain callable for status / event reporting. Invocation from
    any thread is safe — emit is queued, the slot fires on the receiver's
    thread (typically the GUI main thread).
    """

    fired = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def __call__(self, value: Any) -> None:
        self.fired.emit(value)
