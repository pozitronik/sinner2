from typing import Any

from PySide6.QtCore import QObject, Signal

from sinner2.observable import ObservableValue


class ObservableValueBridge(QObject):
    """Forwards ObservableValue changes onto a Qt signal.

    The bridge lives on the GUI thread. When the underlying observable's
    set() is called from a worker thread, the subscriber callback runs on
    that worker thread, then emits the Qt signal. Qt auto-detects the
    cross-thread emission and queues delivery so the slot fires on the
    GUI thread's event loop.

    Bidirectional: callers can call push() to set the observable from the
    GUI side. The `_updating` flag suppresses the re-emit so a slot calling
    push() doesn't loop back through valueChanged.
    """

    valueChanged = Signal(object)

    def __init__(self, observable: ObservableValue[Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._observable = observable
        self._updating = False
        self._unsubscribe = observable.subscribe(self._on_change)

    def _on_change(self, value: Any) -> None:
        if self._updating:
            return
        self.valueChanged.emit(value)

    def push(self, value: Any) -> None:
        """GUI → observable. Sets the observable without retriggering valueChanged."""
        self._updating = True
        try:
            self._observable.set(value)
        finally:
            self._updating = False

    def shutdown(self) -> None:
        self._unsubscribe()
