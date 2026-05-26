import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class ObservableValue(Generic[T]):
    """Thread-safe single-value pub/sub.

    Subscribers are invoked synchronously on whatever thread calls set().
    For Qt GUI integration, wrap with a bridge that re-emits via a queued
    signal so the slot runs on the main thread.

    Equality-based suppression: setting to a value equal to the current one
    does not notify subscribers. This is the property that matters for UI —
    we don't want to repaint at the source's tick rate when the value is
    actually unchanged.
    """

    def __init__(self, initial: T) -> None:
        self._value = initial
        self._subscribers: list[Callable[[T], None]] = []
        self._lock = threading.RLock()

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T) -> None:
        with self._lock:
            if value == self._value:
                return
            self._value = value
            snapshot = list(self._subscribers)
        for sub in snapshot:
            sub(value)

    def subscribe(self, callback: Callable[[T], None]) -> Callable[[], None]:
        """Register callback. Returns an unsubscribe closure."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe
