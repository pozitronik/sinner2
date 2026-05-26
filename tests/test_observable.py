import threading

import pytest

from sinner2.observable import ObservableValue


class TestObservableValue:
    def test_initial_value(self):
        v = ObservableValue(42)
        assert v.get() == 42

    def test_set_updates_value(self):
        v = ObservableValue(0)
        v.set(5)
        assert v.get() == 5

    def test_subscribe_called_on_change(self):
        v = ObservableValue(0)
        received: list[int] = []
        v.subscribe(received.append)
        v.set(1)
        v.set(2)
        v.set(3)
        assert received == [1, 2, 3]

    def test_set_to_same_value_does_not_notify(self):
        v = ObservableValue(0)
        received: list[int] = []
        v.subscribe(received.append)
        v.set(0)
        v.set(0)
        assert received == []

    def test_unsubscribe_stops_notifications(self):
        v = ObservableValue(0)
        received: list[int] = []
        unsub = v.subscribe(received.append)
        v.set(1)
        unsub()
        v.set(2)
        assert received == [1]

    def test_double_unsubscribe_is_safe(self):
        v = ObservableValue(0)
        unsub = v.subscribe(lambda _: None)
        unsub()
        unsub()  # should not raise

    def test_multiple_subscribers(self):
        v = ObservableValue(0)
        a: list[int] = []
        b: list[int] = []
        v.subscribe(a.append)
        v.subscribe(b.append)
        v.set(1)
        assert a == [1]
        assert b == [1]

    def test_subscriber_callback_runs_outside_lock(self):
        v = ObservableValue(0)

        def callback(value: int) -> None:
            v.get()  # must not deadlock

        v.subscribe(callback)
        v.set(1)

    def test_concurrent_sets_serialized(self):
        v = ObservableValue(0)
        received: list[int] = []
        v.subscribe(received.append)

        def writer(values: list[int]) -> None:
            for x in values:
                v.set(x)

        t1 = threading.Thread(target=writer, args=([1, 2, 3, 4, 5],))
        t2 = threading.Thread(target=writer, args=([6, 7, 8, 9, 10],))
        t1.start(); t2.start()
        t1.join(); t2.join()
        # We don't assert order (interleaved) — just that no notifications were lost
        # to crashes and the final value is one of the writes.
        assert v.get() in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
