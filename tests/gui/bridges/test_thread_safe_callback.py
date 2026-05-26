import threading

from sinner2.gui.bridges.thread_safe_callback import ThreadSafeCallback


class TestThreadSafeCallback:
    def test_fires_signal_when_called(self, qtbot):
        cb = ThreadSafeCallback()
        with qtbot.waitSignal(cb.fired, timeout=1000) as blocker:
            cb("hello")
        assert blocker.args == ["hello"]

    def test_arbitrary_payload(self, qtbot):
        cb = ThreadSafeCallback()
        payload = {"status": "ok", "count": 3}
        with qtbot.waitSignal(cb.fired, timeout=1000) as blocker:
            cb(payload)
        assert blocker.args == [payload]

    def test_call_from_worker_thread(self, qtbot):
        cb = ThreadSafeCallback()
        with qtbot.waitSignal(cb.fired, timeout=1000) as blocker:
            t = threading.Thread(target=cb, args=("from worker",))
            t.start()
            t.join()
        assert blocker.args == ["from worker"]
