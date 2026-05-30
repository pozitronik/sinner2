import threading


from sinner2.gui.bridges.observable_bridge import ObservableValueBridge
from sinner2.observable import ObservableValue


class TestObservableValueBridge:
    def test_emits_signal_on_observable_set(self, qtbot):
        obs = ObservableValue(0)
        bridge = ObservableValueBridge(obs)
        with qtbot.waitSignal(bridge.valueChanged, timeout=1000) as blocker:
            obs.set(42)
        assert blocker.args == [42]

    def test_initial_value_does_not_emit(self, qtbot):
        obs = ObservableValue(0)
        bridge = ObservableValueBridge(obs)
        with qtbot.assertNotEmitted(bridge.valueChanged, wait=50):
            pass

    def test_push_updates_observable(self, qtbot):
        obs = ObservableValue(0)
        bridge = ObservableValueBridge(obs)
        bridge.push(10)
        assert obs.get() == 10

    def test_push_does_not_loop_back_to_signal(self, qtbot):
        obs = ObservableValue(0)
        bridge = ObservableValueBridge(obs)
        with qtbot.assertNotEmitted(bridge.valueChanged, wait=100):
            bridge.push(10)

    def test_external_change_still_emits_after_push(self, qtbot):
        obs = ObservableValue(0)
        bridge = ObservableValueBridge(obs)
        bridge.push(5)
        with qtbot.waitSignal(bridge.valueChanged, timeout=1000) as blocker:
            obs.set(7)
        assert blocker.args == [7]

    def test_shutdown_stops_forwarding(self, qtbot):
        obs = ObservableValue(0)
        bridge = ObservableValueBridge(obs)
        bridge.shutdown()
        with qtbot.assertNotEmitted(bridge.valueChanged, wait=100):
            obs.set(99)

    def test_emit_from_worker_thread_reaches_gui(self, qtbot):
        obs = ObservableValue(0)
        bridge = ObservableValueBridge(obs)

        def worker() -> None:
            obs.set(123)

        with qtbot.waitSignal(bridge.valueChanged, timeout=1000) as blocker:
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        assert blocker.args == [123]
