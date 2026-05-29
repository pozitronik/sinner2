import threading
import time

import pytest

from sinner2.pipeline.buffer.bounded_write_executor import BoundedWriteExecutor


class TestSubmitAcceptsBelowCap:
    def test_returns_true_under_cap(self):
        ex = BoundedWriteExecutor(max_workers=2, max_outstanding=4)
        try:
            assert ex.submit(lambda: None) is True
            assert ex.submit(lambda: None) is True
        finally:
            ex.shutdown(wait=True)

    def test_completed_tasks_free_slots(self):
        # After tasks finish, submit should succeed again even after the
        # outstanding count was at the cap — the slot is reclaimed.
        ex = BoundedWriteExecutor(max_workers=4, max_outstanding=2)
        try:
            assert ex.submit(lambda: None)
            assert ex.submit(lambda: None)
            # Wait for completion
            for _ in range(50):
                if ex.metrics_snapshot().completed >= 2:
                    break
                time.sleep(0.01)
            assert ex.submit(lambda: None)
        finally:
            ex.shutdown(wait=True)


class TestSubmitDropsAtCap:
    def test_returns_false_at_cap(self):
        # Block all running tasks so the outstanding count stays maxed.
        gate = threading.Event()
        ex = BoundedWriteExecutor(max_workers=2, max_outstanding=2)
        try:
            assert ex.submit(gate.wait)
            assert ex.submit(gate.wait)
            # Next submit hits the cap before any task can finish.
            assert ex.submit(lambda: None) is False
            assert ex.metrics_snapshot().dropped == 1
        finally:
            gate.set()
            ex.shutdown(wait=True)

    def test_drops_accumulate(self):
        gate = threading.Event()
        ex = BoundedWriteExecutor(max_workers=1, max_outstanding=1)
        try:
            assert ex.submit(gate.wait)
            for _ in range(5):
                ex.submit(lambda: None)
            assert ex.metrics_snapshot().dropped == 5
        finally:
            gate.set()
            ex.shutdown(wait=True)


class TestMetricsSnapshot:
    def test_initial_state(self):
        ex = BoundedWriteExecutor(max_workers=2, max_outstanding=4)
        try:
            m = ex.metrics_snapshot()
            assert m.outstanding == 0
            assert m.max_outstanding == 4
            assert m.submitted == 0
            assert m.completed == 0
            assert m.dropped == 0
            assert m.latency_p50_ms == 0.0
            assert m.latency_p95_ms == 0.0
        finally:
            ex.shutdown(wait=True)

    def test_records_latency_after_completion(self):
        # max_outstanding raised above the burst so no drops bias the count.
        ex = BoundedWriteExecutor(max_workers=2, max_outstanding=16)
        try:
            for _ in range(5):
                ex.submit(lambda: time.sleep(0.01))
            for _ in range(200):
                if ex.metrics_snapshot().completed >= 5:
                    break
                time.sleep(0.01)
            m = ex.metrics_snapshot()
            assert m.completed == 5
            assert m.dropped == 0
            # 10 ms sleep -> p50 should be >0; some overhead expected.
            assert m.latency_p50_ms > 5.0
        finally:
            ex.shutdown(wait=True)


class TestValidation:
    def test_rejects_zero_workers(self):
        with pytest.raises(ValueError):
            BoundedWriteExecutor(max_workers=0, max_outstanding=4)

    def test_rejects_zero_queue(self):
        with pytest.raises(ValueError):
            BoundedWriteExecutor(max_workers=2, max_outstanding=0)


class TestShutdown:
    def test_submit_after_shutdown_returns_false(self):
        ex = BoundedWriteExecutor(max_workers=2, max_outstanding=4)
        ex.shutdown(wait=True)
        assert ex.submit(lambda: None) is False
        assert ex.metrics_snapshot().dropped == 1
