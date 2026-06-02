"""Tests for the smoke-test CLI's setup-readiness handling.

Regression: the executor loads the chain ASYNCHRONOUSLY, so a model-load
failure never raised out of start() — the old try/except was dead code and a
setup failure span the full 300s timeout before printing a misleading message.
``_await_setup`` is the testable seam that turns the async failure into an
immediate error.
"""
from __future__ import annotations


class _Status:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _FakeExecutor:
    def __init__(self, *, ready: bool, status: str) -> None:
        self._ready = ready
        self.status = _Status(status)

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._ready


class TestAwaitSetup:
    def test_returns_none_on_successful_setup(self) -> None:
        from sinner2.cli.smoke import _await_setup

        ex = _FakeExecutor(ready=True, status="")
        assert _await_setup(ex, 5.0) is None

    def test_returns_message_on_chain_setup_failure(self) -> None:
        from sinner2.cli.smoke import _await_setup

        ex = _FakeExecutor(ready=True, status="chain setup failed: boom")
        err = _await_setup(ex, 5.0)
        assert err is not None
        assert "boom" in err

    def test_returns_message_on_setup_timeout(self) -> None:
        # Setup never completes within the timeout: must report instead of
        # spinning the full playback timeout afterward.
        from sinner2.cli.smoke import _await_setup

        ex = _FakeExecutor(ready=False, status="loading models…")
        err = _await_setup(ex, 0.5)
        assert err is not None
        assert "timeout" in err.lower()
