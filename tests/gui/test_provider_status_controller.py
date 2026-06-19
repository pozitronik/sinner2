"""Unit test for ProviderStatusController.stop() (the closeEvent teardown hook).

The failed-provider highlight runs on a QTimer poll; stop() must cancel it so a
queued tick can't fire highlight_failed() against torn-down collaborators during
shutdown. The rest of the controller is exercised via test_main_window's
TensorRT + provider-highlight classes.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from sinner2.gui.provider_status_controller import ProviderStatusController


def test_stop_cancels_the_highlight_poll_timer():
    ctl = ProviderStatusController.__new__(ProviderStatusController)
    timer = MagicMock()
    ctl._highlight_timer = timer
    ctl.stop()
    timer.stop.assert_called_once()
    assert ctl._highlight_timer is None


def test_stop_is_a_noop_when_no_timer_running():
    ctl = ProviderStatusController.__new__(ProviderStatusController)
    ctl._highlight_timer = None
    ctl.stop()  # must not raise
    assert ctl._highlight_timer is None
