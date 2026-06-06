"""A realtime chain-setup failure must be surfaced to the user.

The executor reports setup failure via the status string "chain setup failed:
…" (it can't raise — setup runs on a background thread). The controller's status
filter must route that to errorOccurred, else the failure is silently swallowed
(no dialog) while the executor sits wedged.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls


def test_chain_setup_failure_is_surfaced(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    ctrl = PlayerController(
        frame_display=display,
        transport=transport,
        audio_backend_factory=lambda name: MagicMock(),
    )
    errors: list[str] = []
    ctrl.errorOccurred.connect(lambda m: errors.append(m))
    ctrl._on_status("chain setup failed: boom")  # noqa: SLF001
    assert errors and "chain setup failed" in errors[0]
