"""Tests for the paused-view chain refresh.

When the user toggles a chain-affecting param (enhancer on/off, swapper
detection interval, etc.) while paused, the executor.set_chain() swap
puts the new chain in place but the displayed pixels are still the old
chain's output. apply_session_config compensates by issuing a seek to
the current frame, which causes the new chain to reprocess and the
display to update. When playing, the dispatcher naturally submits next
frames through the new chain, so no extra seek is needed."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sinner2.config.source import Source
from sinner2.gui.player_controller import PlayerController, _DEFAULT_CACHE_SETTINGS
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.pipeline.playback_mode import PlaybackMode
from sinner2.pipeline.processors.face_enhancer import FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams
from sinner2.pipeline.skip_strategy import BestEffortStrategy


@pytest.fixture
def widgets(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return display, transport


@pytest.fixture
def fake_source_path(tmp_path):
    """Source validates path existence; touch a real file so it accepts it."""
    p = tmp_path / "source.jpg"
    p.write_bytes(b"")
    return p


def _make_controller(widgets) -> PlayerController:
    display, transport = widgets
    return PlayerController(frame_display=display, transport=transport)


def _attach_fake_session(
    ctrl: PlayerController,
    *,
    playing: bool,
    current_frame: int,
    source_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    ctrl._current_source = Source(path=source_path)  # noqa: SLF001
    fake_executor = MagicMock()
    fake_executor.is_playing.get.return_value = playing
    fake_executor.current_frame.get.return_value = current_frame
    ctrl._executor = fake_executor  # noqa: SLF001
    # Bypass real chain construction — the test only cares that the
    # controller asks the executor for a set_chain + (conditionally) seek.
    monkeypatch.setattr(ctrl, "_build_chain", lambda src: [object()])
    return fake_executor


def _apply(ctrl: PlayerController, *, enhancer_enabled: bool) -> None:
    """Call apply_session_config with the given enhancer_enabled. Flip the
    value relative to the controller's stored state to force
    chain_changed=True; pass the current value to leave the chain alone."""
    ctrl.apply_session_config(
        swapper_params=FaceSwapperParams(),
        enhancer_params=FaceEnhancerParams(),
        enhancer_enabled=enhancer_enabled,
        strategy=BestEffortStrategy(),
        worker_count=1,
        playback_mode=PlaybackMode.FIXED_30,
        cache_settings=_DEFAULT_CACHE_SETTINGS,
    )


class TestPausedChainRefresh:
    def test_paused_chain_change_seeks_current_frame(
        self, widgets, fake_source_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=42,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        # Default enhancer_enabled is False; flipping forces chain_changed.
        # Controller default is enhancer_enabled=True; flip to False to
        # force chain_changed.
        _apply(ctrl, enhancer_enabled=False)
        fake.set_chain.assert_called_once()
        fake.seek.assert_called_once_with(42)
        ctrl.shutdown()

    def test_playing_chain_change_does_not_seek(
        self, widgets, fake_source_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=True,
            current_frame=42,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        _apply(ctrl, enhancer_enabled=False)
        fake.set_chain.assert_called_once()
        # Playing: dispatcher submits next frames through the new chain
        # on its own — no nudge needed.
        fake.seek.assert_not_called()
        ctrl.shutdown()

    def test_paused_no_chain_change_does_not_seek(
        self, widgets, fake_source_path, monkeypatch
    ):
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=42,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        # Apply the controller's current state — chain_changed=False.
        # No set_chain, no seek.
        _apply(ctrl, enhancer_enabled=ctrl._enhancer_enabled)  # noqa: SLF001
        fake.set_chain.assert_not_called()
        fake.seek.assert_not_called()
        ctrl.shutdown()

    def test_paused_chain_change_at_frame_zero_seeks_zero(
        self, widgets, fake_source_path, monkeypatch
    ):
        # The guard is "current >= 0", so frame 0 must still seek.
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=0,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        _apply(ctrl, enhancer_enabled=False)
        fake.seek.assert_called_once_with(0)
        ctrl.shutdown()

    def test_paused_chain_change_at_negative_frame_does_not_seek(
        self, widgets, fake_source_path, monkeypatch
    ):
        # current_frame=-1 means timeline hasn't produced any frame yet
        # (initial state). Seeking to -1 would be invalid.
        ctrl = _make_controller(widgets)
        fake = _attach_fake_session(
            ctrl,
            playing=False,
            current_frame=-1,
            source_path=fake_source_path,
            monkeypatch=monkeypatch,
        )
        _apply(ctrl, enhancer_enabled=False)
        fake.set_chain.assert_called_once()
        fake.seek.assert_not_called()
        ctrl.shutdown()
