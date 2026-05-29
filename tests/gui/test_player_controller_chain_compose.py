"""The realtime chain honors the swapper/enhancer enable toggles, including
an empty chain (raw passthrough) when both are off."""
from __future__ import annotations

import pytest

from sinner2.config.source import Source
from sinner2.gui.player_controller import PlayerController
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.pipeline.realtime.per_worker import PerWorkerProcessor


class _MarkSwap:
    def __init__(self, source, params, providers=None) -> None:
        self.kind = "swap"
        self.providers = providers


class _MarkEnh:
    name = "FaceEnhancer"  # _build_chain reads FaceEnhancer.name for the wrapper

    def __init__(self, params, device="auto") -> None:
        self.kind = "enh"
        self.device = device


@pytest.fixture(autouse=True)
def _stub_processors(monkeypatch):
    import sinner2.gui.player_controller as pc

    monkeypatch.setattr(pc, "FaceSwapper", _MarkSwap)
    monkeypatch.setattr(pc, "FaceEnhancer", _MarkEnh)


@pytest.fixture
def controller(qtbot):
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return PlayerController(frame_display=display, transport=transport)


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "src.jpg"
    p.write_bytes(b"")  # Source only validates existence
    return Source(path=p)


def _kinds(chain):
    # The enhancer is wrapped in a PerWorkerProcessor (each realtime worker
    # gets its own GFPGAN), so it no longer exposes _MarkEnh.kind directly.
    return [
        "enh" if isinstance(p, PerWorkerProcessor) else p.kind for p in chain
    ]


class TestChainComposition:
    def test_both_enabled(self, controller, source):
        assert _kinds(controller._build_chain(source)) == ["swap", "enh"]  # noqa: SLF001

    def test_swapper_disabled_enhancer_only(self, controller, source):
        controller._swapper_enabled = False  # noqa: SLF001
        assert _kinds(controller._build_chain(source)) == ["enh"]  # noqa: SLF001

    def test_enhancer_disabled_swapper_only(self, controller, source):
        controller._enhancer_enabled = False  # noqa: SLF001
        assert _kinds(controller._build_chain(source)) == ["swap"]  # noqa: SLF001

    def test_both_disabled_is_empty_passthrough(self, controller, source):
        controller._swapper_enabled = False  # noqa: SLF001
        controller._enhancer_enabled = False  # noqa: SLF001
        assert controller._build_chain(source) == []  # noqa: SLF001
