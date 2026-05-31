"""The realtime worker count is capped for a heavy GPU-bound enhancer
(CodeFormer): extra workers only deepen the in-flight queue (worse seek latency)
without adding throughput on the single shared ONNX session."""
from __future__ import annotations

import pytest

from sinner2.gui.player_controller import (
    _CODEFORMER_REALTIME_WORKER_CAP,
    PlayerController,
)
from sinner2.gui.widgets.frame_display import QFrameDisplayWidget
from sinner2.gui.widgets.transport_controls import QTransportControls
from sinner2.pipeline.processors.face_enhancer import (
    EnhancerModel,
    FaceEnhancerParams,
)


@pytest.fixture
def controller(qtbot) -> PlayerController:
    display = QFrameDisplayWidget()
    qtbot.addWidget(display)
    transport = QTransportControls()
    qtbot.addWidget(transport)
    return PlayerController(frame_display=display, transport=transport)


def test_codeformer_enhancer_caps_workers(controller):
    controller._worker_count = 8  # noqa: SLF001
    controller._enhancer_enabled = True  # noqa: SLF001
    controller._enhancer_params = FaceEnhancerParams(  # noqa: SLF001
        model=EnhancerModel.CODEFORMER
    )
    assert controller._effective_worker_count() == _CODEFORMER_REALTIME_WORKER_CAP  # noqa: SLF001
    controller.shutdown()


def test_gfpgan_enhancer_does_not_cap(controller):
    controller._worker_count = 8  # noqa: SLF001
    controller._enhancer_enabled = True  # noqa: SLF001
    controller._enhancer_params = FaceEnhancerParams(  # noqa: SLF001
        model=EnhancerModel.GFPGAN
    )
    assert controller._effective_worker_count() == 8  # noqa: SLF001
    controller.shutdown()


def test_disabled_enhancer_does_not_cap(controller):
    controller._worker_count = 8  # noqa: SLF001
    controller._enhancer_enabled = False  # noqa: SLF001 — CodeFormer selected but off
    controller._enhancer_params = FaceEnhancerParams(  # noqa: SLF001
        model=EnhancerModel.CODEFORMER
    )
    assert controller._effective_worker_count() == 8  # noqa: SLF001
    controller.shutdown()


def test_cap_does_not_raise_below_requested(controller):
    # Already under the cap → unchanged (no spurious bump).
    controller._worker_count = 1  # noqa: SLF001
    controller._enhancer_enabled = True  # noqa: SLF001
    controller._enhancer_params = FaceEnhancerParams(  # noqa: SLF001
        model=EnhancerModel.CODEFORMER
    )
    assert controller._effective_worker_count() == 1  # noqa: SLF001
    controller.shutdown()
