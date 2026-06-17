"""Tests for the shared build_chain (Stage 4): enable toggles compose the chain
in order, empty when all off, providers/detection-sink threaded to the swapper.
Real processors are stubbed (their construction loads models)."""
from __future__ import annotations

import pytest

import sinner2.pipeline.chain_builder as cb
from sinner2.config.source import Source
from sinner2.pipeline.chain_builder import build_chain
from sinner2.pipeline.processors.face_enhancer import FaceEnhancerParams
from sinner2.pipeline.processors.face_swapper import FaceSwapperParams
from sinner2.pipeline.processors.upscaler import UpscalerParams
from sinner2.pipeline.realtime.per_worker import PerWorkerProcessor


class _MarkSwap:
    def __init__(
        self, source, params, providers=None, detection_sink=None, face_map=None
    ):
        self.kind = "swap"
        self.providers = providers
        self.detection_sink = detection_sink
        self.face_map = face_map
        self.geometry = None

    def set_geometry(self, geometry):
        self.geometry = geometry


class _MarkEnh:
    name = "FaceEnhancer"

    def __init__(self, params, device="auto"):
        pass


class _MarkUp:
    name = "Upscaler"

    def __init__(self, params, device="auto"):
        pass


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(cb, "FaceSwapper", _MarkSwap)
    monkeypatch.setattr(cb, "FaceEnhancer", _MarkEnh)
    monkeypatch.setattr(cb, "Upscaler", _MarkUp)


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "s.jpg"
    p.write_bytes(b"")  # Source only validates existence
    return Source(path=p)


def _call(source, **over):
    kw = dict(
        swapper_enabled=True, swapper_params=FaceSwapperParams(),
        swapper_providers=("CPUExecutionProvider",), detection_sink=None,
        enhancer_enabled=True, enhancer_params=FaceEnhancerParams(),
        enhancer_device="auto",
        upscaler_enabled=False, upscaler_params=UpscalerParams(),
        upscaler_device="auto",
    )
    kw.update(over)
    return build_chain(source, **kw)


def _kinds(chain):
    out = []
    for p in chain:
        out.append(p.name if isinstance(p, PerWorkerProcessor) else p.kind)
    return out


def test_full_chain_in_order(source):
    chain = _call(source, upscaler_enabled=True)
    assert _kinds(chain) == ["swap", "FaceEnhancer", "Upscaler"]


def test_swapper_only(source):
    assert _kinds(_call(source, enhancer_enabled=False)) == ["swap"]


def test_enhancer_only(source):
    assert _kinds(_call(source, swapper_enabled=False)) == ["FaceEnhancer"]


def test_all_off_is_empty(source):
    chain = _call(source, swapper_enabled=False, enhancer_enabled=False)
    assert chain == []


def test_swapper_gets_providers_and_sink(source):
    sink = object()
    chain = _call(
        source, enhancer_enabled=False,
        swapper_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        detection_sink=sink,
    )
    swap = chain[0]
    assert swap.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert swap.detection_sink is sink


def test_swapper_gets_face_map(source):
    from sinner2.pipeline.face_map import FaceMap, Identity, normalize

    fm = FaceMap(identities=(Identity("a", normalize([1, 0]), source_path="/s.png"),))
    chain = _call(source, enhancer_enabled=False, face_map=fm)
    assert chain[0].face_map is fm


def test_no_face_map_is_none(source):
    chain = _call(source, enhancer_enabled=False)
    assert chain[0].face_map is None


def test_swapper_gets_geometry(source):
    from sinner2.pipeline.face_map_geometry import FrameGeometry

    geom = FrameGeometry(faces={}, frame_count=5, refined=True)
    chain = _call(source, enhancer_enabled=False, geometry=geom)
    assert chain[0].geometry is geom


def test_no_geometry_is_none(source):
    chain = _call(source, enhancer_enabled=False)
    assert chain[0].geometry is None
