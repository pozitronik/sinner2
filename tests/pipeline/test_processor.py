import numpy as np

from sinner2.pipeline.processor import ChainContext, Processor
from sinner2.types import Frame


class _Compliant:
    name = "Compliant"
    thread_safe = True

    def setup(self) -> None: ...
    def process(self, frame: Frame) -> Frame:
        return frame
    def release(self) -> None: ...


class _MissingSetup:
    name = "MissingSetup"

    def process(self, frame: Frame) -> Frame:
        return frame
    def release(self) -> None: ...


class TestProcessorProtocol:
    def test_compliant_class_is_runtime_processor(self):
        assert isinstance(_Compliant(), Processor)

    def test_missing_method_is_not_processor(self):
        assert not isinstance(_MissingSetup(), Processor)

    def test_chain_application(self):
        a = _Compliant()
        b = _Compliant()
        frame: Frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for p in [a, b]:
            frame = p.process(frame)
        assert frame.shape == (10, 10, 3)


class TestChainContext:
    def test_defaults_to_no_faces(self):
        # None = "no upstream detection ran" → consumers self-detect; an
        # empty LIST is a real no-faces result and must stay distinguishable.
        ctx = ChainContext()
        assert ctx.faces is None
        ctx.faces = []
        assert ctx.faces == []

    def test_plain_processors_do_not_accept_context(self):
        # accepts_context is opt-in — absent means the executor calls the
        # one-argument process(frame).
        assert getattr(_Compliant(), "accepts_context", False) is False
