import numpy as np

from sinner2.pipeline.processor import Processor
from sinner2.types import Frame


class _Compliant:
    name = "Compliant"

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
