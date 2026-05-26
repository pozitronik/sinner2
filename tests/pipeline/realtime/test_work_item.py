from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from sinner2.pipeline.realtime.work_item import WorkItem


class _StubProcessor:
    name = "Stub"

    def setup(self) -> None: ...
    def process(self, frame):  # type: ignore[no-untyped-def]
        return frame
    def release(self) -> None: ...


class TestWorkItem:
    def test_construction(self):
        f = np.zeros((10, 10, 3), dtype=np.uint8)
        item = WorkItem(frame_index=5, source_frame=f, chain_snapshot=(_StubProcessor(),))
        assert item.frame_index == 5
        assert item.source_frame is f
        assert len(item.chain_snapshot) == 1

    def test_is_frozen(self):
        f = np.zeros((10, 10, 3), dtype=np.uint8)
        item = WorkItem(frame_index=0, source_frame=f, chain_snapshot=())
        with pytest.raises(FrozenInstanceError):
            item.frame_index = 1  # type: ignore[misc]

    def test_chain_snapshot_is_tuple(self):
        f = np.zeros((10, 10, 3), dtype=np.uint8)
        item = WorkItem(frame_index=0, source_frame=f, chain_snapshot=(_StubProcessor(),))
        assert isinstance(item.chain_snapshot, tuple)
