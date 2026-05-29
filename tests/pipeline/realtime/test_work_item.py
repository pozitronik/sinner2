from concurrent.futures import Future
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from sinner2.pipeline.realtime.work_item import WorkItem


def _resolved_future(frame=None):
    f: Future = Future()
    if frame is None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
    f.set_result(frame)
    return f


class TestWorkItem:
    def test_construction(self):
        fut = _resolved_future()
        item = WorkItem(frame_index=5, source_future=fut)
        assert item.frame_index == 5
        assert item.source_future is fut
        # Source is reachable via the future once resolved.
        assert item.source_future.result().shape == (10, 10, 3)

    def test_is_frozen(self):
        item = WorkItem(frame_index=0, source_future=_resolved_future())
        with pytest.raises(FrozenInstanceError):
            item.frame_index = 1  # type: ignore[misc]
