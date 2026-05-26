from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from sinner2.pipeline.realtime.work_item import WorkItem


class TestWorkItem:
    def test_construction(self):
        f = np.zeros((10, 10, 3), dtype=np.uint8)
        item = WorkItem(frame_index=5, source_frame=f)
        assert item.frame_index == 5
        assert item.source_frame is f

    def test_is_frozen(self):
        f = np.zeros((10, 10, 3), dtype=np.uint8)
        item = WorkItem(frame_index=0, source_frame=f)
        with pytest.raises(FrozenInstanceError):
            item.frame_index = 1  # type: ignore[misc]
