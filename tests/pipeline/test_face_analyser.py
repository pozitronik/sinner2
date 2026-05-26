from unittest.mock import MagicMock

import numpy as np
import pytest

from sinner2.pipeline import face_analyser
from sinner2.pipeline.face_analyser import FaceAnalyser, reset_shared_face_analysis
from sinner2.types import Frame


@pytest.fixture
def stub_insightface(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    stub.get = MagicMock(return_value=[MagicMock(name="Face")])
    monkeypatch.setattr(face_analyser, "_get_shared_face_analysis", lambda: stub)
    return stub


@pytest.fixture(autouse=True)
def _reset_shared() -> None:
    reset_shared_face_analysis()


def _blank_frame() -> Frame:
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestFaceAnalyser:
    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError):
            FaceAnalyser(detection_interval=0)

    def test_interval_1_detects_every_frame(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=1)
        for _ in range(5):
            a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 5

    def test_interval_3_detects_every_third_frame(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=3)
        for _ in range(9):
            a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 3

    def test_returned_faces_are_cached_between_detections(self, stub_insightface: MagicMock):
        face1 = MagicMock(name="face1")
        face2 = MagicMock(name="face2")
        stub_insightface.get.side_effect = [[face1], [face2]]

        a = FaceAnalyser(detection_interval=3)
        assert a.analyse(_blank_frame()) == [face1]
        assert a.analyse(_blank_frame()) == [face1]
        assert a.analyse(_blank_frame()) == [face1]
        assert a.analyse(_blank_frame()) == [face2]

    def test_analyse_uncached_bypasses_cache(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=10)
        for _ in range(3):
            a.analyse_uncached(_blank_frame())
        assert stub_insightface.get.call_count == 3

    def test_reset_cache_forces_next_detection(self, stub_insightface: MagicMock):
        a = FaceAnalyser(detection_interval=10)
        a.analyse(_blank_frame())
        a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 1

        a.reset_cache()
        a.analyse(_blank_frame())
        assert stub_insightface.get.call_count == 2

    def test_returned_list_is_a_copy(self, stub_insightface: MagicMock):
        face = MagicMock(name="face")
        stub_insightface.get.return_value = [face]

        a = FaceAnalyser(detection_interval=1)
        faces = a.analyse(_blank_frame())
        faces.append("extra")
        assert a.analyse(_blank_frame()) == [face]
