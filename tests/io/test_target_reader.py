from pathlib import Path

import cv2
import numpy as np
import pytest

from sinner2.config.target import Target
from sinner2.io.target_reader import ImageTargetReader, TargetReader


@pytest.fixture
def image_target(tmp_path: Path) -> Target:
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.full((20, 20, 3), 100, dtype=np.uint8))
    return Target(path=p)


class TestImageTargetReader:
    def test_compliant_with_protocol(self, image_target: Target):
        assert isinstance(ImageTargetReader(image_target), TargetReader)

    def test_fps_is_one(self, image_target: Target):
        assert ImageTargetReader(image_target).fps == 1.0

    def test_frame_count_is_one(self, image_target: Target):
        assert ImageTargetReader(image_target).frame_count == 1

    def test_read_zero_returns_frame(self, image_target: Target):
        r = ImageTargetReader(image_target)
        f = r.read(0)
        assert f is not None
        assert f.shape == (20, 20, 3)

    def test_read_non_zero_returns_none(self, image_target: Target):
        r = ImageTargetReader(image_target)
        assert r.read(1) is None
        assert r.read(100) is None

    def test_read_caches_decoded_frame(self, image_target: Target):
        r = ImageTargetReader(image_target)
        f1 = r.read(0)
        f2 = r.read(0)
        assert f1 is f2

    def test_release_clears_cache(self, image_target: Target):
        r = ImageTargetReader(image_target)
        first = r.read(0)
        r.release()
        second = r.read(0)
        assert first is not second  # re-decoded

    def test_unreadable_image_raises(self, tmp_path: Path):
        p = tmp_path / "bad.png"
        p.write_bytes(b"not actually an image")
        r = ImageTargetReader(Target(path=p))
        with pytest.raises(OSError, match="cannot read image"):
            r.read(0)
