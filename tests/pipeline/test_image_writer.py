from pathlib import Path

import cv2
import numpy as np
import pytest

from sinner2.pipeline.image_writer import (
    ImageFormat,
    ImageWriter,
    JPEGImageWriter,
    PNGImageWriter,
    build_image_writer,
)


def _frame() -> np.ndarray:
    return np.full((8, 8, 3), 128, dtype=np.uint8)


class TestPNGImageWriter:
    def test_writes_png(self, tmp_path: Path):
        w = PNGImageWriter()
        path = tmp_path / "x.png"
        w.write(path, _frame())
        assert path.is_file()
        # Round-trip via cv2 to verify it's a valid PNG.
        back = cv2.imread(str(path))
        assert back is not None
        assert back.shape == (8, 8, 3)

    def test_extension_is_png(self):
        assert PNGImageWriter().extension == "png"

    def test_cache_key_includes_compression(self):
        assert PNGImageWriter(compression=3).cache_key == "png-c3"
        assert PNGImageWriter(compression=9).cache_key == "png-c9"

    def test_rejects_invalid_compression(self):
        with pytest.raises(ValueError):
            PNGImageWriter(compression=-1)
        with pytest.raises(ValueError):
            PNGImageWriter(compression=10)

    def test_read_returns_none_for_missing(self, tmp_path: Path):
        assert PNGImageWriter().read(tmp_path / "missing.png") is None


class TestJPEGImageWriter:
    def test_writes_jpeg(self, tmp_path: Path):
        w = JPEGImageWriter()
        path = tmp_path / "x.jpg"
        w.write(path, _frame())
        assert path.is_file()
        back = cv2.imread(str(path))
        assert back is not None
        assert back.shape == (8, 8, 3)

    def test_extension_is_jpg(self):
        assert JPEGImageWriter().extension == "jpg"

    def test_cache_key_includes_quality(self):
        assert JPEGImageWriter(quality=85).cache_key == "jpg-q85"
        assert JPEGImageWriter(quality=100).cache_key == "jpg-q100"

    def test_rejects_invalid_quality(self):
        with pytest.raises(ValueError):
            JPEGImageWriter(quality=0)
        with pytest.raises(ValueError):
            JPEGImageWriter(quality=101)

    def test_read_returns_none_for_missing(self, tmp_path: Path):
        assert JPEGImageWriter().read(tmp_path / "missing.jpg") is None


class TestProtocolConformance:
    def test_png_writer_conforms(self):
        assert isinstance(PNGImageWriter(), ImageWriter)

    def test_jpeg_writer_conforms(self):
        assert isinstance(JPEGImageWriter(), ImageWriter)


class TestBuildImageWriter:
    def test_builds_png(self):
        w = build_image_writer(ImageFormat.PNG, quality=5)
        assert isinstance(w, PNGImageWriter)
        assert w.compression == 5

    def test_builds_jpeg(self):
        w = build_image_writer(ImageFormat.JPEG, quality=85)
        assert isinstance(w, JPEGImageWriter)
        assert w.quality == 85
