"""Tests for the Unicode-safe cv2 wrappers.

cv2.imread / cv2.imwrite go through the OS multibyte codepage on
Windows and break for any non-Latin character in the path. The
imread_unicode / imwrite_unicode wrappers avoid that by reading and
writing the raw bytes in Python and using cv2.imdecode / imencode.

Tested explicitly with a Cyrillic filename — that's the exact path
shape the user reported as broken.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sinner2.io.cv2_unicode import imread_unicode, imwrite_unicode


def _frame() -> np.ndarray:
    return np.full((4, 6, 3), 200, dtype=np.uint8)


class TestRoundtripAscii:
    def test_write_then_read(self, tmp_path: Path):
        p = tmp_path / "ascii.png"
        assert imwrite_unicode(p, _frame())
        img = imread_unicode(p)
        assert img is not None
        assert img.shape == (4, 6, 3)


class TestRoundtripUnicode:
    @pytest.mark.parametrize(
        "name",
        [
            "Буланова.jpg",  # Cyrillic — the user's exact failure case
            "日本語.png",  # CJK
            "café.png",  # Latin-1 accented
            "🙂.jpg",  # emoji — passes Path on all OSes
        ],
    )
    def test_round_trip_non_latin_name(self, tmp_path: Path, name: str):
        p = tmp_path / name
        assert imwrite_unicode(p, _frame()), f"write failed for {name}"
        img = imread_unicode(p)
        assert img is not None, f"read returned None for {name}"
        assert img.shape == (4, 6, 3)


class TestReadFailures:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert imread_unicode(tmp_path / "nope.jpg") is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        p = tmp_path / "empty.jpg"
        p.write_bytes(b"")
        assert imread_unicode(p) is None

    def test_garbage_file_returns_none(self, tmp_path: Path):
        p = tmp_path / "garbage.jpg"
        p.write_bytes(b"not a real image at all")
        assert imread_unicode(p) is None


class TestWriteFailures:
    def test_no_extension_returns_false(self, tmp_path: Path):
        # cv2.imencode needs a leading-dot extension to pick a codec.
        p = tmp_path / "no-ext"
        assert imwrite_unicode(p, _frame()) is False

    def test_writes_to_nonexistent_dir_returns_false(self, tmp_path: Path):
        p = tmp_path / "does-not-exist" / "x.png"
        assert imwrite_unicode(p, _frame()) is False
