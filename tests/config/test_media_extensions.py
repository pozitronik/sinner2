"""Tests for the configurable media-extension detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.config import media_extensions as me


@pytest.fixture(autouse=True)
def reset_extensions():
    me.configure()  # defaults before each test
    yield
    me.configure()  # and restore after, so other test files see defaults


class TestDefaults:
    def test_wmv_is_video(self):
        assert me.is_video_ext(Path("clip.wmv")) is True
        assert me.is_media_ext(Path("clip.wmv")) is True

    def test_case_insensitive(self):
        assert me.is_video_ext(Path("CLIP.WMV")) is True
        assert me.is_image_ext(Path("PHOTO.JPG")) is True

    def test_common_image_and_video(self):
        assert me.is_image_ext(Path("a.png")) is True
        assert me.is_video_ext(Path("a.mp4")) is True

    def test_non_media_rejected(self):
        assert me.is_media_ext(Path("notes.txt")) is False
        assert me.is_image_ext(Path("archive.zip")) is False


class TestConfigure:
    def test_override_replaces_sets(self):
        me.configure(image_exts=["foo"], video_exts=["bar"])
        assert me.is_image_ext(Path("x.foo")) is True
        assert me.is_video_ext(Path("x.bar")) is True
        # The defaults are replaced, not merged.
        assert me.is_video_ext(Path("x.wmv")) is False

    def test_normalizes_case_and_dot(self):
        me.configure(image_exts=[".FOO", "Bar"])
        assert me.is_image_ext(Path("x.foo")) is True
        assert me.is_image_ext(Path("x.bar")) is True

    def test_none_resets_to_defaults(self):
        me.configure(image_exts=["foo"])
        me.configure()  # reset
        assert me.is_image_ext(Path("x.foo")) is False
        assert me.is_image_ext(Path("x.png")) is True


class TestFilters:
    def test_media_filter_includes_video_and_image(self):
        f = me.media_filter()
        assert "*.wmv" in f and "*.png" in f

    def test_images_filter(self):
        assert "*.png" in me.images_filter()
