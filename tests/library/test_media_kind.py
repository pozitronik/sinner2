"""Tests for media_kind.detect_kind — the image/video/None classifier the
library scan uses to filter folder entries.

The thin is_image/is_video/is_media delegators are covered by
tests/config/test_media_extensions.py; here we pin the only real logic:
detect_kind's mapping, its None-for-non-media contract (the silent filter), and
its documented video-before-image precedence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.library import media_kind
from sinner2.library.media_kind import MediaKind, detect_kind


class TestDetectKind:
    @pytest.mark.parametrize("name", ["clip.mp4", "movie.mkv", "scene.mov", "old.wmv"])
    def test_video_extension_returns_video(self, name):
        assert detect_kind(Path(name)) is MediaKind.VIDEO

    @pytest.mark.parametrize("name", ["face.png", "shot.jpg", "pic.jpeg", "art.webp"])
    def test_image_extension_returns_image(self, name):
        assert detect_kind(Path(name)) is MediaKind.IMAGE

    @pytest.mark.parametrize("name", ["notes.txt", "tool.exe", "archive.zip", "noext", ".hidden"])
    def test_non_media_returns_none(self, name):
        """Non-media (and extension-less) paths are filtered out, not raised on."""
        assert detect_kind(Path(name)) is None

    @pytest.mark.parametrize(
        "name,expected",
        [("CLIP.MP4", MediaKind.VIDEO), ("Photo.JPG", MediaKind.IMAGE), ("X.WebP", MediaKind.IMAGE)],
    )
    def test_case_insensitive(self, name, expected):
        assert detect_kind(Path(name)) is expected

    def test_full_path_uses_suffix_not_directory(self, tmp_path):
        """Only the file suffix decides — directory names in the path don't."""
        assert detect_kind(tmp_path / "videos" / "frame.png") is MediaKind.IMAGE

    def test_video_takes_precedence_over_image(self, monkeypatch):
        """When an extension somehow matches both sets, video wins (the order the
        function checks). Pins the branch order independent of the real ext sets."""
        monkeypatch.setattr(media_kind, "is_video_ext", lambda p: True)
        monkeypatch.setattr(media_kind, "is_image_ext", lambda p: True)
        assert detect_kind(Path("ambiguous.xyz")) is MediaKind.VIDEO
