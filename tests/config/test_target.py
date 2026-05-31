from pathlib import Path

import pytest

from sinner2.config.target import Target, TargetKind


class TestTarget:
    def test_image_kind(self, tmp_path: Path):
        p = tmp_path / "img.png"
        p.write_bytes(b"")
        t = Target(path=p)
        assert t.kind == TargetKind.IMAGE

    def test_jpeg_image_kind(self, tmp_path: Path):
        p = tmp_path / "img.jpg"
        p.write_bytes(b"")
        t = Target(path=p)
        assert t.kind == TargetKind.IMAGE

    def test_video_kind(self, tmp_path: Path):
        p = tmp_path / "vid.mp4"
        p.write_bytes(b"")
        t = Target(path=p)
        assert t.kind == TargetKind.VIDEO

    def test_wmv_kind_is_video(self, tmp_path: Path):
        # Extension-based detection recognizes wmv even where the OS mimetypes
        # registry doesn't (the bug this replaced).
        p = tmp_path / "clip.wmv"
        p.write_bytes(b"")
        assert Target(path=p).kind == TargetKind.VIDEO

    def test_unknown_mime_rejected_on_kind_access(self, tmp_path: Path):
        p = tmp_path / "thing.xyz"
        p.write_bytes(b"")
        t = Target(path=p)
        with pytest.raises(ValueError):
            _ = t.kind

    def test_rejects_missing_path(self, tmp_path: Path):
        with pytest.raises(ValueError):
            Target(path=tmp_path / "missing.mp4")

    def test_rejects_directory(self, tmp_path: Path):
        with pytest.raises(ValueError):
            Target(path=tmp_path)
