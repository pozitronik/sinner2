from pathlib import Path

import pytest

from sinner2.config.source import Source


class TestSource:
    def test_accepts_existing_file(self, tmp_path: Path):
        p = tmp_path / "face.png"
        p.write_bytes(b"")
        s = Source(path=p)
        assert s.path == p

    def test_rejects_missing_file(self, tmp_path: Path):
        with pytest.raises(ValueError):
            Source(path=tmp_path / "missing.png")

    def test_rejects_directory(self, tmp_path: Path):
        with pytest.raises(ValueError):
            Source(path=tmp_path)

    def test_accepts_string_path(self, tmp_path: Path):
        p = tmp_path / "face.png"
        p.write_bytes(b"")
        s = Source.model_validate({"path": str(p)})
        assert s.path == p
