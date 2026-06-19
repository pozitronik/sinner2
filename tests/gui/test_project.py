"""Tests for the .sinner project value object (serialization round-trip)."""
from __future__ import annotations

import json
from pathlib import Path

from sinner2.gui.project import PROJECT_VERSION, Project
from sinner2.pipeline.image_writer import ImageFormat


def test_roundtrip_preserves_fields(tmp_path: Path):
    p = Project(
        source_path=tmp_path / "face.png",
        target_path=tmp_path / "clip.mp4",
        sections=[[10, 20], [30, 40]],
        processor={
            "swapper_model": "inswapper_128",
            "realtime_workers": 4,
            "image_format": ImageFormat.JPEG,  # enum → token on save
        },
    )
    f = tmp_path / "proj.sinner"
    p.save(f)
    loaded = Project.load(f)
    assert loaded.source_path == tmp_path / "face.png"
    assert loaded.target_path == tmp_path / "clip.mp4"
    assert loaded.sections == [[10, 20], [30, 40]]
    assert loaded.processor["swapper_model"] == "inswapper_128"
    assert loaded.processor["realtime_workers"] == 4
    # The enum is stored as its stable string token.
    assert loaded.processor["image_format"] == ImageFormat.JPEG.value


def test_none_paths_and_sections_roundtrip(tmp_path: Path):
    p = Project(source_path=None, target_path=None, sections=None, processor={})
    f = tmp_path / "blank.sinner"
    p.save(f)
    loaded = Project.load(f)
    assert loaded.source_path is None
    assert loaded.target_path is None
    assert loaded.sections is None
    assert loaded.processor == {}


def test_file_carries_version(tmp_path: Path):
    f = tmp_path / "p.sinner"
    Project(None, None, None, {}).save(f)
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["version"] == PROJECT_VERSION


def test_from_dict_tolerates_missing_keys():
    # Forward/old-file resilience: a sparse dict loads with sensible blanks.
    loaded = Project.from_dict({"version": 1})
    assert loaded.source_path is None
    assert loaded.target_path is None
    assert loaded.sections is None
    assert loaded.processor == {}
