"""Tests for the persisted Batch Defaults template."""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.batch import defaults as batch_defaults
from sinner2.batch.task import (
    BatchCleanupMode,
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
)


class TestDefaultTemplate:
    def test_default_template_uses_batchtask_defaults(self):
        tmpl = batch_defaults.default_template()
        # Mirrors BatchTask's own field defaults — the template is just a
        # BatchTask, so this stays in lockstep with the schema for free.
        assert tmpl.swapper_enabled is True
        assert tmpl.output_format is BatchOutputFormat.VIDEO
        assert tmpl.cleanup_mode is BatchCleanupMode.KEEP
        assert tmpl.processing_scale == 1.0

    def test_default_template_has_sentinel_paths(self):
        tmpl = batch_defaults.default_template()
        assert tmpl.source_path == Path(".")
        assert tmpl.target_path == Path(".")


class TestLoadSave:
    def test_load_missing_returns_stock_defaults(self, tmp_path):
        loaded = batch_defaults.load_defaults(tmp_path / "nope.json")
        assert loaded.output_format is BatchOutputFormat.VIDEO

    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / "batch_defaults.json"
        tmpl = batch_defaults.default_template().model_copy(
            update={
                "output_format": BatchOutputFormat.FRAMES,
                "cleanup_mode": BatchCleanupMode.AUTO,
                "swapper_model": "ghost_2_256",
                "processing_scale": 0.5,
            }
        )
        batch_defaults.save_defaults(path, tmpl)
        loaded = batch_defaults.load_defaults(path)
        assert loaded.output_format is BatchOutputFormat.FRAMES
        assert loaded.cleanup_mode is BatchCleanupMode.AUTO
        assert loaded.swapper_model == "ghost_2_256"
        assert loaded.processing_scale == 0.5

    def test_save_is_atomic_no_tmp_left(self, tmp_path):
        path = tmp_path / "batch_defaults.json"
        batch_defaults.save_defaults(path, batch_defaults.default_template())
        assert path.is_file()
        assert not (tmp_path / "batch_defaults.json.tmp").exists()

    def test_load_corrupt_returns_stock_defaults(self, tmp_path):
        path = tmp_path / "batch_defaults.json"
        path.write_text("{ this is not json", encoding="utf-8")
        loaded = batch_defaults.load_defaults(path)
        assert isinstance(loaded, BatchTask)
        assert loaded.output_format is BatchOutputFormat.VIDEO


class TestMintTask:
    def test_mint_sets_source_target_and_keeps_config(self):
        tmpl = batch_defaults.default_template().model_copy(
            update={
                "swapper_model": "uniface_256",
                "enhancer_enabled": False,
                "processing_scale": 0.75,
            }
        )
        task = batch_defaults.mint_task(
            tmpl, Path("/s/face.png"), Path("/t/clip.mp4")
        )
        assert task.source_path == Path("/s/face.png")
        assert task.target_path == Path("/t/clip.mp4")
        # Config copied verbatim from the template.
        assert task.swapper_model == "uniface_256"
        assert task.enhancer_enabled is False
        assert task.processing_scale == 0.75

    def test_mint_resets_identity_and_runtime(self):
        tmpl = batch_defaults.default_template().model_copy(
            update={
                "status": BatchTaskStatus.COMPLETED,
                "last_completed_frame": 999,
                "total_frames": 1000,
                "completed_stages": 2,
                "cache_fingerprint": "stale",
                "error_message": "boom",
                "started_at": 1.0,
                "finished_at": 2.0,
                "output_path": Path("/old/out.mp4"),
            }
        )
        task = batch_defaults.mint_task(tmpl, Path("/s.png"), Path("/t.mp4"))
        assert task.status is BatchTaskStatus.PENDING
        assert task.last_completed_frame == -1
        assert task.total_frames == -1
        assert task.completed_stages == 0
        assert task.cache_fingerprint == ""
        assert task.error_message is None
        assert task.started_at is None
        assert task.finished_at is None
        assert task.output_path is None  # auto-derive, not the template's path

    def test_mint_generates_fresh_unique_id(self):
        tmpl = batch_defaults.default_template()
        a = batch_defaults.mint_task(tmpl, Path("/s.png"), Path("/t.mp4"))
        b = batch_defaults.mint_task(tmpl, Path("/s.png"), Path("/t.mp4"))
        assert a.id != b.id
        assert a.id != tmpl.id


class TestDefaultsPath:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        target = tmp_path / "custom_defaults.json"
        monkeypatch.setenv("SINNER2_BATCH_DEFAULTS_PATH", str(target))
        assert batch_defaults.batch_defaults_path() == target

    def test_defaults_sit_beside_settings(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SINNER2_BATCH_DEFAULTS_PATH", raising=False)
        settings_file = tmp_path / "sub" / "settings.json"
        monkeypatch.setenv("SINNER2_SETTINGS_PATH", str(settings_file))
        assert (
            batch_defaults.batch_defaults_path()
            == settings_file.parent / "batch_defaults.json"
        )
