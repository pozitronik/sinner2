"""Tests for the BatchTask schema + output-path resolution."""
from __future__ import annotations

from pathlib import Path

import pytest

from sinner2.batch.task import (
    BatchCleanupMode,
    BatchExtractionMode,
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
    resolve_output_path,
)
from sinner2.io.video_backend import VideoBackend
from sinner2.pipeline.image_writer import ImageFormat


def _task(tmp_path: Path, **overrides) -> BatchTask:
    """Minimal-valid BatchTask. Required positionals filled from tmp_path."""
    kwargs = {
        "source_path": tmp_path / "src.png",
        "target_path": tmp_path / "tgt.mp4",
    }
    kwargs.update(overrides)
    return BatchTask(**kwargs)


class TestBatchTaskDefaults:
    def test_id_is_auto_generated_unique(self, tmp_path):
        a = _task(tmp_path)
        b = _task(tmp_path)
        assert a.id != b.id
        assert len(a.id) == 12  # short uuid prefix

    def test_status_defaults_to_pending(self, tmp_path):
        assert _task(tmp_path).status is BatchTaskStatus.PENDING

    def test_swapper_enabled_defaults_true(self, tmp_path):
        assert _task(tmp_path).swapper_enabled is True

    def test_output_format_defaults_to_video(self, tmp_path):
        assert _task(tmp_path).output_format is BatchOutputFormat.VIDEO

    def test_chain_defaults_match_face_swapper_defaults(self, tmp_path):
        # Field-by-field check that defaults don't drift from FaceSwapperParams.
        from sinner2.pipeline.processors.face_swapper import FaceSwapperParams

        t = _task(tmp_path)
        s = FaceSwapperParams()
        assert t.swapper_detection_interval == s.detection_interval
        assert t.swapper_many_faces == s.many_faces
        assert t.swapper_target_sex == s.target_sex.value

    def test_runtime_state_starts_unset(self, tmp_path):
        t = _task(tmp_path)
        assert t.last_completed_frame == -1
        assert t.total_frames == -1
        assert t.error_message is None
        assert t.started_at is None
        assert t.finished_at is None


class TestBatchTaskRoundtrip:
    def test_json_roundtrip_preserves_fields(self, tmp_path):
        # Round-trip a fully-populated task; assert every persisted
        # field comes back identical.
        t = BatchTask(
            id="abc123",
            source_path=tmp_path / "src.png",
            target_path=tmp_path / "tgt.mp4",
            output_path=tmp_path / "out.mp4",
            output_format=BatchOutputFormat.FRAMES,
            swapper_detection_interval=3,
            swapper_many_faces=False,
            swapper_target_sex="F",
            enhancer_enabled=False,
            enhancer_upscale=4,
            enhancer_only_center_face=True,
            worker_count=8,
            video_backend=VideoBackend.CV2,
            reader_pool_size=4,
            onnx_providers=["CUDAExecutionProvider"],
            image_format=ImageFormat.PNG,
            image_quality=80,
            status=BatchTaskStatus.RUNNING,
            last_completed_frame=1234,
            total_frames=9999,
            error_message="something",
            started_at=1.0,
            finished_at=2.0,
        )
        payload = t.model_dump_json()
        back = BatchTask.model_validate_json(payload)
        assert back == t

    def test_unknown_fields_ignored(self, tmp_path):
        # Forward-compat: a task file written by a newer sinner2 must
        # load without raising. Use json.dumps so Windows backslashes
        # in the temp path are escaped correctly.
        import json

        payload = json.dumps(
            {
                "id": "x",
                "source_path": str(tmp_path / "s.png"),
                "target_path": str(tmp_path / "t.mp4"),
                "newer_field": "ignore me",
            }
        )
        t = BatchTask.model_validate_json(payload)
        assert t.id == "x"


class TestResolveOutputPath:
    def test_explicit_output_path_wins(self, tmp_path):
        explicit = tmp_path / "custom.mp4"
        t = _task(tmp_path, output_path=explicit)
        assert resolve_output_path(t) == explicit

    def test_explicit_output_path_wins_over_global(self, tmp_path):
        explicit = tmp_path / "custom.mp4"
        t = _task(tmp_path, output_path=explicit)
        assert resolve_output_path(t, global_output_dir=tmp_path / "other") == explicit

    def test_global_output_dir_uses_auto_name(self, tmp_path):
        t = _task(tmp_path)
        out = resolve_output_path(
            t, global_output_dir=tmp_path / "batch_out"
        )
        assert out == tmp_path / "batch_out" / "src+tgt.mp4"

    def test_default_is_next_to_target_with_auto_name(self, tmp_path):
        t = _task(tmp_path)
        # target_path is tmp_path/tgt.mp4 → output is tmp_path/src+tgt.mp4
        assert resolve_output_path(t) == tmp_path / "src+tgt.mp4"

    def test_frames_format_strips_extension(self, tmp_path):
        # FRAMES output is a directory; we use the bare base name so
        # the .mp4 suffix doesn't mislead the user about file vs folder.
        t = _task(tmp_path, output_format=BatchOutputFormat.FRAMES)
        assert resolve_output_path(t) == tmp_path / "src+tgt"

    def test_video_format_always_uses_mp4_ext(self, tmp_path):
        # Even with a .mov target, the auto-named output is .mp4 —
        # the encoder always writes mp4 by design (locked above).
        t = _task(tmp_path, target_path=tmp_path / "clip.mov")
        assert resolve_output_path(t) == tmp_path / "src+clip.mp4"


class TestBatchTaskStatus:
    def test_string_tokens_match_sinner1_convention(self):
        # All status tokens are lowercase strings — matches Settings'
        # str-Enum convention so settings.json round-trips cleanly.
        for member in BatchTaskStatus:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()


class TestStageConfigDefaults:
    def test_cleanup_defaults_to_keep(self, tmp_path):
        assert _task(tmp_path).cleanup_mode is BatchCleanupMode.KEEP

    def test_extraction_defaults_to_stream(self, tmp_path):
        assert _task(tmp_path).extraction_mode is BatchExtractionMode.STREAM

    def test_completed_stages_starts_zero(self, tmp_path):
        assert _task(tmp_path).completed_stages == 0

    def test_stage_config_roundtrips(self, tmp_path):
        t = _task(
            tmp_path,
            cleanup_mode=BatchCleanupMode.AUTO,
            extraction_mode=BatchExtractionMode.PREEXTRACT,
            completed_stages=2,
        )
        back = BatchTask.model_validate_json(t.model_dump_json())
        assert back.cleanup_mode is BatchCleanupMode.AUTO
        assert back.extraction_mode is BatchExtractionMode.PREEXTRACT
        assert back.completed_stages == 2

    def test_mode_tokens_are_lowercase(self):
        for enum in (BatchCleanupMode, BatchExtractionMode):
            for member in enum:
                assert member.value == member.value.lower()
