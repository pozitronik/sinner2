"""Tests for the ffmpeg-backed frame-sequence encoder.

Most tests are skipped when ffmpeg/ffprobe aren't on PATH — the CI
machine may not have them, and the missing-ffmpeg path is exercised by
test_encode_raises_when_ffmpeg_missing via monkeypatching shutil.which.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sinner2.io.video_encoder import (
    FfmpegMissingError,
    encode_frames_to_mp4,
    probe_has_audio,
)


_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_HAS_FFMPEG = _FFMPEG is not None and _FFPROBE is not None


def _make_frames(d: Path, n: int = 5, w: int = 64, h: int = 48) -> None:
    """Solid-grey frames numbered 00000000.jpg ... matching the encoder's
    ffmpeg input glob. Brightness varies per frame but stays inside
    uint8 range so numpy doesn't overflow."""
    for i in range(n):
        value = min(255, 50 + i * 15)
        arr = np.full((h, w, 3), value, dtype=np.uint8)
        Image.fromarray(arr).save(d / f"{i:08d}.jpg", quality=80)


class TestMissingFfmpeg:
    """Both should raise FfmpegMissingError when ffmpeg isn't on PATH —
    the caller (batch driver) catches and falls back to frames mode."""

    def test_encode_raises_when_ffmpeg_missing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(FfmpegMissingError):
            encode_frames_to_mp4(
                tmp_path / "frames",
                tmp_path / "out.mp4",
                fps=30.0,
            )

    def test_probe_returns_false_when_ffprobe_missing(
        self, tmp_path, monkeypatch
    ):
        # Defensive: probe_has_audio shouldn't raise; just say "no
        # audio" so the encoder skips the audio-mux arguments cleanly.
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        assert probe_has_audio(tmp_path / "any.mp4") is False


@pytest.mark.skipif(
    not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH"
)
class TestVideoRoundtrip:
    def test_encodes_jpeg_sequence_to_mp4(self, tmp_path):
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        _make_frames(frame_dir, n=10)
        out = tmp_path / "out.mp4"
        encode_frames_to_mp4(frame_dir, out, fps=10.0)
        assert out.is_file()
        # Sanity: ffprobe sees a video stream + 10 frames.
        import subprocess

        result = subprocess.run(
            [
                _FFPROBE,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "10"

    def test_encodes_with_audio_when_source_has_audio(
        self, tmp_path
    ):
        # Generate a brief silent wav, then mux it in. Wav generation
        # uses ffmpeg directly to avoid a wave-module dependency.
        import subprocess

        audio = tmp_path / "audio.wav"
        subprocess.run(
            [
                _FFMPEG,
                "-y", "-f", "lavfi",
                "-i", "anullsrc=channel_layout=mono:sample_rate=22050",
                "-t", "1",
                str(audio),
            ],
            check=True,
            capture_output=True,
        )
        assert probe_has_audio(audio) is True
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        _make_frames(frame_dir, n=10)
        out = tmp_path / "out_with_audio.mp4"
        encode_frames_to_mp4(
            frame_dir, out, fps=10.0, audio_source=audio
        )
        assert out.is_file()
        # Output must now have BOTH a video and an audio stream.
        result = subprocess.run(
            [
                _FFPROBE,
                "-v", "error",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        types = result.stdout.split()
        assert "video" in types
        assert "audio" in types

    def test_encodes_odd_dimension_frames(self, tmp_path):
        # 65x49 (both odd) would fail libx264/yuv420p without the even-scale
        # filter; the output must exist and be cropped to even dimensions.
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        _make_frames(frame_dir, n=6, w=65, h=49)
        out = tmp_path / "odd.mp4"
        encode_frames_to_mp4(frame_dir, out, fps=6.0)
        assert out.is_file()
        import subprocess

        result = subprocess.run(
            [
                _FFPROBE,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        w, h = (int(x) for x in result.stdout.strip().split(","))
        assert w % 2 == 0 and h % 2 == 0

    def test_audio_source_without_audio_stream_silently_drops_remux(
        self, tmp_path
    ):
        # Source is a mp4 we encode WITHOUT audio. Passing it as
        # audio_source must NOT add an audio stream to the output —
        # probe detects no audio and the encoder skips the -map call.
        import subprocess

        # Build a 1-second silent video-only mp4 via ffmpeg first.
        video_only = tmp_path / "video_only.mp4"
        subprocess.run(
            [
                _FFMPEG,
                "-y", "-f", "lavfi",
                "-i", "color=c=red:s=64x48:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_only),
            ],
            check=True,
            capture_output=True,
        )
        assert probe_has_audio(video_only) is False
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        _make_frames(frame_dir, n=5)
        out = tmp_path / "out_no_audio.mp4"
        encode_frames_to_mp4(
            frame_dir, out, fps=5.0, audio_source=video_only
        )
        result = subprocess.run(
            [
                _FFPROBE,
                "-v", "error",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "audio" not in result.stdout.split()


class TestEvenDimensions:
    def test_cmd_forces_even_dimensions(self, tmp_path, monkeypatch):
        # libx264/yuv420p requires even W/H; odd-dimension sources (phone/screen
        # recordings) otherwise fail the encode with an opaque non-zero exit. The
        # encoder must always apply a trunc-to-even scale filter (no-op if even).
        import subprocess

        captured = {}

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **_kw):
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(subprocess, "run", fake_run)
        encode_frames_to_mp4(tmp_path / "frames", tmp_path / "out.mp4", fps=30.0)
        cmd = captured["cmd"]
        assert "-vf" in cmd
        vf = cmd[cmd.index("-vf") + 1]
        assert "trunc(iw/2)*2" in vf and "trunc(ih/2)*2" in vf
