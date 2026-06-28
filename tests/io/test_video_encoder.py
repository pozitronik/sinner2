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
    probe_audio_codec,
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
        # Defensive: the probes shouldn't raise; just report "no audio" so the
        # encoder skips the audio-mux arguments cleanly.
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        assert probe_audio_codec(tmp_path / "any.mp4") is None
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


class TestProgressCallback:
    """The combine/encode step reports frame-count progress (parsed from
    ffmpeg's -progress stream) so the batch bar doesn't freeze while a long
    video muxes. These mock Popen so they run without ffmpeg on PATH."""

    def _fake_popen(self, lines, returncode=0, stderr_text="", captured=None):
        import io

        class _FakePopen:
            def __init__(self, cmd, **_kw):
                if captured is not None:
                    captured["cmd"] = cmd
                self.stdout = iter(lines)
                self.stderr = io.StringIO(stderr_text)
                self.returncode = None

            def wait(self):
                self.returncode = returncode
                return returncode

        return _FakePopen

    def test_callback_receives_parsed_frame_counts(self, tmp_path, monkeypatch):
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        lines = [
            "frame=1\n", "fps=0.0\n", "progress=continue\n",
            "frame=3\n", "frame=5\n", "progress=end\n",
        ]
        monkeypatch.setattr(subprocess, "Popen", self._fake_popen(lines))
        got: list[int] = []
        encode_frames_to_mp4(
            tmp_path / "frames", tmp_path / "out.mp4", fps=10.0,
            progress_callback=got.append,
        )
        assert got == [1, 3, 5]

    def test_progress_flags_added_only_with_callback(self, tmp_path, monkeypatch):
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        captured: dict = {}
        monkeypatch.setattr(
            subprocess, "Popen",
            self._fake_popen(["progress=end\n"], captured=captured),
        )
        encode_frames_to_mp4(
            tmp_path / "frames", tmp_path / "out.mp4", fps=10.0,
            progress_callback=lambda _n: None,
        )
        cmd = captured["cmd"]
        assert "-progress" in cmd and "pipe:1" in cmd and "-nostats" in cmd
        assert cmd[-1] == str(tmp_path / "out.mp4")  # output stays last

    def test_nonzero_exit_raises_with_stderr(self, tmp_path, monkeypatch):
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(
            subprocess, "Popen",
            self._fake_popen(["frame=1\n"], returncode=1, stderr_text="boom"),
        )
        with pytest.raises(subprocess.CalledProcessError) as exc:
            encode_frames_to_mp4(
                tmp_path / "frames", tmp_path / "out.mp4", fps=10.0,
                progress_callback=lambda _n: None,
            )
        assert "boom" in (exc.value.stderr or "")


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
class TestRealProgress:
    def test_real_encode_reports_progress(self, tmp_path):
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        _make_frames(frame_dir, n=10)
        got: list[int] = []
        encode_frames_to_mp4(
            frame_dir, tmp_path / "out.mp4", fps=10.0,
            progress_callback=got.append,
        )
        assert (tmp_path / "out.mp4").is_file()
        # ffmpeg streamed at least one frame count, monotonic up toward 10.
        assert got
        assert got == sorted(got)
        assert max(got) >= 1


class TestCutAudioCommand:
    """A section trim cuts + concatenates the audio (atrim/concat → AAC) so it
    stays in sync with the selected frames, instead of stream-copying it whole."""

    def test_audio_segments_build_filter_concat(self, tmp_path, monkeypatch):
        import subprocess
        from sinner2.io import video_encoder

        captured: dict = {}

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **_kw):
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(video_encoder, "probe_audio_codec", lambda _p: "aac")
        monkeypatch.setattr(subprocess, "run", fake_run)
        encode_frames_to_mp4(
            tmp_path / "frames", tmp_path / "out.mp4", fps=10.0,
            audio_source=tmp_path / "a.mp4",
            audio_segments=[(1.0, 3.0), (5.0, 6.0)],
        )
        cmd = captured["cmd"]
        joined = " ".join(cmd)
        assert "-filter_complex" in cmd
        assert "atrim=start=1.000000:end=3.000000" in joined
        assert "atrim=start=5.000000:end=6.000000" in joined
        assert "concat=n=2:v=0:a=1" in joined
        assert "aac" in cmd          # re-encoded (atrim can't stream-copy)
        assert "copy" not in cmd     # so NOT a stream copy
        assert "-vf" not in cmd      # scale moved into the filter graph

    def test_no_segments_keeps_stream_copy(self, tmp_path, monkeypatch):
        import subprocess
        from sinner2.io import video_encoder

        captured: dict = {}

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **_kw):
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(video_encoder, "probe_audio_codec", lambda _p: "aac")
        monkeypatch.setattr(subprocess, "run", fake_run)
        encode_frames_to_mp4(
            tmp_path / "frames", tmp_path / "out.mp4", fps=10.0,
            audio_source=tmp_path / "a.mp4",
        )
        cmd = captured["cmd"]
        assert "copy" in cmd
        assert "-filter_complex" not in cmd

    def test_incompatible_codec_reencodes_to_aac(self, tmp_path, monkeypatch):
        # WMA (from a .wmv) can't be stream-copied into MP4 — `-c:a copy` fails
        # the mux with EINVAL. The encoder must re-encode it to AAC instead.
        import subprocess
        from sinner2.io import video_encoder

        captured: dict = {}

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **_kw):
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(video_encoder, "probe_audio_codec", lambda _p: "wmav2")
        monkeypatch.setattr(subprocess, "run", fake_run)
        encode_frames_to_mp4(
            tmp_path / "frames", tmp_path / "out.mp4", fps=10.0,
            audio_source=tmp_path / "a.wmv",
        )
        cmd = captured["cmd"]
        assert "aac" in cmd          # re-encoded, not copied
        assert "copy" not in cmd
        assert "1:a:0" in cmd        # audio still mapped through


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
class TestRealCutAudio:
    def test_cut_audio_matches_selection_duration(self, tmp_path):
        import subprocess

        # 10s of silent audio; the selection keeps 2s of it (1..2 + 5..6).
        audio = tmp_path / "audio.wav"
        subprocess.run(
            [
                _FFMPEG, "-y", "-f", "lavfi",
                "-i", "anullsrc=channel_layout=mono:sample_rate=22050",
                "-t", "10", str(audio),
            ],
            check=True, capture_output=True,
        )
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        _make_frames(frame_dir, n=20)  # 2s of video at fps 10
        out = tmp_path / "trim.mp4"
        encode_frames_to_mp4(
            frame_dir, out, fps=10.0, audio_source=audio,
            audio_segments=[(1.0, 2.0), (5.0, 6.0)],
        )
        assert out.is_file()
        # Output has audio, and its duration ~2s (the kept selection), not 10s.
        types = subprocess.run(
            [
                _FFPROBE, "-v", "error", "-show_entries", "stream=codec_type",
                "-of", "csv=p=0", str(out),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert "audio" in types
        dur = float(subprocess.run(
            [
                _FFPROBE, "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(out),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        assert 1.6 < dur < 2.6  # ~2s, not the full 10s


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


class TestAudioMuxCommand:
    def test_audio_mux_does_not_use_shortest(self, tmp_path, monkeypatch):
        # -shortest ends output at the shorter stream; for VFR / fps-rounding the
        # reconstructed video can run longer than the copied source audio, so
        # -shortest silently drops trailing processed frames. The video stream
        # must define duration → no -shortest.
        import subprocess
        from sinner2.io import video_encoder

        captured = {}

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **_kw):
            captured["cmd"] = cmd
            return _R()

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(video_encoder, "probe_audio_codec", lambda _p: "aac")
        monkeypatch.setattr(subprocess, "run", fake_run)
        encode_frames_to_mp4(
            tmp_path / "frames",
            tmp_path / "out.mp4",
            fps=30.0,
            audio_source=tmp_path / "a.wav",
        )
        cmd = captured["cmd"]
        assert "-map" in cmd and "1:a:0" in cmd  # audio path was taken
        assert "-shortest" not in cmd


class TestExtraEncodeArgs:
    """The power-user `encode_args` string is shlex-split and appended to the
    output options just before the output file, so it overrides the defaults."""

    def test_extra_args_appended_before_output_and_override_codec(
        self, tmp_path, monkeypatch
    ):
        import subprocess as sp

        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        captured: dict = {}

        def fake_run(cmd, **_kw):
            captured["cmd"] = list(cmd)
            return sp.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(sp, "run", fake_run)
        frames = tmp_path / "frames"
        frames.mkdir()
        out = tmp_path / "out.mp4"
        encode_frames_to_mp4(
            frames, out, fps=10.0, encode_args="-c:v libx265 -crf 24"
        )
        cmd = captured["cmd"]
        assert cmd[-1] == str(out)  # output stays last
        i = cmd.index("libx265")
        assert cmd[i - 1] == "-c:v"
        assert cmd[i + 1 : i + 3] == ["-crf", "24"]
        # User override is placed AFTER the default libx264 block → it wins.
        assert cmd.index("libx265") > cmd.index("libx264")

    def test_empty_args_is_a_noop(self, tmp_path, monkeypatch):
        import subprocess as sp

        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        captured: dict = {}
        monkeypatch.setattr(
            sp, "run",
            lambda cmd, **_kw: captured.setdefault("cmd", list(cmd))
            or sp.CompletedProcess(cmd, 0, "", ""),
        )
        frames = tmp_path / "frames"
        frames.mkdir()
        encode_frames_to_mp4(frames, tmp_path / "out.mp4", fps=10.0, encode_args="  ")
        assert "libx265" not in captured["cmd"]  # nothing extra spliced

    def test_invalid_args_raise_valueerror(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        frames = tmp_path / "frames"
        frames.mkdir()
        with pytest.raises(ValueError):
            encode_frames_to_mp4(
                frames, tmp_path / "out.mp4", fps=10.0,
                encode_args='-c:v "unbalanced',  # missing closing quote
            )
