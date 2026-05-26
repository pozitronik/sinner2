import shutil
import subprocess
from pathlib import Path

import pytest

from sinner2.config.target import Target
from sinner2.io.target_reader import TargetReader
from sinner2.io.video_target_reader import VideoTargetReader

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg not installed")


@pytest.fixture
def blue_video(tmp_path: Path) -> Target:
    """30-frame blue 64x48 video at 10 fps, encoded with H.264."""
    path = tmp_path / "blue.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", "color=c=blue:s=64x48:d=3:r=10",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )
    return Target(path=path)


class TestVideoTargetReader:
    def test_compliant_with_protocol(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            assert isinstance(r, TargetReader)
        finally:
            r.release()

    def test_fps_matches_source(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            assert abs(r.fps - 10.0) < 0.1
        finally:
            r.release()

    def test_frame_count_matches_source(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            assert r.frame_count == 30
        finally:
            r.release()

    def test_dimensions_match_source(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            assert r.width == 64
            assert r.height == 48
        finally:
            r.release()

    def test_read_first_frame_returns_bgr_blue(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            f = r.read(0)
            assert f is not None
            assert f.shape == (48, 64, 3)
            b_mean = float(f[:, :, 0].mean())
            r_mean = float(f[:, :, 2].mean())
            assert b_mean > r_mean + 100  # blue dominates the red channel
        finally:
            r.release()

    def test_out_of_range_returns_none(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            assert r.read(30) is None
            assert r.read(100) is None
            assert r.read(-1) is None
        finally:
            r.release()

    def test_sequential_reads_succeed(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            for i in range(5):
                assert r.read(i) is not None
        finally:
            r.release()

    def test_random_seek_returns_frames(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        try:
            f15 = r.read(15)
            f5 = r.read(5)
            f25 = r.read(25)
            assert f15 is not None
            assert f5 is not None
            assert f25 is not None
        finally:
            r.release()

    def test_release_is_idempotent(self, blue_video: Target):
        r = VideoTargetReader(blue_video)
        r.release()
        r.release()
