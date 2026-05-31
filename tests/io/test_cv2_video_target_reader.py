import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sinner2.config.target import Target
from sinner2.io.cv2_video_target_reader import CV2VideoTargetReader
from sinner2.io.target_reader import TargetReader

# CV2 itself reads via its bundled ffmpeg, but we still need ffmpeg to
# *encode* the fixture video. Skip the whole file if ffmpeg isn't there.
_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg not installed")


@pytest.fixture
def blue_video(tmp_path: Path) -> Target:
    """30-frame blue 64x48 video at 10 fps. Same fixture as the ffmpeg
    backend tests so behaviour can be compared one-to-one."""
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


class TestCV2VideoTargetReader:
    def test_conforms_to_protocol(self, blue_video: Target):
        r = CV2VideoTargetReader(blue_video)
        try:
            assert isinstance(r, TargetReader)
        finally:
            r.release()

    def test_reports_metadata(self, blue_video: Target):
        r = CV2VideoTargetReader(blue_video)
        try:
            assert r.frame_count == 30
            # cv2 occasionally reports fps as a slight rounding of the
            # encoded value; just check the order of magnitude.
            assert 9.0 <= r.fps <= 11.0
            assert r.width == 64
            assert r.height == 48
        finally:
            r.release()

    def test_sequential_reads(self, blue_video: Target):
        r = CV2VideoTargetReader(blue_video)
        try:
            for i in range(5):
                f = r.read(i)
                assert f is not None
                assert f.shape == (48, 64, 3)
        finally:
            r.release()

    def test_scale_downscales_reads_and_reports_native(self, blue_video: Target):
        r = CV2VideoTargetReader(blue_video, scale=0.5)
        try:
            assert (r.width, r.height) == (32, 24)
            assert (r.native_width, r.native_height) == (64, 48)
            f = r.read(0)
            assert f is not None and f.shape == (24, 32, 3)
        finally:
            r.release()

    def test_random_seek_does_not_restart(self, blue_video: Target):
        # The whole point of this backend — read out of order and still
        # get frames back. The internal _next_index advances normally.
        r = CV2VideoTargetReader(blue_video)
        try:
            assert r.read(20) is not None
            assert r.read(5) is not None
            assert r.read(15) is not None
        finally:
            r.release()

    def test_out_of_range_returns_none(self, blue_video: Target):
        r = CV2VideoTargetReader(blue_video)
        try:
            assert r.read(-1) is None
            assert r.read(9999) is None
        finally:
            r.release()

    def test_release_is_idempotent(self, blue_video: Target):
        r = CV2VideoTargetReader(blue_video)
        r.release()
        r.release()  # must not raise

    def test_corrupt_file_raises(self, tmp_path: Path):
        # Target's pydantic validator already rejects nonexistent paths,
        # so the bad-file case has to be "exists but isn't a video". cv2
        # rejects garbage bytes at VideoCapture-open time.
        garbage = tmp_path / "not_a_video.mp4"
        garbage.write_bytes(b"this is not a video file")
        with pytest.raises(OSError):
            CV2VideoTargetReader(Target(path=garbage))

    def test_returns_a_blue_frame(self, blue_video: Target):
        r = CV2VideoTargetReader(blue_video)
        try:
            f = r.read(0)
            assert f is not None
            # cv2 returns BGR — the blue channel should dominate.
            b, g, red = f[:, :, 0].mean(), f[:, :, 1].mean(), f[:, :, 2].mean()
            assert b > g
            assert b > red
            # And the frame should be mostly the same colour (low std).
            assert np.std(f[:, :, 0]) < 30
        finally:
            r.release()


class TestVideoBackendFactory:
    def test_dispatches_to_cv2_backend(self, blue_video: Target):
        from sinner2.io.video_backend import VideoBackend, build_video_target_reader

        r = build_video_target_reader(blue_video, VideoBackend.CV2)
        try:
            assert isinstance(r, CV2VideoTargetReader)
            assert isinstance(r, TargetReader)
        finally:
            r.release()

    def test_dispatches_to_ffmpeg_backend(self, blue_video: Target):
        from sinner2.io.video_backend import VideoBackend, build_video_target_reader
        from sinner2.io.video_target_reader import FFmpegVideoTargetReader

        r = build_video_target_reader(blue_video, VideoBackend.FFMPEG)
        try:
            assert isinstance(r, FFmpegVideoTargetReader)
            assert isinstance(r, TargetReader)
        finally:
            r.release()

    def test_rejects_unknown_backend(self, blue_video: Target):
        from sinner2.io.video_backend import build_video_target_reader

        class Bogus:
            pass

        with pytest.raises(ValueError):
            build_video_target_reader(blue_video, Bogus())  # type: ignore[arg-type]
