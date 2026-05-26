import json
import subprocess

import numpy as np

from sinner2.config.target import Target
from sinner2.types import Frame, FrameIndex


class VideoTargetReader:
    """Reads BGR frames from a video file via a persistent ffmpeg subprocess.

    Sequential reads share one decoder process — efficient for normal
    playback. An out-of-order or random-seek read restarts the subprocess
    at the target frame, which costs a fork + ffmpeg init (~100-200ms).

    Frame seek uses `-ss <time> -i <input>` which is the fast form — it
    seeks to the nearest preceding keyframe, then drops frames until the
    target. For non-keyframe positions in long-GOP codecs this may land
    a frame or two off the requested index. The design (§5) accepts that
    for v1; exact-frame seek is a future optimization.
    """

    def __init__(self, target: Target) -> None:
        self._target = target
        self._fps, self._frame_count, self._width, self._height = self._probe()
        self._decoder: subprocess.Popen[bytes] | None = None
        self._next_index: FrameIndex = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def read(self, index: FrameIndex) -> Frame | None:
        if index < 0 or index >= self._frame_count:
            return None
        if self._decoder is None or index != self._next_index:
            self._start_decoder_at(index)
        frame = self._read_frame_from_pipe()
        if frame is not None:
            self._next_index = index + 1
        return frame

    def release(self) -> None:
        if self._decoder is None:
            return
        try:
            if self._decoder.stdout is not None:
                self._decoder.stdout.close()
            self._decoder.terminate()
            try:
                self._decoder.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._decoder.kill()
                self._decoder.wait(timeout=1.0)
        except Exception:
            pass
        self._decoder = None

    def _probe(self) -> tuple[float, int, int, int]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,nb_frames,width,height,duration",
                "-of", "json",
                str(self._target.path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        streams = data.get("streams") or []
        if not streams:
            raise OSError(f"no video stream in {self._target.path}")
        s = streams[0]
        num_s, _, den_s = s.get("avg_frame_rate", "30/1").partition("/")
        try:
            den = float(den_s) if den_s else 1.0
            fps = float(num_s) / den if den > 0 else 30.0
        except ValueError:
            fps = 30.0
        try:
            frame_count = int(s["nb_frames"])
        except (KeyError, ValueError, TypeError):
            try:
                frame_count = int(float(s.get("duration", 0)) * fps)
            except (ValueError, TypeError):
                frame_count = 0
        width = int(s["width"])
        height = int(s["height"])
        return fps, frame_count, width, height

    def _start_decoder_at(self, index: FrameIndex) -> None:
        self.release()
        start_time = index / self._fps if self._fps > 0 else 0.0
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-ss", f"{start_time:.6f}",
            "-i", str(self._target.path),
            "-vsync", "0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]
        self._decoder = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._next_index = index

    def _read_frame_from_pipe(self) -> Frame | None:
        if self._decoder is None or self._decoder.stdout is None:
            return None
        frame_size = self._width * self._height * 3
        raw = self._decoder.stdout.read(frame_size)
        if len(raw) < frame_size:
            return None
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self._height, self._width, 3)
            .copy()
        )
