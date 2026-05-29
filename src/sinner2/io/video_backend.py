"""Video reader backend selection.

Two implementations of TargetReader for video files:

  FFMPEG: ffmpeg subprocess + ffprobe. One-shot decoder per session;
          random-access reads cost a process restart, so this is best
          for strictly sequential playback (BestEffortStrategy on a
          fast local source).
  CV2:    cv2.VideoCapture. One persistent capture; cap.set(POS_FRAMES,
          idx) seeks in place. Slightly higher per-frame overhead than
          the ffmpeg pipe but vastly faster random access — the right
          pick when scrubbing or running SyncedStrategy on a slow source
          (network drive, HDD).

Pick via settings. The factory is the only place that knows about both
implementations; everything else talks to the TargetReader Protocol.
"""
from __future__ import annotations

from enum import Enum

from sinner2.config.target import Target
from sinner2.io.target_reader import TargetReader


class VideoBackend(str, Enum):
    FFMPEG = "ffmpeg"
    CV2 = "cv2"


def build_video_target_reader(target: Target, backend: VideoBackend) -> TargetReader:
    if backend is VideoBackend.FFMPEG:
        from sinner2.io.video_target_reader import FFmpegVideoTargetReader

        return FFmpegVideoTargetReader(target)
    if backend is VideoBackend.CV2:
        from sinner2.io.cv2_video_target_reader import CV2VideoTargetReader

        return CV2VideoTargetReader(target)
    raise ValueError(f"unknown video backend: {backend!r}")
