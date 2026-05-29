"""TargetReader backed by cv2.VideoCapture.

One persistent capture object per session; `cap.set(CAP_PROP_POS_FRAMES,
idx)` repositions in place so out-of-order reads are cheap. This is the
backend to pick when:

  - the source is on slow storage (network share, HDD, USB) where
    re-opening the file is expensive;
  - the user scrubs the timeline heavily;
  - SyncedStrategy is active and the system can't keep up (the strategy
    asks for frames out of order; ffmpeg-pipe restarts on every such
    read; cv2 just seeks).

Trade-off vs the ffmpeg-pipe backend: slightly higher per-frame overhead
on strict sequential reads, and seek precision on long-GOP codecs is
approximate (lands on nearest keyframe, may be off by tens of frames).
The ffmpeg backend has the same imprecision via `-ss` keyframe seek,
so this isn't a regression.

Thread safety: cv2.VideoCapture is not thread-safe. The reader is
called only from the dispatcher thread (`_try_submit_next_frame`), so
single-threaded access is the invariant the caller maintains.
"""
from __future__ import annotations

import cv2

from sinner2.config.target import Target
from sinner2.types import Frame, FrameIndex


class CV2VideoTargetReader:
    def __init__(self, target: Target) -> None:
        self._target = target
        cap = cv2.VideoCapture(str(target.path))
        if not cap.isOpened():
            raise OSError(f"cv2.VideoCapture failed to open: {target.path}")
        # CAP_PROP_FPS occasionally reports 0 or NaN for malformed headers;
        # fall back to 30 so downstream math doesn't blow up. Same fallback
        # the ffmpeg backend uses.
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_raw) if fps_raw and fps_raw > 0 else 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise OSError(f"cv2.VideoCapture reports no frames in {target.path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self._fps = fps
        self._frame_count = frame_count
        self._width = width
        self._height = height
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
        # Seek only when the index doesn't match where the capture is
        # already pointing. Sequential reads skip the seek entirely.
        if index != self._next_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._next_index = index + 1
        return frame

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
