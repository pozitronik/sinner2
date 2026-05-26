import threading
import time

from sinner2.types import FrameIndex


class Timeline:
    """Wall-clock ↔ frame index mapping for one playback session.

    Anchor model: we remember a reference frame and the wall-clock time it
    corresponded to. When playing, current_frame = anchor + elapsed * fps.
    When paused, current_frame stays at the anchor. Seek rebases the anchor.

    All state mutations are RLock-protected; reads are also locked to keep
    the (anchor_frame, anchor_time, is_playing) tuple internally consistent.
    """

    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0; got {fps}")
        self._fps = fps
        self._lock = threading.RLock()
        self._anchor_frame: FrameIndex = 0
        self._anchor_time_s: float = 0.0
        self._is_playing = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    def start(self, from_frame: FrameIndex = 0) -> None:
        """Begin (or resume) playing from the given frame index."""
        with self._lock:
            self._anchor_frame = from_frame
            self._anchor_time_s = time.monotonic()
            self._is_playing = True

    def pause(self) -> None:
        """Freeze the current frame; subsequent reads stay there until start/seek."""
        with self._lock:
            if not self._is_playing:
                return
            self._anchor_frame = self._current_frame_locked()
            self._is_playing = False

    def seek(self, frame: FrameIndex) -> None:
        """Jump to frame. Playing state is preserved (still playing or still paused)."""
        with self._lock:
            self._anchor_frame = frame
            self._anchor_time_s = time.monotonic()

    def current_frame(self) -> FrameIndex:
        with self._lock:
            return self._current_frame_locked()

    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since the last start/seek (0 when paused)."""
        with self._lock:
            if not self._is_playing:
                return 0.0
            return time.monotonic() - self._anchor_time_s

    def _current_frame_locked(self) -> FrameIndex:
        if not self._is_playing:
            return self._anchor_frame
        elapsed = time.monotonic() - self._anchor_time_s
        return self._anchor_frame + int(elapsed * self._fps)
