"""Project-root conftest. Sets process-wide env vars that must land
before cv2 / Qt are first imported anywhere in the test process."""
import os

# Silence libavcodec / libavformat stderr noise from cv2's FFmpeg
# backend during video thumbnail extraction. Same setting the app
# applies in sinner2.gui.__main__ — kept in sync so test output
# matches production runtime behaviour. -8 = AV_LOG_QUIET.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
