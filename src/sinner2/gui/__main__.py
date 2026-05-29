import os
import sys

# Silence libavcodec / libavformat stderr noise from cv2's FFmpeg
# backend (the "[h264 @ ...] Invalid NAL unit size" / "Error splitting
# the input into NAL units" floods). These come from probing damaged
# or non-seekable mp4s — we already fall back to frame 0, so the user
# doesn't need the warning storm. -8 = AV_LOG_QUIET.
# MUST be set before cv2 is first imported anywhere — placing it here
# (the entry point) catches every downstream import.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

from PySide6.QtWidgets import QApplication  # noqa: E402

from sinner2.gui.main_window import SinnerMainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    # Set the application identity so the OS associates dialogs, MRU lists,
    # and per-app settings (Windows JumpLists, Recent items) with sinner2.
    app.setApplicationName("sinner2")
    app.setOrganizationName("sinner2")
    window = SinnerMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
