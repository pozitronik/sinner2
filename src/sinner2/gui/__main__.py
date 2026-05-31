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

# Claim PyTorch's CUDA context on the main thread NOW — before Qt, ONNX
# Runtime, or any worker thread touches CUDA. torch otherwise initializes
# CUDA lazily on first use, which in this app is a batch worker thread (after
# ORT already has the device); eager main-thread init avoids the fragile
# ordering that can leave the FaceEnhancer (GFPGAN) stuck on the CPU.
try:
    import torch

    if torch.cuda.is_available():
        torch.cuda.init()
except Exception:  # noqa: BLE001  # CPU-only env / no torch — harmless
    pass

from PySide6.QtWidgets import QApplication  # noqa: E402

from sinner2.gui.icon import app_icon  # noqa: E402
from sinner2.gui.main_window import SinnerMainWindow  # noqa: E402


def _set_windows_app_id() -> None:
    """Give Windows an explicit AppUserModelID so the taskbar uses OUR window
    icon (and groups our windows) instead of falling back to the generic
    python.exe icon. Must run before the first window is shown. No-op off
    Windows / if the shell call isn't available."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("sinner2.app")
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    _set_windows_app_id()
    app = QApplication(sys.argv)
    # Set the application identity so the OS associates dialogs, MRU lists,
    # and per-app settings (Windows JumpLists, Recent items) with sinner2.
    app.setApplicationName("sinner2")
    app.setOrganizationName("sinner2")
    # App-wide icon: covers every window + dialog (the main window also sets it
    # explicitly as a belt-and-suspenders for platforms that prefer per-window).
    app.setWindowIcon(app_icon())
    window = SinnerMainWindow()
    window.show()
    # First-run: if the model files are missing, offer to download them before
    # the user tries to process anything (modal; no-op when they're present).
    from sinner2.gui.model_download import ensure_models_present

    ensure_models_present(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
