import sys

from PySide6.QtWidgets import QApplication

from sinner2.gui.main_window import SinnerMainWindow


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
