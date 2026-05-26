import sys

from PySide6.QtWidgets import QApplication

from sinner2.gui.main_window import SinnerMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = SinnerMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
