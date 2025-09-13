import sys

from PySide6.QtWidgets import QApplication
import qt_material

from src.ui import AstroPoseMainWindow


def main():
    app = QApplication(sys.argv)
    try:
        qt_material.apply_stylesheet(app, theme="dark_blue.xml")
    except Exception:
        pass
    win = AstroPoseMainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()