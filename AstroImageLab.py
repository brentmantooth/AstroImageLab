#build with: pyinstaller AstroImageLab.spec

import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Astro Image Lab")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
