#build with: conda run -n astrolab pyinstaller --clean AstroImageLab.spec
#
# build the local environment: conda env create -f d:\GitHub\AstroImageLab\environment.yml
#


import sys
import os

# Prepend PyQt6's own Qt6 bin directory to PATH so Windows finds the correct Qt6
# DLLs before any conflicting copies installed by other programs (e.g. MiKTeX).
# This must happen before any PyQt6 import.
if not getattr(sys, "frozen", False):
    import importlib.util
    _spec = importlib.util.find_spec("PyQt6")
    if _spec and _spec.submodule_search_locations:
        _qt_bin = os.path.join(list(_spec.submodule_search_locations)[0], "Qt6", "bin")
        if os.path.isdir(_qt_bin):
            os.environ["PATH"] = _qt_bin + os.pathsep + os.environ.get("PATH", "")

import time
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, QTimer
from gui.main_window import MainWindow
from core.models import SPLASH_DURATION_MS


def _load_splash_pixmap() -> QPixmap:
    splash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "resources", "AstroImageLabSplash.png")
    pix = QPixmap(splash_path)
    if not pix.isNull():
        pix = pix.scaledToWidth(640, Qt.TransformationMode.SmoothTransformation)
    return pix


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Astro Image Lab")

    # Application icon
    _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "resources", "icon.ico")
    if os.path.exists(_icon_path):
        app.setWindowIcon(QIcon(_icon_path))

    # Splash screen
    splash = QSplashScreen(_load_splash_pixmap(),
                           Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    # Explicit tooltip colours so they remain readable in both dark and light mode.
    # Qt inherits tooltip colours from the system palette; on Windows dark mode the
    # default produces dark-grey text on a dark-grey background.
    app.setStyleSheet("""
        QToolTip {
            background-color: #1e1e1e;
            color: #f0f0f0;
            border: 1px solid #555555;
            padding: 4px 6px;
            border-radius: 3px;
            font-size: 9pt;
        }
    """)
    t_start = time.monotonic()
    window = MainWindow()
    window.show()
    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    QTimer.singleShot(max(0, SPLASH_DURATION_MS - elapsed_ms), splash.close)
    sys.exit(app.exec())
