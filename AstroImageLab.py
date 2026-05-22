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

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Astro Image Lab")
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
