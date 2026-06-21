#build with: conda run -n astrolab pyinstaller --clean AstroImageLab.spec
#
# build the local environment: conda env create -f d:\GitHub\AstroImageLab\environment.yml
#
# PR → merge to main (CI runs tests + build to verify everything works)
# Tag the merge commit on main → triggers the release workflow
# git tag v0.0.6
# git push origin v0.0.6

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
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QRadialGradient
from PyQt6.QtCore import Qt, QTimer
from gui.main_window import MainWindow
from core.models import SPLASH_DURATION_MS


def _make_splash_pixmap() -> QPixmap:
    W, H = 600, 300
    pix = QPixmap(W, H)
    pix.fill(QColor(13, 17, 23))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Dividing line
    p.setPen(QColor(200, 200, 200))
    p.drawLine(W // 2, 20, W // 2, H - 20)

    # Left panel label (A)
    p.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
    p.setPen(QColor(139, 191, 255))
    p.drawText(18, 44, "A")

    # Right panel label (B)
    p.setPen(QColor(255, 200, 120))
    p.drawText(W - 34, 44, "B")

    # Left star — broad blurry radial glow (Image A, muted blue-white)
    grad_a = QRadialGradient(W // 4, H // 2, H // 3)
    grad_a.setColorAt(0.0, QColor(139, 191, 255, 160))
    grad_a.setColorAt(0.4, QColor(100, 140, 200, 60))
    grad_a.setColorAt(1.0, QColor(13, 17, 23, 0))
    p.setBrush(grad_a)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(W // 4 - H // 3, H // 2 - H // 3, H // 3 * 2, H // 3 * 2)

    # Right star — tight bright radial glow (Image B, warm white)
    grad_b = QRadialGradient(3 * W // 4, H // 2, H // 8)
    grad_b.setColorAt(0.0, QColor(255, 252, 224, 255))
    grad_b.setColorAt(0.3, QColor(255, 220, 120, 120))
    grad_b.setColorAt(1.0, QColor(13, 17, 23, 0))
    p.setBrush(grad_b)
    p.drawEllipse(3 * W // 4 - H // 8, H // 2 - H // 8, H // 4, H // 4)

    # Title
    p.setPen(QColor(240, 240, 240))
    title_font = QFont("Segoe UI", 26, QFont.Weight.Bold)
    p.setFont(title_font)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
               "Astro Image Lab")

    # Subtitle
    p.setPen(QColor(160, 170, 185))
    p.setFont(QFont("Segoe UI", 12))
    sub_rect = pix.rect().adjusted(0, H // 2 + 44, 0, 0)
    p.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
               "Astronomical Image Comparison & Analysis")

    p.end()
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
    splash = QSplashScreen(_make_splash_pixmap(),
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
