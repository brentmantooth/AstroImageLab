"""Regenerate resources/*.png screenshots used by README.md and QuickStart.md.

Constructs the app's widgets directly (no user interaction, no mouse events)
and saves QWidget.grab() output to PNG. This is the same technique documented
in CLAUDE.md's "OS-level screenshots... unreliable" pitfall: .grab() renders
through Qt's own coordinate system, sidestepping OS/DPI virtualization, and
does not require a visible window.

Must be run interactively on a real desktop session — the app follows the OS
theme (dark/light), and the existing screenshots were all captured in Windows
dark mode. Not exercised by CI. Takes a couple of minutes (two full-resolution
synthetic image generations plus one real six-metric analysis run).

Run once from the repo root:
    python tools/generate_screenshots.py
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESOURCES = ROOT / "resources"

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QGroupBox, QMessageBox, QWidget

from core.astro_image import AstroImage
from gui.halo_dialog import HaloAnalyzerDialog
from gui.image_panel import ImagePanel
from gui.main_window import MainWindow
from gui.spatial_target_dialog import SpatialTargetDialog
from gui.synthetic_dialog import SyntheticDialog
from synthetic.generator import SyntheticGenerator

CAMERA = "Player One — Mercury-M"  # smallest full camera in synthetic/cameras.py

_BASE_PARAMS = {
    "camera": CAMERA,
    "focal_length_mm": 500.0,
    "aperture_mm": 100.0,
    "n_stars": 200,           # shared -> star_rng(n_stars) gives A/B matching star positions
    "exposure_s": 300.0,
    "gain_e_per_adu": 1.0,
    "bortle": 4,
    "moffat_beta": 3.5,
    "guiding": 0.3,
    "coma": 0.0,
    "astigmatism": 0.0,
    "spherical": 0.0,
    "collimation": 0.0,
    "backfocus": 0.0,
    "poor_focus": 0.0,
    "field_curvature": 0.0,
    "mag_min": 8.0,
    "mag_max": 14.0,
    "nebula_enabled": True,
}
PARAMS_A = {**_BASE_PARAMS, "fwhm_arcsec": 2.5, "halo": 0.05, "seed": 42}
PARAMS_B = {**_BASE_PARAMS, "fwhm_arcsec": 3.4, "halo": 0.12, "moffat_beta": 3.0, "seed": 43}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def grab(widget: QWidget, path: pathlib.Path) -> None:
    pixmap = widget.grab()
    pixmap.save(str(path))
    print(f"  wrote {path.relative_to(ROOT)}  ({pixmap.width()}x{pixmap.height()})")


def pump(app: QApplication, condition, timeout_s: float = 180, interval_s: float = 0.02) -> None:
    start = time.monotonic()
    while not condition():
        app.processEvents()
        time.sleep(interval_s)
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for condition")


def arm_modal_capture(save_path: pathlib.Path,
                       click_role=QMessageBox.StandardButton.Ok) -> dict:
    """Arm a repeating QTimer that grabs+closes the next QMessageBox to appear.

    Must be armed BEFORE the action that raises the dialog: QMessageBox.information()/
    .question() block via a nested Qt event loop, and only a timer already running
    keeps firing inside that nested loop -- code in our own outer processEvents()
    loop does not resume until the dialog is dismissed.
    """
    state = {"done": False}
    timer = QTimer()
    timer.setInterval(100)

    def _poll() -> None:
        w = QApplication.activeModalWidget()
        if isinstance(w, QMessageBox):
            grab(w, save_path)
            btn = w.button(click_role)
            (btn.click() if btn is not None else w.close())
            state["done"] = True
            timer.stop()

    timer.timeout.connect(_poll)
    timer.start()
    state["_timer"] = timer  # keep a reference alive
    return state


def find_groupbox(parent: QWidget, title: str) -> QGroupBox:
    for box in parent.findChildren(QGroupBox):
        if box.title() == title:
            return box
    raise RuntimeError(f"QGroupBox {title!r} not found")


def grab_side_by_side(widgets: list[QWidget], path: pathlib.Path, gap: int = 12) -> None:
    pixmaps = [w.grab() for w in widgets]
    total_w = sum(p.width() for p in pixmaps) + gap * (len(pixmaps) - 1)
    max_h = max(p.height() for p in pixmaps)
    canvas = QPixmap(total_w, max_h)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    x = 0
    for p in pixmaps:
        painter.drawPixmap(x, 0, p)
        x += p.width() + gap
    painter.end()
    canvas.save(str(path))
    print(f"  wrote {path.relative_to(ROOT)}  ({canvas.width()}x{canvas.height()})")


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def build_sample_data() -> dict:
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="aillab_screenshots_"))
    dir_a, dir_b, dir_lone, dir_report = (tmp_root / n for n in ("a", "b", "lone", "report"))
    for d in (dir_a, dir_b, dir_lone, dir_report):
        d.mkdir(parents=True, exist_ok=True)

    gen = SyntheticGenerator()
    main_a, starless_a = gen.generate({**PARAMS_A, "output_dir": str(dir_a)}, preview=False)
    main_b, starless_b = gen.generate({**PARAMS_B, "output_dir": str(dir_b)}, preview=False)

    # A copy with no "_starless" sibling on disk, to trigger ImagePanel's manual
    # starless prompt (auto-detect only skips it when a "<stem>_starless.<ext>"
    # companion is found next to the loaded file).
    main_lone = dir_lone / pathlib.Path(main_a).name
    shutil.copy(main_a, main_lone)

    return {
        "main_a": main_a, "starless_a": starless_a,
        "main_b": main_b, "starless_b": starless_b,
        "main_lone": str(main_lone),
        "report_dir": str(dir_report),
    }


# ---------------------------------------------------------------------------
# Static window/panel/dialog states
# ---------------------------------------------------------------------------

def capture_static_states(app: QApplication, data: dict) -> MainWindow:
    mw = MainWindow()
    grab(mw, RESOURCES / "01_main_window.png")

    mw._panel_a.load_path(data["main_a"])
    mw._panel_a.set_starless_path(data["starless_a"])
    for _ in range(3):
        app.processEvents()
    grab(mw._panel_a, RESOURCES / "04_image_a_loaded.png")

    mw._panel_b.load_path(data["main_b"])
    mw._panel_b.set_starless_path(data["starless_b"])
    for _ in range(3):
        app.processEvents()
    grab(mw, RESOURCES / "AstroImageLabMain.png")

    # Cross-section line -- call the same handler the mouse-driven flow uses.
    mw._on_line_selected(0.18, 0.72, 0.60, 0.28)
    grab(mw, RESOURCES / "06_drawing_line.png")

    # ROI -- _on_roi_selected only stores state (gui/main_window.py); there is
    # no public setter that draws the overlay, so poke the same private
    # attribute ZoomableImageLabel.mouseReleaseEvent writes on a real drag.
    h, w = mw._panel_a.image.data.shape[:2]
    rx0, ry0, rx1, ry1 = int(0.15 * w), int(0.15 * h), int(0.85 * w), int(0.85 * h)
    mw._on_roi_selected(rx0, ry0, rx1, ry1)
    for panel in (mw._panel_a, mw._panel_b):
        panel._img_label._roi_norm = (0.15, 0.15, 0.85, 0.85)
        panel._img_label.update()
    grab(mw, RESOURCES / "08_drawing_roi.png")

    grab(find_groupbox(mw._control, "2. Parameters"), RESOURCES / "10_parameters.png")

    mw._control._out_dir.setText(data["report_dir"])
    metrics_box = find_groupbox(mw._control, "1. Metrics")
    region_box = find_groupbox(mw._control, "3. Region && Run")
    grab_side_by_side([metrics_box, region_box], RESOURCES / "11_metrics_output.png")

    # Starless prompt -- needs an image with no "_starless" sibling on disk.
    lone_panel = ImagePanel("Image A")
    lone_panel.load_path(data["main_lone"])
    arm_modal_capture(RESOURCES / "03_starless_prompt.png",
                       click_role=QMessageBox.StandardButton.No)
    lone_panel._ask_about_starless(lone_panel.image)  # blocks until timer clicks No

    return mw


def capture_tool_dialogs(app: QApplication, data: dict) -> None:
    dlg = SyntheticDialog()
    dlg.show()
    for _ in range(3):
        app.processEvents()
    grab(dlg, RESOURCES / "15_synthetic_generator.png")
    dlg.close()

    dlg2 = SpatialTargetDialog()
    dlg2.show()
    for _ in range(3):
        app.processEvents()
    grab(dlg2, RESOURCES / "16_spatial_target_generator.png")
    dlg2.close()

    img_a = AstroImage(data["main_a"], label="A")
    img_a.load()
    halo_dlg = HaloAnalyzerDialog(img_a, None, dark_mode=True)
    halo_done = {"flag": False}
    halo_dlg._detect_thread.finished.connect(lambda *a: halo_done.update(flag=True))
    pump(app, lambda: halo_done["flag"], timeout_s=60)

    # Click the brightest detected star so the screenshot shows a populated
    # results table and PSF/RDF chart rather than the blank pre-selection state.
    if halo_dlg._stars_a is not None and len(halo_dlg._stars_a) > 0:
        brightest = halo_dlg._stars_a[halo_dlg._stars_a[:, 2].argmax()]
        halo_dlg._on_star_clicked(float(brightest[0]), float(brightest[1]))
        analyze_done = {"flag": False}

        def _mark_done(*_a):
            analyze_done["flag"] = True

        pump(app, lambda: halo_dlg._analyze_thread is not None, timeout_s=10)
        halo_dlg._analyze_thread.finished.connect(_mark_done)
        pump(app, lambda: analyze_done["flag"], timeout_s=60)

    for _ in range(3):
        app.processEvents()
    grab(halo_dlg, RESOURCES / "17_halo_analyzer.png")
    halo_dlg.close()


# ---------------------------------------------------------------------------
# Real analysis run -> running / complete / inspector
# ---------------------------------------------------------------------------

def capture_running_and_report(app: QApplication, mw: MainWindow) -> None:
    # Avoid every pre-flight confirmation dialog in MainWindow._on_run so the
    # only modal that appears is the completion QMessageBox.
    mw._panel_a._bw_edit.setText("6")
    mw._panel_b._bw_edit.setText("12")

    mw._control._on_run()
    thread = mw._thread

    capture_state = {"count": 0}

    def _on_metric_started(_key: str) -> None:
        capture_state["count"] += 1
        if capture_state["count"] == 3:
            QTimer.singleShot(150, lambda: grab(mw, RESOURCES / "12_running.png"))

    thread.metric_started.connect(_on_metric_started)

    modal_state = arm_modal_capture(RESOURCES / "13_complete.png")
    pump(app, lambda: modal_state["done"], timeout_s=300)

    for _ in range(5):
        app.processEvents()
        time.sleep(0.05)
    grab(mw._data_inspector, RESOURCES / "14_inspector.png")


def main() -> None:
    app = QApplication(sys.argv)

    print("Generating synthetic sample data (two full-resolution images, ~1-2 min)...")
    data = build_sample_data()

    print("Capturing static window/panel/parameter states...")
    mw = capture_static_states(app, data)

    print("Capturing Tools-menu dialogs...")
    capture_tool_dialogs(app, data)

    print("Running a full analysis for running/complete/inspector captures...")
    capture_running_and_report(app, mw)

    print("Done. Screenshots regenerated in resources/.")


if __name__ == "__main__":
    main()
