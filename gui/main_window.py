from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox, QFileDialog,
)

from gui.image_panel import ImagePanel
from gui.control_panel import AnalysisControlPanel
from gui.analysis_thread import AnalysisThread


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Astro Image Lab")
        self.resize(1400, 900)

        self._thread: AnalysisThread | None = None
        self._roi: tuple | None = None
        self._crosshair: dict | None = None

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Image panels side by side
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._panel_a = ImagePanel("Image A")
        self._panel_b = ImagePanel("Image B")
        splitter.addWidget(self._panel_a)
        splitter.addWidget(self._panel_b)
        splitter.setSizes([600, 600])
        main_layout.addWidget(splitter, stretch=1)

        # Control panel below images
        self._control = AnalysisControlPanel()
        self._control.setMaximumHeight(240)
        main_layout.addWidget(self._control)

        # Wire signals
        self._panel_a.image_loaded.connect(self._on_image_loaded)
        self._panel_b.image_loaded.connect(self._on_image_loaded)
        self._panel_a.roi_selected.connect(self._on_roi_selected)
        self._panel_b.roi_selected.connect(self._on_roi_selected)
        self._panel_a.line_selected.connect(self._on_line_selected)
        self._panel_b.line_selected.connect(self._on_line_selected)

        self._control.run_requested.connect(self._on_run)
        self._control.roi_mode_toggled.connect(self._on_roi_mode_toggled)
        self._control.line_mode_toggled.connect(self._on_line_mode_toggled)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_open_a = QAction("Open Image &A…", self)
        act_open_a.triggered.connect(self._panel_a._open_file)
        file_menu.addAction(act_open_a)

        act_open_b = QAction("Open Image &B…", self)
        act_open_b.triggered.connect(self._panel_b._open_file)
        file_menu.addAction(act_open_b)

        file_menu.addSeparator()
        act_open_inspector = QAction("Open Report &Inspector…", self)
        act_open_inspector.triggered.connect(self._open_inspector)
        file_menu.addAction(act_open_inspector)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        analysis_menu = mb.addMenu("&Analysis")
        act_run = QAction("&Run Analysis", self)
        act_run.triggered.connect(lambda: self._control._on_run())
        analysis_menu.addAction(act_run)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_image_loaded(self, img) -> None:
        both_loaded = (self._panel_a.image is not None and
                       self._panel_b.image is not None)
        self._control.set_run_enabled(both_loaded)
        if both_loaded:
            self._control.set_alignment_status("Waiting for analysis…", ok=True)

    def _on_roi_mode_toggled(self, enabled: bool) -> None:
        self._panel_a.set_roi_mode(enabled)
        self._panel_b.set_roi_mode(enabled)
        if not enabled:
            self._control.set_roi(self._roi)

    def _on_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._roi = (x0, y0, x1, y1)
        self._control.set_roi(self._roi)
        # Turn off ROI mode automatically after selection
        self._control._roi_btn.setChecked(False)
        self._on_roi_mode_toggled(False)

    def _on_line_mode_toggled(self, enabled: bool) -> None:
        self._panel_a.set_line_mode(enabled)
        self._panel_b.set_line_mode(enabled)

    def _on_line_selected(self, x0n: float, y0n: float,
                           x1n: float, y1n: float) -> None:
        self._crosshair = {"x0": x0n, "y0": y0n, "x1": x1n, "y1": y1n}
        self._control.set_line(self._crosshair)
        self._panel_a._img_label.set_line_normalised(x0n, y0n, x1n, y1n)
        self._panel_b._img_label.set_line_normalised(x0n, y0n, x1n, y1n)
        self._control._line_btn.setChecked(False)
        self._on_line_mode_toggled(False)

        if self._roi is not None:
            img = self._panel_a.image or self._panel_b.image
            if img is not None:
                H, W = img.data.shape[:2]
                rx0, ry0, rx1, ry1 = self._roi
                lx0, ly0 = x0n * W, y0n * H
                lx1, ly1 = x1n * W, y1n * H
                if not (rx0 <= lx0 <= rx1 and ry0 <= ly0 <= ry1 and
                        rx0 <= lx1 <= rx1 and ry0 <= ly1 <= ry1):
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self, "Cross-section outside ROI",
                        "The drawn cross-section extends outside the selected ROI.\n\n"
                        "Section 8 (Spatial Detail) profiles sample derived maps in ROI-relative "
                        "coordinates. A line that extends beyond the ROI will be clipped to the "
                        "ROI boundary for those subsections.\n\n"
                        "Consider redrawing the line entirely within the ROI, or clear the ROI."
                    )

    def _on_run(self, settings: dict) -> None:
        img_a = self._panel_a.image
        img_b = self._panel_b.image
        if img_a is None or img_b is None:
            QMessageBox.warning(self, "Missing images",
                                "Please load both Image A and Image B before running.")
            self._control.set_run_enabled(True)
            return

        # Push bandwidth and filter thickness values from the input fields into the image objects
        self._panel_a.apply_bandwidth_from_field()
        self._panel_b.apply_bandwidth_from_field()
        self._panel_a.apply_filter_thickness_from_field()
        self._panel_b.apply_filter_thickness_from_field()

        # Warn if either image has no bandwidth set
        missing = [img.label for img in (img_a, img_b) if img.bandwidth_nm is None]
        if missing:
            answer = QMessageBox.question(
                self,
                "Bandwidth not specified",
                f"No filter bandwidth is set for: {', '.join(missing)}.\n\n"
                "Edge contrast ratio and power spectrum results are bandwidth-sensitive. "
                "Proceed without this information?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.No:
                self._control.set_run_enabled(True)
                return

        # Warn if no cross-section line has been drawn
        if self._crosshair is None:
            answer = QMessageBox.question(
                self, "No cross-section line selected",
                "No cross-section line has been drawn on the images.\n\n"
                "Without a cross-section:\n"
                "  • Spatial Detail section will show maps only — no profile overlays\n"
                "  • Edge Analysis will auto-detect the strongest gradient\n\n"
                "Proceed without a cross-section line?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._control.set_run_enabled(True)
                return

        # Merge ROI and crosshair from window state into settings
        settings["roi"] = self._roi
        settings["crosshair"] = self._crosshair

        self._thread = AnalysisThread(
            img_a, img_b, settings,
            starless_a=self._panel_a.starless_image,
            starless_b=self._panel_b.starless_image,
            parent=self,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._control.update_progress(pct, msg)
        if "align" in msg.lower():
            ok = "fail" not in msg.lower()
            self._control.set_alignment_status(
                "Aligned ✓" if ok else "Alignment failed", ok=ok)

    def _on_finished(self, result_a, result_b, report_path: str) -> None:
        self._control.reset_progress()
        self._control.set_run_enabled(True)
        msg = "Analysis complete."
        if report_path:
            msg += f"\nReport saved to:\n{report_path}"
        if result_a.warnings:
            msg += "\n\nWarnings:\n" + "\n".join(f"• {w[:200]}" for w in result_a.warnings)
        QMessageBox.information(self, "Done", msg)

        if report_path:
            from pathlib import Path as _Path
            from gui.report_inspector import ReportInspector
            npz_path = _Path(report_path).with_name(
                _Path(report_path).stem + "_inspector.npz")
            if npz_path.exists():
                self._inspector = ReportInspector(npz_path, parent=self)
                self._inspector.show()

    def _open_inspector(self) -> None:
        from pathlib import Path as _Path
        from gui.report_inspector import ReportInspector
        start_dir = self._control.settings().get("output_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Report Inspector",
            start_dir,
            "Astro Image Lab Report (*.html *_inspector.npz);;All files (*)",
        )
        if not path:
            return
        p = _Path(path)
        if p.suffix.lower() == ".html":
            npz_path = p.with_name(p.stem + "_inspector.npz")
        else:
            npz_path = p
        if not npz_path.exists():
            QMessageBox.warning(
                self,
                "Inspector file not found",
                f"No inspector data file found for this report.\n\n"
                f"Expected: {npz_path.name}\n\n"
                "Run the analysis again to regenerate the inspector file.",
            )
            return
        self._inspector = ReportInspector(npz_path, parent=self)
        self._inspector.show()

    def _on_error(self, msg: str) -> None:
        self._control.reset_progress()
        self._control.set_run_enabled(True)
        QMessageBox.critical(self, "Analysis error", msg)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Filter Image Comparator",
            "<b>Filter Image Comparator</b><br>"
            "Astrophotography narrowband filter characterisation tool.<br><br>"
            "Metrics: PSF/MTF · Halo · Ghost · Edge · Power spectrum · "
            "Spatial detail (std / LoG / wavelet)<br><br>"
            "Supports FITS and XISF input formats.",
        )
