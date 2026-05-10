from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QLabel, QLineEdit, QPushButton, QProgressBar, QFileDialog,
    QFormLayout, QDoubleSpinBox, QSpinBox, QSizePolicy,
)

from core.models import (
    STD_KERNEL_SIZES, LOG_SIGMAS, WAVELET_LEVELS, DEFAULT_PIXEL_SCALE,
    MIN_STAR_SNR, SEEING_WARN_FWHM_ARCS,
)


class AnalysisControlPanel(QWidget):
    """Bottom panel: metric selection, parameters, ROI, output dir, run button."""

    run_requested = pyqtSignal(dict)   # emits settings dict
    roi_mode_toggled = pyqtSignal(bool)
    line_mode_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._roi: tuple | None = None
        self._line: dict | None = None
        self._elapsed_seconds: int = 0
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._on_timer_tick)
        self._build_ui()
        # Restore last used output directory
        saved = QSettings("FilterImageComparator", "FilterImageComparator").value(
            "last_output_dir", "")
        if saved:
            self._out_dir.setText(saved)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Metrics group ──────────────────────────────────────────────
        metrics_box = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        self._checks: dict[str, QCheckBox] = {}
        for key, label in [
            ("psf",     "PSF / MTF"),
            ("halo",    "Halo analysis"),
            ("ghost",   "Ghost detection"),
            ("edge",    "Edge analysis (LSF)"),
            ("power",   "Power spectrum"),
            ("spatial", "Spatial detail (std / LoG / wavelet)"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            metrics_layout.addWidget(cb)
            self._checks[key] = cb

        root.addWidget(metrics_box)

        # ── Parameters group ───────────────────────────────────────────
        params_box = QGroupBox("Parameters")
        params_layout = QFormLayout(params_box)

        self._min_snr = QDoubleSpinBox()
        self._min_snr.setRange(5.0, 500.0)
        self._min_snr.setValue(MIN_STAR_SNR)
        params_layout.addRow("Min star S/N:", self._min_snr)

        self._seeing_thresh = QDoubleSpinBox()
        self._seeing_thresh.setRange(0.5, 10.0)
        self._seeing_thresh.setSingleStep(0.5)
        self._seeing_thresh.setValue(SEEING_WARN_FWHM_ARCS)
        self._seeing_thresh.setSuffix(" \"")
        params_layout.addRow("Seeing warn threshold:", self._seeing_thresh)

        self._pixel_scale_override = QDoubleSpinBox()
        self._pixel_scale_override.setRange(0.0, 20.0)
        self._pixel_scale_override.setDecimals(3)
        self._pixel_scale_override.setValue(0.0)
        self._pixel_scale_override.setSuffix(" \"/px")
        self._pixel_scale_override.setSpecialValueText("(from header)")
        params_layout.addRow("Pixel scale override:", self._pixel_scale_override)

        self._wavelet_levels = QSpinBox()
        self._wavelet_levels.setRange(2, 6)
        self._wavelet_levels.setValue(WAVELET_LEVELS)
        params_layout.addRow("Wavelet levels:", self._wavelet_levels)

        root.addWidget(params_box)

        # ── Output + ROI + Run ─────────────────────────────────────────
        run_box = QGroupBox("Output & Run")
        run_layout = QHBoxLayout(run_box)

        # ── Left column: all selection / status controls ────────────────
        left_col = QVBoxLayout()

        out_row = QHBoxLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Select output directory…")
        out_row.addWidget(self._out_dir)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(btn_browse)
        left_col.addLayout(out_row)

        # Cross-section line row — above ROI
        line_row = QHBoxLayout()
        self._line_btn = QPushButton("Select Line…")
        self._line_btn.setCheckable(True)
        self._line_btn.clicked.connect(self._toggle_line_mode)
        line_row.addWidget(self._line_btn)
        self._line_label = QLabel("No line — skip cross-section")
        self._line_label.setStyleSheet("color: #666;")
        line_row.addWidget(self._line_label)
        line_row.addStretch()
        left_col.addLayout(line_row)

        # ROI row — below line
        roi_row = QHBoxLayout()
        self._roi_btn = QPushButton("Select ROI…")
        self._roi_btn.setCheckable(True)
        self._roi_btn.clicked.connect(self._toggle_roi_mode)
        roi_row.addWidget(self._roi_btn)
        self._roi_label = QLabel("No ROI — auto-detect")
        self._roi_label.setStyleSheet("color: #666;")
        roi_row.addWidget(self._roi_label)
        roi_row.addStretch()
        left_col.addLayout(roi_row)

        align_row = QHBoxLayout()
        align_row.addWidget(QLabel("Alignment:"))
        self._align_label = QLabel("Waiting for images…")
        self._align_label.setStyleSheet("color: #666;")
        align_row.addWidget(self._align_label)
        align_row.addStretch()
        left_col.addLayout(align_row)

        self._parallel_cb = QCheckBox("Run metrics in parallel  (faster, uses more RAM)")
        self._parallel_cb.setChecked(False)
        self._parallel_cb.setToolTip(
            "When checked, all selected analysis metrics run concurrently in separate\n"
            "threads, which can significantly reduce total run time on multi-core CPUs.\n"
            "RAM usage increases because all analyses hold their working data at once.\n"
            "When unchecked, metrics run one at a time using less memory."
        )
        left_col.addWidget(self._parallel_cb)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        left_col.addWidget(self._progress)

        self._status_label = QLabel("Ready")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_col.addWidget(self._status_label)

        left_col.addStretch()
        run_layout.addLayout(left_col, stretch=3)

        # ── Right column: run button + elapsed timer ────────────────────
        right_col = QVBoxLayout()

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setEnabled(False)
        self._run_btn.setMinimumHeight(70)
        self._run_btn.setStyleSheet(
            "QPushButton { background: #2d6da3; color: white; font-weight: bold;"
            "padding: 8px 18px; border-radius: 4px; }"
            "QPushButton:disabled { background: #aaa; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        right_col.addWidget(self._run_btn)

        self._timer_label = QLabel("0:00")
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._timer_label.setStyleSheet(
            "font-family: monospace; font-size: 14pt; color: #444;")
        right_col.addWidget(self._timer_label)

        right_col.addStretch()
        run_layout.addLayout(right_col, stretch=1)

        root.addWidget(run_box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_run_enabled(self, enabled: bool) -> None:
        self._run_btn.setEnabled(enabled)

    def set_alignment_status(self, text: str, ok: bool = True) -> None:
        self._align_label.setText(text)
        color = "#155724" if ok else "#721c24"
        self._align_label.setStyleSheet(f"color: {color};")

    def set_roi(self, roi: tuple | None) -> None:
        self._roi = roi
        if roi:
            x0, y0, x1, y1 = roi
            self._roi_label.setText(f"ROI: ({x0},{y0}) → ({x1},{y1})")
            self._roi_label.setStyleSheet("color: #155724;")
        else:
            self._roi_label.setText("No ROI — auto-detect")
            self._roi_label.setStyleSheet("color: #666;")

    def set_line(self, line: dict | None) -> None:
        self._line = line
        if line:
            x0, y0 = line["x0"], line["y0"]
            x1, y1 = line["x1"], line["y1"]
            self._line_label.setText(f"Line: ({x0:.3f},{y0:.3f})→({x1:.3f},{y1:.3f})")
            self._line_label.setStyleSheet("color: #155724;")
        else:
            self._line_label.setText("No line — skip cross-section")
            self._line_label.setStyleSheet("color: #666;")

    def update_progress(self, pct: int, message: str = "") -> None:
        self._progress.setVisible(True)
        self._progress.setValue(pct)
        if message:
            self._status_label.setText(message)

    def reset_progress(self) -> None:
        self._run_timer.stop()
        self._progress.setVisible(False)
        self._progress.setValue(0)
        self._status_label.setText("Ready")

    def settings(self) -> dict:
        """Return all current settings as a dict for the analysis thread."""
        pso = self._pixel_scale_override.value()
        return {
            "metrics": {k: cb.isChecked() for k, cb in self._checks.items()},
            "min_snr": self._min_snr.value(),
            "seeing_warn_arcsec": self._seeing_thresh.value(),
            "pixel_scale_override": pso if pso > 0 else None,
            "wavelet_levels": self._wavelet_levels.value(),
            "roi": self._roi,
            "crosshair": self._line,
            "output_dir": self._out_dir.text().strip() or str(Path.home() / "filter_reports"),
            "parallel": self._parallel_cb.isChecked(),
        }

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select output directory", self._out_dir.text())
        if d:
            self._out_dir.setText(d)
            QSettings("FilterImageComparator", "FilterImageComparator").setValue(
                "last_output_dir", d)

    def _toggle_roi_mode(self, checked: bool) -> None:
        self._roi_btn.setText("Cancel ROI" if checked else "Select ROI…")
        self.roi_mode_toggled.emit(checked)

    def _toggle_line_mode(self, checked: bool) -> None:
        self._line_btn.setText("Cancel Line" if checked else "Select Line…")
        self.line_mode_toggled.emit(checked)

    def _on_run(self) -> None:
        self._run_btn.setEnabled(False)
        self._elapsed_seconds = 0
        self._timer_label.setText("0:00")
        self._run_timer.start()
        self._status_label.setText("Running…")
        out = self._out_dir.text().strip()
        if out:
            QSettings("FilterImageComparator", "FilterImageComparator").setValue(
                "last_output_dir", out)
        self.run_requested.emit(self.settings())

    def _on_timer_tick(self) -> None:
        self._elapsed_seconds += 1
        m, s = divmod(self._elapsed_seconds, 60)
        self._timer_label.setText(f"{m}:{s:02d}")
