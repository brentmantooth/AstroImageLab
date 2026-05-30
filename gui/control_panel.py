from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QLabel, QLineEdit, QPushButton, QProgressBar, QFileDialog,
    QFormLayout, QDoubleSpinBox, QSpinBox, QSizePolicy,
    QGridLayout, QComboBox,
)

from core.models import (
    STD_KERNEL_SIZES, LOG_SIGMAS, WAVELET_LEVELS, DEFAULT_PIXEL_SCALE,
    MIN_STAR_SNR, SEEING_WARN_FWHM_ARCS, REF_SEEING_ARCSEC, XS_SNR_REGION_WIDTH,
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
        metrics_layout = QGridLayout(metrics_box)
        metrics_layout.setColumnStretch(0, 1)
        metrics_layout.setColumnMinimumWidth(1, 46)   # Export column
        metrics_layout.setColumnMinimumWidth(2, 36)   # ROI column
        metrics_layout.setColumnMinimumWidth(3, 36)   # XS (cross-section) column
        metrics_layout.setColumnMinimumWidth(4, 52)   # Time column

        _hdr_style = "color: #444; font-size: 9pt;"
        hdr_export = QLabel("Export")
        hdr_export.setStyleSheet(_hdr_style)
        hdr_export.setToolTip("Export analysis images as independent .png files")
        metrics_layout.addWidget(hdr_export, 0, 1, Qt.AlignmentFlag.AlignHCenter)

        hdr_roi = QLabel("ROI")
        hdr_roi.setStyleSheet(_hdr_style)
        hdr_roi.setToolTip("Uses the user-drawn ROI region when one is selected")
        metrics_layout.addWidget(hdr_roi, 0, 2, Qt.AlignmentFlag.AlignHCenter)

        hdr_xs = QLabel("XS")
        hdr_xs.setStyleSheet(_hdr_style)
        hdr_xs.setToolTip("Uses the cross-section line when one is drawn")
        metrics_layout.addWidget(hdr_xs, 0, 3, Qt.AlignmentFlag.AlignHCenter)

        hdr_time = QLabel("Time")
        hdr_time.setStyleSheet(_hdr_style)
        hdr_time.setToolTip("Elapsed computation time for each metric")
        metrics_layout.addWidget(hdr_time, 0, 4, Qt.AlignmentFlag.AlignHCenter)

        self._checks: dict[str, QCheckBox] = {}
        self._export_checks: dict[str, QCheckBox] = {}
        self._time_labels: dict[str, QLabel] = {}
        self._metric_timers: dict[str, QTimer] = {}
        self._metric_t0: dict[str, float] = {}
        for row, (key, label, uses_roi, uses_xs) in enumerate([
            #                                                      ROI    XS
            ("snr",     "Signal / Noise (SNR)",                   False, False),
            ("psf",     "PSF / MTF",                              False, False),
            ("halo",    "Halo analysis",                          False, False),
            ("edge",    "Edge analysis (LSF)",                    True,  False),
            ("power",   "Power spectrum",                         True,  False),
            ("spatial", "Spatial detail (std / LoG / wavelet)",   True,  True),
        ], start=1):
            cb = QCheckBox(label)
            cb.setChecked(True)
            metrics_layout.addWidget(cb, row, 0)
            self._checks[key] = cb

            ecb = QCheckBox()
            ecb.setChecked(False)
            ecb.setToolTip(f"Export {label} figures as PNG files to the output folder")
            metrics_layout.addWidget(ecb, row, 1, Qt.AlignmentFlag.AlignHCenter)
            self._export_checks[key] = ecb

            for col, flag in ((2, uses_roi), (3, uses_xs)):
                ind = QLabel("●" if flag else "—")
                ind.setStyleSheet(
                    "color: #2d8a3e; font-size: 10pt;" if flag
                    else "color: #aaa; font-size: 10pt;"
                )
                metrics_layout.addWidget(ind, row, col, Qt.AlignmentFlag.AlignHCenter)

            tlbl = QLabel("—")
            tlbl.setStyleSheet("color: #888; font-size: 9pt;")
            tlbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tlbl.setFixedWidth(52)
            metrics_layout.addWidget(tlbl, row, 4, Qt.AlignmentFlag.AlignRight)
            self._time_labels[key] = tlbl

            timer = QTimer(self)
            timer.setInterval(250)
            timer.timeout.connect(lambda k=key: self._tick_metric_timer(k))
            self._metric_timers[key] = timer

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

        self._xs_snr_width = QSpinBox()
        self._xs_snr_width.setRange(3, 100)
        self._xs_snr_width.setValue(XS_SNR_REGION_WIDTH)
        self._xs_snr_width.setToolTip(
            "Width (px) of the bright/dark sample windows used to compute\n"
            "cross-section SNR in the Spatial Detail section."
        )
        params_layout.addRow("XS SNR region width:", self._xs_snr_width)

        self._ref_seeing_arcsec = QDoubleSpinBox()
        self._ref_seeing_arcsec.setRange(0.5, 10.0)
        self._ref_seeing_arcsec.setSingleStep(0.25)
        self._ref_seeing_arcsec.setDecimals(2)
        self._ref_seeing_arcsec.setValue(REF_SEEING_ARCSEC)
        self._ref_seeing_arcsec.setSuffix(" \"")
        self._ref_seeing_arcsec.setToolTip(
            "FWHM of the synthetic reference PSF shown in the PSF/MTF report section.\n"
            "2.0\" represents typical good ground-based seeing (Kolmogorov atmosphere, β = 4.77)."
        )
        params_layout.addRow("PSF reference seeing:", self._ref_seeing_arcsec)

        root.addWidget(params_box)

        # ── Output + ROI + Run ─────────────────────────────────────────
        run_box = QGroupBox("Output & Run")
        run_layout = QHBoxLayout(run_box)

        # ── Left column: all selection / status controls ────────────────
        left_col = QVBoxLayout()

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Report format:"))
        self._report_fmt = QComboBox()
        self._report_fmt.addItems(["HTML", "PDF"])
        self._report_fmt.setToolTip(
            "PDF requires WeasyPrint (pip install weasyprint).\n"
            "If unavailable the report is saved as HTML automatically."
        )
        fmt_row.addWidget(self._report_fmt)
        fmt_row.addStretch()
        left_col.addLayout(fmt_row)

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

    def reset_metric_timers(self) -> None:
        for key in self._time_labels:
            self._metric_timers[key].stop()
            self._time_labels[key].setText("—")
            self._time_labels[key].setStyleSheet("color: #888; font-size: 9pt;")
        self._metric_t0.clear()

    def on_metric_started(self, key: str) -> None:
        if key not in self._time_labels:
            return
        self._metric_t0[key] = time.monotonic()
        self._time_labels[key].setText("0.0s")
        self._time_labels[key].setStyleSheet("color: #aac8ff; font-size: 9pt;")
        self._metric_timers[key].start()

    def _tick_metric_timer(self, key: str) -> None:
        if key not in self._metric_t0:
            return
        elapsed = time.monotonic() - self._metric_t0[key]
        self._time_labels[key].setText(f"{elapsed:.1f}s")

    def on_metric_done(self, key: str, elapsed: float, success: bool) -> None:
        if key not in self._time_labels:
            return
        self._metric_timers[key].stop()
        self._time_labels[key].setText(f"{elapsed:.1f}s")
        color = "#88dd88" if success else "#ff8888"
        self._time_labels[key].setStyleSheet(f"color: {color}; font-size: 9pt;")

    def settings(self) -> dict:
        """Return all current settings as a dict for the analysis thread."""
        pso = self._pixel_scale_override.value()
        return {
            "metrics": {k: cb.isChecked() for k, cb in self._checks.items()},
            "export_figures": {k: cb.isChecked() for k, cb in self._export_checks.items()},
            "report_format": self._report_fmt.currentText().lower(),
            "min_snr": self._min_snr.value(),
            "seeing_warn_arcsec": self._seeing_thresh.value(),
            "pixel_scale_override": pso if pso > 0 else None,
            "wavelet_levels": self._wavelet_levels.value(),
            "xs_snr_width": self._xs_snr_width.value(),
            "ref_seeing_arcsec": self._ref_seeing_arcsec.value(),
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
        self.reset_metric_timers()
        out = self._out_dir.text().strip()
        if out:
            QSettings("FilterImageComparator", "FilterImageComparator").setValue(
                "last_output_dir", out)
        self.run_requested.emit(self.settings())

    def _on_timer_tick(self) -> None:
        self._elapsed_seconds += 1
        m, s = divmod(self._elapsed_seconds, 60)
        self._timer_label.setText(f"{m}:{s:02d}")
