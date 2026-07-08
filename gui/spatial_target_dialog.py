"""Spatial Detail Test Target Generator dialog.

Generates a 4×3 grid of calibrated spatial test patterns with known frequency
content, enabling calibrated interpretation of the app's spatial-detail metrics.

Workflow:
  1. Configure contrast, sky level, and optional PSF / noise.
  2. Click Generate → produces a clean reference (no PSF/noise) and a
     degraded companion (with configured blur and noise).
  3. Auto-load maps clean → Image A, degraded → Image B (configurable).
  4. Run the analysis; compare wavelet SNR, local-std ratios, and LoG maps
     against the known zone frequencies to calibrate your metric intuition.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QTextBrowser, QVBoxLayout, QWidget,
)

from core.stretch import normalize_for_display
from synthetic.target_generator import SpatialTargetGenerator, ZONE_LABELS
from gui.synthetic_dialog import _SliderRow


# ---------------------------------------------------------------------------
# Zone layout info HTML (shown in the ⓘ dialog)
# ---------------------------------------------------------------------------

_ZONE_INFO_HTML = """
<h2>Spatial Detail Test Target — Zone Layout</h2>

<p>The target is a 4-column × 3-row grid of calibrated test zones.
Each zone has a known waveform type and spatial frequency.
All zones share the same Michelson contrast and DC sky level.</p>

<h3>Zone Grid</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Row</th><th>Col 0</th><th>Col 1</th><th>Col 2</th><th>Col 3</th></tr>
<tr><td><b>0</b></td>
    <td>Sine H<br>f=0.04 c/px</td>
    <td>Sine H<br>f=0.08 c/px</td>
    <td>Sine H<br>f=0.16 c/px</td>
    <td>Sine H<br>f=0.32 c/px</td></tr>
<tr><td><b>1</b></td>
    <td>Square H<br>f=0.04 c/px</td>
    <td>Square H<br>f=0.08 c/px</td>
    <td>Square H<br>f=0.16 c/px</td>
    <td>Square H<br>f=0.32 c/px</td></tr>
<tr><td><b>2</b></td>
    <td>Sine V<br>f=0.08 c/px</td>
    <td>Sine 45°<br>f=0.08 c/px</td>
    <td>Siemens star<br>(radial chirp)</td>
    <td>Slant edge<br>(~5° from vertical)</td></tr>
</table>

<h3>Frequency–Wavelet Level Correspondence</h3>
<p>The four column frequencies are chosen to align with the app's 4-level
wavelet decomposition band centres:</p>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Column</th><th>Frequency (c/px)</th><th>Wavelength (px)</th>
    <th>Wavelet level</th><th>Approx scale</th></tr>
<tr><td>0</td><td>0.04</td><td>25 px</td><td>4 (coarsest)</td><td>~16 px</td></tr>
<tr><td>1</td><td>0.08</td><td>12 px</td><td>3</td><td>~8 px</td></tr>
<tr><td>2</td><td>0.16</td><td>6 px</td><td>2</td><td>~4 px</td></tr>
<tr><td>3</td><td>0.32</td><td>3 px</td><td>1 (finest / near Nyquist)</td><td>~2 px</td></tr>
</table>

<h3>Contrast Ramp</h3>
<p>Each zone ramps Michelson contrast <strong>linearly from top to bottom</strong>:
the top edge has <em>Contrast min</em> and the bottom edge has <em>Contrast max</em>.
Within any horizontal strip across a row, all four columns share the same contrast —
so you can directly compare how different frequencies or waveform types respond at
identical contrast levels.  The ramp restarts at each zone row, so row 0 (sine),
row 1 (square), and row 2 (special) all independently cover the full min→max range.</p>

<h3>How to Use the Target</h3>
<ol>
<li><b>Clean reference (Image A):</b> All zones have perfect contrast at their
    design frequency. Wavelet SNR for zone (0,0) should be high at level 4 and
    near zero at levels 1–3.  Local std at 31 px should respond strongly to the
    f=0.04 zone. Use this to establish the ceiling for each metric.</li>
<li><b>Degraded companion (Image B):</b> After PSF convolution, high-frequency
    zones lose contrast first. The f=0.32 zone (near Nyquist) should show
    near-zero contrast even for moderate FWHM.  Wavelet level 1 SNR drops to
    near 1.0 (noise floor) when the PSF FWHM exceeds ~3 px.</li>
<li><b>Michelson contrast vs frequency:</b> To measure MTF at each frequency,
    load the clean and degraded images, measure local std or wavelet SNR in the
    sine zone for each column.  Ratio (degraded / clean) gives the MTF at that
    spatial frequency.</li>
</ol>

<h3>Waveform Types</h3>
<dl>
<dt><b>Sine waves</b></dt>
<dd>Pure single-frequency content.  After PSF convolution the Michelson contrast
in the zone equals the MTF at that frequency.  Use these for the cleanest
frequency-response measurements.</dd>
<dt><b>Square waves</b></dt>
<dd>Contain all odd harmonics (f, 3f, 5f, …).  PSF blur first kills the high
harmonics, making the square look trapezoidal then sinusoidal.  Gibbs ringing
artifacts from narrow-bandpass filters appear here but not in the sine zones.</dd>
<dt><b>Siemens star</b></dt>
<dd>Radial sinusoidal pattern that sweeps from low frequency at the edge to
very high frequency at the centre (where the pattern aliases).  Shows all
spatial frequencies and all orientations in one zone.  The radius at which
contrast fades to noise is the resolution limit.</dd>
<dt><b>Slant edge (~5° from vertical)</b></dt>
<dd>ISO 12233-style step function.  The slight tilt oversamples the edge
spread function (ESF) by ~1/sin(5°) ≈ 11×, enabling sub-pixel MTF measurement.
After PSF convolution, the app's edge analyzer can extract the MTF from this
zone directly.</dd>
</dl>
"""


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

class _TargetThread(QThread):
    finished = pyqtSignal(str, str)   # clean_path, degraded_path
    failed   = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    def run(self):
        try:
            gen = SpatialTargetGenerator()
            clean_path, degraded_path = gen.generate(self._params, preview=False)
            self.finished.emit(clean_path, degraded_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class _PreviewThread(QThread):
    finished = pyqtSignal(object)   # np.ndarray

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    def run(self):
        try:
            gen = SpatialTargetGenerator()
            arr = gen.generate(self._params, preview=True)
            self.finished.emit(arr)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Info dialog
# ---------------------------------------------------------------------------

class _ZoneInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Detail Target — Zone Layout & Interpretation")
        self.setMinimumSize(680, 540)
        lay = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(_ZONE_INFO_HTML)
        browser.setOpenExternalLinks(False)
        lay.addWidget(browser)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.accept)
        lay.addWidget(btns)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIZE_PRESETS = [
    ("512 × 384  (fast)",       512,  384),
    ("1024 × 768  (default)",  1024,  768),
    ("2048 × 1536  (slow)",    2048, 1536),
]

_LOAD_MODES = [
    ("Clean → A,  Degraded → B", "clean_a_deg_b"),
    ("Degraded → A,  Clean → B", "deg_a_clean_b"),
    ("Degraded → A only",        "deg_a"),
    ("Degraded → B only",        "deg_b"),
    ("Don't load",               ""),
]


# ---------------------------------------------------------------------------
# Main dialog window
# ---------------------------------------------------------------------------

class SpatialTargetDialog(QMainWindow):
    """Standalone window for generating spatial detail test targets."""

    # clean_path, degraded_path, load_mode string (see _LOAD_MODES)
    targets_generated = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AstroImageLab — Spatial Detail Test Target")
        self.setMinimumSize(820, 520)

        self._gen_thread:  _TargetThread  | None = None
        self._prev_thread: _PreviewThread | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._run_preview)

        self._build_ui()
        self._preview_cb.setChecked(True)

        # Restore persisted output directory
        from PyQt6.QtCore import QSettings
        saved = QSettings("FilterImageComparator",
                          "FilterImageComparator").value("target_output_dir", "")
        if saved and os.path.isdir(saved):
            self._outdir_lbl.setText(saved)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._build_pattern_setup(),  stretch=1)
        top.addWidget(self._build_degradation(),    stretch=1)
        top.addWidget(self._build_output(),         stretch=1)
        root.addLayout(top)
        root.addWidget(self._build_preview_section())

    def _build_pattern_setup(self) -> QGroupBox:
        box  = QGroupBox("Pattern Setup")
        form = QFormLayout(box)
        form.setVerticalSpacing(5)

        # Image size preset combo
        self._size_combo = QComboBox()
        for label, *_ in _SIZE_PRESETS:
            self._size_combo.addItem(label)
        self._size_combo.setCurrentIndex(1)   # 1024×768 default
        self._size_combo.setToolTip(
            "Image dimensions for the generated target.\n"
            "Larger images show more cycles at low frequencies but take longer to generate.")
        self._size_combo.currentIndexChanged.connect(self._schedule_preview)
        form.addRow("Image size", self._size_combo)

        # Contrast ramp — min (top of zone) and max (bottom of zone)
        self._contrast_min_sl = _SliderRow(0.00, 1.0, 0.02, 2)
        self._contrast_min_sl.setToolTip(
            "Michelson contrast at the TOP edge of each zone.\n\n"
            "C = (I_max − I_min) / (I_max + I_min)\n\n"
            "Setting this near 0 means the top rows of each zone are nearly flat\n"
            "(pattern invisible) — this shows the detection floor for each metric.\n"
            "Contrast increases linearly toward the bottom (Contrast max).")
        self._contrast_min_sl.valueChanged.connect(self._schedule_preview)
        form.addRow("Contrast min  (top)", self._contrast_min_sl)

        self._contrast_max_sl = _SliderRow(0.00, 1.0, 0.50, 2)
        self._contrast_max_sl.setToolTip(
            "Michelson contrast at the BOTTOM edge of each zone.\n\n"
            "C = (I_max − I_min) / (I_max + I_min)\n\n"
            "Within any horizontal strip across a row, all four columns share\n"
            "the same contrast — enabling direct comparison across frequencies\n"
            "or waveform types at identical contrast levels.\n\n"
            "Typical range: min=0.02, max=0.50 covers the full range from just\n"
            "above noise floor to clearly-visible nebula structure.")
        self._contrast_max_sl.valueChanged.connect(self._schedule_preview)
        form.addRow("Contrast max  (bottom)", self._contrast_max_sl)

        # Sky background ADU
        self._sky_spin = QDoubleSpinBox()
        self._sky_spin.setRange(10, 50_000)
        self._sky_spin.setValue(500)
        self._sky_spin.setDecimals(0)
        self._sky_spin.setSuffix(" ADU")
        self._sky_spin.setToolTip(
            "DC background level in ADU.  Zone patterns are ±amplitude offsets\n"
            "around this value where amplitude = sky × Michelson contrast.\n\n"
            "Use a value comparable to your actual sky background so the wavelet\n"
            "SNR and local-std metrics operate in a similar numerical regime to\n"
            "real images.  Typical values: 200–2000 ADU.")
        self._sky_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Sky DC level", self._sky_spin)

        # Info button
        info_btn = QPushButton("ⓘ  Zone layout & interpretation…")
        info_btn.setToolTip(
            "Open a description of what each zone measures and how to\n"
            "interpret the spatial-detail metrics against these test patterns.")
        info_btn.clicked.connect(self._show_info)
        form.addRow(info_btn)

        return box

    def _build_degradation(self) -> QGroupBox:
        box  = QGroupBox("Degradation  (applies to degraded image only)")
        form = QFormLayout(box)
        form.setVerticalSpacing(5)

        # PSF section
        self._psf_cb = QCheckBox("Apply PSF blur")
        self._psf_cb.setToolTip(
            "Convolve the target with an isotropic Moffat PSF.\n"
            "Models atmospheric seeing without field-dependent aberrations.\n\n"
            "After blur, high-frequency zones lose contrast first:\n"
            "  f=0.32 zone loses ~50 % contrast when FWHM ≈ 1 px\n"
            "  f=0.08 zone loses ~50 % contrast when FWHM ≈ 4 px\n"
            "The ratio (blurred contrast / clean contrast) = MTF at that frequency.")
        self._psf_cb.toggled.connect(self._on_psf_toggled)
        self._psf_cb.toggled.connect(self._schedule_preview)
        form.addRow(self._psf_cb)

        self._fwhm_spin = QDoubleSpinBox()
        self._fwhm_spin.setRange(0.5, 50.0)
        self._fwhm_spin.setValue(4.0)
        self._fwhm_spin.setSuffix(" px")
        self._fwhm_spin.setSingleStep(0.5)
        self._fwhm_spin.setDecimals(1)
        self._fwhm_spin.setEnabled(False)
        self._fwhm_spin.setToolTip(
            "PSF full-width at half-maximum in pixels.\n"
            "Match this to the star FWHM measured in your actual images\n"
            "so the simulated blur matches a realistic observation.")
        self._fwhm_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("FWHM", self._fwhm_spin)

        self._beta_spin = QDoubleSpinBox()
        self._beta_spin.setRange(1.5, 10.0)
        self._beta_spin.setValue(4.77)
        self._beta_spin.setSingleStep(0.1)
        self._beta_spin.setDecimals(2)
        self._beta_spin.setEnabled(False)
        self._beta_spin.setToolTip(
            "Moffat β — controls the power-law slope of the PSF wings.\n"
            "4.77 is the empirical mean for ground-based seeing (Trujillo et al. 2001).\n"
            "Higher β → tighter wings (more Gaussian); lower → broader halos.")
        self._beta_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Moffat β", self._beta_spin)

        form.addRow(QLabel(""))   # visual spacer

        # Noise section
        self._noise_cb = QCheckBox("Apply noise")
        self._noise_cb.setToolTip(
            "Add Poisson sky shot noise and Gaussian read noise.\n"
            "The clean reference is always noise-free so the effect of\n"
            "noise on each metric can be measured independently of PSF blur.\n\n"
            "When both PSF blur and noise are enabled, the degraded image\n"
            "matches a realistic single-sub-frame observation.")
        self._noise_cb.toggled.connect(self._on_noise_toggled)
        self._noise_cb.toggled.connect(self._schedule_preview)
        form.addRow(self._noise_cb)

        self._rn_spin = QDoubleSpinBox()
        self._rn_spin.setRange(0.0, 500.0)
        self._rn_spin.setValue(10.0)
        self._rn_spin.setSuffix(" ADU")
        self._rn_spin.setSingleStep(1.0)
        self._rn_spin.setDecimals(1)
        self._rn_spin.setEnabled(False)
        self._rn_spin.setToolTip(
            "Camera read noise σ in ADU (Gaussian component).\n"
            "Sky Poisson noise is added automatically from the sky DC level.\n"
            "Typical values: 2–20 ADU.  At 500 ADU sky and 10 ADU read noise,\n"
            "sky shot noise (√500 ≈ 22 ADU) dominates.")
        self._rn_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Read noise", self._rn_spin)

        return box

    def _on_psf_toggled(self, checked: bool) -> None:
        self._fwhm_spin.setEnabled(checked)
        self._beta_spin.setEnabled(checked)

    def _on_noise_toggled(self, checked: bool) -> None:
        self._rn_spin.setEnabled(checked)

    def _build_output(self) -> QGroupBox:
        box = QGroupBox("Output")
        lay = QVBoxLayout(box)

        dir_row = QHBoxLayout()
        self._outdir_lbl = QLabel(str(Path.home()))
        self._outdir_lbl.setWordWrap(True)
        self._outdir_lbl.setStyleSheet("font-size: 10px; color: #444;")
        dir_btn = QPushButton("Browse…")
        dir_btn.setToolTip("Choose the folder where generated FITS files will be saved.")
        dir_btn.clicked.connect(self._browse_output)
        dir_row.addWidget(self._outdir_lbl, stretch=1)
        dir_row.addWidget(dir_btn)
        lay.addLayout(dir_row)

        # Auto-load combo
        self._load_combo = QComboBox()
        for label, mode in _LOAD_MODES:
            self._load_combo.addItem(label, mode)
        self._load_combo.setCurrentIndex(0)   # default: clean→A, degraded→B
        self._load_combo.setToolTip(
            "Which panel(s) to auto-load after generation.\n\n"
            "The recommended workflow is Clean → A, Degraded → B so you can\n"
            "run the analysis and see exactly how each metric responds to the\n"
            "known degradation you configured.")
        lay.addWidget(self._load_combo)

        lay.addStretch()

        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setDefault(True)
        self._gen_btn.setFixedHeight(34)
        self._gen_btn.setToolTip(
            "Generate the clean reference and degraded companion FITS files.\n\n"
            "Always produces two files:\n"
            "  • target_…_clean.fits   — pure pattern, no PSF/noise\n"
            "  • target_…_degraded.fits — same pattern with configured degradation\n\n"
            "Load both into Image A and B, then run the analysis to calibrate\n"
            "your intuition for the spatial-detail metrics.")
        self._gen_btn.clicked.connect(self._on_generate)
        lay.addWidget(self._gen_btn)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("font-size: 10px; color: #555;")
        lay.addWidget(self._status_lbl)

        return box

    def _build_preview_section(self) -> QGroupBox:
        box = QGroupBox("Live Preview  (degraded image)")
        lay = QHBoxLayout(box)

        left = QVBoxLayout()
        self._preview_cb = QCheckBox("Enable live preview")
        self._preview_cb.setToolTip(
            "Regenerates a reduced-resolution preview of the degraded target\n"
            "~400 ms after any parameter change.")
        self._preview_cb.toggled.connect(self._on_preview_toggled)
        left.addWidget(self._preview_cb)
        left.addStretch()
        lay.addLayout(left)

        self._preview_lbl = QLabel("Enable preview to see a live update")
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setStyleSheet("background: #1a1a1a; color: #666;")
        self._preview_lbl.setMinimumSize(320, 160)
        self._preview_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay.addWidget(self._preview_lbl, stretch=1)

        return box

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _show_info(self) -> None:
        dlg = _ZoneInfoDialog(self)
        dlg.exec()

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select output directory", self._outdir_lbl.text())
        if d:
            self._outdir_lbl.setText(d)
            from PyQt6.QtCore import QSettings
            QSettings("FilterImageComparator", "FilterImageComparator").setValue(
                "target_output_dir", d)

    def _on_generate(self) -> None:
        if self._gen_thread and self._gen_thread.isRunning():
            return
        params = self._collect_params()
        os.makedirs(params["output_dir"], exist_ok=True)
        from PyQt6.QtCore import QSettings
        QSettings("FilterImageComparator", "FilterImageComparator").setValue(
            "target_output_dir", params["output_dir"])
        self._gen_btn.setEnabled(False)
        self._status_lbl.setText("Generating…")
        self._gen_thread = _TargetThread(params, self)
        self._gen_thread.finished.connect(self._on_gen_done)
        self._gen_thread.failed.connect(self._on_gen_failed)
        self._gen_thread.start()

    @pyqtSlot(str, str)
    def _on_gen_done(self, clean_path: str, degraded_path: str) -> None:
        self._gen_btn.setEnabled(True)
        self._status_lbl.setText(f"Saved: {Path(clean_path).name}")
        mode = self._load_combo.currentData()
        self.targets_generated.emit(clean_path, degraded_path, mode or "")

    @pyqtSlot(str)
    def _on_gen_failed(self, msg: str) -> None:
        self._gen_btn.setEnabled(True)
        self._status_lbl.setText(f"Error: {msg}")

    def _on_preview_toggled(self, checked: bool) -> None:
        if checked:
            self._schedule_preview()
        else:
            self._preview_lbl.setText("Enable preview to see a live update")
            self._preview_lbl.setPixmap(QPixmap())

    def _schedule_preview(self) -> None:
        if self._preview_cb.isChecked():
            self._preview_timer.start()

    def _run_preview(self) -> None:
        if self._prev_thread and self._prev_thread.isRunning():
            return
        params = self._collect_params()
        self._prev_thread = _PreviewThread(params, self)
        self._prev_thread.finished.connect(self._on_preview_done)
        self._prev_thread.start()

    @pyqtSlot(object)
    def _on_preview_done(self, arr: np.ndarray) -> None:
        stretched = normalize_for_display(arr)
        h, w = stretched.shape
        qimg   = QImage(stretched.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self._preview_lbl.setPixmap(
            pixmap.scaled(self._preview_lbl.width(), self._preview_lbl.height(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation))

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------

    def _collect_params(self) -> dict:
        idx    = self._size_combo.currentIndex()
        _, W, H = _SIZE_PRESETS[min(idx, len(_SIZE_PRESETS) - 1)]
        return {
            "width":           W,
            "height":          H,
            "contrast_min":    self._contrast_min_sl.value(),
            "contrast_max":    self._contrast_max_sl.value(),
            "sky_adu":         self._sky_spin.value(),
            "apply_psf":       self._psf_cb.isChecked(),
            "fwhm_px":         self._fwhm_spin.value(),
            "moffat_beta":     self._beta_spin.value(),
            "apply_noise":     self._noise_cb.isChecked(),
            "read_noise_adu":  self._rn_spin.value(),
            "seed":            42,
            "output_dir":      self._outdir_lbl.text(),
        }
