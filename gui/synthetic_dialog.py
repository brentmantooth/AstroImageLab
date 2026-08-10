"""Synthetic Data Generator dialog for AstroImageLab.

Allows users to configure and generate synthetic FITS images with controllable
star fields and optical aberrations for metric validation purposes.
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
    QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QTextBrowser, QVBoxLayout, QWidget,
)

from core.stretch import normalize_for_display
from synthetic.cameras import CAMERAS, DEFAULT_CAMERA
from synthetic.generator import SyntheticGenerator


# ---------------------------------------------------------------------------
# Aberration info text (shown in the ⓘ dialog)
# ---------------------------------------------------------------------------
_ABERRATION_INFO_HTML = """
<h2>Synthetic Data Generator — Model Reference</h2>

<h3>Starless Export</h3>
<p>Every generation produces <strong>two FITS files</strong>:</p>
<ul>
<li><b>Full image</b> (<code>synth_…fits</code>) — stars + nebula (if enabled) + sky background + noise.</li>
<li><b>Starless companion</b> (<code>synth_…_starless.fits</code>) — nebula (if enabled) + sky background + noise; no stars.
    Automatically loaded into the panel's starless slot so halo, edge, and spatial-detail
    analyses can run their full starless-aware code paths.</li>
</ul>
<p>Both images have independent noise realisations (separate Poisson sky draws and read-noise
draws), matching the behaviour of real starless images produced by star-removal software.</p>
<hr>

<h2>Aberration Models</h2>
<p>All aberrations are implemented as <strong>kernel-based PSF modifications</strong>
applied independently to each star at its field position.  This is a deliberate
approximation — not full wavefront (Zernike) optics — intended to produce images
with realistic metric signatures.  The key limitation is noted under each model.</p>
<hr>

<h3>Seeing (atmospheric turbulence)</h3>
<p><b>Physical cause:</b> Refractive index variations in the atmosphere blur starlight.</p>
<p><b>Observable effect:</b> All stars enlarged uniformly; FWHM increases across the image.</p>
<p><b>Model:</b> The <em>Seeing (FWHM)</em> spinbox in Image Setup sets the target star
full-width at half-maximum directly in arcseconds.  Each star is rendered as a Moffat
profile I(r) = (1 + r²/α²)^(−β) where α is derived from the FWHM and the Moffat β
parameter.  A minimum of ~4.7 px FWHM is enforced internally so stars remain visible
in the application display panel.  Seeing is applied uniformly across the field.
<em>Limitation: real seeing has temporal variation and a von Kármán power spectrum;
this model applies a fixed, spatially uniform PSF size.</em></p>

<h3>Poor Focus</h3>
<p><b>Physical cause:</b> The focal plane does not coincide with the detector.</p>
<p><b>Observable effect:</b> Stars form defocused disks or donuts; severe defocus produces
hollow ring patterns.</p>
<p><b>Model:</b> Slider 0→1 blends the core Gaussian with an annulus kernel whose radius
grows from 0 to ~12 px.  Light defocus keeps a filled disk; heavy defocus shows the ring.
<em>Limitation: true defocus has diffraction rings (Airy pattern); this model omits them.</em></p>

<h3>Backfocus Error</h3>
<p><b>Physical cause:</b> The camera sensor is positioned too far from or too close to the
optical focuser's designed back-focal distance.</p>
<p><b>Observable effect:</b> Same as defocus — stars are larger than they should be.</p>
<p><b>Model:</b> Slider −1→+1; magnitude equals |slider|, same defocus kernel as Poor Focus.
The sign is recorded in the FITS header (<code>SYN_BF</code>) for traceability.
<em>Limitation: under-focus and over-focus produce the same PSF shape in this model;
in real optics they can differ when combined with coma.</em></p>

<h3>Guiding Error</h3>
<p><b>Physical cause:</b> Imperfect autoguiding causes the telescope to drift during
the exposure, trailing stars.</p>
<p><b>Observable effect:</b> Stars elongated in a consistent direction; FWHM along the trail
axis is much larger than perpendicular.</p>
<p><b>Model:</b> Applied as a global motion-blur convolution over the entire image.
Slider 0→1 produces trail lengths of 0→20 px at a random angle.
<em>Limitation: real guiding error has a random walk component; this model uses a
straight linear trail.</em></p>

<h3>Coma</h3>
<p><b>Physical cause:</b> Off-axis rays focus at different points than on-axis rays,
producing a comet-like tail pointing away from the field centre.</p>
<p><b>Observable effect:</b> Stars near the field centre look round; stars near edges
develop wedge- or comet-shaped tails radiating outward.</p>
<p><b>Model:</b> Each star receives a secondary Gaussian lobe shifted radially outward
from the field centre.  Offset magnitude = slider × radial position × 10 px.
<em>Limitation: true coma has a specific cubic wavefront dependence and a flared shape;
this model approximates it as a Gaussian secondary lobe.</em></p>

<h3>Field Curvature</h3>
<p><b>Physical cause:</b> The focal surface is curved rather than flat, so the detector
(which is flat) is only in focus across part of the field.</p>
<p><b>Observable effect:</b> Stars in the centre are sharp; stars toward edges are
progressively defocused.</p>
<p><b>Model:</b> The effective σ of each star's Gaussian PSF is scaled by
(1 + slider × r²) where r is the normalised field radius.
<em>Limitation: real field curvature also shifts the focal plane position; this model
only broadens the PSF without the associated donut shape at the edges.</em></p>

<h3>Astigmatism</h3>
<p><b>Physical cause:</b> Different curvatures in two perpendicular planes cause stars to
focus as a line in one direction at one distance and a perpendicular line at another.</p>
<p><b>Observable effect:</b> Stars away from centre appear elliptical; the elongation axis
rotates with field angle (tangential at some positions, radial at others).</p>
<p><b>Model:</b> Each star is rendered as an elliptical Gaussian whose axis ratio and
orientation depend on the field angle.
<em>Limitation: true astigmatism produces a cross pattern in extreme cases; the
elliptical Gaussian only captures the mild-astigmatism regime.</em></p>

<h3>Spherical Aberration</h3>
<p><b>Physical cause:</b> Rays from different zones of the lens/mirror focus at
slightly different distances, causing a halo around all stars.</p>
<p><b>Observable effect:</b> Stars have a bright core with a soft, low-level halo; the
effect is uniform across the field.</p>
<p><b>Model:</b> The PSF is a weighted blend of a narrow core Gaussian and a wide
Gaussian (3× the core σ).  At slider = 1, the wide component carries 65 % of the flux.
<em>Limitation: true spherical aberration has a specific radial intensity profile with
bright and dark rings; this model uses a smooth Gaussian blend.</em></p>

<h3>Poor Collimation</h3>
<p><b>Physical cause:</b> The optical axis of one element (mirror, lens) is tilted relative
to the others, introducing a systematic coma-like aberration across the entire field.</p>
<p><b>Observable effect:</b> All stars show an elongation in the same direction regardless
of field position (unlike true coma which is radially symmetric).</p>
<p><b>Model:</b> A secondary Gaussian lobe is added to every star in the same fixed
direction (horizontal).  Lobe offset = slider × 7 px.
<em>Limitation: real collimation error produces a combination of coma and astigmatism
with a field-dependent pattern; this model only captures the dominant fixed-direction lobe.</em></p>

<h3>Halo</h3>
<p><b>Physical cause:</b> Internal reflections inside filters (especially narrowband) scatter
light from bright stars into a large circular halo around each star.</p>
<p><b>Observable effect:</b> Very wide, faint circular halos around bright stars, clearly
visible in narrowband images.  Directly measured by the Halo Analyzer.</p>
<p><b>Model:</b> A very wide Gaussian (σ up to 55 px ≈ 130 px FWHM) is added to each
star's PSF with a relative intensity of slider × 8 %.  Uniform across the field.
<em>Limitation: real filter halos can have a ring structure with a central dark zone;
this model uses a smooth Gaussian halo.</em></p>

<h3>Bortle Class (sky background)</h3>
<p><b>Physical cause:</b> Light pollution raises the sky surface brightness, increasing
both the background level and the associated shot noise.</p>
<p><b>Observable effect:</b> Higher background reduces contrast and the signal-to-noise
ratio of faint objects.</p>
<p><b>Model:</b> Bortle 1–9 maps to sky surface brightness 22.0–17.0 mag/arcsec².
The background is converted to ADU per pixel using the plate scale, gain, and
exposure time.  Poisson noise is added to the background.
<em>Limitation: real skies have gradients and nebula emission; this model uses a
uniform background.</em></p>

<hr>
<h2>Nebula / Siemens Star</h2>
<p><b>Purpose:</b> A <em>Siemens star</em> (radial sinusoidal sectors) is a standard
spatial-frequency test target used in lens testing and image science.  Placing it at
five field positions (image centre + four corners) lets you see directly how detail is
preserved or degraded from the field centre to the edges under the configured
aberrations — complementing the per-star PSF metrics with an extended-source test.</p>

<p><b>Placement:</b> Centre (50 %, 50 %) plus corners at (15 %, 15 %), (85 %, 15 %),
(15 %, 85 %), (85 %, 85 %) of the image dimensions.  These positions sample the full
range of field-radius-dependent PSF effects.</p>

<p><b>PSF convolution model:</b> Each patch is convolved with the <strong>same
field-position-dependent PSF kernel used for stars</strong>, via
<code>scipy.signal.fftconvolve</code>.  Coma magnitude and direction, astigmatism axis
ratio and orientation, field-curvature broadening, defocus ring shape, and collimation
lobe all vary with field position exactly as they do for stars.  Set coma = 0.5 and
the corner patches will show the expected comet-tail asymmetry while the centre patch
remains round.</p>

<p><b>Halo exclusion:</b> The halo slider is set to zero when computing the nebula PSF.
Filter-reflection halos are a point-source optical effect (the ghost image of a bright
stellar disc inside the filter); they do not apply to diffuse extended sources.</p>

<p><b>Brightness:</b> Controlled as a multiple of the sky background ADU.  At 1.0× the
nebula peak equals the sky level — faint but detectable.  At 3–5× it becomes clearly
visible against the noise floor.</p>

<p><b>Size:</b> Fraction of the shorter sensor dimension.  0.15 ≈ 15 % (626 px on the
ASI2600MM 4176 px sensor).  Maximum 0.50 = half the image height/width.</p>

<p><b>Starless:</b> The nebula is included in both the full image and the starless
companion FITS file; stars appear only in the full image.</p>

<p><em>Limitation: the model applies one PSF kernel per patch (the isoplanatic
approximation — the PSF is constant within each patch).  Real telescopes have a
spatially varying PSF across the patch width, which this model does not capture.  For
small nebula sizes (≤ 10–15 % of the image) the approximation is reasonable.</em></p>
"""


# ---------------------------------------------------------------------------
# Background worker threads
# ---------------------------------------------------------------------------

class _GeneratorThread(QThread):
    finished = pyqtSignal(str, str)   # main_path, starless_path
    failed   = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    def run(self):
        try:
            gen = SyntheticGenerator()
            main_path, starless_path = gen.generate(self._params, preview=False)
            self.finished.emit(main_path, starless_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class _PreviewThread(QThread):
    finished = pyqtSignal(object)   # emits np.ndarray

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    def run(self):
        try:
            gen = SyntheticGenerator()
            arr = gen.generate(self._params, preview=True)
            self.finished.emit(arr)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helper: labelled slider row
# ---------------------------------------------------------------------------

class _SliderRow(QWidget):
    """Horizontal row: label | slider | value-label."""

    valueChanged = pyqtSignal(float)

    def __init__(self, lo: float, hi: float, default: float, decimals: int = 2,
                 parent=None):
        super().__init__(parent)
        self._lo = lo
        self._hi = hi
        self._decimals = decimals
        self._scale = 10 ** decimals

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(lo * self._scale))
        self._slider.setMaximum(int(hi * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.setSingleStep(1)
        self._slider.setTickInterval(max(1, int((hi - lo) * self._scale // 10)))

        self._lbl = QLabel(f"{default:.{decimals}f}")
        self._lbl.setFixedWidth(42)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._slider)
        lay.addWidget(self._lbl)

        self._slider.valueChanged.connect(self._on_change)

    def _on_change(self, raw: int) -> None:
        v = raw / self._scale
        self._lbl.setText(f"{v:.{self._decimals}f}")
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self._slider.value() / self._scale

    def setValue(self, v: float) -> None:
        self._slider.setValue(int(v * self._scale))


# ---------------------------------------------------------------------------
# Aberration info dialog
# ---------------------------------------------------------------------------

class _InfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synthetic Data — Aberration Models")
        self.setMinimumSize(650, 520)
        lay = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(_ABERRATION_INFO_HTML)
        browser.setOpenExternalLinks(False)
        lay.addWidget(browser)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.accept)
        lay.addWidget(btns)


# ---------------------------------------------------------------------------
# Main dialog window
# ---------------------------------------------------------------------------

class SyntheticDialog(QMainWindow):
    """Standalone window for configuring and generating synthetic FITS images."""

    image_generated = pyqtSignal(str, str, str)   # main_path, starless_path, panel

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AstroImageLab — Synthetic Data Generator")
        self.setMinimumSize(900, 640)

        self._gen_thread: _GeneratorThread | None = None
        self._prev_thread: _PreviewThread | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._run_preview)

        self._build_ui()
        # Apply defaults that depend on widgets being fully constructed
        self._nebula_cb.setChecked(True)       # enable nebula by default
        self._preview_cb.setChecked(True)      # live preview on by default
        # Restore persisted output directory
        from PyQt6.QtCore import QSettings
        _s = QSettings("FilterImageComparator", "FilterImageComparator")
        saved_dir = _s.value("synth_output_dir", "")
        if saved_dir and os.path.isdir(saved_dir):
            self._outdir_lbl.setText(saved_dir)
        self._on_camera_changed(DEFAULT_CAMERA)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Three-column top section
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._build_image_setup(), stretch=1)
        top.addWidget(self._build_aberrations(), stretch=1)
        top.addWidget(self._build_nebula(), stretch=1)
        top.addWidget(self._build_output(), stretch=1)
        root.addLayout(top)

        # Live preview section
        root.addWidget(self._build_preview_section())

    def _build_image_setup(self) -> QGroupBox:
        box = QGroupBox("Image Setup")
        form = QFormLayout(box)
        form.setVerticalSpacing(5)

        # Camera combo
        self._cam_combo = QComboBox()
        for name in sorted(CAMERAS.keys()):
            self._cam_combo.addItem(name)
        self._cam_combo.setCurrentText(DEFAULT_CAMERA)
        self._cam_combo.setToolTip(
            "Select the camera model.\n"
            "This sets the sensor dimensions, pixel size, full-well capacity,\n"
            "and read noise used in the simulation."
        )
        self._cam_combo.currentTextChanged.connect(self._on_camera_changed)
        self._cam_combo.currentTextChanged.connect(self._schedule_preview)
        form.addRow("Camera", self._cam_combo)

        # Camera info label
        self._cam_info_lbl = QLabel()
        self._cam_info_lbl.setStyleSheet("color: #555; font-size: 10px;")
        form.addRow("", self._cam_info_lbl)

        # Focal length
        self._focal_spin = QDoubleSpinBox()
        self._focal_spin.setRange(100, 10_000)
        self._focal_spin.setValue(700)
        self._focal_spin.setSuffix(" mm")
        self._focal_spin.setDecimals(0)
        self._focal_spin.setToolTip(
            "Telescope focal length in millimetres.\n"
            "Together with the pixel size this determines the plate scale (arcsec/pixel)\n"
            "and the field of view shown in the Output panel."
        )
        self._focal_spin.valueChanged.connect(self._update_optics_labels)
        self._focal_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Focal length", self._focal_spin)

        # Aperture
        self._aperture_spin = QDoubleSpinBox()
        self._aperture_spin.setRange(30, 1_000)
        self._aperture_spin.setValue(140)
        self._aperture_spin.setSuffix(" mm")
        self._aperture_spin.setDecimals(0)
        self._aperture_spin.setToolTip(
            "Telescope aperture (primary mirror or lens diameter) in millimetres.\n"
            "Used to compute the focal ratio (f/number) which determines how much\n"
            "light each pixel collects per second."
        )
        self._aperture_spin.valueChanged.connect(self._update_optics_labels)
        self._aperture_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Aperture", self._aperture_spin)

        # Seeing (FWHM)
        self._fwhm_spin = QDoubleSpinBox()
        self._fwhm_spin.setRange(0.5, 20.0)
        self._fwhm_spin.setValue(3.0)
        self._fwhm_spin.setSuffix(" arcsec")
        self._fwhm_spin.setSingleStep(0.1)
        self._fwhm_spin.setDecimals(2)
        self._fwhm_spin.setToolTip(
            "Atmospheric seeing expressed as star FWHM (Full Width at Half Maximum)\n"
            "in arcseconds.  This is the total star PSF size on the detector.\n"
            "Typical values: 1.5 arcsec (excellent site), 3.0 (average), 5.0+ (poor).\n"
            "Combined with the Moffat β parameter this fully specifies the star profile\n"
            "used by the PSF/MTF analyzer.  A minimum of ~4.7 px FWHM is enforced\n"
            "internally so stars remain visible in the app display panel."
        )
        self._fwhm_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Seeing (FWHM)", self._fwhm_spin)

        # Moffat β
        self._beta_spin = QDoubleSpinBox()
        self._beta_spin.setRange(1.5, 10.0)
        self._beta_spin.setValue(4.77)
        self._beta_spin.setSingleStep(0.1)
        self._beta_spin.setDecimals(2)
        self._beta_spin.setToolTip(
            "Moffat β parameter — controls the power-law slope of the PSF wings.\n\n"
            "I(r) = (1 + r²/α²)^(−β)\n\n"
            "β = 4.77 is the empirical mean for ground-based seeing-limited PSFs\n"
            "(Trujillo et al. 2001, MNRAS 328, 977).  This is what the PSF analyzer\n"
            "fits and is the best value for metric validation.\n\n"
            "Lower β (2–3): broader wings, more light in the halo — typical of\n"
            "poor seeing or undersampled images.\n"
            "Higher β (6–10): wings fall off faster, approaching a Gaussian profile.\n"
            "β → ∞ is a pure Gaussian (not physically realistic for atmospheric PSFs)."
        )
        self._beta_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Moffat β", self._beta_spin)

        # Number of stars
        self._nstars_spin = QSpinBox()
        self._nstars_spin.setRange(10, 2000)
        self._nstars_spin.setValue(600)
        self._nstars_spin.setToolTip(
            "Number of stars to place in the image.\n"
            "Stars are distributed uniformly across the field with ~20 % near the edges\n"
            "to ensure metrics are sampled across the full field of view.\n"
            "More stars give richer spatial coverage but increase generation time."
        )
        form.addRow("Number of stars", self._nstars_spin)

        # Magnitude range
        mag_row = QHBoxLayout()
        self._mag_min_spin = QDoubleSpinBox()
        self._mag_min_spin.setRange(1, 20)
        self._mag_min_spin.setValue(6)
        self._mag_min_spin.setSingleStep(0.5)
        self._mag_min_spin.setDecimals(1)
        self._mag_min_spin.setToolTip(
            "Minimum (brightest) star magnitude.\n"
            "Each magnitude step of 2.5 corresponds to a factor of 10 in brightness.\n"
            "Very bright stars (mag < 5) may saturate depending on gain and exposure."
        )
        self._mag_max_spin = QDoubleSpinBox()
        self._mag_max_spin.setRange(2, 22)
        self._mag_max_spin.setValue(14)
        self._mag_max_spin.setSingleStep(0.5)
        self._mag_max_spin.setDecimals(1)
        self._mag_max_spin.setToolTip(
            "Maximum (faintest) star magnitude.\n"
            "Stars fainter than ~15–17 may fall below the SNR threshold and\n"
            "not be detected by the star-finding algorithm."
        )
        mag_row.addWidget(self._mag_min_spin)
        mag_row.addWidget(QLabel("→"))
        mag_row.addWidget(self._mag_max_spin)
        form.addRow("Magnitude range", mag_row)

        # Exposure
        self._exp_spin = QDoubleSpinBox()
        self._exp_spin.setRange(1, 7200)
        self._exp_spin.setValue(100)
        self._exp_spin.setSuffix(" s")
        self._exp_spin.setDecimals(0)
        self._exp_spin.setToolTip(
            "Simulated exposure time in seconds.\n"
            "Longer exposures increase both signal and sky background ADU proportionally.\n"
            "The SNR improves as √(exposure time) in a sky-limited regime."
        )
        self._exp_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Exposure", self._exp_spin)

        # Gain
        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(0.01, 10.0)
        self._gain_spin.setValue(1.0)
        self._gain_spin.setSuffix(" e⁻/ADU")
        self._gain_spin.setSingleStep(0.1)
        self._gain_spin.setDecimals(2)
        self._gain_spin.setToolTip(
            "Camera conversion gain in electrons per ADU (written as EGAIN in the FITS header).\n"
            "Lower gain (< 1) means fewer electrons per ADU — finer ADU steps but lower\n"
            "full-well before clipping.  Higher gain means more electrons per ADU.\n"
            "Typical values: 0.1–4 e⁻/ADU depending on camera and gain setting."
        )
        self._gain_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Gain", self._gain_spin)

        return box

    def _build_aberrations(self) -> QGroupBox:
        box = QGroupBox("Aberrations")
        form = QFormLayout(box)
        form.setVerticalSpacing(4)

        def _add(label: str, lo: float, hi: float, default: float,
                 decimals: int, tip: str) -> _SliderRow:
            row = _SliderRow(lo, hi, default, decimals)
            row.setToolTip(tip)
            row.valueChanged.connect(self._schedule_preview)
            form.addRow(label, row)
            return row

        self._focus_sl = _add("Poor focus", 0, 0.5, 0, 2,
            "Defocus — stars form disks or donut shapes instead of sharp points.\n"
            "Slider 0 = perfect focus, 0.5 = heavily defocused ring (~6 px radius).\n"
            "Model: ring/annulus kernel blended with core Gaussian.")

        self._backfocus_sl = _SliderRow(-0.5, 0.5, 0, 2)
        self._backfocus_sl.setToolTip(
            "Backfocus error — camera sensor is not at the correct back-focal distance.\n"
            "Slider 0 = correct distance, ±0.5 = maximum error (under- or over-focused).\n"
            "Produces the same defocused PSF as Poor Focus; sign is recorded in the\n"
            "FITS header (SYN_BF) for traceability but the PSF shape is symmetric.")
        self._backfocus_sl.valueChanged.connect(self._schedule_preview)
        bf_row = QHBoxLayout()
        bf_row.addWidget(QLabel("−0.5"))
        bf_row.addWidget(self._backfocus_sl)
        bf_row.addWidget(QLabel("+0.5"))
        form.addRow("Backfocus error", bf_row)

        self._guiding_sl = _add("Guiding error", 0, 0.5, 0, 2,
            "Autoguider drift — stars are trailed in a consistent direction.\n"
            "Slider 0 = perfect guiding, 0.5 = 10 px trail at a random angle.\n"
            "Model: global motion-blur convolution applied after all stars are placed.")

        self._coma_sl = _add("Coma", 0, 0.5, 0, 2,
            "Coma aberration — stars near edges develop comet-like tails pointing\n"
            "radially outward from the field centre.\n"
            "Slider 0 = none, 0.5 = moderate (up to 5 px offset at field corner).\n"
            "Model: secondary Gaussian lobe shifted radially outward; magnitude ∝ slider × r.")

        self._fc_sl = _add("Field curvature", 0, 0.5, 0, 2,
            "Field curvature — the focal surface is curved so stars at the edges\n"
            "are defocused even when the centre is sharp.\n"
            "Slider 0 = flat field, 0.5 = moderate curvature (edge FWHM ~2× centre).\n"
            "Model: effective σ scaled by (1 + slider × r²) per field position.")

        self._ast_sl = _add("Astigmatism", 0, 0.5, 0, 2,
            "Astigmatism — stars away from centre appear elongated; the elongation\n"
            "axis rotates with field angle (tangential vs radial).\n"
            "Slider 0 = none, 0.5 = moderate (axis ratio up to ~1.75:1 at field edge).\n"
            "Model: elliptical Gaussian with orientation and ratio varying by field angle.")

        self._sph_sl = _add("Spherical aberration", 0, 0.5, 0, 2,
            "Spherical aberration — stars have a bright core with a low-level halo.\n"
            "Uniform across the entire field.\n"
            "Slider 0 = none, 0.5 = moderate (wide halo carries ~33 % of star flux).\n"
            "Model: blend of narrow core Gaussian + wide Gaussian (3× core σ).")

        self._coll_sl = _add("Poor collimation", 0, 0.5, 0, 2,
            "Collimation error — all stars show an elongation in the same direction.\n"
            "Unlike coma, this is field-position independent.\n"
            "Slider 0 = perfect collimation, 0.5 = moderate (3.5 px fixed-direction lobe).\n"
            "Model: secondary Gaussian lobe in a fixed direction added to every star.")

        self._halo_sl = _add("Halo", 0, 0.5, 0, 2,
            "Filter-reflection halo — a wide faint halo around each star,\n"
            "common in narrowband images caused by internal filter reflections.\n"
            "This directly exercises the Halo Analyzer metric.\n"
            "Slider 0 = no halo, 0.5 = moderate halo (σ ≈ 28 px, ~4 % of star flux).\n"
            "Model: very wide Gaussian added to each star's PSF uniformly.")

        self._bortle_sl = _SliderRow(1, 9, 5, 0)
        self._bortle_sl.setToolTip(
            "Bortle dark-sky class — controls sky background brightness.\n"
            "1 = pristine dark sky (22.0 mag/arcsec²), 9 = inner-city sky (17.0 mag/arcsec²).\n"
            "Higher Bortle raises the background ADU and its Poisson shot noise,\n"
            "reducing SNR for faint targets.\n"
            "Model: sky surface brightness converted to ADU/px via plate scale, gain, exposure.")
        self._bortle_sl.valueChanged.connect(self._schedule_preview)
        form.addRow("Bortle class (1–9)", self._bortle_sl)

        self._sky_gradient_sl = _SliderRow(0.0, 0.5, 0.0, 2)
        self._sky_gradient_sl.setToolTip(
            "Linear sky gradient — total corner-to-corner swing as a fraction of\n"
            "the base sky level (0.30 = brightest corner 30 % above the darkest).\n"
            "Candidate real causes: light pollution, moon glow, or vignetting.\n"
            "Model: applied to the Poisson expectation, so shot noise scales as\n"
            "sqrt(local sky) exactly as a real gradient does.")
        self._sky_gradient_sl.valueChanged.connect(self._schedule_preview)
        form.addRow("Sky gradient", self._sky_gradient_sl)

        self._sky_gradient_angle_sl = _SliderRow(0, 360, 0, 0)
        self._sky_gradient_angle_sl.setToolTip(
            "Direction the sky gets brighter, in degrees counter-clockwise from +x.\n"
            "No effect while Sky gradient is 0.")
        self._sky_gradient_angle_sl.valueChanged.connect(self._schedule_preview)
        form.addRow("Gradient angle (°)", self._sky_gradient_angle_sl)

        return box

    def _build_nebula(self) -> QGroupBox:
        box = QGroupBox("Nebula")
        form = QFormLayout(box)
        form.setVerticalSpacing(5)

        self._nebula_cb = QCheckBox("Simulate nebula")
        self._nebula_cb.setToolTip(
            "Places a Siemens-star resolution test pattern at the image centre\n"
            "and four corners.  Each instance is convolved with the local PSF so\n"
            "aberrations (coma, field curvature, astigmatism) vary correctly with\n"
            "field position — the same model used for stars.\n"
            "The nebula appears in both the full image and the starless export.")
        self._nebula_cb.toggled.connect(self._on_nebula_toggled)
        self._nebula_cb.toggled.connect(self._schedule_preview)
        form.addRow(self._nebula_cb)

        self._nebula_brightness_spin = QDoubleSpinBox()
        self._nebula_brightness_spin.setRange(0.10, 10.0)
        self._nebula_brightness_spin.setValue(3.0)
        self._nebula_brightness_spin.setSingleStep(0.1)
        self._nebula_brightness_spin.setDecimals(2)
        self._nebula_brightness_spin.setSuffix(" × sky")
        self._nebula_brightness_spin.setEnabled(False)
        self._nebula_brightness_spin.setToolTip(
            "Peak nebula ADU as a multiple of the sky background.\n"
            "1.0 = equal to sky (faint but detectable).\n"
            "Values > 3 become easily visible above sky noise.")
        self._nebula_brightness_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Brightness", self._nebula_brightness_spin)

        self._nebula_size_sl = _SliderRow(0.05, 1.0, 0.50, 2)
        self._nebula_size_sl.setEnabled(False)
        self._nebula_size_sl.setToolTip(
            "Diameter of each Siemens-star patch as a fraction of the shorter\n"
            "image dimension.  0.50 = 50 % (≈2088 px on ASI2600MM at 4176 px height).\n"
            "Range 0.05–1.0.")
        self._nebula_size_sl.valueChanged.connect(self._schedule_preview)
        form.addRow("Size (fraction)", self._nebula_size_sl)

        return box

    def _on_nebula_toggled(self, checked: bool) -> None:
        self._nebula_brightness_spin.setEnabled(checked)
        self._nebula_size_sl.setEnabled(checked)

    def _build_output(self) -> QGroupBox:
        box = QGroupBox("Output")
        lay = QVBoxLayout(box)

        # Output directory
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
        self._autoload_combo = QComboBox()
        self._autoload_combo.addItem("Load into Image A", "A")
        self._autoload_combo.addItem("Load into Image B", "B")
        self._autoload_combo.addItem("Don't load", "")
        self._autoload_combo.setToolTip(
            "Choose which panel to load the generated image into,\n"
            "or select 'Don't load' to only save the file.")
        lay.addWidget(self._autoload_combo)

        lay.addSpacing(6)

        # Optics info labels
        self._scale_lbl = QLabel("Plate scale: —")
        self._fov_lbl = QLabel("Field of view: —")
        self._scale_lbl.setStyleSheet("font-size: 10px;")
        self._fov_lbl.setStyleSheet("font-size: 10px;")
        lay.addWidget(self._scale_lbl)
        lay.addWidget(self._fov_lbl)

        lay.addStretch()

        # Info button
        info_btn = QPushButton("ⓘ  Aberration models…")
        info_btn.setToolTip("Open a description of how each aberration is physically caused\n"
                            "and how it is modelled in the generator.")
        info_btn.clicked.connect(self._show_info)
        lay.addWidget(info_btn)

        # Generate button
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setDefault(True)
        self._gen_btn.setFixedHeight(34)
        self._gen_btn.setToolTip(
            "Generate a FITS image with the current parameter settings.\n"
            "The filename encodes the key parameters for easy identification.")
        self._gen_btn.clicked.connect(self._on_generate)
        lay.addWidget(self._gen_btn)

        # Status
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("font-size: 10px; color: #555;")
        lay.addWidget(self._status_lbl)

        return box

    def _build_preview_section(self) -> QGroupBox:
        box = QGroupBox("Live Preview")
        lay = QHBoxLayout(box)

        left = QVBoxLayout()
        self._preview_cb = QCheckBox("Enable live preview")
        self._preview_cb.setToolTip(
            "When enabled, a reduced-resolution preview is regenerated automatically\n"
            "~400 ms after any parameter change.  The preview uses 20 stars and matches\n"
            "the selected camera's aspect ratio at up to 640 px on the long edge.\n"
            "Disable if preview generation is slow on your machine.")
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

    @pyqtSlot(str)
    def _on_camera_changed(self, name: str) -> None:
        cam = CAMERAS.get(name)
        if cam is None:
            return
        self._cam_info_lbl.setText(
            f"{cam['width_px']}×{cam['height_px']} px · "
            f"{cam['pixel_um']} μm · "
            f"FW {cam['full_well_e']//1000}ke⁻ · "
            f"RN {cam['read_noise_e']} e⁻"
        )
        self._update_optics_labels()

    def _update_optics_labels(self) -> None:
        cam = CAMERAS.get(self._cam_combo.currentText())
        if cam is None:
            return
        focal = self._focal_spin.value()
        if focal <= 0:
            return
        scale = 206.265 * cam["pixel_um"] / focal
        fov_w = cam["width_px"] * scale / 60.0
        fov_h = cam["height_px"] * scale / 60.0
        fratio = focal / max(self._aperture_spin.value(), 1)
        self._scale_lbl.setText(f"Plate scale: {scale:.3f} arcsec/px")
        self._fov_lbl.setText(f"Field of view: {fov_w:.1f}′ × {fov_h:.1f}′  (f/{fratio:.1f})")

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select output directory", self._outdir_lbl.text())
        if d:
            self._outdir_lbl.setText(d)
            from PyQt6.QtCore import QSettings
            QSettings("FilterImageComparator", "FilterImageComparator").setValue(
                "synth_output_dir", d)

    def _show_info(self) -> None:
        dlg = _InfoDialog(self)
        dlg.exec()

    def _on_generate(self) -> None:
        if self._gen_thread and self._gen_thread.isRunning():
            return
        params = self._collect_params()
        os.makedirs(params["output_dir"], exist_ok=True)
        from PyQt6.QtCore import QSettings
        QSettings("FilterImageComparator", "FilterImageComparator").setValue(
            "synth_output_dir", params["output_dir"])
        self._gen_btn.setEnabled(False)
        self._status_lbl.setText("Generating…")
        self._gen_thread = _GeneratorThread(params, self)
        self._gen_thread.finished.connect(self._on_gen_done)
        self._gen_thread.failed.connect(self._on_gen_failed)
        self._gen_thread.start()

    @pyqtSlot(str, str)
    def _on_gen_done(self, main_path: str, starless_path: str) -> None:
        self._gen_btn.setEnabled(True)
        self._status_lbl.setText(f"Saved: {Path(main_path).name}")
        panel = self._autoload_combo.currentData()
        if panel:
            self.image_generated.emit(main_path, starless_path, panel)

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
        stretched = normalize_for_display(arr)   # STF stretch: sky → ~20 % grey
        h, w = stretched.shape
        qimg = QImage(stretched.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self._preview_lbl.setPixmap(
            pixmap.scaled(self._preview_lbl.width(), self._preview_lbl.height(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation))

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------

    def _collect_params(self) -> dict:
        return {
            "camera":          self._cam_combo.currentText(),
            "focal_length_mm": self._focal_spin.value(),
            "aperture_mm":     self._aperture_spin.value(),
            "fwhm_arcsec":     self._fwhm_spin.value(),
            "n_stars":         self._nstars_spin.value(),
            "mag_min":         self._mag_min_spin.value(),
            "mag_max":         self._mag_max_spin.value(),
            "exposure_s":      self._exp_spin.value(),
            "gain_e_per_adu":  self._gain_spin.value(),
            "bortle":          int(round(self._bortle_sl.value())),
            "sky_gradient":           self._sky_gradient_sl.value(),
            "sky_gradient_angle_deg": self._sky_gradient_angle_sl.value(),
            "moffat_beta":     self._beta_spin.value(),
            "poor_focus":      self._focus_sl.value(),
            "backfocus":       self._backfocus_sl.value(),
            "guiding":         self._guiding_sl.value(),
            "coma":            self._coma_sl.value(),
            "field_curvature": self._fc_sl.value(),
            "astigmatism":     self._ast_sl.value(),
            "spherical":       self._sph_sl.value(),
            "collimation":     self._coll_sl.value(),
            "halo":            self._halo_sl.value(),
            "nebula_enabled":    self._nebula_cb.isChecked(),
            "nebula_brightness": self._nebula_brightness_spin.value(),
            "nebula_size":       self._nebula_size_sl.value(),
            "output_dir":      self._outdir_lbl.text(),
        }
