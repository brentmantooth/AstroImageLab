# Astro Image Lab

A Python desktop application for characterizing narrowband astrophotography filters through quantitative image analysis. Load one or two calibrated images taken through different filters and generate a detailed comparison report covering PSF quality, halo artifacts, edge sharpness, spatial frequency content, multi-scale detail preservation, and signal-to-noise ratio. Includes built-in tools for generating synthetic test images, calibration targets, and interactive per-star halo inspection.

---

## Features

| Metric | Description | Bandwidth-independent? |
|--------|-------------|----------------------|
| **PSF / MTF** | Moffat profile fitting, empirical PSF, MTF curve and MTF50 | ✓ Yes |
| **Halo analysis** | Two-component radial profile fit; halo-to-core ratio | ✓ Yes |
| **Edge analysis (LSF)** | Edge Spread Function, 10–90% edge width, Line Spread Function | ✓ Yes (width) / ⚠ (contrast ratio) |
| **Power spectrum** | Signal-normalised 2D FFT, mid/high spatial frequency ratio | ✓ Normalised |
| **Spatial detail** | Local σ, Laplacian of Gaussian, wavelet, gradient magnitude, and local entropy maps across multiple scales; log-ratio and noise-corrected cross-method comparison; scale-adaptive local-maxima masked metrics | ✓ Normalised |
| **Signal / Noise (SNR)** | Global sky-σ SNR, median star SNR ± IQR, per-pixel SNR map, pixel percentile table | ✓ Yes |

All analysis runs on linear (unstretched) calibrated image data. Images with different filter bandwidths are handled correctly — metrics are clearly labelled as bandwidth-independent or bandwidth-sensitive, and a warning banner appears in the report when bandwidths differ.

---

## Screenshot

![Astro Image Lab](/resources/AstroImageLabMain.png)

---

## Download

Pre-built binaries are produced automatically by GitHub Actions on every push to `main` — no Python or conda required to run them.

| Platform | Artifact | How to run |
| -------- | -------- | ---------- |
| **Windows** | `AstroImageLab-win64.zip` | Unzip and double-click `AstroImageLab.exe` |
| **macOS** | `AstroImageLab-macos.zip` | Unzip, then see the [macOS note](#macos-gatekeeper-note) below |
| **Linux** | `AstroImageLab-linux.zip` | Unzip and run `./AstroImageLab` from a terminal |

Download the latest artifact from the [Actions tab](../../actions) → most recent **CI** run → **Artifacts** section at the bottom of the page.

### macOS Gatekeeper note

CI-built binaries are not code-signed. macOS Gatekeeper will block the app on first launch. To open it, use **either** of these methods:

- Right-click the binary → **Open** → click **Open** in the security dialog, **or**
- Run in Terminal: `xattr -dr com.apple.quarantine AstroImageLab`

This is a one-time step per download. Code signing for seamless distribution requires an Apple Developer Program membership ($99/year) and is not currently configured.

---

## Requirements

### Python
Python 3.10+ (tested with Anaconda 3.12.7 / Python 3.12)

### Recommended: conda environment

The easiest way to get a working environment is via the included `environment.yml`:

```bash
conda env create -f environment.yml
conda activate astrolab
python AstroImageLab.py
```

This creates a conda environment named `astrolab` with all required packages. **PyQt6 is installed via pip** (not conda-forge) because the pip layout is required for PyInstaller to correctly locate Qt DLLs when building a standalone executable.

### Manual installation

Install the whole scientific stack via pip into an existing conda environment (not conda-forge — see the note below):

```bash
pip install numpy scipy seaborn matplotlib astropy photutils bottleneck pywavelets astroalign pillow lz4 zstandard pyqt6 xisf
```

> **Important:** Install the entire stack — including PyQt6 — via `pip`, not `conda install`. The conda-forge builds of PyQt6 and several scientific packages pull in a `qt6-main` package with a DLL layout that conflicts with PyInstaller's hook discovery and with the pip-installed Qt runtime. `environment.yml` installs everything via pip specifically to avoid this.

---

## Installation

```bash
git clone https://github.com/<your-username>/AstroImageLab.git
cd AstroImageLab

# Option A — recommended (creates an isolated conda environment)
conda env create -f environment.yml
conda activate astrolab

# Option B — manual install into an existing environment
pip install numpy scipy seaborn matplotlib astropy photutils bottleneck pywavelets astroalign pillow lz4 zstandard pyqt6 xisf
```

---

## Usage

```bash
conda activate astrolab
python AstroImageLab.py
```

### Image Preparation

**Required — one or two images of the same sky region:**
- Image A is required; Image B is optional — loading only Image A runs the app in single-image analysis mode (per-image metrics still run, but comparison tables and A/B differential metrics are unavailable). When both are loaded, Image A and Image B must cover the same field of view, captured through different filters (or different filter configurations you want to compare).
- Images should be **calibrated and stacked** (bias/dark/flat corrected) but **not stretched**. Linear data is required for valid metric calculations.
- Supported formats: `.fits`, `.fit`, `.fts`, `.xisf`, `.tiff`, `.tif`.

**Suggested — starless versions of each image:**
- Creating starless counterparts (using tools such as [Star XTerminator](https://www.rc-astro.com/resources/StarXTerminator/), [StarNet++](https://www.starnetastro.com/), or equivalent) and loading them alongside the original images significantly improves edge, power spectrum, and spatial detail analysis by removing the PSF contribution of stars from nebula regions.
- Load the starless images when prompted after loading the original image.

**For narrowband filter comparisons — enter filter bandwidths:**
- If your image headers do not contain filter bandwidth information, enter the bandwidth (in nm) manually in the field provided in each image panel (e.g., 3 nm vs 7 nm Ha filters).
- Several metrics are bandwidth-sensitive. Providing accurate bandwidths ensures the report correctly flags and contextualises these differences. A warning banner appears in the report when bandwidths differ.

**Recommended — draw a cross-section line:**
- Before running analysis, click **Select Line…** and draw a line across a region of astrophysical interest (e.g., across a nebula filament or emission edge). This enables per-pixel cross-section profile overlays in the Spatial Detail and Edge Analysis sections of the report.
- Without a cross-section line, edge analysis auto-detects the strongest gradient and spatial detail sections show maps only, with no profile overlays.

---

### Workflow

1. **Load images** — Click **Open FITS / XISF…** in each panel to load Image A and Image B.
2. **Load starless images** *(optional but recommended)* — Click **Open Starless…** in each panel to load the corresponding starless version. Used to isolate nebula structure for edge and spatial detail analysis.
3. **Review metadata** — Telescope, camera, filter, exposure, date, and pixel scale are read from the file headers and displayed automatically. Enter the filter bandwidth (nm) manually if not present in the header.
4. **Draw a cross-section line** *(recommended)* — Click **Select Line…** and drag a line across a region of interest. The line appears overlaid on both images. This drives cross-section profile analysis in the report.
5. **Select ROI** *(optional)* — Click **Select ROI…** and draw a rectangle to target a specific nebula region for edge and power spectrum analysis. If no ROI is selected, the app auto-detects the strongest edge and a star-free region automatically.
6. **Select metrics** — Check or uncheck the metrics you want to run in the control panel. Each metric can also have its figures exported as standalone PNG files using the corresponding export checkbox.
7. **Set output directory** — Browse to where the self-contained HTML report should be saved.
8. **Run** — Click **Run Analysis**. Images are aligned automatically using `astroalign` before per-pixel comparisons. A progress bar and per-metric timer are shown during analysis.
9. **Review report** — The HTML report and an interactive Report Inspector window both open automatically when analysis completes.

The **Tools** menu also has three standalone utilities not part of this linear workflow: a Synthetic Data Generator and a Spatial Target Generator for producing test images, and an interactive Halo Analyzer for click-a-star PSF/halo inspection. See [QuickStart.md](QuickStart.md#additional-tools) for details.

---

## Output Report

The report is saved to your chosen output directory as a single self-contained HTML file (all plots embedded as base64 PNG). HTML is the only output format and requires no additional packages. It contains nine sections:

1. **Image metadata** — Side-by-side header info for both filters; bandwidth warning banner if bandwidths differ
2. **Observation context** — Seeing warning if FWHM > 3″; notes on valid comparison conditions
3. **Signal / Noise (SNR)** — Global sky-σ SNR, median star SNR ± IQR, per-pixel SNR map (side-by-side, plasma colourmap), and a pixel percentile table showing what fraction of the field exceeds 3σ / 5σ / 10σ / 20σ
4. **PSF / MTF** — FWHM, Moffat β, ellipticity, MTF50, MTF at Nyquist; overlaid MTF curves; ePSF images; field aberration scoring (coma, collimation, field curvature)
5. **Halo analysis** — Halo-to-core ratio, halo radius; side-by-side semi-log radial profiles
6. **Edge analysis** — 10–90% edge width in pixels and arcseconds; ESF and LSF plots; edge contrast ratio (flagged ⚠ if bandwidths differ); cross-section profile overlay if a line was drawn
7. **Power spectrum** — Signal-normalised 2D power spectrum; radial power comparison; mid/high ratio and dB ratio curve
8. **Spatial detail** — Local σ, Laplacian of Gaussian, wavelet, gradient magnitude, and local entropy maps at multiple scales; log-ratio comparison and correlation scatter plots per family; noise-corrected cross-method overview; scale-adaptive local-maxima masked metrics; cross-section profile overlays if a line was drawn
9. **Summary table** — All scalar metrics side by side; better value highlighted green, worse value highlighted red

---

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| FITS | `.fits` `.fit` `.fts` | Standard calibrated output from all major acquisition software |
| XISF | `.xisf` | PixInsight native format; requires `pip install xisf` |
| TIFF | `.tiff` `.tif` | 16-/32-bit calibrated TIFF stacks |

---

## On-Sky vs Bench Testing

This tool is designed for **on-sky images**, not optical bench tests. Several important caveats apply:

- **Seeing is the dominant PSF contribution** on most nights. PSF/MTF comparisons between filters are most meaningful when both images were captured on the same night under similar atmospheric conditions.
- The app flags `seeing_dominated = True` and adds a warning in the report when FWHM exceeds 3″.
- **Halo, edge width, and spatial detail metrics** are less sensitive to seeing and are more reliably attributable to filter differences.
- **Astroalign** is used to register Image A onto the coordinate frame of Image B before any per-pixel comparison metrics are computed.

---

## Bandwidth Validity

Filters with different bandwidths (e.g., 3 nm vs 7 nm) produce different absolute ADU levels. The app handles this systematically:

**Bandwidth-independent metrics** (ratio or normalised — valid as-is):
- PSF FWHM and MTF (normalised PSF shape)
- Halo-to-core ratio
- Edge 10–90% width (normalised ESF)
- Spatial detail log-ratio maps and local-maxima masked metrics (all mean-signal normalised)
- Power spectrum mid/high ratio (mean-signal normalised before FFT)
- SNR metrics (all expressed as signal / noise ratios, independent of absolute flux)

**Bandwidth-sensitive metrics** (flagged ⚠ in the report):
- Edge contrast ratio (bright/dark side signal; affected by background level)

When filter bandwidths differ, a banner appears at the top of the report, and each sensitive metric carries an explanatory note.

---

## Project Structure

```text
AstroImageLab.py             # Entry point
environment.yml              # Conda environment specification
core/
  models.py                  # Constants, AnalysisResult dataclass
  astro_image.py              # FITS/XISF/TIFF loading, background estimation
  fig_utils.py                # fig_to_b64(), finalize_layout() — thread-safe figure rendering
  stretch.py                  # STF stretch + normalize_for_display()
  stats_utils.py              # mannwhitney_effect() — shared significance testing
  update_checker.py           # GitHub-release update check
analysis/
  star_catalog.py             # DAOStarFinder + isolation filtering
  psf_analyzer.py              # Moffat fitting, ePSF builder, MTF via FFT
  moffat_fit.py                # Shared Moffat-fitting helpers
  halo_analyzer.py             # Radial profile extraction, two-component Moffat fit
  edge_analyzer.py             # Sobel edge detection, ESF/LSF extraction
  power_spectrum.py            # Signal-normalised 2D FFT and radial average
  image_filters.py             # Local σ, LoG, wavelet, gradient, entropy, local-maxima
  snr_analyzer.py               # Global SNR, per-star SNR, local SNR map, percentile table
report/
  report_builder.py            # Self-contained HTML report generator
gui/
  image_panel.py                # PyQt6 image display with ROI rubber-band and line selection
  control_panel.py              # Metric checkboxes, parameters, output directory
  analysis_thread.py            # QThread orchestrator; runs all engines off the main thread
  main_window.py                # QMainWindow; assembles panels, toolbar, menu, signal wiring
  report_inspector.py           # Interactive side-by-side figure viewer
  synthetic_dialog.py           # Synthetic Data Generator dialog
  spatial_target_dialog.py      # Spatial Target Generator dialog
  halo_dialog.py                 # Halo Analyzer interactive tool
synthetic/
  cameras.py                     # Camera database (24 models)
  generator.py                   # Synthetic star-field image generation engine
  target_generator.py            # Spatial calibration target generation engine
tools/
  generate_icon.py                # Regenerates resources/icon.ico
  generate_screenshots.py         # Regenerates resources/*.png doc screenshots
```

---

## Key Dependencies and Acknowledgements

| Library | Purpose |
|---------|---------|
| [astropy](https://www.astropy.org/) | FITS I/O, Moffat2D model, Background2D |
| [photutils](https://photutils.readthedocs.io/) | DAOStarFinder, EPSFBuilder, morphology |
| [bottleneck](https://bottleneck.readthedocs.io/) | Fast NaN-aware and median reductions on large arrays |
| [scipy](https://scipy.org/) | Optimisation, FFT, image filters |
| [PyWavelets](https://pywavelets.readthedocs.io/) | Daubechies-4 wavelet decomposition |
| [astroalign](https://astroalign.quatrope.org/) | Image registration |
| [xisf](https://pypi.org/project/xisf/) | PixInsight XISF format support |
| [PyQt6](https://riverbankcomputing.com/software/pyqt/) | GUI framework |
| [matplotlib](https://matplotlib.org/) | All plots and figures |

Wavelet noise estimation uses the robust MAD estimator from Donoho & Johnstone (1994).
SNR background estimation uses photutils `Background2D` with `MADStdBackgroundRMS`.
By default stars and extended nebulosity are detected and masked out first, and the sky
level is a BIC-selected low-order surface fitted to the mesh cells that survive — the
plain per-cell estimate assumes every cell is sky-dominated and biases upward over
nebulosity. Report section 3g carries the mask, the per-tier detection thresholds, and
the before/after comparison; the **Source-masked background** checkbox turns it off.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
