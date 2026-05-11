# Astro Image Lab

A Python desktop application for characterizing narrowband astrophotography filters through quantitative image analysis. Load two calibrated images taken through different filters and generate a detailed comparison report covering PSF quality, halo artifacts, ghost images, edge sharpness, spatial frequency content, and multi-scale detail preservation.

---

## Features

| Metric | Description | Bandwidth-independent? |
|--------|-------------|----------------------|
| **PSF / MTF** | Moffat profile fitting, empirical PSF, MTF curve and MTF50 | ✓ Yes |
| **Halo analysis** | Two-component radial profile fit; halo-to-core ratio | ✓ Yes |
| **Ghost detection** | Secondary reflection search around bright stars | ✓ Yes |
| **Edge analysis (LSF)** | Edge Spread Function, 10–90% edge width, Line Spread Function | ✓ Yes (width) / ⚠ (contrast ratio) |
| **Power spectrum** | Signal-normalised 2D FFT, mid/high spatial frequency ratio | ✓ Normalised |
| **Local std maps** | Local standard deviation at 3 kernel scales; contrast ratio metric | ✓ Normalised |
| **Laplacian of Gaussian** | Edge/detail enhancement at 3 spatial scales | ✓ Normalised |
| **Wavelet decomposition** | 4-level Daubechies-4 decomposition; per-level SNR; detail images | ✓ Normalised |

All analysis runs on linear (unstretched) calibrated image data. Images with different filter bandwidths are handled correctly — metrics are clearly labelled as bandwidth-independent or bandwidth-sensitive, and a warning banner appears in the report when bandwidths differ.

---

## Screenshot

![Astro Image Lab](/resources/AstroImageLabMain.png)

---

## Requirements

### Python
Python 3.10+ (tested with Anaconda 3.12.7)

### Conda packages
```bash
conda install -c conda-forge pyqt6 astropy photutils scipy numpy matplotlib astroalign pillow pywavelets
```

### Pip packages
```bash
pip install xisf xhtml2pdf
```

`xisf` provides support for PixInsight's native `.xisf` format. `xhtml2pdf` enables PDF report export (pure-Python, no native library dependencies). All other dependencies are available via conda-forge.

### PDF report export (optional)

The **PDF** format option in the report panel requires a PDF renderer. Two options are supported — the app tries them in order and falls back to HTML if neither is available:

| Renderer | Install | Notes |
|----------|---------|-------|
| **WeasyPrint** | `conda install -c conda-forge weasyprint` | Best layout fidelity. Requires GTK3/Pango native libraries. The conda-forge package bundles these automatically. On Windows with pip, the GTK3 runtime must be installed separately. |
| **xhtml2pdf** | `pip install xhtml2pdf` | Pure-Python fallback. No native dependencies. Complex CSS (flexbox, grid) renders at reduced fidelity. |

If neither renderer is installed, the report is saved as HTML and a warning is shown in the completion dialog.

---

## Installation

```bash
git clone https://github.com/<your-username>/AstroImageLab.git
cd AstroImageLab

# Install dependencies (see Requirements above)
conda install -c conda-forge pyqt6 astropy photutils scipy numpy matplotlib astroalign pillow pywavelets
pip install xisf
```

---

## Usage

```bash
cd AstroImageLab
python AstroImageLab.py
```

### Image Preparation

**Required — two images of the same sky region:**
- Image A and Image B must cover the same field of view, captured through different filters (or different filter configurations you want to compare).
- Images should be **calibrated and stacked** (bias/dark/flat corrected) but **not stretched**. Linear data is required for valid metric calculations.
- Supported formats: `.fits`, `.fit`, `.fts`, `.xisf`.

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
6. **Select metrics** — Check or uncheck the metrics you want to run in the control panel.
7. **Set output directory and format** — Browse to where the report should be saved. Choose HTML (default) or PDF from the format selector. PDF requires WeasyPrint or xhtml2pdf (see Requirements).
8. **Run** — Click **Run Analysis**. Images are aligned automatically using `astroalign` before per-pixel comparisons. A progress bar and elapsed timer are shown during analysis.
9. **Review report** — The HTML report opens automatically in your default browser when analysis completes.

---

## Output Report

The report is saved to your chosen output directory as a self-contained HTML file (all plots embedded as base64 PNG) or as a PDF if a renderer is installed. HTML is the default and requires no additional packages. It contains nine sections:

1. **Image metadata** — Side-by-side header info for both filters; bandwidth warning banner if bandwidths differ
2. **Observation context** — Seeing warning if FWHM > 3″; notes on valid comparison conditions
3. **PSF / MTF** — FWHM, Moffat β, ellipticity, MTF50, MTF at Nyquist; overlaid MTF curves; ePSF images
4. **Halo analysis** — Halo-to-core ratio, halo radius; side-by-side semi-log radial profiles
5. **Ghost detection** — Candidate table (separation, intensity ratio, classification); annotated image
6. **Edge analysis** — 10–90% edge width in pixels and arcseconds; ESF and LSF plots; edge contrast ratio (flagged ⚠ if bandwidths differ); cross-section profile overlay if a line was drawn
7. **Power spectrum** — Signal-normalised 2D power spectrum; radial power comparison; mid/high ratio
8. **Spatial detail** — Local σ maps (3 scales), |LoG| maps (3 scales), wavelet detail images and SNR bar chart; cross-section profile overlays if a line was drawn
9. **Summary table** — All scalar metrics side by side; better value highlighted green, worse value highlighted red

---

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| FITS | `.fits` `.fit` `.fts` | Standard calibrated output from all major acquisition software |
| XISF | `.xisf` | PixInsight native format; requires `pip install xisf` |

---

## On-Sky vs Bench Testing

This tool is designed for **on-sky images**, not optical bench tests. Several important caveats apply:

- **Seeing is the dominant PSF contribution** on most nights. PSF/MTF comparisons between filters are most meaningful when both images were captured on the same night under similar atmospheric conditions.
- The app flags `seeing_dominated = True` and adds a warning in the report when FWHM exceeds 3″.
- **Halo, ghost, edge width, and spatial detail metrics** are less sensitive to seeing and are more reliably attributable to filter differences.
- **Astroalign** is used to register Image A onto the coordinate frame of Image B before any per-pixel comparison metrics are computed.

---

## Bandwidth Validity

Filters with different bandwidths (e.g., 3 nm vs 7 nm) produce different absolute ADU levels. The app handles this systematically:

**Bandwidth-independent metrics** (ratio or normalised — valid as-is):
- PSF FWHM and MTF (normalised PSF shape)
- Halo-to-core ratio and ghost-to-parent ratio
- Edge 10–90% width (normalised ESF)
- Local std contrast ratio, LoG maps, wavelet SNR (all mean-signal normalised)
- Power spectrum mid/high ratio (mean-signal normalised before FFT)

**Bandwidth-sensitive metrics** (flagged ⚠ in the report):
- Edge contrast ratio (bright/dark side signal; affected by background level)

When filter bandwidths differ, a banner appears at the top of the report, and each sensitive metric carries an explanatory note.

---

## Project Structure

```
AstroImageLab.py            # Entry point
requirements.txt
core/
  models.py                 # Constants, AnalysisResult dataclass
  astro_image.py            # FITS/XISF loading, background estimation, statistical stretch
analysis/
  star_catalog.py           # DAOStarFinder + isolation filtering
  psf_analyzer.py           # Moffat fitting, ePSF builder, MTF via FFT
  halo_analyzer.py          # Radial profile extraction, two-component Moffat fit
  ghost_detector.py         # Secondary source search in annular regions
  edge_analyzer.py          # Sobel edge detection, ESF/LSF extraction
  power_spectrum.py         # Signal-normalised 2D FFT and radial average
  image_filters.py          # Local std maps, LoG maps, wavelet decomposition
report/
  report_builder.py         # Self-contained HTML report generator
gui/
  image_panel.py            # PyQt6 image display with ROI rubber-band and line selection
  control_panel.py          # Metric checkboxes, parameters, output directory
  analysis_thread.py        # QThread orchestrator; runs all engines off the main thread
  main_window.py            # QMainWindow; assembles panels, menu, signal wiring
```

---

## Key Dependencies and Acknowledgements

| Library | Purpose |
|---------|---------|
| [astropy](https://www.astropy.org/) | FITS I/O, Moffat2D model, Background2D |
| [photutils](https://photutils.readthedocs.io/) | DAOStarFinder, EPSFBuilder, morphology |
| [scipy](https://scipy.org/) | Optimisation, FFT, image filters |
| [PyWavelets](https://pywavelets.readthedocs.io/) | Daubechies-4 wavelet decomposition |
| [astroalign](https://astroalign.quatrope.org/) | Image registration |
| [xisf](https://pypi.org/project/xisf/) | PixInsight XISF format support |
| [PyQt6](https://riverbankcomputing.com/software/pyqt/) | GUI framework |
| [matplotlib](https://matplotlib.org/) | All plots and figures |
| [WeasyPrint](https://weasyprint.org/) *(optional)* | High-fidelity HTML→PDF rendering |
| [xhtml2pdf](https://xhtml2pdf.readthedocs.io/) *(optional)* | Pure-Python HTML→PDF fallback |

Statistical stretch algorithm adapted from [SETIAstroSuite](https://www.setiastro.com/statistical-stretch) (MTF-based autostretch, equivalent to PixInsight STF).

Wavelet noise estimation uses the robust MAD estimator from Donoho & Johnstone (1994).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
