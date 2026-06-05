# AstroImageLab — Claude Code Instructions

## Project Overview

AstroImageLab is a PyQt6 desktop application that characterises astrophotography
filters by running quantitative image analysis on one or two calibrated FITS/XISF
images. It produces a self-contained HTML report with embedded matplotlib figures.

**Owner:** Brent (solo project)
**Platform:** Windows 11, conda environment `astrolab`
**Entry point:** `AstroImageLab.py`

---

## Architecture

```
AstroImageLab.py       PyQt6 app + animated splash screen
analysis/              Metric engines — each returns a plain dict
  psf_analyzer.py      Moffat/ePSF fitting, MTF via FFT
  halo_analyzer.py     Radial halo profiles, two-component Moffat fit
  snr_analyzer.py      Global/per-star/local SNR, noise factor, sky electrons
  edge_analyzer.py     ESF/LSF edge analysis via Sobel
  power_spectrum.py    Signal-normalised 2D FFT power spectrum
  image_filters.py     Local σ maps, Laplacian of Gaussian, wavelet decomposition
  star_catalog.py      DAOStarFinder star detection and isolation filtering
core/
  astro_image.py       FITS/XISF loading, background estimation (photutils)
  models.py            40+ constants + AnalysisResult dataclass
  fig_utils.py         fig_to_b64() — embeds matplotlib figure as base64 PNG
  stretch.py           STF stretch + normalize_for_display() for 8-bit display output
gui/
  analysis_thread.py   QThread orchestrator; dark-mode rcParams save/restore lives here
  control_panel.py     Settings UI; settings() returns dict consumed by the thread
  image_panel.py       Image display panel; load_path() / set_starless_path() for programmatic load
  report_inspector.py  Interactive side-by-side figure viewer
  synthetic_dialog.py  Synthetic Data Generator dialog (QMainWindow)
report/
  report_builder.py    HTML report generator; consumes AnalysisResult objects
synthetic/
  cameras.py           Camera database — 24 models (ZWO, QHY, Player One)
  generator.py         Image generation engine: Moffat PSF, aberrations, nebula, starless export
```

---

## Key Utilities — Reuse These

| Utility | Location | Purpose |
| --- | --- | --- |
| `_info_box(body, title, open=False, style="")` | `report_builder.py` | Collapsible `<details>/<summary>` HTML panel |
| `_val(v, fmt, fallback="—")` | `report_builder.py:184` | Null-safe table cell formatter |
| `fig_to_b64(fig)` | `core/fig_utils.py` | Embeds matplotlib figure as base64 PNG string |
| `normalize_for_display(arr)` | `core/stretch.py` | STF-stretch float32 array → uint8 [0,255] for QImage display |
| `stf_stretch(data)` | `core/stretch.py` | STF midtone-balance stretch → float32 [0,1]; maps sky to ~20 % grey |
| `load_path(path)` | `gui/image_panel.py` | Load image by path with no dialog and no starless prompt |
| `set_starless_path(path)` | `gui/image_panel.py` | Attach a pre-generated starless FITS to the loaded main image |

---

## Coding Conventions

### Dark mode — check rcParams before any hardcoded colour

```python
import matplotlib
_is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
orig_color = "white" if _is_dark else "black"
```

Place this **before** any loop that references `orig_color`. Dark mode is applied
globally in `analysis_thread.run()` via `plt.style.use("dark_background")` and
restored with `rcParams.update(_saved_params)` in the `finally` block. Never apply
dark mode at module import time — it bleeds into unrelated figure generation.

### Python string encoding — no CSS/HTML hex escapes

CSS hex escapes (`\25B6`, `\00A0`) inside Python string literals are parsed as
**octal escapes**, producing garbled characters in the HTML output. Always use:

- Literal Unicode characters directly: `▶`, `▼`, `—`
- Python Unicode escapes for invisible characters: ` ` (non-breaking space)

### FITS gain keywords — prefer EGAIN, guard against zero

`GAIN` in FITS headers is often a camera *mode index* (0, 100, 200) not the physical
e⁻/ADU conversion factor. Always:

1. Try keywords in this order: `EGAIN`, `GAIN`, `CCDGAIN`, `GAINDB`
2. Accept only values where `g > 0`

### Number formatting — use `g` for scientific values

Fixed decimal formats (`.1f`, `.3f`) silently round sub-ADU values to zero.
Use significant-figure formats for any value that can span orders of magnitude:

| Value type | Format |
| --- | --- |
| ADU sky values (σ_sky, μ_sky) | `.6g` |
| Electron sky values | `.3g` |
| Dimensionless ratios (noise factor) | `.3f` |
| Percentages | `.4f` |

### Long f-string HTML blocks

Pre-compute any Python variable **before** a `return f"""..."""` block. Do not nest
`{f"...{var}..."}` substitutions — they cause confusing `UnboundLocalError` and
syntax errors at runtime.

### PyQt6 signal arity — declaration must match every emit()

`pyqtSignal(str, str)` declared but `.emit(a, b, c)` called raises a `TypeError` at
runtime, not at import or compile time — py_compile will not catch it. Always verify
the number of types in the `pyqtSignal(...)` declaration matches every `.emit()` call
site and every `.connect(slot)` slot signature before running.

### Two-RNG pattern for deterministic vs random draws

When some outputs must be reproducible (e.g. star positions for catalogue matching)
and others should vary independently (noise), use two separate generators:

```python
star_rng  = np.random.default_rng(int(n_stars))   # seeded from count → always same stars
noise_rng = np.random.default_rng(params.get("seed", 42))  # independent noise
```

Apply `star_rng` to positions and magnitudes; `noise_rng` to guiding angle, sky
Poisson, and read noise. This lets the user load two images with different aberrations
into Image A/B and run star-matching analysis — stars are co-located by design.

### Convolution of large patches — use fftconvolve

`scipy.ndimage.convolve` uses direct convolution: O(N²×K²). For PSF convolution of
extended nebula patches (hundreds of pixels) use `scipy.signal.fftconvolve` which is
O(N² log N) and handles any kernel size without performance degradation:

```python
from scipy.signal import fftconvolve
convolved = fftconvolve(patch, psf_kernel, mode="same").astype(np.float64)
```

---

## Collaboration Rules

- **Never commit automatically.** Always ask the user for approval first.
- **Never delete `#` comments** that explain purpose, units, or rationale.
- **Use Edit, not Write,** for any existing file.
- **Ask before changing `core/models.py` constants** — many downstream callers depend on them.
- **Do not re-add PDF export.** WeasyPrint/xhtml2pdf have been intentionally removed.

---

## Build & Run

```bash
conda activate astrolab
python AstroImageLab.py          # run from source
pyinstaller AstroImageLab.spec   # build standalone .exe
```

PyQt6 **must** be installed via `pip`, not `conda` — the conda-forge PyQt6 package
uses a different DLL layout that breaks PyInstaller hook discovery on Windows.

---

## Known Pitfalls

| Pitfall | Rule |
| --- | --- |
| Inspector dark theme bleeds into report figures | Apply dark mode only in `analysis_thread.run()`, never at import time |
| `orig_color` UnboundLocalError | Assign dark-mode variables before any loop that references them |
| CSS characters garbled in HTML output | Use literal Unicode — never CSS hex escapes inside Python strings |
| Sky electron values display as 0.0 | Use `.3g`; `.1f`/`.2f` silently rounds sub-electron values to zero |
| GAIN = 0 accepted from FITS header | Guard with `if g > 0` after parsing the header value |
| `pyqtSignal` arity mismatch silently compiles | `pyqtSignal(str, str)` vs `.emit(a, b, c)` crashes at runtime only — py_compile passes. Count signal args carefully. |
| Single RNG shifts star positions when params change | Use a dedicated `star_rng` seeded from `n_stars`; separate `noise_rng` for sky/read noise |
| Preview PSF too large when image is downsampled | `_star_psf` uses pixel-unit constants (coma offset, halo sigma, etc.). Pass `plate_scale / px_scale` and `px_scale=<downsample_fraction>` so all pixel constants scale correctly with the preview resolution. |

---

## Synthetic Data Generator — Key Patterns

### Signal chain

```text
SyntheticDialog.image_generated = pyqtSignal(str, str, str)  # main_path, starless_path, panel
  → MainWindow._on_synthetic_generated(main_path, starless_path, panel)
      → ImagePanel.load_path(main_path)
      → ImagePanel.set_starless_path(starless_path)
```

`_GeneratorThread.finished = pyqtSignal(str, str)` (main, starless) feeds `_on_gen_done`
which then emits the three-arg `image_generated` signal.

### Generator return types

`SyntheticGenerator.generate(params, preview=False)`:

- `preview=True` → `np.ndarray` (float32, full_image at ¼ camera resolution)
- `preview=False` → `tuple[str, str]` (main_path, starless_path)

Every generation always produces both FITS files. The starless companion is named
`<stem>_starless.fits` in the same directory.

### Nebula PSF convolution

Each Siemens-star patch uses the same field-position-dependent PSF as stars via
`_star_psf(params_no_halo, nx, ny, plate_scale, 61)`. The `halo` parameter is zeroed
for nebula PSF because halos are a point-source effect. Convolution uses
`scipy.signal.fftconvolve(patch, psf, mode="same")`.

### FITS traceability keywords

All generation parameters are written as `SYN_*` keywords. Key ones:
`SYN_FWHM`, `SYN_BETA`, `SYN_BRTL`, `SYN_STRL` (True on starless companion),
`SYN_SSED` (star seed = n_stars), `SYN_NSED` (noise seed).

### _SliderRow widget

`_SliderRow(lo, hi, default, decimals)` — horizontal slider + value label.
Exposes `value()`, `setValue()`, and `valueChanged` signal.
Use `_SliderRow` for all 0–0.5 aberration sliders; use `QDoubleSpinBox` for
parameters with physical units (arcsec, e⁻/ADU, etc.).
Aberration slider max is **0.5** (not 1.0) — values above 0.5 are rarely useful
and the sliders feel oversensitive at full range.

### Preview PSF scaling — `px_scale` parameter

`_star_psf(params, nx, ny, plate_scale, stamp_size, px_scale=1.0)`

When generating a downsampled preview (e.g. ¼ camera resolution), pass:

- `plate_scale / px_scale` as the plate_scale argument (correct arcsec/preview-px)
- `px_scale = preview_width / full_width` (e.g. 0.25 for 4× downsample)

This scales all pixel-unit constants inside `_star_psf` (FWHM floor, coma offset,
collimation offset, halo sigma, defocus ring radius) so the PSF appearance matches
a faithful downsample of the full-resolution image.

### QSettings persistence

Persistent UI state uses `QSettings("FilterImageComparator", "FilterImageComparator")`.
Keys in use: `last_output_dir` (main control panel), `last_data_dir` (image panel),
`synth_output_dir` (synthetic dialog). Always save on user action (browse / generate),
load on widget init after `_build_ui()` completes.

---

## Working Effectively with Claude Code

### The most useful problem statement format

A weak request describes *how* to fix something: "change the format string."
A strong request describes the *problem and the success criterion*:

> "Sky electron values display as `0.0` because `.1f` rounds values like `0.00024` to
> zero. The fix should show at least 2 significant figures for any positive value,
> including sub-electron magnitudes, without adding unnecessary digits for values like
> `150` or `2400`."

Include as many of these four elements as you know:

| Element | Example |
| --- | --- |
| Symptom | "Sky background shows 0.0 and 0.00" |
| Root cause | "GAIN=0 from FITS header was accepted as valid e⁻/ADU" |
| Constraint | "Table must allow hand-verification of σ/√μ = noise factor" |
| Acceptance criterion | "0.00024 must be readable; 150 must not gain unnecessary decimals" |

### When to use plan mode

Prefix your request with `/plan` for any change that:

- Touches more than one file
- Removes or restructures existing behaviour
- Involves a new pattern or convention not yet in this file

Claude will write a specification for your review before editing any code.

### Keeping this file current

After any session where a new pitfall or convention is discovered, ask:
> "Please add that to CLAUDE.md."

This file is the single source of truth for project context — keeping it current means
the next session starts with full context rather than re-deriving it from the code.
