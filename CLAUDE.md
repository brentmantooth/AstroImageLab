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

```text
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
  halo_dialog.py       Halo Analyzer interactive tool (QDialog); click-a-star PSF/RDF inspector
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
| `_extract_cutout(data, xc, yc, radius)` | `gui/halo_dialog.py` | 2r×2r patch centred on star, zero-padded at image edges |
| `_annular_rdf(log_data, xc, yc, radius)` | `gui/halo_dialog.py` | 1-px annular mean/std in log10 space; mirrors `HaloAnalyzer._annular_stats` |
| `_power_ratio_db(freq_a, rp_a, freq_b, rp_b)` | `report_builder.py` | 10·log10 dB ratio between two radial power curves; returns `None` on missing data or misaligned frequency bins |
| `_log_ratio_map(a, b)` | `analysis/image_filters.py` | Per-pixel `log10(\|A\|/\|B\|)` map with percentile-based epsilon floor and defensive shape crop — the Section 8 replacement for plain `A − B` diff |
| `_plot_mask_illustration(base, mask_neb, mask_bg)` | `analysis/image_filters.py` | Translucent steelblue/tomato mask overlay on a grayscale base image |
| `_plot_metric_correlation(map_a, map_b, mask_neb, mask_bg, ...)` | `analysis/image_filters.py` | 1×2 masked-region scatter (A vs B) with a 1:1 line plus overall + tail-restricted dual linear fits |
| `_family_figs_with_corr(rows, map_key_fn)` | `report_builder.py` | Emits a Section 8 family's map figure immediately followed by its `corr_*` correlation scatter, one scale at a time, in numeric order (`_SPATIAL_CORR_ROWS`) — the pattern to follow when adding any new per-scale Section 8 figure pair |

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

For dialogs that own a long-lived `Figure` (created at `__init__` time and reused across
redraws), `plt.style.use("dark_background")` does **not** recolor the existing figure
patch. After `fig.clear()` inside the dark-mode `try` block, explicitly set the patch:

```python
if _is_dark:
    bg = matplotlib.rcParams.get("figure.facecolor", "#121212")
    self._fig.patch.set_facecolor(bg)
    self._canvas.setStyleSheet(f"background-color: {bg};")
else:
    self._fig.patch.set_facecolor("white")
    self._canvas.setStyleSheet("")
```

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
convolved = fftconvolve(patch, psf_kernel, mode="same").astype(np.float32)
```

### Working dtype — always float32

All image data is converted to `np.float32` at load time (`core/astro_image.py:72`).
This is the single working dtype throughout the pipeline — `self.data`, `background_subtracted()`,
background maps from photutils, and all intermediate analysis arrays.

**float32 is sufficient:** source data is at most 16-bit integer before stacking; float32
(24-bit mantissa, ~7 significant digits) represents every possible value exactly. The
switch from float64 halves memory footprint and yields ~1.5–2× faster element-wise
operations through better cache utilisation and wider SIMD lanes.

**Byte order is handled automatically.** FITS `BITPIX=-32` images arrive from astropy as
big-endian `>f4`; `astype(np.float32)` always produces native-endian output, so there is
no need for `.byteswap()` or `.newbyteorder()`.

**Do not add float64 casts in analysis code.** The only legitimate exception in the entire
codebase is the `astroalign` registration call in `gui/analysis_thread.py`, which requires
float64 internally. That explicit cast is already in place and must stay.

**Synthetic generator internals stay float64.** `synthetic/generator.py` and
`synthetic/target_generator.py` accumulate many PSF stamps with `+=` across hundreds of
operations; float64 prevents rounding drift during synthesis. Both generators cast their
output to float32 before writing to FITS.

### Large-array reductions — prefer bottleneck

`bottleneck` (conda-forge) provides drop-in replacements for the numpy NaN-aware
and median functions that are substantially faster on large arrays (full-image or
background-map sized). Use it for any reduction that operates on arrays with
`size > ~10 000` elements. Always import with a transparent fallback:

```python
try:
    import bottleneck as bn
except ImportError:
    bn = np   # transparent fallback; bn = np must come after import numpy as np
```

**Use `bn.*` instead of `np.*` for these functions on large arrays:**

| numpy | bottleneck | Notes |
| --- | --- | --- |
| `np.median(a)` | `bn.median(a)` | Supports `axis=` parameter |
| `np.nanmedian(a)` | `bn.nanmedian(a)` | Supports `axis=` parameter |
| `np.nanmean(a, axis=)` | `bn.nanmean(a, axis=)` | NaN-aware row/col aggregation |
| `np.nanstd(a, axis=)` | `bn.nanstd(a, axis=)` | Same default `ddof=0` as numpy |
| `np.nansum(a)` | `bn.nansum(a)` | Only worthwhile when NaNs are actually present |

**Do not replace:**

- `np.nanpercentile` / `np.percentile` — bottleneck has no equivalent.
- Any reduction on arrays with fewer than ~1 000 elements — call overhead dominates.

**Currently in use:** `core/stretch.py` (stf_stretch, stf_stretch_matched),
`analysis/snr_analyzer.py` (background model median),
`analysis/halo_analyzer.py` (stacked radial profiles, RDF nanmean/nanstd),
`analysis/image_filters.py` (wavelet MAD noise estimate).

### Ratio/comparison curves in report figures — dB convention, avoid twinx()

When adding a new A-vs-B ratio curve to a report figure (precedent: `_power_ratio_db` /
`_plot_radial_ratio_db` in `report_builder.py`, Section 7's power-spectrum ratio):

- **dB convention depends on quantity type.** Power quantities (e.g. `radial_power` in
  `analysis/power_spectrum.py`, `= abs(fft2d)**2 / N**2`) use `10 * np.log10(ratio)`.
  Amplitude-like quantities (e.g. SNR in `analysis/snr_analyzer.py`) use
  `20 * np.log10(ratio)`. Using the wrong constant is silently off by 2× in dB — no
  exception, no obviously-wrong output, just a subtly incorrect number.
- **Don't add the ratio via `ax.twinx()`** onto the existing absolute-value plot unless
  both axes are the same kind of quantity (linear-vs-linear). A linear, zero-centered
  ratio next to a log-scale absolute axis has no principled vertical alignment between
  the two scales — matplotlib's independent autoscaling invents a relationship that
  isn't in the data. Build a separate, dedicated figure/panel instead. (The codebase's
  prior linear-vs-linear precedent, `_draw_cross_section`'s A−B difference line, was
  removed as unnecessary clutter — there is currently no `ax.twinx()` usage anywhere
  in the codebase, so treat this as a rule to apply fresh, not an existing pattern to copy.)
- **Guard bin alignment before dividing two arrays from different analyses.** Two
  per-image radial/frequency arrays are only safely divisible bin-for-bin when they
  share the same shape *and* values (`freq_a.shape == freq_b.shape and
  np.allclose(freq_a, freq_b)` — check shape first, since `np.allclose` raises
  `ValueError` on mismatched shapes rather than returning `False`). This is not
  guaranteed whenever an auto-selected ROI is involved (`_extract_roi` in
  `analysis/power_spectrum.py` computes `N` independently per image when no explicit
  ROI is set). Degrade gracefully — return `None` / skip the curve — rather than crash.
- **Epsilon-flooring a per-pixel ratio map needs a percentile, not a raw minimum.**
  `_power_ratio_db`'s `positive.min() * 0.01` floor is fine for small 1-D arrays
  (frequency bins, ~10²–10³ samples) but fragile at per-pixel map scale (10⁵–10⁷
  samples): the minimum order statistic over that many samples can be pathologically
  tiny and let one spurious pixel dominate the ratio's dynamic range. `_log_ratio_map`
  (`analysis/image_filters.py`) instead floors both operands at a low percentile
  (`SECTION8_LOGRATIO_EPS_PERCENTILE`, default 1st) of the pooled positive `|A|,|B|`
  values — same tool as the existing display-clipping precedent
  (`_plot_side_by_side`'s `np.percentile(arr, 0.5)`), applied to the epsilon floor
  instead of just the color scale.

### Background estimation — compute once via the pre-pass, never redundantly

`AstroImage.estimate_background()` (`core/astro_image.py`) is idempotent: it returns
immediately if `self.background is not None`, since `self.data` is only ever set once,
during `load()`. Every analyzer (`SNRAnalyzer`, `PSFAnalyzer`, `HaloAnalyzer`,
`EdgeAnalyzer`, `PowerSpectrumAnalyzer`, `SpatialDetailAnalyzer`) still calls
`estimate_background()` unconditionally at the top of its `analyze()` — that's
intentional and does not need to change; the idempotency guard just makes each of
those calls a cheap no-op once the object's background has already been computed.

`gui/analysis_thread.py::_execute()` runs a pre-pass — after alignment, before task
dispatch — that calls `estimate_background()` once per distinct `AstroImage` object
(`img_a`, `img_b`, `self._starless_a`, `self._starless_b`) via a small
`ThreadPoolExecutor`. This exists because multiple analyzers share the same image
object and can run concurrently under `parallel=True`; without the pre-pass each one
would independently trigger a full `Background2D` computation (expensive) and race to
write `self.background` / `self.background_rms` on the same object. When adding a new
analyzer that needs background stats, just call `image.estimate_background()` as
normal at the top of `analyze()` — do not add another pre-pass call site; the existing
one in `_execute()` already covers every image object the thread constructs.

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
pyinstaller AstroImageLab.spec   # build standalone binary
```

Output binary names: `AstroImageLab.exe` (Windows), `AstroImageLab` (macOS / Linux).
The spec post-build step automatically creates a platform-labelled zip:
`AstroImageLab-win64.zip`, `AstroImageLab-macos.zip`, or `AstroImageLab-linux.zip`.

PyQt6 **must** be installed via `pip`, not `conda` — the conda-forge PyQt6 package
uses a different DLL layout that breaks PyInstaller hook discovery on Windows.

For CI builds use `requirements-build.txt` (full runtime deps + PyInstaller + PyQt6).
The Linux runner needs system Qt libraries before pip:

```bash
sudo apt-get install -y libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0
```

---

## Testing

### Running tests

```bash
conda activate astrolab
pip install pytest pytest-cov pytest-timeout   # one-time setup; not in environment.yml
pytest tests/ -m "not slow"                   # fast suite (~120 s, 205 tests)
pytest tests/ -m slow                         # slow/integration tests (full FITS generation)
pytest tests/ --cov=analysis,core,synthetic,report --cov-report=html
```

### Design principles

- **Headless only** — no PyQt6 dependency in tests. Covers `analysis/`, `core/`, `synthetic/`, `report/`.
- **Generated fixtures** — `tests/conftest.py` writes a 512×512 hand-crafted FITS (30 Gaussian stars, ~1 s) as the session fixture for all analysis tests. No binary fixtures committed to git.
- **Slow marker** — `@pytest.mark.slow` gates tests that call `SyntheticGenerator.generate(preview=False)` (full 1920×1080 FITS, ~30 s). CI runs with `-m "not slow"`.
- **Smallest camera for generator tests** — `"Player One — Mercury-M"` (1920×1080 full-res; 480×270 in preview mode) is the lightest camera in `synthetic/cameras.py`.

### CI

`.github/workflows/ci.yml` — triggers on push/PR to `main`. Two jobs:

- **test** — matrix across `windows-latest`, `ubuntu-latest`, `macos-latest`; uses `requirements-test.txt` (no PyQt6). Add `CODECOV_TOKEN` to repo secrets to enable Codecov upload; coverage is flagged per OS.
- **build** — runs after all test jobs pass (`needs: test`); same OS matrix; uses `requirements-build.txt`; uploads `AstroImageLab-{win64,macos,linux}.zip` as workflow artifacts downloadable from the Actions tab.

### Test fixture pitfalls

| Pitfall | Rule |
| --- | --- |
| Inline FITS too small to load | `_load_fits` skips HDUs where `max(shape) <= 100`. Test FITS must be at least 101×101; use 128×128 for safety. |
| `float(mtf(array, m))` raises TypeError | `mtf()` returns a same-shape array, not a scalar. Index with `[0]` or pass a scalar input. |
| `PSFAnalyzer.analyze()["figures"]` KeyError | `figures` is only added when `n_stars_used > 0`. Guard with `if result["n_stars_used"] > 0`. |
| `contrast_ratios_b` / `weber_contrast_b` always present | `SpatialDetailAnalyzer.analyze()` always includes `contrast_ratios_b: {}` and `weber_contrast_b: {}` even in single-image mode. Neither is ever `None` or absent — check `not b_ratios` / `not wc_b` instead. |
| Background2D fails on tiny images | `estimate_background()` with default `box_size=64` needs the image to be larger than the box. Any image used in analysis tests should be at least 128×128. |

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
| `secondary_xaxis` accumulates across redraws | `ax.cla()` does not remove secondary axes — always use `fig.clear()` + `fig.subplots(1, N)` when any axis has a secondary x-axis. |
| Pre-created `Figure` stays white in dark mode | `plt.style.use("dark_background")` updates rcParams but does **not** recolor an already-constructed `Figure` object. After `fig.clear()`, explicitly set `fig.patch.set_facecolor(matplotlib.rcParams.get("figure.facecolor", "#121212"))` and `canvas.setStyleSheet(f"background-color: {bg};")` when dark mode is active; reset both to `"white"` / `""` in light mode. |
| Circle overlay after `super().paintEvent()` needs a fresh QPainter | Calling `super().paintEvent(event)` ends the parent's painter. Create `p = QPainter(self)` on the next line to draw custom overlays; do not attempt to reuse the parent's painter object. |
| Closure capture in `secondary_xaxis` lambdas | `lambda x: x * ps` inside a loop captures `ps` by reference. Use default-arg capture: `lambda x, p=ps: x * p` to freeze the value at definition time. |
| macOS binary blocked by Gatekeeper | CI-built binaries are unsigned. Users must right-click → Open, or run `xattr -dr com.apple.quarantine AstroImageLab` in Terminal. Code signing requires an Apple Developer certificate ($99/year). |
| Linux build needs system Qt libraries | PyInstaller must be able to import PyQt6 during analysis. On `ubuntu-latest` run `sudo apt-get install -y libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0` before `pip install -r requirements-build.txt`. |
| `PowerSpectrumAnalyzer` crashes on images smaller than 2048 px | `POWER_SPECTRUM_NPIX = 2048`. The auto-select loop is empty when `min(h, w) < 2048`; the fallback produces negative slice indices → non-square region → `_apply_window` shape mismatch. Fix: `N = min(N, h, w)` before the loop, add `+1` to loop upper bounds, clamp fallback with `max(0, ...)`. |
| `sigma_clip` mask is scalar `False` when nothing is clipped | `clipped.mask` is `np.ma.nomask` (== `False`) when no values are clipped. `region[False]` silently writes only the first row. Use `np.ma.getmaskarray(clipped)` to get a full bool array, then guard with `.any()`. |
| Adding a float64 cast in analysis code | Don't. All image data is float32 after `AstroImage.load()`. The only float64 exception is `astroalign` in `gui/analysis_thread.py`. Redundant float64 casts waste memory and defeat the float32 performance gains. |
| Mixed float32/float64 arithmetic silently widens to float64 | NumPy upcasts when operands differ (e.g. `float32_array - float64_scalar`). If photutils ever returns a float64 background model, `background_subtracted()` will silently return float64. Guard by adding `.astype(np.float32)` at the end of `background_subtracted()` in `astro_image.py` if this is observed. |
| `_section_snr` crashed when SNR metric is unchecked | `_plot_snr_pair` (`report_builder.py`) built a `panels` list filtered to non-`None` entries but never checked whether it was empty before calling `plt.subplots(1, len(panels), ...)` — 0 columns raised `ValueError: Number of columns must be a positive integer, not 0`. Hit whenever SNR is unchecked while another metric (e.g. Power Spectrum) is run. Fixed with an early `if not panels: return None` guard, matching `_plot_radial_overlay`/`_plot_radial_ratio_db`; both call sites already pipe the result through `_img_tag`, which turns `None` into `""`. |
| Whole-population regression slope hides tail-specific divergence | A single OLS fit over an entire A-vs-B scatter is dominated by leverage (distance from the mean), not point count — it can show near-zero correlation even when a sparse high-value tail diverges sharply (or vice versa), because it blends "the bulk agrees" and "the tail diverges" into one ambiguous number. When the signal of interest lives specifically in the tail (Section 8g: "which filter shows more detail in the brightest/most-structured pixels"), fit two lines — an overall fit and a second fit restricted to the top N% by combined magnitude (`_plot_metric_correlation`, `SECTION8_SCATTER_TAIL_PERCENTILE`) — and report slope + R² for each. High tail R² with slope far from 1 is a real, systematic effect; low tail R² means the apparent divergence is mostly noise. |
| New Section 8 panel key doesn't need Report Inspector code changes | `gui/report_inspector.py` is fully generic — driven entirely by a companion `<stem>_inspector.npz` (raw float32/uint8 arrays) plus an embedded `catalog_json` built in `report_builder.py::_write_inspector_file`. `_panel_display_name`/`_panel_concept` dynamically parse any `panels` dict key prefix, so a new `SpatialDetailAnalyzer` panel family auto-appears in the inspector with zero inspector-side changes. A genuinely new *visual type* is a different story: the inspector only knows how to `imshow` 2D/RGB arrays (side-by-side or slider-reveal), so scatter-style plots (Section 8's `corr_*` correlation figures, interleaved into 8b–8f right after each map figure via `_family_figs_with_corr`) must stay static-HTML-only unless new inspector canvas code is written. |
| Renumbering a Section 8 subsection misses caption cross-references | Section 8's sub-heading letters (8a–8g) are referenced by literal string in caption/info-box text scattered throughout `_section_spatial` — not just in the `<h3>` tags (e.g. "see 8g for…", "(8b–8f, 8g)"). After adding, removing, or renumbering a subsection, `grep` the function for every old *and* new heading letter — HTML renders a stale cross-reference without error, it just silently misdirects the reader to the wrong subsection. |
| Stale ROI crashes Section 8 with "index -1 is out of bounds for axis 0 with size 0" | `MainWindow._roi` (`gui/main_window.py`) is never cleared when a new image is loaded into either panel. If the user draws an ROI on one image pair, then loads a smaller replacement pair without clearing it, the stale coordinates go out of bounds for the new image. NumPy doesn't raise on an out-of-range slice — `norm_a[ry0:ry1, rx0:rx1]` silently returns a zero-size array — so the crash surfaces much later and far from the real cause: `SpatialDetailAnalyzer._plot_mask_illustration → _stretch_for_display → np.percentile(empty_array, ...)`. The same unguarded `bgsub[y0:y1, x0:x1]` pattern exists in `power_spectrum.py::_extract_roi` and `edge_analyzer.py::analyze`, so a stale ROI can corrupt those sections too (with different, equally misleading errors) if they happen to run. Fixed at the single real boundary — `MainWindow._on_run()`, which is the only path that constructs `AnalysisThread` — by validating `self._roi` against every loaded image's `data.shape` right before `settings["roi"]` is set; an out-of-bounds ROI is cleared (falls back to auto-detect/full-image) with a `QMessageBox` explaining why, rather than patching each analyzer's slice individually. |

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

## Halo Analyzer Tool — Key Patterns

### Thread architecture

Two background threads are used; they run sequentially (detect → analyze):

```text
_DetectThread   (runs once on dialog open)
  StarCatalogBuilder.build(img_a)  → stores Nx3 ndarray (x, y, peak)
  StarCatalogBuilder.build(img_b)  → same for Image B (if loaded)
  → _on_detect_done: sets self._stars_a / _stars_b, enables clicking

_AnalyzeThread  (runs on each star click or radius change)
  _fit_moffat(bgsub, xc, yc)       → Moffat2D on 25-px core (matches PSFAnalyzer size)
  _shape_metrics(bgsub, xc, yc)    → data_properties: ecc, ell, orientation
  _outer_stats(bgsub, xc, yc, r)   → background, SNR, peak from outer annulus
  _annular_rdf(log_bgsub, xc, yc, r) → 1-px annular mean/std in log10 space
  → _on_analysis_done: updates table + redraws figure
```

When a new analysis request arrives before the previous thread finishes, disconnect
its signals then call `.quit()` — do **not** call `.wait()`, which would block the GUI.
The old thread finishes silently; its result is discarded.

### ZoomableImageLabel subclassing

`_StarImageLabel` extends `ZoomableImageLabel` to add star-click selection and a circle
overlay. Key rules:

- Override `mousePressEvent`: check `not self._roi_mode and not self._line_mode` before
  intercepting left-click; pass everything else to `super().mousePressEvent(event)`.
- Override `paintEvent`: call `super().paintEvent(event)` first, then create a new
  `QPainter(self)` to draw the circle — the parent's painter is already ended.
- Store the circle in normalised coordinates `(xn, yn, rn)` so it scales correctly
  through zoom and pan without any extra math.

### Matplotlib figure in a dialog

Use `matplotlib.figure.Figure()` directly (not `plt.subplots()`) to avoid touching
global pyplot state. Redraw by clearing the whole figure each time:

```python
self._fig = Figure(figsize=(12, 3))
self._canvas = FigureCanvasQTAgg(self._fig)
# ...on each update:
self._fig.clear()
axes = self._fig.subplots(1, 4)
# draw into axes[0..3]
self._fig.tight_layout(pad=1.0)
self._canvas.draw_idle()
```

`fig.clear()` is required (not `ax.cla()`) because secondary x-axes are separate
`Axes` objects that `cla()` does not remove.

### Dual x-axis (pixels + arcseconds)

```python
ps = img_a.pixel_scale   # arcsec/px; always > 0 (DEFAULT_PIXEL_SCALE if no WCS)
if ps > 0:
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(lambda x, p=ps: x * p, lambda x, p=ps: x / p))
    ax_top.set_xlabel('"', fontsize=7)
```

Use default-arg capture (`p=ps`) to freeze the plate-scale value — bare `lambda x: x * ps`
captures `ps` by reference and breaks when used inside loops.

### Star matching between A and B

Images are assumed to be co-registered. The nearest star in Image B's catalog within
50 px of the clicked A star is used. 50 px is intentionally generous — tighter thresholds
reject valid matches when residual alignment offset exists.

---

## Spatial Target Generator — Key Patterns

### Purpose and workflow

`gui/spatial_target_dialog.py` + `synthetic/target_generator.py`

Generates a 4-column × 3-row grid of calibrated test zones at known spatial frequencies,
always as a clean/degraded FITS pair. Load clean → Image A and degraded → Image B to
calibrate the spatial-detail metrics against known inputs.

### Target signal chain

```text
SpatialTargetDialog.targets_generated = pyqtSignal(str, str, str)  # clean_path, degraded_path, mode
  → MainWindow._on_target_generated(clean_path, degraded_path, mode)
      # mode: "clean_a_deg_b" | "deg_a_clean_b" | "deg_a" | "deg_b" | ""
```

`_TargetThread.finished = pyqtSignal(str, str)` (clean, degraded) feeds `_on_gen_done`
which then emits the three-arg `targets_generated` signal.

### Target return types

`SpatialTargetGenerator.generate(params, preview=False)`:

- `preview=True` → `np.ndarray` (float32, degraded image at reduced resolution)
- `preview=False` → `tuple[str, str]` (clean_path, degraded_path)

### Zone layout

```text
Row 0: Sine H  f=0.04 | Sine H  f=0.08 | Sine H  f=0.16 | Sine H  f=0.32  (c/px)
Row 1: Square H  0.04 | Square H  0.08 | Square H  0.16 | Square H  0.32
Row 2: Sine V  0.08  | Sine 45°  0.08 | Siemens star   | Slant edge ~5°
```

Column frequencies align with wavelet levels: 0.04→L4, 0.08→L3, 0.16→L2, 0.32→L1.

### Contrast ramp

Each zone ramps Michelson contrast linearly from `contrast_min` (top edge) to
`contrast_max` (bottom edge). At any horizontal strip, all four columns share
the same contrast — enabling direct cross-frequency comparison. Params:
`contrast_min`, `contrast_max` (both 0–1, default 0.02 / 0.50).

### FITS keywords

`INSTRUME="SpatialTarget"`, `EGAIN=1.0`, `GAIN=1.0`, `TGT_TYPE`, `TGT_ROWS`,
`TGT_COLS`, `TGT_CMIN`, `TGT_CMAX`, `TGT_SKY`, `TGT_CLEN` (bool: clean flag),
per-zone `TGT_{r}{c}F` / `TGT_{r}{c}W`. Optional: `TGT_FWHM`, `TGT_BETA`, `TGT_RN`.
No `FOCALLEN`, `APTDIA`, `FOCRATIO`, `XPIXSZ`, `EXPTIME` — these are set from
`AstroImage` defaults (pixel scale = `DEFAULT_PIXEL_SCALE`).

### Power spectrum on spatial target images

The power spectrum auto-selects one square ROI — not the whole zone grid. The result
reflects whichever zone(s) fall inside that square. Use the explicit crosshair ROI
(user-drawn in the image panel) to target a specific zone for a focused power spectrum.
The frequency axis is always cycles/pixel regardless of ROI size.

### QSettings key

`"target_output_dir"` (separate from the synthetic dialog's `"synth_output_dir"`).

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
