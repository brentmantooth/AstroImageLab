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
AstroImageLab.py       PyQt6 app; splash screen loads resources/AstroImageLabSplash.png
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
  stats_utils.py       mannwhitney_effect() — Mann-Whitney U + Cliff's delta, shared by analysis/ and report/
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
| `_info_box(body, title, open=False, style="")` | `report_builder.py:197` | Collapsible `<details>/<summary>` HTML panel — `body` is raw HTML, so it also wraps whole figure-heavy blocks (not just prose) closed by default to keep the report compact; see "Collapsible figure blocks" below |
| `_val(v, fmt, fallback="—")` | `report_builder.py:184` | Null-safe table cell formatter |
| `fig_to_b64(fig)` | `core/fig_utils.py` | Embeds matplotlib figure as base64 PNG string |
| `finalize_layout(fig, **kwargs)` | `core/fig_utils.py` | Runs `fig.tight_layout(**kwargs)` under the same process-wide lock as `fig_to_b64()`'s `savefig()`. Call this instead of `fig.tight_layout()` directly in any figure-building code that can execute concurrently with other figure-building code — see the mathtext race pitfall below |
| `normalize_for_display(arr)` | `core/stretch.py` | STF-stretch float32 array → uint8 [0,255] for QImage display |
| `stf_stretch(data)` | `core/stretch.py` | STF midtone-balance stretch → float32 [0,1]; maps sky to ~20 % grey |
| `load_path(path)` | `gui/image_panel.py` | Load image by path with no dialog and no starless prompt |
| `set_starless_path(path)` | `gui/image_panel.py` | Attach a pre-generated starless FITS to the loaded main image |
| `_extract_cutout(data, xc, yc, radius)` | `gui/halo_dialog.py` | 2r×2r patch centred on star, zero-padded at image edges |
| `_annular_rdf(log_data, xc, yc, radius)` | `gui/halo_dialog.py` | 1-px annular mean/std in log10 space; mirrors `HaloAnalyzer._annular_stats` |
| `_power_ratio_db(freq_a, rp_a, freq_b, rp_b)` | `report_builder.py` | 10·log10 dB ratio between two radial power curves; returns `None` on missing data or misaligned frequency bins |
| `_log_ratio_map(a, b)` | `analysis/image_filters.py` | Per-pixel `log10(\|A\|/\|B\|)` map with percentile-based epsilon floor and defensive shape crop — the Section 8 replacement for plain `A − B` diff |
| `_log_ratio_color_range(diff)` | `analysis/image_filters.py` | Symmetric `(vmin, vmax)` for the `bwr` log-ratio colormap, shared by the log-ratio map panel, its histogram, and the correlation scatter dot coloring |
| `_plot_mask_illustration(base, mask_neb, mask_bg)` | `analysis/image_filters.py` | Translucent steelblue/tomato mask overlay on a grayscale base image |
| `_plot_metric_correlation(map_a, map_b, log_ratio, mask_neb, mask_bg, ...)` | `analysis/image_filters.py` | 1×2 masked-region scatter (A vs B) with a 1:1 line; each point colored by its pixel's log-ratio value using the same `bwr` scale as the adjacent map figure |
| `_family_figs_with_corr(rows, map_key_fn)` | `report_builder.py` | Emits a Section 8 family's map figure immediately followed by its `corr_*` correlation scatter, one scale at a time, in numeric order (`_SPATIAL_CORR_ROWS`) — the pattern to follow when adding any new per-scale Section 8 figure pair |
| `_family_nrm_figs(rows)` | `report_builder.py` | Sibling of `_family_figs_with_corr` for the noise-normalised (`nrm_*`) trailer figures — same `_SPATIAL_CORR_ROWS`-ordered iteration, no `map_key_fn` needed since the nrm key is always `"nrm_" + row_key`. Always use this (never a raw `sorted(figs)` scan) for any new per-scale trailer block — `sorted()` on figure-key strings is lexicographic and puts `nrm_std_10px` before `nrm_std_3px`/`5px` |
| `SpatialDetailAnalyzer._crosshair_to_cropped_px(crosshair, shape, crop_fraction)` | `image_filters.py` | Converts a normalised `[0,1]` crosshair dict into pixel coords in the frame `_crop_border(arr, crop_fraction)` produces — the pattern for overlaying the user's cross-section line directly onto a Section 8 map panel (`_plot_side_by_side`'s `xs_line` param), as opposed to `xs_data`'s separate line-chart profile panel |
| `SpatialDetailAnalyzer._local_maxima_mask(data, footprint_px, prominence_percentile, region_px, presmooth_sigma)` | `analysis/image_filters.py` | Scale-adaptive peak mask: `maximum_filter` non-max suppression + percentile prominence threshold + optional pre-smoothing (suppresses noise-driven false peaks) + `binary_dilation` region growth. All params are relative to the caller's own characteristic scale, not fixed pixel counts — see the Local-maxima detection convention below |
| `SpatialDetailAnalyzer._combined_localmax_mask(abs_a, abs_b, footprint_px, prominence_percentile, region_px, presmooth_sigma, top_percent)` / `_top_percent_mask(abs_a, abs_b, top_percent)` | `analysis/image_filters.py` | `_local_maxima_mask` unioned (OR) with a top-`top_percent`% brightness mask — catches broad bright plateaus a sharp-peak detector alone would miss. Used identically by `_localmax_entry` (Section 8j table stats) and the mask-grid figure builder in `analyze()`, so the displayed mask always matches the mask backing that row's numbers |
| `mannwhitney_effect(va, vb)` | `core/stats_utils.py` | Mann-Whitney U p-value + Cliff's delta in O(n log n) via `delta = 2·U/(n1·n2) − 1`; shared by `_psf_stat_test` (Section 4) and `SpatialDetailAnalyzer._localmax_stats` (Section 8j) — never reimplement Cliff's delta via a pairwise `arr_a[:, None] - arr_b[None, :]` matrix, it's O(n·m) memory |
| `_format_significance_html(p, delta)` / `_sig_td(html, p)` | `report_builder.py` | Shared star-rating/p-value HTML cell + colored `<td>` wrapper for any Mann-Whitney significance column — used by both Section 4's PSF table and Section 8j's local-maxima table |
| `SpatialDetailAnalyzer._ratio_series_with_errors(ratios_by_method, errors_by_method=None)` | `analysis/image_filters.py` | `{method: {scale: value}}` (+ optional matching errors) → sorted `{method: [(x_px, value, error_or_None), ...]}` point lists for a cross-method overview line plot; shared by `_plot_nc_ratio_overview` (8i) and `_plot_localmax_ratio_overview` (8j) |
| `_nc_ratio_rows(score_a, score_b, ratio, scale_label, val_fmt=".3f", method_label=None)` | `report_builder.py` | `<tr>` rows for a noise-corrected score table (Scale \| A \| B \| Ratio A/B). Pass `method_label` to prepend a Method-name `<td>` when consolidating several per-family tables that share this schema into one combined table — precedent: Section 8j's cross-method NC table, which concatenates the row-strings from all five `_nc_ratio_rows` calls (LoG/Wavelet/Gradient/Std/Entropy) into a single `<table>` |

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

### Collapsible figure blocks — reuse `_info_box` for images, not just text

`_info_box(body, title, open=False, style="")` (`report_builder.py:197`) only ever
wrapped prose/methodology text until Section 8's families got collapsed — its `body`
parameter is raw HTML, so it works unchanged for a block of embedded `<img>` tags.
To collapse a figure-heavy block (precedent: Section 8's five metric families 8d–8h,
and Section 4's PSF test-chart image sequence in `_psf_simulation_html`), **pre-compute
the image HTML as its own variable before the enclosing `return f"""..."""`** — same
rule as "Long f-string HTML blocks" above, since an f-string expression can't contain
statements — then interpolate `{_info_box(images_html, title="Show ...", open=False)}`
in place of the raw figure calls. Keep the section heading, methodology `_info_box`,
and any data table **outside** the collapsible: those are what a reader needs even with
the images hidden, and default-closed means the collapsible content shouldn't be load-
bearing for understanding the section at a glance.

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

### Local-maxima / peak detection — scale-relative parameters, not fixed pixel values

`SpatialDetailAnalyzer._local_maxima_mask` (Section 8j) detects peaks via
`data == maximum_filter(data, size=footprint_px)` AND `data > percentile(data, prominence_percentile)`,
then grows survivors with `binary_dilation(mask, iterations=region_px)`. All four
tunables are expressed **relative to the caller's own characteristic scale**
(`footprint_px = footprint_mult * scale_px`, `region_px = region_fraction * footprint_px`,
`presmooth_sigma = presmooth_fraction * scale_px`), never as fixed absolute pixel
counts — a std-3px kernel and a wavelet level-3 (~8px) feature need genuinely
different "how local" and "how tall" thresholds, and one fixed setting either misses
fine structure or over-merges coarse structure. Pre-smooth the peak-source array
(`gaussian_filter`, sigma also scale-relative) **before** non-max suppression to
prevent single-pixel noise from registering as spurious peaks — reuse this whenever
applying the pattern to new noisy per-pixel data; skipping the pre-smooth step lets
shot noise dominate the detected-peak count. See `core/models.py`'s five
`SECTION8_LOCALMAX_*` constants for the current default multipliers/fractions.

A pure peak detector still undersamples broad bright plateaus that never register as a
sharp local maximum (every pixel ties for "the max of its own neighbourhood").
`_combined_localmax_mask` unions the peak mask with `_top_percent_mask` — pixels in the
top `SECTION8_LOCALMAX_TOP_PERCENT`% of Image A's or Image B's own value distribution —
so broad-but-real bright regions are still captured. Both the table's per-row
statistics (`_localmax_entry`) and the mask-grid display figure (built inside
`analyze()`) call this **one shared method**; never let the mask backing a statistic
and the mask drawn in its illustrative figure be two independent implementations of
the same idea — they will silently drift apart the next time either one gets a
formula change.

### Ratio uncertainty / error bars — exact when pixel-paired, approximate (CV-propagated) otherwise

When adding an error bar to a ratio-vs-scale plot (precedent: Section 8i/8j's
cross-method overview figures, `_plot_nc_ratio_overview` / `_plot_localmax_ratio_overview`
in `image_filters.py`), first check whether the two populations behind the ratio are
**pixel-paired** (same pixel coordinates in both images):

- **Pixel-paired** (Section 8j: `diff[mask]` is a genuine per-pixel `log10(|A|/|B|)`
  population) → take `std(diff[mask])` directly and delta-method-propagate it into
  linear ratio units (`ratio * ln(10) * log_std`) — an exact spread measure.
- **Not pixel-paired** (Section 8i: nebula vs. background populations, computed
  independently per image) → there is no per-pixel ratio distribution to take a std
  of. Use a standard relative-uncertainty (coefficient-of-variation) propagation
  instead: `err = |ratio| * sqrt((std_a/median_a)² + (std_b/median_b)²)`. **This is
  an approximation, not a formal confidence interval** — caption it as such in the
  report (see 8i's methodology caption) rather than presenting it as exact.

Both plot functions share `_ratio_series_with_errors` for the point-list-building
loop; only the upstream computation of the error value differs by data source.

**Prefer presenting a log-space quantity in log space throughout, rather than
converting back to linear for display.** Section 8j's ratio column and cross-method
overview originally stored `ratio = 10**mean(diff[mask])` and converted `log_ratio_std`
into a linear error bar via the delta method (`ratio * ln(10) * log_std`) — correct,
but an avoidable approximation-flavoured extra step. Since `diff[mask]` *is* a log10
population, carrying `log_ratio_mean`/`log_ratio_std` straight through to the table
(`_val_pm`) and the overview plot's y-axis removes the conversion entirely — the error
bar becomes exact by construction instead of merely well-approximated, and the table
header should say so (`"log ratio A/B (geo. mean ± SD)"`, not `"Ratio A/B"`). Shade
this kind of column a fixed neutral color (not `_better_worse_class` red/green) when
it isn't a value judgement between A and B, just a measured quantity.

### Toolbar actions that mirror control-panel widgets — share QActions, proxy clicks, sync reactively

`gui/main_window.py::_build_toolbar()` is the app's first `QToolBar`, sitting above the
image-panel splitter to give the load → select-region → run workflow a visible
left-to-right order (Open Image A / Open Image B │ Select ROI / Select Line │ Run
Analysis). It never introduces a second state machine alongside `AnalysisControlPanel`'s
existing widgets — extend this pattern for any future toolbar addition:

- **Reuse the same `QAction` object in both the menu and the toolbar** when one already
  exists (`self._act_open_a`, `self._act_open_b`, `self._act_run` — promoted from local
  variables in `_build_menu` to instance attributes specifically so `_build_toolbar` can
  add them a second time). A `QAction` can live in multiple containers simultaneously —
  this is what it's for, not a hack.
- **For a checkable control-panel `QPushButton` with no existing menu equivalent** (ROI,
  Line), add a plain **non-checkable** `QAction` whose `triggered` handler calls
  `.click()` on the real button (`self._control._roi_btn.click()`) — this drives the
  entire existing toggle/signal chain unmodified instead of maintaining a second
  checked-state to keep in sync. Reflect the active/cancel state by setting the
  toolbar action's text inside the *existing* mode-toggled slot
  (`_on_roi_mode_toggled`/`_on_line_mode_toggled`), which already fires on every path
  that changes mode (control-panel click, toolbar click, and post-selection auto-reset).
- **For enable/disable state** that must track a control-panel widget (Run), add one
  wrapper method (`_set_run_enabled`) that updates both the button and the action, and
  route every call site through it — do not leave the toolbar/menu action's enabled
  state to drift independently from the button's.
- **Do not use a `QToolBar.addWidget(spacer)` with an `Expanding` `QSizePolicy` to
  right-align a trailing action.** In this environment it made every action added after
  the spacer disappear entirely — not merely misplaced — regardless of whether the
  spacer's vertical policy was `Preferred` or `Expanding`. Removing the spacer (`Run
  Analysis` simply follows the last separator, left-aligned like everything else) fixed
  it immediately. If a right-aligned toolbar action is ever genuinely needed, verify the
  spacer approach renders before relying on it — don't assume the common Qt idiom is
  safe in this codebase's environment without checking.

### Fixed-size markers in report figures — matplotlib `markersize`, not a data-space patch

When a marker must stay a constant *visual* size regardless of the image's zoom or
pixel scale (precedent: the red "scan start" square in Section 6's edge ROI figure and
its ESF/LSF profile chart, `EdgeAnalyzer._plot_results` / `_plot_esf_lsf_pair`), use
`ax.plot(x, y, marker='s', markersize=N, color=...)` — `markersize` is in points
(1/72 inch), independent of the axes' data-coordinate scaling. Do **not** use a
`Rectangle` patch sized in data coordinates for this (e.g. `Rectangle((x, y), 5, 5)`)
— its apparent size changes with the image's pixel scale/zoom, the opposite of what
"fixed size" means here.

### Locating a point across `scipy.ndimage.rotate()` without hand-deriving its angle-sign convention

`EdgeAnalyzer._extract_esf` rotates an ROI so the edge runs vertical, then (to place
the "scan start" marker) needs to know which point in the *original*, unrotated frame
corresponds to a specific column of the *rotated* frame. Hand-deriving `rotate()`'s
counter-clockwise-vs-clockwise / y-down handedness risks a silently mirrored result
with no exception raised. Instead, use `rotate()` itself in both directions: place a
single-pixel impulse at the target column in a same-shaped zero array, apply the
**inverse** rotation (`rotate(impulse, -angle, reshape=False, order=1)`), and take
`argmax` of the result. This is correct by construction — the same function, forward
then backward — regardless of the library's sign convention, and generalizes to any
"where did this rotated-frame coordinate come from" problem.

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
pytest tests/ -m "not slow"                   # fast suite (~11 min, 430 tests)
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
| `contrast_ratios_b` / `entropy_contrast_ratio_b` always present | `SpatialDetailAnalyzer.analyze()` always includes `contrast_ratios_b: {}` and `entropy_contrast_ratio_b: {}` even in single-image mode. Neither is ever `None` or absent — check `not b_ratios` / `not ecr_b` instead. |
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
| New Section 8 panel key doesn't need Report Inspector code changes | `gui/report_inspector.py` is fully generic — driven entirely by a companion `<stem>_inspector.npz` (raw float32/uint8 arrays) plus an embedded `catalog_json` built in `report_builder.py::_write_inspector_file`. `_panel_display_name`/`_panel_concept` dynamically parse any `panels` dict key prefix, so a new `SpatialDetailAnalyzer` panel family auto-appears in the inspector with zero inspector-side changes. A genuinely new *visual type* is a different story: the inspector only knows how to `imshow` 2D/RGB arrays (side-by-side or slider-reveal), so scatter-style plots (Section 8's `corr_*` correlation figures, interleaved into 8d–8h right after each map figure via `_family_figs_with_corr`) must stay static-HTML-only unless new inspector canvas code is written. |
| Renumbering a Section 8 subsection misses caption cross-references | Section 8's sub-heading letters (currently 8a–8j: 8a Background/Key Terms, 8b Original Image, 8c Mask Overview, 8d–8f detail-based families [LoG, Wavelet, Gradient], 8g–8h contrast/texture-based families [Local σ, Local Entropy], 8i Noise-Corrected Cross-Method Overview, 8j Local-Maxima Masked Metrics) are referenced by literal string in caption/info-box text scattered throughout `_section_spatial` — not just in the `<h3>` tags (e.g. "see 8i for…", "(8d–8h, 8i)"). After adding, removing, or renumbering a subsection, `grep` the function (and `_SPATIAL_GLOSSARY_HTML`) for every old *and* new heading letter — HTML renders a stale cross-reference without error, it just silently misdirects the reader to the wrong subsection. A self-reference inside a family's own info box (e.g. Gradient's "same framework as the other N families") must enumerate the other letters explicitly rather than use a dash range — under a non-contiguous lettering (Gradient kept letter `8f` while Std/Entropy moved past it into the Contrast group), a `8d–8h` range would wrongly include Gradient's own letter. Precedent: when Weber contrast (formerly 8h) was replaced by Local Entropy, keeping the same letter for the new family avoided a renumbering pass entirely — every existing `8h`/`8d–8h` cross-reference stayed numerically valid, only the prose describing 8h's content changed. |
| Stale ROI crashes Section 8 with "index -1 is out of bounds for axis 0 with size 0" | `MainWindow._on_image_loaded()` (`gui/main_window.py`) now resets `self._roi`/`self._crosshair` to `None` (plus both panels' visual overlays, via `ImagePanel.clear_roi_overlay()`/`clear_line_overlay()`) on every new main-image load — the choke point is `ImagePanel.image_loaded`, emitted only from `_open_file`/`load_path`, never from `set_starless_path`, so attaching a starless companion correctly does *not* wipe an existing ROI/line. Before this proactive reset existed, a stale ROI drawn against a previous, larger image pair silently went out of bounds for a smaller replacement: NumPy doesn't raise on an out-of-range slice — `norm_a[ry0:ry1, rx0:rx1]` silently returns a zero-size array — so the crash surfaced much later and far from the real cause: `SpatialDetailAnalyzer._plot_mask_illustration → _stretch_for_display → np.percentile(empty_array, ...)`. The same unguarded `bgsub[y0:y1, x0:x1]` pattern exists in `power_spectrum.py::_extract_roi` and `edge_analyzer.py::analyze`, so a stale ROI could corrupt those sections too. `MainWindow._on_run()`'s validation (checking `self._roi` against every loaded image's `data.shape` right before `settings["roi"]` is set, clearing it with a `QMessageBox` if it no longer fits) is kept as defense-in-depth for any future code path that changes loaded-image dimensions without going through `_on_image_loaded`, but the normal load→run flow now clears stale state at the source instead of catching it reactively at Run time. |
| `binary_dilation` on a loose sigma-threshold mask amplifies noise, not signal | Growing a boolean mask straight from a threshold cut (e.g. Section 8's nebula mask at 1.7σ) dilates *every* True pixel, including scattered single/few-pixel noise-driven false positives — expected in bulk at a loose sigma cut (~4.5% of pixels at 1.7σ one-sided). Each isolated speck balloons into a `~(2·dilation_px+1)²`-px blob, inflating the mask area by 4x+ and diluting any signal-vs-background metric computed over it. Fix: strip small isolated connected components (`scipy.ndimage.label` + `np.bincount` size filter, same size threshold used for hole-filling) *before* calling `binary_dilation` — see `SpatialDetailAnalyzer._remove_small_objects` / `_fill_small_holes` in `image_filters.py`. Caught by comparing mask pixel counts with dilation on vs off on real (noisy) fixture data — a clean synthetic square mask (no noise) will not reveal this bug. |
| Adding GUI parameter rows clips existing text in the Parameters group | `gui/main_window.py`'s `AnalysisControlPanel.setMaximumHeight(...)` caps the whole control panel's height. Metrics / Parameters / Region & Run are laid out side-by-side (`QHBoxLayout` in `control_panel.py::_build_ui`), so the cap must fit the *tallest* group box's natural content height. Parameters (`control_panel.py`, "2. Parameters") is itself split into two side-by-side `QFormLayout`s — "General / PSF" (`form1`) and "Nebula & Local-Maxima" (`form2`) — so its own height is driven by `max(form1_rows, form2_rows)`, not the total row count across both. When adding a new parameter row, add it to whichever column keeps the two roughly balanced, then re-measure: construct `AnalysisControlPanel` headlessly and read `QGroupBox.sizeHint()` for all three boxes (see the measurement approach used when this split was introduced — a small script that imports the widget, calls `.adjustSize()`, and prints each `findChildren(QGroupBox)` entry's `sizeHint()` — is far more precise than eyeballing a screenshot, and sidesteps OS/DPI screenshot-scaling inconsistencies entirely) rather than assuming a fixed per-row pixel cost. Set `setMaximumHeight(...)` to comfortably cover the tallest of the three measured heights. |
| Cliff's delta via pairwise sign matrix doesn't scale past ~100s of samples | `arr_a[:, None] - arr_b[None, :]` is O(n1·n2) memory — fine for PSF's per-star counts, a multi-gigabyte blowup for per-pixel populations (Section 8j masks can hold 10⁴–10⁵+ pixels). Use the exact O(n log n) identity instead: `delta = 2·U/(n1·n2) − 1`, where `U` is the Mann-Whitney U statistic `scipy.stats.mannwhitneyu` already computes. See `core/stats_utils.py::mannwhitney_effect`. |
| Section HTML block gated on the wrong figure's output | A conditional HTML block that wraps *multiple* pieces of content but gates on only *one* figure's presence (e.g. `dist_html = ("<h3>...</h3>" + mask_html + dist_img + ...) if dist_img else ""`) will silently delete the other content too if that one figure is ever removed or becomes empty — this hit Section 8c, whose Nebula/Background mask illustration disappeared along with the (later-removed) log-ratio violin figure it happened to share a block with. Gate on the actual content being wrapped (e.g. `mask_fig`, if that's what must be present for the block to be worth showing), not on a sibling figure that currently always co-occurs with it. |
| Extending an analyzer method's return-tuple arity misses a call site | `_nc_score` gained a 3rd return value (`neb_std`) to support Section 8i's error bars; it has 10 call sites (2 per metric family × 5 families), each needing `nc_a, noise_a = ...` changed to `nc_a, noise_a, neb_std_a = ...`. `grep` for every call site before changing a shared method's return signature — a missed site raises `TypeError: cannot unpack non-sequence`, or worse, silently mis-assigns if old and new arity both happen to unpack without error. |
| `np.percentile` threshold on a flat/constant array selects everything | A percentile-based `>=`-threshold mask (e.g. `_top_percent_mask`) degenerates when the source array has little/no spread: if every value is equal, every percentile equals that one value, so `arr >= threshold` matches 100% of pixels, not the intended top N%. Hit writing a unit test for `_top_percent_mask` with an all-zero second operand — its own 90th-percentile threshold was also `0`, making `arr_b >= 0` trivially true everywhere and swamping the assertion. When constructing a synthetic array to exercise percentile-threshold logic, give it genuine spread (or reuse the same non-flat array for both operands) rather than an all-zero/constant placeholder. |
| A cached render-to-attribute call duplicates report content | `_psf_simulation_html` called `self._psf_retention_table(sim)` twice with identical arguments — once inline into its own returned HTML, once purely to populate `self._cached_retention_html` so `_section_summary` could re-splice the same table into Section 9 later. Both calls are pure functions of `sim`, so the two copies were byte-identical, and the report silently showed the same table twice. If a value needs to reach a second section, thread the already-computed *result* through (return value, parameter, or a plain instance attribute set once) rather than re-invoking the render function a second time purely as a caching side-effect. |
| `QGroupBox` title containing a bare `&` silently swallows it (mnemonic) | `QGroupBox("3. Region & Run")` rendered as "3. Region Run" — Qt interprets `&` in a group box title as a mnemonic prefix for the next character, same as `QAction`/`QPushButton` text. Escape a literal ampersand as `&&` (`QGroupBox("3. Region && Run")`). Plain `QLabel` text does **not** have this problem (no mnemonic support unless `setBuddy()` is used), so this is specific to titles/text that Qt treats as mnemonic-aware (menus, actions, buttons, group boxes). |
| `QToolBar.addWidget(spacer)` with an `Expanding` size policy can make every action after it vanish | Adding a bare `QWidget` spacer (`QSizePolicy.Policy.Expanding` horizontal, tried both `Preferred` and `Expanding` vertical) to right-align a trailing toolbar action made that action disappear from the rendered toolbar entirely — not shifted, not overflowed into a `»` chevron, just absent — confirmed by re-rendering with the spacer removed (the action reappeared immediately, left-aligned after the preceding separator). Root cause not fully isolated; treat any `addWidget(spacer)`-for-right-alignment idiom in a `QToolBar` in this codebase as unverified until the rendered result is actually checked (see the next pitfall for how to check it without a real display). |
| OS-level screenshots of the PyQt6 GUI are unreliable for verifying a layout change | `Get-WindowRect`/`SetWindowPos`/`Screen.Bounds` from a non-DPI-aware PowerShell process and the actual rendered window disagreed with each other by inconsistent ratios (not a single uniform scale factor) on a scaled Windows display, making a window that should fit on-screen appear clipped, and vice versa — even maximizing the window didn't reliably show all of it. Prefer `QWidget.grab()` (or `QMainWindow.grab()`) from a small headless script that constructs the widget/window directly and saves the returned `QPixmap` to a PNG — this renders through Qt's own coordinate system with a consistent `devicePixelRatio`, sidestepping OS/DPI virtualization entirely, and doesn't require a visible window at all. For precise sizing decisions (e.g. tuning a `setMaximumHeight`), read `QWidget.sizeHint()`/`minimumSizeHint()` directly instead of eyeballing a rendered image — see the "Adding GUI parameter rows..." pitfall above for the exact approach. |
| Overview figure's box/marker count silently drifted from its own caption | `EdgeAnalyzer.analyze()`'s gradient-magnitude overview map was given `rois_used` — every *searched* candidate ROI (`N_CANDIDATE_EDGES = EDGE_N_TOP_EDGES * 3`, e.g. 9) — instead of only the edges actually *accepted* into `edges` (capped at `EDGE_N_TOP_EDGES`, e.g. 3), so the map drew 9 cyan boxes while its own caption said "three selected." Root cause: `rois_used` was assigned once, early, from the full candidate list, and never reassigned after low-quality candidates got filtered out of `edges`. Whenever a display figure loops over a list to draw one marker/box per entry, verify that list is the actually-used subset the caption describes, not the broader search/candidate pool that produced it — fixed by reassigning `rois_used = [e["roi_used"] for e in edges]` right after `edges` is finalized, before it's read by `_plot_gradient_map`/stored in the result dict. |
| A "plausible-looking" derived metric can still be measuring the wrong thing entirely | `EdgeAnalyzer._extract_esf`'s `rotation_angle = -(90.0 - angle_deg)` (present since the file's first commit) looked like a reasonable 90°-complement but actually rotated the edge **horizontally**, not vertically as its own design requires (`esf_raw = nanmean(rotated, axis=0)` averages *down columns*, so the edge must run vertically for that average to stay on one side of the transition). The bug was invisible to `tests/test_analysis/test_edge_analyzer.py` because every test there checked `_esf_quality` (a monotonicity ratio) or structural shape (no NaN, normalized range) — never the actual measured width against a *known* ground truth — and the wrong-orientation artifact (a disc-boundary/interpolation trend) happened to also be smooth and monotonic, so it passed every existing gate while over-measuring width by 7–14x on a synthetic edge with a known Gaussian blur sigma. Diagnosed by rendering `rotate()`'s output for several known angles and inspecting it directly (ASCII-art / value dump, not just numbers) — the same "don't hand-derive `rotate()`'s convention" principle documented above ("Locating a point across `scipy.ndimage.rotate()`..."), applied to the rotation *angle formula* itself rather than just a post-hoc point lookup. When a metric's test suite only checks shape/monotonicity/range properties, add at least one test with an analytically-known true value (`tests/test_analysis/test_edge_analyzer.py::TestEdgeWidthAccuracy`, using the erf-profile width of a Gaussian-blurred step edge) — shape-only checks can pass on a metric that's confidently, monotonically, consistently wrong. |
| Locking `savefig()` alone doesn't fix the mathtext `ParseException` race | `core/fig_utils.py`'s `_MPL_DRAW_LOCK` (formerly `_SAVEFIG_LOCK`) was originally applied only inside `fig_to_b64()`'s `savefig()` call, on the theory that `savefig()` was the only draw-triggering call site. It wasn't: `fig.tight_layout()` also runs a full draw pass to measure text extents (titles, tick labels, legends), which hits the same non-thread-safe pyparsing packrat cache mathtext uses. `PowerSpectrumAnalyzer.analyze()` calls `fig.tight_layout()` unprotected inside its `_plot_results()`, and `gui/analysis_thread.py` runs image A/B's `PowerSpectrumAnalyzer().analyze()` concurrently in a 2-worker `ThreadPoolExecutor` (and, in parallel mode, alongside every other analyzer's figure building too) — so an unlocked `tight_layout()` in one thread reliably corrupted the cache mid-parse in another, surfacing as `⚠ Analysis failed: ... ParseException: exception raised in parse action (at char 0), (line:1, col:1)` on Section 7. Reproduced directly: 60 concurrent `tight_layout()` + locked-`savefig()` calls threw the exact same `ParseException` in a stress test; wrapping `tight_layout()` in the same lock (`core/fig_utils.py::finalize_layout()`) brought the error count to zero over the same stress test. Fixed at every call site that can run concurrently with other figure-building code: `power_spectrum.py`, `snr_analyzer.py`, `psf_analyzer.py`, `image_filters.py`, `halo_analyzer.py`, `edge_analyzer.py`, and `gui/halo_dialog.py` (its `_AnalyzeThread` isn't gated against a concurrent `Run Analysis` pass, so it's a real concurrent path too). `report/report_builder.py`'s `tight_layout()` calls were deliberately left as plain `fig.tight_layout()` — report generation runs strictly serially after every analyzer thread has already joined (`gui/analysis_thread.py`'s comment: "Report generation (always serial — needs all results)"), so there's nothing for those calls to race against; wrapping them would be lock overhead with no behavioral benefit. When adding a new figure-building method anywhere that *can* run inside a `ThreadPoolExecutor` alongside other figure code, call `finalize_layout(fig, **kwargs)` instead of `fig.tight_layout(**kwargs)` — never assume only `savefig()` needs the lock. |
| Dropping a new file into `resources/` is enough — no `.spec` edit needed | `AstroImageLab.spec` bundles the entire directory in one line (`datas += [("resources", "resources")]`), not a per-file list. Any new asset placed in `resources/` (e.g. `AstroImageLabSplash.png`) is automatically included in the Windows/macOS/Linux PyInstaller builds without touching the spec file — confirmed when the splash screen was switched from a procedurally-painted `QPixmap` to loading `resources/AstroImageLabSplash.png` directly via `QPixmap(path).scaledToWidth(...)`. |

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
