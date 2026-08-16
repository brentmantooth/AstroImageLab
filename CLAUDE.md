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
  power_spectrum.py    Signal-normalised 2D FFT power spectrum; per-image relative
                       "spectral MTF50" proxy from the radial curve — NOT a true MTF,
                       see the MTF-source-labeling convention below
  image_filters.py     Local σ maps, Laplacian of Gaussian, wavelet decomposition
  star_catalog.py      DAOStarFinder star detection and isolation filtering
  inspector_regions.py Headless array maths for the Data Inspector — display ranges,
                       A/B comparison, ROI/threshold masks, correlation sampling
  background_fit.py    Section 3f — BIC-selected constant/plane/quadric WLS fit to a
                       Background2D mesh; gradient/isotropic/anisotropic decomposition
  background_mask.py   Section 3e — reproduces Background2D's own per-cell sigma-clip
                       as a pixel-level accept/reject mask (photutils exposes only the
                       resulting per-cell scalar values, never a pixel mask itself);
                       takes the same `mask=` the real call was given
  source_mask.py       4-stage source-masked background re-estimate: one-sided sky
                       scaffold → multi-scale detection → extent-based point/extended
                       classification → order-capped parametric fit. Array-in/array-out
                       (no AstroImage), and must never import core.astro_image — see the
                       cycle note in its module docstring. **Adopted** by
                       `estimate_background()` as `AstroImage.background_model` (Section
                       3e); Section 3g is its audit trail. Diagnostic-only when the
                       `use_source_masked_background` setting is off
core/
  astro_image.py       FITS/XISF loading, background estimation (photutils)
  models.py            40+ constants + AnalysisResult dataclass
  fig_utils.py         fig_to_b64() — embeds matplotlib figure as base64 PNG
  stretch.py           STF stretch + normalize_for_display() for 8-bit display output
  stats_utils.py       mannwhitney_effect() / combined_se_z_test() — significance/
                       comparison primitives shared by analysis/ and report/
  spatial_stats.py     Spatial block bootstrap for autocorrelated per-pixel maps —
                       honest CIs where the per-pixel p-values are saturated. Reports
                       non-convergence rather than hiding it
  practical_significance.py  The four practical labels + the FWHM/SNR currency they
                       anchor to. Returns label=None for an uncalibrated metric rather
                       than guessing
  inspector_catalog.py _inspector.npz catalog reader + panel display names/descriptions,
                       shared by report_builder (writer) and both inspectors (readers)
gui/
  analysis_thread.py   QThread orchestrator; dark-mode rcParams save/restore lives here
  control_panel.py     Settings UI; settings() returns dict consumed by the thread
  image_panel.py       Image display panel; load_path() / set_starless_path() for programmatic load
  report_inspector.py  Interactive side-by-side figure viewer (matplotlib; being replaced)
  data_inspector.py    Data Inspector (QMainWindow) — pyqtgraph successor to the above
  inspector_widgets.py pyqtgraph widgets for the Data Inspector (drawing only, no maths)
  bg_region_dialog.py  Background exclusion regions (QMainWindow, pyqtgraph) — freehand
                       lasso over the image, region list, and a threaded preview of the
                       Section 3g mask. Opened from the toolbar before Run; the dialog
                       owns the drawing mode, so the control panel has a status label
                       but no second checkable button to keep in sync
  synthetic_dialog.py  Synthetic Data Generator dialog (QMainWindow)
  halo_dialog.py       Halo Analyzer interactive tool (QDialog); click-a-star PSF/RDF inspector
report/
  report_builder.py    HTML report generator; consumes AnalysisResult objects
synthetic/
  cameras.py           Camera database — 24 models (ZWO, QHY, Player One)
  generator.py         Image generation engine: Moffat PSF, aberrations, nebula, starless export
tools/
  generate_icon.py         One-off dev script: regenerates resources/icon.ico
  generate_screenshots.py  One-off dev script: regenerates resources/*.png screenshots used by README.md/QuickStart.md
  sensitivity_sweep.py     Metric sensitivity / calibration harness. Subcommands:
                           validate (label the six reports in AstroLabTestData/
                           FilterCompare), null-ensemble (repeated disjoint-partition
                           null floors + mask sweep), signal-grid, blur-grid,
                           recalibrate, add-null-floors, health, m31-factorial.
                           Runs headless with make_figures=False
  metric_atlas.py          Renders the Metric Atlas: 1:1 filmstrip, response curves,
                           and the monotone/sign scorecard deciding which metrics may
                           carry a practical label
  make_starless_stacks.py  Builds outlier-rejected stacks + their SyQon Axiom starless
                           counterparts, the image class Section 8 actually analyses
```

---

## Key Utilities — Reuse These

| Utility | Location | Purpose |
| --- | --- | --- |
| `_info_box(body, title, open=False, style="")` | `report_builder.py:197` | Collapsible `<details>/<summary>` HTML panel — `body` is raw HTML, so it also wraps whole figure-heavy blocks (not just prose) closed by default to keep the report compact; see "Collapsible figure blocks" below |
| `_val(v, fmt, fallback="—")` | `report_builder.py:184` | Null-safe table cell formatter |
| `fig_to_b64(fig)` | `core/fig_utils.py` | Embeds matplotlib figure as base64 PNG string |
| `finalize_layout(fig, **kwargs)` | `core/fig_utils.py` | Runs `fig.tight_layout(**kwargs)` under the same process-wide lock as `fig_to_b64()`'s `savefig()`. Call this instead of `fig.tight_layout()` directly in any figure-building code that can execute concurrently with other figure-building code — see the mathtext race pitfall below |
| `locked_draw_call(fn, *args, **kwargs)` | `core/fig_utils.py` | Runs a matplotlib call that can trigger draw-adjacent text/layout measurement (`fig.colorbar(...)`, `ax.legend(...)` — especially `loc="best"` auto-placement) under the same `_MPL_DRAW_LOCK` as `finalize_layout()`/`fig_to_b64()`. Call this instead of `fig.colorbar(...)`/`ax.legend(...)` directly in any figure-building code that can execute concurrently with other figure-building code — see the mathtext race pitfall below |
| `panel_display_name(pkey)` / `panel_concept(pkey)` | `core/inspector_catalog.py` | Human-readable name and description for a `SpatialDetailAnalyzer` panels key. **Add a branch here whenever a new metric family is added**, or the family shows in both inspectors as a raw key (`localgrad_1.5`) with a blank description box — exactly the gap 8i/8j sat in. `report_builder.py` re-exports them as `_panel_display_name`/`_panel_concept` |
| `load_inspector_data(npz)` | `core/inspector_catalog.py` | Parses `catalog_json` into `InspectorData`/`Entry`. Filters options against arrays actually present, supports the legacy `{key_a, key_b}` format, and back-fills a missing description from the npz key so older `.npz` files gain new prose without regeneration |
| `value_range(arrays)` | `analysis/inspector_regions.py` | Shared percentile display range, character-for-character `_plot_side_by_side`'s formula **including the unconditional `max(0.0, …)` floor**. Never "improve" it for signed data — matching the report is the point |
| `comparison_map(a, b, mode)` / `comparison_range(diff)` | `analysis/inspector_regions.py` | A-vs-B map (`COMPARE_LOGRATIO` / `COMPARE_DIFFERENCE`) and its symmetric colour range, both delegating to `SpatialDetailAnalyzer`'s own helpers. Raises on an unknown mode rather than silently falling through to log-ratio |
| `roi_mask` / `threshold_mask` / `refine_mask` / `correlation_sample` / `ids_to_mask` / `select_points_in_polygon` | `analysis/inspector_regions.py` | The Data Inspector's region pipeline. `threshold_mask` returns a `ThresholdResult(mask, warning)`; `correlation_sample` returns flat pixel ids as the stable identity linking scatter points to pixels |
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
| `SpatialDetailAnalyzer._combined_localmax_mask(abs_a, abs_b, footprint_px, prominence_percentile, region_px, presmooth_sigma, top_percent)` / `_top_percent_mask(abs_a, abs_b, top_percent)` | `analysis/image_filters.py` | `_local_maxima_mask` unioned (OR) with a top-`top_percent`% brightness mask — catches broad bright plateaus a sharp-peak detector alone would miss. Used identically by `_localmax_entry` (Section 8l table stats) and the mask-grid figure builder in `analyze()`, so the displayed mask always matches the mask backing that row's numbers |
| `mannwhitney_effect(va, vb)` | `core/stats_utils.py` | Mann-Whitney U p-value + Cliff's delta in O(n log n) via `delta = 2·U/(n1·n2) − 1`; shared by `_psf_stat_test` (Section 4) and `SpatialDetailAnalyzer._localmax_stats` (Section 8l) — never reimplement Cliff's delta via a pairwise `arr_a[:, None] - arr_b[None, :]` matrix, it's O(n·m) memory |
| `_format_significance_html(p, delta)` / `_sig_td(html, p)` | `report_builder.py` | Shared star-rating/p-value HTML cell + colored `<td>` wrapper for any Mann-Whitney significance column — used by both Section 4's PSF table and Section 8l's local-maxima table |
| `SpatialDetailAnalyzer._ratio_series_with_errors(ratios_by_method, errors_by_method=None)` | `analysis/image_filters.py` | `{method: {scale: value}}` (+ optional matching errors) → sorted `{method: [(x_px, value, error_or_None), ...]}` point lists for a cross-method overview line plot; shared by `_plot_nc_ratio_overview` (8k) and `_plot_localmax_ratio_overview` (8l) |
| `_nc_ratio_rows(score_a, score_b, ratio, scale_label, val_fmt=".3f", method_label=None)` | `report_builder.py` | `<tr>` rows for a noise-corrected score table (Scale \| A \| B \| Ratio A/B). Pass `method_label` to prepend a Method-name `<td>` when consolidating several per-family tables that share this schema into one combined table — precedent: Section 8l's cross-method NC table, which concatenates the row-strings from all seven `_nc_ratio_rows` calls (LoG/Wavelet/Gradient/Std/Entropy/Local grad. energy/Local Laplacian var.) into a single `<table>` |
| `_better_worse_class(val_a, val_b, higher_is_better=True, *, neutral=False)` | `report_builder.py` | Returns a `("better", "worse")` (or reversed) CSS class-name pair for a comparison table's `<td class="...">` — direction is per-metric via `higher_is_better`, not global. Used across Section 4's PSF table, Section 6's edge table, Section 7's power-spectrum table, and Section 9's summary table; reuse this rather than hand-rolling a new green/red comparison whenever a table gains an A-vs-B metric column. `neutral=True` (keyword-only, so all 28 existing call sites are unaffected) renders `.neutral` instead — the dead-band. **The caller decides, deliberately: this takes no tolerance argument** so that the grey cell and the practical label are driven by one measured threshold and cannot disagree (Section 9 passes `neutral=(verdict_for_fwhm(a, b).label == NONE)`) |
| `ReportBuilder._section_verdict(ra, rb, scale_known=True)` | `report_builder.py` | The unnumbered Executive Summary. Three axes — sharpness (`fwhm_px`), noise (`snr_global`, starless-preferred), detail — with a headline that is `overall_label` across the *labelled* axes only. Reads `fwhm_px` never `fwhm_arcsec` (see the bad-header convention). The detail axis deliberately reports magnitude and no label |
| `AstroImage.set_background_exclusion_mask(mask)` | `core/astro_image.py` | Attach the user's rasterised exclusion mask **and invalidate any background already computed**. The idempotency guard keys on `background_model`, so assigning the slot directly after estimation is silently ignored rather than an error. Rasterise in the caller (`exclusion_mask`) — importing it here would close a cycle |
| `classify_background_pixels(data, box_size, sigma=…, maxiters=…, mask=None)` | `analysis/background_mask.py` | `mask` is withheld from the per-cell clip via `np.ma.masked_array`, not just relabelled afterwards — leaving a bright region in lets it set the cell's own centre and scale, so the result would no longer describe the clip that actually ran. Masked pixels return `False`; Section 3e derives its extra classes by subtracting the masks it already holds |
| `plane_quadric_agreement(mesh, rms_mesh, box_size, image_shape, valid, sky_sigma)` | `analysis/source_mask.py` | Fits the same cells at order 1 and 2 and returns whole-frame vs at-the-cells divergence plus their ratio. **Both bounds are needed** — see the convention above for why divergence alone demotes a correct quadric on a vignetted frame |
| `AstroImage.background_display(kind="model"\|"rms"\|"unmasked_model", max_dim=BACKGROUND_DISPLAY_MAX_DIM)` | `core/astro_image.py` | Stride-decimated float32 view of `background_model` / `background_rms` / `background.background`, for Section 3e figures and Data Inspector npz entries. `"unmasked_model"` exists so Section 3g's difference map has something to subtract — against `"model"` it would be identically zero once the masked estimate is adopted. Stride decimation (`arr[::step, ::step]`), not `_inspector_display`'s LANCZOS resize — resampling would blur the actual computed background/RMS values, which is the opposite of what a diagnostic view needs |
| `decimation_step(h, w, max_dim=BACKGROUND_DISPLAY_MAX_DIM)` | `core/astro_image.py` | The one shared striding formula behind `background_display()` and every Section 3e/3f/3g array built independently at a different call site (report figure vs. `_write_inspector_file`) — two arrays that must land on the *identical* pixel grid for an element-wise op (e.g. `model_disp - fitted_surface`) need to agree on `step` exactly, so this is factored out even though the identical one-line formula is tolerated duplicated elsewhere (`edge_analyzer.py`, `snr_analyzer.py`) where no cross-array alignment requirement exists |
| `fit_background_surface(mesh, rms_mesh, box_size, image_shape)` / `evaluate_surface(fit_result, image_shape, step)` / `compare_fits(fit_a, fit_b)` | `analysis/background_fit.py` | Section 3f's gradient/curvature fit: weighted least squares (inverse-variance from `rms_mesh`) on 3 nested polynomial orders, minimum-BIC selection, decomposed into `gradient`/`isotropic_curvature`/`anisotropic_curvature` dicts (`magnitude`/`se`/`p_value`/derived ADU quantity, or `None` if the selected order doesn't include that term). `evaluate_surface`'s sampled-coordinate formula must exactly match `arr[::step, ::step]`'s shape — that's the load-bearing invariant keeping the residual (`background_display("model") - evaluate_surface(...)`) a valid element-wise subtraction |
| `classify_background_pixels(data, box_size, sigma=BACKGROUND_SIGCLIP_SIGMA, maxiters=BACKGROUND_SIGCLIP_MAXITERS)` | `analysis/background_mask.py` | Reproduces `Background2D`'s own per-cell `SigmaClip` as a full-resolution boolean mask (`True` = kept as background) — see the "Reconstructing a mask" pitfall below for why this can't just be read off `Background2D` |
| `source_masked_background(data, box_size=64, fwhm_px=4.0, step=1, n_passes=SOURCEMASK_N_PASSES, user_exclusion=None, mesh=None, rms_mesh=None)` | `analysis/source_mask.py` | Section 3g's single entry point → `MaskedBackgroundResult`. Pass the caller's already-computed `mesh`/`rms_mesh` (every `AstroImage` has them after `estimate_background()`) to skip a redundant full `Background2D` pass. `user_exclusion` is a full-res bool mask applied **twice** — to the stage-1 cell selection and to the final mask — because those do different jobs (better pedestal vs. guaranteed exclusion). `.ok` is False when the mask left too little sky; `.fallback_reason` says why. Carries `rms_map` (the masked pass's full-res RMS, so an adopting caller does not pair a masked background with an unmasked noise floor) and `agreement` (`plane_quadric_agreement`'s record). Measured ~18 s on a 24 MP frame |
| `fit_sky_scaffold(mesh, rms_mesh, box_size, image_shape, reject_nsigma=…, max_iters=…, reject_order=1, final_max_order=2, excluded_cells=None)` | `analysis/source_mask.py` | Stage 1: doubly one-sided robust fit to an *unmasked* mesh (rejects only cells above the fit; scale from the lowest quartile), **two-stage** — rejection at `reject_order`, final refit with BIC free up to `final_max_order`. Returns a `fit_background_surface()` dict plus `kept_cells`/`n_iters`. Reach for this, not a `SigmaClip`, whenever the contamination being rejected is single-signed |
| `exclusion_mask(shape, regions)` | `analysis/inspector_regions.py` | OR of several normalised `roi_mask`-shaped dicts → True where **excluded**. Empty/None gives all-**False**, the exact inverse of `roi_mask(shape, None)` — see the pitfall row |
| `LassoViewBox` (was `_BrushViewBox`) | `gui/inspector_widgets.py` | Freehand/rect polygon capture on any pyqtgraph ViewBox. Coordinate-system agnostic (emits whatever `mapToView` returns), which is why it is public: the correlation scatter and the image panels both use it |
| `LinkedImageView.set_lasso_locked(locked)` / `polygon_drawn` / `set_overlay("exclusion", …)` | `gui/inspector_widgets.py` | Freehand drawing on an image panel, emitting a normalised `{"kind": "polygon", "points": …}` dict straight into `exclusion_mask`. Third overlay slot sits above mask/selection so user input stays visible where it overlaps |
| `vignetted_frame()` / `vignetted_nebula_frame()` / `bounded_nebula_frame()` | `tests/bg_frames.py` | Regression guards for the failures the Gaussian frames could not expose — see the fixture-profile convention below |
| `build_source_mask(data, pedestal_surface, sky_sigma, fwhm_px, step=1)` | `analysis/source_mask.py` | Stages 2+3 → `SourceMaskResult` (`mask`/`point_mask`/`extended_mask` plus the `tiers`/`segments` audit records the report renders). `step` decimates the extended tiers only |
| `masked_background_estimate(data, source_mask, box_size, exclude_percentile=…, min_cell_unmasked_frac=…)` | `analysis/source_mask.py` | Stage 4 → `(payload, fallback_reason)`. Selects cells by *its own* unmasked-fraction rule, not by photutils' exclusion — see the interpolated-fill pitfall below |
| `_cell_unmasked_fraction(source_mask, mesh_shape, box_size)` | `analysis/source_mask.py` | Per-mesh-cell surviving-pixel fraction via `np.add.reduceat`. The only reliable way to tell a *measured* cell from one photutils excluded and back-filled; also drives Section 3g's audit figure |
| `_smoothed_sigma(sky_sigma, kernel_sigma)` | `analysis/source_mask.py` | `σ/sqrt(4π σ_k²)` — noise after a sum-normalised Gaussian convolution. Any threshold applied to a smoothed image must use this, not the raw per-pixel σ (at σ_k = 8 px they differ by ~28×) |
| `make_bg_frame(...)` / `star_only_frame()` / `moderate_nebula_frame()` / `heavy_nebula_frame()` / `gradient_nebula_frame()` | `tests/bg_frames.py` | Ground-truth frames returning `(data, true_sky, truth)` so background tests assert measured accuracy rather than shape properties. Import these rather than hand-rolling a frame; a scratch script can reproduce the exact frame a failing assertion refers to |
| `_sourcemask_for(cache, img, fwhm_px)` / `_adopted_sourcemask(img)` | `report_builder.py` | `_sourcemask_for` returns `img.source_mask_result` when present — reading back what `estimate_background()` actually used is what guarantees 3g audits the numbers 3e presents — and only falls back to computing (memoised per build; `generate()` resets the cache, same rule `_export_ctx` follows) for an `AstroImage` built outside the GUI. `_adopted_sourcemask` is the stricter question 3e/3f must ask: is this the background *in use*, i.e. attached by `estimate_background()` **and** `.ok` **and** the flag on. Don't use the first where you mean the second — it will happily hand back a diagnostic estimate for an image whose background was left unmasked |
| `combined_se_z_test(val_a, se_a, val_b, se_b)` | `core/stats_utils.py` | Generic two-sided z-test for comparing two *independent* point estimates with known SEs (`z = (val_a-val_b)/sqrt(se_a²+se_b²)`, normal not Student-t — no shared `df` between two separately-fit models). Sibling of `mannwhitney_effect`: same "generic primitive in `core/`, domain-specific call site in `analysis/`" layering. Used by `analysis/background_fit.py::compare_fits` for Section 3f's A-vs-B column; reach for this (not a paired test) whenever the two things being compared are separate fits/estimates rather than a pixel-paired population |
| `_bgfit_value_td(value, se, p_term, css_class, fmt=".3g")` / `_bgfit_table_html(fit_a, fit_b, label_a, label_b)` | `report_builder.py` | Section 3f's table-cell/table builder — composes `_sig_td` (per-term significance: is this specific magnitude distinguishable from zero) around a `_better_worse_class`-colored `<span>` (A-vs-B judgement). Deliberately **not** built on `_format_significance_html`/`_psf_stat_test` — those are shaped around Mann-Whitney U + Cliff's delta for two *distributions* of values, a different statistical object from a parametric regression coefficient's t-test against zero |
| `find_level_crossing(x, y, level=0.5)` | `core/stats_utils.py` | Linear-interpolated first *descending* crossing of `level` in a `(x, y)` curve; `None` if `y` never crosses (starts below `level`, or never drops below it). Extracted from `PSFAnalyzer`'s original ePSF-MTF50 search so `PowerSpectrumAnalyzer._spectral_mtf` (`analysis/power_spectrum.py`) can reuse the identical algorithm for its own, differently-sourced MTF50 proxy instead of re-deriving it — see the MTF-source-labeling convention below |
| `_db_ratio_at(freq_a, rp_a, freq_b, rp_b, ref_freq)` | `report_builder.py` | Interpolates `_power_ratio_db`'s dB curve at a single reference frequency; `None` on missing input, misaligned bins, or `ref_freq=None`. Used to collapse a whole ratio *curve* into one table-cell scalar wherever a report row needs "the dB ratio at the point this row's own headline number sits," rather than requiring the reader to eyeball it off an adjacent figure |
| `block_bootstrap_ci(field, mask, block_px, n_boot, ci, seed, fn)` | `core/spatial_stats.py` | Spatial block bootstrap of a masked mean → `BootstrapResult` (point, lo/hi, se, `block_px`, `n_blocks`, `converged`, `naive_se_understatement`). Feed it `panels[key]["diff"]` — the per-pixel log-ratio map Section 8 already builds — so a CI costs no metric recomputation. **Always check `.converged`**: on this project's real maps it is often False, and the interval is then a lower bound on the true uncertainty, not the uncertainty |
| `se_ladder(field, mask, ...)` / `auto_block_size(...)` | `core/spatial_stats.py` | Bootstrap SE at a geometric ladder of block sizes, and the block choice derived from it. The block size **must** come from where the SE plateaus, never from the correlation length — see the long-range-dependence convention below |
| `tile_reduce(field, mask, block_px, fn)` | `core/spatial_stats.py` | Field → one value per tile, plus per-tile valid-pixel weights. The weighted mean of the tile values reproduces the plain masked mean exactly, which is what keeps the bootstrap's point estimate equal to the number the report's table shows. Tile acceptance scales with the mask's own density (see the sparse-mask pitfall) |
| `verdict_for_fwhm(a, b)` / `verdict_for_snr(a, b)` / `verdict_for_metric(key, log_ratio, cal)` | `core/practical_significance.py` | A measured change → `PracticalVerdict` (label, currency, equivalent, calibrated). FWHM and SNR need no calibration — they *are* the currency. A Section 8 metric returns `label=None` with `calibrated=False` unless `resources/metric_calibration.json` covers it; never guess a label |
| `consensus_label(verdicts)` vs `overall_label(verdicts)` | `core/practical_significance.py` | **Median** within one family of correlated estimates, **max** across independent axes. Using the wrong one is a real failure mode, not a style choice — see the aggregation convention below |
| `starless(arr, tag)` / `_maybe_starless(...)` | `tools/sensitivity_sweep.py` | SyQon Axiom v2.1 star removal on an in-memory array via the CLI at `D:\Astro\SyQon\starless_cli\starless_cli.exe`. `--use-mtf` is mandatory (stacks are linear, the model expects display-stretched input). Mandatory before any spatial-detail calibration — see the star-contamination convention |
| `_blur_preserving_noise(arr, sigma, rng)` | `tools/sensitivity_sweep.py` | Gaussian blur that adds back the noise the blur removed, so detail loss is not confounded with noise reduction. Survival fraction is measured by blurring actual white noise, never from the asymptotic `σ/sqrt(4π σ_k²)` closed form |
| `_reject_outlier_frames(cache, paths, hi, lo)` | `tools/sensitivity_sweep.py` | Per-frame fine-scale-structure screen. **Run before any null floor is measured** — one bad sub inflated a floor 26× |

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

**Use literal `e` format (`.2e`/`.3e`), not `.Ng`, when scientific notation is
specifically required.** `.Ng` only switches to exponential form when the value's
exponent falls outside roughly `[-4, N)` — `0.0034` under `.3g` prints as plain
`0.0034`, not scientific notation. For a column that must always render in
`d.dde±dd` form regardless of magnitude (e.g. Section 6's Gradient magnitude,
typically sub-0.01 ADU-scale Sobel gradients), use `.2e` directly rather than
reaching for this codebase's usual `.Ng` significant-figure convention — the two
solve different problems (never-collapses-to-zero vs. always-scientific).

### Long f-string HTML blocks

Pre-compute any Python variable **before** a `return f"""..."""` block. Do not nest
`{f"...{var}..."}` substitutions — they cause confusing `UnboundLocalError` and
syntax errors at runtime.

### Collapsible figure blocks — reuse `_info_box` for images, not just text

`_info_box(body, title, open=False, style="")` (`report_builder.py:197`) only ever
wrapped prose/methodology text until Section 8's families got collapsed — its `body`
parameter is raw HTML, so it works unchanged for a block of embedded `<img>` tags.
To collapse a figure-heavy block (precedent: Section 8's seven metric families 8d–8j,
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

**Do not add float64 casts in analysis code.** There are two legitimate exceptions in the
codebase, both explicit and commented at the cast site — do not add a third without the
same justification (tiny array, precision-sensitive result):

1. The `astroalign` registration call in `gui/analysis_thread.py`, which requires float64
   internally.
2. `analysis/background_fit.py::fit_background_surface`'s weighted-least-squares solve
   (`np.linalg.lstsq`/`np.linalg.pinv` on the mesh's design matrix). The array is tiny
   (tens–hundreds of mesh cells, never image-scale) so there's no memory/perf cost, and
   the fit's SE/p-value inference needs the numerical conditioning float64 gives the
   matrix inversion. Only the internal solve is float64 — `mesh`/`rms_mesh` inputs and
   every returned scalar/array (`coefficients`, `evaluate_surface`'s output, etc.) stay
   float32/plain Python float, matching how `snr_global = float(...)` elsewhere in the
   pipeline is a scalar float regardless of the surrounding array dtype.

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
- **Resample onto a common grid instead of requiring exact bin alignment when the
  domain is fixed by a shared constant, even if bin count varies per-image.** The
  "guard bin alignment, return `None`" rule above is correct when two frequency axes
  come from independently-sized data with no guaranteed relationship (Section 7's
  power-spectrum ROI). It's the wrong tool when the axis *domain* is fixed by a
  global constant even though *bin count* varies per-image — e.g. Section 4's MTF
  ratio (`_mtf_ratio_db`/`_plot_mtf_ratio_db`), where each image's MTF bin count
  depends on that image's own median star FWHM (`_compute_mtf`'s `nbins = n // 2`,
  with `n` derived from `box_size`/`fwhm_estimate`) but the frequency domain is
  always `[0, 0.5 * EPSF_OVERSAMPLING]` cycles/native-px since `EPSF_OVERSAMPLING`
  is a fixed module constant, not per-image. Requiring exact alignment there would
  make the ratio figure almost never render — differing FWHM between the two images
  being compared is the point of the comparison, not an edge case. Resample both
  curves onto a shared `np.interp`-built grid before dividing instead (precedented
  on this exact data: `mtf_nyq = float(np.interp(0.5, freq, mtf))`,
  `psf_analyzer.py:142`).

### MTF/MTF50 values must state their source — a natural scene's power spectrum is not directly an MTF

Two pipelines each produce a number called "MTF50," and they measure genuinely
different things. `PSFAnalyzer` (Section 4) builds an ePSF from the with-stars
image's own stars — a *known* point-source input — FFTs it, and reports the true
system MTF: this is the only one of the two that is an MTF in the optical-engineering
sense. `PowerSpectrumAnalyzer._spectral_mtf` (Section 7, `analysis/power_spectrum.py`)
instead FFTs a real, unknown nebula scene: `P_image(f) ≈ |H(f)|² · P_scene(f)`, where
`P_scene(f)` — the nebula's own true spatial-frequency content — is unknown and
cannot be separated from the system response `H(f)` using a single image alone. No
absolute MTF is recoverable this way, full stop.

What *is* recoverable: normalise each image's own radial power to the mean power in
its low-frequency band (`freq <= LOW_FREQ_MAX`, 0.10 cyc/px — the same boundary
`_compute_mid_high_ratio` already uses), then take the square root (power → an
amplitude-like quantity) so a 50%-crossing (via the shared `find_level_crossing`,
see the utilities table above) carries the same *meaning* as a true MTF50. This is
valid only as a same-target, same-session A-vs-B comparator. It is not comparable in
isolation to `PSFAnalyzer`'s ePSF-based MTF50 (different physical quantity, not a
second measurement of the same one), is unsmoothed (a single noisy realisation's
radial-averaged FFT power, unlike the ePSF curve's stacked many-star fit — so a `None`
crossing on a noisy or very smooth spectrum is expected, not a bug), and is not
guaranteed to start near 1.0 the way the ePSF MTF is explicitly clamped to be via
`otf /= otf.max()` in `_compute_mtf` — it is normalised to the low band's *mean*, not
a peak.

**Every MTF/MTF50 value the report shows must say which of the two it is.** Table row
labels use the pattern `"MTF50 — <source> (cyc/px)"` (`"— ePSF"` / `"— power
spectrum"` / `"— power spectrum, with stars"`), never a bare `"MTF50 (cyc/px)"` —
Section 4, Section 7, and Section 9's summary table all follow this, and Section 4's
"How the MTF is derived" info box explicitly cross-references Section 7's number as a
different, complementary quantity so a reader who notices the two disagree doesn't
conclude one is broken. Apply the same reasoning before adding a third source of an
MTF-shaped number: if it comes from a known calibration input (point/edge/sinusoid),
it's a real MTF; if it comes from an arbitrary scene, it's at best a same-target
relative proxy and must be labelled and caveated as such. The dB "ratio of the
difference" column for the power-spectrum MTF50 table reuses the existing
power-quantity dB convention above (`10·log10`, via `_db_ratio_at`), sampled at the
mean of the two images' MTF50 frequencies — not a fresh ad hoc ratio formula.

### `AstroImage`'s three background attributes mean three different things

Since the source-masked estimate was adopted, "the background" is no longer one array:

| Attribute | Is | Used by |
| --- | --- | --- |
| `background` | the **unmasked** `Background2D` object | stage-1's scaffold (needs an unmasked mesh *by design*), `box_size`, Section 3g's "before" column |
| `background_model` | the sky level **actually subtracted** — the masked fit's surface, or `background.background` when masking is off/failed | `background_subtracted()`, `background_display("model")`, `snr_analyzer`'s `background_median` |
| `background_rms` | the noise floor **actually used** — the masked pass's RMS map, or the unmasked one | every analyzer (`star_catalog`, `image_filters`, `snr_analyzer`, `halo_analyzer`, `psf_analyzer`) |

The blast radius stayed small precisely because most consumers read the *attributes*,
not `background.background`. When adding a consumer, read `background_model` — reaching
into `background.background` gets you the unmasked estimate, which nothing else in the
report was computed against. `background_display(kind="unmasked_model")` exists solely
so Section 3g can still show the comparison.

Feeding the source mask into `Background2D(mask=…)` instead of fitting a surface is the
known-bad configuration, for the interpolated-fill reason documented below — at high
coverage most cells go empty and photutils back-fills them.

### A single background sample must be re-expressed in each plot's own normalisation

Section 5 (Halo Analysis) draws a dashed background-noise reference line on every
cross-section and radial-profile plot. `HaloAnalyzer._bg_rms_at()` samples
`image.background_rms` once per star, at that star's own pixel location — but every
plot in this section has already re-normalised its curve differently (cross-section:
raw ADU; RDF: log10-relative to the star's own centre-bin value; Moffat profile:
linear-relative to the star's own peak), so the *same* physical noise sample has to be
converted into three different units before it means anything on a given axis:

| Plot | Existing normalisation (already computed) | Field | Formula |
| --- | --- | --- | --- |
| Cross-section (`ax_xs`) | none — raw ADU | `bg_rms` | `bg_rms` as-is |
| RDF (`ax_rdf`) | `norm = mu[0]` (log10 value at centre bin) | `rdf_bg_fraction` | `bg_rms / 10**norm` |
| Moffat profile (`_plot_profile`) | `peak = I[0]` (linear centre-bin value) | `bg_over_peak` | `bg_rms / peak` |

Each conversion happens once, at the exact point in `HaloAnalyzer.analyze()` where
that normalisation constant is already in scope — never in `report_builder.py`,
which only reads the pre-converted field and calls `axhline`. Guard `bg_over_peak`
on `I[0] > 0`: `peak` silently falls back to `1.0` when the true centre-bin value is
non-positive (`peak = I[0] if I[0] > 0 else 1.0`, `analysis/halo_analyzer.py`), and
dividing by that fallback would produce a number that looks like a fraction but
isn't one. Aggregate figures (the two `_plot_rdf_comparison` calls, the stacked
Moffat profile) take the *median* of the per-star fractions across whichever star
population that aggregate was built from, not a fresh whole-image scalar — same
principle as `consensus_label`'s median-within-a-correlated-family aggregation.

This is a different quantity from the interactive Halo Analyzer dialog's own
background line (`gui/halo_dialog.py`'s `_rdf_bg_level`/`_xs_bg_level`): the dialog
derives a proxy from the mean of the profile's own outer 20% tail, independent of
`AstroImage.background_rms` entirely, and is not wired to `HaloAnalyzer` at all. The
two are not expected to agree exactly — the report's line is the real,
independently-characterised sky noise; the dialog's is a self-referential estimate
from the same curve it's drawn on. This was a deliberate scope decision (report-only)
rather than an oversight — see the dialog's own thread architecture notes below.

### The exclusion mask must be attached before the background is computed

`estimate_background()`'s idempotency guard now keys on `background_model`, and the mask
changes the answer, so a mask attached afterwards would be silently ignored — a wrong
result, not an error. Two things enforce the ordering:

- **`set_background_exclusion_mask(mask)`** clears `background`/`background_model`/
  `background_rms`/`source_mask_result`. Always use it; never assign the slot directly.
- **`AnalysisThread._execute()` attaches before the pre-pass**, not just before the
  report call as it did while Section 3g was diagnostic-only. It sets
  `use_source_masked_background`, `psf_fwhm_hint_px` and the rasterised mask on *every*
  image including the starless companions, so the starless SNR comparison stays
  like-for-like.

Rasterisation happens in the caller because `core/astro_image.py` cannot import
`analysis.inspector_regions` — that reaches `analysis.image_filters`, which imports
`core.astro_image`. `analysis.source_mask` *is* safe to import from `core` (it reaches
only `analysis.background_fit` plus numpy/scipy/photutils), which is what makes the
adoption possible at all.

`psf_fwhm_hint_px` is a hint, not a measurement: PSF analysis needs the background first,
so `AnalysisThread` derives it from `ref_seeing_arcsec / pixel_scale`. A 2× error changes
the point mask's dilation margin, not the qualitative result, and the mask is displayed.

### Adopting a better background changes what a metric *selects*, not just its value

`snr_global` is `median(bgsub[bgsub > 3σ]) / σ`. With the unmasked background the nebula
was being subtracted away as if it were sky, so only star cores cleared 3σ. With the
masked background the nebula clears it too, and the median of a much larger, fainter
population is lower. Measured on `gradient_nebula_frame`:

| | pixels > 3σ | `snr_global` | recovered flux vs truth |
| --- | --- | --- | --- |
| unmasked | 1.47 % | **30.30** | 46 % |
| masked | 11.41 % | **3.70** | **84 %** |

The headline number fell 8× while the thing it is a proxy for nearly doubled. Section 3e
says this explicitly and a test pins the paragraph, because without it a user flipping
the checkbox reasonably concludes the feature is broken. **When a change alters the
population a summary statistic is computed over, "did the number go up?" is the wrong
acceptance test** — check the quantity the statistic stands in for (here, recovered flux
against a known truth) and say so in the report.

### Headless analysis — `make_figures=False`, and it is not a small saving

`SNRAnalyzer`, `PSFAnalyzer`, `PowerSpectrumAnalyzer` and `SpatialDetailAnalyzer` all take
`make_figures: bool = True`. Passing `False` skips every matplotlib call and omits the `figures`
key; every other returned key is byte-identical (verified: 621/621 metric keys on
`SpatialDetailAnalyzer`). The saving is **25×** on that analyzer (53.2 s → 2.1 s), because ~30
figures at dpi=150 dominate its runtime — which is what makes a 400-point sweep practical at all.

Default stays `True` so the GUI and every existing call site are unchanged. Note the early-return
paths already omit `figures`, so callers must treat it as optional regardless.

Two things that must stay outside the flag because they are metrics, not pictures: `xs_snr`
(computed, then only its rendering skipped) and the local-maxima table statistics.

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

### Model order selection: BIC over Lasso for small nested candidate sets

`analysis/background_fit.py`'s Section 3f gradient/curvature fit chooses between 3
nested polynomial models (constant / plane / full quadric — 1/3/6 candidate terms
total) by comparing BIC (`n·ln(WRSS/n) + k·ln(n)`) across weighted-least-squares fits
of each, not by Lasso-regularizing a single larger fit. BIC is the right tool whenever
the candidate models are few, nested, and each tier has a standalone physical meaning
worth naming (here: "no gradient" / "linear gradient" / "gradient + curvature") — it's
exact (closed-form WLS per candidate), needs no regularization-strength hyperparameter,
and its ΔBIC has a standard interpretability scale (Kass & Raftery: >2 positive, >6
strong, >10 very strong evidence for the more complex model). Reach for Lasso instead
only when the candidate feature set is large and/or the features are correlated in a
way that makes an exhaustive per-subset BIC comparison impractical — that is not the
shape of this problem, and would additionally need cross-validation to pick a
regularization strength, which is awkward on spatially-autocorrelated mesh cells (CV's
fold-independence assumption doesn't hold for a smooth spatial field). Apply the same
reasoning to any future "which of a few nested physically-meaningful models fits this
data" decision before defaulting to a more general regularized regression.

### A model-order gate expressed as an absolute count does not survive a change of scale

`SOURCEMASK_MIN_CELLS_QUADRIC = 35` was tuned on the 64-cell mesh of a 512² fixture,
where it means **54.7 % of cells**. On a 24 MP frame's 2925-cell mesh the same number
means **1.2 %** — the guard evaporates on exactly the large real images where a quadric
has the most masked area to extrapolate across. Observed: a full quadric fitted from 492
of 2925 cells.

The gate is now `max(absolute_floor, fraction × n_cells_total)`. Express the risk in the
units the risk actually lives in — here the danger is extrapolation *across the frame*,
so the question is what fraction of the frame was measured, not how many cells there
happen to be. `n_cells_total=0` defaults the fraction term away, so the old rule is the
degenerate case and every existing single-argument call is unchanged.

Deliberately **not** applied to the plane: on a heavily-masked frame a plane still beats
a constant, so gating it would be a regression.

### Two numbers, not one, separate real curvature from extrapolated curvature

`plane_quadric_agreement` fits the same surviving cells at order 1 and order 2 and
reports where they part company — over the whole frame, and at the surviving cell
centres. **Divergence alone cannot make this call.** On a genuinely vignetted frame the
plane simply cannot fit, so the two surfaces differ everywhere and the quadric is
*right*: measured 5.32 σ divergence on `vignetted_frame`, which is correct behaviour.
What identifies invented curvature is the *ratio* — agreement where cells survive,
disagreement only out across the mask. The vignetted frames sit at ratio 1.58/1.69,
comfortably under the 3.0 bound, so demotion never fires on them.

Demotion requires **both** conditions and ships enabled on that evidence: it never fired
on any of the seven ground-truth frames, so it is a guard for the large-mesh case those
512² fixtures cannot reach, not a knob trading accuracy on the tested ones. The numbers
are reported in Section 3g either way — a reader cannot otherwise tell a curved fit
following real curvature from one extrapolating across a mask.

Maximise the difference by **sampling** a grid over the frame, not by a closed form: the
difference of two quadrics attains its maximum on the boundary *or* at an interior
critical point depending on the form's definiteness, and 128² evaluations settle both
cases without a case analysis.

### Detection thresholds — statistical significance is only half the question

A threshold set purely by "is this real?" runs away as soon as the test integrates over
an area. Section 3g's broadest tier smooths with a σ=16 px kernel, which drops the noise
~57×, so a 2σ cut triggered on structure at **3.5% of sky noise** — genuinely detectable,
and completely irrelevant, since masking it shifts the background by nothing while
consuming the mesh cells the fit needs. Coverage hit 62% on a frame whose visible
nebulosity was 12.9%.

Ask the second question too: **is acting on this worth what it costs?** Thresholds are
now `max(statistical, SOURCEMASK_MIN_SURFACE_BRIGHTNESS_SIGMA × sky_sigma)`, and the audit
record carries `threshold_statistical_adu`, `threshold_floor_adu` and `threshold_source`
so the report can say which bound was binding — without that, a reader cannot tell a
correctly-quiet tier from a broken one. Apply the same shape to any future threshold
whose test statistic improves with integration area or exposure.

### A robust fit that also has to choose a model order — reject conservatively, then refit

`fit_sky_scaffold` iterates its one-sided rejection with a **plane**, then refits the
survivors with BIC free to pick a **quadric**. Neither order works alone, and the failure
modes are opposite:

| Scaffold | Fails on | How |
| --- | --- | --- |
| quadric throughout | nebulosity | the curved fit absorbs the nebula, the residual flattens, less is detected, pedestal ends up biased (heavy frame 0.299 → 1.053 σ) |
| plane throughout | vignetting | cannot represent a bowl, so the whole bowl stays in the residual and gets masked as if it were source — estimate came out *worse than doing nothing* |

The general rule: while a robust loop is still deciding *which points to trust*, give it
the model that cannot explain away the contamination. Only once the outliers are gone is
it safe to let the model complexity float. Measured best-or-near-best on all seven
ground-truth frames, where each single order was catastrophic on at least one.

### Some ambiguities are degenerate — no threshold resolves them, so ask the user

Two pairs in this domain are *mathematically* indistinguishable from pixel values alone:

- **smooth nebulosity vs. a sky gradient** — a nebula edge running parallel to the
  gradient is exactly a steeper gradient. A test case had to be deleted over this;
- **a centred nebula vs. vignetting** — both are radially symmetric with the same
  curvature sign, so even the isotropic/anisotropic decomposition cannot separate them.

Before spending effort tuning a discriminator, check whether the two hypotheses produce
the *same* observable. If they do, the only fix is external information — hence the
Background Regions dialog. Be honest in the report about which it was: Section 3g reports
user-drawn pixels as their own mask class rather than folding them into the detector's,
so the algorithm is never credited with the user's judgement.

### Single-signed contamination — reject one side, and estimate scale from the clean side

`SigmaClip` is the reflex for "throw out the outliers", and it is the wrong tool whenever the
contamination can only push one way. Nebulosity strictly *adds* flux, so a symmetric clip
(a) cannot remove a bias carried by more than half the samples, and (b) discards genuine
dark-sky samples that carry exactly the information the fit needs. `fit_sky_scaffold`
(`analysis/source_mask.py`) is the pattern: reject only residuals **above** the running fit.

The subtler half is the *scale* estimate, which was got wrong twice before it was right:

| Estimator | Breaks down at | Why |
| --- | --- | --- |
| two-sided MAD | ~any contamination | inflated by the contamination it must detect — it grew until 62 of 64 cells survived and the scaffold stayed biased **+1.26 σ** |
| below-median MAD | >50% | once the median itself moves into the contaminated population, the "lower half" is polluted too — at 72% contamination it rejected **nothing** |
| **lowest quartile** (current) | ~75% | q05/q25 stay clean while ≤75% is contaminated; `σ = (q25−q05)/0.9704` and the uncontaminated median is recovered as `q25 + 0.6745σ` |

Anchor the cut on that reconstructed clean median, not on `np.median(residuals)` — with heavy
contamination the raw median *is* the bias. Reach for the quartile form for any future
"fit the sky/floor/baseline under one-signed contamination" problem.

### An iterative detect → mask → re-estimate loop diverges; do not assume it converges

The intuition that a second pass refines the first is wrong when each pass feeds the next a
*lower* pedestal: more gets detected, the mask grows, fewer and more spatially-clustered cells
survive, the estimate drops again. Measured on a heavy-nebula frame, mask coverage ran away
**54% → 84% → 95%**, and a gradient frame went from +0.10 σ at one pass to **−2.17 σ** at two
and −2.84 σ at three. `SOURCEMASK_N_PASSES` is therefore **1**.

The parameter is kept (tests exercise the loop) but raising it needs ground-truth measurement,
not reasoning. More generally: any feedback loop where the mask can only ever grow has no
restoring force — verify convergence empirically before shipping a fixed iteration count, and
prefer one well-anchored pass over several unanchored ones.

### photutils back-fills excluded mesh cells — `n_pixels_mesh` cannot tell you which are real

`Background2D` does not drop a cell it excludes via `exclude_percentile`; it **interpolates a
replacement** into `background_mesh` that is indistinguishable from a measurement in every
downstream figure and fit. `n_pixels_mesh` does not disambiguate them either. Fitting those
fills gave a mesh RMS error of **55.5 ADU** where genuinely-measured cells sat at **5.1**, and —
the part that hid the bug — it flattened the recovered gradient to under half its true size
while the *median* bias still looked fine.

The fix is to invert the responsibility: set `exclude_percentile` **permissive** so photutils
reports every cell it can actually measure, then select cells yourself from the mask
(`_cell_unmasked_fraction` ≥ `SOURCEMASK_MIN_CELL_UNMASKED_FRAC`). That also makes the
selection auditable, which a threshold buried in photutils is not. Note the bias here is *not*
truncation — the surviving pixels in a half-masked cell are genuine sky, so such a cell is
still unbiased; the cut exists because few surviving pixels means a noisy cell and, at the
extreme, an interpolated fill rather than a measurement.

### Detecting extended structure — excise point sources before convolving at large scales

A star is compact but *bright*, and convolution trades one for the other. A 30 000 ADU star
smoothed with a σ=16 px kernel still has a ~339 ADU peak against that tier's ~1.06 ADU
threshold, so it registers as a **54 px-radius** "extended" blob. Forty ordinary stars alone
masked **63%** of a star-only frame before this was fixed.

Run the point-source tier first, dilate it, and zero those pixels in the residual the broad
tiers convolve (zero is correct because the residual is already pedestal-subtracted, so zero
*is* the local sky level). Two related rules from the same work:

- **Threshold against the smoothed noise, not the per-pixel noise.** Use `_smoothed_sigma`;
  at σ_k = 8 px the two differ by ~28× and a per-pixel threshold detects essentially nothing.
- **Classify by extent, never by eccentricity.** Eccentricity separates round from elongated,
  not compact from extended — filamentary nebulosity, blended pairs, trailed stars and
  diffraction spikes all sit at its high end together, so it misclassifies in both directions.
  Gate on `equivalent_radius` against the PSF plus the detecting tier's scale; record
  eccentricity for the audit table without gating on it.

### Local-maxima / peak detection — scale-relative parameters, not fixed pixel values

`SpatialDetailAnalyzer._local_maxima_mask` (Section 8l) detects peaks via
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

### Judging a fitted surface — median bias alone hides the failure that matters

A surface fitted to samples clustered in one region can pass *through* the truth near the frame
centre while diverging badly at the corners. An early Section 3g revision looked excellent at
**+0.09 σ median bias** while carrying a **3.9 σ** corner error — the median was averaging a
positive centre against a negative edge. Two bugs (the gradient-flattening interpolated fills,
and an over-permissive order cap) were invisible until RMS and worst-case error were checked.

Always report/assert all three — `median`, `rms`, `max(|err|)` — for any 2D surface or map
compared against a truth, and for a fitted plane also check the recovered *tilt* (least-squares
fit through the output surface) rather than the fit's own linear coefficients: when BIC selects
the quadric, those coefficients are the tangent slope at frame centre, not the frame-averaged
tilt, so comparing them to an injected plane is not like-for-like.
`tests/test_analysis/test_source_mask.py::test_error_is_bounded_across_the_whole_frame` is the
guard; the same reasoning as the edge-analyzer rotation bug, one dimension up.

### Ground-truth fixtures — return the truth, not just the data

`tests/bg_frames.py` returns `(data, true_sky, truth)` so background assertions are quantitative
(`|recovered − true| < tolerance`) instead of structural. This is the direct application of the
edge-analyzer lesson: a metric can be smooth, monotonic, plausible and confidently wrong, and
only a known answer catches it. Frames are exported as named functions (`heavy_nebula_frame()`
etc.) rather than fixtures alone, so a scratch script can reproduce exactly the frame a failing
assertion refers to.

**The fixture's profile decides which bugs it can expose.** `bg_frames.py` originally
offered only Gaussian nebulae, whose wings never reach zero — so *every* frame was
contaminated everywhere, masking more always improved measured accuracy, and an
over-aggressive detector scored well. That single choice hid three real failures at once.
`nebula_profile="bounded"` (compact, reaches exactly zero) and `vignetting=` were added
after the fact, and immediately exposed all of them: the floor's benefit inverted sign,
a plane-only scaffold turned out to mask >50% of a source-free frame, and the quadric
cell threshold turned out to be too low. When a synthetic fixture makes a whole class of
error impossible by construction, it is not testing the thing you think it is — vary the
*shape* of the truth, not just its amplitude.

**Stamp point sources into a bounded patch.** The first version evaluated a full-image `exp()`
per star — fine at 512×512, but O(n_stars × H × W) made a 24 MP / 600-star frame take over ten
minutes and it read as a hang in the code under test rather than in the helper. Use a
`4σ` half-width slice; the profile beyond that is numerically zero anyway.

### Ratio uncertainty / error bars — exact when pixel-paired, approximate (CV-propagated) otherwise

When adding an error bar to a ratio-vs-scale plot (precedent: Section 8k/8l's
cross-method overview figures, `_plot_nc_ratio_overview` / `_plot_localmax_ratio_overview`
in `image_filters.py`), first check whether the two populations behind the ratio are
**pixel-paired** (same pixel coordinates in both images):

- **Pixel-paired** (Section 8l: `diff[mask]` is a genuine per-pixel `log10(|A|/|B|)`
  population) → take `std(diff[mask])` directly and delta-method-propagate it into
  linear ratio units (`ratio * ln(10) * log_std`) — an exact spread measure.
- **Not pixel-paired** (Section 8k: nebula vs. background populations, computed
  independently per image) → there is no per-pixel ratio distribution to take a std
  of. Use a standard relative-uncertainty (coefficient-of-variation) propagation
  instead: `err = |ratio| * sqrt((std_a/median_a)² + (std_b/median_b)²)`. **This is
  an approximation, not a formal confidence interval** — caption it as such in the
  report (see 8k's methodology caption) rather than presenting it as exact.

Both plot functions share `_ratio_series_with_errors` for the point-list-building
loop; only the upstream computation of the error value differs by data source.

**Prefer presenting a log-space quantity in log space throughout, rather than
converting back to linear for display.** Section 8l's ratio column and cross-method
overview originally stored `ratio = 10**mean(diff[mask])` and converted `log_ratio_std`
into a linear error bar via the delta method (`ratio * ln(10) * log_std`) — correct,
but an avoidable approximation-flavoured extra step. Since `diff[mask]` *is* a log10
population, carrying `log_ratio_mean`/`log_ratio_std` straight through to the table
(`_val_pm`) and the overview plot's y-axis removes the conversion entirely — the error
bar becomes exact by construction instead of merely well-approximated, and the table
header should say so (`"log ratio A/B (geo. mean ± SD)"`, not `"Ratio A/B"`). Shade
this kind of column a fixed neutral color (not `_better_worse_class` red/green) when
it isn't a value judgement between A and B, just a measured quantity.

### A report figure that grows with the metric count — one figure per grouping, collapsed

When a single combined figure's row count scales with *how many metric families/scales
exist* rather than a fixed number, it will eventually become "very tall" as new families
are added — precedent: Section 8l's local-maxima distribution figures. Originally two
giant N-row × 1-column figures (`_localmax_distributions_figure` /
`_localmax_log_ratio_distribution_figure`) spanned *every* family's rows combined — 17
rows once 8i/8j (Local Gradient Energy / Local Variance of Laplacian) existed, always
rendered open (not collapsed). Refactored into `_localmax_family_distributions_figure`:
one **N×2 grid per family** (that family's own scales as rows; left column = magnitude
violin, right column = ratio violin), called once per family via the existing per-family
row-filter lists (`_std_rows`, `_log_rows`, etc. — reuse these, don't re-derive the
grouping), with all resulting figures concatenated and wrapped in a **single** collapsed
`_info_box`. Two extraction moves made this possible: (1) hoist the duplicated
`_draw_boxwhisker` helper to module scope once, instead of nested identically inside both
original functions; (2) extract each original function's per-row body into a standalone
`_draw_*_ax(ax, ...)` helper that draws one row's content into a *given* axis and returns
whether it drew anything (`False`/`ax.set_visible(False)` on insufficient data) — the new
per-family wrapper then just loops rows, calling both `_draw_*_ax` helpers into
`axes[i, 0]`/`axes[i, 1]`. Apply this same shape (extract per-row draw helpers, wrap in a
per-grouping outer function, concatenate + single collapse) to any other report figure
whose height is a function of "how many things currently exist" rather than a fixed
count.

### pyqtgraph — keep maths out of `gui/`, and theme per widget

The Data Inspector (`gui/data_inspector.py` + `gui/inspector_widgets.py`) is the only
pyqtgraph code in the project. Two structural rules make it maintainable:

**Every array formula lives in `analysis/inspector_regions.py`, never in the widget.**
`gui/` has no test coverage by design (the headless suite has no PyQt6), so anything
computed there is unverifiable. This is not theoretical: `value_range` was first
written inside `inspector_widgets.py` with an invented conditional that preserved
negative values for signed maps, so the inspector shaded the Original panel from
−9.23 where the report starts at 0. Nothing could catch it until the formula moved
somewhere a test could reach — it was found by rendering the report figure and
comparing numbers by hand. When adding a new interactive view, put the maths in
`analysis/` or `core/` first and have the widget import it.

**Theme per widget, never through `pg.setConfigOption`.** `background`/`foreground`
are process-global — the same mistake as the Report Inspector's import-time
matplotlib `rcParams` mutation, which CLAUDE.md already records as a pitfall. Use
`GraphicsLayoutWidget.setBackground(...)` plus `AxisItem.setPen/setTextPen` (see
`theme_colors()` / `style_plot_item()`), driven by the `dark_mode_graphics` setting
passed into the constructor. `imageAxisOrder="row-major"` and `antialias=False` *are*
set globally at module import — those are correctness/performance defaults with no
per-widget meaning, unlike colour.

**Pin the Qt binding before importing pyqtgraph:**
`os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")` at the top of any module that
imports it. pyqtgraph probes for every binding, and a frozen build that sees another
one on the path can bind to the wrong Qt. The spec excludes `PySide6`/`PySide2`/`PyQt5`
for the same reason, and `MainWindow._open_data_inspector` wraps its deferred import
in `try/except ImportError` so a stale env shows a `QMessageBox` instead of a traceback.

**Only one tool accepts the mouse at a time.** With several draggable items in one
panel (cross-section line, ROI, swipe divider) a click is ambiguous unless the
inactive ones are genuinely inert — not merely hidden. Locking takes three steps
together: `item.translatable = False`, `handle.setVisible(False)` **and**
`handle.setEnabled(False)`, plus `setAcceptedMouseButtons(Qt.MouseButton.NoButton)`.
See `LinkedImageView.set_line_locked` / `set_roi_locked`, driven by the window's
`Tool:` combo, with a hint label stating the current click behaviour.

### Overlays on a pyqtgraph image — indexed label image + LUT, not an RGBA rebuild

Draw a mask as its own `ImageItem` above the base image, holding a **uint8 {0,1}
array with a 2-entry RGBA lookup table** (`[[0,0,0,0], [r,g,b,alpha]]`), not an
`H×W×4` RGBA buffer. Changing colour or opacity then rewrites two LUT rows instead
of rebuilding megabytes, and the overlay costs 1 byte/px rather than 4. See
`LinkedImageView.set_overlay` / `set_overlay_style` / `_overlay_lut`.

### Registering several ImageItems with one ColorBarItem

`ColorBarItem.setImageItem` **replaces** its `img_list` rather than appending, and it
accepts a list — so pass every item in one call:
`setImageItem([base, overlay], insert_in=plot)`. This is what guarantees the two
halves of an A/B wipe can never drift onto different scales, including when the user
drags the bar's level handles. Do **not** hand-sync via `sigLevelsChanged`: that
signal fires only on interactive drags, never on a programmatic `setLevels`, so a
hand-rolled connection silently does nothing (`_update_items` is what propagates).

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

### Getting GUI state to the report — attach to the image, not to `generate()`

`AnalysisThread` unpacks only two scalars from `settings` into `ReportBuilder.generate()`
(`output_dir`, `ref_seeing_arcsec`, plus format flags) — **the settings dict never reaches
the report builder**. Anything else the report needs has to travel on the objects that do
get passed: the `AstroImage`s or the `AnalysisResult`s.

The background exclusion regions use the `starless_image` precedent — a slot *declared* in
`AstroImage.__init__` with a comment naming its external populator, rather than the
undeclared-attribute style of `image.catalog`. `report_builder` reads through
`getattr(img, …, None)` so an `AstroImage` built outside the GUI (tests, `tools/`) still
works. No signature anywhere changes.

**Where in `_execute()` the assignment happens is load-bearing, and it moved.** While
Section 3g was diagnostic-only, assigning just before the report call was fine because
nothing upstream read the slot. Now `estimate_background()` does, so the assignment sits
**before the background pre-pass** — and it is `set_background_exclusion_mask()`, which
invalidates, rather than a bare attribute write. Anything the background depends on has to
be attached before that pre-pass or the idempotency guard hands every later caller a
background estimated without it.

### A dialog that owns a drawing mode needs no second checkable button

The ROI and Line tools are checkable `QPushButton`s on the control panel, mirrored by
non-checkable toolbar `QAction`s that call `.click()` on the real button — that pattern
exists to avoid a second checked-state to keep in sync. A tool that lives in its **own
window** sidesteps the problem entirely: `Background Regions…` is a plain `QAction` that
opens a dialog, and the control panel gets only a **status label** (`set_bg_exclusions`,
mirroring `set_roi`/`set_line`) plus the `settings()` key. There is no mode on the main
window to toggle, so there is nothing to desynchronise. Prefer this shape for any future
tool complex enough to want its own window.

Clear the state in `MainWindow._on_image_loaded` alongside `_roi`/`_crosshair`. Normalised
regions can never fall out of bounds, so unlike the ROI they need no re-validation ladder
in `_on_run` — but they were still drawn against different sky, so they must not survive a
new image.

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

### Statistical significance is saturated in this report — magnitude is what discriminates

Every per-pixel Mann-Whitney in Section 8 runs over 10⁴–10⁶ spatially autocorrelated pixels.
Measured against a spatial block bootstrap on the project's own maps, the naive per-pixel SE
is understated by **6–46×** (`std_3px` 38×, `std_10px` 46×, `log_1.5` 18×), so every p-value
in those tables saturates at p < 0.001 regardless of effect size and the `n.s.` branch of
`_format_significance_html` essentially never fires.

**Fixing the interval does not fix the question.** The decisive test: two disjoint 20-frame
stacks of the *same* M31 sub-frames — an image pair whose correct answer is exactly zero —
still produced bootstrap intervals excluding zero on every metric. A correct CI cannot say
"no visible difference"; only a magnitude threshold can. `core/spatial_stats.py` gives the
honest interval, `core/practical_significance.py` gives the label, and both are needed.

### Block size comes from the SE ladder, never from the correlation length

The textbook rule (block > 2× the integral autocorrelation length) is wrong on these fields.
Section 8's `std_3px` difference map has an integral length of ~7 px yet its bootstrap SE keeps
growing out to 224 px blocks, with the growth per doubling *accelerating*: 1.13, 1.14, 1.26,
1.53, 1.78, 1.86. The autocorrelation explains it — ρ falls to 0.01 by lag 4 then sits at ~0.008
out to lag 256 without reaching zero. A long, weak tail contributes almost nothing to the
correlation length and dominates the variance of a whole-frame mean.

Two consequences. `auto_block_size` walks `se_ladder` and picks the rung past the **last**
violation, not the first rung that passes — on an accelerating curve the smallest rung always
passes a local check, which is how an earlier version declared a badly non-converged field
converged. And `converged=False` is a *result*: it means no valid block size exists for that
field, and the caller must say so rather than quote a still-growing interval.

### Aggregating labels: median within a family, max across axes

`consensus_label` (median) and `overall_label` (max) are not interchangeable.

Section 8's metrics are ~11 strongly-correlated estimates of one quantity, and the maximum of a
set of correlated noisy estimates is just its noisiest member. Measured on the SCT Ha6/Ha12 pair:
ten of eleven calibrated metrics reported "no visible difference" and one reading only 2× its own
null floor reported "noticeable" — taking the max made that single weak detection the report's
headline verdict on a filter pair its owner expects to be indistinguishable.

Across *independent* axes (sharpness / noise / detail) use the max, for the opposite reason: a
difference material on any one axis is material to the viewer. This two-level structure is what
recovers the deconvolution-sharpen case — Section 8's own consensus reports nothing on it, and
the headline reaches "material" through the FWHM axis.

### Practical labels anchor to FWHM and SNR; the detail axis needs a per-comparison reference

FWHM and global SNR are direct physical measurements needing no reference, and together they rank
all six real comparisons the way their owner reads them. They can always carry a label.

The Section 8 detail metrics cannot. Their null floor — the reading between two stacks of
identical data — spans **15× across 14 filter sets** (58× for the acutance scalars), and nothing
measurable predicts it: pixel fraction above 3σ r²=0.35, detail-map contrast 0.20, global SNR
0.19, SNR×√N 0.16, sub count 0.14, sky σ 0.04. It does not split by binning, plate scale, or
broadband-vs-narrowband either.

So **do not ship a floor constant.** For a same-conditions A/B comparison the floor is obtainable
from the comparison's own data — split A's sub-frames into disjoint halves, and B's likewise —
and that is the only defensible source. Absent one, report the detail magnitude with no practical
reading rather than a borrowed number.

### Reject outlier sub-frames before measuring anything about a floor

A single bad sub does not raise a floor slightly; it *becomes* the floor, in a way that looks
exactly like a real measurement. Dumbbell/Ha6 initially measured 0.46 log units (a 2.9× apparent
difference between identical data). The null pairs read −0.462, −0.458, −0.449, **+0.429** —
near-identical magnitude with the sign flipping according to which half the bad frame landed in.
One frame carried 9.0× the set's median fine-scale structure; three more sat at 1.5–1.7×; the
remaining ten at 0.91–1.07×. After rejection the floor fell **26×** (`std_3px` 0.5007 → 0.0192).

Taking a max or upper quantile over partitions therefore writes a data-quality defect straight
into the floor and suppresses every real difference beneath it. `_reject_outlier_frames` runs
first, and the same reasoning explains an earlier unexplained spike at 3-frame depth.

### Stars dominate the detail mask — remove them before any spatial-detail calibration

`gui/analysis_thread.py` hands Section 8 `self._starless_a or img_a`, so the report analyses the
**starless** image whenever one is attached. Any calibration run on star-containing stacks is
therefore fitting the wrong image class — and stars are the sharpest features in a frame and the
most affected by blur, so the "signal" being calibrated is largely star response.

Measured: the top-5% detail mask is **56% star pixels on M31** against 8.8% star coverage (6.4×
enrichment), and 14.8× enrichment on IC1179. After SyQon Axiom removal the enrichment falls to
**0.4×** — the mask actively avoids former star sites, so no residual artefacts re-enter it.

Star removal is applied to the **stack**, never to individual subs. It is a non-linear,
content-dependent CNN operation: run per-sub it does something different to each of N noisy
frames, plausibly removing noise peaks as stars and altering the noise realisation differently in
every sub, which corrupts the very quantity a null pair measures. Applied to a block-stack it runs
once, on input inside the model's trained SNR regime. Never let a block fall below ~3 subs.

### Degradation studies must preserve the noise floor

A plain post-stack Gaussian blur lowers detail *and* noise together: measured on a real frame,
σ=2 px drops the noise floor to 0.46× and *raises* `snr_global` from 17.4 to 20.5. Calibrating
against that conflates two independent things, since real filter differences change detail and
noise for unrelated physical reasons. A softer optical system instead blurs the photon
distribution before shot and read noise are realised.

`_blur_preserving_noise` adds back the missing variance; the same blurs then give 17.3 / 16.3 /
16.0, falling as they should, and it holds the noise floor to within 1.6% on a frame of known σ.
Two ways of sizing the compensation were wrong first: a MAD-of-high-pass noise estimate counts
pervasive nebula texture as noise and over-added enough to *invert* the sign of bulk `std_3px`;
and `_smoothed_sigma`'s asymptotic `σ/sqrt(4π σ_k²)` predicts 0.318 survival at σ_b=0.5 where a
discrete half-pixel Gaussian is nearly a delta. Measure the kernel's response on actual white
noise instead — signal-free by construction and exact for the discrete kernel.

### Acutance: the best detail metric, and an absolute value that is not a quality score

Discriminability (signal from a known 1.0 px blur ÷ null floor) across three nebula targets:

| metric | population | M31 Lum | Horsehead Ha | Dumbbell Ha6 |
| --- | --- | --- | --- | --- |
| `loclap_1.5` | top 5% | **7.9** | 2.2 | 2.3 |
| `lap_var_1.5` | nebula mask | 3.1 | **3.1** | 1.7 |
| `std_3px` | top 5% | 0.6 | 1.5 | **4.4** |
| `grad_energy_1.5` | nebula mask | 0.3 | 1.4 | 0.5 |
| `grad_energy_3.0` | nebula mask | 0.1 | 0.9 | 1.4 |

The two best are the *same quantity* at different aggregation — variance of the Laplacian,
localised (8j `loclap`) or whole-nebula (8m `lap_var`). **`grad_energy` mostly cannot separate a
real 1 px blur from sampling noise** despite Section 8m presenting it as `lap_var`'s equal.
σ=1.5 dominates σ=3.0 for both. `lap_var` takes no `top_percent` parameter, which makes it immune
to the unresolved question below.

**But its absolute value is a content measure, not a sharpness measure.** Correlated across 14
filter sets: `lap_var_1.5` vs pixel-fraction-above-3σ **r = −0.94**, vs global SNR r = −0.77.
Targets that fill the frame with smooth structure have low Laplacian variance *because they are
smooth*; a star field with stars removed has high variance from residual point structure. It is
valid only as an A-vs-B ratio on the same target — never as an absolute quality score across
images. (`lap_var_1.5` vs `lap_var_3.0` is r = 0.98, so the acutance scalars are near-redundant
with each other; the ratio-based `nc_score` metrics instead track SNR *positively* at r = 0.92.)

### What the report claims, and what it deliberately refuses to claim

The report carries practical labels on **two axes only**: sharpness (PSF FWHM)
and noise (global SNR). Both are direct physical measurements needing no
reference image, and together they label all six real comparisons the way their
owner reads them.

Section 8's detail metrics carry **no label**, and this is a designed refusal
rather than an unfinished feature. Labelling them needs a null floor, that floor
does not transfer between datasets (15× spread across 14 filter sets, six
candidate predictors all r² ≤ 0.35), and `resources/metric_calibration.json`'s
floors were all measured on M31. Wiring them in would give every comparison
labels derived from a different target's noise. The Executive Summary says so in
prose. **Do not "finish" this by pointing `verdict_for_metric` at the shipped
calibration** — the honest completion is a floor measured from the comparison's
own sub-frames.

Section 8l reports a **block-bootstrap CI** instead, computed in
`_localmax_stats` (`analysis/image_filters.py`) where the mask already exists and
where it runs inside the analyzer's own 5-way pool, rather than serially at
report time. A non-converged interval is labelled "still widening" in the table —
it is a lower bound on the uncertainty, not the uncertainty.

### `SECTION8_LOCALMAX_TOP_PERCENT` has no universal optimum

Sweeping it 1–100% and scoring by discriminability gives a trimodal answer across 15
(set × metric) combinations: `1,1,1,1,1, 5,5, 10,10,10, 50, 100,100,100,100`. Some metrics want
the brightest 1%, others the whole frame, and it flips by target. The shipped 5% is a reasonable
middle choice that no single constant improves on; setting it *per metric* is defensible, guessing
a better global value is not.

`SECTION8_LOCALMAX_REGION_FRACTION` (the dilation) is settled: mean discriminability 2.37 at 0.5
versus 2.34 at 1.0 across 105 combinations. It makes no material difference and does not need
exposing as a user control.

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

**Locating `astrolab` from a Claude Code tool call.** The Bash/PowerShell tools run a
non-interactive shell that has not sourced the user's profile, so `conda`/`python` are
not on `PATH` and `conda activate astrolab` fails with "command not found" even though
it works fine in the user's own terminal. Skip environment activation entirely and
call the env's interpreter directly — this is the most reliable path and needs no
`conda` on `PATH` at all:

```text
C:\Users\bmant\anaconda3\envs\astrolab\python.exe script.py
```

Only reach for `conda.exe` itself (env creation, package install/list) — its `Scripts\`
folder sits next to the base install, e.g. `C:\Users\bmant\anaconda3\Scripts\conda.exe`.
If either path has moved, `Get-ChildItem -Path "$env:USERPROFILE" -Filter python.exe
-Recurse -Depth 4` (PowerShell) will re-find it — but check these known paths first
rather than re-deriving them every session.

Output binary names: `AstroImageLab.exe` (Windows), `AstroImageLab` (macOS / Linux).
The spec post-build step automatically creates a platform-labelled zip:
`AstroImageLab-win64.zip`, `AstroImageLab-macos.zip`, or `AstroImageLab-linux.zip`.

PyQt6 **must** be installed via `pip`, not `conda` — the conda-forge PyQt6 package
uses a different DLL layout that breaks PyInstaller hook discovery on Windows.

`pyqtgraph` backs the Data Inspector. It is declared in `environment.yml`,
`requirements.txt` and `requirements-build.txt`, and bundled via
`collect_all("pyqtgraph")` in the spec (it ships `icons/`, `colors/maps/*.csv` and
`.ui` files and resolves submodules lazily, so static analysis misses them). The
spec also excludes `PySide6`, `PySide2` and `PyQt5` so pyqtgraph's binding shim
cannot pull a second Qt into the bundle. It is deliberately **not** in
`requirements-test.txt` — the suite stays headless.

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
pip install -r requirements-test.txt          # one-time setup; not in environment.yml
pytest tests/ -m "not slow" -n auto           # fast suite (945 tests; parallel via pytest-xdist)
pytest tests/ -m slow                         # slow/integration tests (full FITS generation)
pytest tests/ --cov=analysis,core,synthetic,report --cov-report=html
```

### Design principles

- **Headless only** — no PyQt6 dependency in tests. Covers `analysis/`, `core/`, `synthetic/`, `report/`. **`gui/` is therefore untested, which is a design constraint, not an oversight**: any logic a GUI feature needs must be pushed down into `analysis/` or `core/` to be verifiable at all. `analysis/inspector_regions.py` and `core/inspector_catalog.py` exist for exactly that reason — see the pyqtgraph convention above for the bug that motivated the split.
- **Generated fixtures** — `tests/conftest.py` writes a 512×512 hand-crafted FITS (30 Gaussian stars, ~1 s) as the session fixture for all analysis tests. No binary fixtures committed to git. `tests/bg_frames.py` is the ground-truth variant for background work — it returns the true sky surface alongside the data so assertions can be quantitative; see the "Ground-truth fixtures" convention above.
- **Report-section tests build the section directly** — `tests/test_report/test_executive_summary.py` calls `ReportBuilder()._section_verdict(...)` / `._section_summary(...)` with hand-built `AnalysisResult`s rather than generating a whole report, which is what keeps them fast enough to be worth running. Two gotchas it encodes: assert against `html.split("<details")[0]` when checking that a label is *absent*, because the methodology info-box legitimately names all four labels; and set `builder._single_image` explicitly, since both sections early-return on it.
- **Acceptance against known answers** — `tests/test_core/test_practical_acceptance.py` pins the six real comparisons in `AstroLabTestData/FilterCompare/` against the reading their owner gives each pair. It runs against the *shipped* `resources/metric_calibration.json`, so regenerating the calibration cannot silently break the known answers, and it skips (rather than fails) when no calibration exists so a fresh checkout stays green. When a labelling rule changes, this is the test that decides whether the change was an improvement.
- **Slow marker** — `@pytest.mark.slow` gates tests that call `SyntheticGenerator.generate(preview=False)` (full 1920×1080 FITS, ~30 s). CI runs with `-m "not slow"`.
- **Smallest camera for generator tests** — `"Player One — Mercury-M"` (1920×1080 full-res; 480×270 in preview mode) is the lightest camera in `synthetic/cameras.py`.
- **Share identical-input analyzer/generator calls via a fixture** — if two or more tests in a file call `SomeAnalyzer().analyze(astro_image_a)` (or `StarCatalogBuilder().build(...)`, `SyntheticGenerator().generate(...)`) with the exact same arguments, that result must come from a class- or module-scoped fixture, not a fresh call in each test body. `SpatialDetailAnalyzer.analyze()` in particular runs LoG/wavelet/local-sigma/entropy/local-maxima detection across every configured scale — cheap to assert against, expensive to recompute. Only call it fresh when the test genuinely needs different arguments (a different ROI, crosshair, monkeypatch, or a second call specifically to test determinism/parameter-wiring, e.g. `test_second_run_consistent` / `test_higher_top_percent_does_not_shrink_masked_pixel_count`). A class-scoped fixture that wraps a value (not just computes one) still needs the `@pytest.fixture(scope="class") @classmethod def name(cls, ...):` form — a plain `def name(self, ...):` class-scoped fixture is deprecated in pytest 9 and breaks in pytest 10.

### CI

`.github/workflows/ci.yml` — triggers on push/PR to `main`. Two jobs:

- **test** — matrix across `windows-latest`, `ubuntu-latest`, `macos-latest`; uses `requirements-test.txt` (no PyQt6); runs `pytest -n auto` (pytest-xdist) under a 20-minute job timeout. Add `CODECOV_TOKEN` to repo secrets to enable Codecov upload; coverage is flagged per OS.
- **build** — runs after all test jobs pass (`needs: test`); same OS matrix; uses `requirements-build.txt`; uploads `AstroImageLab-{win64,macos,linux}.zip` as workflow artifacts downloadable from the Actions tab.

### Test fixture pitfalls

| Pitfall | Rule |
| --- | --- |
| Inline FITS too small to load | `_load_fits` skips HDUs where `max(shape) <= 100`. Test FITS must be at least 101×101; use 128×128 for safety. |
| `float(mtf(array, m))` raises TypeError | `mtf()` returns a same-shape array, not a scalar. Index with `[0]` or pass a scalar input. |
| `PSFAnalyzer.analyze()["figures"]` KeyError | `figures` is only added when `n_stars_used > 0`. Guard with `if result["n_stars_used"] > 0`. |
| `contrast_ratios_b` / `entropy_contrast_ratio_b` always present | `SpatialDetailAnalyzer.analyze()` always includes `contrast_ratios_b: {}` and `entropy_contrast_ratio_b: {}` even in single-image mode. Neither is ever `None` or absent — check `not b_ratios` / `not ecr_b` instead. |
| Background2D fails on tiny images | `estimate_background()` with default `box_size=64` needs the image to be larger than the box. Any image used in analysis tests should be at least 128×128. |
| CI fast suite crossed the 20-min job timeout | Root cause was not one slow test — it was ~80 near-identical `AnalyzerX().analyze(astro_image_a)` / `.build(astro_image_a)` calls scattered across `test_psf_analyzer.py`, `test_edge_analyzer.py`, `test_snr_analyzer.py`, `test_halo_analyzer.py`, `test_power_spectrum.py`, `test_star_catalog.py`, and `test_spatial_detail.py` (one parametrized case there alone re-ran a full two-image `SpatialDetailAnalyzer.analyze()` 15× just to check one absent dict key). Fixed by collapsing same-input calls into shared fixtures (see the "Share identical-input analyzer/generator calls" design principle above) plus adding `pytest-xdist`/`-n auto`. Before assuming a CI timeout needs a longer `timeout-minutes`, run `pytest --durations=25` locally — a large `setup` duration shared by many tests is fine (paid once), but many `call` entries at the same duration for the same analyzer is a missing-fixture symptom, not a slow-runner problem. |

---

## Known Pitfalls

| Pitfall | Rule |
| --- | --- |
| Inspector dark theme bleeds into report figures | Apply dark mode only in `analysis_thread.run()`, never at import time |
| An aspect-locked pyqtgraph `ViewBox` answers a resize by preserving scale, not refitting | Any layout change leaves the image at the wrong size until the range is re-applied. The Data Inspector opened at **30.9%** of the area it could fill, because `_load_panels()` ran during `__init__` against a placeholder-sized widget (585×410) and Qt then laid out to 409×432. Hiding Image B for swipe mode reproduced it far worse — the range blew up to `[-2228, 3063]` for an 835-px-wide image. Two fixes together: a one-shot `QTimer.singleShot(0, …)` from `showEvent` for the initial fit, and a **debounced handler on the ViewBox's own `sigResized`** for everything after. Reacting to the *window's* `resizeEvent` is not enough — hiding a splitter child resizes the panel without resizing the window — and a refit issued from the handler that triggered the layout change is undone, because Qt finishes the re-layout after that handler returns. The refit itself is `_capture_view()` + `_apply_view()`, which is idempotent only because the capture clamps to the image bounds (see the next row). |
| Round-tripping an aspect-locked view range makes it drift outward | Capture-then-reapply compounds: aspect lock *expands* whichever axis has spare room, so storing the expanded range and re-applying it lets the other axis expand in turn. Measured drift to `(-0.331, 1.331, -0.69, 1.623)` in normalised coords after a few panel changes. Clamp the captured rectangle to `[0,1]` before storing — "which part of the image am I looking at" is the only part worth carrying to a differently-sized panel, and clamping makes the round trip converge. |
| `pg.BarGraphItem` ignores `PlotItem.setLogMode` | `setLogMode` only transforms `PlotDataItem`s. With a log y-axis, bars plotted with raw counts render against an axis labelled `10^173`, `10^-227` — nonsense. Pass **log10 heights** in the ViewBox's own (already-log) coordinates instead, drop empty bins (`log10(0) = -inf`), and give the bars a `y0` slightly below zero so single-count bins stay visible. See `RatioHistogramPlot.update_histogram`. |
| pyqtgraph axes relabel small dimensionless values with an SI prefix | A log ratio spanning ±0.15 gets rendered as "±800" with a `(x0.001)` suffix on the label — which reads as a completely different quantity. Call `axis.enableAutoSIPrefix(False)` on every axis showing a ratio or a count. Easy to fix on one widget and miss on another: it was fixed on the cross-section's axes and missed on the histogram's until a screenshot showed it. |
| `QWidget.grab()` misrenders a `GraphicsLayoutWidget` | It showed the Data Inspector's swipe panel at roughly **1/5** its real size while `viewRange()`, `mapViewToDevice()`, and `pg.exporters.ImageExporter` all independently agreed the layout was correct — and it did so consistently, through a settled real event loop, in a state where the side-by-side panel grabbed correctly. Nearly caused a "fix" to a non-bug. For pyqtgraph panels use **`pg.exporters.ImageExporter(plot_item).export(path)`** for visuals and `viewRange()` / `mapViewToDevice()` for geometry; treat `grab()` output as untrustworthy. This is a sharper instance of the "OS-level screenshots are unreliable" row below — here even Qt's own in-process grab lies. |
| `ColorBarItem.setLevels` does not emit `sigLevelsChanged` | That signal fires only on an interactive drag of the bar's handles. A hand-rolled `sigLevelsChanged` connection to keep a second `ImageItem` in step therefore silently never runs. Register every item with the bar instead — `setImageItem([base, overlay], insert_in=plot)` — and let `_update_items` drive them all. Note `setImageItem` *replaces* `img_list` rather than appending, so one call must carry the whole list. |
| A composable region model needs the no-threshold case spelled out | With ROI-as-domain and threshold-as-split, writing `in = domain & split; out = domain & ~split` unconditionally makes plot 2 **always empty** when no threshold is set, because `split` is all-True. The four cases only work if the no-threshold branch is separate: `in, out = domain, ~domain` — so an ROI alone splits inside-vs-outside, and with no ROI either, `~domain` is empty and plot 2 is correctly blank. |
| A "top N%" mask built as `A_cut AND B_cut` is much smaller than N% | Per the spec the threshold applies to each image independently and the masks are AND-ed, so the result is the *overlap* of two tails and can only be ≤ N%. On real noise-dominated Local-σ maps, top-10% of each gave 52,271 and 52,272 pixels but only **9,347** in common — 1.8% of the frame, not 10%. Expected behaviour, not a bug; assert against `min(solo_a, solo_b)` rather than against N% when testing it. |
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
| `astroalign.register()` fails with "Input type for source not supported." on a large/bright target — this is not a dtype problem | `gui/analysis_thread.py::_align()` already casts both images to float64 before calling `aa.register()`, and astroalign's own `_find_sources` re-casts to float32 internally regardless (`image.astype("float32")` right before `sep.extract`), so dtype was never the mechanism. `find_transform()` wraps its whole source-detection step in a blanket `except Exception: raise TypeError("Input type for source not supported.")`, so *any* failure inside `sep` gets relabeled with that one misleading message. The real cause, confirmed by calling astroalign's internal `_find_sources` directly on the failing array: `sep.extract`'s internal pixel buffer (default `extract_pixstack = 300_000` active above-threshold pixels) overflows when a large, bright, extended target — measured on the Orion Nebula core, 523 061 pixels (4.5 % of a 2822×4144 frame) above a 5σ threshold — forms one connected blob past that cap. `_align()` now calls `sep.set_extract_pixstack(SEP_EXTRACT_PIXSTACK)` (1,000,000, a process-global sep setting, harmless on ordinary starfields) before every `aa.register()` call. When astroalign raises this exact message, reproduce with `astroalign._find_sources(astroalign._bw(arr), detection_sigma=5, min_area=5, mask=astroalign._mask(arr))` on each image directly — it surfaces the real underlying exception instead of the generic one `register()`/`find_transform()` re-raise. |
| Adding a float64 cast in analysis code | Don't. All image data is float32 after `AstroImage.load()`. The only float64 exception is `astroalign` in `gui/analysis_thread.py`. Redundant float64 casts waste memory and defeat the float32 performance gains. |
| Mixed float32/float64 arithmetic silently widens to float64 | NumPy upcasts when operands differ (e.g. `float32_array - float64_scalar`). If photutils ever returns a float64 background model, `background_subtracted()` will silently return float64. Guard by adding `.astype(np.float32)` at the end of `background_subtracted()` in `astro_image.py` if this is observed. |
| `_section_snr` crashed when SNR metric is unchecked | `_plot_snr_pair` (`report_builder.py`) built a `panels` list filtered to non-`None` entries but never checked whether it was empty before calling `plt.subplots(1, len(panels), ...)` — 0 columns raised `ValueError: Number of columns must be a positive integer, not 0`. Hit whenever SNR is unchecked while another metric (e.g. Power Spectrum) is run. Fixed with an early `if not panels: return None` guard, matching `_plot_radial_overlay`/`_plot_radial_ratio_db`; both call sites already pipe the result through `_img_tag`, which turns `None` into `""`. |
| New Section 8 panel key doesn't need Report Inspector code changes | `gui/report_inspector.py` is fully generic — driven entirely by a companion `<stem>_inspector.npz` (raw float32/uint8 arrays) plus an embedded `catalog_json` built in `report_builder.py::_write_inspector_file`. `_panel_display_name`/`_panel_concept` dynamically parse any `panels` dict key prefix, so a new `SpatialDetailAnalyzer` panel family auto-appears in the inspector with zero inspector-side changes. A genuinely new *visual type* is a different story: the inspector only knows how to `imshow` 2D/RGB arrays (side-by-side or slider-reveal), so scatter-style plots (Section 8's `corr_*` correlation figures, interleaved into 8d–8j right after each map figure via `_family_figs_with_corr`) must stay static-HTML-only unless new inspector canvas code is written. |
| Section 3's sub-headings have the same cross-reference hazard as Section 8's | Section 3 currently runs 3a–3g (3e Background Model, 3f Background Gradient Analysis, 3g Source-Masked Background Check), all inside the single `_section_snr` function — there is no `_section_3e`. 3g was **appended** after 3f, so it needed zero relettering (the same move as 8k/8l/8m). Adding it did require updating the prose that described the gap it closes: 3e's "Known limitation" paragraph and the Data Inspector's Background/RMS `concept=` text both asserted that no source mask exists anywhere, and now point at 3g instead. When a new section removes or narrows a documented limitation, `grep` for the text describing that limitation — HTML renders a stale claim without error. **This bit twice.** Adopting the masked estimate inverted 3g's own framing — "this section is diagnostic only… every SNR still uses the unmasked estimate" became flatly false, not merely stale — and the same claim was duplicated in the npz `concept=` string. Both branches are now built from a single `_masked_now` flag hoisted above the 3e block (3f and 3g both branch on it too), and tests assert the *absence* of `"diagnostic only"` as well as the presence of the replacement. When a section's premise can flip, gate the prose on one computed flag rather than writing the claim into several literals. **A third instance, of the opposite kind:** `test_warns_that_global_snr_is_not_comparable_across_the_setting` asserted two substrings that `git log -S` shows never existed in `report_builder.py` at all — the test shipped in `883f90f` without its implementation and stayed red for months. 3e now carries that warn-box (global SNR moves sharply when the masked background is toggled because it changes *which pixels qualify*, not just their values). A long-red test is worth `git log -S`-ing before assuming the prose merely drifted; "the feature was never written" and "the wording changed" need different fixes. |
| Section 3f would fit a polynomial to a polynomial | Once 3e's adopted model *is* a fitted surface, refitting `background_display("model")` returns the surface's own coefficients with a residual of exactly zero. 3f now reuses `res.fit` (already a `fit_background_surface` result, so `_bgfit_table_html` is unchanged) and plots residuals **against the mesh cells** — the actual independent measurements — with mask-removed cells left NaN so they read as "no data" rather than as a zero residual the fit never earned. `_plot_background_map_pair` gained `axis_unit=` because that array is mesh-resolution, not pixels. |
| Renumbering a Section 8 subsection misses caption cross-references | Section 8's sub-heading letters (currently 8a–8m: 8a Background/Key Terms, 8b Original Image, 8c Mask Overview, 8d–8f detail-based families [LoG, Wavelet, Gradient], 8g–8h contrast/texture-based families [Local σ, Local Entropy], 8i Local Gradient Energy [windowed mean(\|G\|²) sharpness-*concentration map*, distinct from 8f's raw per-pixel \|G\| and 8m's single scalar], 8j Local Variance of Laplacian [windowed var(signed LoG) focus-*concentration map*, distinct from 8d's raw per-pixel \|LoG\| and 8m's single scalar — variance rather than 8i's mean-of-squares, since the Laplacian is signed], 8k Noise-Corrected Cross-Method Overview, 8l Local-Maxima Masked Metrics [its **95% CI (block bootstrap)** column replaced a Mann-Whitney `Significance` column — see the practical-labels convention], 8m Global Acutance/Perceived Sharpness [whole-nebula absolute variance/energy scalars — var(LoG), gradient energy — as opposed to 8d–8l's nebula/background median ratio]) are referenced by literal string in caption/info-box text scattered throughout `_section_spatial` — not just in the `<h3>` tags (e.g. "see 8k for…", "(8d–8j)"). After adding, removing, or renumbering a subsection, `grep` the function (and `_SPATIAL_GLOSSARY_HTML`) for every old *and* new heading letter — HTML renders a stale cross-reference without error, it just silently misdirects the reader to the wrong subsection. A self-reference inside a family's own info box (e.g. Gradient's "same framework as the other N families") must enumerate the other letters explicitly rather than use a dash range — under a non-contiguous lettering (Gradient kept letter `8f`, so its own sibling list must skip it), a range would wrongly include Gradient's own letter. Precedent: when Weber contrast (formerly 8h) was replaced by Local Entropy, keeping the same letter for the new family avoided a renumbering pass entirely — every existing `8h`/`8d–8h` cross-reference stayed numerically valid, only the prose describing 8h's content changed. Second precedent: 8k, 8l, and 8m (in that order) were first *appended* after 8j rather than inserted mid-sequence, so adding them required zero renumbering of the sequence that existed at the time. Third precedent — a full reletter, not just an append: when the user later asked to move the (by-then-existing) Local Gradient Energy/Local Variance of Laplacian sections to before the NC Cross-Method Overview section, physically moving the HTML blocks without relettering would have produced a document reading `8a…8h, 8l, 8m, 8i, 8j, 8k` — non-sequential and defeating the point of lettered headings. The fix was a full content-to-letter remap (old 8i→8k, old 8j→8l, old 8k→8m, old 8l→8i, old 8m→8j) applied consistently across `_section_spatial`, `_SPATIAL_GLOSSARY_HTML`, and every test file reference — plus fixing positional language that became stale from the *reorder itself*, not just the rename (e.g. 8m's own box said its 8i/8j companions were "further down this section" when they were still 8l/8m; after the reorder they sit *earlier*, so the text had to change from "further down" to "earlier in this section", independent of the letter substitution). The reorder also had a welcome side effect: several previously-fragmented cross-reference lists (`8d–8h, 8l, 8m`) became genuinely contiguous ranges (`8d–8j`) once the physical and alphabetical orders realigned — prefer the simplified contiguous range over the old enumeration wherever the new physical adjacency makes it accurate, don't just find-and-replace old letters for new ones. Fourth precedent: adding 8j also surfaced a reusable formula — `_windowed_variance` was factored out of 8g's `_compute_std_map` (same `mean_sq - mean**2` box-filter identity, `_compute_std_map` now just `sqrt`s it) specifically so 8j could call the unsquashed variance directly; when a new family needs a formula an *existing* family already computes, check for this kind of extraction before duplicating the math. |
| Two metrics computed inside one family method must not share a scale-keyed dict | 8i (Local Gradient Energy) is computed *inside* `_gradient_analysis` alongside 8f's own gradient-magnitude metric, reusing its already-computed `gm_a`/`gm_b` maps rather than dispatching a 6th concurrent `ThreadPoolExecutor` future — cheaper (no duplicate convolution) and avoids bumping the hard-coded `max_workers=5`. But `_gradient_analysis`'s `partial` dict already had `localmax_log_ratio`/`localmax_log_ratio_err` sub-dicts keyed by `sigma` for the gradient family's own local-maxima entries; naively writing 8i's local-maxima log-ratio into the *same* sigma-keyed dict would silently overwrite gradient's entries with local-gradient-energy's (both use identical sigma keys 1.5/3.0/6.0, and dict assignment never raises `KeyError` on overwrite). Fixed with distinct sub-dict names (`localgrad_localmax_log_ratio`/`_err`) so 8l's `localmax_log_ratios_by_method` dict can read both as independent `{sigma: value}` series. Whenever folding a second metric into an existing family method to reuse its intermediate arrays, audit every dict inside that method's `partial` for whether its keys are the metric's own scale/level (safe to add new same-shaped dicts) versus something the two metrics would collide on if merged into one. Confirmed as a repeatable convention, not a one-off: the identical pattern was applied again when 8j (Local Variance of Laplacian) was folded into `_log_analysis` alongside 8d's own \|LoG\| metric — `loclap_localmax_log_ratio`/`_err`, kept distinct from `_log_analysis`'s existing `localmax_log_ratio`/`_err`. (These prefix strings — `localgrad_*`, `loclap_*` — are internal dict/result keys, not the section's display letter, so they were correctly left unchanged by the later 8i/8j/8k/8l/8m reletter; only the `<h3>` labels and prose describing them moved.) |
| Stale ROI crashes Section 8 with "index -1 is out of bounds for axis 0 with size 0" | `MainWindow._on_image_loaded()` (`gui/main_window.py`) now resets `self._roi`/`self._crosshair` to `None` (plus both panels' visual overlays, via `ImagePanel.clear_roi_overlay()`/`clear_line_overlay()`) on every new main-image load — the choke point is `ImagePanel.image_loaded`, emitted only from `_open_file`/`load_path`, never from `set_starless_path`, so attaching a starless companion correctly does *not* wipe an existing ROI/line. Before this proactive reset existed, a stale ROI drawn against a previous, larger image pair silently went out of bounds for a smaller replacement: NumPy doesn't raise on an out-of-range slice — `norm_a[ry0:ry1, rx0:rx1]` silently returns a zero-size array — so the crash surfaced much later and far from the real cause: `SpatialDetailAnalyzer._plot_mask_illustration → _stretch_for_display → np.percentile(empty_array, ...)`. The same unguarded `bgsub[y0:y1, x0:x1]` pattern exists in `power_spectrum.py::_extract_roi` and `edge_analyzer.py::analyze`, so a stale ROI could corrupt those sections too. `MainWindow._on_run()`'s validation (checking `self._roi` against every loaded image's `data.shape` right before `settings["roi"]` is set, clearing it with a `QMessageBox` if it no longer fits) is kept as defense-in-depth for any future code path that changes loaded-image dimensions without going through `_on_image_loaded`, but the normal load→run flow now clears stale state at the source instead of catching it reactively at Run time. |
| `binary_dilation` on a loose sigma-threshold mask amplifies noise, not signal | Growing a boolean mask straight from a threshold cut (e.g. Section 8's nebula mask at 1.7σ) dilates *every* True pixel, including scattered single/few-pixel noise-driven false positives — expected in bulk at a loose sigma cut (~4.5% of pixels at 1.7σ one-sided). Each isolated speck balloons into a `~(2·dilation_px+1)²`-px blob, inflating the mask area by 4x+ and diluting any signal-vs-background metric computed over it. Fix: strip small isolated connected components (`scipy.ndimage.label` + `np.bincount` size filter, same size threshold used for hole-filling) *before* calling `binary_dilation` — see `SpatialDetailAnalyzer._remove_small_objects` / `_fill_small_holes` in `image_filters.py`. Caught by comparing mask pixel counts with dilation on vs off on real (noisy) fixture data — a clean synthetic square mask (no noise) will not reveal this bug. |
| Adding GUI parameter rows clips existing text in the Parameters group | `gui/main_window.py`'s `AnalysisControlPanel.setMaximumHeight(...)` caps the whole control panel's height. Metrics / Parameters / Region & Run are laid out side-by-side (`QHBoxLayout` in `control_panel.py::_build_ui`), so the cap must fit the *tallest* group box's natural content height. Parameters (`control_panel.py`, "2. Parameters") is itself split into two side-by-side `QFormLayout`s — "General / PSF" (`form1`) and "Nebula & Local-Maxima" (`form2`) — so its own height is driven by `max(form1_rows, form2_rows)`, not the total row count across both. When adding a new parameter row, add it to whichever column keeps the two roughly balanced, then re-measure: construct `AnalysisControlPanel` headlessly and read `QGroupBox.sizeHint()` for all three boxes (see the measurement approach used when this split was introduced — a small script that imports the widget, calls `.adjustSize()`, and prints each `findChildren(QGroupBox)` entry's `sizeHint()` — is far more precise than eyeballing a screenshot, and sidesteps OS/DPI screenshot-scaling inconsistencies entirely) rather than assuming a fixed per-row pixel cost. Set `setMaximumHeight(...)` to comfortably cover the tallest of the three measured heights. |
| Cliff's delta via pairwise sign matrix doesn't scale past ~100s of samples | `arr_a[:, None] - arr_b[None, :]` is O(n1·n2) memory — fine for PSF's per-star counts, a multi-gigabyte blowup for per-pixel populations (Section 8l masks can hold 10⁴–10⁵+ pixels). Use the exact O(n log n) identity instead: `delta = 2·U/(n1·n2) − 1`, where `U` is the Mann-Whitney U statistic `scipy.stats.mannwhitneyu` already computes. See `core/stats_utils.py::mannwhitney_effect`. |
| Section HTML block gated on the wrong figure's output | A conditional HTML block that wraps *multiple* pieces of content but gates on only *one* figure's presence (e.g. `dist_html = ("<h3>...</h3>" + mask_html + dist_img + ...) if dist_img else ""`) will silently delete the other content too if that one figure is ever removed or becomes empty — this hit Section 8c, whose Nebula/Background mask illustration disappeared along with the (later-removed) log-ratio violin figure it happened to share a block with. Gate on the actual content being wrapped (e.g. `mask_fig`, if that's what must be present for the block to be worth showing), not on a sibling figure that currently always co-occurs with it. |
| Extending an analyzer method's return-tuple arity misses a call site | `_nc_score` gained a 3rd return value (`neb_std`) to support the Section 8 cross-method overview's error bars (Section 8k as of the current lettering — this pitfall predates the 8i/8j/8k/8l/8m relettering, so the section was "8i" at the time); it has 10 call sites (2 per metric family × 5 families), each needing `nc_a, noise_a = ...` changed to `nc_a, noise_a, neb_std_a = ...`. `grep` for every call site before changing a shared method's return signature — a missed site raises `TypeError: cannot unpack non-sequence`, or worse, silently mis-assigns if old and new arity both happen to unpack without error. |
| `np.percentile` threshold on a flat/constant array selects everything | A percentile-based `>=`-threshold mask (e.g. `_top_percent_mask`) degenerates when the source array has little/no spread: if every value is equal, every percentile equals that one value, so `arr >= threshold` matches 100% of pixels, not the intended top N%. Hit writing a unit test for `_top_percent_mask` with an all-zero second operand — its own 90th-percentile threshold was also `0`, making `arr_b >= 0` trivially true everywhere and swamping the assertion. When constructing a synthetic array to exercise percentile-threshold logic, give it genuine spread (or reuse the same non-flat array for both operands) rather than an all-zero/constant placeholder. |
| A cached render-to-attribute call duplicates report content | `_psf_simulation_html` called `self._psf_retention_table(sim)` twice with identical arguments — once inline into its own returned HTML, once purely to populate `self._cached_retention_html` so `_section_summary` could re-splice the same table into Section 9 later. Both calls are pure functions of `sim`, so the two copies were byte-identical, and the report silently showed the same table twice. If a value needs to reach a second section, thread the already-computed *result* through (return value, parameter, or a plain instance attribute set once) rather than re-invoking the render function a second time purely as a caching side-effect. |
| `QGroupBox` title containing a bare `&` silently swallows it (mnemonic) | `QGroupBox("3. Region & Run")` rendered as "3. Region Run" — Qt interprets `&` in a group box title as a mnemonic prefix for the next character, same as `QAction`/`QPushButton` text. Escape a literal ampersand as `&&` (`QGroupBox("3. Region && Run")`). Plain `QLabel` text does **not** have this problem (no mnemonic support unless `setBuddy()` is used), so this is specific to titles/text that Qt treats as mnemonic-aware (menus, actions, buttons, group boxes). |
| `QToolBar.addWidget(spacer)` with an `Expanding` size policy can make every action after it vanish | Adding a bare `QWidget` spacer (`QSizePolicy.Policy.Expanding` horizontal, tried both `Preferred` and `Expanding` vertical) to right-align a trailing toolbar action made that action disappear from the rendered toolbar entirely — not shifted, not overflowed into a `»` chevron, just absent — confirmed by re-rendering with the spacer removed (the action reappeared immediately, left-aligned after the preceding separator). Root cause not fully isolated; treat any `addWidget(spacer)`-for-right-alignment idiom in a `QToolBar` in this codebase as unverified until the rendered result is actually checked (see the next pitfall for how to check it without a real display). |
| OS-level screenshots of the PyQt6 GUI are unreliable for verifying a layout change | `Get-WindowRect`/`SetWindowPos`/`Screen.Bounds` from a non-DPI-aware PowerShell process and the actual rendered window disagreed with each other by inconsistent ratios (not a single uniform scale factor) on a scaled Windows display, making a window that should fit on-screen appear clipped, and vice versa — even maximizing the window didn't reliably show all of it. Prefer `QWidget.grab()` (or `QMainWindow.grab()`) from a small headless script that constructs the widget/window directly and saves the returned `QPixmap` to a PNG — this renders through Qt's own coordinate system with a consistent `devicePixelRatio`, sidestepping OS/DPI virtualization entirely, and doesn't require a visible window at all. For precise sizing decisions (e.g. tuning a `setMaximumHeight`), read `QWidget.sizeHint()`/`minimumSizeHint()` directly instead of eyeballing a rendered image — see the "Adding GUI parameter rows..." pitfall above for the exact approach. |
| Overview figure's box/marker count silently drifted from its own caption | `EdgeAnalyzer.analyze()`'s gradient-magnitude overview map was given `rois_used` — every *searched* candidate ROI (`N_CANDIDATE_EDGES = EDGE_N_TOP_EDGES * 3`, e.g. 9) — instead of only the edges actually *accepted* into `edges` (capped at `EDGE_N_TOP_EDGES`, e.g. 3), so the map drew 9 cyan boxes while its own caption said "three selected." Root cause: `rois_used` was assigned once, early, from the full candidate list, and never reassigned after low-quality candidates got filtered out of `edges`. Whenever a display figure loops over a list to draw one marker/box per entry, verify that list is the actually-used subset the caption describes, not the broader search/candidate pool that produced it — fixed by reassigning `rois_used = [e["roi_used"] for e in edges]` right after `edges` is finalized, before it's read by `_plot_gradient_map`/stored in the result dict. |
| A "plausible-looking" derived metric can still be measuring the wrong thing entirely | `EdgeAnalyzer._extract_esf`'s `rotation_angle = -(90.0 - angle_deg)` (present since the file's first commit) looked like a reasonable 90°-complement but actually rotated the edge **horizontally**, not vertically as its own design requires (`esf_raw = nanmean(rotated, axis=0)` averages *down columns*, so the edge must run vertically for that average to stay on one side of the transition). The bug was invisible to `tests/test_analysis/test_edge_analyzer.py` because every test there checked `_esf_quality` (a monotonicity ratio) or structural shape (no NaN, normalized range) — never the actual measured width against a *known* ground truth — and the wrong-orientation artifact (a disc-boundary/interpolation trend) happened to also be smooth and monotonic, so it passed every existing gate while over-measuring width by 7–14x on a synthetic edge with a known Gaussian blur sigma. Diagnosed by rendering `rotate()`'s output for several known angles and inspecting it directly (ASCII-art / value dump, not just numbers) — the same "don't hand-derive `rotate()`'s convention" principle documented above ("Locating a point across `scipy.ndimage.rotate()`..."), applied to the rotation *angle formula* itself rather than just a post-hoc point lookup. When a metric's test suite only checks shape/monotonicity/range properties, add at least one test with an analytically-known true value (`tests/test_analysis/test_edge_analyzer.py::TestEdgeWidthAccuracy`, using the erf-profile width of a Gaussian-blurred step edge) — shape-only checks can pass on a metric that's confidently, monotonically, consistently wrong. |
| Locking `savefig()` alone doesn't fix the mathtext `ParseException` race | `core/fig_utils.py`'s `_MPL_DRAW_LOCK` (formerly `_SAVEFIG_LOCK`) was originally applied only inside `fig_to_b64()`'s `savefig()` call, on the theory that `savefig()` was the only draw-triggering call site. It wasn't: `fig.tight_layout()` also runs a full draw pass to measure text extents (titles, tick labels, legends), which hits the same non-thread-safe pyparsing packrat cache mathtext uses. `PowerSpectrumAnalyzer.analyze()` calls `fig.tight_layout()` unprotected inside its `_plot_results()`, and `gui/analysis_thread.py` runs image A/B's `PowerSpectrumAnalyzer().analyze()` concurrently in a 2-worker `ThreadPoolExecutor` (and, in parallel mode, alongside every other analyzer's figure building too) — so an unlocked `tight_layout()` in one thread reliably corrupted the cache mid-parse in another, surfacing as `⚠ Analysis failed: ... ParseException: exception raised in parse action (at char 0), (line:1, col:1)` on Section 7. Reproduced directly: 60 concurrent `tight_layout()` + locked-`savefig()` calls threw the exact same `ParseException` in a stress test; wrapping `tight_layout()` in the same lock (`core/fig_utils.py::finalize_layout()`) brought the error count to zero over the same stress test. Fixed at every call site that can run concurrently with other figure-building code: `power_spectrum.py`, `snr_analyzer.py`, `psf_analyzer.py`, `image_filters.py`, `halo_analyzer.py`, `edge_analyzer.py`, and `gui/halo_dialog.py` (its `_AnalyzeThread` isn't gated against a concurrent `Run Analysis` pass, so it's a real concurrent path too). `report/report_builder.py`'s `tight_layout()` calls were deliberately left as plain `fig.tight_layout()` — report generation runs strictly serially after every analyzer thread has already joined (`gui/analysis_thread.py`'s comment: "Report generation (always serial — needs all results)"), so there's nothing for those calls to race against; wrapping them would be lock overhead with no behavioral benefit. When adding a new figure-building method anywhere that *can* run inside a `ThreadPoolExecutor` alongside other figure code, call `finalize_layout(fig, **kwargs)` instead of `fig.tight_layout(**kwargs)` — never assume only `savefig()` needs the lock. The lock's own comment already said "savefig() is not the only trigger," but coverage still had a gap: `image_filters.py`'s `fig.colorbar(...)`/`ax.legend(...)` calls (13 sites, several using `loc="best"` auto-placement, which needs text-extent measurement to find a non-overlapping spot) ran unlocked inside `SpatialDetailAnalyzer`'s always-on 5-way `ThreadPoolExecutor`. Extended in the same session that fixed the `_compute_std_map` bottleneck below: added `locked_draw_call(fn, *args, **kwargs)` to `core/fig_utils.py` and wrapped every colorbar/legend call site in `image_filters.py` with it. |
| Dropping a new file into `resources/` is enough — no `.spec` edit needed | `AstroImageLab.spec` bundles the entire directory in one line (`datas += [("resources", "resources")]`), not a per-file list. Any new asset placed in `resources/` (e.g. `AstroImageLabSplash.png`) is automatically included in the Windows/macOS/Linux PyInstaller builds without touching the spec file — confirmed when the splash screen was switched from a procedurally-painted `QPixmap` to loading `resources/AstroImageLabSplash.png` directly via `QPixmap(path).scaledToWidth(...)`. |
| A headless script can't regain control while a `QMessageBox` is up — only a pre-armed `QTimer` can | `QMessageBox.information()`/`.question()` block the caller via a *nested* Qt event loop. A plain Python `while` loop calling `app.processEvents()` cannot run any of its own code again until the dialog closes, so it can neither detect nor dismiss the dialog itself. A `QTimer` already running *before* the blocking call is made keeps firing inside that nested loop (the same mechanism that keeps the rest of the UI responsive during any modal dialog) — so `tools/generate_screenshots.py`'s `arm_modal_capture()` arms a repeating 100 ms `QTimer` that polls `QApplication.activeModalWidget()`, grabs+saves it, and clicks its button to unblock the caller. It must be armed *before* triggering the action that raises the dialog, never after. For a mid-analysis-run screenshot, gate the capture on the real `metric_started` signal count (not a wall-clock `QTimer.singleShot` guess) so timing stays correct regardless of machine speed. |
| `MainWindow._on_roi_selected` only stores ROI state — it never draws the overlay | Unlike the line overlay (`ZoomableImageLabel.set_line_normalised()`, a public method `_on_line_selected` calls directly), there is no equivalent public setter for the ROI box. The mouse-driven `mouseReleaseEvent` writes straight to `ZoomableImageLabel`'s private `_roi_norm` attribute and calls `.update()`. Any code that needs to draw an ROI programmatically (e.g. `tools/generate_screenshots.py`) must poke `panel._img_label._roi_norm = (x0n, y0n, x1n, y1n)` + `.update()` itself, in addition to calling `_on_roi_selected(x0, y0, x1, y1)` (pixel coords, not normalised) to keep `MainWindow`/`AnalysisControlPanel` state consistent. |
| Removing a feature doesn't auto-sync README.md/QuickStart.md | Ghost detection (removed 2026-05-22) and PDF export (removed 2026-06-01) both stayed documented as live, working features in README.md and QuickStart.md for nearly two months after removal — QuickStart's own Troubleshooting section told users to `pip install weasyprint` to fix a feature that no longer existed anywhere in the codebase. Root cause: the removal commits didn't touch either doc, and nothing else prompts a check. QuickStart.md is opened directly by the app's own **Help → Quick Start Guide** menu item, so this isn't just GitHub-browsing staleness — it's live in-app UX. When removing or fundamentally changing a user-facing feature, grep README.md and QuickStart.md for it as part of the same change, not as a separate later cleanup pass. |
| `_compute_std_map` used `generic_filter(np.std)` long after entropy was migrated off the same pattern | `SpatialDetailAnalyzer._compute_std_map` (`analysis/image_filters.py`) computed local σ via a per-pixel `generic_filter(data, np.std, size=kernel_size)` callback — the exact pattern `_compute_entropy_map`'s own docstring documents as "~10-50x slower" than a vectorized `uniform_filter` approach (entropy was migrated off it; std was not). It ran up to 6x sequentially (3 `STD_KERNEL_SIZES` × A/B) inside `_std_analysis`, one of 5 families `SpatialDetailAnalyzer.analyze()` runs concurrently via its own always-on `ThreadPoolExecutor(max_workers=5)` — the likely cause of a user report where running "Spatial Detail" alone took 10+ minutes and never completed instead of the usual ~2. (The user's own hypothesis — a serial-vs-parallel dispatch bug in `gui/analysis_thread.py` — did not hold up: see the next pitfall row.) Fixed by computing variance as `mean_sq - mean**2` from two `uniform_filter` passes (float64 accumulation to avoid cancellation error, `mode="reflect"`, cast to float32 on return), mirroring the entropy precedent exactly. When adding any new windowed per-pixel statistic, check `_compute_entropy_map`'s docstring first — `generic_filter` with a Python callable is a documented anti-pattern in this codebase, not a reasonable first draft. |
| "Run metrics in parallel" checkbox is a no-op whenever only one metric is selected | `gui/analysis_thread.py`'s dispatch gate is `if parallel and len(tasks) > 1: self._run_parallel(...) else: self._run_serial(...)`. With a single metric checked, `len(tasks) == 1`, so this is always `False` regardless of the checkbox — both settings take the identical `_run_serial()` path, calling that one metric's closure directly with no executor at all. A slow/hung single-metric run is therefore never explained by the parallel checkbox; look inside that analyzer's own internal concurrency instead (e.g. `SpatialDetailAnalyzer`'s always-on 5-way `ThreadPoolExecutor`, unrelated to this outer setting). Separately, the background/RMS pre-pass at `gui/analysis_thread.py:115-127` ("Pre-compute background once per distinct image object") also runs unconditionally regardless of `parallel` or which metrics are selected — it cannot explain a serial-only slowdown either, though it is real, currently-unfiltered waste when a selected metric doesn't touch every image object. `control_panel.py`'s `_parallel_cb` now defaults to checked (`setChecked(True)`) — parallel mode has no known downside besides RAM, and multi-metric runs benefit from it by default. |
| MTF frequency axis mislabeled by `EPSF_OVERSAMPLING²` | `PSFAnalyzer._compute_mtf` (`analysis/psf_analyzer.py`) builds the ePSF at `EPSF_OVERSAMPLING`× finer sampling than native pixels, so the FFT's own Nyquist bin (`r = max_r = n/2`) actually corresponds to `0.5 * EPSF_OVERSAMPLING` cycles/native-px — oversampling lets you resolve frequencies *beyond* the native Nyquist, so native Nyquist (0.5 cyc/px) sits at the *midpoint* of the array, not its edge. The code instead set `freq_max = 0.5 / EPSF_OVERSAMPLING` (dividing, not multiplying), compressing the whole axis by `EPSF_OVERSAMPLING²` (4× at the default oversampling=2): the Section 4 MTF plot silently stalled at 0.25 cyc/px instead of reaching 0.5, `mtf50_cycles_per_px` was under-reported by ~4×, and `mtf_nyquist = np.interp(0.5, freq, mtf)` (`psf_analyzer.py:142`) clamped to the array's stale edge instead of interpolating at true Nyquist, since `freq` never actually reached 0.5. The bug passed every existing test because `_compute_mtf`'s output stayed monotonic and bounded [0,1] — it just meant something different than its own axis label claimed. Confirmed and fixed by feeding a synthetic ePSF containing a pure cosine at a *known* frequency through the real function: the peak was mislabeled ~0.10 cyc/px for a true 0.4 cyc/px signal before the fix, ~0.41 after. Fixed with `freq_max = 0.5 * EPSF_OVERSAMPLING`. When adding any new oversampled-grid frequency axis, verify calibration with a known-frequency synthetic test signal rather than trusting the scaling formula by inspection. |
| Asserting on `fig.text()`/`ax.set_title()` content by searching the rendered report HTML | `_localmax_family_distributions_figure`'s (Section 8l) column headers ("Magnitude (Image A vs B)" / "Log ratio (A / B)") are drawn with `fig.text(...)` directly onto the matplotlib `Figure`, which is then embedded as a base64 PNG — the text exists only as *pixels* inside the image, never as a string in the HTML document. A test asserting `"Log ratio (A / B)" in section_html` fails every time, regardless of whether the figure is correct, because that substring can never appear in the HTML — only in the undecoded PNG bytes. Caught immediately by running the new test rather than assuming it would pass. Any assertion about matplotlib-drawn text (titles, axis labels, legends, `fig.text`/`fig.suptitle`) must instead check adjacent *literal* HTML the report builder also emits — a `<p class="caption">` string, an `alt="..."` tag, a heading — never the rendered figure's own visual content. |
| Reconstructing a pixel-level mask from an algorithm that only exposes cell-level statistics | `photutils.background.Background2D` sigma-clips raw pixel values *inside* each mesh cell before computing that cell's background/RMS, but never exposes which individual pixels survived the clip — only the resulting per-cell scalars (`background_mesh`/`background_rms_mesh`). Confirmed directly against the photutils API reference before writing any code: no `mask`-shaped output attribute exists. Section 3e's pixel-classification histogram/overlay needed that mask, so `analysis/background_mask.py::classify_background_pixels` re-runs `astropy.stats.SigmaClip` with the *same* sigma/maxiters independently, purely to recover it — which only stays correct if those two parameters are a single named source of truth (`BACKGROUND_SIGCLIP_SIGMA`/`BACKGROUND_SIGCLIP_MAXITERS` in `core/astro_image.py`, used by both `estimate_background()` and `classify_background_pixels()`), not two independently-hardcoded `3.0`/`10` literals that could silently drift apart. General pattern: before writing a second pass to recover "what did this library step actually do internally," check its public API surface first (don't assume it's exposed, and don't assume it isn't) — and if it must be recomputed, name-share every parameter the original call used. |
| A closed-form "range across the frame" derived from a fitted plane needs corner evaluation, not `magnitude × diagonal_length` | `analysis/background_fit.py`'s gradient term reports the ADU swing across the image from its fitted plane `b·dx + c·dy`. The true corner-to-corner range of a linear function over a rectangle is `max(corner values) − min(corner values)`, which reduces to `abs(b)·w + abs(c)·h` — **not** `sqrt(b²+c²) · sqrt(w²+h²)` (magnitude times diagonal length), which overstates the true range by Cauchy-Schwarz whenever the gradient direction isn't aligned with the diagonal. Caught before it shipped by a dedicated test (`test_exact_adu_range_is_corner_based_not_diagonal`) constructing a deliberately non-diagonal gradient and asserting the two formulas differ meaningfully for it — a test using a diagonal-aligned gradient would have passed either (wrong) formula, since they only diverge off-diagonal. When deriving any "extreme value of a fitted function over a bounded region" quantity, evaluate at the actual boundary rather than reaching for a magnitude-times-extent shortcut, and specifically test a case where the shortcut and the correct answer diverge. |
| A quadratic form's cross-term coefficient is `2×` the matrix's off-diagonal entry | `analysis/background_fit.py`'s anisotropic curvature term decomposes `d·x² + e·y² + f·xy` via the form's symmetric matrix `[[d, f/2], [f/2, e]]` — the `xy` polynomial coefficient `f` is twice the matrix entry `f/2`, because both `dx·dy` and `dy·dx` contribute to the quadratic form. The traceless-matrix eigenvalue magnitude is `sqrt(((d-e)/2)² + (f/2)²)`, not `sqrt(((d-e)/2)² + f²)` — using raw `f` overstates the anisotropic magnitude by an amount that depends on the ratio of the two terms (exactly 2× in the pure-`f` case). Caught by a dedicated test (`test_anisotropic_magnitude_uses_f_over_2_not_raw_f`) constructing a synthetic case with `d == e` so the anisotropic term is driven purely by `f`, making the correct answer exactly `abs(f)/2` and an `abs(f)`-using bug exactly 2× wrong — an easy, unambiguous numeric distinction a mixed-term synthetic case wouldn't give as cleanly. `atan2`-based *direction* formulas are scale-invariant, so this class of factor-of-2 bug only ever affects a *magnitude*, never an angle derived via `atan2` on the same two quantities — don't assume a magnitude bug implies a direction bug too. |
| A plane-only robust scaffold on a vignetted frame | It cannot represent a bowl, so the whole vignette stays in the residual and the detector masks it as source: **53% of a frame with no nebulosity in it**, and an estimate worse than doing nothing (−0.267 σ against a +0.023 σ unmasked baseline). Vignetting is near-universal in real data, so this is a default-path bug, not an edge case. Reject with a plane, refit survivors with BIC free to add curvature. |
| A detection threshold set only by statistical significance | It runs away as the test integrates over area — a σ=16 px kernel drops the noise ~57×, so a 2σ cut fired on structure at 3.5% of sky noise and masked 62% of a frame whose visible nebulosity was 12.9%. Add a physical floor (`max(statistical, k × sky_sigma)`) and record which bound won. |
| `roi_mask(shape, None)` returns all-**True**, `exclusion_mask(shape, [])` all-**False** | Opposite conventions for the same-shaped data, and both are correct: "no ROI" means the whole frame is the domain of interest, "no exclusions" means exclude nothing. Reusing `roi_mask`'s None-guard for an exclusion list would mask every pixel of every image in the *default* case. A parametrized test pins `None`/`[]`/`()`/`[None]`. |
| A class-scoped fixture that mutates shared state, plus a sibling test asserting the unmutated case | A generator fixture's teardown runs at the *end of the class*, so any other test in that class sees the mutation. `test_absent_when_no_regions_drawn` failed for exactly this reason against the shared `image_pair`. A test asserting the absence of something must set that state itself in a `try/finally` rather than rely on fixture ordering. |
| A model-order gate expressed as an absolute cell count | It means a different thing at every image scale — 35 cells is 54.7 % of a 64-cell test mesh and 1.2 % of a 2925-cell real one, so the guard is absent exactly where it matters. Gate on `max(floor, fraction × n_total)`. |
| Judging a plane-vs-quadric disagreement by its size alone | A vignetted frame's plane genuinely cannot fit, so the two surfaces differ everywhere (measured 5.32 σ) and the quadric is correct. Only the *ratio* of whole-frame to at-the-cells divergence separates real curvature from extrapolation. Require both bounds. |
| Assuming a mask attached to an `AstroImage` reaches its background | `estimate_background()` is idempotent and guards on `background_model`, so anything attached afterwards is silently ignored. Use `set_background_exclusion_mask()`, which invalidates, and attach **before** `AnalysisThread`'s pre-pass. A class-scoped test fixture that rebuilds a shared image's background must rebuild it back in teardown — clearing `bg_exclusion_regions` alone no longer undoes it. |
| Treating "the summary number went down" as a regression after changing what it selects over | `snr_global` is `median(pixels > 3σ)/σ`. Fixing the background so it stops absorbing the nebula moved that population from 1.5 % to 11.4 % of the frame, dropping the statistic 30.3 → 3.7 while recovered flux rose 46 % → 84 % of truth. Check the quantity the statistic proxies for, and state the discontinuity in the report or the user will read it as breakage. |
| `core/astro_image.py` importing `analysis.inspector_regions` | Closes a cycle through `analysis.image_filters`, which imports `core.astro_image`. `analysis.source_mask` *is* safe (it reaches only `analysis.background_fit` + numpy/scipy/photutils). Rasterise region dicts in the caller and hand `AstroImage` the finished bool array. |
| A symmetric `SigmaClip` used against single-signed contamination | It cannot remove a bias carried by >50% of samples, and its scale estimate is inflated by the contamination itself. Reject one side only, and take the scale from the lowest quartile — see the "Single-signed contamination" convention above for the two intermediate estimators that also failed. |
| Assuming a detect → mask → re-estimate loop converges | It diverges: the mask can only grow, so coverage ran away 54% → 84% → 95% and bias went +0.10 σ → −2.17 σ → −2.84 σ. `SOURCEMASK_N_PASSES = 1`. Verify convergence with ground truth before raising any iteration count. |
| Treating `Background2D.background_mesh` values as measurements | Cells excluded by `exclude_percentile` are silently back-filled by interpolation, and `n_pixels_mesh` cannot distinguish them. Fitting the fills gave 55.5 ADU mesh error vs 5.1 for real cells, and halved a recovered gradient while the median bias still looked fine. Set `exclude_percentile` permissive and select cells yourself via `_cell_unmasked_fraction`. |
| Detecting extended structure without first removing stars | A 30 000 ADU star convolved at σ=16 px still stands ~640× above that tier's threshold and becomes a 54 px-radius "extended" blob; 40 stars masked 63% of a *star-only* frame. Excise the dilated point mask from the residual (fill with 0 — it is pedestal-subtracted) before the broad tiers convolve. |
| Thresholding a smoothed image with the per-pixel sigma | Noise drops to `σ/sqrt(4π σ_k²)` after a sum-normalised Gaussian — a factor of ~28 at σ_k = 8 px. Use `_smoothed_sigma`; the raw σ detects essentially nothing and looks like a broken detector rather than a wrong constant. |
| Judging a fitted 2D surface by median error alone | A fit through edge-clustered cells can sit on the truth at centre and diverge at the corners: measured +0.09 σ median while carrying 3.9 σ worst-case. Assert `median`, `rms` **and** `max(\|err\|)`, and compare a plane's recovered *tilt* (lstsq through the output surface) rather than the fit's own linear coefficients, which are the centre tangent when BIC picks the quadric. |
| A per-source full-image `exp()` in a test frame generator | O(n_stars × H × W) is invisible at 512×512 and takes >10 min at 24 MP with 600 stars — and reads as a hang in the code under test. Stamp into a `4σ` bounded slice (`tests/bg_frames.py`). |
| Dropping sparse tiles from a **count-weighted** tile reduction | It breaks the identity that the weighted tile mean equals the plain masked mean, and the symptom is a confidence interval that does not contain its own point estimate — which reads as a broken table. `tile_reduce`'s original 16-pixel floor discarded 568 of 900 tiles on a real Section 8l mask (7.6% coverage), and because the discarded tiles were the mask-sparse ones rather than a random subset it biased the estimate **2×**. The sparse-cell concern that motivates a floor applies to *unweighted* statistics; here a 3-pixel tile simply carries weight 3. Both bounds are now 0/1, and `test_point_estimate_equals_the_masked_mean_on_a_concentrated_mask` pins it using a mask shaped like the real one. Guarding against under-determined input belongs in the bootstrap (which needs ≥ 2 tiles), not in the reduction. |
| Two reducers over the same domain that accept different input types | `consensus_label` returns a plain label string while `overall_label` read `.rank`/`.label` off its inputs — so following `consensus_label`'s own docstring ("feed this the per-axis consensus labels") raised `AttributeError`. It shipped undetected because the only caller at the time reduced labels by hand. When two functions are documented as composing, make the second accept what the first returns, and add the composition itself as a test (`test_two_level_reduction_composes`). |
| Enumerating Section 8 metrics from the `panels` dict | It silently omits Section 8m. `lap_var_*` / `grad_energy_*` are **scalars** in the result dict, not panels, so a `METRIC_KEYS` list built by iterating `panels` covered 20 metrics and missed the six acutance ones entirely — through an entire calibration study, and acutance turned out to be among the most sensitive metrics in the suite. Enumerate from what the *report* reports, not from one container. |
| Selecting a bootstrap block size from the correlation length | Fails on long-range-dependent fields: integral length ~7 px while the SE still grows at 224 px blocks. Use `se_ladder`, take the rung past the **last** violation, and treat `converged=False` as a reportable result. |
| Sorting calibration points by ratio before fitting the curve | It *manufactures* monotonicity out of a non-monotone response and lets exactly the broken metrics through the screen. Judge monotonicity in **blur order**, and reject on wrong sign or on a response range too flat to invert (`gradient_3.0` spanned 0.0085 log units across a 0–141% FWHM range, so a noise-sized reading mapped to "material"). |
| A fraction-of-tile-area rule in `tile_reduce` | Section 8l's local-maxima mask selects ~5–12% of the frame, and a flat 25%-of-area requirement rejects essentially every tile — silently disabling the bootstrap for the population that best tracks perceived sharpness. Scale the requirement by the mask's own density, with an absolute pixel floor underneath. |
| Assuming star removal parallelises | It is GPU-bound via DirectML: 2 concurrent Axiom calls give 1.17×, **4 give 0.63× — slower than sequential**. The lever is doing less work, not more at once. Block-level removal (star-remove each block once, assemble pairs from already-starless blocks) cut a 14-set run from 4.4 h to 52 min; `--jobs` only became worthwhile *after* that flipped the balance toward the CPU-bound analysis half. |
| A `str.replace` patch script matching identical boilerplate in two functions | `cmd_null_ensemble` and `cmd_signal_grid` open with byte-identical `rng = ...` / `rows = []` / `mask_metrics = ...` blocks, so a single-target patch spliced itself into both and the second definition shadowed the first (`NameError: SNRAnalyzer`). When patching by literal text in this file, assert the match count or use the Edit tool, which fails loudly on ambiguity. |
| `ax.hist(..., density=True)` on percentile-clipped bins can silently divide by zero when a population's whole range falls outside those bins | Section 3e's background-vs-excluded pixel histogram originally computed bin range from the *pooled* (kept + excluded) data's 0.5–99.5th percentile. The excluded population is by construction the outlier tail (e.g. >3σ pixels), so on realistic data its values can fall almost entirely *outside* the pooled 99.5th-percentile cutoff — `numpy`'s `density=True` normalization then divides by a zero bin-count sum for that population, throwing `RuntimeWarning: invalid value encountered in divide` silently (no crash, no visible symptom besides the warning — the histogram bar for that population just doesn't render). A `size > 0` guard on the input array does **not** catch this, since the array is non-empty, just entirely out of the chosen bin range. Fixed by computing the bin range from *each* population's own percentiles (kept ∪ excluded, unioned), not the pooled percentile — so a minority outlier population's own bulk always lands inside the shared range. When a figure shows two populations with very different scales/tails on shared bins, derive the bin range from each population's own spread, not from percentiles of the combination. |

---

## Documentation Screenshots — Key Patterns

`tools/generate_screenshots.py` regenerates every screenshot embedded in README.md and
QuickStart.md (`resources/*.png`) from synthetic data — no real FITS files or manual
GUI interaction required. Run it (`python tools/generate_screenshots.py`, needs a real
interactive desktop session so dark-mode chrome matches the OS theme) whenever a GUI
change makes existing screenshots stale, rather than leaving them to drift the way the
pre-toolbar screenshots did.

### Sample data

Two full-resolution (1920×1080) images from `SyntheticGenerator`, camera `"Player One —
Mercury-M"` (the smallest full camera — see the Testing section below), sharing the same
`n_stars` so the two-RNG convention gives matching star positions while differing
`fwhm_arcsec`/`halo`/`moffat_beta` make Image A vs B visibly distinguishable. Reusing the
generator (rather than real data) keeps the script fully reproducible with zero external
dependencies.

### Capture technique

`QWidget.grab()` from a normal (non-`offscreen`) `QApplication` — construct each
widget/dialog directly, no user interaction, no visible window required (see the
"OS-level screenshots... unreliable" pitfall above for why this is the right primitive).
Modal dialogs and mid-run states need the two techniques captured as their own pitfall
rows above (`arm_modal_capture()`'s pre-armed `QTimer`; `metric_started`-signal-gated
mid-run capture) — both live in the script as reusable helpers, not one-off hacks.

### Manifest

15 files total: 8 full/panel/group-box states of the main window (empty, both images
loaded, line drawn, ROI drawn, Parameters group, Metrics+Region&Run composite via
`grab_side_by_side()`), the manual starless prompt, a mid-run and a completion-dialog
capture from one real six-metric analysis run, the Report Inspector that run produces,
and the three Tools-menu dialogs (Synthetic Data Generator, Spatial Target Generator,
Halo Analyzer — the latter with a star pre-clicked via `dlg._on_star_clicked(x, y)` so
the results table and PSF/RDF charts are populated rather than blank).

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
`SYN_SSED` (star seed = n_stars), `SYN_NSED` (noise seed),
`SYN_SGRD` / `SYN_SGAN` (sky gradient fraction / direction — note `SYN_SKYA` was
already taken by the sky ADU level).

### Sky gradient — applied to the Poisson expectation, not added afterwards

`sky_gradient` (0–0.5, total corner-to-corner swing as a fraction of the base sky) and
`sky_gradient_angle_deg` (0–360, CCW from +x) exist so gradient+nebula frames can be produced
for the Section 3g diagnostic. `_sky_gradient_map()` turns the scalar `sky_e` into an (h, w)
expectation array, which `noise_rng.poisson(lam_array)` accepts natively — so shot noise scales
as √(local sky) exactly as a real light-pollution gradient does. Adding a ramp *after* the
Poisson draw would leave a flat noise level under a sloping sky, which is unphysical and would
quietly invalidate the very thing Section 3g is being tested on.

The ramp is deterministic, so it is drawn from neither RNG and the two-RNG convention is
untouched. The projection is renormalised by its own span so the requested swing means the same
thing at any angle (a diagonal spans √2 more than an axis-aligned one), and the frame mean stays
at the base sky level. `sky_gradient = 0` returns a genuinely uniform array, so the default
reproduces the pre-gradient behaviour exactly.

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

## Data Inspector — Key Patterns

Four modules: `gui/data_inspector.py` (window), `gui/inspector_widgets.py` (pyqtgraph
widgets), `analysis/inspector_regions.py` (all array maths), `core/inspector_catalog.py`
(`.npz` reader). Opened from **File → Open Data Inspector…**; the Report Inspector is
unchanged and still auto-opens after a run. Both are kept until this one is proven.

### Layout

Three splitter rows: `Image A | Image B | Comparison` on top, `cross-section |
comparison histogram` in the middle, `correlation in-mask | correlation out-of-mask`
at the bottom. Three control rows above: data selection (Section → Image set → A/B →
Compare), view/tool/reset + a hint line, and the region controls.

### Everything positional is normalised `[0,1]`

The cross-section line, the ROI, the swipe divider and the view rectangle are all
stored as fractions, so they rescale onto a panel of any size and can never fall out
of bounds — the same convention `gui/image_panel.py` uses. That is why no
"selection no longer fits, reset it?" prompt is needed anywhere.

### Comparison is log-ratio or difference — never a linear ratio

`Compare:` offers `log10(|A|/|B|)` (default) and `A − B`, both symmetric about zero.
"Ratio" always means the log₁₀ ratio, following Section 8's convention; a linear
ratio is neither symmetric about its no-change point nor safe on a diverging
colormap. `COMPARE_MODES` has no `"ratio"` entry and a test asserts it never gains one.

### Region model — ROI is the domain, threshold is the split

| ROI | Threshold | Correlation plot 1 | Correlation plot 2 |
| --- | --- | --- | --- |
| — | — | all pixels | empty |
| set | — | inside ROI | outside ROI |
| — | set | mask | ¬mask |
| set | set | ROI ∧ mask | ROI ∧ ¬mask |

The threshold applies to A and B **independently and is then AND-ed**, so a "top 10%"
setting yields well under 10% of the frame (see the pitfall row). Percentile mode is
scale-free across metric families; absolute mode re-seeds its spin box from the
current panel's own range when you switch to it.

### Stable identity: flat pixel index

`correlation_sample` returns `flat_ids` (`row * W + col` in the common-cropped frame)
alongside the plotted values. Selections are carried as those ids — never as
positions within the subsampled arrays, which change on every resample. `ids_to_mask`
turns a selection back into the cyan pixel overlay, and both correlation plots
highlight from the same id set. A panel change drops the selection, because the ids
index a frame that may have resized.

### Threading

`_RegionThread` computes masks, both correlation samples and the histogram off the
GUI thread. Three mechanisms together, following `gui/halo_dialog.py`'s convention:
a **120 ms debounce** (an ROI drag emits a continuous stream), a **monotonic token**
echoed back in the result so a slow run cannot overwrite newer state, and
**supersede** — disconnect and `quit()` a running thread, never `wait()` on the GUI
thread. Cheap work (cross-section sampling, LUT swaps, selection→overlay) stays
inline. The worker catches everything and emits `{"error": …}` so a bad request
surfaces in the warning label rather than taking the window down.

### Compute functions return data; the presentation layer renders it

`_plot_psf_simulation` used to return an `RdBu_r`-colormapped RGB `diff` plus uint8
panels — pictures, not data. Storing uint8 quantises to 1/255, and a typical A−B
range on that chart is a few percent, so the stored panels could not support the
comparison they existed for (measured on real data: the whole A−B range spanned
under 3 quantisation steps). It now returns float32 values, and the colormapping
plus uint8 cast live in `_psf_simulation_html` where the `<img>` is built. Apply the
same split to any new figure helper whose output something else might want to measure.

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
