"""Render the local HTML artifact for the synthetic spatial-detail
screening sweep (tools/spatial_detail_screen.py).

Filmstrips + response-curve graphs, matching tools/metric_atlas.py's shape:
this module reuses that file's build_response_curves / build_scorecard /
_png_b64 / _CSS unmodified. metric_atlas.py already does the equivalent
cross-import from tools/sensitivity_sweep.py, so this is an established
pattern in this codebase, not a new one.

This page is diagnostic-only. It is entirely separate from
resources/metric_calibration.json and core/practical_significance.py -- not
wired into either -- and it carries no FWHM/practical-significance labels
anywhere, because every target is deliberately star-free (no PSF to fit).
See tools/spatial_detail_screen.py and synthetic/detail_targets.py for the
sweep design and why noise is swept as an axis independent of blur.

Usage
-----
    python tools/spatial_detail_screen.py -o sweep_out/synthetic_detail
    python tools/spatial_detail_atlas.py --csv sweep_out/synthetic_detail/spatial_detail_screen.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from PIL import Image                                                  # noqa: E402

from analysis.image_filters import SpatialDetailAnalyzer               # noqa: E402
from core.fig_utils import fig_to_b64                                  # noqa: E402
from core.stretch import normalize_for_display                         # noqa: E402
from synthetic.detail_targets import apply_camera_noise, apply_gaussian_blur  # noqa: E402
from tools.metric_atlas import _CSS, _png_b64, build_response_curves  # noqa: E402
from tools.sensitivity_sweep import (                                  # noqa: E402
    METRIC_KEYS, POP_BULK, POP_TOP, _SCALE_OF, _localmax_mask, _mem_image,
)
from tools.spatial_detail_screen import ALL_TARGETS, POP_NEBULA, build_target  # noqa: E402


# Per-target caveats discovered while running the actual sweep (not
# predicted in advance) -- surfaced here rather than silently left for the
# reader to notice from an odd-looking curve.
TARGET_CAVEATS = {
    "zebra": (
        "This target's nebula-mask coverage reached ~98% of the frame at "
        "some grid point during the sweep, right at the "
        "SOURCEMASK_MAX_COVERAGE guard -- a stark, full-bleed stripe "
        "pattern leaves the source mask almost no genuine flat-sky region "
        "to anchor a background estimate on. Read this target's "
        "nebula-mask-based numbers (Section 8k lap_var_*/grad_energy_*, "
        "and any nrm_* row whose noise floor comes from a nearly-empty "
        "background mask) with that in mind."),
}


def build_filmstrip_row(panels: list[tuple[str, np.ndarray]]) -> str:
    """One row of same-canvas panels, each with its own caption.

    Unlike tools/metric_atlas.py's build_filmstrip, this carries no
    FWHM/practical-significance chip -- every target here is deliberately
    star-free, so there is no PSF to fit and no calibrated label to attach.
    """
    cells = []
    for label, img in panels:
        cells.append(
            f'<div class="cell"><img src="data:image/png;base64,{_png_b64(img)}" '
            f'width="{img.shape[1]}" height="{img.shape[0]}" alt="{label}">'
            f'<div><b>{label}</b></div></div>')
    return f'<div class="strip">{"".join(cells)}</div>'


# One representative scale per family -- 7 of the 20 METRIC_KEYS -- so the
# overlay filmstrip below stays to a manageable image count while still
# covering every family.
OVERLAY_FAMILY_KEYS = ["std_3px", "log_1.5", "gradient_1.5", "wavelet_2",
                       "entropy_5px", "localgrad_1.5", "loclap_1.5"]
OVERLAY_DEPTHS = [1, 25]           # noisiest realistic vs. well-stacked
OVERLAY_PANEL_PX = 320             # downscaled from the 640px canvas -- the
                                    # mask pattern stays legible, file size stays bounded
OVERLAY_COLOR = np.array([255.0, 0.0, 255.0])   # magenta
OVERLAY_ALPHA = 0.40                             # "light" overlay -- structure
                                                   # underneath stays visible


def _pick_illustrative_sigmas(blur_ladder: list[float]) -> list[float]:
    """Up to 3 points from whatever blur ladder this target actually used:
    no blur, a middle value, the max -- read back from the CSV, not a
    hardcoded list, so this always matches the grid that actually ran.

    Indexes into the deduplicated, sorted candidate list itself (rather than
    picking `nonzero[len(nonzero)//2]` and unioning with 0 separately) so
    the three picks are guaranteed distinct whenever >=3 unique values
    exist -- the earlier version collapsed to 2 points whenever the ladder
    had exactly 2 nonzero values, since `nonzero[len//2]` and `nonzero[-1]`
    landed on the same element and the union silently deduplicated them.
    """
    candidates = sorted({0.0} | {float(s) for s in blur_ladder if s >= 0})
    if len(candidates) <= 3:
        return candidates
    mid = candidates[len(candidates) // 2]
    return sorted({candidates[0], mid, candidates[-1]})


def _overlay_img_b64(magnitude_map: np.ndarray, mask: np.ndarray,
                     size_px: int = OVERLAY_PANEL_PX) -> str:
    """Grayscale STF-stretched magnitude map (same stretch _png_b64 uses)
    with a translucent magenta tint where mask is True, downscaled to
    size_px after compositing (not before -- compositing at full resolution
    keeps the mask's true shape, downscaling afterward only controls file
    size).

    Saved as JPEG, not PNG like every other image on this page: the
    noisiest OVERLAY_DEPTHS panels are dominated by per-pixel shot-noise
    texture, which lossless PNG compresses very poorly (measured ~145KB per
    320px panel -- 420 panels would add ~60MB to the file). These panels
    exist for spotting gross mask-shape shifts by eye, not per-pixel
    fidelity, so JPEG's lossy compression (which is efficient on exactly
    this kind of noisy/photographic content) is the right tradeoff here,
    not a corner cut everywhere -- every other image on this page (filmstrips,
    response/value-curve figures) stays PNG via the existing _png_b64.
    """
    gray = normalize_for_display(magnitude_map)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float64)
    m = mask[:rgb.shape[0], :rgb.shape[1]]
    rgb[m] = (1.0 - OVERLAY_ALPHA) * rgb[m] + OVERLAY_ALPHA * OVERLAY_COLOR
    img = Image.fromarray(rgb.astype(np.uint8))
    if img.width != size_px or img.height != size_px:
        img = img.resize((size_px, size_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_metric_overlay_section(target_id: str, blur: list[float]) -> str:
    """For each representative metric family, one filmstrip row: image B's
    own magnitude map at a few illustrative blur sigmas x two stack depths,
    each with the mask that actually backs this page's 'top5' curves
    overlaid in magenta -- context for what's driving a top5 response
    curve's value at that point, and (the reason this exists) a way to see
    by eye whether the masked pixel set is shifting non-smoothly as blur
    increases, which is the leading suspect for this page's response-curve
    discontinuities: the top5 population is a discrete pixel set
    (_combined_localmax_mask = local-maxima peaks unioned with the
    top-N%-brightness mask, not the brightness-only mask 8d-8j's report
    figures use), and which pixels qualify can jump even when the
    underlying image is changing smoothly.

    One analyze() call per (depth, sigma) pair gives every family's panels
    at once, so this is len(OVERLAY_DEPTHS) x 3 calls per target, not
    x len(OVERLAY_FAMILY_KEYS) -- reuses exactly the A/B pairing
    tools/spatial_detail_screen.py::run_sweep uses.
    """
    sigmas = _pick_illustrative_sigmas(blur)
    clean = build_target(target_id)
    rng = np.random.default_rng(1)

    by_family: dict[str, list[str]] = {k: [] for k in OVERLAY_FAMILY_KEYS}
    for depth in OVERLAY_DEPTHS:
        image_a = _mem_image(apply_camera_noise(clean, rng, depth), "A")
        for sigma in sigmas:
            blurred = clean if sigma == 0 else apply_gaussian_blur(clean, sigma)
            image_b = _mem_image(apply_camera_noise(blurred, rng, depth), "B")
            res = SpatialDetailAnalyzer().analyze(image_a, image_b, make_figures=False)
            panels = res.get("panels", {})
            for key in OVERLAY_FAMILY_KEYS:
                panel = panels.get(key)
                if panel is None or panel.get("a") is None:
                    continue
                scale = _SCALE_OF.get(key)
                mask = _localmax_mask(np.abs(panel["a"]), np.abs(panel["b"]), scale)
                b64 = _overlay_img_b64(panel["b"], mask)
                label = f"{key} &middot; stack={depth} &middot; &sigma;={sigma}px"
                by_family[key].append(
                    f'<div class="cell"><img src="data:image/jpeg;base64,{b64}" '
                    f'width="{OVERLAY_PANEL_PX}" height="{OVERLAY_PANEL_PX}" '
                    f'alt="{label}"><div>{label}</div></div>')

    rows_html = [f'<div class="strip">{"".join(cells)}</div>'
                for cells in by_family.values() if cells]
    return "".join(rows_html)


def build_value_curves(rows: list[dict]) -> str:
    """Per-metric raw value (image B's own mean magnitude) vs blur sigma,
    one line per noise level -- the within-B trend the log-ratio panel
    abstracts away. No axhline(0) (not meaningful for a raw value) and no
    forced pass through the origin; y-axis units are that metric's own,
    not comparable across families or against the log-ratio panel."""
    by_metric: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        if not r.get("value_b"):
            continue
        try:
            sigma = float(r["factor_sharpness"].split("=")[1])
            noise = r["factor_snr"]
        except (IndexError, ValueError, KeyError):
            continue
        by_metric[r["metric_key"]][noise].append((sigma, float(r["value_b"])))

    if not by_metric:
        return "<p>No rows with a raw value in this population.</p>"

    keys = sorted(by_metric)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        for noise, pts in sorted(by_metric[key].items()):
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", ms=3, lw=1.2, label=noise)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("blur σ (px)", fontsize=8)
        ax.set_ylabel("raw mean magnitude (B)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, title="added noise", title_fontsize=6)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.tight_layout()
    return (f'<img src="data:image/png;base64,{fig_to_b64(fig, dpi=110)}" '
           f'alt="raw metric value trend">')


# The reference noise level for monotone/sign screening. tools/metric_atlas.py's
# build_scorecard hardcodes "noise=0.0" (a literal zero-added-noise case,
# valid for its own real-data blur-grid sweep where the base image still
# carries its own real sensor noise). This study's sweep no longer generates
# a perfectly noiseless case at all -- it was found to behave atypically and
# doesn't represent a real astro image -- so a real, always-present stack
# depth is used instead: the noisiest realistic single-sub case, the most
# demanding condition a metric could be screened against.
SCORECARD_SNR_LABEL = "stack=1"


def _metric_pass_fail(rows: list[dict], snr_label: str = SCORECARD_SNR_LABEL) -> dict[str, bool]:
    """Same monotone / correct-sign screen as tools/metric_atlas.py's
    build_scorecard (range is intentionally not required here -- see
    build_consensus_table), returned as a plain {metric_key: passed} dict
    rather than rendered HTML.

    This exists because a genuine cross-target consensus is an AND across
    several *single-target* scorecards, not one build_scorecard call over
    every target's rows pooled together: different targets share blur-sigma
    values but have unrelated response magnitudes, so sorting the pooled
    rows by sigma and checking monotonicity there is invalid -- it interleaves
    unrelated curves and (confirmed by running it) fails essentially
    everything, which is a methodology artifact, not a finding.
    """
    by: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if not r.get("log_ratio") or r.get("factor_snr") != snr_label:
            continue
        try:
            sigma = float(r["factor_sharpness"].split("=")[1])
        except (IndexError, ValueError):
            continue
        by.setdefault(r["metric_key"], []).append((sigma, float(r["log_ratio"])))
    out: dict[str, bool] = {}
    for key, pts in by.items():
        vals = [v for _, v in sorted(pts)]
        diffs = np.diff(vals)
        monotone = bool(np.all(diffs >= -1e-12))
        sign_ok = all(v >= -1e-9 for v in vals)
        out[key] = monotone and sign_ok
    return out


def build_scorecard_at(rows: list[dict], snr_label: str = SCORECARD_SNR_LABEL) -> str:
    """Monotone / sign / dynamic-range screen -> who may carry a label, at
    one specific noise level.

    A local reimplementation of tools/metric_atlas.py's build_scorecard,
    not a call to it: that function hardcodes "noise=0.0" as the screening
    level, which no longer exists in this study's data (see
    SCORECARD_SNR_LABEL above) -- the algorithm is otherwise identical.
    """
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if not r.get("log_ratio") or r.get("factor_snr") != snr_label:
            continue
        try:
            sigma = float(r["factor_sharpness"].split("=")[1])
        except (IndexError, ValueError):
            continue
        by[r["metric_key"]].append((sigma, float(r["log_ratio"])))

    out = ["<table><tr><th>Metric</th><th>Monotone in blur</th>"
          "<th>Sign correct</th><th>Dynamic range</th><th>Verdict</th></tr>"]
    for key in sorted(by):
        vals = [v for _, v in sorted(by[key])]
        diffs = np.diff(vals)
        monotone = bool(np.all(diffs >= -1e-12))
        sign_ok = all(v >= -1e-9 for v in vals)
        rng_span = max(vals) - min(vals) if vals else 0.0
        ok = monotone and sign_ok
        verdict = ("may carry a label" if ok
                  else '<span class="disq">disqualified</span>')
        out.append(f"<tr><td><b>{key}</b></td><td>{'yes' if monotone else 'NO'}</td>"
                  f"<td>{'yes' if sign_ok else 'NO'}</td><td>{rng_span:.3f}</td>"
                  f"<td>{verdict}</td></tr>")
    out.append("</table>")
    return "".join(out)


def build_consensus_table(per_target: dict[str, dict[str, bool]]) -> str:
    """Metric -> how many targets it passes monotone+sign on, and whether it
    passes on every one of them. A strict "every target" count is dominated
    by known-pathological targets (e.g. zebra, see TARGET_CAVEATS), so the
    fractional count is the more informative column; "every target" is kept
    alongside it as the strict reading.
    """
    all_keys = sorted({k for d in per_target.values() for k in d})
    n_targets = len(per_target)
    rows_html = []
    for key in all_keys:
        n_pass = sum(1 for d in per_target.values() if d.get(key, False))
        all_pass = n_pass == n_targets
        rows_html.append(
            f"<tr><td><b>{key}</b></td><td>{n_pass}/{n_targets}</td>"
            f"<td>{'yes' if all_pass else 'no'}</td></tr>")
    return ("<table><tr><th>Metric</th>"
            f"<th>Targets monotone + correctly signed (at {SCORECARD_SNR_LABEL})</th>"
            "<th>True on every target</th></tr>"
            + "".join(rows_html) + "</table>")


def _rows_for(rows: list[dict], population: str | None = None) -> list[dict]:
    """Primary-family (raw, non-nrm_) rows, optionally restricted to one
    population."""
    out = []
    for r in rows:
        if r["metric_key"] not in METRIC_KEYS:
            continue
        if population is not None and r["population"] != population:
            continue
        out.append(r)
    return out


def _nrm_acutance_rows(rows: list[dict], top5: bool = False) -> list[dict]:
    """nrm_* rows for one population, plus -- only in the bulk grouping --
    the 8k global-acutance scalars (population=nebula), which have no
    top-N% population concept at all (they're whole-nebula scalars, not a
    ratio against a top-N% mask) so they only ever belong with bulk."""
    out = []
    for r in rows:
        is_nrm = r["metric_key"].startswith("nrm_")
        is_8k = r["population"] == POP_NEBULA
        if top5:
            if is_nrm and r["population"] == POP_TOP:
                out.append(r)
        else:
            if (is_nrm and r["population"] == POP_BULK) or is_8k:
                out.append(r)
    return out


def _ladders(rows: list[dict], target_id: str) -> tuple[list[float], list[int | None]]:
    """Blur/stack-depth values actually present for one target, read back
    from the CSV rather than the module's own default ladders, so a
    filmstrip always matches whatever grid actually produced this CSV --
    including a --blur/--stack-depths override run."""
    t_rows = [r for r in rows if r["case_id"] == target_id]
    blur = sorted({float(r["factor_sharpness"].split("=")[1]) for r in t_rows
                  if r["factor_sharpness"]})
    depths = sorted(
        {None if r["factor_snr"] == "noise=0.0" else int(r["factor_snr"].split("=")[1])
         for r in t_rows if r["factor_snr"]},
        key=lambda d: (d is None, d))
    return blur, depths


def build_target_section(target_id: str, rows: list[dict]) -> str:
    blur, depths = _ladders(rows, target_id)
    clean = build_target(target_id)
    rng = np.random.default_rng(0)

    blur_panels = [("clean", clean)]
    for sigma in blur:
        if sigma > 0:
            blur_panels.append((f"blur σ={sigma}px", apply_gaussian_blur(clean, sigma)))

    noise_panels = [("noiseless", clean)]
    for depth in depths:
        if depth is not None:
            noise_panels.append((f"stack={depth}", apply_camera_noise(clean, rng, depth)))

    t_rows = [r for r in rows if r["case_id"] == target_id]
    primary_bulk = _rows_for(t_rows, population=POP_BULK)
    primary_top5 = _rows_for(t_rows, population=POP_TOP)
    nrm_bulk_and_acutance = _nrm_acutance_rows(t_rows, top5=False)
    nrm_top5 = _nrm_acutance_rows(t_rows, top5=True)

    caveat = TARGET_CAVEATS.get(target_id, "")
    caveat_html = f'<div class="warn"><b>Caveat.</b> {caveat}</div>' if caveat else ""

    return f"""
<h2>{target_id}</h2>
{caveat_html}

<h3>Filmstrip &mdash; blur axis (noiseless)</h3>
{build_filmstrip_row(blur_panels)}

<h3>Filmstrip &mdash; noise axis (no blur)</h3>
{build_filmstrip_row(noise_panels)}

<h3>Metric maps &amp; the top5 mask, by family</h3>
<p class="caption">Each row is one metric family's own map for the blurred
image (B) at a few illustrative blur levels and two stack depths (noisiest
realistic vs. well-stacked), downscaled for size, with a light magenta
overlay marking exactly the pixel set behind this page's "top5" curves
below &mdash; local-maxima peaks unioned with the top-N%-brightness mask
(<code>_combined_localmax_mask</code>), <b>not</b> the brightness-only mask
Section 8d&ndash;8j's report figures use. Watch how the magenta region's
shape and extent change (or jump) across blur levels: it is a discrete
pixel set, not a smooth field, so which pixels qualify can shift
non-smoothly as blur redistributes brightness even though the image itself
is changing smoothly &mdash; the leading suspect for this page's
response-curve discontinuities.</p>
{build_metric_overlay_section(target_id, blur)}

<h3>Primary families &mdash; whole-frame (bulk)</h3>
<p class="caption">Diagnostic population only. This project's own convention
(see tools/sensitivity_sweep.py's POP_* notes) is that the whole-frame mean
can run the wrong sign entirely under noise-dominated conditions. The
top-5%-brightest population below is the one calibration and the report's
own headline read.</p>
{build_response_curves(primary_bulk)}
{build_value_curves(primary_bulk)}
{build_scorecard_at(primary_bulk)}

<h3>Primary families &mdash; top-5%-brightest (the population that matters)</h3>
{build_response_curves(primary_top5)}
{build_value_curves(primary_top5)}
{build_scorecard_at(primary_top5)}

<h3>Noise-normalised variants &amp; global acutance (8k) &mdash; whole-frame (bulk)</h3>
<p class="caption">nrm_* rows divide the raw magnitude map by that image's
own per-scale noise floor (median magnitude over that image's own background
mask, from _nc_score) before comparing -- this is the population that
accounts for the varying SNR a real stack depth produces, which is why it
tracks a genuine sharpness/detail trend far more cleanly than the raw
(un-normalised) families above across different stack depths. lap_var_*/
grad_energy_* are Section 8k's whole-nebula absolute second-moment scalars
-- a magnitude, not a ratio against a top-N% population, so they belong in
their own scale on this shared axis and have no top-5% counterpart (next
section). Watch entropy_5px/9px (and their nrm_* variants) in particular:
they're the least numerically stable family on this page, sometimes by a
near-zero-denominator effect (dividing by a background noise floor that
sits very close to zero on stark/binary or heavily-smoothed content, e.g.
zebra and composite_dense at heavy blur, producing large unstable log-ratio
excursions) and sometimes just from the 5x5/9x9-pixel window being a
genuinely high-variance histogram-entropy estimator under strong noise --
confirmed on this exact grid: one single (target, stack depth) sigma=0
self-comparison spiked to -0.37 on fractal_00053 at stack=10 even though
every other target's same self-comparison stayed under 0.02. Neither is a
degradation signal or a pairing bug; both are real properties of the
entropy estimator, not something any other family showed across the whole
grid.</p>
{build_response_curves(nrm_bulk_and_acutance)}
{build_value_curves(nrm_bulk_and_acutance)}
{build_scorecard_at(nrm_bulk_and_acutance)}

<h3>Noise-normalised variants &mdash; top-5%-brightest</h3>
<p class="caption">The <b>same</b> top5/combined-localmax mask as "Primary
families &mdash; top-5%-brightest" above (computed from the raw magnitude
maps, not recomputed on the noise-normalised ones) &mdash; literally the
same pixel set, with the noise-normalised value averaged within it instead
of the raw one. No 8k row here: lap_var_*/grad_energy_* are whole-nebula
scalars with no top-N% population concept to restrict them to. No bootstrap
CI on this population either (unlike the raw top-5% section's, which comes
free from analyze()'s own local-maxima stats) -- these are point
estimates.</p>
{build_response_curves(nrm_top5)}
{build_value_curves(nrm_top5)}
{build_scorecard_at(nrm_top5)}
"""


def build_html(rows: list[dict], sections: list[str],
               consensus_bulk: str, consensus_top5: str,
               consensus_nrm_bulk: str, consensus_nrm_top5: str, n_targets: int) -> str:
    sections_html = "\n<hr>\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic Spatial-Detail Atlas</title><style>{_CSS}</style></head><body>

<h1>Synthetic Spatial-Detail Screening Atlas</h1>
<p class="caption">Generated by tools/spatial_detail_atlas.py from a
tools/spatial_detail_screen.py sweep &mdash; {len(rows)} rows, {n_targets} targets.</p>

<div class="warn">
<b>Diagnostic screening only.</b> This page is entirely separate from
resources/metric_calibration.json and core/practical_significance.py &mdash;
not wired into either &mdash; and carries no FWHM/practical-significance
labels anywhere, because every target below is deliberately star-free (there
is no PSF to fit). It exists to build intuition for how the Section 8
spatial-detail metrics respond to a known amount of Gaussian blur and a
known amount of camera-realistic shot/read noise, swept as two
<i>independent</i> axes, on synthetic targets standing in for real
nebula-like structure. Real-data calibration/validation against actual
filter comparisons is an explicit follow-on phase, not attempted here.
</div>

<div class="info">
<b>Reading the fractal/zebra targets.</b> The fractal-noise and zebra-stripe
targets are built from 8-bit PNG source images
(AstroLabTestData/Fractal Source/) that were already display-stretched by
whatever tool produced them. They are linearly rescaled here to a chosen
peak ADU, which preserves relative spatial structure but is not a
photometric de-stretch &mdash; read their absolute SNR as illustrative, not
calibrated. The Siemens-star and composite-accent content is generated
directly at a chosen ADU level and carries no such caveat.
</div>

<div class="info">
<b>Every case here has real noise.</b> Earlier passes of this sweep included
a perfectly noiseless reference case (no added shot/read noise at all) at
every blur level, shown as its own "noise=0.0" line on every response/value
curve. It was dropped after review: it behaved atypically compared to every
noisy case, obscured the trend that matters (how a metric responds under
conditions a real astro image would actually have), and no real image is
ever truly noiseless. Every case on this page now has camera-realistic shot
and read noise at some stack depth &mdash; the noisiest is a single sub
(<code>stack=1</code>), the cleanest a 50-sub stack (<code>stack=50</code>
on primary targets).
</div>

<div class="info">
<b>How "monotone" and "correctly signed" are decided.</b> For one
target/population/metric, take its log10(A/B) values at
{SCORECARD_SNR_LABEL} (the noisiest realistic single-sub case, present on
every target, and the most demanding condition to screen a metric against
now that there's no noiseless reference to fall back on) across the whole
blur ladder, sort by blur sigma:
<ul>
<li><b>Monotone</b> means the sequence never <i>decreases</i> as blur
increases (checked as <code>diffs = np.diff(vals); monotone =
np.all(diffs &gt;= -1e-12)</code> &mdash; that tolerance is numerical slack,
not a real allowance). A is the fixed, less-blurred reference and B gets
progressively blurrier, so the measured gap between them should only ever
grow or hold steady, never shrink and then grow again. A non-monotone curve
means the metric registered <i>less</i> difference between the sharp
reference and a moderately-blurred B than it did against a
<i>less</i>-blurred B &mdash; internally inconsistent as a sharpness
indicator, whatever its absolute value looks like.</li>
<li><b>Correctly signed</b> means every value in that same sequence stays
&ge; 0 (<code>sign_ok = all(v &gt;= -1e-9 for v in vals)</code>, again
float-noise slack only). Since A always has at least as much fine-scale
structure as any blurred version of itself, log10(A/B) should never go
negative &mdash; a negative value means the metric said the blurred image
has <i>more</i> of this kind of structure than the sharp original, which is
backwards.</li>
</ul>
This is the identical screen tools/sensitivity_sweep.py's
<code>_write_calibration</code> uses to decide what may enter
resources/metric_calibration.json &mdash; same two failure modes, same
reasoning, applied here purely diagnostically (see the warning banner above:
nothing on this page writes to that file).
</div>

<h2>Cross-target consensus</h2>
<p class="caption">For each metric, how many of the {n_targets} targets'
own <i>individual</i> blur sweeps (at {SCORECARD_SNR_LABEL}, the noisiest
realistic single-sub case) are monotone and correctly signed. This is an
AND across per-target scorecards, not one scorecard over
every target's rows pooled together &mdash; different targets share
blur-sigma values but have unrelated response magnitudes, so a single pooled
sweep would interleave unrelated curves and call it non-monotone almost
everywhere, which would be a methodology artifact, not a finding. The
bulk table is included for comparison, not as the recommended reading (see
the per-target sections' own caption; this project's own convention is that
top-5% carries the real signal).</p>
<h3>Whole-frame (bulk)</h3>
{consensus_bulk}
<h3>Top-5%-brightest</h3>
{consensus_top5}
<h3>Noise-normalised &mdash; whole-frame (bulk) &amp; global acutance (8k)</h3>
{consensus_nrm_bulk}
<h3>Noise-normalised &mdash; top-5%-brightest</h3>
{consensus_nrm_top5}

<hr>
{sections_html}

</body></html>"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path,
                   default=_REPO / "sweep_out" / "synthetic_detail" / "spatial_detail_screen.csv")
    p.add_argument("-o", "--out", type=Path,
                   default=_REPO / "sweep_out" / "synthetic_detail")
    args = p.parse_args(argv)

    if not args.csv.exists():
        print(f"{args.csv} not found -- run tools/spatial_detail_screen.py first")
        return 1
    with args.csv.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["tier"] == "synthetic-detail"]
    print(f"{len(rows)} rows")

    targets_present = sorted({r["case_id"] for r in rows},
                             key=lambda t: ALL_TARGETS.index(t) if t in ALL_TARGETS else 999)

    sections = [build_target_section(t, rows) for t in targets_present]

    per_target_bulk = {t: _metric_pass_fail(
        _rows_for([r for r in rows if r["case_id"] == t], population=POP_BULK))
        for t in targets_present}
    per_target_top5 = {t: _metric_pass_fail(
        _rows_for([r for r in rows if r["case_id"] == t], population=POP_TOP))
        for t in targets_present}
    per_target_nrm_bulk = {t: _metric_pass_fail(
        _nrm_acutance_rows([r for r in rows if r["case_id"] == t], top5=False))
        for t in targets_present}
    per_target_nrm_top5 = {t: _metric_pass_fail(
        _nrm_acutance_rows([r for r in rows if r["case_id"] == t], top5=True))
        for t in targets_present}
    consensus_bulk = build_consensus_table(per_target_bulk)
    consensus_top5 = build_consensus_table(per_target_top5)
    consensus_nrm_bulk = build_consensus_table(per_target_nrm_bulk)
    consensus_nrm_top5 = build_consensus_table(per_target_nrm_top5)

    html = build_html(rows, sections, consensus_bulk, consensus_top5,
                      consensus_nrm_bulk, consensus_nrm_top5, len(targets_present))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "spatial_detail_atlas.html"
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
