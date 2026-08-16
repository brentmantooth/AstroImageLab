"""Metric sensitivity sweep and calibration harness.

Answers two questions the HTML report cannot:

  1. How large a change in each metric corresponds to a change an imager would
     actually see? (calibration -> resources/metric_calibration.json)
  2. Which metrics respond monotonically and in the correct direction to a known
     degradation at all? (health screen)

Runs entirely headless with make_figures=False, which is ~25x faster than the
report path on SpatialDetailAnalyzer, so a grid of ~100 points is practical.

Subcommands
-----------
    validate       Re-read the six generated reports' _inspector.npz files and
                   label each with core.practical_significance. No re-analysis.
    m31-factorial  The real 2x2: SNR (3 vs 75 frames) x sharpness (native vs
                   BlurXTerminator), all four cells already on disk.
    m31-null       Two disjoint stacks of the same size from the 75 registered
                   sub-frames: a real image pair whose correct answer is
                   "identical". The strongest anchor available for the lowest
                   label.
    blur-grid      Known Gaussian blur x known added noise on a real frame.
                   Blur is noise-compensated by default (see
                   _blur_preserving_noise); --plain-blur reproduces a
                   post-stack blur instead. Produces the calibration curves.
    recalibrate    Rebuild the calibration from an existing blur-grid CSV,
                   without re-measuring (the grid is ~20 min, the screen is
                   a few lines).
    add-null-floors  Merge measured null floors from an m31-null CSV into the
                   calibration. Required: without floors, a reading at the
                   noise level maps through a steep curve to a large label.
    health         Monotonicity / sign screen over a blur-grid result.

Every metric is measured over two populations -- the whole frame (bulk) and
Section 8l's local-maxima mask (top5). Calibration and the headline read top5;
see the POP_* constants for why the bulk arm cannot be trusted here.

Typical full rebuild
--------------------
    python tools/sensitivity_sweep.py -o sweep_out blur-grid --crop 1200
    python tools/sensitivity_sweep.py -o sweep_out m31-null --n 37 --crop 1000
    python tools/sensitivity_sweep.py add-null-floors --csv sweep_out/m31_null.csv
    python tools/sensitivity_sweep.py validate
    python tools/metric_atlas.py --csv sweep_out/blur_grid.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.astro_image import AstroImage                      # noqa: E402
from core.practical_significance import (                    # noqa: E402
    Calibration, fwhm_change_pct, overall_label, verdict_for_fwhm,
    verdict_for_metric, verdict_for_snr,
)
from core.spatial_stats import block_bootstrap_ci            # noqa: E402

# ── Data locations ────────────────────────────────────────────────────────────
FILTER_COMPARE = _REPO / "AstroLabTestData" / "FilterCompare"
M31_DIR = _REPO / "AstroLabTestData" / "M31 Integrations"
M31_FRAMES = Path(r"D:\Astro\DataProcessing\M31 Andromeda\PixOut\registered"
                  r"\Light_BIN-1_6248x4176_EXPOSURE-60.00s_FILTER-Lum_mono")

# The six generated comparisons, with the reading their owner gives each one.
# These are the acceptance set: a labelling scheme that disagrees with the
# right-hand column is wrong regardless of what its p-values say.
# (npz filename, owner's reading, global SNR A, global SNR B).
# The SNR pair is read from each report's Section 3 and is a first-class axis:
# Rosette's SV advantage is almost entirely in SNR (+2.14 dB), so a headline
# built from sharpness alone reports that pair as no different at all.
REPORT_CASES = {
    "HyperHa 6 vs 12 (f/2.2)":
        ("report_HyperHa6_HyperHa12_20260812_202540_inspector.npz",
         "faster optics; blue shift costs the 6nm real signal", 5.1025, 5.0015),
    "SCT Ha 6 vs 12 (f/7)":
        ("report_SCTHa6_SCT_Ha12_20260812_205546_inspector.npz",
         "may or may not differ at this light-pollution level", 4.7934, 4.3575),
    "Rosette CXB vs SV":
        ("report_Rosette_CXB_5_Stacked_stacked_Rosette_SV_5_Stacked_stacked"
         "_20260811_220846_inspector.npz",
         "SV genuinely better on SNR and detail; how much is the question",
         11.2445, 14.3746),
    "OIII Orion CXB vs Opt":
        ("report_OIII_Orion_CXB_stacked_OIII_Orion_Opt_stacked"
         "_20260812_201332_inspector.npz",
         "3nm vs 12nm at f/10; blue shift negligible", 9.9475, 9.3959),
    "CXB crop: deconvolution sharpen":
        ("report_Image_A_Image_B_20260814_085623_inspector.npz",
         "VISIBLE difference — must come out material", 9.7403, 4.8199),
    "CXB crop: gaussian blur":
        ("report_Image_A_Image_B_20260814_090831_inspector.npz",
         "VISIBLE difference — must come out material", 8.7668, 5.6251),
}

# Section 8 families, in the order the report presents them.
METRIC_KEYS = [
    "std_3px", "std_5px", "std_10px",
    "log_1.5", "log_3.0", "log_6.0",
    "gradient_1.5", "gradient_3.0", "gradient_6.0",
    "wavelet_2", "wavelet_3",
    "entropy_5px", "entropy_9px", "entropy_17px",
    "localgrad_1.5", "localgrad_3.0", "localgrad_6.0",
    "loclap_1.5", "loclap_3.0", "loclap_6.0",
]


# The two populations a Section 8 log-ratio can be summarised over. Measured on
# a visibly-sharpened real pair, the top-5% population carries 2.6-3.6x more
# signal than the whole-frame mean -- and on std_3px the bulk mean ranks that
# pair BELOW a possibly-null filter pair, while top-5% separates them 5.8x the
# right way. Perceived sharpness lives in the bright structure, not the bulk.
POP_NEB = "nebula"    # Section 8m's whole-nebula-mask scalars -- no top_pct at all
POP_BULK = "bulk"     # every pixel of the frame
POP_TOP = "top5"      # Section 8l's local-maxima mask: peaks OR the top
                      # SECTION8_LOCALMAX_TOP_PERCENT% of A's or B's own values
#
# Measured under noise-compensated blur, 15 of 20 metrics are calibratable on
# top5 and only 1 of 20 on bulk. The bulk arm is not merely less sensitive, it
# is actively wrong: std_3px runs NEGATIVE and saturates at -0.19, i.e. it
# reports the blurred image as having more fine detail than its original.
#
# The cause is specific and worth knowing. `_noise_sigma` reads sky noise off a
# textured target (M31), so it is inflated by real galaxy structure and the
# compensation over-adds white noise by ~30% in sigma -- 1.23-1.32x measured on
# M31 against 0.98x on a synthetic frame with no texture. In the sky-dominated
# bulk that surplus noise dominates the small-scale statistics and flips the
# sign. Inside the top-5% mask the signal dominates and the surplus is
# negligible, which is why that arm stays clean and monotone.
#
# So bulk rows are kept for diagnostics and comparison only; calibration and
# the headline read top5.


@dataclass
class Row:
    """One (case, metric, population) measurement. The sweep's output schema."""
    tier: str
    case_id: str
    factor_sharpness: str = ""
    factor_snr: str = ""
    metric_key: str = ""
    population: str = POP_BULK   # POP_BULK | POP_TOP -- which pixels were summarised
    log_ratio: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    se: float | None = None
    block_px: int | None = None
    n_blocks: int | None = None
    converged: bool | None = None
    naive_se_understatement: float | None = None
    fwhm_pct: float | None = None
    label: str | None = None
    top_pct: float | None = None     # localmax_top_percent used for this row
    dilation: float | None = None    # localmax_region_fraction used for this row
    depth: int | None = None         # sub-frames per half-stack
    repeat: int | None = None        # which random partition
    note: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mem_image(arr: np.ndarray, label: str, pixel_scale: float = 1.0) -> AstroImage:
    """AstroImage over an in-memory array. AstroImage.__init__ never touches the
    filesystem -- only load() does -- so the path argument is just a label
    source. Same informal route the codebase already uses for externally
    populated slots; there is no from_array factory."""
    img = AstroImage(f"{label}.fits", label=label)
    img.data = np.ascontiguousarray(arr, dtype=np.float32)
    img.pixel_scale = pixel_scale
    return img


def _centre_crop(arr: np.ndarray, size: int | None) -> np.ndarray:
    if not size or size <= 0 or min(arr.shape[:2]) <= size:
        return arr
    h, w = arr.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def _load(path: Path, crop: int | None = None) -> np.ndarray:
    img = AstroImage(str(path))
    img.load()
    return _centre_crop(img.data, crop)


def _bootstrap_row(tier: str, case_id: str, metric_key: str,
                   diff: np.ndarray, calibration: Calibration,
                   seed: int = 0, mask: np.ndarray | None = None,
                   population: str = POP_BULK, **factors) -> Row:
    res = block_bootstrap_ci(diff, mask=mask, seed=seed)
    if res is None:
        return Row(tier=tier, case_id=case_id, metric_key=metric_key,
                   population=population, note="bootstrap failed", **factors)
    # Calibration is keyed by metric AND population -- a top-5% reading must
    # not be interpreted through a bulk-derived curve, they differ by 2-4x.
    verdict = verdict_for_metric(_cal_key(metric_key, population),
                                 res.point, calibration)
    return Row(
        tier=tier, case_id=case_id, metric_key=metric_key,
        population=population,
        log_ratio=res.point, ci_lo=res.lo, ci_hi=res.hi, se=res.se,
        block_px=res.block_px, n_blocks=res.n_blocks, converged=res.converged,
        naive_se_understatement=res.se_understatement,
        fwhm_pct=verdict.equivalent, label=verdict.label,
        note=verdict.note if not verdict.calibrated else
             ("" if res.converged else "SE still growing at largest block"),
        **factors)


def _cal_key(metric_key: str, population: str) -> str:
    """Calibration entries are per (metric, population)."""
    return metric_key if population == POP_BULK else f"{metric_key}@{population}"


def _measure_all(tier: str, case_id: str, arr_a, arr_b,
                 cal: Calibration, seed: int, **factors) -> list[Row]:
    """Both populations for every metric, as separate rows."""
    rows: list[Row] = []
    for key, (diff, mask) in _spatial_measures(arr_a, arr_b).items():
        rows.append(_bootstrap_row(tier, case_id, key, diff, cal, seed=seed,
                                   population=POP_BULK, **factors))
        if mask is not None and mask.any():
            rows.append(_bootstrap_row(tier, case_id, key, diff, cal, seed=seed,
                                       mask=mask, population=POP_TOP, **factors))
    return rows


def _localmax_mask(abs_a: np.ndarray, abs_b: np.ndarray,
                   scale_px: float) -> np.ndarray:
    """Rebuild Section 8l's combined local-maxima mask for one scale.

    Calls SpatialDetailAnalyzer's own `_combined_localmax_mask` with the shipped
    constants, so the population measured here is the same one the report's 8l
    table reports on -- never a second implementation of the same idea.
    """
    from analysis.image_filters import SpatialDetailAnalyzer
    from core.models import (SECTION8_LOCALMAX_FOOTPRINT_MULT,
                             SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                             SECTION8_LOCALMAX_REGION_FRACTION,
                             SECTION8_LOCALMAX_PRESMOOTH_FRACTION,
                             SECTION8_LOCALMAX_TOP_PERCENT)
    footprint = max(3, int(round(SECTION8_LOCALMAX_FOOTPRINT_MULT * scale_px)) | 1)
    region = max(1, int(round(SECTION8_LOCALMAX_REGION_FRACTION * footprint)))
    presmooth = max(0.5, SECTION8_LOCALMAX_PRESMOOTH_FRACTION * scale_px)
    return SpatialDetailAnalyzer()._combined_localmax_mask(
        abs_a, abs_b, footprint, SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
        region, presmooth, SECTION8_LOCALMAX_TOP_PERCENT)


# Characteristic scale per metric key, for the local-maxima footprint.
_SCALE_OF = {f"std_{k}px": float(k) for k in (3, 5, 10)}
_SCALE_OF.update({f"entropy_{k}px": float(k) for k in (5, 9, 17)})
for _s in (1.5, 3.0, 6.0):
    _SCALE_OF.update({f"log_{_s}": _s, f"gradient_{_s}": _s,
                      f"localgrad_{_s}": _s, f"loclap_{_s}": _s})
_SCALE_OF.update({f"wavelet_{lv}": float(2 ** lv) for lv in (2, 3)})


def _spatial_measures(arr_a: np.ndarray, arr_b: np.ndarray,
                      pixel_scale: float = 1.0
                      ) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Run SpatialDetailAnalyzer headlessly -> {metric: (diff_map, top5_mask)}.

    The diff map is the same `panels[key]["diff"]` the report builds. The mask
    restricts it to the brightest/peak structure, which is where perceived
    sharpness actually lives: measured on a visibly-sharpened real pair, the
    top-5% population carries 2.6-3.6x more signal than the whole-frame mean,
    and it is the difference between ranking that pair above a possibly-null
    filter pair and ranking it below.
    """
    from analysis.image_filters import SpatialDetailAnalyzer
    a = _mem_image(arr_a, "A", pixel_scale)
    b = _mem_image(arr_b, "B", pixel_scale)
    res = SpatialDetailAnalyzer().analyze(a, b, make_figures=False)
    out: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for key, panel in res.get("panels", {}).items():
        if key not in METRIC_KEYS or panel.get("diff") is None:
            continue
        diff = panel["diff"]
        mask = None
        pa, pb, scale = panel.get("a"), panel.get("b"), _SCALE_OF.get(key)
        if pa is not None and pb is not None and scale:
            h = min(pa.shape[0], pb.shape[0], diff.shape[0])
            w = min(pa.shape[1], pb.shape[1], diff.shape[1])
            mask = _localmax_mask(np.abs(pa[:h, :w]), np.abs(pb[:h, :w]), scale)
            diff = diff[:h, :w]
        out[key] = (diff, mask)
    return out


def _noise_sigma(arr: np.ndarray) -> float:
    """Sky noise, from photutils' sigma-clipped per-cell RMS.

    Deliberately not "MAD of a high-pass residual": that counts real
    small-scale nebula texture as noise, because pervasive texture is not an
    outlier a MAD rejects. Using it to size the blur compensation over-added
    noise badly enough to invert the sign of the bulk std_3px log-ratio -- the
    grid then claimed a blurred image had *more* fine detail than its original.
    Background2D's clip is per-cell and signal-robust, which is the property
    that matters here.
    """
    img = _mem_image(arr, "noise")
    img.estimate_background()
    rms = img.background_rms
    return float(np.median(rms)) if rms is not None else float(np.std(arr))


def _blur_noise_survival(sigma_b: float, shape=(512, 512), seed: int = 12345) -> float:
    """Fraction of white-noise *variance* surviving a Gaussian blur.

    Measured by blurring an actual white-noise field with the actual kernel,
    rather than from `_smoothed_sigma`'s sigma/sqrt(4*pi*sigma_k**2). That
    closed form is asymptotic and badly wrong for the sub-pixel kernels this
    grid cares most about: at sigma_b=0.5 it predicts 0.318 survival, while a
    discrete half-pixel Gaussian is nearly a delta function and barely
    attenuates. Over-adding compensation noise there would corrupt exactly the
    region where the response knee lives.

    Signal-free by construction, deterministic, and costs one 512x512 blur.
    """
    from scipy.ndimage import gaussian_filter
    noise = np.random.default_rng(seed).normal(0.0, 1.0, shape)
    return float(min(1.0, np.var(gaussian_filter(noise, sigma_b)) / np.var(noise)))


def _blur_preserving_noise(arr: np.ndarray, sigma: float,
                           rng) -> np.ndarray:
    """Blur, then restore the noise the blur removed.

    A plain post-stack Gaussian blur lowers detail *and* noise together --
    measured on a real frame, sigma=2 px drops the noise floor to 0.46x and
    *raises* snr_global from 17.4 to 20.5. Calibrating against that conflates
    two independent things, and real filter differences change detail and noise
    for unrelated physical reasons.

    A softer optical system instead blurs the photon distribution before shot
    and read noise are realised, so detail falls at a roughly constant noise
    floor. Adding back the missing variance reproduces that: the same blurs
    then give 17.3 / 16.3 / 16.0, falling, as they should.
    """
    from scipy.ndimage import gaussian_filter
    sigma0 = _noise_sigma(arr)                      # measured on the CLEAN frame
    survival = _blur_noise_survival(sigma)          # closed form, signal-free
    var_add = max(0.0, sigma0 ** 2 * (1.0 - survival))
    out = gaussian_filter(arr, sigma)
    if var_add > 0:
        out = out + rng.normal(0.0, np.sqrt(var_add), out.shape).astype(out.dtype)
    return out.astype(np.float32)


def _psf_fwhm(arr: np.ndarray, pixel_scale: float = 1.0) -> float | None:
    from analysis.psf_analyzer import PSFAnalyzer
    r = PSFAnalyzer(epsf_max_stars=200).analyze(
        _mem_image(arr, "psf", pixel_scale), make_figures=False)
    return r.get("fwhm_px")


def _global_snr(arr: np.ndarray, pixel_scale: float = 1.0) -> float | None:
    from analysis.snr_analyzer import SNRAnalyzer
    return SNRAnalyzer().analyze(
        _mem_image(arr, "snr", pixel_scale), make_figures=False).get("snr_global")


def _write(rows: list[Row], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(Row.__dataclass_fields__))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"\nwrote {len(rows)} rows -> {path}")
    return path


# ── validate: the six existing reports, no re-analysis ────────────────────────

def cmd_validate(args) -> list[Row]:
    cal = Calibration.load()
    rows: list[Row] = []
    print(f"calibration: {cal.source} ({len(cal.metrics)} metrics)\n")

    for case, (fname, expectation, snr_a, snr_b) in REPORT_CASES.items():
        path = FILTER_COMPARE / fname
        if not path.exists():
            print(f"  [skip] {case}: {fname} not found")
            continue
        z = np.load(path, allow_pickle=True)
        print(f"{case}\n    expected: {expectation}")

        # FWHM is the currency and needs no calibration, so it can be labelled
        # even before any sweep has run.
        fa = _map_median(z, "psf_fwhm_map_a")
        fb = _map_median(z, "psf_fwhm_map_b")
        v = verdict_for_fwhm(fa, fb)
        rows.append(Row(tier="report", case_id=case, metric_key="psf_fwhm_px",
                        log_ratio=None, fwhm_pct=v.equivalent, label=v.label,
                        note="currency anchor"))
        print(f"    FWHM {fa:.3f} -> {fb:.3f} px  ({v.equivalent:+.1f}%)"
              f"   => {v.label}")

        keys = [k for k in METRIC_KEYS if f"sp_{k}_diff" in z]
        if args.quick:
            keys = keys[:args.quick]
        before = len(rows)
        for key in keys:
            diff = z[f"sp_{key}_diff"].astype(np.float32)
            rows.append(_bootstrap_row("report", case, key, diff, cal,
                                       seed=args.seed, population=POP_BULK))
            pa = z[f"sp_{key}_a"] if f"sp_{key}_a" in z else None
            pb = z[f"sp_{key}_b"] if f"sp_{key}_b" in z else None
            scale = _SCALE_OF.get(key)
            if pa is None or pb is None or not scale:
                continue
            h = min(pa.shape[0], pb.shape[0], diff.shape[0])
            w = min(pa.shape[1], pb.shape[1], diff.shape[1])
            mask = _localmax_mask(np.abs(pa[:h, :w]), np.abs(pb[:h, :w]), scale)
            rows.append(_bootstrap_row("report", case, key, diff[:h, :w], cal,
                                       seed=args.seed, mask=mask,
                                       population=POP_TOP))
        # The headline reads the top-5% population: it is what tracks perceived
        # sharpness. The bulk mean ranks a visibly-sharpened pair BELOW a
        # possibly-null filter pair on std_3px (-0.0163 vs -0.0198).
        labelled = [r.label for r in rows[before:]
                    if r.label and r.population == POP_TOP]
        section8 = _max_label(labelled) if labelled else None
        print(f"    {len(keys)} Section-8 metrics; consensus "
              f"{section8 or 'uncalibrated'}")

        # Headline across distinct axes: max, because sharpness and detail are
        # independent. This is what recovers the sharpen case -- Section 8
        # genuinely cannot see deconvolution sharpening, while FWHM can.
        vs = verdict_for_snr(snr_a, snr_b)
        rows.append(Row(tier="report", case_id=case, metric_key="snr_global",
                        label=vs.label, note="currency anchor"))
        print(f"    SNR  {snr_a:.3f} -> {snr_b:.3f} ({vs.equivalent:+.2f} dB)"
              f"   => {vs.label}")

        from core.practical_significance import PRACTICAL_LABELS, PRACTICAL_ORDER
        axes = [x for x in (v.label, vs.label, section8) if x in PRACTICAL_ORDER]
        headline = PRACTICAL_LABELS[max(PRACTICAL_ORDER[x] for x in axes)] if axes else None
        print(f"    HEADLINE: {headline}\n")

    return rows


def _map_median(z, key: str) -> float | None:
    if key not in z:
        return None
    a = z[key]
    good = a[np.isfinite(a) & (a > 0)]
    return float(np.median(good)) if good.size else None


def _max_label(labels) -> str | None:
    """Median, not max: Section 8's metrics are correlated estimates of one
    quantity, so consensus_label is the right reducer for them. Name kept for
    the call site above; see core.practical_significance for the reasoning."""
    from core.practical_significance import PRACTICAL_LABELS, PRACTICAL_ORDER
    ranks = sorted(PRACTICAL_ORDER[x] for x in labels if x in PRACTICAL_ORDER)
    return PRACTICAL_LABELS[ranks[len(ranks) // 2]] if ranks else None


# ── m31: real SNR ladder, real 2x2, real null ─────────────────────────────────

def _m31(name: str) -> Path:
    return M31_DIR / name


def cmd_m31_factorial(args) -> list[Row]:
    """Tier 1a: sharpness (native vs blurEX) x SNR (3 vs 75 frames).

    IMPORTANT -- blurEX is NOT a controlled factor, so do not read the two SNR
    columns as a clean interaction measurement:

      * BlurXTerminator sharpens *stars* independently of nebula structure, and
        strongly. FWHM is measured on stars while the Section 8 metrics are
        measured over the nebula mask, so the two are tracking different things
        that this tool changed by different amounts. FWHM cannot serve as the
        currency for a blurEX pair the way it can for a uniform blur.
      * It is a CNN, so it is not the same operator at both SNR levels -- given
        3 frames it has far less signal to work with and treats nebulosity far
        more conservatively than at 75. A difference between the columns is
        therefore partly the algorithm behaving differently and partly the
        metrics responding differently, and this arm cannot separate them.

    The controlled interaction number has to come from `blur-grid`, where a
    fixed Gaussian kernel is applied identically at every noise level. What this
    arm *is* good for: a realistic, strong, real-data sharpening case to check
    that the metrics move in the right direction at all.
    """
    cal = Calibration.load()
    cells = {
        "3 frames":  (_m31("M31 integration3_ABE1.xisf"),
                      _m31("M31 integration3_ABE1_blurEX.xisf")),
        "75 frames": (_m31("M31 integration75_ABE1.xisf"),
                      _m31("M31 integration75_ABE1_blurEX.xisf")),
    }
    rows: list[Row] = []
    for snr_level, (pa, pb) in cells.items():
        if not (pa.exists() and pb.exists()):
            print(f"  [skip] {snr_level}: missing {pa.name} / {pb.name}")
            continue
        print(f"\n=== BlurXTerminator effect at {snr_level} ===")
        a, b = _load(pa, args.crop), _load(pb, args.crop)
        fa, fb = _psf_fwhm(a), _psf_fwhm(b)
        pct = fwhm_change_pct(fa, fb)
        print(f"  FWHM {fa} -> {fb} px ({pct:+.1f}%)" if pct is not None
              else "  FWHM unavailable")
        rows.append(Row(tier="m31-2x2", case_id=f"blurEX @ {snr_level}",
                        factor_sharpness="blurEX", factor_snr=snr_level,
                        metric_key="psf_fwhm_px", fwhm_pct=pct,
                        label=verdict_for_fwhm(fa, fb).label))
        rows.extend(_measure_all("m31-2x2", f"blurEX @ {snr_level}", a, b,
                                 cal, args.seed,
                                 factor_sharpness="blurEX",
                                 factor_snr=snr_level))
    _report_interaction(rows)
    return rows


def _report_interaction(rows: list[Row]) -> None:
    """The point of the 2x2: does the same sharpening register differently at
    different SNR? A flat answer here would mean the interaction measured
    synthetically (2.5x) does not survive on real data -- worth knowing either
    way, so it is printed rather than asserted."""
    by = {}
    for r in rows:
        if r.metric_key in METRIC_KEYS and r.log_ratio is not None:
            by.setdefault(r.metric_key, {})[r.factor_snr] = r.log_ratio
    pairs = [(k, v["3 frames"], v["75 frames"]) for k, v in by.items()
             if {"3 frames", "75 frames"} <= set(v)]
    if not pairs:
        return
    print("\n=== interaction: same sharpening, two SNR levels ===")
    print(f"{'metric':16s}{'@3 frames':>12s}{'@75 frames':>12s}{'ratio':>9s}")
    for k, lo, hi in sorted(pairs):
        ratio = hi / lo if lo else float("nan")
        print(f"{k:16s}{lo:+12.4f}{hi:+12.4f}{ratio:9.2f}")


def cmd_m31_null(args) -> list[Row]:
    """Tier 1c: two DISJOINT stacks of N frames each.

    Same target, same optics, same night, same exposure, same sharpness, same
    SNR. The correct answer is "no difference", known with certainty -- which is
    what makes this the strongest test in the harness. Any metric that calls
    this pair material is disqualified.
    """
    cal = Calibration.load()
    frames = sorted(M31_FRAMES.glob("*_c_r.xisf"))
    if len(frames) < 2 * args.n:
        print(f"need {2 * args.n} frames, found {len(frames)} in {M31_FRAMES}")
        return []
    print(f"stacking two disjoint sets of {args.n} from {len(frames)} frames")

    stack_a = _stack(frames[:args.n], args.crop)
    stack_b = _stack(frames[args.n:2 * args.n], args.crop)

    rows: list[Row] = []
    fa, fb = _psf_fwhm(stack_a), _psf_fwhm(stack_b)
    v = verdict_for_fwhm(fa, fb)
    print(f"\nFWHM {fa} vs {fb} px -> {v.equivalent:+.2f}%  => {v.label}")
    rows.append(Row(tier="m31-null", case_id=f"disjoint {args.n}+{args.n}",
                    metric_key="psf_fwhm_px", fwhm_pct=v.equivalent,
                    label=v.label, note="known-null pair"))

    sa, sb = _global_snr(stack_a), _global_snr(stack_b)
    vs = verdict_for_snr(sa, sb)
    print(f"SNR  {sa} vs {sb} -> {vs.equivalent:+.3f} dB  => {vs.label}")
    rows.append(Row(tier="m31-null", case_id=f"disjoint {args.n}+{args.n}",
                    metric_key="snr_global", fwhm_pct=None, label=vs.label,
                    note="known-null pair"))

    print(f"\n{'metric':16s}{'pop':>6s}{'log ratio':>12s}{'95% CI':>24s}")
    for r in _measure_all("m31-null", f"disjoint {args.n}+{args.n}",
                          stack_a, stack_b, cal, args.seed):
        rows.append(r)
        ci = (f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]"
              if r.ci_lo is not None else "—")
        print(f"{r.metric_key:16s}{r.population:>6s}"
              f"{(r.log_ratio or 0):+12.4f}{ci:>24s}")
    return rows


def _stack(paths: list[Path], crop: int | None) -> np.ndarray:
    """Mean-stack pre-registered frames via a running sum.

    The frames come out of PixInsight already calibrated and registered, so this
    is a plain average -- no alignment, and deliberately no astroalign (warping
    them again would add exactly the resampling bias this harness exists to
    avoid). Running sum rather than np.stack: 75 x 26 MP would be ~7.8 GB.
    """
    total = None
    for i, p in enumerate(paths, 1):
        arr = _load(p, crop).astype(np.float64)
        total = arr if total is None else total + arr
        if i % 10 == 0 or i == len(paths):
            print(f"    stacked {i}/{len(paths)}")
    return (total / len(paths)).astype(np.float32)


# ── blur-grid: the calibration curves ─────────────────────────────────────────

def cmd_blur_grid(args) -> list[Row]:
    """Tier 2: known Gaussian blur x known added noise on a real frame.

    Produces the metric -> equivalent-FWHM curves that everything else needs,
    and is dense below sigma 0.5 px because that is where the response knee and
    every real filter comparison actually live.
    """
    from scipy.ndimage import gaussian_filter

    src = Path(args.image) if args.image else _default_blur_base()
    if src is None or not src.exists():
        print(f"no base image (looked for {src}); pass --image")
        return []
    base = _load(src, args.crop)
    print(f"base: {src.name}  shape={base.shape}")

    sky_sigma = float(np.std(base))
    noise_levels = [float(x) for x in args.noise]
    blur_sigmas = [float(x) for x in args.blur]
    rng = np.random.default_rng(args.seed)

    cal = Calibration.load()
    rows: list[Row] = []
    curve: dict[str, list[tuple[float, float, float]]] = {}

    for noise in noise_levels:
        ref = (base + rng.normal(0, noise * sky_sigma, base.shape).astype(np.float32)
               if noise > 0 else base)
        fwhm_ref = _psf_fwhm(ref)
        for sigma in blur_sigmas:
            if sigma <= 0:
                continue
            t0 = time.perf_counter()
            deg = (gaussian_filter(ref, sigma) if args.plain_blur
                   else _blur_preserving_noise(ref, sigma, rng))
            # PSFAnalyzer returns None once heavy blur plus added noise leaves
            # fewer than 3 fittable stars -- expected at the far corner of the
            # grid, not an error. The Section 8 rows are still valid there; only
            # the FWHM currency is unavailable, so those points cannot join the
            # calibration curve.
            fwhm_deg = _psf_fwhm(deg)
            pct = fwhm_change_pct(fwhm_ref, fwhm_deg)
            case = f"blur{sigma}_noise{noise}"
            for r in _measure_all("blur-grid", case, ref, deg, cal, args.seed,
                                  factor_sharpness=f"blur sigma={sigma}",
                                  factor_snr=f"noise={noise}"):
                r.fwhm_pct = pct       # measured truth, not the calibration
                rows.append(r)
                if r.log_ratio is not None and pct is not None and noise == noise_levels[0]:
                    # Keep sigma: _write_calibration must judge monotonicity in
                    # *blur* order, not in ratio order.
                    curve.setdefault(_cal_key(r.metric_key, r.population),
                                     []).append((sigma, r.log_ratio, abs(pct)))
            fwhm_txt = (f"FWHM {fwhm_ref:.3f}->{fwhm_deg:.3f} ({pct:+.1f}%)"
                        if pct is not None else
                        "FWHM unavailable (too few fittable stars)")
            print(f"  sigma={sigma:<5} noise={noise:<5} {fwhm_txt} "
                  f"[{time.perf_counter()-t0:.1f}s]", flush=True)

    if curve:
        _write_calibration(curve, args.out, src.name)
    return rows


def _default_blur_base() -> Path | None:
    for cand in (M31_DIR / "M31 integration75_ABE1.xisf",
                 M31_DIR / "M31 integration75.xisf"):
        if cand.exists():
            return cand
    return None


# A metric must span at least this many log-ratio units across the whole blur
# sweep to be invertible. Below it the curve is so flat that a noise-sized
# reading maps to an enormous equivalent-FWHM: gradient_3.0 covers 0.0085 log
# units over a 0-141% FWHM range, so a 0.01 measurement -- well inside its own
# null floor -- would be reported as "material".
MIN_CALIBRATION_RANGE = 0.05


def _write_calibration(curve: dict[str, list[tuple[float, float, float]]],
                       out_dir: Path, source: str) -> Path:
    """Emit resources/metric_calibration.json.

    Only metrics that pass the same screen `health` applies get in. Three ways
    to fail, all of which mean "cannot carry a practical label":

      * non-monotone in blur -- the response doubles back, so a ratio maps to
        more than one FWHM change and the inverse is not a function;
      * wrong sign -- blurring B must raise log10(|A|/|B|), and several coarse-
        scale metrics go the other way;
      * range below MIN_CALIBRATION_RANGE -- technically invertible, but so
        flat that inverting it amplifies noise into a confident wrong answer.

    Sorting the points by ratio and keeping the increasing ones (the earlier
    approach) *manufactured* monotonicity out of a non-monotone response and
    let exactly the broken metrics through. Judge in blur order instead.
    """
    metrics = {}
    dropped: dict[str, str] = {}
    for key, pts in curve.items():
        pts = sorted(pts)                       # by blur sigma
        ratios = [r for _, r, _ in pts]
        diffs = np.diff(ratios)

        if not np.all(diffs >= -1e-12):
            dropped[key] = "not monotone in blur"
            continue
        if any(r < -1e-9 for r in ratios):
            dropped[key] = "wrong sign"
            continue
        span = max(ratios) - min(ratios)
        if span < MIN_CALIBRATION_RANGE:
            dropped[key] = f"range {span:.4f} below {MIN_CALIBRATION_RANGE}"
            continue

        xs, ys = [0.0], [0.0]
        for _, ratio, pct in pts:
            if abs(ratio) > xs[-1] and pct >= ys[-1]:
                xs.append(abs(ratio))
                ys.append(pct)
        if len(xs) >= 3:
            metrics[key] = {"abs_log_ratio": xs, "fwhm_pct": ys}
        else:
            dropped[key] = "too few usable points"

    payload = {
        "source": f"tools/sensitivity_sweep.py blur-grid on {source}",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    # Always resources/, regardless of --out: this is the shipped artifact and
    # the only place Calibration.load() looks. --out controls the CSVs.
    path = _REPO / "resources" / "metric_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\ncalibrated {len(metrics)} metrics -> {path}")
    for key, why in sorted(dropped.items()):
        print(f"  uncalibrated  {key:16s} {why}")
    return path


# ── null-ensemble: repeated disjoint partitions across configurations ─────────

RAW_ROOT = _REPO / "AstroLabTestData" / "RawFiles"


def _list_subs(set_spec: str) -> list[Path]:
    """All sub-frames for a 'Target/Filter' spec, sorted for reproducibility."""
    d = RAW_ROOT / set_spec
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in (".xisf", ".fit", ".fits"))


def _load_cache(paths: list[Path], crop: int) -> np.ndarray:
    """Load every sub once, centre-crop, and hold the stack in memory.

    This is what makes the ensemble affordable. 396 sub-frames at 1024^2 float32
    is ~1.6 GB, so the disk cost is paid once and every partition afterwards is
    an array sum rather than a re-read of 26 GB.
    """
    out = np.empty((len(paths), crop, crop), dtype=np.float32)
    for i, p in enumerate(paths):
        out[i] = _centre_crop(_load(p), crop)
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            print(f"      cached {i + 1}/{len(paths)}", flush=True)
    return out


STARLESS_CLI = Path(r"D:\Astro\SyQon\starless_cli\starless_cli.exe")
_STARLESS_TMP = _REPO / "sweep_out" / "_starless_tmp"


def starless(arr: np.ndarray, tag: str = "s") -> np.ndarray | None:
    """Remove stars via SyQon Axiom, returning a linear array.

    Applied to the *stacked* image rather than to individual subs, because that
    is what the report does -- gui/analysis_thread.py hands Section 8
    `self._starless_a or img_a`, and the starless companion is generated from
    the stack.

    This matters more than it sounds. Measured on IC1179, the top-5% detail mask
    is 29.7% star pixels at a 14.8x enrichment over their frame coverage; after
    star removal that falls to 0.7%, an enrichment of 0.4x. Calibrating on
    star-containing stacks was therefore fitting the response of stars, not of
    the nebula detail the report reports on.

    --use-mtf is required: the stacks are linear and the model expects
    display-stretched input, so the CLI stretches, infers, and inverts.
    """
    import subprocess
    from astropy.io import fits
    if not STARLESS_CLI.exists():
        return None
    _STARLESS_TMP.mkdir(parents=True, exist_ok=True)
    src = _STARLESS_TMP / f"{tag}_in.fits"
    dst = _STARLESS_TMP / f"{tag}_out.fits"
    fits.PrimaryHDU(np.asarray(arr, dtype=np.float32)).writeto(src, overwrite=True)
    if dst.exists():
        dst.unlink()
    res = subprocess.run(
        [str(STARLESS_CLI), "--input", str(src), "--output", str(dst), "--use-mtf"],
        capture_output=True, text=True)
    if res.returncode != 0 or not dst.exists():
        print(f"      [starless failed] {(res.stderr or '')[-200:]}")
        return None
    return np.asarray(fits.getdata(dst), dtype=np.float32)


def _maybe_starless(arr: np.ndarray, enabled: bool, tag: str) -> np.ndarray:
    out = starless(arr, tag) if enabled else None
    return arr if out is None else out


def _frame_quality(cache: np.ndarray) -> np.ndarray:
    """Per-frame median local sigma at 3 px, relative to the set's median.

    A cheap proxy for fine-scale frame quality: clouds, trailing, focus drift
    and satellite streaks all change small-scale structure. Values near 1.0 are
    consistent with the rest of the set.
    """
    from scipy.ndimage import uniform_filter
    q = np.empty(len(cache))
    for i, frame in enumerate(cache):
        a = frame.astype(np.float64)
        m = uniform_filter(a, 3)
        m2 = uniform_filter(a * a, 3)
        q[i] = np.median(np.sqrt(np.clip(m2 - m * m, 0.0, None)))
    med = np.median(q)
    return q / med if med > 0 else np.ones_like(q)


def _reject_outlier_frames(cache: np.ndarray, paths: list[Path],
                           hi: float, lo: float) -> tuple[np.ndarray, list[int]]:
    """Drop frames whose fine-scale structure is inconsistent with the set.

    This must run *before* any null floor is measured. A single bad sub does not
    raise the floor a little -- it dominates it completely, and in a way that
    looks exactly like a real floor unless the distribution is inspected.
    Measured on Dumbell/Ha6: one frame at 7.93x the set median produced null
    pairs of -0.462, -0.458, -0.449, +0.429 -- near-identical magnitude with the
    sign flipping according to which half that frame landed in, i.e. a 2.9x
    apparent difference between two stacks of identical data. Taking a max or
    upper quantile over partitions would have written that straight into the
    floor and suppressed every real difference below 2.9x.
    """
    q = _frame_quality(cache)
    bad = [i for i, v in enumerate(q) if v > hi or v < lo]
    for i in bad:
        print(f"      reject [{i}] {q[i]:.2f}x  {paths[i].name[:56]}")
    if not bad:
        return cache, []
    keep = [i for i in range(len(cache)) if i not in bad]
    return cache[keep], bad


def _block_stacks(cache: np.ndarray, n_blocks: int, rng,
                  starless_on: bool, tag: str) -> tuple[np.ndarray, int]:
    """Partition subs into disjoint blocks; stack and star-remove each ONCE.

    The restructure that makes the ensemble affordable. Star-removing every
    half-stack meant 840 Axiom calls for 420 pairs, ~2.3 h, and almost all of
    it was redundant -- the half-stacks are drawn from the same pool. Removing
    stars at block level instead lets every pair be assembled from
    already-starless building blocks: one blocking of 10 blocks supports 126
    distinct 5v5 pairs plus everything at 2v2 and 1v1, for 10 Axiom calls.

    Measured: star removal does not parallelise (2 concurrent calls give 1.17x,
    4 give 0.63x -- worse than sequential, since Axiom is GPU-bound via
    DirectML), so cutting the call count is the only lever that works.

    It also keeps the model in its trained regime: a block is a stack of
    several subs, never a single noisy frame, which is the concern that ruled
    out per-sub removal.
    """
    idx = rng.permutation(len(cache))
    size = len(cache) // n_blocks
    blocks = np.empty((n_blocks, cache.shape[1], cache.shape[2]), dtype=np.float32)
    for k in range(n_blocks):
        sel = idx[k * size:(k + 1) * size]
        blocks[k] = _maybe_starless(cache[sel].mean(axis=0), starless_on,
                                    f"{tag}b{k}")
    return blocks, size


def _block_pairs(n_blocks: int, half: int, n_pairs: int,
                 rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n_pairs` (A, B) block-index pairs, each side using `half` blocks.

    Every pair is internally disjoint, which is what null validity requires.
    Pairs drawn from one blocking do share blocks with each other, so they are
    not mutually independent -- run several blockings to keep the spread of the
    floor honest rather than optimistic.
    """
    out = []
    if 2 * half > n_blocks:
        return out
    for _ in range(n_pairs):
        sel = rng.choice(n_blocks, size=2 * half, replace=False)
        out.append((sel[:half], sel[half:]))
    return out


def _disjoint_pairs(n: int, groups: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle n indices into `groups` equal disjoint blocks, pair them up.

    Every returned pair is two *disjoint* sets of sub-frames, so the correct
    answer for the pair is exactly zero difference. Splitting into more groups
    gives shallower but more numerous pairs, which is how depth is varied
    without ever reusing a frame across the two halves -- the flaw in the
    pre-made nested integrations.
    """
    idx = rng.permutation(n)
    size = n // groups
    if size < 1:
        return []
    blocks = [idx[i * size:(i + 1) * size] for i in range(groups)]
    return [(blocks[i], blocks[i + 1]) for i in range(0, groups - 1, 2)]


def cmd_null_ensemble(args) -> list[Row]:
    """Null-floor distribution over repeated disjoint partitions.

    Answers the question a single null pair cannot: how big is a metric's
    difference between two images that are *known* to be the same, and how
    reproducible is that number? Measured across four stack depths on one
    target, the same metric's floor moved non-monotonically by up to 50x, which
    is why floors are now taken from a distribution rather than a point.

    Also sweeps the two mask parameters that decide which pixels the detail
    metrics are summarised over -- `localmax_top_percent` and the dilation
    `localmax_region_fraction`. Both were set by intuition; the ensemble lets
    them be chosen by measurement, since the figure of merit (blur response
    divided by this floor) needs both halves and only the floor half is new.
    """
    from analysis.image_filters import SpatialDetailAnalyzer

    if args.jobs > 1:
        # Sets are completely independent -- no shared state, and with
        # make_figures=False there is no matplotlib in the path, so the
        # mathtext/draw race CLAUDE.md documents cannot apply. Processes rather
        # than threads because SpatialDetailAnalyzer is CPU-bound under the GIL.
        # Star removal still serialises on the GPU (measured: 2 concurrent
        # Axiom calls give 1.17x, 4 give 0.63x), but block-level removal cut it
        # to a minority of the runtime, so the analysis half now dominates and
        # that half parallelises cleanly.
        from concurrent.futures import ProcessPoolExecutor
        rows: list[Row] = []
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for got in ex.map(_ensemble_one_set,
                              [(spec, args) for spec in args.sets]):
                rows.extend(got)
        return rows
    return _ensemble_sets(args.sets, args)


def _ensemble_one_set(payload):
    spec, args = payload
    return _ensemble_sets([spec], args)


def _ensemble_sets(specs, args) -> list[Row]:
    from analysis.image_filters import SpatialDetailAnalyzer
    rng = np.random.default_rng(args.seed)
    rows: list[Row] = []
    mask_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    for spec in specs:
        paths = _list_subs(spec)
        if len(paths) < 4:
            print(f"[skip] {spec}: only {len(paths)} subs")
            continue
        print(f"\n=== {spec}: {len(paths)} subs ===", flush=True)
        cache = _load_cache(paths, args.crop)
        if not args.keep_outliers:
            cache, bad = _reject_outlier_frames(
                cache, paths, args.outlier_hi, args.outlier_lo)
            if bad:
                print(f"      rejected {len(bad)} of {len(paths)} frames")
        n = len(cache)

        # Never let a block fall below min_block_subs. Blocks are what Axiom
        # sees, and a 1-sub block is precisely the low-SNR single frame that
        # per-sub star removal was rejected for -- the model would be working
        # outside its trained regime and the --use-mtf stretch would be set
        # from a noise-dominated median. Small sets therefore get fewer, deeper
        # blocks rather than more, shallower ones.
        n_blocks = min(args.n_blocks, max(2, n // max(1, args.min_block_subs)))
        if n_blocks < 2:
            print(f"[skip] {spec}: too few frames for {args.n_blocks} blocks")
            del cache
            continue

        for blocking in range(args.blockings):
            t0 = time.perf_counter()
            blocks, block_size = _block_stacks(
                cache, n_blocks, rng, args.starless, f"{blocking}")
            print(f"   blocking {blocking}: {n_blocks} blocks of {block_size} "
                  f"subs [{time.perf_counter() - t0:.1f}s]", flush=True)
            for half in args.block_halves:
                depth = half * block_size
                pairs = _block_pairs(n_blocks, half, args.pairs_per_depth, rng)
                for pi, (ba, bb) in enumerate(pairs):
                    a = blocks[ba].mean(axis=0)
                    b = blocks[bb].mean(axis=0)
                    t1 = time.perf_counter()
                    res = SpatialDetailAnalyzer().analyze(
                        _mem_image(a, "A"), _mem_image(b, "B"),
                        make_figures=False)
                    rows.extend(_ensemble_rows(
                        spec, res, depth, blocking * 100 + pi, mask_metrics,
                        args.top_pcts, args.dilations))
                    print(f"      depth={depth:<4d} blk={blocking} pair={pi} "
                          f"[{time.perf_counter() - t1:.1f}s]", flush=True)
            del blocks
        del cache
    return rows


def _ensemble_rows(spec: str, res: dict, depth: int, repeat: int,
                   mask_metrics: list[str], top_pcts: list[float],
                   dilations: list[float]) -> list[Row]:
    """Bulk row per metric, plus the (top_pct x dilation) grid for a subset.

    The mask grid is restricted to `mask_metrics` because every combination
    needs its own `_combined_localmax_mask` call; the full 20-metric cross
    product would dominate the run for information the representative subset
    already carries.
    """
    from analysis.image_filters import SpatialDetailAnalyzer
    from core.models import (SECTION8_LOCALMAX_FOOTPRINT_MULT,
                             SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                             SECTION8_LOCALMAX_PRESMOOTH_FRACTION)
    sd = SpatialDetailAnalyzer()
    out: list[Row] = []

    # Section 8m acutance. These are *scalars* over the nebula mask, not
    # `panels` entries, which is why an earlier version of this sweep missed
    # them entirely -- METRIC_KEYS was built from the panels dict rather than
    # from what the report reports. They matter: 8m was the one section that
    # responded correctly and reproducibly to a real deconvolution sharpen
    # where the per-pixel families barely moved. Structurally they also
    # sidestep the whole unresolved top_pct question, because
    # `_masked_reduce(values, mask_neb_shared, ...)` takes no percentile.
    for fam, ka, kb in (("lap_var", "lap_var_a", "lap_var_b"),
                        ("grad_energy", "grad_energy_a", "grad_energy_b")):
        a_by_sigma = res.get(ka) or {}
        b_by_sigma = res.get(kb) or {}
        for sigma, va in a_by_sigma.items():
            vb = b_by_sigma.get(sigma)
            if va and vb and va > 0 and vb > 0:
                out.append(Row(
                    tier="null-ensemble", case_id=spec,
                    metric_key=f"{fam}_{sigma}", population=POP_NEB,
                    depth=depth, repeat=repeat,
                    log_ratio=float(np.log10(va / vb))))

    for key, panel in res.get("panels", {}).items():
        if key not in METRIC_KEYS or panel.get("diff") is None:
            continue
        diff, pa, pb = panel["diff"], panel.get("a"), panel.get("b")
        out.append(Row(tier="null-ensemble", case_id=spec, metric_key=key,
                       population=POP_BULK, depth=depth, repeat=repeat,
                       log_ratio=float(np.nanmean(diff))))
        if key not in mask_metrics or pa is None or pb is None:
            continue
        scale = _SCALE_OF.get(key)
        if not scale:
            continue
        h = min(pa.shape[0], pb.shape[0], diff.shape[0])
        w = min(pa.shape[1], pb.shape[1], diff.shape[1])
        aa, bb, dd = np.abs(pa[:h, :w]), np.abs(pb[:h, :w]), diff[:h, :w]
        footprint = max(3, int(round(SECTION8_LOCALMAX_FOOTPRINT_MULT * scale)) | 1)
        presmooth = max(0.5, SECTION8_LOCALMAX_PRESMOOTH_FRACTION * scale)
        for dil in dilations:
            region = max(1, int(round(dil * footprint)))
            for tp in top_pcts:
                m = sd._combined_localmax_mask(
                    aa, bb, footprint, SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                    region, presmooth, tp)
                if not m.any():
                    continue
                out.append(Row(
                    tier="null-ensemble", case_id=spec, metric_key=key,
                    population=POP_TOP, depth=depth, repeat=repeat,
                    top_pct=tp, dilation=dil,
                    log_ratio=float(np.nanmean(dd[m])),
                    note=f"{100.0 * m.mean():.1f}% of frame"))
    return out


def cmd_signal_grid(args) -> list[Row]:
    """Blur response over the same (top_pct x dilation) grid as null-ensemble.

    The missing half of the mask-parameter question. `null-ensemble` gives the
    floor for each combination; this gives the signal, and the combination
    worth using is the one maximising discriminability = signal / floor. Either
    number alone is misleading: a smaller top_pct lowers the floor (fewer,
    brighter, more signal-dominated pixels) but also samples less of the frame,
    so the two effects have to be weighed against each other rather than
    assumed.

    Also records each stack's pct_above_N sigma from SNRAnalyzer, to test
    whether an image can choose its own top_pct from a quantity the report
    already computes instead of inheriting a fixed constant.
    """
    from analysis.image_filters import SpatialDetailAnalyzer
    from analysis.snr_analyzer import SNRAnalyzer

    rng = np.random.default_rng(args.seed)
    rows: list[Row] = []
    mask_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    for spec in args.sets:
        paths = _list_subs(spec)
        if len(paths) < 4:
            print(f"[skip] {spec}: only {len(paths)} subs")
            continue
        print(f"\n=== {spec}: {len(paths)} subs ===", flush=True)
        cache = _load_cache(paths, args.crop)
        if not args.keep_outliers:
            cache, bad = _reject_outlier_frames(
                cache, paths, args.outlier_hi, args.outlier_lo)
            if bad:
                print(f"      rejected {len(bad)} of {len(paths)} frames")
        ref = _maybe_starless(cache.mean(axis=0), args.starless, "sgref")
        del cache

        snr = SNRAnalyzer().analyze(_mem_image(ref, "ref"), make_figures=False)
        pcts = {n: snr.get(f"pct_above_{n}") for n in (3, 5, 10, 20)}
        print("      pct of pixels above N sigma: "
              + "  ".join(f"{n}s={pcts[n]:.2f}%" for n in (3, 5, 10, 20)
                          if pcts[n] is not None), flush=True)
        for n, v in pcts.items():
            if v is not None:
                rows.append(Row(tier="signal-grid", case_id=spec,
                                metric_key=f"pct_above_{n}", log_ratio=float(v),
                                note="SNRAnalyzer pixel fraction"))

        for sigma in args.blur:
            deg = _blur_preserving_noise(ref, float(sigma), rng)
            t0 = time.perf_counter()
            res = SpatialDetailAnalyzer().analyze(
                _mem_image(ref, "A"), _mem_image(deg, "B"), make_figures=False)
            for r in _ensemble_rows(spec, res, 0, 0, mask_metrics,
                                    args.top_pcts, args.dilations):
                r.tier = "signal-grid"
                r.factor_sharpness = f"blur sigma={sigma}"
                rows.append(r)
            print(f"   blur sigma={sigma} [{time.perf_counter() - t0:.1f}s]",
                  flush=True)
    return rows


def cmd_add_null_floors(args) -> list[Row]:
    """Merge measured null floors from an m31-null CSV into the calibration.

    Monotonicity, sign and dynamic range together are not enough to decide
    whether a metric can carry a label -- local entropy passes all three with
    the largest range of any family and is still unusable, because two stacks of
    the *same* sub-frames differ on it by more than any two real filters do.
    The floor is the only screen that catches that, and it has to be measured
    rather than reasoned about.

    Floors are target- and depth-specific (these come from M31 at the stack
    depth you ran), so they are indicative for other targets rather than exact.
    A per-report floor would be better and needs that target's own sub-frames.
    """
    cal_path = _REPO / "resources" / "metric_calibration.json"
    if not cal_path.exists():
        print("no calibration yet; run blur-grid or recalibrate first")
        return []
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"{csv_path} not found; run m31-null first")
        return []

    # Take the MAX across every null pair present, not the last or the mean.
    #
    # A floor measured from a single disjoint pair is badly under-determined:
    # across four stack depths the same metric's floor moved non-monotonically
    # by up to 50x (std_3px top5: 0.0005 at N=1, 0.0246 at N=3, 0.0003 at N=10,
    # 0.0010 at N=37). Two images at the same depth have the same noise level,
    # so this statistic measures the reproducibility of the *mean log-ratio*
    # rather than the noise itself, and one realisation of it is dominated by
    # which particular frames landed in each half -- the N=3 spike is most
    # likely a single poor sub-frame.
    #
    # Max is the conservative reduction: it errs toward calling real
    # differences "no visible difference" rather than calling sampling scatter
    # "material". Pass a CSV concatenating several disjoint pairs (ideally
    # several at the same depth) for a better-determined bound.
    floors: dict[str, float] = {}
    n_rows = 0
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["tier"] == "m31-null" and row["log_ratio"]:
                key = _cal_key(row["metric_key"], row.get("population", POP_BULK))
                floors[key] = max(floors.get(key, 0.0), abs(float(row["log_ratio"])))
                n_rows += 1
    print(f"read {n_rows} null measurements covering {len(floors)} keys")

    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    applied, unusable = 0, []
    for key, entry in payload.get("metrics", {}).items():
        if key in floors:
            entry["null_floor"] = floors[key]
            applied += 1
            # If the floor swallows the metric's whole calibrated range, it can
            # never report anything but "no visible difference".
            top = entry.get("abs_log_ratio", [0])[-1]
            if floors[key] >= top:
                unusable.append(key)
    payload["null_floor_source"] = f"{csv_path.name}"
    cal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"applied {applied} null floors -> {cal_path}")
    for key in sorted(unusable):
        print(f"  floor exceeds full calibrated range: {key}")
    missing = sorted(set(payload.get("metrics", {})) - set(floors))
    for key in missing:
        print(f"  no floor measured: {key}")
    return []


def cmd_recalibrate(args) -> list[Row]:
    """Rebuild the calibration from an existing blur-grid CSV.

    The grid is ~18 minutes; the screen that decides which metrics survive it is
    a few lines. Separating them means a change to the screening rules can be
    re-applied to measured data instead of re-measuring it.
    """
    path = Path(args.csv)
    if not path.exists():
        print(f"{path} not found; run blur-grid first")
        return []
    want_noise = f"noise={float(args.noise)}"
    curve: dict[str, list[tuple[float, float, float]]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row["tier"] != "blur-grid" or not row["log_ratio"]
                    or not row["fwhm_pct"] or row.get("factor_snr") != want_noise):
                continue
            try:
                sigma = float(row["factor_sharpness"].split("=")[1])
            except (IndexError, ValueError):
                continue
            key = _cal_key(row["metric_key"], row.get("population", POP_BULK))
            curve.setdefault(key, []).append(
                (sigma, float(row["log_ratio"]), abs(float(row["fwhm_pct"]))))
    if not curve:
        print(f"no usable rows at {want_noise}")
        return []
    _write_calibration(curve, args.out, f"{path.name} @ {want_noise}")
    return []


# ── health: monotonicity / sign screen ────────────────────────────────────────

def cmd_health(args) -> list[Row]:
    """Screen a blur-grid CSV: does each metric move monotonically, and in the
    right direction, as a known blur increases?

    A metric that inverts sign partway through the sweep cannot carry a
    practical label no matter how significant its p-value is. Section 8m's
    acutance does exactly this on real data (correct at sigma=1.5, reversed at
    sigma>=3), which is what this screen exists to catch automatically.
    """
    path = Path(args.csv)
    if not path.exists():
        print(f"{path} not found; run blur-grid first")
        return []
    # One noise level at a time. Pooling them all would put four different
    # log-ratios at each sigma and make every metric look non-monotone by
    # construction -- monotonicity is a statement about the blur axis at a
    # fixed noise level, not about the whole grid.
    want_noise = f"noise={float(args.noise)}"
    by: dict[str, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["tier"] != "blur-grid" or not row["log_ratio"]:
                continue
            if row.get("factor_snr") != want_noise:
                continue
            try:
                sigma = float(row["factor_sharpness"].split("=")[1])
            except (IndexError, ValueError):
                continue
            by.setdefault(row["metric_key"], []).append(
                (sigma, float(row["log_ratio"])))
    if not by:
        print(f"no rows at {want_noise}; try --noise with one of the grid's levels")
        return []
    print(f"screening at {want_noise}\n")

    print(f"{'metric':16s}{'monotone':>10s}{'sign ok':>9s}{'range':>10s}  verdict")
    for key in sorted(by):
        pts = sorted(by[key])
        vals = [v for _, v in pts]
        diffs = np.diff(vals)
        monotone = bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))
        # Blurring A's counterpart lowers B's detail, so log10(|A|/|B|) must rise.
        sign_ok = all(v >= -1e-9 for v in vals)
        rng_ = max(vals) - min(vals)
        ok = monotone and sign_ok
        print(f"{key:16s}{str(monotone):>10s}{str(sign_ok):>9s}{rng_:10.3f}"
              f"  {'usable' if ok else 'DISQUALIFIED'}")
    return []


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path, default=_REPO / "sweep_out")
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="label the six existing reports")
    v.add_argument("--quick", type=int, default=0,
                   help="only the first N metrics per case")
    v.set_defaults(fn=cmd_validate, stem="validate")

    f = sub.add_parser("m31-factorial", help="real 2x2 sharpness x SNR")
    f.add_argument("--crop", type=int, default=1500)
    f.set_defaults(fn=cmd_m31_factorial, stem="m31_factorial")

    n = sub.add_parser("m31-null", help="two disjoint stacks: known-null pair")
    n.add_argument("--n", type=int, default=37)
    n.add_argument("--crop", type=int, default=1500)
    n.set_defaults(fn=cmd_m31_null, stem="m31_null")

    b = sub.add_parser("blur-grid", help="calibration curves")
    b.add_argument("--image", type=str, default=None)
    b.add_argument("--crop", type=int, default=1500)
    b.add_argument("--blur", nargs="+",
                   default=[0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0])
    b.add_argument("--noise", nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    b.add_argument("--plain-blur", action="store_true",
                   help="blur without restoring the noise it removes -- "
                        "matches a post-stack blur, but conflates detail "
                        "loss with noise reduction (snr_global RISES)")
    b.set_defaults(fn=cmd_blur_grid, stem="blur_grid")

    ne = sub.add_parser("null-ensemble",
                        help="repeated disjoint-partition null floors + mask sweep")
    ne.add_argument("--sets", nargs="+", required=True,
                    help="'Target/Filter' specs under AstroLabTestData/RawFiles")
    ne.add_argument("--crop", type=int, default=1024)
    ne.add_argument("--n-blocks", type=int, default=10,
                    help="disjoint blocks per blocking; star removal runs once "
                         "per block, so this sets the Axiom call count")
    ne.add_argument("--blockings", type=int, default=2,
                    help="independent re-blockings; pairs within one blocking "
                         "share blocks, so >1 keeps the floor's spread honest")
    ne.add_argument("--block-halves", nargs="+", type=int, default=[1, 2, 5],
                    help="blocks per side (depth = half * block_size)")
    ne.add_argument("--pairs-per-depth", type=int, default=5)
    ne.add_argument("--jobs", type=int, default=1,
                    help="sets processed in parallel (2-3 is the sweet spot: "
                         "Axiom serialises on the GPU, analysis does not)")
    ne.add_argument("--min-block-subs", type=int, default=3,
                    help="floor on subs per block, so Axiom never sees a "
                         "near-single-sub stack")
    ne.add_argument("--top-pcts", nargs="+", type=float,
                    default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    ne.add_argument("--dilations", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ne.add_argument("--starless", action="store_true",
                    help="run SyQon Axiom on each stack first -- the image "
                         "class Section 8 actually analyses")
    ne.add_argument("--keep-outliers", action="store_true",
                    help="skip frame-quality rejection (to reproduce its effect)")
    ne.add_argument("--outlier-hi", type=float, default=1.35)
    ne.add_argument("--outlier-lo", type=float, default=0.70)
    ne.add_argument("--metrics", type=str,
                    default="std_3px,log_1.5,wavelet_2,localgrad_1.5,loclap_1.5",
                    help="metrics carried through the mask grid")
    ne.set_defaults(fn=cmd_null_ensemble, stem="null_ensemble")

    sg = sub.add_parser("signal-grid",
                        help="blur response over the top_pct x dilation grid")
    sg.add_argument("--sets", nargs="+", required=True)
    sg.add_argument("--crop", type=int, default=1024)
    sg.add_argument("--blur", nargs="+", type=float, default=[0.5, 1.0])
    sg.add_argument("--top-pcts", nargs="+", type=float,
                    default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    sg.add_argument("--dilations", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    sg.add_argument("--starless", action="store_true",
                    help="run SyQon Axiom on each stack first -- the image "
                         "class Section 8 actually analyses")
    sg.add_argument("--keep-outliers", action="store_true")
    sg.add_argument("--outlier-hi", type=float, default=1.35)
    sg.add_argument("--outlier-lo", type=float, default=0.70)
    sg.add_argument("--metrics", type=str,
                    default="std_3px,log_1.5,wavelet_2,localgrad_1.5,loclap_1.5")
    sg.set_defaults(fn=cmd_signal_grid, stem="signal_grid")

    nf = sub.add_parser("add-null-floors",
                        help="merge measured null floors into the calibration")
    nf.add_argument("--csv", type=str,
                    default=str(_REPO / "sweep_out" / "m31_null.csv"))
    nf.set_defaults(fn=cmd_add_null_floors, stem="add_null_floors")

    rc = sub.add_parser("recalibrate", help="rebuild calibration from a grid CSV")
    rc.add_argument("--csv", type=str,
                    default=str(_REPO / "sweep_out" / "blur_grid.csv"))
    rc.add_argument("--noise", type=float, default=0.0)
    rc.set_defaults(fn=cmd_recalibrate, stem="recalibrate")

    h = sub.add_parser("health", help="monotonicity / sign screen")
    h.add_argument("--csv", type=str, default=str(_REPO / "sweep_out" / "blur_grid.csv"))
    h.add_argument("--noise", type=float, default=0.0,
                   help="which grid noise level to screen (one at a time)")
    h.set_defaults(fn=cmd_health, stem="health")

    args = p.parse_args(argv)
    t0 = time.perf_counter()
    rows = args.fn(args)
    if rows:
        _write(rows, args.out, args.stem)
    print(f"done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
