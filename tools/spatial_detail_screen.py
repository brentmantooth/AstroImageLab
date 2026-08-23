"""Spatial-detail metric screening sweep over synthetic, star-free targets.

Measurement half of the "Big Spatial Detail Calibration" synthetic screening
phase. Blurs known, star-free test targets (a Siemens star, real fractal-
noise images standing in for nebula-like diffuse structure, high-contrast
zebra stripes, and diffuse+sharp-accent composites) by a swept amount, adds
swept levels of camera-realistic shot noise, and measures every Section 8
spatial-detail metric (including the noise-normalised nrm_* variants and the
8k global-acutance scalars) via SpatialDetailAnalyzer.analyze(make_figures=
False). tools/spatial_detail_atlas.py turns the resulting CSV into the local
HTML artifact.

This is deliberately separate from tools/sensitivity_sweep.py and does NOT
write to resources/metric_calibration.json -- it is a diagnostic screening
pass over synthetic star-free targets, not a real-data calibration run. Real-
data validation is an explicit follow-on phase. See
synthetic/detail_targets.py's module docstring for why noise is swept as an
independent axis rather than compensated the way
sensitivity_sweep.py::_blur_preserving_noise compensates it.

Usage
-----
    python tools/spatial_detail_screen.py -o sweep_out/synthetic_detail
    python tools/spatial_detail_screen.py --targets siemens_mid --blur 1.0 --stack-depths 10
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.image_filters import SpatialDetailAnalyzer               # noqa: E402
from synthetic.detail_targets import (                                 # noqa: E402
    CANVAS_PX,
    apply_camera_noise, apply_gaussian_blur, build_composite_target,
    build_siemens_target, load_fractal_source, load_zebra,
)
from tools.sensitivity_sweep import (                                  # noqa: E402
    METRIC_KEYS, POP_BULK, POP_TOP, _SCALE_OF, _localmax_mask, _mem_image,
)

FRACTAL_DIR = _REPO / "AstroLabTestData" / "Fractal Source"

# 8k global-acutance families -- scalars over the shared nebula mask, not
# `panels` entries, so they must be enumerated separately (see CLAUDE.md's
# "Enumerating Section 8 metrics from the panels dict" pitfall).
ACUTANCE_FAMILIES = (("lap_var", "lap_var_a", "lap_var_b"),
                     ("grad_energy", "grad_energy_a", "grad_energy_b"))
POP_NEBULA = "nebula"

# Capped at 3px (was 8.0): heavier blur was swamping the smaller, more
# realistic variations this study actually cares about -- the dense sampling
# below 1px is where the interesting response-curve behavior lives anyway.
BLUR_FULL = [0.0, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
BLUR_REDUCED = [0.0, 0.3, 0.75, 1.5, 3.0]
# No noiseless (None) case anymore -- a perfectly zero-noise image behaves
# atypically (see tools/spatial_detail_atlas.py's build_scorecard_at) and
# doesn't represent a real astro image, which always has some noise floor.
# depth=50 anchors a genuinely "well-stacked" case; depth=1 is the noisiest
# realistic single-sub case, present on every target (not just primary) so
# there's always a real-noise anchor to screen against.
STACK_DEPTHS_FULL: list[int] = [1, 4, 10, 25, 50]
STACK_DEPTHS_REDUCED: list[int] = [1, 10, 25]

# Peak ADU levels, well under a ZWO ASI2600MM Pro's ~51000 ADU full well.
# Lowered against the brighter PEDESTAL_ADU (synthetic/detail_targets.py) so
# the nebula-like targets sit at a more realistic contrast against sky --
# roughly 3-5x peak-over-pedestal rather than the original ~50x. ZEBRA_PEAK
# is deliberately left unchanged: zebra exists specifically as the stark
# high-contrast reference the low-contrast targets are compared against.
SIEMENS_PEAK = {"siemens_low": 1000.0, "siemens_mid": 3000.0, "siemens_high": 8000.0}
FRACTAL_PEAK = 8000.0
ZEBRA_PEAK = 15000.0
COMPOSITE_BASE_PEAK = 3000.0
COMPOSITE_ACCENT_PEAK = 8000.0

FRACTAL_FILES = {
    "fractal_00018": "BigImage_00018.png",
    "fractal_00045": "BigImage_00045.png",
    "fractal_00053": "BigImage_00053.png",
    "fractal_00078": "BigImage_00078.png",
}

PRIMARY_TARGETS = ("siemens_mid", "fractal_00018", "zebra",
                   "composite_sparse", "composite_dense")
SECONDARY_TARGETS = ("siemens_low", "siemens_high",
                     "fractal_00045", "fractal_00053", "fractal_00078")
ALL_TARGETS = PRIMARY_TARGETS + SECONDARY_TARGETS


def _stack_label(depth: int | None) -> str:
    return "noise=0.0" if depth is None else f"stack={depth}"


def build_target(target_id: str, canvas_px: int = CANVAS_PX) -> np.ndarray:
    """Build one target's clean (unblurred, noiseless) array."""
    if target_id in SIEMENS_PEAK:
        return build_siemens_target(SIEMENS_PEAK[target_id], canvas_px=canvas_px)
    if target_id in FRACTAL_FILES:
        return load_fractal_source(FRACTAL_DIR / FRACTAL_FILES[target_id],
                                   FRACTAL_PEAK, canvas_px=canvas_px)
    if target_id == "zebra":
        return load_zebra(FRACTAL_DIR / "Zebra.png", ZEBRA_PEAK, canvas_px=canvas_px)
    if target_id in ("composite_sparse", "composite_dense"):
        base = load_fractal_source(FRACTAL_DIR / FRACTAL_FILES["fractal_00018"],
                                   COMPOSITE_BASE_PEAK, canvas_px=canvas_px)
        n_accents = 3 if target_id == "composite_sparse" else 8
        return build_composite_target(base, n_accents, COMPOSITE_ACCENT_PEAK,
                                      seed=n_accents)
    raise ValueError(f"unknown target {target_id!r}; known: {ALL_TARGETS}")


@dataclass
class DetailRow:
    """One (target, blur, noise, metric, population) measurement.

    Superset of tools/sensitivity_sweep.py's Row -- same field names/order
    (so a plain column-by-name CSV reader is unaffected), plus
    p_value/cliffs_delta/n_px/pct_area, which Row has no slot for and that
    file must not be modified to add for this diagnostic-only study.
    """
    tier: str = "synthetic-detail"
    case_id: str = ""
    factor_sharpness: str = ""
    factor_snr: str = ""
    metric_key: str = ""
    population: str = POP_BULK
    log_ratio: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    se: float | None = None
    block_px: int | None = None
    n_blocks: int | None = None
    converged: bool | None = None
    naive_se_understatement: float | None = None
    fwhm_pct: float | None = None    # always None -- no PSF-fittable stars in any target
    label: str | None = None         # always None -- no Calibration loaded, by design
    top_pct: float | None = None
    dilation: float | None = None
    depth: int | None = None         # equivalent stack depth; None = noiseless
    repeat: int | None = None        # unused; kept for schema symmetry with Row
    note: str = ""
    p_value: float | None = None
    cliffs_delta: float | None = None
    n_px: int | None = None
    pct_area: float | None = None
    value_a: float | None = None     # raw mean magnitude, image A (fixed sigma=0 reference)
    value_b: float | None = None     # raw mean magnitude, image B (this row's blur sigma)


def _write_detail_rows(rows: list[DetailRow], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(DetailRow.__dataclass_fields__))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"\nwrote {len(rows)} rows -> {path}")
    return path


def _extract_rows(target_id: str, sigma: float, depth: int | None,
                  res: dict) -> list[DetailRow]:
    """Every row for one analyze() call."""
    factors = dict(case_id=target_id, factor_sharpness=f"blur sigma={sigma}",
                   factor_snr=_stack_label(depth), depth=depth)
    rows: list[DetailRow] = []
    panels = res.get("panels", {})

    for key in METRIC_KEYS:
        panel = panels.get(key)
        if panel is not None and panel.get("diff") is not None:
            # Bulk: plain mean of the per-pixel log-ratio, matching
            # sensitivity_sweep.py::_ensemble_rows's cheaper convention for
            # this population -- CLAUDE.md's own convention is that top5,
            # not bulk, carries the real signal, so a second expensive
            # block-bootstrap pass here would double per-case cost for the
            # population that matters less.
            rows.append(DetailRow(metric_key=key, population=POP_BULK,
                                  log_ratio=float(np.nanmean(panel["diff"])),
                                  value_a=float(np.nanmean(panel["a"])),
                                  value_b=float(np.nanmean(panel["b"])),
                                  **factors))

        # Top5 / local-maxima: free, already computed inside analyze() --
        # read it rather than re-deriving the mask and re-bootstrapping.
        entry = res.get("localmax", {}).get(key)
        if entry is not None and entry.get("n_px"):
            rows.append(DetailRow(
                metric_key=key, population=POP_TOP,
                log_ratio=entry["log_ratio_mean"],
                ci_lo=entry["ci_lo"], ci_hi=entry["ci_hi"], se=entry["ci_se"],
                block_px=entry["ci_block_px"], n_blocks=entry["ci_n_blocks"],
                converged=entry["ci_converged"],
                naive_se_understatement=entry["se_understatement"],
                p_value=entry["p_value"], cliffs_delta=entry["cliffs_delta"],
                n_px=entry["n_px"], pct_area=entry["pct_area"],
                value_a=entry["mean_a"], value_b=entry["mean_b"],
                **factors))

        # Noise-normalised variant. At the noiseless depth this is NOT
        # reliably absent the way a real, perfectly flat sky would make it:
        # the clean synthetic template itself still carries smooth
        # structure (the Siemens taper, fractal texture) into the pixels
        # classified as background, so _nc_score's noise floor there is
        # small but not exactly zero. Verified against the smoke-test CSV
        # rather than assumed -- read these noiseless-depth nrm_* rows as
        # "normalised against the template's own residual structure", not
        # "normalised against camera noise".
        nrm_panel = panels.get(f"nrm_{key}")
        if nrm_panel is not None and nrm_panel.get("a") is not None:
            nrm_diff = SpatialDetailAnalyzer._log_ratio_map(nrm_panel["a"], nrm_panel["b"])
            rows.append(DetailRow(metric_key=f"nrm_{key}", population=POP_BULK,
                                  log_ratio=float(np.nanmean(nrm_diff)),
                                  value_a=float(np.nanmean(nrm_panel["a"])),
                                  value_b=float(np.nanmean(nrm_panel["b"])),
                                  **factors))

            # Noise-normalised, top5: the SAME combined-localmax mask the
            # raw top5 row above already uses (derived from the RAW abs_a/
            # abs_b -- deliberately not recomputed on the nrm_ arrays), so
            # this is the identical pixel set as "top5" for this key, just
            # with the noise-normalised value averaged within it instead of
            # the raw one. No bootstrap CI here (unlike the free one on the
            # raw top5 row from res["localmax"]) -- computing one would be a
            # genuine extra block_bootstrap_ci call, not a free read, so
            # this stays a point estimate like the nrm bulk row above.
            scale = _SCALE_OF.get(key)
            if panel is not None and panel.get("a") is not None and scale:
                mask = _localmax_mask(np.abs(panel["a"]), np.abs(panel["b"]), scale)
                h = min(nrm_diff.shape[0], mask.shape[0])
                w = min(nrm_diff.shape[1], mask.shape[1])
                m = mask[:h, :w]
                if m.any():
                    masked_diff = nrm_diff[:h, :w][m]
                    nrm_a_m = np.abs(nrm_panel["a"])[:h, :w][m]
                    nrm_b_m = np.abs(nrm_panel["b"])[:h, :w][m]
                    rows.append(DetailRow(
                        metric_key=f"nrm_{key}", population=POP_TOP,
                        log_ratio=float(np.mean(masked_diff)),
                        value_a=float(np.mean(nrm_a_m)), value_b=float(np.mean(nrm_b_m)),
                        n_px=int(m.sum()), pct_area=100.0 * float(m.sum()) / m.size,
                        **factors))

    # 8k global acutance scalars, recomputed from the raw a/b dicts rather
    # than trusting the ratio dict's own sign convention -- mirrors
    # sensitivity_sweep.py::_ensemble_rows's identical precedent.
    for fam, ka, kb in ACUTANCE_FAMILIES:
        a_by_sigma, b_by_sigma = res.get(ka) or {}, res.get(kb) or {}
        for s, va in a_by_sigma.items():
            vb = b_by_sigma.get(s)
            if va and vb and va > 0 and vb > 0:
                rows.append(DetailRow(metric_key=f"{fam}_{s}", population=POP_NEBULA,
                                      log_ratio=float(np.log10(va / vb)),
                                      value_a=float(va), value_b=float(vb),
                                      **factors))
    return rows


def run_sweep(target_ids, blur_sigmas, stack_depths, seed: int = 0,
             canvas_px: int = CANVAS_PX, verbose: bool = True) -> list[DetailRow]:
    """Full (target x depth x blur) grid. See the module docstring for the
    A/B pairing design: A is the clean signal at a given stack depth,
    rebuilt once per depth and reused across the whole blur ladder; B is an
    independent noise draw at the same depth, over the (possibly) blurred
    clean signal.
    """
    rows: list[DetailRow] = []
    rng = np.random.default_rng(seed)
    for target_id in target_ids:
        clean = build_target(target_id, canvas_px=canvas_px)
        nebula_frac_seen: list[float] = []
        for depth in stack_depths:
            image_a = _mem_image(apply_camera_noise(clean, rng, depth), "A")
            for sigma in blur_sigmas:
                t0 = time.perf_counter()
                blurred = clean if sigma == 0 else apply_gaussian_blur(clean, sigma)
                image_b = _mem_image(apply_camera_noise(blurred, rng, depth), "B")
                res = SpatialDetailAnalyzer().analyze(image_a, image_b, make_figures=False)
                rows.extend(_extract_rows(target_id, sigma, depth, res))
                nebula_frac = res.get("nc_shared_nebula_pixels", 0) / (canvas_px ** 2)
                nebula_frac_seen.append(nebula_frac)
                if verbose:
                    print(f"  {target_id:<18} depth={str(depth):<5} sigma={sigma:<5} "
                          f"nebula_frac={nebula_frac:.2f} "
                          f"[{time.perf_counter() - t0:.2f}s]", flush=True)
        if nebula_frac_seen and max(nebula_frac_seen) > 0.9:
            print(f"  [warn] {target_id}: nebula-mask coverage reached "
                  f"{max(nebula_frac_seen):.0%} at some grid point "
                  f"(SOURCEMASK_MAX_COVERAGE risk -- check the atlas for this target)")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path,
                   default=_REPO / "sweep_out" / "synthetic_detail")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets", nargs="+", default=None, choices=ALL_TARGETS,
                   help="target ids to run (default: all primary + secondary)")
    p.add_argument("--blur", nargs="+", type=float, default=None,
                   help="override the blur ladder for every selected target")
    p.add_argument("--stack-depths", nargs="+", default=None,
                   help="override the stack-depth ladder ('none' for noiseless)")
    p.add_argument("--canvas", type=int, default=CANVAS_PX)
    args = p.parse_args(argv)

    def _parse_depth(s):
        return None if str(s).lower() in ("none", "noiseless") else int(s)

    targets = args.targets if args.targets else list(ALL_TARGETS)

    rows: list[DetailRow] = []
    t0 = time.perf_counter()
    for target_id in targets:
        is_primary = target_id in PRIMARY_TARGETS
        blur = args.blur if args.blur is not None else (
            BLUR_FULL if is_primary else BLUR_REDUCED)
        depths = ([_parse_depth(d) for d in args.stack_depths]
                 if args.stack_depths is not None
                 else (STACK_DEPTHS_FULL if is_primary else STACK_DEPTHS_REDUCED))
        print(f"\n=== {target_id} ({'primary' if is_primary else 'secondary'}) ===",
              flush=True)
        rows.extend(run_sweep([target_id], blur, depths, seed=args.seed,
                              canvas_px=args.canvas))

    # Built-in correctness check: sigma=0 self-comparisons should sit near
    # log_ratio=0 for every metric at every depth (A and B are the same
    # clean signal, independent noise draws at the same depth) -- catches a
    # pairing/extraction bug immediately, no pytest needed.
    zero_sigma = [r for r in rows if r.factor_sharpness == "blur sigma=0.0"
                 and r.log_ratio is not None]
    if zero_sigma:
        worst = max(zero_sigma, key=lambda r: abs(r.log_ratio))
        print(f"\nsigma=0 self-comparison check: worst |log_ratio| = "
              f"{abs(worst.log_ratio):.4f} ({worst.metric_key}, {worst.case_id}, "
              f"{worst.factor_snr})")
        if abs(worst.log_ratio) > 0.2:
            print("  [warn] larger than expected for a same-signal comparison -- "
                  "check the pairing/extraction logic before trusting the grid")

    if rows:
        _write_detail_rows(rows, args.out, "spatial_detail_screen")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s, {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
