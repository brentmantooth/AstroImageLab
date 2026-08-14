"""Source-masked background estimation — the report's background model.

Adopted by `AstroImage.estimate_background()` as the sky model actually
subtracted (Section 3e), with Section 3g carrying the audit trail: the mask,
the per-tier thresholds, the surviving mesh cells, and the before/after
comparison against the plain unmasked estimate. It runs as a diagnostic only
when the user switches source masking off.

photutils' `Background2D` sigma-clips each mesh cell independently, which
assumes every cell is sky-dominated. That assumption fails when extended
nebulosity covers a large fraction of the frame: the clip converges on the
nebula level rather than the sky level and the background is biased upward
(measured at +1.46 sigma_sky on a synthetic frame with 54% nebula coverage).

Four stages, each producing an auditable intermediate:

  1. Sky scaffold      unmasked mesh -> ONE-SIDED robust polynomial fit
  2. Detection         multi-scale detect_sources on (data - scaffold)
  3. Classification    extent-based point/extended split -> differential dilation
  4. Final estimate    masked mesh -> order-capped polynomial fit + fallback ladder

Two design points are load-bearing and non-obvious, both established by
measurement against a known true sky surface (see CLAUDE.md):

  * Stage 1's rejection is ONE-SIDED. Nebula only ever adds flux, so the
    contamination is single-signed; a symmetric sigma-clip cannot remove a bias
    carried by >50% of the cells, and additionally throws away genuine low-sky
    cells. Rejecting only cells ABOVE the running fit is what makes the
    scaffold gradient-robust without needing any segmentation.

  * Stage 4 fits a PARAMETRIC surface to the surviving cells rather than
    letting Background2D interpolate across the excluded ones. With a
    high-coverage mask most cells go empty, and zoom-interpolating that sparse
    remainder produced a worse worst-case error than the original bias.
    Polynomial order is additionally capped by surviving-cell count, because
    BIC alone will choose a quadric on a handful of cells and extrapolate
    wildly across the masked region.

This module is deliberately array-in / array-out (no `AstroImage`), matching
analysis/background_fit.py. It must not import core.astro_image -- that is the
direction analysis/background_mask.py already imports, and reversing it would
create a cycle once core/ calls into this module.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from photutils.background import (Background2D, MADStdBackgroundRMS,
                                  SExtractorBackground)
from photutils.segmentation import SourceCatalog, detect_sources
from scipy.ndimage import binary_dilation, binary_fill_holes, label
from scipy.signal import fftconvolve

from analysis.background_fit import evaluate_surface_at, fit_background_surface
from core.models import (SOURCEMASK_EXCLUDE_PERCENTILE,
                         SOURCEMASK_EXT_DILATION_SIGMA_MULT,
                         SOURCEMASK_EXT_KERNEL_SIGMAS, SOURCEMASK_EXT_NPIX_MULT,
                         SOURCEMASK_EXT_NSIGMA, SOURCEMASK_EXTENDED_RADIUS_MULT,
                         SOURCEMASK_MASKED_FILTER_SIZE, SOURCEMASK_MAX_COVERAGE,
                         SOURCEMASK_MAX_HOLE_PX, SOURCEMASK_MIN_CELL_UNMASKED_FRAC,
                         SOURCEMASK_AGREEMENT_GRID,
                         SOURCEMASK_DEMOTE_ON_DIVERGENCE,
                         SOURCEMASK_MAX_EXTRAPOLATION_RATIO,
                         SOURCEMASK_MAX_MODEL_DIVERGENCE_SIGMA,
                         SOURCEMASK_MIN_CELLS_CONSTANT, SOURCEMASK_MIN_CELLS_PLANE,
                         SOURCEMASK_MIN_CELLS_QUADRIC,
                         SOURCEMASK_MIN_SURFACE_BRIGHTNESS_SIGMA,
                         SOURCEMASK_N_PASSES, SOURCEMASK_QUADRIC_CELL_FRACTION,
                         SOURCEMASK_POINT_NPIXELS, SOURCEMASK_POINT_NSIGMA,
                         SOURCEMASK_SCAFFOLD_MAX_ITERS,
                         SOURCEMASK_SCAFFOLD_REJECT_NSIGMA,
                         SOURCEMASK_STAR_DILATION_FWHM_MULT)

# Background2D's own defaults, mirrored here so the masked pass clips raw pixels
# identically to the unmasked pass it is being compared against. Imported from
# core.astro_image would be the single-source-of-truth ideal, but that is the
# import direction this module must not take (see module docstring), so the two
# are kept in step by the comment rather than by the import.
_SIGCLIP_SIGMA = 3.0
_SIGCLIP_MAXITERS = 10

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493   # 2*sqrt(2*ln 2)


@dataclass
class SourceMaskResult:
    """Stage 2+3 output: which pixels are source, and the audit trail for why."""

    mask: np.ndarray            # bool, True = source (excluded from background)
    point_mask: np.ndarray      # bool, point-source class only
    extended_mask: np.ndarray   # bool, extended-structure class only
    coverage: float             # fraction of the frame masked
    sky_sigma: float            # per-pixel sky noise the thresholds were built from
    tiers: list[dict] = field(default_factory=list)      # per-tier threshold record
    segments: list[dict] = field(default_factory=list)   # per-segment classification record
    # Pixels the user excluded by hand. Kept as its own class rather than folded
    # into extended_mask so the report can say "you excluded this" instead of
    # crediting the detector with the user's judgement.
    user_mask: np.ndarray | None = None

    @property
    def n_point(self) -> int:
        return sum(1 for s in self.segments if s["classification"] == "point")

    @property
    def n_extended(self) -> int:
        return sum(1 for s in self.segments if s["classification"] == "extended")

    @property
    def user_coverage(self) -> float:
        """Fraction of the frame the user excluded by hand."""
        return 0.0 if self.user_mask is None else float(self.user_mask.mean())


@dataclass
class MaskedBackgroundResult:
    """Stage 4 output: the source-masked background estimate plus its provenance."""

    surface: np.ndarray | None       # float32 (h, w) background model, or None if aborted
    rms_median: float | None
    background_median: float | None
    fit: dict | None                 # fit_background_surface() result
    n_cells: int                     # surviving (unmasked) mesh cells
    n_cells_total: int
    order_cap: int | None
    coverage: float
    fallback_reason: str | None
    source_mask: SourceMaskResult | None
    scaffold: dict | None            # stage-1 fit, kept for the report's audit box
    # Full-resolution noise floor from the masked Background2D pass. Carried so a
    # caller adopting this estimate gets the masked RMS map too -- removing the
    # nebulosity lowers the measured noise as well as the level, and using the
    # masked background with the unmasked RMS would mix the two.
    rms_map: np.ndarray | None = None
    # plane_quadric_agreement() record, or None when no quadric was on the table.
    agreement: dict | None = None

    @property
    def ok(self) -> bool:
        return self.surface is not None


# ----------------------------------------------------------------------
# Stage 1 — sky scaffold
# ----------------------------------------------------------------------

def _mesh_cell_centres(shape: tuple[int, int], box_size: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Native-pixel (x, y) centres of a Background2D mesh, same reconstruction
    fit_background_surface() uses internally."""
    n_rows, n_cols = shape
    yy, xx = np.mgrid[0:n_rows, 0:n_cols]
    return ((xx.astype(np.float64) + 0.5) * box_size,
            (yy.astype(np.float64) + 0.5) * box_size)


def fit_sky_scaffold(mesh: np.ndarray, rms_mesh: np.ndarray, box_size: int,
                     image_shape: tuple[int, int], *,
                     reject_nsigma: float = SOURCEMASK_SCAFFOLD_REJECT_NSIGMA,
                     max_iters: int = SOURCEMASK_SCAFFOLD_MAX_ITERS,
                     reject_order: int = 1, final_max_order: int = 2,
                     excluded_cells: np.ndarray | None = None) -> dict:
    """One-sided robust polynomial fit to an *unmasked* Background2D mesh.

    Returns a fit_background_surface() dict with two extra keys: `kept_cells`
    (bool array over the mesh) and `n_iters`.

    The rejection is deliberately asymmetric -- a cell is dropped only when it
    sits `reject_nsigma` scale units ABOVE the running fit, never when it sits
    below. Cells below the fit are genuine sky (or downward noise) and carry
    the information the fit exists to recover; only upward excursions can be
    source contamination, since nebulosity strictly adds flux.

    **Two-stage by design.** The rejection loop runs at `reject_order` (a
    plane), then the surviving cells are re-fitted with BIC free to choose up
    to `final_max_order` (a quadric). Neither order works alone:

      * iterating with a quadric lets the scaffold absorb broad nebulosity --
        the residual flattens, less is detected, and the pedestal ends up
        biased (heavy-nebula frame: median error 0.299 -> 1.053 sigma);
      * but a plane cannot represent vignetting, so on a vignetted frame the
        residual keeps a large bowl, the detector masks it as if it were
        source, and the estimate came out *worse than doing nothing* (0.267
        sigma against an unmasked baseline of 0.023, masking 53% of a frame
        with no nebula in it at all).

    Rejecting with a plane keeps the quadric from eating the nebula; fitting
    the survivors with BIC lets real curvature back in. Measured best-or-
    near-best on every ground-truth frame, where each single order was
    catastrophic on some frame.

    Note vignetting and a centred nebula are *genuinely degenerate* here -- both
    are radially symmetric with the same curvature sign -- so no threshold can
    separate them. That case is what the user-drawn exclusion regions exist for.

    `excluded_cells` is a boolean mesh-shaped array of cells the user excluded
    by hand; they are dropped before the first fit. This is the strongest place
    to apply that knowledge: a scaffold no longer bent by the nebula produces a
    flatter residual, which improves detection across the *whole* frame rather
    than only inside the region that was drawn.
    """
    finite = np.isfinite(mesh) & np.isfinite(rms_mesh) & (rms_mesh > 0)
    if excluded_cells is not None:
        finite = finite & ~np.asarray(excluded_cells, dtype=bool)
        if int(finite.sum()) < SOURCEMASK_MIN_CELLS_CONSTANT:
            # The drawn regions would starve the fit. Better to ignore them here
            # and let the one-sided rejection do what it can than to return a
            # surface fitted through two cells.
            finite = np.isfinite(mesh) & np.isfinite(rms_mesh) & (rms_mesh > 0)
    keep = finite.copy()
    cx, cy = _mesh_cell_centres(mesh.shape, box_size)

    fit: dict = {}
    n_iters = 0
    for n_iters in range(1, max_iters + 1):
        masked_mesh = np.where(keep, mesh, np.nan)
        masked_rms = np.where(keep, rms_mesh, np.nan)
        fit = fit_background_surface(masked_mesh, masked_rms, box_size,
                                     image_shape, max_order=reject_order)
        if fit.get("insufficient_data"):
            break
        resid = mesh - evaluate_surface_at(fit, cx, cy)
        kept_resid = resid[keep]
        if kept_resid.size < 2:
            break
        # Scale and anchor are both estimated from the LOWEST QUARTILE of the
        # residuals. Every wider estimator breaks down under the contamination
        # levels this module exists for:
        #   * a two-sided MAD is inflated by the very contamination it must
        #     detect (measured: 62 of 64 cells kept, scaffold biased +1.26 sigma);
        #   * a below-median MAD fails once contamination passes 50%, because
        #     the median itself moves into the contaminated population (at 72%
        #     contamination it rejected nothing at all).
        # The lowest quartile stays clean up to ~75% one-sided contamination,
        # since nebulosity only ever adds flux. q25/q05 of a normal are
        # -0.6745/-1.6449 sigma, which inverts to both quantities below.
        q05, q25 = np.percentile(kept_resid, [5.0, 25.0])
        sigma = float(q25 - q05) / 0.9704
        if not np.isfinite(sigma) or sigma <= 0:
            # Degenerate quartiles (many cells sharing one exact value). Fall
            # back to the two-sided MAD: it is the inflated estimator, but an
            # inflated scale only under-rejects, whereas bailing out here would
            # accept every contaminated cell.
            med = float(np.median(kept_resid))
            sigma = 1.4826 * float(np.median(np.abs(kept_resid - med)))
            anchor = med
        else:
            anchor = float(q25) + 0.6745 * sigma   # the uncontaminated median
        if not np.isfinite(sigma) or sigma <= 0:
            break   # residuals are genuinely constant; nothing to reject
        new_keep = finite & (resid < anchor + reject_nsigma * sigma)
        # Never reject below the floor the order needs, and stop once stable.
        if int(new_keep.sum()) < SOURCEMASK_MIN_CELLS_CONSTANT:
            break
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    # Second stage: refit the survivors with BIC free to add curvature. The
    # rejection above has already removed the contaminated cells, so a quadric
    # chosen here is responding to real structure (vignetting) rather than to
    # nebulosity it was allowed to swallow.
    if final_max_order > reject_order and not fit.get("insufficient_data"):
        refit = fit_background_surface(np.where(keep, mesh, np.nan),
                                       np.where(keep, rms_mesh, np.nan),
                                       box_size, image_shape,
                                       max_order=final_max_order)
        if not refit.get("insufficient_data"):
            fit = refit

    fit = dict(fit)
    fit["kept_cells"] = keep
    fit["n_iters"] = n_iters
    return fit


def _scaffold_sky_sigma(rms_mesh: np.ndarray, kept_cells: np.ndarray) -> float:
    """Per-pixel sky noise from the scaffold-accepted cells only.

    The rejected cells are the source-contaminated ones; their MADStd is
    inflated by real structure, so pooling them would raise every detection
    threshold in stage 2 and systematically under-detect.
    """
    vals = rms_mesh[kept_cells & np.isfinite(rms_mesh) & (rms_mesh > 0)]
    if vals.size == 0:
        vals = rms_mesh[np.isfinite(rms_mesh) & (rms_mesh > 0)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


# ----------------------------------------------------------------------
# Stage 2 — multi-scale detection
# ----------------------------------------------------------------------

def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Sum-normalised 2D Gaussian, truncated at 4 sigma (odd side length)."""
    radius = max(1, int(np.ceil(4.0 * sigma)))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    g1 = np.exp(-0.5 * (ax / sigma) ** 2)
    k = np.outer(g1, g1)
    return k / k.sum()


def _smoothed_sigma(sky_sigma: float, kernel_sigma: float) -> float:
    """Noise sigma after convolving white noise with a sum-normalised Gaussian.

    For a kernel normalised to sum 1, Var_out = Var_in * sum(k**2), and for a
    2D Gaussian sum(k**2) = 1/(4*pi*sigma**2) -- so the threshold applied to
    the smoothed image must use this, not the raw per-pixel sigma. Using
    sky_sigma directly would over-threshold by a factor of 2*sqrt(pi)*sigma_k
    (a factor of ~28 at sigma_k = 8 px) and detect essentially nothing.
    """
    return float(sky_sigma / np.sqrt(4.0 * np.pi * kernel_sigma ** 2))


def _decimate(arr: np.ndarray, step: int) -> np.ndarray:
    return arr if step <= 1 else arr[::step, ::step]


def _upsample_mask(mask: np.ndarray, step: int, shape: tuple[int, int]) -> np.ndarray:
    """Exact inverse of `arr[::step, ::step]` for a boolean mask.

    np.repeat on both axes reproduces the stride grid cell-for-cell; the
    trailing crop/pad handles a final partial block when the dimension isn't
    an exact multiple of step.
    """
    if step <= 1:
        return mask
    out = np.repeat(np.repeat(mask, step, axis=0), step, axis=1)
    h, w = shape
    if out.shape[0] < h or out.shape[1] < w:
        padded = np.zeros(shape, dtype=bool)
        padded[:out.shape[0], :out.shape[1]] = out[:h, :w]
        return padded
    return out[:h, :w]


def _detect_tier(residual: np.ndarray, kernel_sigma: float, n_sigma: float,
                 npixels: int, sky_sigma: float,
                 min_surface_brightness: float = SOURCEMASK_MIN_SURFACE_BRIGHTNESS_SIGMA
                 ) -> tuple[np.ndarray, dict]:
    """Convolve, threshold, and label one detection tier.

    Returns (bool detection mask, audit record). The kernel is sum-normalised
    so the convolved image stays in ADU and the threshold is directly
    interpretable; see _smoothed_sigma for why the threshold is not n*sky_sigma.

    The threshold is the LARGER of two bounds, and the audit record says which
    one won:

      * statistical -- n_sigma above the smoothed noise, i.e. "is this real?"
      * surface brightness -- min_surface_brightness x sky_sigma, i.e. "is this
        worth masking?"

    Statistical significance alone runs away at large kernels: integrating over
    a 16 px kernel drops the noise ~57x, so that tier triggered on structure at
    3.5% of sky noise -- real, but it biases the background by nothing while
    costing mesh cells the fit needs. Masking is not free, so the threshold has
    to answer the second question too.
    """
    statistical = n_sigma * _smoothed_sigma(sky_sigma, kernel_sigma)
    floor = max(0.0, float(min_surface_brightness)) * sky_sigma
    threshold = max(statistical, floor)

    def _record(n_segments: int, skipped: bool) -> dict:
        return {
            "kernel_sigma": float(kernel_sigma), "n_sigma": float(n_sigma),
            "threshold_adu": float(threshold), "npixels": int(npixels),
            "n_segments": int(n_segments), "skipped": bool(skipped),
            "threshold_statistical_adu": float(statistical),
            "threshold_floor_adu": float(floor),
            # Which bound is binding. Without this the report cannot explain
            # why a deep tier stopped detecting.
            "threshold_source": "surface brightness" if floor > statistical else "statistical",
        }

    kernel = _gaussian_kernel(kernel_sigma)
    if min(residual.shape) <= 1 or kernel.shape[0] > min(residual.shape):
        # Kernel larger than the frame: nothing meaningful to detect at this scale.
        return np.zeros(residual.shape, dtype=bool), _record(0, True)
    convolved = fftconvolve(residual.astype(np.float64), kernel, mode="same")
    segm = detect_sources(convolved, threshold, n_pixels=npixels)
    if segm is None:
        return np.zeros(residual.shape, dtype=bool), _record(0, False)
    rec = _record(segm.n_labels, False)
    rec["_segm"] = segm
    rec["_convolved"] = convolved
    return segm.data > 0, rec


# ----------------------------------------------------------------------
# Stage 3 — classification + morphology
# ----------------------------------------------------------------------

def _fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Fill enclosed gaps inside mask up to (max_hole_px)**2 area."""
    if max_hole_px <= 0 or not mask.any():
        return mask
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    labeled, n_holes = label(holes)
    if n_holes == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros(sizes.size, dtype=bool)
    keep[1:] = sizes[1:] <= max_hole_px * max_hole_px   # label 0 is background
    return mask | keep[labeled]


def _remove_small_objects(mask: np.ndarray, max_size_px: int) -> np.ndarray:
    """Strip isolated foreground specks up to (max_size_px)**2 area.

    Applied before dilation: at a loose detection threshold a scattered
    few-pixel noise excursion would otherwise be inflated into a much larger
    blob far from any real structure, polluting genuine blank sky.
    """
    if max_size_px <= 0 or not mask.any():
        return mask
    labeled, n_objects = label(mask)
    if n_objects == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros(sizes.size, dtype=bool)
    keep[1:] = sizes[1:] > max_size_px * max_size_px
    return keep[labeled]


def _classify_segments(segm, convolved: np.ndarray, fwhm_px: float,
                       tier_label: str, kernel_sigma: float) -> list[dict]:
    """Per-segment audit record with an extent-based point/extended verdict.

    Classification gates on `equivalent_radius` (= sqrt(area/pi)) relative to
    the PSF's own FWHM, plus the detecting tier's scale. Eccentricity and the
    concentration proxies are recorded but never gated on: eccentricity
    separates round from elongated, not compact from extended, and filamentary
    nebulosity, blended pairs and trailed stars all sit at its high end
    together -- it puts members of both classes at both ends of the axis.
    """
    if segm is None or segm.n_labels == 0:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat = SourceCatalog(convolved, segm)
        r_eq = np.atleast_1d(np.asarray(cat.equivalent_radius, dtype=float))
        areas = np.atleast_1d(np.asarray(cat.area, dtype=float))
        ecc = np.atleast_1d(np.asarray(cat.eccentricity, dtype=float))
        peak = np.atleast_1d(np.asarray(cat.max_value, dtype=float))
        labels = np.atleast_1d(np.asarray(cat.labels))

    radius_cut = SOURCEMASK_EXTENDED_RADIUS_MULT * fwhm_px
    # A tier whose kernel is already much broader than the PSF can only be
    # responding to extended structure, whatever the segment's own shape.
    tier_is_extended = kernel_sigma > 2.0 * fwhm_px * _FWHM_TO_SIGMA

    out: list[dict] = []
    for i, lbl in enumerate(labels):
        is_ext = bool(tier_is_extended or r_eq[i] > radius_cut)
        out.append({
            "label": int(lbl),
            "tier": tier_label,
            "kernel_sigma": float(kernel_sigma),
            "area": float(areas[i]),
            "equivalent_radius": float(r_eq[i]),
            "eccentricity": float(ecc[i]),
            "max_value": float(peak[i]),
            "classification": "extended" if is_ext else "point",
        })
    return out


def _segment_mask_for(segm, records: list[dict], classification: str) -> np.ndarray:
    """Boolean mask of just the segments carrying the given classification."""
    labels = [r["label"] for r in records if r["classification"] == classification]
    if segm is None or not labels:
        return np.zeros(segm.shape if segm is not None else (0, 0), dtype=bool)
    return np.isin(segm.data, labels)


def build_source_mask(data: np.ndarray, pedestal_surface: np.ndarray,
                      sky_sigma: float, fwhm_px: float, *,
                      step: int = 1,
                      ext_kernel_sigmas: tuple[float, ...] = SOURCEMASK_EXT_KERNEL_SIGMAS,
                      ext_nsigma: tuple[float, ...] = SOURCEMASK_EXT_NSIGMA,
                      user_exclusion: np.ndarray | None = None,
                      ) -> SourceMaskResult:
    """Stages 2+3: multi-scale detection on `data - pedestal_surface`.

    `pedestal_surface` must already be full-resolution and aligned with `data`;
    subtracting it is what flattens the frame so a single global threshold is
    valid everywhere, making detection robust to a large-scale gradient.

    `step` decimates the *extended* tiers only. Extended structure survives
    decimation by construction, and the mask upsamples back exactly, so this
    buys a step**2 cost reduction on the expensive large-kernel convolutions
    without changing what they can resolve. The point-source tier always runs
    at full resolution -- a star is exactly what decimation destroys.

    `user_exclusion` is a hand-drawn mask that is always excluded, whatever the
    detector concludes. It exists for structure no threshold can identify: a
    smooth nebula is genuinely degenerate with a sky gradient, and a centred
    nebula with vignetting, so those cases are unresolvable from pixel values
    alone. It is kept as its own class in the result rather than merged into
    `extended_mask`.
    """
    h, w = data.shape[:2]
    residual = (data.astype(np.float64) - pedestal_surface.astype(np.float64))
    tiers: list[dict] = []
    segments: list[dict] = []
    point_mask = np.zeros((h, w), dtype=bool)
    extended_mask = np.zeros((h, w), dtype=bool)

    user_mask = None
    if user_exclusion is not None:
        user_mask = np.asarray(user_exclusion, dtype=bool)
        if user_mask.shape[:2] != (h, w):
            raise ValueError(
                f"user_exclusion shape {user_mask.shape[:2]} does not match "
                f"data shape {(h, w)}")

    if not np.isfinite(sky_sigma) or sky_sigma <= 0:
        # Degenerate noise estimate: detect nothing, but still honour the user's
        # own regions — those do not depend on any threshold.
        combined = (user_mask.copy() if user_mask is not None
                    else np.zeros((h, w), dtype=bool))
        return SourceMaskResult(mask=combined,
                                point_mask=point_mask, extended_mask=extended_mask,
                                coverage=float(combined.mean()),
                                sky_sigma=float(sky_sigma),
                                tiers=[], segments=[], user_mask=user_mask)

    # --- point-source tier, full resolution ---
    point_kernel_sigma = max(0.8, fwhm_px * _FWHM_TO_SIGMA)
    det, rec = _detect_tier(residual, point_kernel_sigma, SOURCEMASK_POINT_NSIGMA,
                            SOURCEMASK_POINT_NPIXELS, sky_sigma)
    rec["tier"] = "point"
    segm, conv = rec.pop("_segm", None), rec.pop("_convolved", None)
    if segm is not None:
        recs = _classify_segments(segm, conv, fwhm_px, "point", point_kernel_sigma)
        segments += recs
        point_mask |= _segment_mask_for(segm, recs, "point")
        extended_mask |= _segment_mask_for(segm, recs, "extended")
    tiers.append(rec)

    # Finalise the point mask *before* the extended tiers run, so their
    # convolutions can be fed a star-free residual (see below).
    point_mask = _remove_small_objects(point_mask, 1)   # drop single-pixel hits only
    star_grow = int(round(SOURCEMASK_STAR_DILATION_FWHM_MULT * fwhm_px))
    if star_grow > 0 and point_mask.any():
        point_mask = binary_dilation(point_mask, iterations=star_grow)

    # Excise the point sources before smoothing at the extended scales. A star
    # is compact but *bright*: convolved with a 16 px kernel a 30000 ADU peak
    # still stands ~640x above that tier's threshold, so it registers as a
    # ~54 px-radius "extended" blob. Left in, 40 ordinary stars alone masked
    # 63% of a star-only frame. Zero is the correct fill because `residual` is
    # already pedestal-subtracted, so zero *is* the local sky level.
    residual_ext = residual.copy()
    residual_ext[point_mask] = 0.0

    # --- extended tiers, optionally decimated ---
    residual_dec = _decimate(residual_ext, step)
    for k_sigma, n_sig in zip(ext_kernel_sigmas, ext_nsigma):
        # Kernel sigma is expressed in native pixels; on the decimated grid the
        # same physical scale is step times smaller.
        k_dec = max(1.0, k_sigma / step)
        npix = max(5, int(round(SOURCEMASK_EXT_NPIX_MULT * k_dec ** 2)))
        det, rec = _detect_tier(residual_dec, k_dec, n_sig, npix, sky_sigma)
        rec["tier"] = f"extended σ={k_sigma:g}px"
        rec["kernel_sigma"] = float(k_sigma)   # report the native-scale sigma
        rec["step"] = int(step)
        segm, conv = rec.pop("_segm", None), rec.pop("_convolved", None)
        if segm is not None:
            recs = _classify_segments(segm, conv, max(1.0, fwhm_px / step),
                                      rec["tier"], k_dec)
            for r in recs:
                r["kernel_sigma"] = float(k_sigma)
                # Areas/radii were measured on the decimated grid; report them
                # in native pixels so the audit table is in one unit system.
                r["area"] *= step ** 2
                r["equivalent_radius"] *= step
            segments += recs
            pm = _upsample_mask(_segment_mask_for(segm, recs, "point"), step, (h, w))
            em = _upsample_mask(_segment_mask_for(segm, recs, "extended"), step, (h, w))
            # Grow each extended tier by its own kernel scale before union, so
            # the margin tracks the structure that produced it.
            grow = int(round(SOURCEMASK_EXT_DILATION_SIGMA_MULT * k_sigma))
            if grow > 0 and em.any():
                em = binary_dilation(em, iterations=grow)
            point_mask |= pm
            extended_mask |= em
        tiers.append(rec)

    # --- morphology on the extended mask (the point mask is already final) ---
    extended_mask = _fill_small_holes(extended_mask, SOURCEMASK_MAX_HOLE_PX)
    extended_mask = _remove_small_objects(extended_mask, SOURCEMASK_MAX_HOLE_PX)

    # The three reported classes are kept mutually exclusive so their pixel
    # counts sum to the total coverage. Precedence is user > extended > point:
    # a pixel the user excluded is reported as theirs even where the detector
    # also found something, since attributing it to the algorithm would
    # overstate what the algorithm actually achieved.
    if user_mask is not None:
        extended_mask = extended_mask & ~user_mask
        point_mask = point_mask & ~user_mask
    point_mask = point_mask & ~extended_mask
    combined = point_mask | extended_mask
    if user_mask is not None:
        combined = combined | user_mask

    return SourceMaskResult(
        mask=combined, point_mask=point_mask, extended_mask=extended_mask,
        coverage=float(combined.mean()), sky_sigma=float(sky_sigma),
        tiers=tiers, segments=segments, user_mask=user_mask,
    )


# ----------------------------------------------------------------------
# Stage 4 — masked estimate with order cap and fallback ladder
# ----------------------------------------------------------------------

def _cell_unmasked_fraction(source_mask: np.ndarray, mesh_shape: tuple[int, int],
                            box_size: int) -> np.ndarray:
    """Fraction of each mesh cell's pixels that survived the source mask.

    Computed here rather than read off Background2D, because photutils'
    `n_pixels_mesh` cannot distinguish a cell it measured from one it excluded
    and back-filled by interpolation -- both report a plausible value. This is
    the quantity stage 4 uses to decide which cells are real measurements, so
    it needs to be ours and inspectable.
    """
    h, w = source_mask.shape[:2]
    unmasked = (~source_mask).astype(np.float32)
    row_idx = np.arange(0, h, box_size)
    col_idx = np.arange(0, w, box_size)
    sums = np.add.reduceat(np.add.reduceat(unmasked, row_idx, axis=0), col_idx, axis=1)
    counts = np.add.reduceat(np.add.reduceat(np.ones_like(unmasked), row_idx, axis=0),
                             col_idx, axis=1)
    frac = np.divide(sums, counts, out=np.zeros_like(sums),
                     where=counts > 0).astype(np.float64)
    # photutils pads a ragged edge, so its mesh can be a cell wider/taller than
    # the block grid above; align defensively rather than assume they match.
    out = np.zeros(mesh_shape, dtype=np.float64)
    r = min(mesh_shape[0], frac.shape[0])
    c = min(mesh_shape[1], frac.shape[1])
    out[:r, :c] = frac[:r, :c]
    return out


def _order_cap_for(n_cells: int, n_cells_total: int = 0) -> int | None:
    """Max polynomial order the surviving-cell count can honestly support.

    The quadric must clear an absolute floor AND a fraction of the whole mesh.
    An absolute count alone does not survive a change of image scale: 35 cells
    is 54.7% of the 64-cell mesh it was tuned on but 1.2% of a 24 MP frame's
    2925-cell mesh, so the guard evaporates on exactly the large images where a
    quadric has the most masked area to extrapolate across.

    `n_cells_total=0` makes the fraction term vanish, so a single-argument call
    reproduces the old absolute-only rule exactly.
    """
    quadric_min = max(SOURCEMASK_MIN_CELLS_QUADRIC,
                      int(np.ceil(SOURCEMASK_QUADRIC_CELL_FRACTION * max(0, n_cells_total))))
    if n_cells >= quadric_min:
        return 2
    if n_cells >= SOURCEMASK_MIN_CELLS_PLANE:
        return 1
    if n_cells >= SOURCEMASK_MIN_CELLS_CONSTANT:
        return 0
    return None


def plane_quadric_agreement(mesh: np.ndarray, rms_mesh: np.ndarray, box_size: int,
                            image_shape: tuple[int, int], valid: np.ndarray,
                            sky_sigma: float,
                            n_grid: int = SOURCEMASK_AGREEMENT_GRID) -> dict | None:
    """Fit the same cells at order 1 and order 2 and measure where they part company.

    A quadric fitted to cells that survive a dense mask can pass through every
    measurement and still swing wildly across the region with no data. BIC cannot
    see that -- it only scores the fit against the cells it was given -- and no
    cell-count threshold generalises across image scale, so the honest test is to
    fit both orders and look at the disagreement itself.

    Two numbers, because one is not enough:

      * `max_divergence_adu` -- worst |quadric - plane| anywhere on the frame.
      * `max_divergence_measured_adu` -- worst |quadric - plane| at the surviving
        cell centres, i.e. where both models are actually constrained.

    Divergence alone cannot separate real curvature from invented curvature: on a
    genuinely vignetted frame the plane simply cannot fit, so the two surfaces
    differ a lot *everywhere*, including where the data is -- and the quadric is
    right. What identifies extrapolation is the ratio: agreement where cells
    survive, disagreement only out across the mask.

    Returns None when either fit is unavailable (too few cells, degenerate mesh).
    """
    finite = np.isfinite(mesh) & np.isfinite(rms_mesh) & (rms_mesh > 0) & valid
    if int(finite.sum()) < SOURCEMASK_MIN_CELLS_PLANE:
        return None
    masked_mesh = np.where(finite, mesh, np.nan)
    masked_rms = np.where(finite, rms_mesh, np.nan)

    plane = fit_background_surface(masked_mesh, masked_rms, box_size, image_shape,
                                   max_order=1)
    quadric = fit_background_surface(masked_mesh, masked_rms, box_size, image_shape,
                                     max_order=2)
    if plane.get("insufficient_data") or quadric.get("insufficient_data"):
        return None

    h, w = image_shape[:2]
    n = max(2, int(n_grid))
    gx, gy = np.meshgrid(np.linspace(0.0, float(w - 1), n),
                         np.linspace(0.0, float(h - 1), n))
    frame_diff = np.abs(evaluate_surface_at(quadric, gx, gy)
                        - evaluate_surface_at(plane, gx, gy))
    max_frame = float(np.nanmax(frame_diff)) if frame_diff.size else 0.0

    cx, cy = _mesh_cell_centres(mesh.shape, box_size)
    cell_diff = np.abs(evaluate_surface_at(quadric, cx[finite], cy[finite])
                       - evaluate_surface_at(plane, cx[finite], cy[finite]))
    max_cells = float(np.nanmax(cell_diff)) if cell_diff.size else 0.0

    sigma_units = (max_frame / sky_sigma
                   if np.isfinite(sky_sigma) and sky_sigma > 0 else None)
    # Guard the ratio on the noise floor, not on zero: two surfaces that agree to
    # within a rounding error everywhere would otherwise produce an enormous and
    # meaningless ratio from dividing one tiny number by another.
    floor = (0.01 * sky_sigma if np.isfinite(sky_sigma) and sky_sigma > 0
             else max(max_frame, 1e-12) * 1e-6)
    ratio = float(max_frame / max(max_cells, floor))

    diverges = bool(
        sigma_units is not None
        and sigma_units > SOURCEMASK_MAX_MODEL_DIVERGENCE_SIGMA
        and ratio > SOURCEMASK_MAX_EXTRAPOLATION_RATIO)

    return {
        "plane_fit": plane,
        "quadric_fit": quadric,
        "max_divergence_adu": max_frame,
        "max_divergence_measured_adu": max_cells,
        "max_divergence_sigma": sigma_units,
        "extrapolation_ratio": ratio,
        "verdict": "diverge" if diverges else "agree",
        "demoted": False,          # set by masked_background_estimate if it acts
        "n_cells": int(finite.sum()),
    }


def _geometry_supported_order(cx: np.ndarray, cy: np.ndarray,
                              keep: np.ndarray, order: int) -> int:
    """Drop the order until the surviving cells' geometry can actually constrain it.

    All-surviving-cells-in-one-row is the realistic failure (a nebula spanning
    the frame leaves only a top or bottom strip): a plane through a collinear
    set is unconstrained in the perpendicular direction, and the WLS solve
    returns a garbage coefficient rather than raising.
    """
    if order <= 0:
        return 0
    xs, ys = cx[keep], cy[keep]
    if xs.size == 0:
        return 0
    x_spread = float(np.ptp(xs))
    y_spread = float(np.ptp(ys))
    if x_spread <= 0 or y_spread <= 0:
        return 0                       # collinear: only a constant is defensible
    if order >= 2:
        # A quadric additionally needs >2 distinct coordinates on each axis.
        if np.unique(xs).size < 3 or np.unique(ys).size < 3:
            return 1
    return order


def masked_background_estimate(data: np.ndarray, source_mask: np.ndarray,
                               box_size: int, *,
                               exclude_percentile: float = SOURCEMASK_EXCLUDE_PERCENTILE,
                               min_cell_unmasked_frac: float = SOURCEMASK_MIN_CELL_UNMASKED_FRAC,
                               sky_sigma: float | None = None,
                               ) -> tuple[dict | None, str | None]:
    """Run the masked Background2D pass and fit its surviving mesh cells.

    Returns (payload, fallback_reason). `payload` carries the fit, the
    surviving-cell bookkeeping and the masked RMS; it is None whenever the mask
    leaves too little sky, in which case fallback_reason says why.

    `sky_sigma` scales the plane-vs-quadric agreement check into sky-noise units;
    omitted, the surviving cells' own median RMS stands in for it.

    The fit -- not Background2D's interpolated mesh -- is the product here.
    Once the mask is dense, most cells are empty and photutils interpolates
    them from a sparse, spatially-clustered remainder, which measured worse
    than the unmasked bias it was meant to remove.
    """
    h, w = data.shape[:2]
    coverage = float(source_mask.mean())
    if coverage >= SOURCEMASK_MAX_COVERAGE:
        return None, (f"source mask covers {coverage:.1%} of the frame "
                      f"(limit {SOURCEMASK_MAX_COVERAGE:.0%}) — too little sky remains "
                      f"to estimate a background")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=AstropyUserWarning)
            bkg = Background2D(
                data, box_size=box_size, filter_size=SOURCEMASK_MASKED_FILTER_SIZE,
                sigma_clip=SigmaClip(sigma=_SIGCLIP_SIGMA, maxiters=_SIGCLIP_MAXITERS),
                bkg_estimator=SExtractorBackground(),
                bkg_rms_estimator=MADStdBackgroundRMS(),
                mask=source_mask, exclude_percentile=exclude_percentile,
            )
    except ValueError as exc:
        # photutils raises outright when the mask leaves no usable pixels.
        return None, f"masked Background2D failed: {exc}"

    mesh = np.asarray(bkg.background_mesh, dtype=np.float64).copy()
    rms_mesh = np.asarray(bkg.background_rms_mesh, dtype=np.float64).copy()
    # Keep only cells that are genuinely *measured*. A cell photutils excluded
    # still carries a plausible-looking interpolated value in background_mesh,
    # and n_pixels_mesh cannot tell the two apart -- fitting the fills is what
    # produced a 55 ADU mesh RMS error against a 5 ADU truth on a gradient
    # frame, and flattened the recovered gradient to under half its real size.
    cell_frac = _cell_unmasked_fraction(source_mask, mesh.shape, box_size)
    unmeasured = cell_frac < min_cell_unmasked_frac
    mesh[unmeasured] = np.nan
    rms_mesh[unmeasured] = np.nan

    valid = np.isfinite(mesh) & np.isfinite(rms_mesh) & (rms_mesh > 0)
    n_cells = int(valid.sum())
    n_cells_total = int(mesh.size)

    cap = _order_cap_for(n_cells, n_cells_total)
    if cap is None:
        return None, (f"only {n_cells} of {n_cells_total} mesh cells survived the "
                      f"source mask (minimum {SOURCEMASK_MIN_CELLS_CONSTANT})")

    cx, cy = _mesh_cell_centres(mesh.shape, box_size)
    cap = _geometry_supported_order(cx, cy, valid, cap)

    # Plane-vs-quadric agreement. Computed whenever a quadric is on the table at
    # all, because the number is worth reporting even when it does not change the
    # decision -- a reader cannot otherwise tell a curved fit that follows real
    # curvature from one that is extrapolating across the mask.
    agreement = None
    if cap >= 2:
        sigma = (float(sky_sigma) if sky_sigma is not None
                 and np.isfinite(sky_sigma) and sky_sigma > 0
                 else float(np.median(rms_mesh[valid])))
        agreement = plane_quadric_agreement(mesh, rms_mesh, box_size, (h, w),
                                            valid, sigma)
        if (agreement is not None and agreement["verdict"] == "diverge"
                and SOURCEMASK_DEMOTE_ON_DIVERGENCE):
            cap = 1
            agreement["demoted"] = True

    fit = fit_background_surface(mesh, rms_mesh, box_size, (h, w), max_order=cap)
    if fit.get("insufficient_data"):
        return None, f"masked mesh fit had insufficient data ({n_cells} cells)"

    return {
        "fit": fit,
        "n_cells": n_cells,
        "n_cells_total": n_cells_total,
        "order_cap": cap,
        "coverage": coverage,
        "rms_median": float(np.median(rms_mesh[valid])),
        # The full-resolution RMS map from this same masked pass. photutils has
        # already computed it; returning it is what lets a caller adopt the masked
        # noise floor rather than only the masked background level.
        "rms_map": np.asarray(bkg.background_rms, dtype=np.float32),
        "n_pixels_mesh": np.asarray(bkg.n_pixels_mesh).copy(),
        "cell_unmasked_fraction": cell_frac,
        "mesh_valid": valid,
        "agreement": agreement,
    }, None


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

def source_masked_background(data: np.ndarray, *, box_size: int = 64,
                             fwhm_px: float = 4.0,
                             step: int = 1,
                             n_passes: int = SOURCEMASK_N_PASSES,
                             exclude_percentile: float = SOURCEMASK_EXCLUDE_PERCENTILE,
                             user_exclusion: np.ndarray | None = None,
                             mesh: np.ndarray | None = None,
                             rms_mesh: np.ndarray | None = None,
                             ) -> MaskedBackgroundResult:
    """Full 4-stage source-masked background estimate — the single entry point.

    `mesh`/`rms_mesh` let a caller hand in the unmasked Background2D grids it
    has already computed (every AstroImage has them after estimate_background),
    avoiding a redundant full Background2D pass; omit them and they are
    computed here.

    `user_exclusion` is a full-resolution boolean mask of regions the user drew
    by hand. It is applied twice — to the stage-1 scaffold's cell selection and
    to the final mask — because those do different jobs: the first improves the
    pedestal (and therefore detection everywhere), the second guarantees the
    pixels are excluded from the fit whatever the detector decided.

    `n_passes` repeats stages 2-4 with the previous pass's fitted surface as
    the detection pedestal. It defaults to 1 because that loop *diverges*: a
    lower pedestal detects more, which grows the mask, which leaves fewer and
    more spatially-clustered cells, which pushes the estimate lower again. See
    SOURCEMASK_N_PASSES for the measured numbers.
    """
    data = np.asarray(data)
    h, w = data.shape[:2]

    if mesh is None or rms_mesh is None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=AstropyUserWarning)
                base = Background2D(
                    data, box_size=box_size, filter_size=3,
                    sigma_clip=SigmaClip(sigma=_SIGCLIP_SIGMA, maxiters=_SIGCLIP_MAXITERS),
                    bkg_estimator=SExtractorBackground(),
                    bkg_rms_estimator=MADStdBackgroundRMS(),
                )
        except ValueError as exc:
            return MaskedBackgroundResult(
                surface=None, rms_median=None, background_median=None, fit=None,
                n_cells=0, n_cells_total=0, order_cap=None, coverage=0.0,
                fallback_reason=f"unmasked Background2D failed: {exc}",
                source_mask=None, scaffold=None)
        mesh = np.asarray(base.background_mesh, dtype=np.float64)
        rms_mesh = np.asarray(base.background_rms_mesh, dtype=np.float64)

    user_mask = None
    excluded_cells = None
    if user_exclusion is not None:
        user_mask = np.asarray(user_exclusion, dtype=bool)
        if user_mask.shape[:2] != (h, w):
            raise ValueError(
                f"user_exclusion shape {user_mask.shape[:2]} does not match "
                f"data shape {(h, w)}")
        # A cell is dropped from the scaffold on the same "is it still a real
        # measurement" rule stage 4 uses, so the two stages agree about what
        # counts as usable.
        frac = _cell_unmasked_fraction(user_mask, np.asarray(mesh).shape, box_size)
        excluded_cells = frac < SOURCEMASK_MIN_CELL_UNMASKED_FRAC

    # --- Stage 1: gradient-robust scaffold, no segmentation involved ---
    scaffold = fit_sky_scaffold(mesh, rms_mesh, box_size, (h, w),
                                excluded_cells=excluded_cells)
    if scaffold.get("insufficient_data"):
        return MaskedBackgroundResult(
            surface=None, rms_median=None, background_median=None, fit=None,
            n_cells=0, n_cells_total=int(np.asarray(mesh).size), order_cap=None,
            coverage=0.0, fallback_reason="sky scaffold fit had insufficient data",
            source_mask=None, scaffold=scaffold)

    sky_sigma = _scaffold_sky_sigma(rms_mesh, scaffold["kept_cells"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pedestal = np.asarray(evaluate_surface_at(scaffold, xx, yy), dtype=np.float64)

    smask: SourceMaskResult | None = None
    payload: dict | None = None
    reason: str | None = None
    surface: np.ndarray | None = None

    for _ in range(max(1, n_passes)):
        # --- Stages 2+3 ---
        # Passed *into* build_source_mask rather than merged afterwards: this
        # loop re-invokes it each pass, so a post-merge would be dropped on
        # every pass after the first.
        smask = build_source_mask(data, pedestal, sky_sigma, fwhm_px, step=step,
                                  user_exclusion=user_mask)
        # --- Stage 4 ---
        payload, reason = masked_background_estimate(
            data, smask.mask, box_size, exclude_percentile=exclude_percentile,
            sky_sigma=sky_sigma)
        if payload is None:
            break
        surface = np.asarray(evaluate_surface_at(payload["fit"], xx, yy),
                             dtype=np.float64)
        # Next pass detects against the masked estimate rather than the
        # stage-1 scaffold, so the pedestal improves with the mask.
        pedestal = surface

    if payload is None:
        return MaskedBackgroundResult(
            surface=None, rms_median=None, background_median=None, fit=None,
            n_cells=0, n_cells_total=int(np.asarray(mesh).size), order_cap=None,
            coverage=smask.coverage if smask is not None else 0.0,
            fallback_reason=reason, source_mask=smask, scaffold=scaffold)

    return MaskedBackgroundResult(
        surface=surface.astype(np.float32),
        rms_median=payload["rms_median"],
        background_median=float(np.median(surface)),
        fit=payload["fit"],
        n_cells=payload["n_cells"],
        n_cells_total=payload["n_cells_total"],
        order_cap=payload["order_cap"],
        coverage=payload["coverage"],
        fallback_reason=None,
        source_mask=smask,
        scaffold=scaffold,
        rms_map=payload.get("rms_map"),
        agreement=payload.get("agreement"),
    )
