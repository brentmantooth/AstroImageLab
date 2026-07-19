from __future__ import annotations

import concurrent.futures
import numpy as np
try:
    import bottleneck as bn
except ImportError:
    bn = np
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import generic_filter, uniform_filter, gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude, map_coordinates, zoom, maximum_filter, binary_dilation, binary_fill_holes, label
import pywt

from core.astro_image import AstroImage
from core.fig_utils import fig_to_b64, figs_to_b64
from core.models import (STD_KERNEL_SIZES, LOG_SIGMAS, WAVELET_NAME, WAVELET_LEVELS,
                         ENTROPY_KERNEL_SIZES,
                         XS_LINE_ALPHA, SECTION8_BORDER_CROP_FRACTION, SECTION8_ANALYSIS_CMAP,
                         XS_SNR_REGION_WIDTH,
                         SECTION8_LOGRATIO_EPS_PERCENTILE, SECTION8_SCATTER_MAX_SAMPLES,
                         SECTION8_NEBULA_MASK_SIGMA, SECTION8_NEBULA_MASK_DILATION_PX,
                         SECTION8_NEBULA_MASK_MAX_HOLE_PX,
                         SECTION8_LOCALMAX_FOOTPRINT_MULT, SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                         SECTION8_LOCALMAX_PRESMOOTH_FRACTION, SECTION8_LOCALMAX_REGION_FRACTION,
                         SECTION8_LOCALMAX_DIST_MAX_SAMPLES, SECTION8_LOCALMAX_TOP_PERCENT,
                         SECTION8_ENTROPY_N_BINS, SECTION8_ENTROPY_CLIP_PERCENTILE)

MAX_DIM_FOR_STD = 2048   # downsample to this before generic_filter (performance)
_DISPLAY_SMOOTH_SIGMA = 1.0   # applied to maps before plotting; does NOT affect metrics


class SpatialDetailAnalyzer:
    """Multi-scale spatial detail comparison: local std, LoG, and wavelet.

    Takes both images simultaneously and produces side-by-side comparison
    figures. All processing uses mean-signal-normalised data.
    """

    def analyze(self, image_a: AstroImage, image_b: AstroImage | None = None,
                kernel_sizes: tuple = STD_KERNEL_SIZES,
                log_sigmas: tuple = LOG_SIGMAS,
                wavelet: str = WAVELET_NAME,
                levels: int = WAVELET_LEVELS,
                entropy_kernel_sizes: tuple = ENTROPY_KERNEL_SIZES,
                crosshair: dict | None = None,
                roi: tuple | None = None,
                xs_snr_width: int | None = None,
                nebula_sigma: float = SECTION8_NEBULA_MASK_SIGMA,
                nebula_dilation_px: int = SECTION8_NEBULA_MASK_DILATION_PX,
                nebula_max_hole_px: int = SECTION8_NEBULA_MASK_MAX_HOLE_PX,
                localmax_footprint_mult: float = SECTION8_LOCALMAX_FOOTPRINT_MULT,
                localmax_prominence_percentile: float = SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                localmax_region_fraction: float = SECTION8_LOCALMAX_REGION_FRACTION,
                localmax_presmooth_fraction: float = SECTION8_LOCALMAX_PRESMOOTH_FRACTION,
                localmax_top_percent: float = SECTION8_LOCALMAX_TOP_PERCENT) -> dict:

        image_a.estimate_background()
        if image_b is not None:
            image_b.estimate_background()

        norm_a = self._normalise(image_a)
        norm_b = self._normalise(image_b) if image_b is not None else None

        result: dict = {
            "contrast_ratios_a": {},
            "contrast_ratios_b": {},
            "wavelet_snr_a": {},
            "wavelet_snr_b": {},
            "sigma_noise_a": None,
            "sigma_noise_b": None,
            "nebula_sigma": nebula_sigma,
            "nebula_dilation_px": nebula_dilation_px,
            "nebula_max_hole_px": nebula_max_hole_px,
            "localmax_footprint_mult": localmax_footprint_mult,
            "localmax_prominence_percentile": localmax_prominence_percentile,
            "localmax_region_fraction": localmax_region_fraction,
            "localmax_presmooth_fraction": localmax_presmooth_fraction,
            "localmax_top_percent": localmax_top_percent,
            "entropy_contrast_ratio_a": {},
            "entropy_contrast_ratio_b": {},
            "panels": {},
            "localmax": {},
            "nc_shared_nebula_pixels": 0,
            "std_nc_score_a": {}, "std_nc_score_b": {},
            "std_nc_noise_a": {}, "std_nc_noise_b": {}, "std_nc_ratio": {},
            "std_nc_neb_std_a": {}, "std_nc_neb_std_b": {}, "std_nc_ratio_err": {},
            "log_nc_score_a": {}, "log_nc_score_b": {},
            "log_nc_noise_a": {}, "log_nc_noise_b": {}, "log_nc_ratio": {},
            "log_nc_neb_std_a": {}, "log_nc_neb_std_b": {}, "log_nc_ratio_err": {},
            "wavelet_nc_score_a": {}, "wavelet_nc_score_b": {},
            "wavelet_nc_noise_a": {}, "wavelet_nc_noise_b": {}, "wavelet_nc_ratio": {},
            "wavelet_nc_neb_std_a": {}, "wavelet_nc_neb_std_b": {}, "wavelet_nc_ratio_err": {},
            "entropy_nc_score_a": {}, "entropy_nc_score_b": {},
            "entropy_nc_noise_a": {}, "entropy_nc_noise_b": {}, "entropy_nc_ratio": {},
            "entropy_nc_neb_std_a": {}, "entropy_nc_neb_std_b": {}, "entropy_nc_ratio_err": {},
            "gm_nc_score_a": {}, "gm_nc_score_b": {},
            "gm_nc_noise_a": {}, "gm_nc_noise_b": {}, "gm_nc_ratio": {},
            "gm_nc_neb_std_a": {}, "gm_nc_neb_std_b": {}, "gm_nc_ratio_err": {},
        }
        figures: dict = {}

        if norm_a is None or (image_b is not None and norm_b is None):
            result["warning"] = "Mean signal ≤ 0 in one or both images; spatial analysis skipped."
            return result

        # Nebula / background masks (used for contrast ratio)
        mask_neb_a, mask_bg_a = self._make_masks(image_a, nebula_sigma, nebula_dilation_px, nebula_max_hole_px)
        mask_neb_b = mask_bg_b = None
        if image_b is not None:
            mask_neb_b, mask_bg_b = self._make_masks(image_b, nebula_sigma, nebula_dilation_px, nebula_max_hole_px)

        # When a user ROI is provided, crop the analysis arrays to that region after
        # normalisation so the global signal mean is used, not the ROI's local mean.
        # The std/LoG/wavelet analysis and all display maps are then restricted to the
        # ROI only.  Crosshair sampling always uses the full-image arrays because
        # crosshair coordinates are always in full-image pixel space.
        if roi is not None:
            rx0, ry0, rx1, ry1 = roi
            analysis_a  = norm_a[ry0:ry1, rx0:rx1]
            analysis_b  = norm_b[ry0:ry1, rx0:rx1] if norm_b is not None else None
            mask_neb_a  = mask_neb_a[ry0:ry1, rx0:rx1]
            mask_bg_a   = mask_bg_a[ry0:ry1, rx0:rx1]
            if mask_neb_b is not None:
                mask_neb_b  = mask_neb_b[ry0:ry1, rx0:rx1]
                mask_bg_b   = mask_bg_b[ry0:ry1, rx0:rx1]
            display_roi = None   # arrays are already the analysis region; no further cropping
            result["roi_used"] = roi
            if crosshair is not None:
                roi_w = rx1 - rx0
                roi_h = ry1 - ry0
                img_h, img_w = norm_a.shape[:2]
                def _clip01(v): return max(0.0, min(1.0, v))
                crosshair_roi = {
                    "x0": _clip01((crosshair["x0"] * img_w - rx0) / roi_w),
                    "y0": _clip01((crosshair["y0"] * img_h - ry0) / roi_h),
                    "x1": _clip01((crosshair["x1"] * img_w - rx0) / roi_w),
                    "y1": _clip01((crosshair["y1"] * img_h - ry0) / roi_h),
                }
            else:
                crosshair_roi = None
        else:
            analysis_a  = norm_a
            analysis_b  = norm_b  # may be None in single-image mode
            # Bright-feature bounding box for display cropping (uses image A's mask)
            display_roi = self._nebula_bounding_box(mask_neb_a, norm_a.shape)
            crosshair_roi = crosshair   # full-image coords are correct when no ROI

        result["display_roi"] = display_roi

        # Shared nebula/background regions for noise-corrected A/B scoring and diff
        # distributions. Nebula is the two-image UNION: a pixel counts as Nebula if
        # either image independently classifies it that way, so nebula signal that's
        # marginal in one image (registration offset, PSF, local noise) still counts.
        # Background stays the two-image INTERSECTION: a pixel counts as Background
        # only if both images agree, keeping the noise-floor reference population
        # clean. Per-image nebula-dominance in _make_masks (bg_mask excludes
        # nebula_mask) guarantees these two combinations never overlap.
        # None in single-image mode.
        mask_neb_shared = mask_bg_shared = None
        if mask_neb_b is not None:
            h_s = min(mask_neb_a.shape[0], mask_neb_b.shape[0])
            w_s = min(mask_neb_a.shape[1], mask_neb_b.shape[1])
            mask_neb_shared = mask_neb_a[:h_s, :w_s] | mask_neb_b[:h_s, :w_s]
            mask_bg_shared = mask_bg_a[:h_s, :w_s] & mask_bg_b[:h_s, :w_s]
        result["nc_shared_nebula_pixels"] = (
            int(np.count_nonzero(mask_neb_shared)) if mask_neb_shared is not None else 0
        )

        # Fixed seed so the correlation-scatter and local-maxima-distribution
        # subsampling below is reproducible across report generations for the
        # same input images.
        rng = np.random.default_rng(42)

        # Export the exact preprocessed array every std/LoG/wavelet/entropy/gradient
        # map is computed from (mean-normalised, ROI-cropped if a ROI was given) as
        # its own panel, so the Report Inspector can show the source image content
        # for a map alongside the map itself, pixel-aligned to the same crop.
        original_diff = self._log_ratio_map(analysis_a, analysis_b) if analysis_b is not None else None
        result["panels"]["original"] = {
            "a":    analysis_a.astype(np.float32),
            "b":    analysis_b.astype(np.float32) if analysis_b is not None else None,
            "diff": original_diff,
        }
        # Mask illustration: only meaningful in two-image mode.
        if mask_neb_shared is not None:
            mask_fig = self._plot_mask_illustration(
                result["panels"]["original"]["a"], mask_neb_shared, mask_bg_shared)
            figures["mask_illustration"] = fig_to_b64(mask_fig, dpi=150)

        _label_b = image_b.label if image_b is not None else None

        # Original image family: the raw source data itself, analysed with the same
        # map-pair layout (A|B, log-ratio, cross-section, histogram, correlation) as
        # every derived metric below — not a filter, just the input they all share.
        xs_raw_orig = None
        xs_line_orig = None
        if crosshair_roi is not None and analysis_b is not None:
            pos, pa = self._sample_line(analysis_a, **crosshair_roi)
            _, pb = self._sample_line(analysis_b, **crosshair_roi)
            xs_raw_orig = (pos, pa, pb, image_a.label, _label_b,
                           "Cross-section — Original (normalised image)")
            xs_line_orig = self._crosshair_to_cropped_px(
                crosshair_roi, analysis_a.shape, SECTION8_BORDER_CROP_FRACTION)

        if analysis_b is not None:
            orig_fig = self._plot_side_by_side(
                self._crop_border(analysis_a, SECTION8_BORDER_CROP_FRACTION),
                self._crop_border(analysis_b, SECTION8_BORDER_CROP_FRACTION),
                f"Original (normalised) — {image_a.label}",
                f"Original (normalised) — {_label_b}",
                diff_title="Log ratio (A/B), original image",
                cmap=SECTION8_ANALYSIS_CMAP,
                display_roi=None,
                xs_data=xs_raw_orig,
                xs_line=xs_line_orig,
            )
        else:
            orig_fig = self._plot_single(
                self._crop_border(analysis_a, SECTION8_BORDER_CROP_FRACTION),
                f"Original (normalised) — {image_a.label}",
                cmap=SECTION8_ANALYSIS_CMAP,
            )
        figures["original"] = fig_to_b64(orig_fig, dpi=150)

        if mask_neb_shared is not None:
            orig_corr_fig = self._plot_metric_correlation(
                analysis_a, analysis_b, original_diff, mask_neb_shared, mask_bg_shared,
                image_a.label, _label_b, "Original (normalised image)", rng)
            if orig_corr_fig is not None:
                figures["corr_original"] = fig_to_b64(orig_corr_fig, dpi=150)

        # 1-5. Local std, LoG, wavelet, entropy, gradient — all read norm_a/norm_b with no
        # shared mutable state, so they run concurrently. Each method returns
        # (b64_figs, partial_result).
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as _ex:
            _f_std = _ex.submit(self._std_analysis,
                analysis_a, analysis_b,
                mask_neb_a, mask_bg_a,
                mask_neb_b, mask_bg_b,
                kernel_sizes,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared,
                mask_bg_shared=mask_bg_shared, rng=rng,
                localmax_footprint_mult=localmax_footprint_mult,
                localmax_prominence_percentile=localmax_prominence_percentile,
                localmax_region_fraction=localmax_region_fraction,
                localmax_presmooth_fraction=localmax_presmooth_fraction,
                localmax_top_percent=localmax_top_percent,
            )
            _f_log = _ex.submit(self._log_analysis,
                analysis_a, analysis_b, log_sigmas,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared, mask_bg_a=mask_bg_a, mask_bg_b=mask_bg_b,
                mask_bg_shared=mask_bg_shared, rng=rng,
                localmax_footprint_mult=localmax_footprint_mult,
                localmax_prominence_percentile=localmax_prominence_percentile,
                localmax_region_fraction=localmax_region_fraction,
                localmax_presmooth_fraction=localmax_presmooth_fraction,
                localmax_top_percent=localmax_top_percent,
            )
            _f_wav = _ex.submit(self._wavelet_analysis,
                analysis_a, analysis_b, wavelet, levels,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared, mask_bg_a=mask_bg_a, mask_bg_b=mask_bg_b,
                mask_bg_shared=mask_bg_shared, rng=rng,
                localmax_footprint_mult=localmax_footprint_mult,
                localmax_prominence_percentile=localmax_prominence_percentile,
                localmax_region_fraction=localmax_region_fraction,
                localmax_presmooth_fraction=localmax_presmooth_fraction,
                localmax_top_percent=localmax_top_percent,
            )
            _f_ent = _ex.submit(self._entropy_analysis,
                analysis_a, analysis_b,
                mask_neb_a, mask_bg_a,
                mask_neb_b, mask_bg_b,
                entropy_kernel_sizes,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared,
                mask_bg_shared=mask_bg_shared, rng=rng,
                localmax_footprint_mult=localmax_footprint_mult,
                localmax_prominence_percentile=localmax_prominence_percentile,
                localmax_region_fraction=localmax_region_fraction,
                localmax_presmooth_fraction=localmax_presmooth_fraction,
                localmax_top_percent=localmax_top_percent,
            )
            _f_grad = _ex.submit(self._gradient_analysis,
                analysis_a, analysis_b, log_sigmas,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared, mask_bg_a=mask_bg_a, mask_bg_b=mask_bg_b,
                mask_bg_shared=mask_bg_shared, rng=rng,
                localmax_footprint_mult=localmax_footprint_mult,
                localmax_prominence_percentile=localmax_prominence_percentile,
                localmax_region_fraction=localmax_region_fraction,
                localmax_presmooth_fraction=localmax_presmooth_fraction,
                localmax_top_percent=localmax_top_percent,
            )
            std_b64, std_partial = _f_std.result()
            log_b64, log_partial = _f_log.result()
            wav_b64, wav_partial = _f_wav.result()
            ent_b64, ent_partial = _f_ent.result()
            grad_b64, grad_partial = _f_grad.result()

        figures.update(std_b64)
        figures.update(log_b64)
        figures.update(wav_b64)
        figures.update(ent_b64)
        figures.update(grad_b64)
        result["contrast_ratios_a"].update(std_partial["contrast_ratios_a"])
        result["contrast_ratios_b"].update(std_partial["contrast_ratios_b"])
        result["sigma_noise_a"] = wav_partial["sigma_noise_a"]
        result["sigma_noise_b"] = wav_partial["sigma_noise_b"]
        result["wavelet_snr_a"].update(wav_partial["wavelet_snr_a"])
        result["wavelet_snr_b"].update(wav_partial["wavelet_snr_b"])
        result["entropy_contrast_ratio_a"].update(ent_partial["entropy_contrast_ratio_a"])
        result["entropy_contrast_ratio_b"].update(ent_partial["entropy_contrast_ratio_b"])
        result["panels"].update(std_partial["panels"])
        result["panels"].update(log_partial["panels"])
        result["panels"].update(wav_partial["panels"])
        result["panels"].update(ent_partial["panels"])
        result["panels"].update(grad_partial["panels"])
        result["localmax"].update(std_partial["localmax"])
        result["localmax"].update(log_partial["localmax"])
        result["localmax"].update(wav_partial["localmax"])
        result["localmax"].update(ent_partial["localmax"])
        result["localmax"].update(grad_partial["localmax"])

        # Merge noise-corrected scores/noise-floors and compute A/B ratios centrally.
        for prefix, partial in (("std", std_partial), ("log", log_partial),
                                 ("wavelet", wav_partial), ("entropy", ent_partial),
                                 ("gm", grad_partial)):
            for suffix in ("nc_score_a", "nc_score_b", "nc_noise_a", "nc_noise_b",
                            "nc_neb_std_a", "nc_neb_std_b"):
                result[f"{prefix}_{suffix}"].update(partial[f"{prefix}_{suffix}"])
            result[f"{prefix}_nc_ratio"] = self._compute_nc_ratios(
                result[f"{prefix}_nc_score_a"], result[f"{prefix}_nc_score_b"])
            result[f"{prefix}_nc_ratio_err"] = self._compute_nc_ratio_errors(
                result[f"{prefix}_nc_ratio"], result[f"{prefix}_nc_score_a"], result[f"{prefix}_nc_score_b"],
                result[f"{prefix}_nc_noise_a"], result[f"{prefix}_nc_noise_b"],
                result[f"{prefix}_nc_neb_std_a"], result[f"{prefix}_nc_neb_std_b"])

        if image_b is not None:
            nc_errors_by_method = {
                "std": result["std_nc_ratio_err"], "log": result["log_nc_ratio_err"],
                "wavelet": result["wavelet_nc_ratio_err"], "entropy": result["entropy_nc_ratio_err"],
                "gradient": result["gm_nc_ratio_err"],
            }
            nc_fig = self._plot_nc_ratio_overview({
                "std": result["std_nc_ratio"],
                "log": result["log_nc_ratio"],
                "wavelet": result["wavelet_nc_ratio"],
                "entropy": result["entropy_nc_ratio"],
                "gradient": result["gm_nc_ratio"],
            }, nc_errors_by_method)
            if nc_fig is not None:
                figures["nc_ratio_overview"] = fig_to_b64(nc_fig, dpi=150)

            localmax_log_ratios_by_method = {
                "std": std_partial["localmax_log_ratio"],
                "log": log_partial["localmax_log_ratio"],
                "wavelet": wav_partial["localmax_log_ratio"],
                "entropy": ent_partial["localmax_log_ratio"],
                "gradient": grad_partial["localmax_log_ratio"],
            }
            localmax_log_ratio_errors_by_method = {
                "std": std_partial["localmax_log_ratio_err"],
                "log": log_partial["localmax_log_ratio_err"],
                "wavelet": wav_partial["localmax_log_ratio_err"],
                "entropy": ent_partial["localmax_log_ratio_err"],
                "gradient": grad_partial["localmax_log_ratio_err"],
            }
            lm_ratio_fig = self._plot_localmax_ratio_overview(
                localmax_log_ratios_by_method, localmax_log_ratio_errors_by_method)
            if lm_ratio_fig is not None:
                figures["localmax_ratio_overview"] = fig_to_b64(lm_ratio_fig, dpi=150)

            # Local-maxima mask grid: one row per metric family, columns = kernel/scale
            # sizes smallest -> largest. Every panel's mask is computed with the same
            # _combined_localmax_mask formula _localmax_entry uses for that row's own
            # table statistics -- this reuses panels already cached in
            # result["panels"], no new map computation. Wavelet has only 2 display
            # scales, so its 3rd column is left blank.
            _grid_families = [
                ("Local σ",        [(f"std_{ks}px",    float(ks),      f"Local σ — {ks} px")       for ks in kernel_sizes]),
                ("|LoG|",          [(f"log_{s}",        float(s),       f"|LoG| — σ={s} px")        for s in log_sigmas]),
                ("Gradient |G|",   [(f"gradient_{s}",   float(s),       f"Gradient |G| — σ={s} px") for s in log_sigmas]),
                ("Wavelet",        [(f"wavelet_{lvl}",  float(2 ** lvl), f"Wavelet — level {lvl}")   for lvl in (2, 3)]),
                ("Local entropy",  [(f"entropy_{ks}px", float(ks),      f"Entropy — {ks} px")       for ks in entropy_kernel_sizes]),
            ]
            grid_rows = []
            for family_label, entries in _grid_families:
                cells = []
                for key, scale_px, panel_title in entries:
                    panel = result["panels"].get(key)
                    if panel is None or panel["a"] is None or panel["b"] is None:
                        continue
                    h_g = min(panel["a"].shape[0], panel["b"].shape[0])
                    w_g = min(panel["a"].shape[1], panel["b"].shape[1])
                    abs_a = np.abs(panel["a"][:h_g, :w_g])
                    abs_b = np.abs(panel["b"][:h_g, :w_g])
                    footprint_px = max(3, int(round(localmax_footprint_mult * scale_px)) | 1)
                    region_px = max(1, int(round(localmax_region_fraction * footprint_px)))
                    presmooth_sigma = max(0.5, localmax_presmooth_fraction * scale_px)
                    mask = self._combined_localmax_mask(abs_a, abs_b, footprint_px,
                                                          localmax_prominence_percentile,
                                                          region_px, presmooth_sigma,
                                                          localmax_top_percent)
                    cells.append((abs_a, mask, panel_title))
                grid_rows.append((family_label, cells))
            grid_fig = self._plot_localmax_mask_grid(grid_rows)
            if grid_fig is not None:
                figures["localmax_mask_illustration"] = fig_to_b64(grid_fig, dpi=150)

        if crosshair is not None:
            pos_a, prof_a = self._sample_line(norm_a, **crosshair)
            pos_a_raw, prof_a_raw = self._sample_line(image_a.data, **crosshair)
            if image_b is not None and norm_b is not None:
                figures["xs_context"] = fig_to_b64(self._plot_context_figure(
                    image_a, image_b, image_a.label, image_b.label, crosshair), dpi=150)
                pos_b, prof_b = self._sample_line(norm_b, **crosshair)
                figures["xs_image_profile"] = fig_to_b64(self._plot_image_profile(
                    pos_a, prof_a, pos_b, prof_b, image_a.label, image_b.label), dpi=150)
                pos_b_raw, prof_b_raw = self._sample_line(image_b.data, **crosshair)
                figures["xs_image_profile_raw"] = fig_to_b64(self._plot_image_profile(
                    pos_a_raw, prof_a_raw, pos_b_raw, prof_b_raw,
                    image_a.label, image_b.label,
                    title="Cross-section brightness profile (raw counts)",
                    ylabel="Pixel value (raw ADU)",
                ), dpi=150)
                _width = xs_snr_width if xs_snr_width is not None else XS_SNR_REGION_WIDTH
                xs_snr = self._compute_xs_snr(
                    pos_a_raw, prof_a_raw, pos_b_raw, prof_b_raw,
                    image_a.label, image_b.label, _width)
                if xs_snr:
                    result["xs_snr"] = {k: v for k, v in xs_snr.items() if k != "fig"}
                    figures["xs_snr_profile"] = fig_to_b64(xs_snr["fig"], dpi=150)
                    plt.close(xs_snr["fig"])
            else:
                # Single-image: produce cross-section profile for image A only
                figures["xs_image_profile"] = fig_to_b64(self._plot_image_profile_single(
                    pos_a, prof_a, image_a.label), dpi=150)
                figures["xs_image_profile_raw"] = fig_to_b64(self._plot_image_profile_single(
                    pos_a_raw, prof_a_raw, image_a.label,
                    title="Cross-section brightness profile (raw counts)",
                    ylabel="Pixel value (raw ADU)",
                ), dpi=150)

        result["crosshair"] = crosshair
        result["figures"] = figures
        return result

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _normalise(self, image: AstroImage) -> np.ndarray | None:
        bgsub = image.background_subtracted()
        positive = bgsub[bgsub > 0]
        if positive.size == 0:
            return None
        mean_signal = float(np.mean(positive))
        if mean_signal <= 0:
            return None
        return bgsub / mean_signal

    def _make_masks(self, image: AstroImage,
                     nebula_sigma: float = SECTION8_NEBULA_MASK_SIGMA,
                     dilation_px: int = SECTION8_NEBULA_MASK_DILATION_PX,
                     max_hole_px: int = SECTION8_NEBULA_MASK_MAX_HOLE_PX) -> tuple[np.ndarray, np.ndarray]:
        rms = image.background_rms
        if rms is None:
            rms_val = float(np.std(image.background_subtracted()))
        else:
            rms_val = float(np.median(rms))

        bgsub = image.background_subtracted()
        nebula_mask = bgsub > nebula_sigma * rms_val
        bg_mask = bgsub < 0.5 * rms_val

        # Fallback: use top-5% as nebula if no pixels pass threshold
        if not np.any(nebula_mask):
            threshold = np.percentile(bgsub, 95)
            nebula_mask = bgsub >= threshold

        # Fill small enclosed background gaps, strip small isolated noise-driven
        # specks (same size threshold — a scattered 1-2px false positive at this
        # sigma level would otherwise balloon into a ~(2*dilation_px+1)^2 blob per
        # speck once dilated, polluting blank-sky area far from any real nebula),
        # then grow into adjacent dim/dark transition regions at nebula edges.
        # All three are applied before the bg_mask exclusion below, since any of
        # them can pull previously bg-classified pixels into the nebula mask.
        nebula_mask = self._fill_small_holes(nebula_mask, max_hole_px)
        nebula_mask = self._remove_small_objects(nebula_mask, max_hole_px)
        if dilation_px > 0:
            nebula_mask = binary_dilation(nebula_mask, iterations=dilation_px)

        # Nebula dominates: keep the two masks mutually exclusive after growth.
        bg_mask = bg_mask & ~nebula_mask

        return nebula_mask, bg_mask

    def _fill_small_holes(self, mask: np.ndarray, max_hole_px: int) -> np.ndarray:
        """Fill enclosed background gaps inside mask up to (max_hole_px)**2 area."""
        if max_hole_px <= 0:
            return mask
        filled = binary_fill_holes(mask)
        holes = filled & ~mask
        labeled, n_holes = label(holes)
        if n_holes == 0:
            return mask
        sizes = np.bincount(labeled.ravel())
        max_area = max_hole_px * max_hole_px
        keep = np.zeros(sizes.size, dtype=bool)
        keep[1:] = sizes[1:] <= max_area   # label 0 is background, not a hole
        return mask | keep[labeled]

    def _remove_small_objects(self, mask: np.ndarray, max_size_px: int) -> np.ndarray:
        """Strip isolated foreground specks up to (max_size_px)**2 area.

        Scattered single/few-pixel noise excursions above the nebula sigma
        threshold are common at a loose (~1.7 sigma) cut. Left in place, dilation
        would inflate each one into a much larger blob far from any real nebula
        structure, so small islands are dropped before growth is applied.
        """
        if max_size_px <= 0:
            return mask
        labeled, n_objects = label(mask)
        if n_objects == 0:
            return mask
        sizes = np.bincount(labeled.ravel())
        max_area = max_size_px * max_size_px
        keep = np.zeros(sizes.size, dtype=bool)
        keep[1:] = sizes[1:] > max_area   # label 0 is background; drop small islands
        return keep[labeled]

    @staticmethod
    def _local_maxima_mask(data: np.ndarray, footprint_px: int,
                            prominence_percentile: float,
                            region_px: int,
                            presmooth_sigma: float = 0.0) -> np.ndarray:
        """Binary mask marking the local region of pixels around each detected
        peak in `data` — not just the single maximal pixel.

        Detection runs on a lightly Gaussian-smoothed copy of `data`
        (presmooth_sigma) to suppress single-pixel noise-driven false maxima;
        the mask itself is built from that smoothed detection pass, but
        callers should measure statistics from the raw (unsmoothed) data
        within the returned mask, not from the smoothed copy. A pixel is a
        peak if it equals the max of its own footprint_px neighbourhood
        (scipy.ndimage.maximum_filter) AND exceeds the
        prominence_percentile-th percentile of the smoothed data — the
        percentile floor is relative to the data's own distribution (mirrors
        the SECTION8_LOGRATIO_EPS_PERCENTILE precedent in _log_ratio_map), so
        peak "height" auto-scales per metric instead of needing an absolute
        cutoff. Each surviving peak pixel is then grown by region_px
        (binary_dilation) to cover the local neighbourhood around it — sized
        proportionally to footprint_px by the caller — rather than a single
        pixel.
        """
        if data.size == 0:
            return np.zeros_like(data, dtype=bool)
        smoothed = gaussian_filter(data, sigma=presmooth_sigma) if presmooth_sigma > 0 else data
        footprint = max(3, int(footprint_px) | 1)
        local_max = smoothed == maximum_filter(smoothed, size=footprint)
        threshold = np.percentile(smoothed, prominence_percentile)
        mask = local_max & (smoothed > threshold)
        if region_px > 0 and np.any(mask):
            mask = binary_dilation(mask, iterations=region_px)
        return mask

    @staticmethod
    def _top_percent_mask(abs_a: np.ndarray, abs_b: np.ndarray, top_percent: float) -> np.ndarray:
        """Boolean mask of pixels in the top `top_percent`% of Image A's OR Image B's
        own value distribution -- catches broad bright regions that local-maxima peak
        detection alone would miss (e.g. an extended plateau, not a sharp point peak)."""
        thresh_a = np.percentile(abs_a, 100.0 - top_percent)
        thresh_b = np.percentile(abs_b, 100.0 - top_percent)
        return (abs_a >= thresh_a) | (abs_b >= thresh_b)

    def _combined_localmax_mask(self, abs_a: np.ndarray, abs_b: np.ndarray,
                                 footprint_px: int, prominence_percentile: float,
                                 region_px: int, presmooth_sigma: float,
                                 top_percent: float) -> np.ndarray:
        """Local-maxima peak mask (_local_maxima_mask) unioned with a top-percent
        brightness mask (_top_percent_mask). Used identically by _localmax_entry
        (Section 8j table/figure stats) and the mask-grid figure builder in analyze(),
        so every displayed panel matches the mask actually backing that row's numbers."""
        peak_source = np.maximum(abs_a, abs_b)
        mask = self._local_maxima_mask(peak_source, footprint_px, prominence_percentile,
                                        region_px, presmooth_sigma)
        mask |= self._top_percent_mask(abs_a, abs_b, top_percent)
        return mask

    @staticmethod
    def _localmax_stats(abs_a: np.ndarray, abs_b: np.ndarray,
                         diff: np.ndarray, mask: np.ndarray,
                         rng: np.random.Generator) -> dict:
        """Masked summary stats for one metric/scale's local-maxima mask.

        abs_a/abs_b must already be magnitude (|.|) arrays so wavelet's signed
        reconstructions don't cancel when averaged (a no-op for the other four
        families, which are already >= 0). ratio = 10**mean(diff[mask]) — the
        geometric mean of the per-pixel A/B ratio at the masked pixels,
        expressed as a plain ×-factor; diff is the already-computed
        log10(|A|/|B|) map every family builds via _log_ratio_map, not
        recomputed here. std_a/std_b are the sample standard deviation of the
        masked magnitudes. p_value/cliffs_delta are a Mann-Whitney U test
        (two-sided) + Cliff's delta comparing the full masked-pixel
        populations of A vs B (core.stats_utils.mannwhitney_effect) — delta
        > 0 means A tends higher. log_ratio_mean/log_ratio_std are the mean/standard
        deviation of the per-pixel log10(|A|/|B|) population at the masked pixels
        (masked_diff itself, a genuinely pixel-paired quantity) — log_ratio_mean is
        the same quantity `ratio` is exponentiated from (ratio = 10**log_ratio_mean),
        kept unexponentiated for the Section 8j table/overview plot, which present
        this column directly in log10 space rather than converting back to a linear
        ×-factor. vals_a/vals_b/vals_log_ratio are each a
        SECTION8_LOCALMAX_DIST_MAX_SAMPLES-capped random subsample of the
        corresponding masked population, retained only for the Section 8j
        distribution figures; every other returned value is computed from the
        FULL population. Returns None values / n_px=0 when the mask selects
        no pixels.
        """
        from core.stats_utils import mannwhitney_effect
        h = min(abs_a.shape[0], abs_b.shape[0], diff.shape[0], mask.shape[0])
        w = min(abs_a.shape[1], abs_b.shape[1], diff.shape[1], mask.shape[1])
        m = mask[:h, :w]
        n_px = int(np.count_nonzero(m))
        if n_px == 0:
            return {"mean_a": None, "mean_b": None, "std_a": None, "std_b": None,
                    "ratio": None, "log_ratio_mean": None, "log_ratio_std": None,
                    "p_value": None, "cliffs_delta": None,
                    "n_px": 0, "pct_area": 0.0,
                    "vals_a": np.empty(0, dtype=np.float32), "vals_b": np.empty(0, dtype=np.float32),
                    "vals_log_ratio": np.empty(0, dtype=np.float32)}
        vals_a = abs_a[:h, :w][m]
        vals_b = abs_b[:h, :w][m]
        mean_a, std_a = float(np.mean(vals_a)), float(np.std(vals_a))
        mean_b, std_b = float(np.mean(vals_b)), float(np.std(vals_b))
        masked_diff = diff[:h, :w][m]
        log_ratio_mean = float(np.mean(masked_diff))
        log_ratio_std = float(np.std(masked_diff))
        ratio = float(10.0 ** log_ratio_mean)
        p_value, delta = mannwhitney_effect(vals_a, vals_b)
        sub_a, sub_b, sub_log_ratio = vals_a, vals_b, masked_diff
        if vals_a.size > SECTION8_LOCALMAX_DIST_MAX_SAMPLES:
            sub_a = vals_a[rng.choice(vals_a.size, SECTION8_LOCALMAX_DIST_MAX_SAMPLES, replace=False)]
        if vals_b.size > SECTION8_LOCALMAX_DIST_MAX_SAMPLES:
            sub_b = vals_b[rng.choice(vals_b.size, SECTION8_LOCALMAX_DIST_MAX_SAMPLES, replace=False)]
        if masked_diff.size > SECTION8_LOCALMAX_DIST_MAX_SAMPLES:
            sub_log_ratio = masked_diff[rng.choice(masked_diff.size, SECTION8_LOCALMAX_DIST_MAX_SAMPLES, replace=False)]
        return {"mean_a": mean_a, "mean_b": mean_b, "std_a": std_a, "std_b": std_b,
                "ratio": ratio, "log_ratio_mean": log_ratio_mean, "log_ratio_std": log_ratio_std,
                "p_value": p_value, "cliffs_delta": delta,
                "n_px": n_px, "pct_area": 100.0 * n_px / m.size,
                "vals_a": sub_a.astype(np.float32), "vals_b": sub_b.astype(np.float32),
                "vals_log_ratio": sub_log_ratio.astype(np.float32)}

    def _localmax_entry(self, map_a: np.ndarray, map_b: np.ndarray,
                         diff: np.ndarray, scale_px: float,
                         footprint_mult: float, prominence_percentile: float,
                         region_fraction: float, presmooth_fraction: float,
                         top_percent: float,
                         rng: np.random.Generator) -> dict:
        """One partial['localmax'][key] entry. Builds the local-maxima mask via
        _combined_localmax_mask (peaks over np.maximum(|A|, |B|) — a peak strong in
        either image counts, mirroring the existing Nebula-mask union rationale —
        unioned with the top-percent brightness mask) then returns its masked stats
        measured from the raw (unsmoothed) magnitude maps. footprint_px, region_px
        (the neighbourhood grown around each peak), and presmooth_sigma
        (detection-only smoothing) all scale with this call's own scale_px, so both
        "how local" and "how tall" a peak must be, plus how much denoising happens
        before detection, auto-adapt per metric/scale instead of using one fixed
        setting everywhere.
        """
        h = min(map_a.shape[0], map_b.shape[0])
        w = min(map_a.shape[1], map_b.shape[1])
        abs_a, abs_b = np.abs(map_a[:h, :w]), np.abs(map_b[:h, :w])
        footprint_px = max(3, int(round(footprint_mult * scale_px)) | 1)
        region_px = max(1, int(round(region_fraction * footprint_px)))
        presmooth_sigma = max(0.5, presmooth_fraction * scale_px)
        mask = self._combined_localmax_mask(abs_a, abs_b, footprint_px, prominence_percentile,
                                             region_px, presmooth_sigma, top_percent)
        return self._localmax_stats(abs_a, abs_b, diff, mask, rng)

    def _nebula_bounding_box(self, mask: np.ndarray,
                              shape: tuple) -> tuple[int, int, int, int] | None:
        """Return (r0, r1, c0, c1) bounding box of the nebula mask with 5% padding.
        Returns None if the mask is empty or covers the whole image."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        r0 = int(np.where(rows)[0][0])
        r1 = int(np.where(rows)[0][-1])
        c0 = int(np.where(cols)[0][0])
        c1 = int(np.where(cols)[0][-1])
        pad = max(30, int(0.05 * max(r1 - r0, c1 - c0)))
        r0 = max(0, r0 - pad)
        r1 = min(shape[0], r1 + pad)
        c0 = max(0, c0 - pad)
        c1 = min(shape[1], c1 + pad)
        # Only use the ROI if it covers <90% of the image
        if (r1 - r0) / shape[0] > 0.9 and (c1 - c0) / shape[1] > 0.9:
            return None
        return (r0, r1, c0, c1)

    @staticmethod
    def _smooth_for_display(arr: np.ndarray) -> np.ndarray:
        """Gaussian σ=1.0 smoothing for visualisation only."""
        return gaussian_filter(arr.astype(float), sigma=_DISPLAY_SMOOTH_SIGMA)

    # ------------------------------------------------------------------
    # Local standard deviation maps
    # ------------------------------------------------------------------

    def _std_analysis(self, norm_a, norm_b,
                       mask_neb_a, mask_bg_a,
                       mask_neb_b, mask_bg_b,
                       kernel_sizes, label_a, label_b,
                       display_roi=None,
                       crosshair=None,
                       mask_neb_shared=None,
                       mask_bg_shared=None, rng=None,
                       localmax_footprint_mult=SECTION8_LOCALMAX_FOOTPRINT_MULT,
                       localmax_prominence_percentile=SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                       localmax_region_fraction=SECTION8_LOCALMAX_REGION_FRACTION,
                       localmax_presmooth_fraction=SECTION8_LOCALMAX_PRESMOOTH_FRACTION,
                       localmax_top_percent=SECTION8_LOCALMAX_TOP_PERCENT) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "contrast_ratios_a": {}, "contrast_ratios_b": {},
            "std_nc_score_a": {}, "std_nc_score_b": {},
            "std_nc_noise_a": {}, "std_nc_noise_b": {},
            "std_nc_neb_std_a": {}, "std_nc_neb_std_b": {},
            "panels": {},
            "localmax": {},
            "localmax_log_ratio": {},
            "localmax_log_ratio_err": {},
        }
        single = norm_b is None
        for ks in kernel_sizes:
            std_a = self._compute_std_map(norm_a, ks)
            std_b = self._compute_std_map(norm_b, ks) if not single else None

            # Contrast ratios (computed on unsmoothed maps)
            cr_a = self._contrast_ratio(std_a, mask_neb_a, mask_bg_a)
            partial["contrast_ratios_a"][ks] = cr_a
            if not single:
                cr_b = self._contrast_ratio(std_b, mask_neb_b, mask_bg_b)
                partial["contrast_ratios_b"][ks] = cr_b

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a, neb_std_a = self._nc_score(std_a, mask_neb_shared, mask_bg_a)
                partial["std_nc_score_a"][ks] = nc_a
                partial["std_nc_noise_a"][ks] = noise_a
                partial["std_nc_neb_std_a"][ks] = neb_std_a
                nc_b, noise_b, neb_std_b = self._nc_score(std_b, mask_neb_shared, mask_bg_b)
                partial["std_nc_score_b"][ks] = nc_b
                partial["std_nc_noise_b"][ks] = noise_b
                partial["std_nc_neb_std_b"][ks] = neb_std_b

            diff = self._log_ratio_map(std_a, std_b) if not single else None
            partial["panels"][f"std_{ks}px"] = {
                "a":    std_a.astype(np.float32),
                "b":    std_b.astype(np.float32) if std_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                lm_entry = self._localmax_entry(
                    std_a, std_b, diff, ks,
                    localmax_footprint_mult, localmax_prominence_percentile,
                    localmax_region_fraction, localmax_presmooth_fraction,
                    localmax_top_percent, rng)
                partial["localmax"][f"std_{ks}px"] = lm_entry
                partial["localmax_log_ratio"][ks] = lm_entry["log_ratio_mean"]
                partial["localmax_log_ratio_err"][ks] = lm_entry["log_ratio_std"]
            if not single:
                corr_fig = self._plot_metric_correlation(
                    std_a, std_b, diff, mask_neb_shared, mask_bg_shared,
                    label_a, label_b, f"Local σ (kernel {ks}px)", rng)
                if corr_fig is not None:
                    figures[f"corr_std_{ks}px"] = corr_fig
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_std_{ks}px"] = {
                    "a": (std_a / noise_a).astype(np.float32),
                    "b": (std_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            xs_raw = None
            xs_line = None
            if crosshair is not None and not single:
                pos, pa = self._sample_line(std_a, **crosshair)
                _, pb = self._sample_line(std_b, **crosshair)
                xs_raw = (pos, pa, pb, label_a, label_b,
                          f"Cross-section — Local σ, kernel {ks}px")
                xs_line = self._crosshair_to_cropped_px(crosshair, std_a.shape, SECTION8_BORDER_CROP_FRACTION)

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(std_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(std_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Local σ — kernel {ks}px — {label_a}",
                    f"Local σ — kernel {ks}px — {label_b}",
                    diff_title=f"Log ratio (A/B), kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                    display_roi=None,
                    xs_data=xs_raw,
                    xs_line=xs_line,
                )
            else:
                fig = self._plot_single(
                    self._crop_border(std_a, SECTION8_BORDER_CROP_FRACTION),
                    f"Local σ — kernel {ks}px — {label_a}",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                )
            figures[f"std_{ks}px"] = fig

            if not single and noise_a and noise_b:
                xs_nrm = None
                if crosshair is not None:
                    pos_n, pa_n = self._sample_line(std_a / noise_a, **crosshair)
                    _, pb_n = self._sample_line(std_b / noise_b, **crosshair)
                    xs_nrm = (pos_n, pa_n, pb_n, label_a, label_b,
                              f"Cross-section — Local σ (× noise floor), kernel {ks}px")
                figures[f"nrm_std_{ks}px"] = self._plot_side_by_side(
                    self._crop_border(std_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(std_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Local σ (× noise floor) — kernel {ks}px — {label_a}",
                    f"Local σ (× noise floor) — kernel {ks}px — {label_b}",
                    diff_title=f"Log ratio (A/B), noise-normalised, kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                    xs_data=xs_nrm,
                    xs_line=xs_line,
                )

        return figs_to_b64(figures, dpi=150), partial

    def _compute_std_map(self, norm: np.ndarray, kernel_size: int) -> np.ndarray:
        # Downsample for performance if image is large
        factor = 1.0
        data = norm
        if max(norm.shape) > MAX_DIM_FOR_STD:
            factor = MAX_DIM_FOR_STD / max(norm.shape)
            new_h = int(norm.shape[0] * factor)
            new_w = int(norm.shape[1] * factor)
            data = zoom(norm, (new_h / norm.shape[0], new_w / norm.shape[1]), order=1)
            kernel_size = max(3, int(kernel_size * factor) | 1)

        std_map = generic_filter(data, np.std, size=kernel_size)

        if factor < 1.0:
            std_map = zoom(std_map,
                           (norm.shape[0] / std_map.shape[0],
                            norm.shape[1] / std_map.shape[1]),
                           order=1)
        return std_map

    def _contrast_ratio(self, std_map: np.ndarray,
                         nebula_mask: np.ndarray,
                         bg_mask: np.ndarray) -> float | None:
        neb_vals = std_map[nebula_mask[:std_map.shape[0], :std_map.shape[1]]]
        bg_vals = std_map[bg_mask[:std_map.shape[0], :std_map.shape[1]]]
        if neb_vals.size == 0 or bg_vals.size == 0:
            return None
        bg_med = float(np.median(bg_vals))
        if bg_med <= 0:
            return None
        return float(np.median(neb_vals)) / bg_med

    def _nc_score(self, detail_map: np.ndarray,
                  mask_neb_shared: np.ndarray | None,
                  mask_bg: np.ndarray) -> tuple[float | None, float | None, float | None]:
        """Noise-corrected local-contrast score for one detail map at one scale.

        score = median(|detail|) over the pixels either image classifies as nebula
        (mask_neb_shared), divided by median(|detail|) over THIS image's own
        background mask — its empirical per-scale noise floor for this operator.
        Returns (score, noise_floor, neb_std); all None if a mask selects zero
        pixels, mask_neb_shared is unavailable (single-image mode), or
        noise_floor <= 0. neb_std is the sample standard deviation of the same
        nebula-region |detail| population the median score is computed from —
        used to propagate an approximate uncertainty onto the 8i cross-method
        overview's ratio-of-scores error bars (see _compute_nc_ratio_errors).
        """
        if mask_neb_shared is None:
            return None, None, None
        h = min(detail_map.shape[0], mask_neb_shared.shape[0], mask_bg.shape[0])
        w = min(detail_map.shape[1], mask_neb_shared.shape[1], mask_bg.shape[1])
        absmap = np.abs(detail_map[:h, :w])
        neb_vals = absmap[mask_neb_shared[:h, :w]]
        bg_vals = absmap[mask_bg[:h, :w]]
        if neb_vals.size == 0 or bg_vals.size == 0:
            return None, None, None
        noise_floor = float(bn.median(bg_vals))
        if noise_floor <= 0:
            return None, None, None
        neb_std = float(np.std(neb_vals))
        return float(bn.median(neb_vals)) / noise_floor, noise_floor, neb_std

    @staticmethod
    def _log_ratio_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-pixel log10(|A|/|B|) map, replacing plain A-B difference.

        Sign is discarded (via abs()) before the ratio so the result is always
        well-defined even for map families that can go negative (original
        background-subtracted flux, wavelet band-pass reconstructions) — this
        turns the comparison into "which image shows more structure/contrast at
        this scale", not "which image is signed-brighter". Non-negative families
        (std, |LoG|, gradient magnitude, local entropy) are unaffected by the
        abs() since they're already >= 0.

        Epsilon-floors both operands using a low percentile (not a raw minimum —
        a raw minimum over a multi-megapixel array can be pathologically tiny and
        let a single spurious pixel dominate the log-ratio's dynamic range) of the
        pooled positive values from both inputs, mirroring report_builder.py's
        _power_ratio_db epsilon pattern but adapted for per-pixel map sizes.

        A and B are defensively cropped to their common shape first — two-image
        analysis can reach here with mismatched shapes when astroalign
        registration fails and analysis proceeds unaligned.
        """
        abs_a, abs_b = np.abs(a), np.abs(b)
        h = min(abs_a.shape[0], abs_b.shape[0])
        w = min(abs_a.shape[1], abs_b.shape[1])
        abs_a, abs_b = abs_a[:h, :w], abs_b[:h, :w]

        positive = np.concatenate([abs_a[abs_a > 0].ravel(), abs_b[abs_b > 0].ravel()])
        eps = max(float(np.percentile(positive, SECTION8_LOGRATIO_EPS_PERCENTILE)), 1e-12) \
            if positive.size > 0 else 1e-12

        ratio = np.clip(abs_a, eps, None) / np.clip(abs_b, eps, None)
        return np.log10(ratio).astype(np.float32)

    @staticmethod
    def _log_ratio_color_range(diff: np.ndarray) -> tuple[float, float]:
        """Symmetric (vmin, vmax) for the bwr log-ratio colormap, shared by the
        log-ratio map panel, its histogram, and the correlation scatter dots."""
        d_max = float(np.percentile(np.abs(diff), 99.5)) or 1.0
        return -d_max, d_max

    @staticmethod
    def _compute_nc_ratios(score_a: dict, score_b: dict) -> dict:
        """Per-scale A/B ratio of noise-corrected scores; {} if either side is empty
        (single-image mode)."""
        if not score_a or not score_b:
            return {}
        out = {}
        for scale, va in score_a.items():
            vb = score_b.get(scale)
            out[scale] = None if (va is None or vb is None or vb == 0) else va / vb
        return out

    @staticmethod
    def _compute_nc_ratio_errors(ratio: dict, score_a: dict, score_b: dict,
                                  noise_a: dict, noise_b: dict,
                                  neb_std_a: dict, neb_std_b: dict) -> dict:
        """Approximate symmetric uncertainty on each scale's nc_ratio, propagated
        from the coefficient of variation (std/median) of each image's own
        nebula-region pixel population -- a standard relative-uncertainty
        propagation for a ratio of two independent quantities. This is NOT a
        formal confidence interval on the median; it is an approximate indicator
        of spread, captioned as such in the report (see 8i methodology).
        median_neb is recovered as score * noise_floor (score = median_neb /
        noise_floor by construction in _nc_score) rather than recomputed.
        """
        out = {}
        for scale, r in ratio.items():
            sa, sb = score_a.get(scale), score_b.get(scale)
            na, nb = noise_a.get(scale), noise_b.get(scale)
            sda, sdb = neb_std_a.get(scale), neb_std_b.get(scale)
            if None in (r, sa, sb, na, nb, sda, sdb):
                out[scale] = None
                continue
            median_neb_a, median_neb_b = sa * na, sb * nb
            if median_neb_a == 0 or median_neb_b == 0:
                out[scale] = None
                continue
            cv_a, cv_b = sda / median_neb_a, sdb / median_neb_b
            out[scale] = abs(r) * (cv_a ** 2 + cv_b ** 2) ** 0.5
        return out

    # ------------------------------------------------------------------
    # Laplacian of Gaussian maps
    # ------------------------------------------------------------------

    def _log_analysis(self, norm_a, norm_b, sigmas,
                       label_a, label_b,
                       display_roi=None,
                       crosshair=None,
                       mask_neb_shared=None, mask_bg_a=None, mask_bg_b=None,
                       mask_bg_shared=None, rng=None,
                       localmax_footprint_mult=SECTION8_LOCALMAX_FOOTPRINT_MULT,
                       localmax_prominence_percentile=SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                       localmax_region_fraction=SECTION8_LOCALMAX_REGION_FRACTION,
                       localmax_presmooth_fraction=SECTION8_LOCALMAX_PRESMOOTH_FRACTION,
                       localmax_top_percent=SECTION8_LOCALMAX_TOP_PERCENT) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "log_nc_score_a": {}, "log_nc_score_b": {},
            "log_nc_noise_a": {}, "log_nc_noise_b": {},
            "log_nc_neb_std_a": {}, "log_nc_neb_std_b": {},
            "panels": {},
            "localmax": {},
            "localmax_log_ratio": {},
            "localmax_log_ratio_err": {},
        }
        single = norm_b is None
        for sigma in sigmas:
            log_a = np.abs(gaussian_laplace(norm_a, sigma=sigma))
            log_b = np.abs(gaussian_laplace(norm_b, sigma=sigma)) if not single else None

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a, neb_std_a = self._nc_score(log_a, mask_neb_shared, mask_bg_a)
                partial["log_nc_score_a"][sigma] = nc_a
                partial["log_nc_noise_a"][sigma] = noise_a
                partial["log_nc_neb_std_a"][sigma] = neb_std_a
                nc_b, noise_b, neb_std_b = self._nc_score(log_b, mask_neb_shared, mask_bg_b)
                partial["log_nc_score_b"][sigma] = nc_b
                partial["log_nc_noise_b"][sigma] = noise_b
                partial["log_nc_neb_std_b"][sigma] = neb_std_b

            diff = self._log_ratio_map(log_a, log_b) if log_b is not None else None
            partial["panels"][f"log_{sigma}"] = {
                "a":    log_a.astype(np.float32),
                "b":    log_b.astype(np.float32) if log_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                lm_entry = self._localmax_entry(
                    log_a, log_b, diff, sigma,
                    localmax_footprint_mult, localmax_prominence_percentile,
                    localmax_region_fraction, localmax_presmooth_fraction,
                    localmax_top_percent, rng)
                partial["localmax"][f"log_{sigma}"] = lm_entry
                partial["localmax_log_ratio"][sigma] = lm_entry["log_ratio_mean"]
                partial["localmax_log_ratio_err"][sigma] = lm_entry["log_ratio_std"]
            if not single:
                corr_fig = self._plot_metric_correlation(
                    log_a, log_b, diff, mask_neb_shared, mask_bg_shared,
                    label_a, label_b, f"|LoG| (σ={sigma}px)", rng)
                if corr_fig is not None:
                    figures[f"corr_log_{sigma}"] = corr_fig
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_log_{sigma}"] = {
                    "a": (log_a / noise_a).astype(np.float32),
                    "b": (log_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            xs_raw = None
            xs_line = None
            if crosshair is not None and not single:
                pos, pa = self._sample_line(log_a, **crosshair)
                _, pb = self._sample_line(log_b, **crosshair)
                xs_raw = (pos, pa, pb, label_a, label_b,
                          f"Cross-section — |LoG|, σ={sigma}px")
                xs_line = self._crosshair_to_cropped_px(crosshair, log_a.shape, SECTION8_BORDER_CROP_FRACTION)

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(log_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(log_b, SECTION8_BORDER_CROP_FRACTION),
                    f"|LoG| σ={sigma}px — {label_a}",
                    f"|LoG| σ={sigma}px — {label_b}",
                    diff_title=f"|LoG| log-ratio (A/B), σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                    display_roi=None,
                    xs_data=xs_raw,
                    xs_line=xs_line,
                )
            else:
                fig = self._plot_single(
                    self._crop_border(log_a, SECTION8_BORDER_CROP_FRACTION),
                    f"|LoG| σ={sigma}px — {label_a}",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                )
            figures[f"log_sigma{sigma}"] = fig

            if not single and noise_a and noise_b:
                xs_nrm = None
                if crosshair is not None:
                    pos_n, pa_n = self._sample_line(log_a / noise_a, **crosshair)
                    _, pb_n = self._sample_line(log_b / noise_b, **crosshair)
                    xs_nrm = (pos_n, pa_n, pb_n, label_a, label_b,
                              f"Cross-section — |LoG| (× noise floor), σ={sigma}px")
                figures[f"nrm_log_{sigma}"] = self._plot_side_by_side(
                    self._crop_border(log_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(log_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"|LoG| (× noise floor) σ={sigma}px — {label_a}",
                    f"|LoG| (× noise floor) σ={sigma}px — {label_b}",
                    diff_title=f"Log ratio (A/B), noise-normalised, σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                    xs_data=xs_nrm,
                    xs_line=xs_line,
                )
        return figs_to_b64(figures, dpi=150), partial

    # ------------------------------------------------------------------
    # Gradient magnitude (edge sharpness)
    # ------------------------------------------------------------------

    def _gradient_analysis(self, norm_a, norm_b, sigmas,
                            label_a, label_b,
                            display_roi=None,
                            crosshair=None,
                            mask_neb_shared=None, mask_bg_a=None, mask_bg_b=None,
                            mask_bg_shared=None, rng=None,
                            localmax_footprint_mult=SECTION8_LOCALMAX_FOOTPRINT_MULT,
                            localmax_prominence_percentile=SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                            localmax_region_fraction=SECTION8_LOCALMAX_REGION_FRACTION,
                            localmax_presmooth_fraction=SECTION8_LOCALMAX_PRESMOOTH_FRACTION,
                            localmax_top_percent=SECTION8_LOCALMAX_TOP_PERCENT) -> tuple[dict, dict]:
        """G = |gradient| at Gaussian scale sigma (first spatial derivative magnitude).
        Reuses the LOG_SIGMAS scale set so gradient and |LoG| are directly comparable
        at identical spatial scales. Structured identically to _log_analysis."""
        figures = {}
        partial: dict = {
            "gm_nc_score_a": {}, "gm_nc_score_b": {},
            "gm_nc_noise_a": {}, "gm_nc_noise_b": {},
            "gm_nc_neb_std_a": {}, "gm_nc_neb_std_b": {},
            "panels": {},
            "localmax": {},
            "localmax_log_ratio": {},
            "localmax_log_ratio_err": {},
        }
        single = norm_b is None
        for sigma in sigmas:
            gm_a = gaussian_gradient_magnitude(norm_a, sigma=sigma)
            gm_b = gaussian_gradient_magnitude(norm_b, sigma=sigma) if not single else None

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a, neb_std_a = self._nc_score(gm_a, mask_neb_shared, mask_bg_a)
                partial["gm_nc_score_a"][sigma] = nc_a
                partial["gm_nc_noise_a"][sigma] = noise_a
                partial["gm_nc_neb_std_a"][sigma] = neb_std_a
                nc_b, noise_b, neb_std_b = self._nc_score(gm_b, mask_neb_shared, mask_bg_b)
                partial["gm_nc_score_b"][sigma] = nc_b
                partial["gm_nc_noise_b"][sigma] = noise_b
                partial["gm_nc_neb_std_b"][sigma] = neb_std_b

            diff = self._log_ratio_map(gm_a, gm_b) if gm_b is not None else None
            partial["panels"][f"gradient_{sigma}"] = {
                "a":    gm_a.astype(np.float32),
                "b":    gm_b.astype(np.float32) if gm_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                lm_entry = self._localmax_entry(
                    gm_a, gm_b, diff, sigma,
                    localmax_footprint_mult, localmax_prominence_percentile,
                    localmax_region_fraction, localmax_presmooth_fraction,
                    localmax_top_percent, rng)
                partial["localmax"][f"gradient_{sigma}"] = lm_entry
                partial["localmax_log_ratio"][sigma] = lm_entry["log_ratio_mean"]
                partial["localmax_log_ratio_err"][sigma] = lm_entry["log_ratio_std"]
            if not single:
                corr_fig = self._plot_metric_correlation(
                    gm_a, gm_b, diff, mask_neb_shared, mask_bg_shared,
                    label_a, label_b, f"Gradient |G| (σ={sigma}px)", rng)
                if corr_fig is not None:
                    figures[f"corr_gradient_{sigma}"] = corr_fig
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_gradient_{sigma}"] = {
                    "a": (gm_a / noise_a).astype(np.float32),
                    "b": (gm_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            xs_raw = None
            xs_line = None
            if crosshair is not None and not single:
                pos, pa = self._sample_line(gm_a, **crosshair)
                _, pb = self._sample_line(gm_b, **crosshair)
                xs_raw = (pos, pa, pb, label_a, label_b,
                          f"Cross-section — Gradient, σ={sigma}px")
                xs_line = self._crosshair_to_cropped_px(crosshair, gm_a.shape, SECTION8_BORDER_CROP_FRACTION)

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(gm_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(gm_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Gradient |G| σ={sigma}px — {label_a}",
                    f"Gradient |G| σ={sigma}px — {label_b}",
                    diff_title=f"Gradient log-ratio (A/B), σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                    display_roi=None,
                    xs_data=xs_raw,
                    xs_line=xs_line,
                )
            else:
                fig = self._plot_single(
                    self._crop_border(gm_a, SECTION8_BORDER_CROP_FRACTION),
                    f"Gradient |G| σ={sigma}px — {label_a}",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                )
            figures[f"gradient_{sigma}"] = fig

            if not single and noise_a and noise_b:
                xs_nrm = None
                if crosshair is not None:
                    pos_n, pa_n = self._sample_line(gm_a / noise_a, **crosshair)
                    _, pb_n = self._sample_line(gm_b / noise_b, **crosshair)
                    xs_nrm = (pos_n, pa_n, pb_n, label_a, label_b,
                              f"Cross-section — Gradient (× noise floor), σ={sigma}px")
                figures[f"nrm_gradient_{sigma}"] = self._plot_side_by_side(
                    self._crop_border(gm_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(gm_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Gradient (× noise floor) σ={sigma}px — {label_a}",
                    f"Gradient (× noise floor) σ={sigma}px — {label_b}",
                    diff_title=f"Log ratio (A/B), noise-normalised, σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                    xs_data=xs_nrm,
                    xs_line=xs_line,
                )
        return figs_to_b64(figures, dpi=150), partial

    # ------------------------------------------------------------------
    # Wavelet decomposition
    # ------------------------------------------------------------------

    def _wavelet_analysis(self, norm_a, norm_b, wavelet, levels,
                           label_a, label_b,
                           display_roi=None,
                           crosshair=None,
                           mask_neb_shared=None, mask_bg_a=None, mask_bg_b=None,
                           mask_bg_shared=None, rng=None,
                           localmax_footprint_mult=SECTION8_LOCALMAX_FOOTPRINT_MULT,
                           localmax_prominence_percentile=SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                           localmax_region_fraction=SECTION8_LOCALMAX_REGION_FRACTION,
                           localmax_presmooth_fraction=SECTION8_LOCALMAX_PRESMOOTH_FRACTION,
                           localmax_top_percent=SECTION8_LOCALMAX_TOP_PERCENT) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "sigma_noise_a": None, "sigma_noise_b": None,
            "wavelet_snr_a": {}, "wavelet_snr_b": {},
            "wavelet_nc_score_a": {}, "wavelet_nc_score_b": {},
            "wavelet_nc_noise_a": {}, "wavelet_nc_noise_b": {},
            "wavelet_nc_neb_std_a": {}, "wavelet_nc_neb_std_b": {},
            "panels": {},
            "localmax": {},
            "localmax_log_ratio": {},
            "localmax_log_ratio_err": {},
        }
        single = norm_b is None

        coeffs_a = pywt.wavedec2(norm_a, wavelet, level=levels, mode="periodization")
        coeffs_b = pywt.wavedec2(norm_b, wavelet, level=levels, mode="periodization") if not single else None

        sigma_a = self._estimate_noise(coeffs_a)
        sigma_b = self._estimate_noise(coeffs_b) if coeffs_b is not None else None
        partial["sigma_noise_a"] = sigma_a
        partial["sigma_noise_b"] = sigma_b

        # Per-level SNR (index 1 = coarsest detail, -1 = finest detail)
        for lvl_idx in range(1, levels + 1):
            coeff_idx = levels + 1 - lvl_idx  # map human level to list index
            snr_a = self._level_snr(coeffs_a, coeff_idx, sigma_a, lvl_idx)
            snr_b = self._level_snr(coeffs_b, coeff_idx, sigma_b, lvl_idx) if coeffs_b is not None else None
            partial["wavelet_snr_a"][lvl_idx] = snr_a
            partial["wavelet_snr_b"][lvl_idx] = snr_b

        # SNR bar chart
        figures["wavelet_snr"] = self._plot_snr_bars(
            partial["wavelet_snr_a"], partial["wavelet_snr_b"], label_a, label_b, levels)

        # Reconstruct every level for noise-corrected scoring (raw subband coefficients
        # live at reduced spatial resolution and don't align pixel-for-pixel with the
        # full-resolution nebula mask — only the inverse-DWT output does). Display
        # figures/panels stay restricted to levels 2-3 ("best signal content").
        for human_level in range(1, levels + 1):
            coeff_idx = levels + 1 - human_level
            rec_a = self._reconstruct_level(coeffs_a, coeff_idx, wavelet, levels)
            rec_b = self._reconstruct_level(coeffs_b, coeff_idx, wavelet, levels) if coeffs_b is not None else None

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a, neb_std_a = self._nc_score(rec_a, mask_neb_shared, mask_bg_a)
                partial["wavelet_nc_score_a"][human_level] = nc_a
                partial["wavelet_nc_noise_a"][human_level] = noise_a
                partial["wavelet_nc_neb_std_a"][human_level] = neb_std_a
                nc_b, noise_b, neb_std_b = self._nc_score(rec_b, mask_neb_shared, mask_bg_b)
                partial["wavelet_nc_score_b"][human_level] = nc_b
                partial["wavelet_nc_noise_b"][human_level] = noise_b
                partial["wavelet_nc_neb_std_b"][human_level] = neb_std_b

            if human_level not in (2, 3):
                continue   # display/panels only for levels 2-3, unchanged from prior behaviour

            display_level = human_level
            diff = self._log_ratio_map(rec_a, rec_b) if rec_b is not None else None
            partial["panels"][f"wavelet_{display_level}"] = {
                "a":    rec_a.astype(np.float32),
                "b":    rec_b.astype(np.float32) if rec_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                lm_entry = self._localmax_entry(
                    rec_a, rec_b, diff, 2 ** display_level,
                    localmax_footprint_mult, localmax_prominence_percentile,
                    localmax_region_fraction, localmax_presmooth_fraction,
                    localmax_top_percent, rng)
                partial["localmax"][f"wavelet_{display_level}"] = lm_entry
                partial["localmax_log_ratio"][display_level] = lm_entry["log_ratio_mean"]
                partial["localmax_log_ratio_err"][display_level] = lm_entry["log_ratio_std"]
            if not single:
                # Raw signed reconstructions (not abs()) — complementary to the
                # sign-discarding log-ratio map, shows whether band-pass detail
                # flips sign between the two filters at a given pixel.
                corr_fig = self._plot_metric_correlation(
                    rec_a, rec_b, diff, mask_neb_shared, mask_bg_shared,
                    label_a, label_b, f"Wavelet level {display_level}", rng)
                if corr_fig is not None:
                    figures[f"corr_wavelet_{display_level}"] = corr_fig
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_wavelet_{display_level}"] = {
                    "a": (rec_a / noise_a).astype(np.float32),
                    "b": (rec_b / noise_b).astype(np.float32),
                    "diff": None,
                }
            xs_raw = None
            xs_line = None
            if crosshair is not None and not single:
                pos, pa = self._sample_line(rec_a, **crosshair)
                _, pb = self._sample_line(rec_b, **crosshair)
                xs_raw = (pos, pa, pb, label_a, label_b,
                          f"Cross-section — Wavelet level {display_level}")
                xs_line = self._crosshair_to_cropped_px(crosshair, rec_a.shape, SECTION8_BORDER_CROP_FRACTION)

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(rec_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(rec_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Wavelet level {display_level} — {label_a}",
                    f"Wavelet level {display_level} — {label_b}",
                    diff_title=f"Level {display_level} log-ratio (A/B)",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                    xs_data=xs_raw,
                    xs_line=xs_line,
                )
            else:
                fig = self._plot_single(
                    self._crop_border(rec_a, SECTION8_BORDER_CROP_FRACTION),
                    f"Wavelet level {display_level} — {label_a}",
                    cmap=SECTION8_ANALYSIS_CMAP,
                )
            figures[f"wavelet_level{display_level}"] = fig

            if not single and noise_a and noise_b:
                xs_nrm = None
                if crosshair is not None:
                    pos_n, pa_n = self._sample_line(rec_a / noise_a, **crosshair)
                    _, pb_n = self._sample_line(rec_b / noise_b, **crosshair)
                    xs_nrm = (pos_n, pa_n, pb_n, label_a, label_b,
                              f"Cross-section — Wavelet level {display_level} (× noise floor)")
                figures[f"nrm_wavelet_{display_level}"] = self._plot_side_by_side(
                    self._crop_border(rec_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(rec_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Wavelet level {display_level} (× noise floor) — {label_a}",
                    f"Wavelet level {display_level} (× noise floor) — {label_b}",
                    diff_title=f"Level {display_level} log-ratio (A/B), noise-normalised",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                    xs_data=xs_nrm,
                    xs_line=xs_line,
                )

        return figs_to_b64(figures, dpi=150), partial

    def _estimate_noise(self, coeffs) -> float:
        # Finest-level horizontal detail (last element, first sub-band)
        lh1 = coeffs[-1][0]
        return float(bn.median(np.abs(lh1))) / 0.6745

    def _level_snr(self, coeffs, coeff_idx: int,
                    sigma_noise: float, human_level: int) -> float | None:
        if coeff_idx < 1 or coeff_idx >= len(coeffs):
            return None
        lh, hl, hh = coeffs[coeff_idx]
        signal_energy = float(np.sum(lh**2 + hl**2 + hh**2))
        # Noise amplifies by sqrt(3) * 2^(level/2) at each wavelet level
        noise_amp = sigma_noise * np.sqrt(3.0) * (2.0 ** (human_level / 2.0))
        noise_energy = (noise_amp ** 2) * lh.size
        if noise_energy <= 0:
            return None
        return signal_energy / noise_energy

    def _reconstruct_level(self, coeffs, target_coeff_idx: int,
                             wavelet: str, levels: int) -> np.ndarray:
        zeroed = [np.zeros_like(coeffs[0])]
        for i, detail in enumerate(coeffs[1:], start=1):
            if i == target_coeff_idx:
                zeroed.append(detail)
            else:
                zeroed.append(tuple(np.zeros_like(d) for d in detail))
        return pywt.waverec2(zeroed, wavelet, mode='periodization')

    # ------------------------------------------------------------------
    # Shared plotting helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Local entropy maps
    # ------------------------------------------------------------------

    def _entropy_analysis(self, norm_a, norm_b,
                        mask_neb_a, mask_bg_a,
                        mask_neb_b, mask_bg_b,
                        kernel_sizes, label_a, label_b,
                        display_roi=None,
                        crosshair=None,
                        mask_neb_shared=None,
                        mask_bg_shared=None, rng=None,
                        localmax_footprint_mult=SECTION8_LOCALMAX_FOOTPRINT_MULT,
                        localmax_prominence_percentile=SECTION8_LOCALMAX_PROMINENCE_PERCENTILE,
                        localmax_region_fraction=SECTION8_LOCALMAX_REGION_FRACTION,
                        localmax_presmooth_fraction=SECTION8_LOCALMAX_PRESMOOTH_FRACTION,
                        localmax_top_percent=SECTION8_LOCALMAX_TOP_PERCENT) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "entropy_contrast_ratio_a": {}, "entropy_contrast_ratio_b": {},
            "entropy_nc_score_a": {}, "entropy_nc_score_b": {},
            "entropy_nc_noise_a": {}, "entropy_nc_noise_b": {},
            "entropy_nc_neb_std_a": {}, "entropy_nc_neb_std_b": {},
            "panels": {},
            "localmax": {},
            "localmax_log_ratio": {},
            "localmax_log_ratio_err": {},
        }
        single = norm_b is None
        for ks in kernel_sizes:
            ent_a = self._compute_entropy_map(norm_a, ks)
            ent_b = self._compute_entropy_map(norm_b, ks) if not single else None

            # Contrast ratios (computed on unsmoothed maps)
            cr_a = self._contrast_ratio(ent_a, mask_neb_a, mask_bg_a)
            partial["entropy_contrast_ratio_a"][ks] = cr_a
            if not single:
                cr_b = self._contrast_ratio(ent_b, mask_neb_b, mask_bg_b)
                partial["entropy_contrast_ratio_b"][ks] = cr_b

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a, neb_std_a = self._nc_score(ent_a, mask_neb_shared, mask_bg_a)
                partial["entropy_nc_score_a"][ks] = nc_a
                partial["entropy_nc_noise_a"][ks] = noise_a
                partial["entropy_nc_neb_std_a"][ks] = neb_std_a
                nc_b, noise_b, neb_std_b = self._nc_score(ent_b, mask_neb_shared, mask_bg_b)
                partial["entropy_nc_score_b"][ks] = nc_b
                partial["entropy_nc_noise_b"][ks] = noise_b
                partial["entropy_nc_neb_std_b"][ks] = neb_std_b

            diff = self._log_ratio_map(ent_a, ent_b) if not single else None
            partial["panels"][f"entropy_{ks}px"] = {
                "a":    ent_a.astype(np.float32),
                "b":    ent_b.astype(np.float32) if ent_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                lm_entry = self._localmax_entry(
                    ent_a, ent_b, diff, ks,
                    localmax_footprint_mult, localmax_prominence_percentile,
                    localmax_region_fraction, localmax_presmooth_fraction,
                    localmax_top_percent, rng)
                partial["localmax"][f"entropy_{ks}px"] = lm_entry
                partial["localmax_log_ratio"][ks] = lm_entry["log_ratio_mean"]
                partial["localmax_log_ratio_err"][ks] = lm_entry["log_ratio_std"]
            if not single:
                corr_fig = self._plot_metric_correlation(
                    ent_a, ent_b, diff, mask_neb_shared, mask_bg_shared,
                    label_a, label_b, f"Local entropy (kernel {ks}px)", rng)
                if corr_fig is not None:
                    figures[f"corr_entropy_{ks}px"] = corr_fig
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_entropy_{ks}px"] = {
                    "a": (ent_a / noise_a).astype(np.float32),
                    "b": (ent_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            xs_raw = None
            xs_line = None
            if crosshair is not None and not single:
                pos, pa = self._sample_line(ent_a, **crosshair)
                _, pb = self._sample_line(ent_b, **crosshair)
                xs_raw = (pos, pa, pb, label_a, label_b,
                          f"Cross-section — Local entropy, kernel {ks}px")
                xs_line = self._crosshair_to_cropped_px(crosshair, ent_a.shape, SECTION8_BORDER_CROP_FRACTION)

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(ent_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(ent_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Local entropy — kernel {ks}px — {label_a}",
                    f"Local entropy — kernel {ks}px — {label_b}",
                    diff_title=f"Log ratio (A/B), kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                    display_roi=None,       # _crop_border already applied
                    xs_data=xs_raw,
                    xs_line=xs_line,
                )
            else:
                fig = self._plot_single(
                    self._crop_border(ent_a, SECTION8_BORDER_CROP_FRACTION),
                    f"Local entropy — kernel {ks}px — {label_a}",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                )
            figures[f"entropy_{ks}px"] = fig

            if not single and noise_a and noise_b:
                xs_nrm = None
                if crosshair is not None:
                    pos_n, pa_n = self._sample_line(ent_a / noise_a, **crosshair)
                    _, pb_n = self._sample_line(ent_b / noise_b, **crosshair)
                    xs_nrm = (pos_n, pa_n, pb_n, label_a, label_b,
                              f"Cross-section — Local entropy (× noise floor), kernel {ks}px")
                figures[f"nrm_entropy_{ks}px"] = self._plot_side_by_side(
                    self._crop_border(ent_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(ent_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Local entropy (× noise floor) — kernel {ks}px — {label_a}",
                    f"Local entropy (× noise floor) — kernel {ks}px — {label_b}",
                    diff_title=f"Log ratio (A/B), noise-normalised, kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                    xs_data=xs_nrm,
                    xs_line=xs_line,
                )

        return figs_to_b64(figures, dpi=150), partial

    def _compute_entropy_map(self, norm: np.ndarray, kernel_size: int,
                              n_bins: int = SECTION8_ENTROPY_N_BINS) -> np.ndarray:
        """Local Shannon entropy (bits, log2) of the gray-level histogram within a
        square window. norm is deliberately quantized into n_bins integer levels
        from a percentile-clipped range *before* filtering -- computing entropy
        directly on continuous float32 data returns ~log2(window_area) almost
        everywhere (every pixel value in a window is unique), measuring float
        precision rather than genuine tonal diversity. The clip/bin range is
        computed independently per image (matches the existing per-image
        convention used by every other Section 8 map).

        Vectorized via n_bins uniform_filter passes (one per gray level) rather
        than a per-pixel generic_filter callback: uniform_filter(mask_b) is a
        box filter of a 0/1 mask, which *is* exactly the local proportion p_b
        of bin b within each window -- summing -p_b*log2(p_b) across bins then
        gives the entropy map with no per-pixel Python-level histogram ever
        built. A per-pixel generic_filter callback (the initial approach here,
        mirroring _compute_std_map) measured ~10-50x slower than Weber's fully
        vectorized maximum/minimum/median_filter maps at the same array size;
        this reuses that same vectorization principle for entropy.
        """
        factor = 1.0
        data = norm
        if max(norm.shape) > MAX_DIM_FOR_STD:
            factor = MAX_DIM_FOR_STD / max(norm.shape)
            new_h = int(norm.shape[0] * factor)
            new_w = int(norm.shape[1] * factor)
            data = zoom(norm, (new_h / norm.shape[0], new_w / norm.shape[1]), order=1)
            kernel_size = max(3, int(kernel_size * factor) | 1)

        lo, hi = np.percentile(data, [SECTION8_ENTROPY_CLIP_PERCENTILE,
                                       100 - SECTION8_ENTROPY_CLIP_PERCENTILE])
        if hi <= lo:
            entropy_map = np.zeros_like(data, dtype=np.float64)
        else:
            binned = np.clip(((data - lo) / (hi - lo) * n_bins), 0, n_bins - 1).astype(np.intp)
            entropy_map = np.zeros(data.shape, dtype=np.float64)
            for b in range(n_bins):
                p_b = uniform_filter((binned == b).astype(np.float64), size=kernel_size, mode="reflect")
                with np.errstate(divide="ignore", invalid="ignore"):
                    entropy_map -= np.where(p_b > 0, p_b * np.log2(p_b), 0.0)

        if factor < 1.0:
            entropy_map = zoom(entropy_map,
                                (norm.shape[0] / entropy_map.shape[0],
                                 norm.shape[1] / entropy_map.shape[1]),
                                order=1)
        return entropy_map.astype(np.float32)

    def _plot_side_by_side(self, arr_a: np.ndarray, arr_b: np.ndarray,
                            title_a: str, title_b: str,
                            diff_title: str = "",
                            cmap: str = "viridis",
                            nonlinear_norm: bool = False,
                            display_roi=None,
                            smooth_display: bool = True,
                            xs_data: tuple | None = None,
                            xs_line: tuple | None = None) -> plt.Figure:
        """xs_data, if given, is (pos, prof_a, prof_b, label_a, label_b, xs_title)
        for the embedded cross-section panel; None leaves that panel blank.
        xs_line, if given, is (x0, y0, x1, y1) pixel coords (in arr_a/arr_b's own
        frame, post any cropping already applied by the caller) of the user's
        cross-section line, overlaid directly on the Image A/B panels above."""
        # Crop to bright-feature ROI if available
        if display_roi is not None:
            r0, r1, c0, c1 = display_roi
            arr_a = arr_a[r0:r1, c0:c1]
            arr_b = arr_b[r0:r1, c0:c1]

        # Smooth for display only (does not affect any metric values)
        if smooth_display:
            arr_a = self._smooth_for_display(arr_a)
            arr_b = self._smooth_for_display(arr_b)

        # Shared color scale: use percentile clipping to prevent bright outliers
        # from compressing the interesting nebula detail range.
        vmin = max(0.0, float(min(np.percentile(arr_a, 0.5), np.percentile(arr_b, 0.5))))
        vmax = float(max(np.percentile(arr_a, 99.5), np.percentile(arr_b, 99.5)))
        if vmax <= vmin:
            vmax = vmin + 1e-9

        # Sqrt (PowerNorm gamma=0.5) compresses bright stars, reveals faint nebula
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax) if nonlinear_norm else None

        # Log-ratio panel (helper handles any shape mismatch defensively)
        diff = self._log_ratio_map(arr_a, arr_b)

        # Symmetric about zero so the "bwr" midpoint (white) always means no difference.
        dvmin, dvmax = self._log_ratio_color_range(diff)

        # Dark-mode-aware reference-line color (project convention).
        is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        orig_color = "white" if is_dark else "black"

        # 3-row grid: A|B on top, log-ratio diff | cross-section in the middle,
        # a log-ratio pixel-value histogram spanning both columns on the bottom.
        # A/B/diff share the source array's pixel aspect ratio (aspect="equal"
        # imshow); the cross-section panel is a line plot with no such constraint
        # but occupies an equal-size grid cell so the other three panels stay
        # geometrically identical whether or not a crosshair (and thus xs_data)
        # is set.
        h, w = arr_a.shape[:2]
        aspect_ratio = h / max(w, 1)
        panel_w = 5.0   # half the old single-column width — 2 columns now share it
        panel_h = panel_w * aspect_ratio
        hist_h = 2.2
        fig = plt.figure(figsize=(panel_w * 2, panel_h * 2 + hist_h + 1.5),
                          constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[panel_h, panel_h, hist_h])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_diff = fig.add_subplot(gs[1, 0])
        ax_xs = fig.add_subplot(gs[1, 1])
        ax_hist = fig.add_subplot(gs[2, :])

        for ax, arr, title in zip([ax_a, ax_b], [arr_a, arr_b], [title_a, title_b]):
            im = ax.imshow(arr, origin="upper", cmap=cmap,
                           norm=norm if norm is not None else None,
                           vmin=None if norm is not None else vmin,
                           vmax=None if norm is not None else vmax,
                           interpolation="nearest", aspect="equal")
            if xs_line is not None:
                # Lock the view before plotting — otherwise matplotlib autoscales to
                # include line endpoints outside the (already-cropped) array, adding
                # unwanted blank padding around the image.
                ax.set_xlim(-0.5, arr.shape[1] - 0.5)
                ax.set_ylim(arr.shape[0] - 0.5, -0.5)   # origin="upper"
                lx0, ly0, lx1, ly1 = xs_line
                ax.plot([lx0, lx1], [ly0, ly1], color="#ff7f0e",
                        linewidth=1.5, alpha=XS_LINE_ALPHA, zorder=5)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        im_diff = ax_diff.imshow(diff, origin="upper", cmap="bwr",
                                  vmin=dvmin, vmax=dvmax,
                                  interpolation="nearest", aspect="equal")
        ax_diff.set_title(diff_title, fontsize=10)
        ax_diff.axis("off")
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

        if xs_data is not None:
            pos, prof_a, prof_b, xs_label_a, xs_label_b, xs_title = xs_data
            self._draw_cross_section(ax_xs, pos, prof_a, prof_b,
                                      xs_label_a, xs_label_b, xs_title)
        else:
            ax_xs.axis("off")

        # Histogram of the log-ratio map's pixel distribution, colored to match
        # the diff panel above: full data range on the x-axis (no pixels hidden),
        # but each bin's fill color is clipped to [dvmin, dvmax] so extreme-tail
        # bins saturate to the same end colors imshow already uses for its own
        # outliers.
        counts, bin_edges, patches = ax_hist.hist(diff.ravel(), bins=60)
        hist_norm = mcolors.Normalize(vmin=dvmin, vmax=dvmax, clip=True)
        hist_cmap = plt.get_cmap("bwr")
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for patch, center in zip(patches, bin_centers):
            patch.set_facecolor(hist_cmap(hist_norm(center)))
        ax_hist.axvline(0.0, color=orig_color, linestyle="--", linewidth=1.0, label="A = B")
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel("Log ratio, log10(|A|/|B|)", fontsize=8)
        ax_hist.set_ylabel("Pixel count", fontsize=8)
        ax_hist.tick_params(labelsize=7)
        ax_hist.legend(fontsize=6.5, loc="upper right")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_title("Log-ratio pixel distribution", fontsize=9)

        return fig

    def _plot_single(self, arr_a: np.ndarray, title_a: str,
                     cmap: str = "viridis",
                     nonlinear_norm: bool = False,
                     smooth_display: bool = True) -> plt.Figure:
        if smooth_display:
            arr_a = self._smooth_for_display(arr_a)
        vmin = float(np.percentile(arr_a, 0.5))
        vmax = float(np.percentile(arr_a, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax) if nonlinear_norm else None
        h, w = arr_a.shape[:2]
        aspect_ratio = h / max(w, 1)
        panel_w = 10.0
        panel_h = panel_w * aspect_ratio
        fig, ax = plt.subplots(1, 1, figsize=(panel_w, panel_h), constrained_layout=True)
        im = ax.imshow(arr_a, origin="upper", cmap=cmap,
                       norm=norm if norm is not None else None,
                       vmin=None if norm is not None else vmin,
                       vmax=None if norm is not None else vmax,
                       interpolation="nearest", aspect="equal")
        ax.set_title(title_a, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    def _plot_mask_illustration(self, base: np.ndarray, mask_neb: np.ndarray,
                                 mask_bg: np.ndarray, alpha: float = 0.45) -> plt.Figure:
        """Image A shown with a translucent nebula/background mask overlay.

        Uses the same steelblue/tomato color convention as the Section 8a violin
        plots (report_builder.py's palette = {"Nebula": "steelblue",
        "Background": "tomato"}) so the two figures read as one visual language.
        Unclassified pixels (neither mask) are left plain grayscale. mask_neb/
        mask_bg are expected to already be the two-image shared classification
        (mask_neb_a | mask_neb_b for Nebula, mask_bg_a & mask_bg_b for Background)
        — the exact masks that feed the violin plots
        and correlation scatter plots — so this figure depicts what those plots
        are actually gated on.
        """
        gray = self._stretch_for_display(base)
        h = min(gray.shape[0], mask_neb.shape[0], mask_bg.shape[0])
        w = min(gray.shape[1], mask_neb.shape[1], mask_bg.shape[1])
        gray = gray[:h, :w]
        mask_neb = mask_neb[:h, :w]
        mask_bg = mask_bg[:h, :w]

        rgb = np.stack([gray, gray, gray], axis=-1)
        neb_color = np.array(mcolors.to_rgb("steelblue"))
        bg_color = np.array(mcolors.to_rgb("tomato"))
        rgb[mask_neb] = (1 - alpha) * rgb[mask_neb] + alpha * neb_color
        rgb[mask_bg] = (1 - alpha) * rgb[mask_bg] + alpha * bg_color

        aspect_ratio = h / max(w, 1)
        panel_w = 8.0
        fig, ax = plt.subplots(figsize=(panel_w, panel_w * aspect_ratio + 1.0))
        ax.imshow(rgb, origin="upper", interpolation="nearest", aspect="equal")
        ax.axis("off")
        ax.set_title("Nebula / background mask regions (shared A∩B classification), "
                      "shown on Image A", fontsize=10)
        legend_handles = [
            Patch(facecolor="steelblue", edgecolor="none", alpha=0.7, label="Nebula"),
            Patch(facecolor="tomato", edgecolor="none", alpha=0.7, label="Background"),
            Patch(facecolor="0.5", edgecolor="none", label="Unclassified"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.8)
        fig.tight_layout()
        return fig

    def _plot_localmax_mask_grid(self, grid_rows: list,
                                  color: str = "magenta", alpha: float = 0.55) -> plt.Figure | None:
        """Grid figure: one row per metric family, columns = kernel/scale sizes
        smallest to largest. Each panel shows that exact scale's local-maxima
        mask (same computation _localmax_entry uses for the Section 8j table)
        overlaid on that metric's own |A| magnitude map. Families with fewer
        scales than the widest row (Wavelet: 2 vs. 3) leave trailing panels
        blank. grid_rows: list of (family_label, [(base_img, mask, panel_title), ...]).
        Returns None if no row has any panel (e.g. single-image mode never calls this).
        """
        n_rows = len(grid_rows)
        n_cols = max((len(cells) for _, cells in grid_rows), default=0)
        if n_cols == 0 or not any(cells for _, cells in grid_rows):
            return None
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows))
        axes = np.atleast_2d(axes)
        col_rgb = np.array(mcolors.to_rgb(color))
        for r, (family_label, cells) in enumerate(grid_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                if c >= len(cells):
                    ax.axis("off")
                    continue
                base, mask, panel_title = cells[c]
                gray = self._stretch_for_display(base)
                h = min(gray.shape[0], mask.shape[0])
                w = min(gray.shape[1], mask.shape[1])
                gray, m = gray[:h, :w], mask[:h, :w]
                rgb = np.stack([gray, gray, gray], axis=-1)
                rgb[m] = (1 - alpha) * rgb[m] + alpha * col_rgb
                ax.imshow(rgb, origin="upper", interpolation="nearest", aspect="equal")
                ax.axis("off")
                ax.set_title(panel_title, fontsize=8)
            axes[r, 0].text(-0.08, 0.5, family_label, transform=axes[r, 0].transAxes,
                             fontsize=9, fontweight="bold", ha="right", va="center", rotation=90)
        legend_handle = Patch(facecolor=color, edgecolor="none", alpha=0.8, label="Local maxima (dilated) ∪ top-N% bright")
        fig.legend(handles=[legend_handle], loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("Local-maxima masks by metric (rows) and scale, smallest → largest (columns)", fontsize=11)
        fig.tight_layout(rect=[0.03, 0.02, 1, 0.96])
        return fig

    def _plot_snr_bars(self, snr_a: dict, snr_b: dict,
                        label_a: str, label_b: str | None, levels: int) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, levels + 1)
        vals_a = [snr_a.get(lvl) or 0.0 for lvl in x]
        if label_b is not None:
            width = 0.35
            vals_b = [snr_b.get(lvl) or 0.0 for lvl in x]
            ax.bar(x - width / 2, vals_a, width, label=label_a, color="steelblue")
            ax.bar(x + width / 2, vals_b, width, label=label_b, color="tomato")
            ax.set_title("Wavelet per-level SNR comparison")
        else:
            ax.bar(x, vals_a, 0.6, label=label_a, color="steelblue")
            ax.set_title("Wavelet per-level SNR")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                   label="SNR = 1 (signal = noise)")
        ax.set_xlabel("Wavelet level (1 = finest ~2px, 4 = coarsest ~16px)")
        ax.set_ylabel("Signal energy / Noise energy")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Level {i}" for i in x])
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def _ratio_series_with_errors(ratios_by_method: dict, errors_by_method: dict | None = None) -> dict:
        """{method: {scale: value}} (+ optional matching {method: {scale: error}}) ->
        {method: [(x_px, value, error_or_None), ...]} sorted by x. Wavelet scale
        keys are human levels, converted to approximate px via 2**level. Shared
        by _plot_nc_ratio_overview (8i) and _plot_localmax_ratio_overview (8j)."""
        series = {}
        for method, ratios in ratios_by_method.items():
            if not ratios:
                continue
            errs = (errors_by_method or {}).get(method, {})
            pts = []
            for scale, v in ratios.items():
                if v is None:
                    continue
                x = 2 ** scale if method == "wavelet" else float(scale)
                pts.append((x, v, errs.get(scale)))
            if pts:
                pts.sort(key=lambda p: p[0])
                series[method] = pts
        return series

    def _plot_nc_ratio_overview(self, ratios_by_method: dict,
                                 errors_by_method: dict | None = None) -> plt.Figure | None:
        """One line per method: noise-corrected A/B ratio vs. approximate spatial
        scale (px, log-x). None if no method has any usable (non-None) value.
        errors_by_method (optional): matching {method: {scale: error}} — an
        approximate symmetric uncertainty (see _compute_nc_ratio_errors),
        rendered as error bars when present for a given point."""
        _SCALE_LABEL = {
            "std": "px", "entropy": "px", "log": "σ px",
            "gradient": "σ px", "wavelet": "level (≈px)",
        }
        _COLORS = {
            "std": "steelblue", "log": "tomato", "wavelet": "mediumpurple",
            "entropy": "seagreen", "gradient": "goldenrod",
        }
        series = self._ratio_series_with_errors(ratios_by_method, errors_by_method)
        if not series:
            return None

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, pts in series.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            label = f"{method} ({_SCALE_LABEL.get(method, 'px')})"
            if any(e is not None for e in es):
                yerr = [e if e is not None else 0.0 for e in es]
                ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linestyle="-",
                            label=label, color=_COLORS.get(method))
            else:
                ax.plot(xs, ys, marker="o", label=label, color=_COLORS.get(method))
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                   label="Ratio = 1 (A = B)")
        ax.set_xscale("log")
        ax.set_xlabel("Approximate spatial scale (px)")
        ax.set_ylabel("Noise-corrected score ratio (A / B)")
        ax.set_title("Noise-corrected local contrast — cross-method overview")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_localmax_ratio_overview(self, log_ratios_by_method: dict,
                                       errors_by_method: dict | None = None) -> plt.Figure | None:
        """One line per method: local-maxima masked log10(A/B) geometric-mean vs.
        approximate spatial scale (px, log-x). Structurally identical to
        _plot_nc_ratio_overview (Section 8i); y-axis is the local-maxima-masked
        log-ratio (Section 8j) instead of the whole-nebula noise-corrected score
        ratio. None if no method has any usable (non-None) value.
        errors_by_method (optional): matching {method: {scale: error}} — ±1
        standard deviation of the per-pixel log10(A/B) population within each
        scale's mask, plotted directly with no unit conversion (both the value
        and its error already live in log10 space, so this is an exact spread
        measure, not an approximation), rendered as error bars when present for
        a given point."""
        _SCALE_LABEL = {
            "std": "px", "entropy": "px", "log": "σ px",
            "gradient": "σ px", "wavelet": "level (≈px)",
        }
        _COLORS = {
            "std": "steelblue", "log": "tomato", "wavelet": "mediumpurple",
            "entropy": "seagreen", "gradient": "goldenrod",
        }
        series = self._ratio_series_with_errors(log_ratios_by_method, errors_by_method)
        if not series:
            return None

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, pts in series.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            label = f"{method} ({_SCALE_LABEL.get(method, 'px')})"
            if any(e is not None for e in es):
                yerr = [e if e is not None else 0.0 for e in es]
                ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linestyle="-",
                            label=label, color=_COLORS.get(method))
            else:
                ax.plot(xs, ys, marker="o", label=label, color=_COLORS.get(method))
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8,
                   label="log ratio = 0 (A = B)")
        ax.set_xscale("log")
        ax.set_xlabel("Approximate spatial scale (px)")
        ax.set_ylabel("Local-maxima masked log₁₀(A / B) (geometric mean ± SD)")
        ax.set_title("Local-maxima masked contrast — cross-method overview (log ratio)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def _sample_line(arr: np.ndarray, x0: float, y0: float,
                      x1: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
        """Sample arr along the line defined by normalised [0,1] coords.
        Returns (positions_px, values) using bilinear interpolation."""
        H, W = arr.shape[:2]
        c0, r0 = x0 * W, y0 * H
        c1, r1 = x1 * W, y1 * H
        length = float(np.hypot(c1 - c0, r1 - r0))
        n = max(2, int(length))
        cols = np.linspace(c0, c1, n)
        rows = np.linspace(r0, r1, n)
        values = map_coordinates(arr, [rows, cols], order=1, mode='nearest')
        positions = np.linspace(0.0, length, n)
        return positions, values

    @staticmethod
    def _draw_cross_section(ax, pos: np.ndarray, prof_a: np.ndarray, prof_b: np.ndarray,
                             label_a: str, label_b: str, title: str) -> None:
        """Draw a cross-section profile into an existing Axes: the bottom-right
        quadrant of _plot_side_by_side's 2×2 grid. Fonts/linewidths are tuned down
        from the metric's old standalone-figure sizes since this panel is now
        roughly a quarter the area."""
        # Images with slightly different pixel dimensions produce different-length profiles
        n = min(len(pos), len(prof_a), len(prof_b))
        pos, prof_a, prof_b = pos[:n], prof_a[:n], prof_b[:n]
        ax.plot(pos, prof_a, color="steelblue", linewidth=1.1, alpha=XS_LINE_ALPHA, label=label_a)
        ax.plot(pos, prof_b, color="tomato", linewidth=1.1, alpha=XS_LINE_ALPHA, label=label_b)
        ax.set_xlabel("Position along line (px)", fontsize=8)
        ax.set_ylabel("Map value", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper left", fontsize=6.5, labelspacing=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=9)

    @staticmethod
    def _plot_metric_correlation(map_a: np.ndarray, map_b: np.ndarray,
                                  log_ratio: np.ndarray,
                                  mask_neb_shared: np.ndarray, mask_bg_shared: np.ndarray,
                                  label_a: str, label_b: str, metric_title: str,
                                  rng: np.random.Generator,
                                  max_samples: int = SECTION8_SCATTER_MAX_SAMPLES) -> plt.Figure | None:
        """1x2 correlation scatter (Nebula | Background): metric value in A (y)
        vs. metric value in B (x), with a dashed 1:1 reference line. Each point
        is colored by that pixel's log-ratio value (log_ratio — the same array
        driving the adjacent log-ratio map panel), using the same bwr colormap
        and range, so the scatter visually links back to the map.

        Axis limits reflect the FULL pooled population range per subplot (not
        percentile-clipped like the Section 8a violin plots) so upper-tail
        divergence from the 1:1 line — the signal this plot exists to surface —
        stays visible. Point clouds are randomly subsampled (up to max_samples)
        for render cost only; the axis range is always computed from the full,
        unsampled population. Returns None if both subplots have too few points.
        """
        h = min(map_a.shape[0], map_b.shape[0], log_ratio.shape[0],
                mask_neb_shared.shape[0], mask_bg_shared.shape[0])
        w = min(map_a.shape[1], map_b.shape[1], log_ratio.shape[1],
                mask_neb_shared.shape[1], mask_bg_shared.shape[1])
        map_a = map_a[:h, :w]
        map_b = map_b[:h, :w]
        log_ratio = log_ratio[:h, :w]
        mask_neb_shared = mask_neb_shared[:h, :w]
        mask_bg_shared = mask_bg_shared[:h, :w]

        # Dark-mode-aware reference-line color (project convention).
        is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        orig_color = "white" if is_dark else "black"

        # Shared across both panels so a given color always means the same
        # log-ratio value, matching the adjacent map figure's own scale.
        dvmin, dvmax = SpatialDetailAnalyzer._log_ratio_color_range(log_ratio)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        any_data = False
        for ax, region_name, mask in zip(axes, ("Nebula", "Background"),
                                          (mask_neb_shared, mask_bg_shared)):
            a_vals = map_a[mask]
            b_vals = map_b[mask]
            c_vals = log_ratio[mask]
            n = a_vals.size
            if n < 3:
                ax.set_visible(False)
                continue
            any_data = True

            lo = float(min(a_vals.min(), b_vals.min()))
            hi = float(max(a_vals.max(), b_vals.max()))
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            lo -= pad
            hi += pad

            if n > max_samples:
                idx = rng.choice(n, max_samples, replace=False)
                a_plot, b_plot, c_plot = a_vals[idx], b_vals[idx], c_vals[idx]
            else:
                a_plot, b_plot, c_plot = a_vals, b_vals, c_vals

            sc = ax.scatter(b_plot, a_plot, c=c_plot, cmap="bwr", vmin=dvmin, vmax=dvmax,
                             alpha=0.55, s=8, zorder=3, edgecolors="none", rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=orig_color, linestyle="--",
                    linewidth=1.2, zorder=4, label="Slope = 1 (A = B)")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log10(|A|/|B|)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{metric_title} — {label_b} (x)", fontsize=8)
            ax.set_ylabel(f"{metric_title} — {label_a} (y)", fontsize=8)
            ax.set_title(f"{region_name} (n={n})", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, loc="best", labelspacing=0.3)
            ax.grid(True, alpha=0.3)

        if not any_data:
            plt.close(fig)
            return None
        fig.tight_layout()
        return fig

    @staticmethod
    def _crop_border(arr: np.ndarray, fraction: float) -> np.ndarray:
        n = max(1, int(min(arr.shape[0], arr.shape[1]) * fraction))
        if arr.shape[0] > 2 * n and arr.shape[1] > 2 * n:
            return arr[n:-n, n:-n]
        return arr

    @staticmethod
    def _crosshair_to_cropped_px(crosshair: dict | None, shape: tuple,
                                  crop_fraction: float) -> tuple[float, float, float, float] | None:
        """Convert a normalised [0,1] crosshair dict (in the coordinate frame of an
        array with `shape`) to pixel coords in the frame _crop_border(arr, crop_fraction)
        produces for that array — mirrors _crop_border's own offset math exactly, so the
        overlay lines up pixel-for-pixel with the already-cropped display arrays."""
        if crosshair is None:
            return None
        H, W = shape[:2]
        n = max(1, int(min(H, W) * crop_fraction))
        off_y, off_x = (n, n) if (H > 2 * n and W > 2 * n) else (0, 0)
        return (crosshair["x0"] * W - off_x, crosshair["y0"] * H - off_y,
                crosshair["x1"] * W - off_x, crosshair["y1"] * H - off_y)

    @staticmethod
    def _stretch_for_display(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(arr, [0.5, 99.9])
        if hi > lo:
            return np.clip((arr.astype(float) - lo) / (hi - lo), 0.0, 1.0)
        return np.zeros_like(arr, dtype=float)

    def _plot_context_figure(self, img_a: AstroImage, img_b: AstroImage,
                              label_a: str, label_b: str,
                              crosshair: dict) -> plt.Figure:
        """2×1 zoomed crop of both images with the cross-section line overlaid."""
        H_a, W_a = img_a.data.shape[:2]
        H_b, W_b = img_b.data.shape[:2]

        x0a = crosshair["x0"] * W_a;  y0a = crosshair["y0"] * H_a
        x1a = crosshair["x1"] * W_a;  y1a = crosshair["y1"] * H_a
        pad_a = max(30, int(0.15 * float(np.hypot(x1a - x0a, y1a - y0a))))
        rx0a = max(0,   int(min(x0a, x1a) - pad_a))
        ry0a = max(0,   int(min(y0a, y1a) - pad_a))
        rx1a = min(W_a, int(max(x0a, x1a) + pad_a))
        ry1a = min(H_a, int(max(y0a, y1a) + pad_a))
        crop_a = self._stretch_for_display(img_a.data[ry0a:ry1a, rx0a:rx1a])

        x0b = crosshair["x0"] * W_b;  y0b = crosshair["y0"] * H_b
        x1b = crosshair["x1"] * W_b;  y1b = crosshair["y1"] * H_b
        pad_b = max(30, int(0.15 * float(np.hypot(x1b - x0b, y1b - y0b))))
        rx0b = max(0,   int(min(x0b, x1b) - pad_b))
        ry0b = max(0,   int(min(y0b, y1b) - pad_b))
        rx1b = min(W_b, int(max(x0b, x1b) + pad_b))
        ry1b = min(H_b, int(max(y0b, y1b) + pad_b))
        crop_b = self._stretch_for_display(img_b.data[ry0b:ry1b, rx0b:rx1b])

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        ax_a.imshow(crop_a, origin="upper", cmap="gray", interpolation="nearest")
        ax_a.plot([x0a - rx0a, x1a - rx0a], [y0a - ry0a, y1a - ry0a],
                  color="#ff7f0e", linewidth=2)
        ax_a.set_title(label_a, fontsize=10)
        ax_a.axis("off")

        ax_b.imshow(crop_b, origin="upper", cmap="gray", interpolation="nearest")
        ax_b.plot([x0b - rx0b, x1b - rx0b], [y0b - ry0b, y1b - ry0b],
                  color="#1f77b4", linewidth=2)
        ax_b.set_title(label_b, fontsize=10)
        ax_b.axis("off")

        return fig

    @staticmethod
    def _plot_image_profile(pos_a: np.ndarray, prof_a: np.ndarray,
                             pos_b: np.ndarray, prof_b: np.ndarray,
                             label_a: str, label_b: str,
                             title: str = "Cross-section brightness profile",
                             ylabel: str = "Pixel value (normalised)") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(pos_a, prof_a, color="#ff7f0e", linewidth=1.5,
                alpha=XS_LINE_ALPHA, label=label_a)
        ax.plot(pos_b, prof_b, color="#1f77b4", linewidth=1.5,
                alpha=XS_LINE_ALPHA, label=label_b)
        ax.set_xlabel("Position along line (px)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def _plot_image_profile_single(pos: np.ndarray, prof: np.ndarray,
                                    label: str,
                                    title: str = "Cross-section brightness profile",
                                    ylabel: str = "Pixel value (normalised)") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(pos, prof, color="#ff7f0e", linewidth=1.5,
                alpha=XS_LINE_ALPHA, label=label)
        ax.set_xlabel("Position along line (px)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def _compute_xs_snr(pos_a: np.ndarray, prof_a: np.ndarray,
                         pos_b: np.ndarray, prof_b: np.ndarray,
                         label_a: str, label_b: str,
                         width: int) -> dict | None:
        """Compute std-based SNR for both profiles and produce a shaded profile figure.

        Bright region centred on prof_a argmax; dark region on prof_a argmin.
        Both images sample the same profile positions for a fair comparison.
        Returns a dict with scalar metrics and a 'fig' key, or None if arrays are too short.
        """
        n = min(len(prof_a), len(prof_b))
        if n < width * 2:
            return None
        prof_a = prof_a[:n].astype(float)
        prof_b = prof_b[:n].astype(float)
        pos    = pos_a[:n]

        def _bounds(idx: int, w: int, length: int) -> tuple[int, int]:
            half  = w // 2
            start = max(0, idx - half)
            end   = min(length, start + w)
            start = max(0, end - w)
            return start, end

        bright_idx = int(np.argmax(prof_a))
        dark_idx   = int(np.argmin(prof_a))
        bs, be = _bounds(bright_idx, width, n)
        ds, de = _bounds(dark_idx,   width, n)

        def _snr(prof: np.ndarray) -> float:
            b_mean, b_std = float(np.mean(prof[bs:be])), float(np.std(prof[bs:be]))
            d_mean, d_std = float(np.mean(prof[ds:de])), float(np.std(prof[ds:de]))
            denom = (b_std ** 2 + d_std ** 2) / width
            if denom <= 0:
                return float("nan")
            return (b_mean - d_mean) / float(np.sqrt(denom))

        snr_a = _snr(prof_a)
        snr_b = _snr(prof_b)

        abs_a = abs(snr_a) if not np.isnan(snr_a) else 0.0
        abs_b = abs(snr_b) if not np.isnan(snr_b) else 0.0
        if abs_a > 0 and abs_b > 0:
            hi, lo = max(abs_a, abs_b), min(abs_a, abs_b)
            exposure_factor = (hi / lo) ** 2
            higher_label = label_a if abs_a >= abs_b else label_b
            lower_label  = label_b if abs_a >= abs_b else label_a
        else:
            exposure_factor = float("nan")
            higher_label = label_a
            lower_label  = label_b

        # ── Figure ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(pos, prof_a, color="steelblue", linewidth=1.5,
                alpha=XS_LINE_ALPHA, label=label_a)
        ax.plot(pos, prof_b, color="tomato", linewidth=1.5,
                alpha=XS_LINE_ALPHA, label=label_b)

        bright_pos_lo = float(pos[bs]) if bs < len(pos) else 0.0
        bright_pos_hi = float(pos[be - 1]) if be - 1 < len(pos) else float(pos[-1])
        dark_pos_lo   = float(pos[ds]) if ds < len(pos) else 0.0
        dark_pos_hi   = float(pos[de - 1]) if de - 1 < len(pos) else float(pos[-1])

        ax.axvspan(bright_pos_lo, bright_pos_hi, alpha=0.25, color="gold",
                   label=f"Bright region ({width} px)")
        ax.axvspan(dark_pos_lo,   dark_pos_hi,   alpha=0.20, color="gray",
                   label=f"Dark region ({width} px)")
        ax.set_xlabel("Distance (px)")
        ax.set_ylabel("Pixel value (ADU)")
        ax.set_title("Cross-section SNR — bright/dark sample regions")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        return {
            "snr_a":           snr_a,
            "snr_b":           snr_b,
            "exposure_factor": exposure_factor,
            "higher_label":    higher_label,
            "lower_label":     lower_label,
            "width":           width,
            "bright_start":    bs,
            "bright_end":      be,
            "dark_start":      ds,
            "dark_end":        de,
            "fig":             fig,
        }
