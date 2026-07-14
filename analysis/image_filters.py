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
from scipy.ndimage import generic_filter, gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude, map_coordinates, zoom, maximum_filter, minimum_filter, median_filter
import pywt

from core.astro_image import AstroImage
from core.fig_utils import fig_to_b64, figs_to_b64
from core.models import (STD_KERNEL_SIZES, LOG_SIGMAS, WAVELET_NAME, WAVELET_LEVELS,
                         WEBER_KERNEL_SIZES,
                         XS_LINE_ALPHA, SECTION8_BORDER_CROP_FRACTION, SECTION8_ANALYSIS_CMAP,
                         XS_SNR_REGION_WIDTH, SECTION8_DIFF_DIST_MAX_SAMPLES)

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
                weber_kernel_sizes: tuple = WEBER_KERNEL_SIZES,
                crosshair: dict | None = None,
                roi: tuple | None = None,
                xs_snr_width: int | None = None) -> dict:

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
            "weber_contrast_a": {},
            "weber_contrast_b": {},
            "panels": {},
            "diff_dist": {},
            "nc_shared_nebula_pixels": 0,
            "std_nc_score_a": {}, "std_nc_score_b": {},
            "std_nc_noise_a": {}, "std_nc_noise_b": {}, "std_nc_ratio": {},
            "log_nc_score_a": {}, "log_nc_score_b": {},
            "log_nc_noise_a": {}, "log_nc_noise_b": {}, "log_nc_ratio": {},
            "wavelet_nc_score_a": {}, "wavelet_nc_score_b": {},
            "wavelet_nc_noise_a": {}, "wavelet_nc_noise_b": {}, "wavelet_nc_ratio": {},
            "weber_nc_score_a": {}, "weber_nc_score_b": {},
            "weber_nc_noise_a": {}, "weber_nc_noise_b": {}, "weber_nc_ratio": {},
            "gm_nc_score_a": {}, "gm_nc_score_b": {},
            "gm_nc_noise_a": {}, "gm_nc_noise_b": {}, "gm_nc_ratio": {},
        }
        figures: dict = {}

        if norm_a is None or (image_b is not None and norm_b is None):
            result["warning"] = "Mean signal ≤ 0 in one or both images; spatial analysis skipped."
            return result

        # Nebula / background masks (used for contrast ratio)
        mask_neb_a, mask_bg_a = self._make_masks(image_a)
        mask_neb_b = mask_bg_b = None
        if image_b is not None:
            mask_neb_b, mask_bg_b = self._make_masks(image_b)

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
        # distributions: pixels BOTH images independently classify the same way.
        # None in single-image mode.
        mask_neb_shared = mask_bg_shared = None
        if mask_neb_b is not None:
            h_s = min(mask_neb_a.shape[0], mask_neb_b.shape[0])
            w_s = min(mask_neb_a.shape[1], mask_neb_b.shape[1])
            mask_neb_shared = mask_neb_a[:h_s, :w_s] & mask_neb_b[:h_s, :w_s]
            mask_bg_shared = mask_bg_a[:h_s, :w_s] & mask_bg_b[:h_s, :w_s]
        result["nc_shared_nebula_pixels"] = (
            int(np.count_nonzero(mask_neb_shared)) if mask_neb_shared is not None else 0
        )

        # Fixed seed so the diff-distribution subsampling below is reproducible
        # across report generations for the same input images.
        diff_dist_rng = np.random.default_rng(42)

        # Export the exact preprocessed array every std/LoG/wavelet/Weber/gradient
        # map is computed from (mean-normalised, ROI-cropped if a ROI was given) as
        # its own panel, so the Report Inspector can show the source image content
        # for a map alongside the map itself, pixel-aligned to the same crop.
        original_diff = (analysis_a - analysis_b).astype(np.float32) if analysis_b is not None else None
        result["panels"]["original"] = {
            "a":    analysis_a.astype(np.float32),
            "b":    analysis_b.astype(np.float32) if analysis_b is not None else None,
            "diff": original_diff,
        }
        if original_diff is not None:
            result["diff_dist"]["original"] = self._diff_distribution(
                original_diff, mask_neb_shared, mask_bg_shared, diff_dist_rng)

        _label_b = image_b.label if image_b is not None else None

        # 1-5. Local std, LoG, wavelet, Weber, gradient — all read norm_a/norm_b with no
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
                mask_bg_shared=mask_bg_shared, diff_dist_rng=diff_dist_rng,
            )
            _f_log = _ex.submit(self._log_analysis,
                analysis_a, analysis_b, log_sigmas,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared, mask_bg_a=mask_bg_a, mask_bg_b=mask_bg_b,
                mask_bg_shared=mask_bg_shared, diff_dist_rng=diff_dist_rng,
            )
            _f_wav = _ex.submit(self._wavelet_analysis,
                analysis_a, analysis_b, wavelet, levels,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared, mask_bg_a=mask_bg_a, mask_bg_b=mask_bg_b,
                mask_bg_shared=mask_bg_shared, diff_dist_rng=diff_dist_rng,
            )
            _f_web = _ex.submit(self._weber_analysis,
                analysis_a, analysis_b,
                weber_kernel_sizes,
                image_a.label, _label_b,
                display_roi=display_roi,
                mask_neb_shared=mask_neb_shared, mask_bg_a=mask_bg_a, mask_bg_b=mask_bg_b,
                mask_bg_shared=mask_bg_shared, diff_dist_rng=diff_dist_rng,
            )
            _f_grad = _ex.submit(self._gradient_analysis,
                analysis_a, analysis_b, log_sigmas,
                image_a.label, _label_b,
                display_roi=display_roi,
                crosshair=crosshair_roi,
                mask_neb_shared=mask_neb_shared, mask_bg_a=mask_bg_a, mask_bg_b=mask_bg_b,
                mask_bg_shared=mask_bg_shared, diff_dist_rng=diff_dist_rng,
            )
            std_b64, std_partial = _f_std.result()
            log_b64, log_partial = _f_log.result()
            wav_b64, wav_partial = _f_wav.result()
            web_b64, web_partial = _f_web.result()
            grad_b64, grad_partial = _f_grad.result()

        figures.update(std_b64)
        figures.update(log_b64)
        figures.update(wav_b64)
        figures.update(web_b64)
        figures.update(grad_b64)
        result["contrast_ratios_a"].update(std_partial["contrast_ratios_a"])
        result["contrast_ratios_b"].update(std_partial["contrast_ratios_b"])
        result["sigma_noise_a"] = wav_partial["sigma_noise_a"]
        result["sigma_noise_b"] = wav_partial["sigma_noise_b"]
        result["wavelet_snr_a"].update(wav_partial["wavelet_snr_a"])
        result["wavelet_snr_b"].update(wav_partial["wavelet_snr_b"])
        result["weber_contrast_a"].update(web_partial["weber_contrast_a"])
        result["weber_contrast_b"].update(web_partial["weber_contrast_b"])
        result["panels"].update(std_partial["panels"])
        result["panels"].update(log_partial["panels"])
        result["panels"].update(wav_partial["panels"])
        result["panels"].update(web_partial["panels"])
        result["panels"].update(grad_partial["panels"])
        result["diff_dist"].update(std_partial["diff_dist"])
        result["diff_dist"].update(log_partial["diff_dist"])
        result["diff_dist"].update(wav_partial["diff_dist"])
        result["diff_dist"].update(web_partial["diff_dist"])
        result["diff_dist"].update(grad_partial["diff_dist"])

        # Merge noise-corrected scores/noise-floors and compute A/B ratios centrally.
        for prefix, partial in (("std", std_partial), ("log", log_partial),
                                 ("wavelet", wav_partial), ("weber", web_partial),
                                 ("gm", grad_partial)):
            for suffix in ("nc_score_a", "nc_score_b", "nc_noise_a", "nc_noise_b"):
                result[f"{prefix}_{suffix}"].update(partial[f"{prefix}_{suffix}"])
            result[f"{prefix}_nc_ratio"] = self._compute_nc_ratios(
                result[f"{prefix}_nc_score_a"], result[f"{prefix}_nc_score_b"])

        if image_b is not None:
            nc_fig = self._plot_nc_ratio_overview({
                "std": result["std_nc_ratio"],
                "log": result["log_nc_ratio"],
                "wavelet": result["wavelet_nc_ratio"],
                "weber": result["weber_nc_ratio"],
                "gradient": result["gm_nc_ratio"],
            })
            if nc_fig is not None:
                figures["nc_ratio_overview"] = fig_to_b64(nc_fig, dpi=150)

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

    def _make_masks(self, image: AstroImage) -> tuple[np.ndarray, np.ndarray]:
        rms = image.background_rms
        if rms is None:
            rms_val = float(np.std(image.background_subtracted()))
        else:
            rms_val = float(np.median(rms))

        bgsub = image.background_subtracted()
        nebula_mask = bgsub > 2.0 * rms_val
        bg_mask = bgsub < 0.5 * rms_val

        # Fallback: use top-5% as nebula if no pixels pass threshold
        if not np.any(nebula_mask):
            threshold = np.percentile(bgsub, 95)
            nebula_mask = bgsub >= threshold

        return nebula_mask, bg_mask

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
                       mask_bg_shared=None, diff_dist_rng=None) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "contrast_ratios_a": {}, "contrast_ratios_b": {},
            "std_nc_score_a": {}, "std_nc_score_b": {},
            "std_nc_noise_a": {}, "std_nc_noise_b": {},
            "panels": {},
            "diff_dist": {},
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
                nc_a, noise_a = self._nc_score(std_a, mask_neb_shared, mask_bg_a)
                partial["std_nc_score_a"][ks] = nc_a
                partial["std_nc_noise_a"][ks] = noise_a
                nc_b, noise_b = self._nc_score(std_b, mask_neb_shared, mask_bg_b)
                partial["std_nc_score_b"][ks] = nc_b
                partial["std_nc_noise_b"][ks] = noise_b

            diff = (std_a - std_b).astype(np.float32) if not single else None
            partial["panels"][f"std_{ks}px"] = {
                "a":    std_a.astype(np.float32),
                "b":    std_b.astype(np.float32) if std_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                partial["diff_dist"][f"std_{ks}px"] = self._diff_distribution(
                    diff, mask_neb_shared, mask_bg_shared, diff_dist_rng)
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_std_{ks}px"] = {
                    "a": (std_a / noise_a).astype(np.float32),
                    "b": (std_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(std_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(std_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Local σ — kernel {ks}px — {label_a}",
                    f"Local σ — kernel {ks}px — {label_b}",
                    diff_title=f"Diff (A−B), kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                    display_roi=None,
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
                figures[f"nrm_std_{ks}px"] = self._plot_side_by_side(
                    self._crop_border(std_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(std_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Local σ (× noise floor) — kernel {ks}px — {label_a}",
                    f"Local σ (× noise floor) — kernel {ks}px — {label_b}",
                    diff_title=f"Diff (A−B), noise-normalised, kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                )

            if crosshair is not None:
                pos, pa = self._sample_line(std_a, **crosshair)
                if not single:
                    _, pb = self._sample_line(std_b, **crosshair)
                    figures[f"xs_std_{ks}px"] = self._plot_cross_section(
                        pos, pa, pb, label_a, label_b,
                        f"Cross-section — Local σ, kernel {ks}px")

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
                  mask_bg: np.ndarray) -> tuple[float | None, float | None]:
        """Noise-corrected local-contrast score for one detail map at one scale.

        score = median(|detail|) over the pixels BOTH images classify as nebula
        (mask_neb_shared), divided by median(|detail|) over THIS image's own
        background mask — its empirical per-scale noise floor for this operator.
        Returns (score, noise_floor); either is None if a mask selects zero pixels,
        mask_neb_shared is unavailable (single-image mode), or noise_floor <= 0.
        """
        if mask_neb_shared is None:
            return None, None
        h = min(detail_map.shape[0], mask_neb_shared.shape[0], mask_bg.shape[0])
        w = min(detail_map.shape[1], mask_neb_shared.shape[1], mask_bg.shape[1])
        absmap = np.abs(detail_map[:h, :w])
        neb_vals = absmap[mask_neb_shared[:h, :w]]
        bg_vals = absmap[mask_bg[:h, :w]]
        if neb_vals.size == 0 or bg_vals.size == 0:
            return None, None
        noise_floor = float(bn.median(bg_vals))
        if noise_floor <= 0:
            return None, None
        return float(bn.median(neb_vals)) / noise_floor, noise_floor

    @staticmethod
    def _diff_distribution(diff_map: np.ndarray,
                            mask_neb_shared: np.ndarray | None,
                            mask_bg_shared: np.ndarray | None,
                            rng: np.random.Generator) -> dict:
        """Random-subsampled A-B diff pixel populations for nebula vs background.

        Returns {"nebula": ndarray, "background": ndarray} (float32, signed diff
        values, up to SECTION8_DIFF_DIST_MAX_SAMPLES each). Either array is empty
        if the corresponding mask is unavailable (single-image mode) or selects
        zero pixels.
        """
        out = {"nebula": np.empty(0, dtype=np.float32),
               "background": np.empty(0, dtype=np.float32)}
        if mask_neb_shared is None or mask_bg_shared is None:
            return out
        h = min(diff_map.shape[0], mask_neb_shared.shape[0], mask_bg_shared.shape[0])
        w = min(diff_map.shape[1], mask_neb_shared.shape[1], mask_bg_shared.shape[1])
        cropped = diff_map[:h, :w]
        for key, mask in (("nebula", mask_neb_shared), ("background", mask_bg_shared)):
            vals = cropped[mask[:h, :w]]
            if vals.size > SECTION8_DIFF_DIST_MAX_SAMPLES:
                idx = rng.choice(vals.size, SECTION8_DIFF_DIST_MAX_SAMPLES, replace=False)
                vals = vals[idx]
            out[key] = vals.astype(np.float32)
        return out

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

    # ------------------------------------------------------------------
    # Laplacian of Gaussian maps
    # ------------------------------------------------------------------

    def _log_analysis(self, norm_a, norm_b, sigmas,
                       label_a, label_b,
                       display_roi=None,
                       crosshair=None,
                       mask_neb_shared=None, mask_bg_a=None, mask_bg_b=None,
                       mask_bg_shared=None, diff_dist_rng=None) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "log_nc_score_a": {}, "log_nc_score_b": {},
            "log_nc_noise_a": {}, "log_nc_noise_b": {},
            "panels": {},
            "diff_dist": {},
        }
        single = norm_b is None
        for sigma in sigmas:
            log_a = np.abs(gaussian_laplace(norm_a, sigma=sigma))
            log_b = np.abs(gaussian_laplace(norm_b, sigma=sigma)) if not single else None

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a = self._nc_score(log_a, mask_neb_shared, mask_bg_a)
                partial["log_nc_score_a"][sigma] = nc_a
                partial["log_nc_noise_a"][sigma] = noise_a
                nc_b, noise_b = self._nc_score(log_b, mask_neb_shared, mask_bg_b)
                partial["log_nc_score_b"][sigma] = nc_b
                partial["log_nc_noise_b"][sigma] = noise_b

            diff = (log_a - log_b).astype(np.float32) if log_b is not None else None
            partial["panels"][f"log_{sigma}"] = {
                "a":    log_a.astype(np.float32),
                "b":    log_b.astype(np.float32) if log_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                partial["diff_dist"][f"log_{sigma}"] = self._diff_distribution(
                    diff, mask_neb_shared, mask_bg_shared, diff_dist_rng)
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_log_{sigma}"] = {
                    "a": (log_a / noise_a).astype(np.float32),
                    "b": (log_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(log_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(log_b, SECTION8_BORDER_CROP_FRACTION),
                    f"|LoG| σ={sigma}px — {label_a}",
                    f"|LoG| σ={sigma}px — {label_b}",
                    diff_title=f"LoG diff (A−B), σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                    display_roi=None,
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
                figures[f"nrm_log_{sigma}"] = self._plot_side_by_side(
                    self._crop_border(log_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(log_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"|LoG| (× noise floor) σ={sigma}px — {label_a}",
                    f"|LoG| (× noise floor) σ={sigma}px — {label_b}",
                    diff_title=f"Diff (A−B), noise-normalised, σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                )

            if crosshair is not None and not single:
                pos, pa = self._sample_line(log_a, **crosshair)
                _, pb = self._sample_line(log_b, **crosshair)
                figures[f"xs_log_sigma{sigma}"] = self._plot_cross_section(
                    pos, pa, pb, label_a, label_b,
                    f"Cross-section — |LoG|, σ={sigma}px")
        return figs_to_b64(figures, dpi=150), partial

    # ------------------------------------------------------------------
    # Gradient magnitude (edge sharpness)
    # ------------------------------------------------------------------

    def _gradient_analysis(self, norm_a, norm_b, sigmas,
                            label_a, label_b,
                            display_roi=None,
                            crosshair=None,
                            mask_neb_shared=None, mask_bg_a=None, mask_bg_b=None,
                            mask_bg_shared=None, diff_dist_rng=None) -> tuple[dict, dict]:
        """G = |gradient| at Gaussian scale sigma (first spatial derivative magnitude).
        Reuses the LOG_SIGMAS scale set so gradient and |LoG| are directly comparable
        at identical spatial scales. Structured identically to _log_analysis."""
        figures = {}
        partial: dict = {
            "gm_nc_score_a": {}, "gm_nc_score_b": {},
            "gm_nc_noise_a": {}, "gm_nc_noise_b": {},
            "panels": {},
            "diff_dist": {},
        }
        single = norm_b is None
        for sigma in sigmas:
            gm_a = gaussian_gradient_magnitude(norm_a, sigma=sigma)
            gm_b = gaussian_gradient_magnitude(norm_b, sigma=sigma) if not single else None

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a = self._nc_score(gm_a, mask_neb_shared, mask_bg_a)
                partial["gm_nc_score_a"][sigma] = nc_a
                partial["gm_nc_noise_a"][sigma] = noise_a
                nc_b, noise_b = self._nc_score(gm_b, mask_neb_shared, mask_bg_b)
                partial["gm_nc_score_b"][sigma] = nc_b
                partial["gm_nc_noise_b"][sigma] = noise_b

            diff = (gm_a - gm_b).astype(np.float32) if gm_b is not None else None
            partial["panels"][f"gradient_{sigma}"] = {
                "a":    gm_a.astype(np.float32),
                "b":    gm_b.astype(np.float32) if gm_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                partial["diff_dist"][f"gradient_{sigma}"] = self._diff_distribution(
                    diff, mask_neb_shared, mask_bg_shared, diff_dist_rng)
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_gradient_{sigma}"] = {
                    "a": (gm_a / noise_a).astype(np.float32),
                    "b": (gm_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(gm_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(gm_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Gradient |G| σ={sigma}px — {label_a}",
                    f"Gradient |G| σ={sigma}px — {label_b}",
                    diff_title=f"Gradient diff (A−B), σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                    display_roi=None,
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
                figures[f"nrm_gradient_{sigma}"] = self._plot_side_by_side(
                    self._crop_border(gm_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(gm_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Gradient (× noise floor) σ={sigma}px — {label_a}",
                    f"Gradient (× noise floor) σ={sigma}px — {label_b}",
                    diff_title=f"Diff (A−B), noise-normalised, σ={sigma}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                )

            if crosshair is not None and not single:
                pos, pa = self._sample_line(gm_a, **crosshair)
                _, pb = self._sample_line(gm_b, **crosshair)
                figures[f"xs_gradient_{sigma}"] = self._plot_cross_section(
                    pos, pa, pb, label_a, label_b,
                    f"Cross-section — Gradient, σ={sigma}px")
        return figs_to_b64(figures, dpi=150), partial

    # ------------------------------------------------------------------
    # Wavelet decomposition
    # ------------------------------------------------------------------

    def _wavelet_analysis(self, norm_a, norm_b, wavelet, levels,
                           label_a, label_b,
                           display_roi=None,
                           crosshair=None,
                           mask_neb_shared=None, mask_bg_a=None, mask_bg_b=None,
                           mask_bg_shared=None, diff_dist_rng=None) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "sigma_noise_a": None, "sigma_noise_b": None,
            "wavelet_snr_a": {}, "wavelet_snr_b": {},
            "wavelet_nc_score_a": {}, "wavelet_nc_score_b": {},
            "wavelet_nc_noise_a": {}, "wavelet_nc_noise_b": {},
            "panels": {},
            "diff_dist": {},
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
                nc_a, noise_a = self._nc_score(rec_a, mask_neb_shared, mask_bg_a)
                partial["wavelet_nc_score_a"][human_level] = nc_a
                partial["wavelet_nc_noise_a"][human_level] = noise_a
                nc_b, noise_b = self._nc_score(rec_b, mask_neb_shared, mask_bg_b)
                partial["wavelet_nc_score_b"][human_level] = nc_b
                partial["wavelet_nc_noise_b"][human_level] = noise_b

            if human_level not in (2, 3):
                continue   # display/panels only for levels 2-3, unchanged from prior behaviour

            display_level = human_level
            diff = (rec_a - rec_b).astype(np.float32) if rec_b is not None else None
            partial["panels"][f"wavelet_{display_level}"] = {
                "a":    rec_a.astype(np.float32),
                "b":    rec_b.astype(np.float32) if rec_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                partial["diff_dist"][f"wavelet_{display_level}"] = self._diff_distribution(
                    diff, mask_neb_shared, mask_bg_shared, diff_dist_rng)
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_wavelet_{display_level}"] = {
                    "a": (rec_a / noise_a).astype(np.float32),
                    "b": (rec_b / noise_b).astype(np.float32),
                    "diff": None,
                }
            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(rec_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(rec_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Wavelet level {display_level} — {label_a}",
                    f"Wavelet level {display_level} — {label_b}",
                    diff_title=f"Level {display_level} diff (A−B)",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                )
            else:
                fig = self._plot_single(
                    self._crop_border(rec_a, SECTION8_BORDER_CROP_FRACTION),
                    f"Wavelet level {display_level} — {label_a}",
                    cmap=SECTION8_ANALYSIS_CMAP,
                )
            figures[f"wavelet_level{display_level}"] = fig

            if not single and noise_a and noise_b:
                figures[f"nrm_wavelet_{display_level}"] = self._plot_side_by_side(
                    self._crop_border(rec_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(rec_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Wavelet level {display_level} (× noise floor) — {label_a}",
                    f"Wavelet level {display_level} (× noise floor) — {label_b}",
                    diff_title=f"Level {display_level} diff (A−B), noise-normalised",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                )

            if crosshair is not None and not single:
                pos, pa = self._sample_line(rec_a, **crosshair)
                _, pb = self._sample_line(rec_b, **crosshair)
                figures[f"xs_wavelet_level{display_level}"] = self._plot_cross_section(
                    pos, pa, pb, label_a, label_b,
                    f"Cross-section — Wavelet level {display_level}")

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
    # Weber fraction contrast maps
    # ------------------------------------------------------------------

    def _weber_analysis(self, norm_a, norm_b, kernel_sizes,
                        label_a, label_b,
                        display_roi=None,
                        mask_neb_shared=None, mask_bg_a=None, mask_bg_b=None,
                        mask_bg_shared=None, diff_dist_rng=None) -> tuple[dict, dict]:
        figures = {}
        partial: dict = {
            "weber_contrast_a": {},
            "weber_contrast_b": {},
            "weber_nc_score_a": {}, "weber_nc_score_b": {},
            "weber_nc_noise_a": {}, "weber_nc_noise_b": {},
            "panels": {},
            "diff_dist": {},
        }
        single = norm_b is None
        for ks in kernel_sizes:
            wc_a = self._compute_weber_map(norm_a, ks)
            wc_b = self._compute_weber_map(norm_b, ks) if not single else None

            # 99th percentile avoids dark-sky floor driving the scalar metric to extremes
            partial["weber_contrast_a"][ks] = float(np.percentile(wc_a, 99))
            if not single:
                partial["weber_contrast_b"][ks] = float(np.percentile(wc_b, 99))

            noise_a = noise_b = None
            if not single:
                nc_a, noise_a = self._nc_score(wc_a, mask_neb_shared, mask_bg_a)
                partial["weber_nc_score_a"][ks] = nc_a
                partial["weber_nc_noise_a"][ks] = noise_a
                nc_b, noise_b = self._nc_score(wc_b, mask_neb_shared, mask_bg_b)
                partial["weber_nc_score_b"][ks] = nc_b
                partial["weber_nc_noise_b"][ks] = noise_b

            diff = (wc_a - wc_b).astype(np.float32) if wc_b is not None else None
            partial["panels"][f"weber_{ks}px"] = {
                "a":    wc_a.astype(np.float32),
                "b":    wc_b.astype(np.float32) if wc_b is not None else None,
                "diff": diff,
            }
            if diff is not None:
                partial["diff_dist"][f"weber_{ks}px"] = self._diff_distribution(
                    diff, mask_neb_shared, mask_bg_shared, diff_dist_rng)
            if not single and noise_a and noise_b:
                partial["panels"][f"nrm_weber_{ks}px"] = {
                    "a": (wc_a / noise_a).astype(np.float32),
                    "b": (wc_b / noise_b).astype(np.float32),
                    "diff": None,
                }

            if not single:
                fig = self._plot_side_by_side(
                    self._crop_border(wc_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(wc_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Weber contrast — kernel {ks}px — {label_a}",
                    f"Weber contrast — kernel {ks}px — {label_b}",
                    diff_title=f"Weber diff (A−B), kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,    # unbounded output; PowerNorm compresses dynamic range
                    display_roi=None,       # _crop_border already applied
                )
            else:
                fig = self._plot_single(
                    self._crop_border(wc_a, SECTION8_BORDER_CROP_FRACTION),
                    f"Weber contrast — kernel {ks}px — {label_a}",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    nonlinear_norm=True,
                )
            figures[f"weber_{ks}px"] = fig

            if not single and noise_a and noise_b:
                figures[f"nrm_weber_{ks}px"] = self._plot_side_by_side(
                    self._crop_border(wc_a / noise_a, SECTION8_BORDER_CROP_FRACTION),
                    self._crop_border(wc_b / noise_b, SECTION8_BORDER_CROP_FRACTION),
                    f"Weber (× noise floor) — kernel {ks}px — {label_a}",
                    f"Weber (× noise floor) — kernel {ks}px — {label_b}",
                    diff_title=f"Diff (A−B), noise-normalised, kernel {ks}px",
                    cmap=SECTION8_ANALYSIS_CMAP,
                    display_roi=None,
                )

        return figs_to_b64(figures, dpi=150), partial

    def _compute_weber_map(self, norm: np.ndarray, kernel_size: int) -> np.ndarray:
        """Weber fraction contrast c = ΔL / L where ΔL = max−min and L = median of kernel.
        L uses median (not mean) — robust background luminance unaffected by bright filaments.
        Output is unbounded >= 0; nonlinear_norm=True is used for display.
        """
        _EPS = 1e-6   # L (median) can be very small over dark sky; larger EPS prevents extremes
        factor = 1.0
        data = norm
        if max(norm.shape) > MAX_DIM_FOR_STD:
            factor = MAX_DIM_FOR_STD / max(norm.shape)
            new_h = int(norm.shape[0] * factor)
            new_w = int(norm.shape[1] * factor)
            data = zoom(norm, (new_h / norm.shape[0], new_w / norm.shape[1]), order=1)
            kernel_size = max(3, int(kernel_size * factor) | 1)

        i_max = maximum_filter(data, size=kernel_size)
        i_min = minimum_filter(data, size=kernel_size)
        i_med = median_filter(data, size=kernel_size)

        delta_L = i_max - i_min                  # always >= 0
        L = np.maximum(i_med, 0.0)              # bg-subtracted images can have negative medians
        weber = delta_L / (L + _EPS)

        if factor < 1.0:
            weber = zoom(weber,
                         (norm.shape[0] / weber.shape[0],
                          norm.shape[1] / weber.shape[1]),
                         order=1)
        return np.maximum(weber, 0.0)

    def _plot_side_by_side(self, arr_a: np.ndarray, arr_b: np.ndarray,
                            title_a: str, title_b: str,
                            diff_title: str = "",
                            cmap: str = "viridis",
                            nonlinear_norm: bool = False,
                            display_roi=None,
                            smooth_display: bool = True) -> plt.Figure:
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

        # Difference panel (computed before any possible shape mismatch)
        h_min = min(arr_a.shape[0], arr_b.shape[0])
        w_min = min(arr_a.shape[1], arr_b.shape[1])
        diff = arr_a[:h_min, :w_min] - arr_b[:h_min, :w_min]

        # Symmetric about zero so the "bwr" midpoint (white) always means no difference.
        d_max = float(np.percentile(np.abs(diff), 99.5)) or 1.0
        dvmin, dvmax = -d_max, d_max

        # 3×1 column layout — A, B, then diff stacked vertically.
        # Use 1:1 pixel aspect; size figure based on the cropped array dimensions.
        h, w = arr_a.shape[:2]
        aspect_ratio = h / max(w, 1)
        panel_w = 10.0
        panel_h = panel_w * aspect_ratio
        fig_h = panel_h * 3 + 2.0   # 3 panels + headroom for colorbars/titles
        fig, axes = plt.subplots(3, 1, figsize=(panel_w, fig_h),
                                  constrained_layout=True)
        ax_a, ax_b, ax_diff = axes

        for ax, arr, title in zip([ax_a, ax_b], [arr_a, arr_b], [title_a, title_b]):
            im = ax.imshow(arr, origin="upper", cmap=cmap,
                           norm=norm if norm is not None else None,
                           vmin=None if norm is not None else vmin,
                           vmax=None if norm is not None else vmax,
                           interpolation="nearest", aspect="equal")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        im_diff = ax_diff.imshow(diff, origin="upper", cmap="bwr",
                                  vmin=dvmin, vmax=dvmax,
                                  interpolation="nearest", aspect="equal")
        ax_diff.set_title(diff_title, fontsize=10)
        ax_diff.axis("off")
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

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

    def _plot_nc_ratio_overview(self, ratios_by_method: dict) -> plt.Figure | None:
        """One line per method: noise-corrected A/B ratio vs. approximate spatial
        scale (px, log-x). None if no method has any usable (non-None) value."""
        _SCALE_LABEL = {
            "std": "px", "weber": "px", "log": "σ px",
            "gradient": "σ px", "wavelet": "level (≈px)",
        }
        _COLORS = {
            "std": "steelblue", "log": "tomato", "wavelet": "mediumpurple",
            "weber": "seagreen", "gradient": "goldenrod",
        }
        series = {}
        for method, ratios in ratios_by_method.items():
            if not ratios:
                continue
            if method == "wavelet":
                pts = [(2 ** scale, v) for scale, v in ratios.items() if v is not None]
            else:
                pts = [(float(scale), v) for scale, v in ratios.items() if v is not None]
            if pts:
                pts.sort(key=lambda p: p[0])
                series[method] = pts

        if not series:
            return None

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, pts in series.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=f"{method} ({_SCALE_LABEL.get(method, 'px')})",
                    color=_COLORS.get(method))
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
    def _plot_cross_section(pos: np.ndarray, prof_a: np.ndarray, prof_b: np.ndarray,
                             label_a: str, label_b: str, title: str) -> plt.Figure:
        # Images with slightly different pixel dimensions produce different-length profiles
        n = min(len(pos), len(prof_a), len(prof_b))
        pos, prof_a, prof_b = pos[:n], prof_a[:n], prof_b[:n]
        fig, ax1 = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax1.plot(pos, prof_a, color="steelblue", linewidth=1.5, alpha=XS_LINE_ALPHA, label=label_a)
        ax1.plot(pos, prof_b, color="tomato", linewidth=1.5, alpha=XS_LINE_ALPHA, label=label_b)
        ax1.set_xlabel("Position along line (px)")
        ax1.set_ylabel("Map value")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        n = min(len(prof_a), len(prof_b))
        diff = prof_a[:n] - prof_b[:n]
        ax2.plot(pos[:n], diff, color="#2ca02c", linewidth=1.2,
                 linestyle="--", alpha=0.85, label="A−B")
        ax2.axhline(0, color="#2ca02c", linewidth=0.8, alpha=0.3)   # zero-crossing reference
        ax2.set_ylabel("Difference (A−B)", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax2.legend(loc="upper right", fontsize=8)
        ax1.set_title(title, fontsize=10)
        return fig

    @staticmethod
    def _crop_border(arr: np.ndarray, fraction: float) -> np.ndarray:
        n = max(1, int(min(arr.shape[0], arr.shape[1]) * fraction))
        if arr.shape[0] > 2 * n and arr.shape[1] > 2 * n:
            return arr[n:-n, n:-n]
        return arr

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
