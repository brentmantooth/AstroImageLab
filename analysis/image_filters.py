from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, gaussian_filter, gaussian_laplace, map_coordinates, zoom
import pywt

from core.astro_image import AstroImage
from core.models import STD_KERNEL_SIZES, LOG_SIGMAS, WAVELET_NAME, WAVELET_LEVELS, XS_LINE_ALPHA

MAX_DIM_FOR_STD = 2048   # downsample to this before generic_filter (performance)
_DISPLAY_SMOOTH_SIGMA = 1.0   # applied to maps before plotting; does NOT affect metrics


class SpatialDetailAnalyzer:
    """Multi-scale spatial detail comparison: local std, LoG, and wavelet.

    Takes both images simultaneously and produces side-by-side comparison
    figures. All processing uses mean-signal-normalised data.
    """

    def analyze(self, image_a: AstroImage, image_b: AstroImage,
                kernel_sizes: tuple = STD_KERNEL_SIZES,
                log_sigmas: tuple = LOG_SIGMAS,
                wavelet: str = WAVELET_NAME,
                levels: int = WAVELET_LEVELS,
                crosshair: dict | None = None) -> dict:

        image_a.estimate_background()
        image_b.estimate_background()

        norm_a = self._normalise(image_a)
        norm_b = self._normalise(image_b)

        result: dict = {
            "contrast_ratios_a": {},
            "contrast_ratios_b": {},
            "wavelet_snr_a": {},
            "wavelet_snr_b": {},
            "sigma_noise_a": None,
            "sigma_noise_b": None,
        }
        figures: dict = {}

        if norm_a is None or norm_b is None:
            result["warning"] = "Mean signal ≤ 0 in one or both images; spatial analysis skipped."
            return result

        # Nebula / background masks (used for contrast ratio)
        mask_neb_a, mask_bg_a = self._make_masks(image_a)
        mask_neb_b, mask_bg_b = self._make_masks(image_b)

        # Bright-feature bounding box for display cropping (uses image A's mask)
        display_roi = self._nebula_bounding_box(mask_neb_a, norm_a.shape)
        result["display_roi"] = display_roi

        # 1. Local standard deviation maps
        std_figs = self._std_analysis(
            norm_a, norm_b,
            mask_neb_a, mask_bg_a,
            mask_neb_b, mask_bg_b,
            kernel_sizes,
            image_a.label, image_b.label,
            result,
            display_roi=display_roi,
            crosshair=crosshair,
        )
        figures.update(std_figs)

        # 2. Laplacian of Gaussian maps
        log_figs = self._log_analysis(
            norm_a, norm_b, log_sigmas,
            image_a.label, image_b.label,
            display_roi=display_roi,
            crosshair=crosshair,
        )
        figures.update(log_figs)

        # 3. Wavelet decomposition
        wav_figs = self._wavelet_analysis(
            norm_a, norm_b, wavelet, levels,
            image_a.label, image_b.label,
            result,
            display_roi=display_roi,
            crosshair=crosshair,
        )
        figures.update(wav_figs)

        if crosshair is not None:
            figures["xs_context"] = self._plot_context_figure(
                image_a, image_b, image_a.label, image_b.label, crosshair)
            pos_a, prof_a = self._sample_line(norm_a, **crosshair)
            pos_b, prof_b = self._sample_line(norm_b, **crosshair)
            figures["xs_image_profile"] = self._plot_image_profile(
                pos_a, prof_a, pos_b, prof_b, image_a.label, image_b.label)
            pos_a_raw, prof_a_raw = self._sample_line(image_a.data, **crosshair)
            pos_b_raw, prof_b_raw = self._sample_line(image_b.data, **crosshair)
            figures["xs_image_profile_raw"] = self._plot_image_profile(
                pos_a_raw, prof_a_raw, pos_b_raw, prof_b_raw,
                image_a.label, image_b.label,
                title="Cross-section brightness profile (raw counts)",
                ylabel="Pixel value (raw ADU)",
            )

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
                       result: dict,
                       display_roi=None,
                       crosshair=None) -> dict:
        figures = {}
        for ks in kernel_sizes:
            std_a = self._compute_std_map(norm_a, ks)
            std_b = self._compute_std_map(norm_b, ks)

            # Contrast ratios (computed on unsmoothed maps)
            cr_a = self._contrast_ratio(std_a, mask_neb_a, mask_bg_a)
            cr_b = self._contrast_ratio(std_b, mask_neb_b, mask_bg_b)
            result["contrast_ratios_a"][ks] = cr_a
            result["contrast_ratios_b"][ks] = cr_b

            fig = self._plot_side_by_side(
                std_a, std_b,
                f"Local σ — kernel {ks}px — {label_a}",
                f"Local σ — kernel {ks}px — {label_b}",
                diff_title=f"Diff (A−B), kernel {ks}px",
                cmap="viridis",
                nonlinear_norm=True,
                display_roi=display_roi,
            )
            figures[f"std_{ks}px"] = fig

            if crosshair is not None:
                pos, pa = self._sample_line(std_a, **crosshair)
                _, pb = self._sample_line(std_b, **crosshair)
                figures[f"xs_std_{ks}px"] = self._plot_cross_section(
                    pos, pa, pb, label_a, label_b,
                    f"Cross-section — Local σ, kernel {ks}px")

        return figures

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

    # ------------------------------------------------------------------
    # Laplacian of Gaussian maps
    # ------------------------------------------------------------------

    def _log_analysis(self, norm_a, norm_b, sigmas,
                       label_a, label_b,
                       display_roi=None,
                       crosshair=None) -> dict:
        figures = {}
        for sigma in sigmas:
            log_a = np.abs(gaussian_laplace(norm_a, sigma=sigma))
            log_b = np.abs(gaussian_laplace(norm_b, sigma=sigma))
            fig = self._plot_side_by_side(
                log_a, log_b,
                f"|LoG| σ={sigma}px — {label_a}",
                f"|LoG| σ={sigma}px — {label_b}",
                diff_title=f"LoG diff (A−B), σ={sigma}px",
                cmap="hot",
                nonlinear_norm=True,
                display_roi=display_roi,
            )
            figures[f"log_sigma{sigma}"] = fig
            if crosshair is not None:
                pos, pa = self._sample_line(log_a, **crosshair)
                _, pb = self._sample_line(log_b, **crosshair)
                figures[f"xs_log_sigma{sigma}"] = self._plot_cross_section(
                    pos, pa, pb, label_a, label_b,
                    f"Cross-section — |LoG|, σ={sigma}px")
        return figures

    # ------------------------------------------------------------------
    # Wavelet decomposition
    # ------------------------------------------------------------------

    def _wavelet_analysis(self, norm_a, norm_b, wavelet, levels,
                           label_a, label_b, result: dict,
                           display_roi=None,
                           crosshair=None) -> dict:
        figures = {}

        coeffs_a = pywt.wavedec2(norm_a, wavelet, level=levels,
                                   mode="periodization")
        coeffs_b = pywt.wavedec2(norm_b, wavelet, level=levels,
                                   mode="periodization")

        sigma_a = self._estimate_noise(coeffs_a)
        sigma_b = self._estimate_noise(coeffs_b)
        result["sigma_noise_a"] = sigma_a
        result["sigma_noise_b"] = sigma_b

        # Per-level SNR (index 1 = coarsest detail, -1 = finest detail)
        for lvl_idx in range(1, levels + 1):
            # coeffs layout: [approx, detail_coarsest, ..., detail_finest]
            # level 1 = finest = coeffs[-1]; level N = coarsest = coeffs[1]
            coeff_idx = levels + 1 - lvl_idx  # map human level to list index
            snr_a = self._level_snr(coeffs_a, coeff_idx, sigma_a, lvl_idx)
            snr_b = self._level_snr(coeffs_b, coeff_idx, sigma_b, lvl_idx)
            result["wavelet_snr_a"][lvl_idx] = snr_a
            result["wavelet_snr_b"][lvl_idx] = snr_b

        # SNR bar chart
        figures["wavelet_snr"] = self._plot_snr_bars(
            result["wavelet_snr_a"], result["wavelet_snr_b"], label_a, label_b, levels)

        # Reconstruct and display levels 2 and 3 (best signal content)
        for display_level in [2, 3]:
            if display_level > levels:
                continue
            coeff_idx = levels + 1 - display_level
            rec_a = self._reconstruct_level(coeffs_a, coeff_idx, wavelet, levels)
            rec_b = self._reconstruct_level(coeffs_b, coeff_idx, wavelet, levels)
            fig = self._plot_side_by_side(
                rec_a, rec_b,
                f"Wavelet level {display_level} — {label_a}",
                f"Wavelet level {display_level} — {label_b}",
                diff_title=f"Level {display_level} diff (A−B)",
                cmap="RdBu_r",
                symmetric_diff=True,
                display_roi=display_roi,
            )
            figures[f"wavelet_level{display_level}"] = fig
            if crosshair is not None:
                pos, pa = self._sample_line(rec_a, **crosshair)
                _, pb = self._sample_line(rec_b, **crosshair)
                figures[f"xs_wavelet_level{display_level}"] = self._plot_cross_section(
                    pos, pa, pb, label_a, label_b,
                    f"Cross-section — Wavelet level {display_level}")

        return figures

    def _estimate_noise(self, coeffs) -> float:
        # Finest-level horizontal detail (last element, first sub-band)
        lh1 = coeffs[-1][0]
        return float(np.median(np.abs(lh1))) / 0.6745

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

    def _plot_side_by_side(self, arr_a: np.ndarray, arr_b: np.ndarray,
                            title_a: str, title_b: str,
                            diff_title: str = "",
                            cmap: str = "viridis",
                            symmetric_diff: bool = False,
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

        if symmetric_diff:
            d_max = float(np.percentile(np.abs(diff), 99.5)) or 1.0
            dvmin, dvmax = -d_max, d_max
        else:
            dvmin = float(np.percentile(diff, 0.5))
            dvmax = float(np.percentile(diff, 99.5))

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
            im = ax.imshow(arr, origin="lower", cmap=cmap,
                           norm=norm if norm is not None else None,
                           vmin=None if norm is not None else vmin,
                           vmax=None if norm is not None else vmax,
                           interpolation="nearest", aspect="equal")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        im_diff = ax_diff.imshow(diff, origin="lower", cmap="RdBu_r",
                                  vmin=dvmin, vmax=dvmax,
                                  interpolation="nearest", aspect="equal")
        ax_diff.set_title(diff_title, fontsize=10)
        ax_diff.axis("off")
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

        return fig

    def _plot_snr_bars(self, snr_a: dict, snr_b: dict,
                        label_a: str, label_b: str, levels: int) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, levels + 1)
        width = 0.35
        vals_a = [snr_a.get(lvl) or 0.0 for lvl in x]
        vals_b = [snr_b.get(lvl) or 0.0 for lvl in x]
        bars_a = ax.bar(x - width / 2, vals_a, width, label=label_a, color="steelblue")
        bars_b = ax.bar(x + width / 2, vals_b, width, label=label_b, color="tomato")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                   label="SNR = 1 (signal = noise)")
        ax.set_xlabel("Wavelet level (1 = finest ~2px, 4 = coarsest ~16px)")
        ax.set_ylabel("Signal energy / Noise energy")
        ax.set_title("Wavelet per-level SNR comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Level {i}" for i in x])
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
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
        ax2.set_ylabel("Difference (A−B)", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax2.legend(loc="upper right", fontsize=8)
        ax1.set_title(title, fontsize=10)
        return fig

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
