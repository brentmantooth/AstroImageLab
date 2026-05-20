from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.astro_image import AstroImage
from core.fig_utils import fig_to_b64


class SNRAnalyzer:
    """Compute signal-to-noise ratio metrics from pre-computed background estimates."""

    def analyze(self, image: AstroImage) -> dict:
        image.estimate_background()
        bgsub = image.background_subtracted().astype(np.float64)
        rms = image.background_rms  # 2D array from Background2D

        noise = float(np.median(rms)) if rms is not None else 1.0
        if noise <= 0:
            noise = 1.0

        # --- Global SNR --------------------------------------------------
        # Median signal of pixels clearly above the noise floor (> 3 σ),
        # divided by the median sky RMS.  Pixels below 3 σ are sky-dominated
        # and would bias the signal estimate downward.
        above = bgsub[bgsub > 3.0 * noise]
        snr_global = float(np.median(above) / noise) if above.size > 0 else 0.0

        # --- Per-star SNR ------------------------------------------------
        # Uses peak count from DAOStarFinder catalog, divided by the median
        # background RMS (same estimator used in star_catalog.py filtering).
        cat = getattr(image, "catalog", None)
        star_snr_median: float | None = None
        star_snr_iqr: float | None = None
        if cat is not None and len(cat) > 0 and noise > 0:
            peaks = np.asarray(cat["peak"], dtype=np.float64)
            star_snrs = peaks / noise
            star_snrs = star_snrs[np.isfinite(star_snrs) & (star_snrs > 0)]
            if star_snrs.size > 0:
                star_snr_median = float(np.median(star_snrs))
                star_snr_iqr = float(
                    np.percentile(star_snrs, 75) - np.percentile(star_snrs, 25)
                )

        # --- Local SNR map -----------------------------------------------
        if rms is not None:
            denom = np.where(rms > 0, rms.astype(np.float64), np.nan)
        else:
            denom = noise
        snr_map = bgsub / denom

        # --- Percentile table -------------------------------------------
        total = snr_map.size
        pcts = {
            thr: float(np.nansum(snr_map > thr) / total * 100.0)
            for thr in (3, 5, 10, 20)
        }

        # --- Downsampled display array for shared-scale report rendering ---
        h_s, w_s = snr_map.shape
        step_s = max(1, max(h_s, w_s) // 800)
        snr_display = snr_map[::step_s, ::step_s].astype(np.float32)
        finite_pos = snr_display[np.isfinite(snr_display) & (snr_display > 0)]
        snr_p2  = float(np.percentile(finite_pos, 2))  if finite_pos.size > 0 else 0.0
        snr_p98 = float(np.percentile(finite_pos, 98)) if finite_pos.size > 0 else 10.0

        # --- dB equivalent -----------------------------------------------
        # Amplitude-ratio conversion: SNR_dB = 20 log10(SNR_σ)
        snr_global_db = (20.0 * float(np.log10(snr_global))
                         if snr_global > 0 else None)

        # --- Sky background level ----------------------------------------
        # Median of the 2D background model in ADU; lower = darker sky.
        background_median = (float(np.median(image.background.background))
                             if image.background is not None else None)

        # --- Figure (per-image, for PNG export) ---------------------------
        snr_map_fig = self._plot_snr_map(snr_map, image.label)

        return {
            "snr_global":        snr_global,
            "snr_global_db":     snr_global_db,
            "noise_median":      noise,     # median sky RMS (σ_sky) in ADU
            "background_median": background_median,  # median sky background (μ_sky) in ADU
            "star_snr_median": star_snr_median,
            "star_snr_iqr":    star_snr_iqr,
            "pct_above_3":     pcts[3],
            "pct_above_5":     pcts[5],
            "pct_above_10":    pcts[10],
            "pct_above_20":    pcts[20],
            "snr_display":     snr_display,
            "snr_p2":          snr_p2,
            "snr_p98":         snr_p98,
            "figures":         {"snr_map": snr_map_fig},
        }

    def _plot_snr_map(self, snr_map: np.ndarray, label: str) -> str:
        finite_pos = snr_map[np.isfinite(snr_map) & (snr_map > 0)]
        vmin = float(np.percentile(finite_pos, 2)) if finite_pos.size > 0 else 0.0
        vmax = float(np.percentile(finite_pos, 98)) if finite_pos.size > 0 else 10.0

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            snr_map,
            origin="lower",
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label="SNR (σ)", fraction=0.046, pad=0.04)
        ax.set_title(f"SNR map — {label}", fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig.tight_layout()
        return fig_to_b64(fig, dpi=120)
