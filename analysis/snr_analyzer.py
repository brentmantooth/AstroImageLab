from __future__ import annotations

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import bottleneck as bn
except ImportError:
    bn = np

from core.astro_image import AstroImage, _resolve_gain
from core.fig_utils import fig_to_b64, finalize_layout


class SNRAnalyzer:
    """Compute signal-to-noise ratio metrics from pre-computed background estimates."""

    def analyze(self, image: AstroImage) -> dict:
        image.estimate_background()
        bgsub = image.background_subtracted()
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
        # image.catalog may be None if PSFAnalyzer hasn't run yet (parallel mode),
        # so build it on demand here without filtering.
        cat = getattr(image, "catalog", None)
        if cat is None:
            from analysis.star_catalog import StarCatalogBuilder
            cat = StarCatalogBuilder().build(image)
        star_snr_median: float | None = None
        star_snr_iqr: float | None = None
        if cat is not None and len(cat) > 0 and noise > 0:
            peaks = np.asarray(cat["peak"], dtype=np.float32)
            star_snrs = peaks / noise
            star_snrs = star_snrs[np.isfinite(star_snrs) & (star_snrs > 0)]
            if star_snrs.size > 0:
                star_snr_median = float(np.median(star_snrs))
                star_snr_iqr = float(
                    np.percentile(star_snrs, 75) - np.percentile(star_snrs, 25)
                )

        # --- Local SNR map -----------------------------------------------
        if rms is not None:
            denom = np.where(rms > 0, rms.astype(np.float32), np.nan)
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
        # Median of the 2D background model in ADU; lower = darker sky. Reads the
        # adopted model, not background.background -- the latter is always the
        # unmasked phase-1 estimate, so it would report a sky level that nothing
        # else in this result was actually computed against.
        background_median = (float(bn.median(image.background_model))
                             if image.background_model is not None else None)

        # --- Camera gain from FITS header --------------------------------
        # See core.astro_image._resolve_gain for the EGAIN-priority rationale
        # (GAIN alone is often an ambiguous camera mode index, not e⁻/ADU).
        gain_e_per_adu: float | None = _resolve_gain(getattr(image, 'header', None))

        # --- Noise factor: σ_sky / √μ_sky --------------------------------
        # 1.0 = pure sky shot noise (Poisson); >1.0 = read noise / thermal contributions.
        # Narrowband images through narrow filters are commonly read-noise dominated (factor 2–10).
        noise_factor: float | None = None
        if background_median is not None and background_median > 0:
            noise_factor = noise / math.sqrt(background_median)

        # --- Sky in electrons (requires gain from FITS header) -----------
        sky_bg_electrons: float | None = (
            background_median * gain_e_per_adu
            if (background_median is not None and gain_e_per_adu is not None) else None)
        sky_noise_electrons: float | None = (
            noise * gain_e_per_adu if gain_e_per_adu is not None else None)

        # --- Figure (per-image, for PNG export) ---------------------------
        snr_map_fig = self._plot_snr_map(snr_map, image.label)

        return {
            "snr_global":        snr_global,
            "snr_global_db":     snr_global_db,
            "noise_median":      noise,     # median sky RMS (σ_sky) in ADU
            "background_median": background_median,  # median sky background (μ_sky) in ADU
            "star_snr_median":   star_snr_median,
            "star_snr_iqr":      star_snr_iqr,
            "pct_above_3":       pcts[3],
            "pct_above_5":       pcts[5],
            "pct_above_10":      pcts[10],
            "pct_above_20":      pcts[20],
            "snr_display":       snr_display,
            "snr_p2":            snr_p2,
            "snr_p98":           snr_p98,
            "gain_e_per_adu":    gain_e_per_adu,      # e⁻/ADU from FITS header, or None
            "noise_factor":      noise_factor,          # σ_sky / √μ_sky
            "sky_bg_electrons":  sky_bg_electrons,      # μ_sky × gain, or None
            "sky_noise_electrons": sky_noise_electrons, # σ_sky × gain, or None
            "figures":           {"snr_map": snr_map_fig},
        }

    def _plot_snr_map(self, snr_map: np.ndarray, label: str) -> str:
        finite_pos = snr_map[np.isfinite(snr_map) & (snr_map > 0)]
        vmin = float(np.percentile(finite_pos, 2)) if finite_pos.size > 0 else 0.0
        vmax = float(np.percentile(finite_pos, 98)) if finite_pos.size > 0 else 10.0

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            snr_map,
            origin="upper",
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label="SNR (σ)", fraction=0.046, pad=0.04)
        ax.set_title(f"SNR map — {label}", fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        finalize_layout(fig)
        return fig_to_b64(fig, dpi=120)
