from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from core.astro_image import AstroImage
from core.models import HALO_FIT_RADIUS_PX, HALO_MIN_STAR_SNR
from analysis.star_catalog import StarCatalogBuilder

RADIAL_BIN_WIDTH = 0.5   # px


def _moffat1d(r: np.ndarray, amp: float, gamma: float, alpha: float) -> np.ndarray:
    return amp / (1.0 + (r / gamma) ** 2) ** alpha


def _two_component(r: np.ndarray,
                    amp_core: float, gamma_core: float, alpha_core: float,
                    amp_halo: float, gamma_halo: float, alpha_halo: float
                    ) -> np.ndarray:
    return _moffat1d(r, amp_core, gamma_core, alpha_core) + \
           _moffat1d(r, amp_halo, gamma_halo, alpha_halo)


class HaloAnalyzer:
    """Extract radial profiles and fit a two-component core+halo model."""

    def __init__(self):
        self._catalog_builder = StarCatalogBuilder()

    def analyze(self, image: AstroImage) -> dict:
        image.estimate_background()
        catalog = self._catalog_builder.build(image)

        result: dict = {
            "halo_radius_px": None,
            "halo_to_core_ratio": None,
            "n_stars_fitted": 0,
            "radial_profile": None,
            "radial_radii": None,
        }

        halo_stars = self._select_halo_stars(catalog, image)
        if len(halo_stars) == 0:
            return result

        bgsub = image.background_subtracted()
        profiles = []
        per_star = []
        for row in halo_stars:
            xc, yc = float(row["xcentroid"]), float(row["ycentroid"])
            r, I = self._extract_radial_profile(bgsub, xc, yc, HALO_FIT_RADIUS_PX)
            if len(r) < 10:
                continue
            # Normalize to peak
            peak = I[0] if I[0] > 0 else 1.0
            I_norm = I / peak
            profiles.append((r, I_norm))
            # Per-star halo fit for ranking
            fit = self._fit_two_component(r, I_norm)
            if fit is not None:
                per_star.append({
                    "xc": xc, "yc": yc,
                    "halo_to_core_ratio": fit["halo_to_core_ratio"],
                    "halo_radius_px": fit["halo_radius_px"],
                })

        if not profiles:
            return result

        per_star.sort(key=lambda s: s["halo_to_core_ratio"], reverse=True)
        result["star_data"] = per_star

        # Median stack normalized profiles
        common_r = profiles[0][0]
        stacked = np.median(
            np.array([np.interp(common_r, p[0], p[1]) for p in profiles]),
            axis=0
        )
        result["n_stars_fitted"] = len(profiles)
        result["radial_radii"] = common_r
        result["radial_profile"] = stacked

        fit = self._fit_two_component(common_r, stacked)
        if fit is not None:
            result["halo_to_core_ratio"] = fit["halo_to_core_ratio"]
            result["halo_radius_px"] = fit["halo_radius_px"]
            result["figures"] = {
                "halo_profile": self._plot_profile(
                    common_r, stacked, fit, image.label)
            }

        return result

    def _select_halo_stars(self, catalog, image: AstroImage):
        if len(catalog) == 0:
            return catalog
        rms = image.background_rms
        rms_val = float(np.median(rms)) if rms is not None else 1.0
        sat = image.saturation_threshold()
        h, w = image.data.shape

        keep = []
        for row in catalog:
            if row["peak"] >= sat:
                continue
            snr = row["peak"] / rms_val if rms_val > 0 else 0.0
            if snr < HALO_MIN_STAR_SNR:
                continue
            x, y = row["xcentroid"], row["ycentroid"]
            margin = HALO_FIT_RADIUS_PX + 5
            if x < margin or x > w - margin or y < margin or y > h - margin:
                continue
            keep.append(row)

        if not keep:
            return catalog[:0]

        # Isolation: no other halo-candidate within HALO_FIT_RADIUS_PX
        isolated = []
        xs = np.array([r["xcentroid"] for r in keep])
        ys = np.array([r["ycentroid"] for r in keep])
        for i in range(len(keep)):
            dists = np.sqrt((xs - xs[i])**2 + (ys - ys[i])**2)
            dists[i] = np.inf
            if np.min(dists) >= HALO_FIT_RADIUS_PX:
                isolated.append(keep[i])

        from astropy.table import Table
        if not isolated:
            return catalog[:0]
        return Table(rows=isolated)

    def _extract_radial_profile(self, data: np.ndarray,
                                  xc: float, yc: float,
                                  max_radius: float) -> tuple[np.ndarray, np.ndarray]:
        h, w = data.shape
        x0 = max(0, int(xc - max_radius))
        y0 = max(0, int(yc - max_radius))
        x1 = min(w, int(xc + max_radius + 1))
        y1 = min(h, int(yc + max_radius + 1))

        sub = data[y0:y1, x0:x1]
        yg, xg = np.mgrid[y0:y1, x0:x1]
        r_map = np.sqrt((xg - xc)**2 + (yg - yc)**2)

        max_r = min(max_radius, r_map.max())
        edges = np.arange(0, max_r + RADIAL_BIN_WIDTH, RADIAL_BIN_WIDTH)
        radii, intensities = [], []
        for i in range(len(edges) - 1):
            mask = (r_map >= edges[i]) & (r_map < edges[i + 1])
            if mask.sum() > 0:
                radii.append((edges[i] + edges[i + 1]) / 2.0)
                intensities.append(float(np.median(sub[mask])))

        return np.array(radii), np.array(intensities)

    def _fit_two_component(self, r: np.ndarray,
                             I_norm: np.ndarray) -> dict | None:
        # Fit in log space for better numerical behaviour
        log_I = np.log10(np.clip(I_norm, 1e-6, None))

        p0 = [1.0, 2.0, 2.5,   # core: amp, gamma, alpha
              0.1, 20.0, 1.5]  # halo: amp, gamma, alpha
        bounds = (
            [0, 0.1, 0.5,   0,  5.0, 0.5],
            [5, 15.0, 10.0, 2, 150.0, 5.0],
        )

        def model_log(r, *p):
            return np.log10(np.clip(_two_component(r, *p), 1e-12, None))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model_log, r, log_I, p0=p0, bounds=bounds,
                                    maxfev=5000)
        except RuntimeError:
            return None

        amp_core, gamma_core, alpha_core, amp_halo, gamma_halo, alpha_halo = popt

        # Guard against role swap
        if gamma_core > gamma_halo:
            amp_core, gamma_core, alpha_core, amp_halo, gamma_halo, alpha_halo = \
                amp_halo, gamma_halo, alpha_halo, amp_core, gamma_core, alpha_core

        halo_to_core = amp_halo / amp_core if amp_core > 0 else float("inf")
        # Halo half-power radius: where halo component drops to amp_halo/2
        halo_r = gamma_halo * np.sqrt(2.0 ** (1.0 / alpha_halo) - 1.0)

        return {
            "popt": popt,
            "halo_to_core_ratio": halo_to_core,
            "halo_radius_px": halo_r,
        }

    def _plot_profile(self, r: np.ndarray, I_norm: np.ndarray,
                       fit: dict, label: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(r, I_norm, "k.", markersize=3, label="Measured")

        popt = fit["popt"]
        r_fine = np.linspace(r[0], r[-1], 300)
        core = _moffat1d(r_fine, popt[0], popt[1], popt[2])
        halo = _moffat1d(r_fine, popt[3], popt[4], popt[5])
        total = core + halo
        ax.semilogy(r_fine, total, "b-", linewidth=1.5, label="Total fit")
        ax.semilogy(r_fine, core, "g--", linewidth=1, label="Core")
        ax.semilogy(r_fine, halo, "r--", linewidth=1, label="Halo")

        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Normalised intensity")
        ax.set_title(f"Halo profile — {label}\n"
                     f"Halo/core = {fit['halo_to_core_ratio']:.3f}, "
                     f"R_halo = {fit['halo_radius_px']:.1f} px")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
