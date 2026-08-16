from __future__ import annotations

import warnings

import numpy as np
try:
    import bottleneck as bn
except ImportError:
    bn = np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit

from core.astro_image import AstroImage
from core.fig_utils import figs_to_b64, finalize_layout
from core.models import HALO_FIT_RADIUS_PX, HALO_MIN_STAR_SNR, RDF_BIN_WIDTH
from analysis.star_catalog import StarCatalogBuilder

RADIAL_BIN_WIDTH = 0.5   # px; bin width for the median-profile used in Moffat fitting
N_SATURATED_DISPLAY = 10


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

        bgsub = image.background_subtracted()
        h, w = image.data.shape
        rms_map = image.background_rms

        # Log10-transform for RDF: gives balanced statistics across the halo's
        # dynamic range; floor at the 1st percentile of positive pixels.
        _pos = bgsub[bgsub > 0]
        _log_floor = float(np.percentile(_pos, 1)) if _pos.size > 0 else 1.0
        log_bgsub = np.log10(np.clip(bgsub, _log_floor, None))

        halo_stars = self._select_halo_stars(catalog, image)

        profiles = []
        per_star = []

        for row in halo_stars:
            xc, yc = float(row["x_centroid"]), float(row["y_centroid"])
            r, I = self._extract_radial_profile(bgsub, xc, yc, HALO_FIT_RADIUS_PX)
            if len(r) < 10:
                continue
            peak = I[0] if I[0] > 0 else 1.0
            I_norm = I / peak
            profiles.append((r, I_norm))

            fit = self._fit_two_component(r, I_norm)
            star_entry: dict = {"xc": xc, "yc": yc, "peak": float(row["peak"])}
            if fit is not None:
                star_entry["halo_to_core_ratio"] = fit["halo_to_core_ratio"]
                star_entry["halo_radius_px"]     = fit["halo_radius_px"]

            # Local 1σ background noise at this star's own position, expressed in
            # each plot's own normalisation so report_builder can axhline it directly.
            bg_rms = self._bg_rms_at(rms_map, xc, yc, h, w)
            if bg_rms is not None:
                star_entry["bg_rms"] = bg_rms
                if I[0] > 0:   # only the real peak, not the 1.0 fallback
                    star_entry["bg_over_peak"] = bg_rms / peak

            rdf_result = self._annular_stats(log_bgsub, xc, yc, HALO_FIT_RADIUS_PX)
            if rdf_result is not None:
                r_rdf, mu, sig = rdf_result
                norm = mu[0]                    # log10 value at centre bin
                star_entry["rdf_radii"] = r_rdf
                star_entry["rdf_mean"]  = mu - norm   # 0 at r=0 → 10^0 = 1 in display
                star_entry["rdf_std"]   = sig          # already in log10 units
                if bg_rms is not None:
                    star_entry["rdf_bg_fraction"] = bg_rms / (10.0 ** norm)

            per_star.append(star_entry)

        figs_dict: dict = {}

        if profiles:
            result["star_data"] = per_star

            common_r = profiles[0][0]
            stacked = bn.median(
                np.array([np.interp(common_r, p[0], p[1]) for p in profiles]),
                axis=0
            )
            result["n_stars_fitted"] = len(profiles)
            result["radial_radii"]   = common_r
            result["radial_profile"] = stacked

            bg_over_peak_vals = [s["bg_over_peak"] for s in per_star if "bg_over_peak" in s]
            profile_bg_level = float(np.median(bg_over_peak_vals)) if bg_over_peak_vals else None
            result["profile_bg_level"] = profile_bg_level

            fit = self._fit_two_component(common_r, stacked)
            if fit is not None:
                result["halo_to_core_ratio"] = fit["halo_to_core_ratio"]
                result["halo_radius_px"]     = fit["halo_radius_px"]
                figs_dict["halo_profile"]    = self._plot_profile(
                    common_r, stacked, fit, image.label, bg_level=profile_bg_level)

            agg_unsat = self._compute_aggregate_rdf(
                [s for s in per_star if "rdf_mean" in s])
            if agg_unsat is not None:
                result["rdf_radii"]       = agg_unsat["rdf_radii"]
                result["rdf_mean"]        = agg_unsat["rdf_mean"]
                result["rdf_std"]         = agg_unsat["rdf_std"]
                result["rdf_star_std"]    = agg_unsat["rdf_star_std"]
                result["rdf_n_stars"]     = agg_unsat["rdf_n_stars"]
                result["rdf_bg_fraction"] = agg_unsat["rdf_bg_fraction"]

        # --- Saturated stars (always collected, regardless of unsaturated count) ---
        sat_stars = self._collect_saturated_stars(catalog, image)
        for s in sat_stars:
            bg_rms = self._bg_rms_at(rms_map, s["xc"], s["yc"], h, w)
            if bg_rms is not None:
                s["bg_rms"] = bg_rms
            rdf_result = self._annular_stats(log_bgsub, s["xc"], s["yc"], HALO_FIT_RADIUS_PX)
            if rdf_result is not None:
                r_rdf, mu, sig = rdf_result
                norm = mu[0]                    # log10 value at centre bin
                s["rdf_radii"] = r_rdf
                s["rdf_mean"]  = mu - norm      # 0 at r=0 → 10^0 = 1 in display
                s["rdf_std"]   = sig            # already in log10 units
                if bg_rms is not None:
                    s["rdf_bg_fraction"] = bg_rms / (10.0 ** norm)

        result["saturated_star_data"] = sat_stars

        agg_sat = self._compute_aggregate_rdf(
            [s for s in sat_stars if "rdf_mean" in s])
        if agg_sat is not None:
            result["sat_rdf_radii"]       = agg_sat["rdf_radii"]
            result["sat_rdf_mean"]        = agg_sat["rdf_mean"]
            result["sat_rdf_std"]         = agg_sat["rdf_std"]
            result["sat_rdf_star_std"]    = agg_sat["rdf_star_std"]
            result["sat_rdf_bg_fraction"] = agg_sat["rdf_bg_fraction"]
            result["sat_rdf_n_stars"]  = agg_sat["rdf_n_stars"]

        if figs_dict:
            result["figures"] = figs_to_b64(figs_dict)
        return result

    # ------------------------------------------------------------------
    # Star selection — unsaturated halo stars (unchanged)
    # ------------------------------------------------------------------

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
            x, y = row["x_centroid"], row["y_centroid"]
            margin = HALO_FIT_RADIUS_PX + 5
            if x < margin or x > w - margin or y < margin or y > h - margin:
                continue
            keep.append(row)

        if not keep:
            return catalog[:0]

        # Isolation: no other halo-candidate within HALO_FIT_RADIUS_PX
        isolated = []
        xs = np.array([r["x_centroid"] for r in keep])
        ys = np.array([r["y_centroid"] for r in keep])
        for i in range(len(keep)):
            dists = np.sqrt((xs - xs[i])**2 + (ys - ys[i])**2)
            dists[i] = np.inf
            if np.min(dists) >= HALO_FIT_RADIUS_PX:
                isolated.append(keep[i])

        from astropy.table import Table
        if not isolated:
            return catalog[:0]
        return Table(rows=isolated)

    # ------------------------------------------------------------------
    # Saturated star detection — connected components on raw pixels
    # ------------------------------------------------------------------

    def _collect_saturated_stars(self, catalog, image: AstroImage) -> list:
        """Find saturated stars via connected-component analysis on saturated pixels.

        Uses scipy.ndimage.label on pixels >= saturation threshold instead of the
        catalog, because DAOStarFinder fails to detect heavily blown-out cores
        (flat plateau — no identifiable Gaussian peak).
        """
        from scipy.ndimage import label as _label
        sat = image.saturation_threshold()
        h, w = image.data.shape
        margin = HALO_FIT_RADIUS_PX + 5

        # Primary: connected regions of saturated pixels → one star each
        sat_mask = image.data >= sat
        labeled, n_features = _label(sat_mask)

        cc_stars: list[dict] = []
        for region_idx in range(1, n_features + 1):
            ys, xs = np.where(labeled == region_idx)
            if len(xs) < 4:             # skip cosmic rays / hot pixels
                continue
            xc = float(np.mean(xs))
            yc = float(np.mean(ys))
            if xc < margin or xc > w - margin or yc < margin or yc > h - margin:
                continue
            peak = float(image.data[ys, xs].max())
            cc_stars.append({"xc": xc, "yc": yc, "peak": peak,
                              "n_sat_px": len(xs)})

        # Supplementary: catalog entries that are saturated but form no large
        # connected region (moderately saturated — not fully clipped)
        for row in catalog:
            if row["peak"] < sat:
                continue
            x, y = float(row["x_centroid"]), float(row["y_centroid"])
            if x < margin or x > w - margin or y < margin or y > h - margin:
                continue
            if any(np.sqrt((x - s["xc"])**2 + (y - s["yc"])**2) < 30.0
                   for s in cc_stars):
                continue    # already captured by a CC region
            cc_stars.append({"xc": x, "yc": y, "peak": float(row["peak"]),
                              "n_sat_px": 0})

        # Sort by saturated-pixel area (proxy for total brightness), then peak
        cc_stars.sort(key=lambda s: (s["n_sat_px"], s["peak"]), reverse=True)
        return [{"xc": s["xc"], "yc": s["yc"], "peak": s["peak"]}
                for s in cc_stars[:N_SATURATED_DISPLAY]]

    # ------------------------------------------------------------------
    # Background reference
    # ------------------------------------------------------------------

    @staticmethod
    def _bg_rms_at(rms_map: np.ndarray | None, xc: float, yc: float,
                    h: int, w: int) -> float | None:
        """Local 1σ background noise (ADU) at a star's pixel location."""
        if rms_map is None:
            return None
        iy = min(max(int(round(yc)), 0), h - 1)
        ix = min(max(int(round(xc)), 0), w - 1)
        val = float(rms_map[iy, ix])
        return val if val > 0 else None

    # ------------------------------------------------------------------
    # Radial profiles
    # ------------------------------------------------------------------

    def _extract_radial_profile(self, data: np.ndarray,
                                  xc: float, yc: float,
                                  max_radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Median intensity in each 0.5 px annular bin — used for Moffat fitting."""
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

    def _annular_stats(self, data: np.ndarray, xc: float, yc: float,
                        max_radius: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Mean and std of pixel values in each 1 px annular bin — used for RDF."""
        h, w = data.shape
        x0 = max(0, int(xc - max_radius))
        y0 = max(0, int(yc - max_radius))
        x1 = min(w, int(xc + max_radius + 1))
        y1 = min(h, int(yc + max_radius + 1))
        sub = data[y0:y1, x0:x1].astype(float)
        yg, xg = np.mgrid[y0:y1, x0:x1]
        r_map = np.sqrt((xg - xc)**2 + (yg - yc)**2)

        edges = np.arange(0, max_radius + RDF_BIN_WIDTH, RDF_BIN_WIDTH)
        r_centers, means, stds = [], [], []
        for i in range(len(edges) - 1):
            mask = (r_map >= edges[i]) & (r_map < edges[i + 1])
            if mask.sum() > 2:
                vals = sub[mask]
                r_centers.append((edges[i] + edges[i + 1]) / 2.0)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))

        if len(r_centers) < 3:
            return None
        return np.array(r_centers), np.array(means), np.array(stds)

    def _compute_aggregate_rdf(self, rdf_list: list) -> dict | None:
        """Stack per-star RDF dicts and return cross-star aggregate statistics.

        Stars near image edges produce shorter radius arrays, so each profile is
        interpolated onto the longest common grid; np.nanmean handles bins that
        fall outside a given star's covered radius range.
        """
        if not rdf_list:
            return None

        # Reference grid = the star with the most radial bins
        r_ref = max((d["rdf_radii"] for d in rdf_list), key=len)

        stacked_means = []
        stacked_stds  = []
        bg_fractions  = []
        for d in rdf_list:
            r = d["rdf_radii"]
            m = d["rdf_mean"]
            s = d["rdf_std"]
            if len(r) < 3:
                continue
            # Extrapolated bins (beyond this star's range) become NaN
            m_i = np.interp(r_ref, r, m, left=np.nan, right=np.nan)
            s_i = np.interp(r_ref, r, s, left=np.nan, right=np.nan)
            stacked_means.append(m_i)
            stacked_stds.append(s_i)
            if "rdf_bg_fraction" in d:
                bg_fractions.append(d["rdf_bg_fraction"])

        if not stacked_means:
            return None

        all_means = np.array(stacked_means)
        all_stds  = np.array(stacked_stds)
        return {
            "rdf_radii":       r_ref,
            "rdf_mean":        bn.nanmean(all_means, axis=0),
            "rdf_std":         bn.nanmean(all_stds,  axis=0),   # avg per-bin σ
            "rdf_star_std":    bn.nanstd(all_means,  axis=0),   # star-to-star σ
            "rdf_n_stars":     len(stacked_means),
            "rdf_bg_fraction": float(np.median(bg_fractions)) if bg_fractions else None,
        }

    # ------------------------------------------------------------------
    # Moffat fitting
    # ------------------------------------------------------------------

    def _fit_two_component(self, r: np.ndarray,
                             I_norm: np.ndarray) -> dict | None:
        log_I = np.log10(np.clip(I_norm, 1e-6, None))

        p0 = [1.0, 2.0, 2.5,   # core: amp, gamma, alpha
              0.1, 20.0, 1.5]  # halo: amp, gamma, alpha
        bounds = (
            [0,  0.1,  0.5,  0,               5.0, 0.5],
            [5, 15.0, 10.0,  2, HALO_FIT_RADIUS_PX, 5.0],
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

        if gamma_core > gamma_halo:
            amp_core, gamma_core, alpha_core, amp_halo, gamma_halo, alpha_halo = \
                amp_halo, gamma_halo, alpha_halo, amp_core, gamma_core, alpha_core

        halo_to_core = amp_halo / amp_core if amp_core > 0 else float("inf")
        halo_r: float | None = gamma_halo * np.sqrt(2.0 ** (1.0 / alpha_halo) - 1.0)
        # HWHM beyond the data window is extrapolation — mark as unreliable
        if halo_r > HALO_FIT_RADIUS_PX:
            halo_r = None

        return {
            "popt": popt,
            "halo_to_core_ratio": halo_to_core,
            "halo_radius_px": halo_r,
        }

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _plot_profile(self, r: np.ndarray, I_norm: np.ndarray,
                       fit: dict, label: str,
                       bg_level: float | None = None) -> plt.Figure:
        halo_r = fit["halo_radius_px"]   # may be None if HWHM > data window
        popt = fit["popt"]

        r_max_plot = max(r[-1], (halo_r or r[-1]) * 1.2)
        r_fine = np.linspace(r[0], r_max_plot, 400)

        import matplotlib
        _is_dark = matplotlib.rcParams.get("figure.facecolor", "white") not in ("white", "#ffffff", 1.0)
        meas_color = "w" if _is_dark else "k"

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(r, I_norm, f"{meas_color}.", markersize=3, label="Measured")

        core = _moffat1d(r_fine, popt[0], popt[1], popt[2])
        halo = _moffat1d(r_fine, popt[3], popt[4], popt[5])
        total = core + halo
        ax.semilogy(r_fine, total, "b-", linewidth=1.5, label="Total fit")
        ax.semilogy(r_fine, core, "g--", linewidth=1, label="Core")
        ax.semilogy(r_fine, halo, "r--", linewidth=1, label="Halo")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2g}'))

        ax.axvline(r[-1], color="gray", linestyle=":", linewidth=1.2,
                   label=f"Data limit ({r[-1]:.0f} px)")

        if halo_r is not None:
            ax.axvline(halo_r, color="tomato", linestyle="--", linewidth=1.0,
                       alpha=0.8, label=f"R_halo = {halo_r:.0f} px")

        if bg_level is not None and bg_level > 0:
            ax.axhline(bg_level, color="cyan", linestyle="--", linewidth=0.8,
                       alpha=0.9, label=f"Background (1σ) ≈ {bg_level:.2g}")

        halo_r_str = f"{halo_r:.1f} px" if halo_r is not None else "N/A (> data window)"
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Normalised intensity")
        ax.set_title(f"Halo profile — {label}\n"
                     f"Halo/core = {fit['halo_to_core_ratio']:.5f}, "
                     f"R_halo = {halo_r_str}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        finalize_layout(fig)
        return fig
