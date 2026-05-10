from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from photutils.detection import DAOStarFinder

from core.astro_image import AstroImage
from core.models import GHOST_SEARCH_RADIUS_PX
from analysis.star_catalog import StarCatalogBuilder

N_BRIGHTEST = 10
INNER_EXCLUSION_FWHM = 3.0   # avoid PSF wings inside this multiple of FWHM
GHOST_DETECTION_SIGMA = 3.0


class GhostDetector:
    """Search for ghost images (secondary reflections) near bright stars."""

    def __init__(self):
        self._catalog_builder = StarCatalogBuilder()

    def analyze(self, image: AstroImage,
                psf_fwhm_px: float = 4.0) -> dict:
        image.estimate_background()
        catalog = self._catalog_builder.build(image, fwhm_guess=psf_fwhm_px)

        result: dict = {
            "ghost_candidates": [],
            "n_bright_stars_searched": 0,
        }

        if len(catalog) == 0:
            return result

        bgsub = image.background_subtracted()
        rms_val = float(np.median(image.background_rms)) \
            if image.background_rms is not None else float(np.std(bgsub))
        sat = image.saturation_threshold()

        bright_stars = self._identify_bright_stars(catalog, sat)
        result["n_bright_stars_searched"] = len(bright_stars)

        candidates = []
        inner_r = INNER_EXCLUSION_FWHM * psf_fwhm_px

        for parent in bright_stars:
            found = self._search_for_ghosts(
                bgsub, parent, catalog, rms_val, inner_r, image.data.shape
            )
            candidates.extend(found)

        result["ghost_candidates"] = candidates

        if candidates:
            result["figures"] = {
                "ghost_map": self._plot_ghost_map(
                    bgsub, bright_stars, candidates, image.label)
            }

        return result

    def _identify_bright_stars(self, catalog: Table, sat_thresh: float) -> Table:
        unsaturated = catalog[catalog["peak"] < sat_thresh]
        if len(unsaturated) == 0:
            return catalog[:0]
        sort_idx = np.argsort(unsaturated["peak"])[::-1]
        return unsaturated[sort_idx[:N_BRIGHTEST]]

    def _search_for_ghosts(self, bgsub: np.ndarray,
                             parent, full_catalog: Table,
                             rms_val: float,
                             inner_r: float,
                             shape: tuple) -> list[dict]:
        h, w = shape
        px = float(parent["xcentroid"])
        py = float(parent["ycentroid"])

        # Build residual: subtract all known catalog sources as point estimates
        residual = bgsub.copy()
        for row in full_catalog:
            xi = int(round(row["xcentroid"]))
            yi = int(round(row["ycentroid"]))
            r_local = max(5, int(inner_r))
            x0 = max(0, xi - r_local)
            y0 = max(0, yi - r_local)
            x1 = min(w, xi + r_local + 1)
            y1 = min(h, yi + r_local + 1)
            # Replace with local background estimate (median of annulus)
            residual[y0:y1, x0:x1] = 0.0

        # Define annular search region
        yg, xg = np.mgrid[0:h, 0:w]
        dist_from_parent = np.sqrt((xg - px)**2 + (yg - py)**2)
        search_mask = (dist_from_parent > inner_r) & \
                      (dist_from_parent <= GHOST_SEARCH_RADIUS_PX)

        if not np.any(search_mask):
            return []

        # Run a second detection pass on the residual in the search region
        search_data = np.where(search_mask, residual, 0.0)
        threshold = GHOST_DETECTION_SIGMA * rms_val

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = DAOStarFinder(fwhm=4.0, threshold=threshold)
            ghost_table = finder(search_data)

        if ghost_table is None or len(ghost_table) == 0:
            return []

        # Cross-match against original catalog; flag sources not already known
        results = []
        for g in ghost_table:
            gx, gy = g["xcentroid"], g["ycentroid"]
            dists_to_known = np.sqrt(
                (full_catalog["xcentroid"] - gx)**2 +
                (full_catalog["ycentroid"] - gy)**2
            )
            if np.min(dists_to_known) < 5.0:
                continue  # already in catalog

            sep = float(np.sqrt((gx - px)**2 + (gy - py)**2))
            ratio = float(g["peak"] / parent["peak"]) if parent["peak"] > 0 else 0.0
            classification = self._classify(gx, gy, px, py)
            results.append({
                "dx": float(gx - px),
                "dy": float(gy - py),
                "separation_px": sep,
                "intensity_ratio": ratio,
                "classification": classification,
                "parent_x": px,
                "parent_y": py,
            })

        return results

    def _classify(self, gx: float, gy: float,
                   px: float, py: float) -> str:
        angle = abs(np.degrees(np.arctan2(gy - py, gx - px))) % 90
        if angle < 10 or angle > 80:
            return "diffraction_spike"
        # Mirror-symmetric ghosts cluster around specific axis offsets
        return "probable_ghost"

    def _plot_ghost_map(self, bgsub: np.ndarray, bright_stars: Table,
                         candidates: list[dict], label: str) -> plt.Figure:
        from astropy.visualization import ImageNormalize, AsinhStretch
        fig, ax = plt.subplots(figsize=(7, 7))
        norm = ImageNormalize(bgsub, stretch=AsinhStretch(a=0.05))
        ax.imshow(bgsub, origin="lower", cmap="gray", norm=norm)

        for row in bright_stars:
            circ = plt.Circle((row["xcentroid"], row["ycentroid"]),
                               radius=15, color="cyan", fill=False, linewidth=1.2)
            ax.add_patch(circ)

        top_cands = sorted(candidates,
                           key=lambda c: c.get("intensity_ratio", 0),
                           reverse=True)[:25]
        for c in top_cands:
            ax.annotate("",
                        xy=(c["parent_x"] + c["dx"], c["parent_y"] + c["dy"]),
                        xytext=(c["parent_x"], c["parent_y"]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
            ax.plot(c["parent_x"] + c["dx"], c["parent_y"] + c["dy"],
                    "o", markersize=9, markeredgewidth=1.5,
                    markeredgecolor="red", markerfacecolor="none")

        total = len(candidates)
        note = f"  (showing top 25 of {total})" if total > 25 else ""
        ax.set_title(f"Ghost candidates — {label}\n"
                     f"Cyan=searched stars, Red=ghost candidates{note}")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        fig.tight_layout()
        return fig
