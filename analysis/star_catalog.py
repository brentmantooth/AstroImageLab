from __future__ import annotations

import warnings

import numpy as np
from astropy.table import Table
from photutils.detection import DAOStarFinder

from core.models import MIN_STAR_SNR, ISOLATION_RADIUS_FWHM, SATURATION_FRACTION
from core.astro_image import AstroImage


class StarCatalogBuilder:
    """Detect and filter stars suitable for PSF and halo analysis."""

    def build(self, image: AstroImage, fwhm_guess: float = 4.0) -> Table:
        """Run DAOStarFinder on the background-subtracted image."""
        bgsub = image.background_subtracted()
        rms = image.background_rms
        if rms is None:
            rms_val = float(np.std(bgsub))
        else:
            rms_val = float(np.median(rms))

        threshold = 5.0 * rms_val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold)
            catalog = finder(bgsub)

        if catalog is None:
            catalog = Table(names=["xcentroid", "ycentroid", "peak", "flux",
                                    "sharpness", "roundness1", "roundness2"],
                             dtype=[float] * 7)
        image.catalog = catalog
        return catalog

    def filter_psf_stars(self,
                          catalog: Table,
                          image: AstroImage,
                          fwhm_estimate: float,
                          border_px: int = 50,
                          min_snr: float = MIN_STAR_SNR) -> Table:
        """Remove saturated, low-SNR, non-isolated, and border-adjacent stars."""
        if len(catalog) == 0:
            return catalog

        sat_thresh = image.saturation_threshold()
        rms = image.background_rms
        rms_val = float(np.median(rms)) if rms is not None else 1.0

        height, width = image.data.shape

        keep = np.ones(len(catalog), dtype=bool)

        for i, row in enumerate(catalog):
            # Saturation check
            if row["peak"] >= sat_thresh:
                keep[i] = False
                continue

            # S/N check
            snr = row["peak"] / rms_val if rms_val > 0 else 0.0
            if snr < min_snr:
                keep[i] = False
                continue

            # Border check
            x, y = row["xcentroid"], row["ycentroid"]
            if x < border_px or x > width - border_px:
                keep[i] = False
                continue
            if y < border_px or y > height - border_px:
                keep[i] = False
                continue

        # Isolation check — only among candidates that passed above filters
        candidates = catalog[keep]
        cand_indices = np.where(keep)[0]
        isolation_radius = ISOLATION_RADIUS_FWHM * fwhm_estimate

        isolation_ok = np.ones(len(candidates), dtype=bool)
        xs = np.array(candidates["xcentroid"])
        ys = np.array(candidates["ycentroid"])
        for i in range(len(candidates)):
            dists = np.sqrt((xs - xs[i])**2 + (ys - ys[i])**2)
            dists[i] = np.inf  # ignore self
            if np.min(dists) < isolation_radius:
                isolation_ok[i] = False

        final_indices = cand_indices[isolation_ok]
        return catalog[final_indices]
