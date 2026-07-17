from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation as mad, mad_std
from astropy.table import Table
from photutils.psf import EPSFBuilder, extract_stars

from core.astro_image import AstroImage
from core.fig_utils import figs_to_b64
from core.models import (SEEING_WARN_FWHM_ARCS, PSF_BETA_MIN, PSF_BETA_MAX, PSF_FWHM_CLIP_NSIGMA,
                          ABERRATION_MIN_STARS, ABERRATION_OUTER_RADIUS_FRAC, EPSF_MAX_STARS)
from analysis.moffat_fit import fit_moffat2d_core
from analysis.star_catalog import StarCatalogBuilder

CUTOUT_SIZE = 25   # pixels per side for per-star cutouts
EPSF_OVERSAMPLING = 2
EPSF_MAXITERS = 15


class PSFAnalyzer:
    """Fit Moffat PSF to stars, build empirical PSF, compute MTF."""

    def __init__(self, epsf_max_stars: int = EPSF_MAX_STARS):
        self._catalog_builder = StarCatalogBuilder()
        self._epsf_max_stars = epsf_max_stars

    def analyze(self, image: AstroImage) -> dict:
        image.estimate_background()
        catalog = self._catalog_builder.build(image)
        fwhm_guess = 4.0
        psf_stars = self._catalog_builder.filter_psf_stars(catalog, image, fwhm_guess)

        result: dict = {
            "n_stars_total": len(catalog),
            "n_psf_candidates": 0,
            "n_outliers_rejected": 0,
            "n_stars_used": 0,
            "fwhm_px": None,
            "fwhm_arcsec": None,
            "beta": None,
            "ellipticity": None,
            "position_angle": None,
            "mtf50_cycles_per_px": None,
            "mtf_nyquist": None,
            "seeing_dominated": False,
            "star_data": [],
        }

        if len(psf_stars) < 3:
            return result

        bgsub = image.background_subtracted()
        n_psf_candidates = len(psf_stars)
        moffat_fits = self._fit_moffat_all(bgsub, psf_stars)
        if not moffat_fits:
            return result

        # Second-pass: FWHM sigma-clip (needs full fit set to compute median/MAD)
        moffat_fits = self._sigma_clip_fwhm(moffat_fits)
        if not moffat_fits:
            return result

        # Build clean_psf_stars matching only the sigma-clipped fits so that
        # ellipticity measurement and ePSF build use the same clean star set.
        clean_xy = {(f["x"], f["y"]) for f in moffat_fits}
        clean_psf_stars = psf_stars[np.array([
            (int(round(row["x_centroid"])), int(round(row["y_centroid"]))) in clean_xy
            for row in psf_stars
        ])]

        fwhms = [f["fwhm"] for f in moffat_fits]
        betas = [f["alpha"] for f in moffat_fits]
        median_fwhm = float(np.median(fwhms))
        median_beta = float(np.median(betas))
        fwhm_arcsec = median_fwhm * image.pixel_scale
        fwhm_mad        = float(mad(fwhms))         if len(fwhms) > 1 else 0.0
        beta_mad        = float(mad(betas))          if len(betas) > 1 else 0.0
        fwhm_arcsec_mad = fwhm_mad * image.pixel_scale

        # Ellipticity + per-star eccentricity from image moments — use clean_psf_stars
        ell, pa, ell_per_star = self._measure_ellipticity(bgsub, clean_psf_stars)
        ecc_by_pos    = {(s["x"], s["y"]): s["eccentricity"]  for s in ell_per_star}
        ell_by_pos    = {(s["x"], s["y"]): s["ellipticity"]   for s in ell_per_star}
        orient_by_pos = {(s["x"], s["y"]): s["orientation"]   for s in ell_per_star}

        ecc_values = [s["eccentricity"] for s in ell_per_star]
        ell_values = [s["ellipticity"]  for s in ell_per_star]
        ell_mad = float(mad(ell_values)) if len(ell_values) > 1 else 0.0
        ecc_mad = float(mad(ecc_values)) if len(ecc_values) > 1 else 0.0

        _rms = image.background_rms
        _noise = float(np.median(_rms)) if _rms is not None else 1.0
        _noise = max(_noise, 1e-9)

        result.update({
            "n_psf_candidates":    n_psf_candidates,
            "n_outliers_rejected": n_psf_candidates - len(moffat_fits),
            "n_stars_used":        len(moffat_fits),
            "fwhm_px":           median_fwhm,
            "fwhm_arcsec":       fwhm_arcsec,
            "beta":              median_beta,
            "seeing_dominated":  fwhm_arcsec > SEEING_WARN_FWHM_ARCS,
            "fwhm_px_mad":       fwhm_mad,
            "fwhm_arcsec_mad":   fwhm_arcsec_mad,
            "beta_mad":          beta_mad,
            "star_data": [
                {
                    "x":           f["x"],
                    "y":           f["y"],
                    "fwhm":        f["fwhm"],
                    "fwhm_arcsec": f["fwhm"] * image.pixel_scale,
                    "beta":        f["alpha"],
                    "eccentricity": ecc_by_pos.get((f["x"], f["y"])),
                    "ellipticity":  ell_by_pos.get((f["x"], f["y"])),
                    "orientation":  orient_by_pos.get((f["x"], f["y"])),  # radians, photutils convention
                    "snr":          f["peak"] / _noise,
                }
                for f in moffat_fits
            ],
        })

        result["ellipticity"]     = ell
        result["ellipticity_mad"] = ell_mad
        result["position_angle"]  = pa
        result["eccentricity"] = (
            float(np.median(ecc_values)) if ecc_values else None
        )
        result["eccentricity_mad"] = ecc_mad

        img_h, img_w = bgsub.shape
        result["aberration"] = self._aberration_analysis(result["star_data"], img_h, img_w)

        # Empirical PSF and MTF — use clean_psf_stars (same set as metric statistics)
        epsf, freq, mtf = self._build_epsf_and_mtf(image, clean_psf_stars, median_fwhm)
        if epsf is not None:
            mtf50 = self._find_mtf50(freq, mtf)
            mtf_nyq = float(np.interp(0.5, freq, mtf))
            result["mtf50_cycles_per_px"] = mtf50
            result["mtf_nyquist"] = mtf_nyq
            result["epsf_data"] = epsf                       # raw 2D array for convolution
            result["epsf_oversampling"] = EPSF_OVERSAMPLING  # = 2
            result["epsf_n_stars"]    = getattr(self, "_last_epsf_n_stars",    None)
            result["epsf_converged"]  = getattr(self, "_last_epsf_converged",  None)
            result["epsf_iterations"] = getattr(self, "_last_epsf_iterations", None)
            result["mtf_freq"] = freq
            result["mtf_curve"] = mtf
            result["figures"] = figs_to_b64({
                "mtf": self._plot_mtf(freq, mtf, mtf50, image.label),
                "epsf": self._plot_epsf(epsf, image.label),
            })

        return result

    # ------------------------------------------------------------------
    # Moffat fitting
    # ------------------------------------------------------------------

    def _fit_moffat_all(self, bgsub: np.ndarray, stars: Table) -> list[dict]:
        results = []
        h, w = bgsub.shape
        half = CUTOUT_SIZE // 2

        for row in stars:
            xc = int(round(row["x_centroid"]))
            yc = int(round(row["y_centroid"]))
            x0 = max(0, xc - half)
            y0 = max(0, yc - half)
            x1 = min(w, xc + half + 1)
            y1 = min(h, yc + half + 1)
            cutout = bgsub[y0:y1, x0:x1].copy()

            fit = fit_moffat2d_core(
                cutout,
                alpha_bounds=(PSF_BETA_MIN, PSF_BETA_MAX),
                gamma_min=0.1,
                fwhm_bounds=(0.5, CUTOUT_SIZE),
            )
            if fit is None:
                continue
            results.append({"x": xc, "y": yc, **fit})

        return results

    @staticmethod
    def _sigma_clip_fwhm(fits: list[dict]) -> list[dict]:
        """Remove fits whose FWHM deviates more than PSF_FWHM_CLIP_NSIGMA*sigma from median.

        Uses mad_std (the sigma-scaled MAD, ~1.4826x raw MAD) rather than the raw
        median_absolute_deviation used for the reported *_mad dispersion fields, so
        PSF_FWHM_CLIP_NSIGMA is an actual sigma multiple rather than a raw-MAD multiple.
        Requires at least 5 fits; returns the list unchanged if the scaled MAD is zero.
        """
        if len(fits) < 5:
            return fits
        fwhms = np.array([f["fwhm"] for f in fits])
        med = float(np.median(fwhms))
        m = float(mad_std(fwhms))
        if m == 0:
            return fits
        return [f for f in fits if abs(f["fwhm"] - med) <= PSF_FWHM_CLIP_NSIGMA * m]

    # ------------------------------------------------------------------
    # Field aberration analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _aberration_analysis(stars: list[dict], img_h: int, img_w: int) -> dict:
        """Compute per-field aberration indices from per-star orientation + eccentricity data.

        Returns a dict with scalar scores and per-star arrays for plotting.
        Returns a warning key instead when fewer than ABERRATION_MIN_STARS stars
        have valid orientation measurements.
        """
        from scipy.stats import circstd

        stars_ok = [s for s in stars
                    if s.get("orientation") is not None
                    and s.get("eccentricity") is not None
                    and s.get("fwhm") is not None]
        n = len(stars_ok)
        if n < ABERRATION_MIN_STARS:
            return {
                "warning": (f"Only {n} stars with orientation data "
                            f"(minimum {ABERRATION_MIN_STARS}); aberration analysis skipped."),
                "n_stars_used": n,
            }

        xs    = np.array([s["x"]            for s in stars_ok], dtype=float)
        ys    = np.array([s["y"]            for s in stars_ok], dtype=float)
        ecc   = np.array([s["eccentricity"] for s in stars_ok], dtype=float)
        theta = np.array([s["orientation"]  for s in stars_ok], dtype=float)
        fwhm  = np.array([s["fwhm"]         for s in stars_ok], dtype=float)

        cx, cy = img_w / 2.0, img_h / 2.0
        ri  = np.hypot(xs - cx, ys - cy)
        phi = np.arctan2(ys - cy, xs - cx)

        # Radial/tangential decomposition of eccentricity
        er = ecc * np.cos(2.0 * (theta - phi))  # positive = elongated away from centre
        et = ecc * np.sin(2.0 * (theta - phi))  # positive = elongated tangentially CCW

        # Coma index: Pearson r between |er| and radial distance
        abs_er = np.abs(er)
        if np.std(ri) > 0 and np.std(abs_er) > 0:
            coma_index = float(np.corrcoef(ri, abs_er)[0, 1])
        else:
            coma_index = 0.0

        # Radial elongation fraction: mean |er|/ecc for sufficiently elongated stars
        mask_elong = ecc > 0.05
        radial_frac = float(np.mean(abs_er[mask_elong] / ecc[mask_elong])) if mask_elong.sum() > 0 else 0.0

        # Collimation uniformity: circular std of orientation angles
        # Use double-angle trick so π-periodicity is handled correctly
        collimation_circstd_deg = float(np.degrees(circstd(2.0 * theta)) / 2.0)

        # FWHM radial gradient: quadratic fit on normalised radius
        r_norm = ri / max(float(ri.max()), 1.0)
        coeffs = np.polyfit(r_norm, fwhm, 2)
        fwhm_radial_linear    = float(coeffs[1])   # linear coefficient
        fwhm_radial_quadratic = float(coeffs[0])   # quadratic coefficient

        # Corner/centre FWHM ratio
        r_frac = ri / max(float(ri.max()), 1.0)
        outer_mask = r_frac >= (1.0 - ABERRATION_OUTER_RADIUS_FRAC)
        inner_mask = r_frac <= ABERRATION_OUTER_RADIUS_FRAC
        corner_centre_ratio = None
        if outer_mask.sum() >= 3 and inner_mask.sum() >= 3:
            corner_centre_ratio = float(np.median(fwhm[outer_mask]) / np.median(fwhm[inner_mask]))

        # Quadrant mean orientations (degrees) for astigmatism hint
        quads = {
            "NW": (xs < cx) & (ys < cy),
            "NE": (xs >= cx) & (ys < cy),
            "SW": (xs < cx) & (ys >= cy),
            "SE": (xs >= cx) & (ys >= cy),
        }
        quad_mean_deg: dict[str, float] = {}
        for qname, qmask in quads.items():
            if qmask.sum() >= 3:
                t = theta[qmask]
                mean_rad = float(np.arctan2(np.mean(np.sin(2.0 * t)),
                                             np.mean(np.cos(2.0 * t))) / 2.0)
                quad_mean_deg[qname] = float(np.degrees(mean_rad))

        return {
            "n_stars_used":             n,
            "coma_index":               coma_index,
            "radial_frac":              radial_frac,
            "collimation_circstd_deg":  collimation_circstd_deg,
            "fwhm_radial_linear":       fwhm_radial_linear,
            "fwhm_radial_quadratic":    fwhm_radial_quadratic,
            "corner_centre_ratio":      corner_centre_ratio,
            "quad_mean_deg":            quad_mean_deg,
            # Per-star arrays for plotting (stored as Python lists for JSON-ability)
            "star_xs":    xs.tolist(),
            "star_ys":    ys.tolist(),
            "star_ecc":   ecc.tolist(),
            "star_theta": theta.tolist(),
            "star_er":    er.tolist(),
            "star_et":    et.tolist(),
        }

    # ------------------------------------------------------------------
    # Ellipticity via image moments
    # ------------------------------------------------------------------

    def _measure_ellipticity(self, bgsub: np.ndarray,
                               stars: Table) -> tuple[float, float, list]:
        from photutils.morphology import data_properties
        ellipticities = []
        pas = []
        per_star = []
        h, w = bgsub.shape
        half = CUTOUT_SIZE // 2

        for row in stars:
            xc = int(round(row["x_centroid"]))
            yc = int(round(row["y_centroid"]))
            x0 = max(0, xc - half)
            y0 = max(0, yc - half)
            x1 = min(w, xc + half + 1)
            y1 = min(h, yc + half + 1)
            cutout = bgsub[y0:y1, x0:x1].copy()
            cutout = np.clip(cutout, 0, None)
            if cutout.sum() == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    props = data_properties(cutout)
                ellipticities.append(float(props.ellipticity.value))
                pas.append(float(props.orientation.value))
                per_star.append({
                    "x": xc, "y": yc,
                    "eccentricity": float(props.eccentricity.value),
                    "ellipticity":  float(props.ellipticity.value),
                    "orientation":  float(props.orientation.value),  # radians
                })
            except Exception:
                continue

        if not ellipticities:
            return 0.0, 0.0, []
        return float(np.median(ellipticities)), float(np.median(pas)), per_star

    # ------------------------------------------------------------------
    # Empirical PSF and MTF
    # ------------------------------------------------------------------

    def _build_epsf_and_mtf(self, image: AstroImage, stars: Table,
                              fwhm_estimate: float
                              ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        nddata = image.nddata()

        # Limit to the brightest N stars to keep ePSF build time manageable
        if self._epsf_max_stars > 0 and len(stars) > self._epsf_max_stars:
            order = np.argsort(np.asarray(stars["peak"]))[::-1]
            stars = stars[order[:self._epsf_max_stars]]

        stars_tbl = Table()
        stars_tbl["x"] = stars["x_centroid"]
        stars_tbl["y"] = stars["y_centroid"]

        box_size = max(CUTOUT_SIZE, int(fwhm_estimate * 6) | 1)
        if box_size % 2 == 0:
            box_size += 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                extracted = extract_stars(nddata, stars_tbl, size=box_size)
                builder = EPSFBuilder(
                    oversampling=EPSF_OVERSAMPLING,
                    maxiters=EPSF_MAXITERS,
                    progress_bar=False,
                )
                epsf, epsf_result = builder(extracted)
            epsf_data = epsf.data
            # EPSFBuilder returns (EPSFModel, EPSFStars) in most photutils versions.
            # EPSFStars is a star collection — len() gives the count actually used.
            # Newer photutils may return a result object with .converged/.iterations;
            # getattr handles both APIs without raising AttributeError.
            self._last_epsf_n_stars    = len(epsf_result)
            self._last_epsf_converged  = getattr(epsf_result, "converged",  None)
            self._last_epsf_iterations = getattr(epsf_result, "iterations", None)
        except Exception:
            return None, np.array([]), np.array([])

        freq, mtf = self._compute_mtf(epsf_data)
        return epsf_data, freq, mtf

    def _compute_mtf(self, epsf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Azimuthally averaged MTF from ePSF via FFT."""
        # Normalize PSF to unit sum
        epsf_norm = epsf / (epsf.sum() or 1.0)
        fft2d = np.fft.fftshift(np.fft.fft2(epsf_norm))
        otf = np.abs(fft2d)
        otf /= otf.max() or 1.0  # MTF(0) = 1

        n = epsf.shape[0]
        cx, cy = n // 2, n // 2
        y_idx, x_idx = np.mgrid[0:n, 0:n]
        r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

        # Frequency in cycles/native-pixel (account for oversampling)
        max_r = n / 2.0
        freq_max = 0.5 / EPSF_OVERSAMPLING   # Nyquist of native pixels

        nbins = n // 2
        freq_edges = np.linspace(0, max_r, nbins + 1)
        freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2.0
        freq_axis = freq_centers / max_r * freq_max

        mtf = np.array([
            np.mean(otf[(r >= freq_edges[i]) & (r < freq_edges[i + 1])])
            if np.any((r >= freq_edges[i]) & (r < freq_edges[i + 1])) else 0.0
            for i in range(nbins)
        ])

        return freq_axis, mtf

    def _find_mtf50(self, freq: np.ndarray, mtf: np.ndarray) -> float | None:
        if len(freq) < 2:
            return None
        for i in range(len(mtf) - 1):
            if mtf[i] >= 0.5 >= mtf[i + 1]:
                slope = (mtf[i + 1] - mtf[i]) / (freq[i + 1] - freq[i])
                return float(freq[i] + (0.5 - mtf[i]) / slope)
        return None

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _plot_mtf(self, freq: np.ndarray, mtf: np.ndarray,
                   mtf50: float | None, label: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(freq, mtf, linewidth=2, label=label)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="MTF = 0.5")
        ax.axvline(0.5, color="red", linestyle=":", linewidth=0.8, label="Nyquist")
        if mtf50 is not None:
            ax.axvline(mtf50, color="blue", linestyle="--", linewidth=1.0,
                       label=f"MTF50 = {mtf50:.3f} cyc/px")
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("MTF")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_epsf(self, epsf: np.ndarray, label: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(np.log1p(epsf - epsf.min()),
                       origin="upper", cmap="viridis", interpolation="nearest")
        ax.set_title(f"ePSF — {label}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig
