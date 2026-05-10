from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.modeling import fitting
from astropy.modeling.models import Moffat2D
from astropy.table import Table
from photutils.psf import EPSFBuilder, extract_stars

from core.astro_image import AstroImage
from core.models import SEEING_WARN_FWHM_ARCS
from analysis.star_catalog import StarCatalogBuilder

CUTOUT_SIZE = 25   # pixels per side for per-star cutouts
EPSF_OVERSAMPLING = 2
EPSF_MAXITERS = 5


def _moffat_fwhm(gamma: float, alpha: float) -> float:
    """FWHM from astropy Moffat2D gamma/alpha parameters."""
    return 2.0 * gamma * np.sqrt(2.0 ** (1.0 / alpha) - 1.0)


class PSFAnalyzer:
    """Fit Moffat PSF to stars, build empirical PSF, compute MTF."""

    def __init__(self):
        self._catalog_builder = StarCatalogBuilder()

    def analyze(self, image: AstroImage) -> dict:
        image.estimate_background()
        catalog = self._catalog_builder.build(image)
        fwhm_guess = 4.0
        psf_stars = self._catalog_builder.filter_psf_stars(catalog, image, fwhm_guess)

        result: dict = {
            "n_stars_total": len(catalog),
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
        moffat_fits = self._fit_moffat_all(bgsub, psf_stars)
        if not moffat_fits:
            return result

        fwhms = [f["fwhm"] for f in moffat_fits]
        betas = [f["alpha"] for f in moffat_fits]
        median_fwhm = float(np.median(fwhms))
        median_beta = float(np.median(betas))
        fwhm_arcsec = median_fwhm * image.pixel_scale

        # Ellipticity + per-star eccentricity from image moments (needs bgsub + psf_stars)
        ell, pa, ell_per_star = self._measure_ellipticity(bgsub, psf_stars)
        ecc_by_pos = {(s["x"], s["y"]): s["eccentricity"] for s in ell_per_star}

        result.update({
            "n_stars_used": len(moffat_fits),
            "fwhm_px": median_fwhm,
            "fwhm_arcsec": fwhm_arcsec,
            "beta": median_beta,
            "seeing_dominated": fwhm_arcsec > SEEING_WARN_FWHM_ARCS,
            "star_data": [
                {"x": f["x"], "y": f["y"], "fwhm": f["fwhm"],
                 "eccentricity": ecc_by_pos.get((f["x"], f["y"]))}
                for f in moffat_fits
            ],
        })

        result["ellipticity"] = ell
        result["position_angle"] = pa
        result["eccentricity"] = (
            float(np.median([s["eccentricity"] for s in ell_per_star]))
            if ell_per_star else None
        )

        # Empirical PSF and MTF
        epsf, freq, mtf = self._build_epsf_and_mtf(image, psf_stars, median_fwhm)
        if epsf is not None:
            mtf50 = self._find_mtf50(freq, mtf)
            mtf_nyq = float(np.interp(0.5, freq, mtf))
            result["mtf50_cycles_per_px"] = mtf50
            result["mtf_nyquist"] = mtf_nyq
            result["epsf_data"] = epsf                       # raw 2D array for convolution
            result["epsf_oversampling"] = EPSF_OVERSAMPLING  # = 2
            result["figures"] = {
                "mtf": self._plot_mtf(freq, mtf, mtf50, image.label),
                "epsf": self._plot_epsf(epsf, image.label),
            }

        return result

    # ------------------------------------------------------------------
    # Moffat fitting
    # ------------------------------------------------------------------

    def _fit_moffat_all(self, bgsub: np.ndarray, stars: Table) -> list[dict]:
        fitter = fitting.LevMarLSQFitter()
        results = []
        h, w = bgsub.shape
        half = CUTOUT_SIZE // 2

        for row in stars:
            xc = int(round(row["xcentroid"]))
            yc = int(round(row["ycentroid"]))
            x0 = max(0, xc - half)
            y0 = max(0, yc - half)
            x1 = min(w, xc + half + 1)
            y1 = min(h, yc + half + 1)
            cutout = bgsub[y0:y1, x0:x1].copy()
            if cutout.size == 0:
                continue

            cy, cx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
            amp = float(np.max(cutout))
            cx0 = cutout.shape[1] / 2.0
            cy0 = cutout.shape[0] / 2.0

            model = Moffat2D(amplitude=amp, x_0=cx0, y_0=cy0, gamma=2.0, alpha=2.5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fitted = fitter(model, cx, cy, cutout)
                except Exception:
                    continue

            gamma = abs(fitted.gamma.value)
            alpha = abs(fitted.alpha.value)
            if alpha < 0.5 or gamma < 0.1:
                continue
            fwhm = _moffat_fwhm(gamma, alpha)
            if fwhm < 0.5 or fwhm > CUTOUT_SIZE:
                continue
            results.append({"x": xc, "y": yc, "fwhm": fwhm, "alpha": alpha, "gamma": gamma})

        return results

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
            xc = int(round(row["xcentroid"]))
            yc = int(round(row["ycentroid"]))
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
        stars_tbl = Table()
        stars_tbl["x"] = stars["xcentroid"]
        stars_tbl["y"] = stars["ycentroid"]

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
                epsf, _ = builder(extracted)
            epsf_data = epsf.data
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
                       origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_title(f"ePSF — {label}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig
