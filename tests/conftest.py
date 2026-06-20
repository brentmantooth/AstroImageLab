"""Session-scoped fixtures shared by all test modules.

Hand-crafted FITS fixture for analysis tests (fast, no PSF rendering).
SyntheticGenerator preview fixture for generator/kernel tests.
SyntheticGenerator full FITS fixture for slow integration tests only.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

import numpy as np
import pytest
from astropy.io import fits as ap_fits
from pathlib import Path

from core.astro_image import AstroImage

# Smallest available camera: 1920x1080 -> 480x270 in preview mode
_CAMERA = "Player One — Mercury-M"

SMALL_PARAMS: dict = {
    "camera": _CAMERA,
    "focal_length_mm": 500.0,
    "aperture_mm": 100.0,
    "n_stars": 20,
    "exposure_s": 300.0,
    "gain_e_per_adu": 1.0,
    "bortle": 4,
    "fwhm_arcsec": 3.0,
    "moffat_beta": 3.5,
    "halo": 0.05,
    "guiding": 0.3,
    "coma": 0.0,
    "astigmatism": 0.0,
    "spherical": 0.0,
    "collimation": 0.0,
    "backfocus": 0.0,
    "poor_focus": 0.0,
    "field_curvature": 0.0,
    "mag_min": 8.0,
    "mag_max": 14.0,
    "nebula_enabled": False,
    "seed": 42,
}


def _make_test_fits(path: Path) -> None:
    """Write a 512x512 FITS with Poisson sky and 30 isolated Gaussian stars.

    Stars are grid-placed at ~70px horizontal / ~58px vertical spacing so they
    survive the 20px isolation filter in StarCatalogBuilder.filter_psf_stars().
    GAIN=0 in the header intentionally tests the gain-guard logic.
    """
    rng = np.random.default_rng(0)
    h, w = 512, 512
    sky_level = 1000.0
    sky_noise = 32.0
    sigma_px = 1.7   # FWHM ~4 px, matches DAOStarFinder fwhm default

    data = rng.normal(sky_level, sky_noise, (h, w))

    # 5-col x 6-row grid, stars safely inside the 50-px border exclusion zone
    for row in range(6):
        for col in range(5):
            cx = 80 + col * 70 + int(rng.uniform(-8, 8))
            cy = 80 + row * 58 + int(rng.uniform(-8, 8))
            peak = float(rng.uniform(8000, 35000))
            r = 12
            yslice = slice(max(0, cy - r), min(h, cy + r + 1))
            xslice = slice(max(0, cx - r), min(w, cx + r + 1))
            yy, xx = np.mgrid[yslice, xslice]
            g = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma_px ** 2)
            data[yslice, xslice] += peak * g

    data = np.clip(data, 0, 65535).astype(np.float32)

    hdr = ap_fits.Header()
    hdr["EGAIN"]    = 1.0       # physical e-/ADU — must be preferred over GAIN
    hdr["GAIN"]     = 0         # mode index, zero — must be rejected by SNR analyzer
    hdr["FOCALLEN"] = 500.0     # mm
    hdr["XPIXSZ"]   = 3.76     # um -> pixel_scale ~1.55 arcsec/px
    hdr["EXPTIME"]  = 300.0
    hdr["INSTRUME"] = "TestCam"
    ap_fits.writeto(str(path), data, hdr, overwrite=True)


@pytest.fixture(scope="session")
def test_fits_path(tmp_path_factory) -> Path:
    """Path to the hand-crafted test FITS, written once per session."""
    out = tmp_path_factory.mktemp("fits") / "test.fits"
    _make_test_fits(out)
    return out


@pytest.fixture(scope="session")
def astro_image_a(test_fits_path) -> AstroImage:
    """Loaded + background-estimated AstroImage shared across all analysis tests."""
    img = AstroImage(str(test_fits_path), label="TestA")
    img.load()
    img.estimate_background()
    return img


@pytest.fixture(scope="session")
def synth_params() -> dict:
    """Default synthetic generator parameter set (smallest camera, fixed seed)."""
    return dict(SMALL_PARAMS)


@pytest.fixture(scope="session")
def synth_preview(synth_params) -> np.ndarray:
    """Preview-mode synthetic array (480x270, float32, no file I/O)."""
    from synthetic.generator import SyntheticGenerator
    return SyntheticGenerator().generate(synth_params, preview=True)


@pytest.fixture(scope="session")
def synth_fits_path(tmp_path_factory, synth_params) -> tuple[str, str]:
    """Full-resolution FITS pair (1920x1080). Used only by slow/integration tests."""
    out = tmp_path_factory.mktemp("synth")
    params = {**synth_params, "output_dir": str(out)}
    from synthetic.generator import SyntheticGenerator
    return SyntheticGenerator().generate(params, preview=False)
