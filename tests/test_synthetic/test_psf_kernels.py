"""Unit tests for PSF kernel helper functions in synthetic/generator.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from synthetic.generator import (
    _gaussian_2d,
    _moffat_2d,
    _ring_kernel,
    _fwhm_to_alpha,
    _star_psf,
)

SIZE = 31   # odd stamp size used for all kernel tests


class TestGaussian2D:
    def test_all_nonnegative(self):
        k = _gaussian_2d(SIZE, sigma_x=2.0)
        assert np.all(k >= 0)

    def test_peak_at_centre(self):
        k = _gaussian_2d(SIZE, sigma_x=2.0)
        cy, cx = SIZE // 2, SIZE // 2
        assert float(k[cy, cx]) == pytest.approx(float(k.max()), rel=1e-6)

    def test_symmetric_no_offset(self):
        k = _gaussian_2d(SIZE, sigma_x=3.0)
        np.testing.assert_allclose(k, k[::-1, ::-1], atol=1e-10)

    def test_with_offset_shifts_peak(self):
        k_off = _gaussian_2d(SIZE, sigma_x=2.0, cx=2.0)
        k_ctr = _gaussian_2d(SIZE, sigma_x=2.0, cx=0.0)
        assert not np.allclose(k_off, k_ctr)

    def test_anisotropic_wider_in_y(self):
        k = _gaussian_2d(SIZE, sigma_x=2.0, sigma_y=5.0)
        cy, cx = SIZE // 2, SIZE // 2
        hw_x = int(np.sum(k[cy, :] > 0.5 * k[cy, cx]))
        hw_y = int(np.sum(k[:, cx] > 0.5 * k[cy, cx]))
        assert hw_x < hw_y

    def test_correct_shape(self):
        k = _gaussian_2d(SIZE, sigma_x=2.0)
        assert k.shape == (SIZE, SIZE)


class TestMoffat2D:
    def test_all_nonnegative(self):
        k = _moffat_2d(SIZE, alpha=3.0, beta=4.0)
        assert np.all(k >= 0)

    def test_peak_at_centre(self):
        k = _moffat_2d(SIZE, alpha=3.0, beta=4.0)
        cy, cx = SIZE // 2, SIZE // 2
        assert float(k[cy, cx]) == pytest.approx(float(k.max()), rel=1e-6)

    def test_symmetric(self):
        k = _moffat_2d(SIZE, alpha=3.0, beta=4.0)
        np.testing.assert_allclose(k, k[::-1, ::-1], atol=1e-10)

    def test_lower_beta_wider_wings(self):
        k_sharp = _moffat_2d(SIZE, alpha=3.0, beta=8.0)
        k_wide  = _moffat_2d(SIZE, alpha=3.0, beta=2.0)
        cy, cx  = SIZE // 2, SIZE // 2
        # Normalise by peak
        k_sharp_n = k_sharp / k_sharp[cy, cx]
        k_wide_n  = k_wide  / k_wide[cy, cx]
        # At a fixed radius, lower beta (wider PSF) has more flux
        r = 8
        assert float(k_sharp_n[cy, cx + r]) < float(k_wide_n[cy, cx + r])

    def test_correct_shape(self):
        k = _moffat_2d(SIZE, alpha=3.0, beta=4.0)
        assert k.shape == (SIZE, SIZE)


class TestRingKernel:
    def test_sums_to_one(self):
        k = _ring_kernel(radius_px=8.0, stamp_size=SIZE)
        assert float(k.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_centre_near_zero(self):
        k = _ring_kernel(radius_px=8.0, stamp_size=SIZE)
        cy, cx = SIZE // 2, SIZE // 2
        assert float(k[cy, cx]) < 0.01

    def test_ring_has_mass_at_radius(self):
        k = _ring_kernel(radius_px=8.0, stamp_size=SIZE)
        cy, cx = SIZE // 2, SIZE // 2
        assert float(k[cy, cx + 8]) > 0.0

    def test_correct_shape(self):
        k = _ring_kernel(radius_px=5.0, stamp_size=SIZE)
        assert k.shape == (SIZE, SIZE)


class TestFwhmToAlpha:
    @pytest.mark.parametrize("beta", [2.0, 3.5, 4.77, 6.0])
    def test_roundtrip(self, beta):
        fwhm = 5.0
        alpha = _fwhm_to_alpha(fwhm, beta)
        # FWHM = 2*alpha*sqrt(2^(1/beta) - 1)
        recovered = 2.0 * alpha * math.sqrt(2.0 ** (1.0 / beta) - 1.0)
        assert recovered == pytest.approx(fwhm, rel=1e-6)

    def test_larger_fwhm_larger_alpha(self):
        alpha4 = _fwhm_to_alpha(4.0, 4.0)
        alpha8 = _fwhm_to_alpha(8.0, 4.0)
        assert alpha8 > alpha4

    def test_positive_output(self):
        assert _fwhm_to_alpha(4.0, 4.0) > 0.0


class TestStarPsf:
    _BASE = {
        "fwhm_arcsec": 3.0, "moffat_beta": 4.0,
        "halo": 0.0, "guiding": 0.0, "coma": 0.0,
        "astigmatism": 0.0, "spherical": 0.0, "collimation": 0.0,
        "backfocus": 0.0, "poor_focus": 0.0, "field_curvature": 0.0,
    }

    def test_sums_to_one(self):
        psf = _star_psf(self._BASE, nx=0.0, ny=0.0,
                        plate_scale=1.5, stamp_size=SIZE)
        assert float(psf.sum()) == pytest.approx(1.0, rel=0.02)

    def test_correct_shape(self):
        psf = _star_psf(self._BASE, nx=0.0, ny=0.0,
                        plate_scale=1.5, stamp_size=SIZE)
        assert psf.shape == (SIZE, SIZE)

    def test_peak_near_centre_no_aberrations(self):
        psf = _star_psf(self._BASE, nx=0.0, ny=0.0,
                        plate_scale=1.5, stamp_size=SIZE)
        cy, cx = SIZE // 2, SIZE // 2
        assert float(psf[cy, cx]) == pytest.approx(float(psf.max()), rel=0.1)

    def test_nonnegative(self):
        psf = _star_psf(self._BASE, nx=0.5, ny=0.3,
                        plate_scale=1.5, stamp_size=SIZE)
        assert float(psf.min()) >= -1e-9   # tiny FFT ringing allowed

    def test_both_field_positions_normalise_to_one(self):
        center = _star_psf(self._BASE, nx=0.0, ny=0.0,
                           plate_scale=1.5, stamp_size=SIZE)
        corner = _star_psf(self._BASE, nx=0.8, ny=0.8,
                           plate_scale=1.5, stamp_size=SIZE)
        assert float(center.sum()) == pytest.approx(1.0, rel=0.05)
        assert float(corner.sum()) == pytest.approx(1.0, rel=0.05)

    def test_halo_broadens_psf(self):
        params_halo = {**self._BASE, "halo": 0.4}
        psf_clean = _star_psf(self._BASE,    nx=0.0, ny=0.0,
                               plate_scale=1.5, stamp_size=SIZE)
        psf_halo  = _star_psf(params_halo, nx=0.0, ny=0.0,
                               plate_scale=1.5, stamp_size=SIZE)
        cy, cx = SIZE // 2, SIZE // 2
        # Halo redistributes flux to wings: core value should be lower with halo
        assert float(psf_halo[cy, cx]) < float(psf_clean[cy, cx])
