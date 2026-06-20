"""Unit tests for analysis/psf_analyzer.py."""
from __future__ import annotations

import pytest

from analysis.psf_analyzer import PSFAnalyzer

_REQUIRED_KEYS = {
    "n_stars_total", "n_psf_candidates", "n_stars_used",
    "fwhm_px", "fwhm_arcsec", "beta", "ellipticity",
}


class TestAnalyze:
    def test_returns_dict(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        assert isinstance(result, dict)

    def test_required_keys_present(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_n_stars_total_nonnegative(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        assert result["n_stars_total"] >= 0

    def test_n_stars_used_le_total(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        assert result["n_stars_used"] <= result["n_stars_total"]

    def test_fwhm_positive_or_none(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        fwhm = result["fwhm_px"]
        assert fwhm is None or fwhm > 0.0

    def test_fwhm_arcsec_consistent_with_px(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        if result["fwhm_px"] and result["fwhm_arcsec"]:
            # fwhm_arcsec = fwhm_px * pixel_scale
            ratio = result["fwhm_arcsec"] / result["fwhm_px"]
            assert 0.5 < ratio < 5.0   # plausible pixel scale range

    def test_beta_in_plausible_range(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        beta = result["beta"]
        if beta is not None:
            assert 0.5 < beta < 20.0

    def test_ellipticity_in_unit_range(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        e = result["ellipticity"]
        if e is not None:
            assert 0.0 <= e <= 1.0

    def test_figures_key_present_when_analysis_succeeds(self, astro_image_a):
        result = PSFAnalyzer().analyze(astro_image_a)
        if result["n_stars_used"] > 0:
            assert "figures" in result

    def test_second_run_consistent(self, astro_image_a):
        r1 = PSFAnalyzer().analyze(astro_image_a)
        r2 = PSFAnalyzer().analyze(astro_image_a)
        # Both runs should report the same star count
        assert r1["n_stars_total"] == r2["n_stars_total"]
