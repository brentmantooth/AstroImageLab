"""Unit tests for analysis/image_filters.py (SpatialDetailAnalyzer)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as ap_fits

from analysis.image_filters import SpatialDetailAnalyzer
from core.astro_image import AstroImage


class TestAnalyze:
    def test_returns_dict(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert isinstance(result, dict)

    def test_contrast_ratios_a_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "contrast_ratios_a" in result

    def test_wavelet_snr_a_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "wavelet_snr_a" in result

    def test_panels_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "panels" in result

    def test_contrast_ratios_are_positive(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        ratios = result.get("contrast_ratios_a") or []
        for r in ratios:
            if r is not None:
                assert r >= 0.0

    def test_wavelet_snr_finite(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        snrs = result.get("wavelet_snr_a") or []
        for s in snrs:
            if s is not None:
                assert np.isfinite(s)

    def test_single_image_b_ratios_empty(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        # Single-image mode: contrast_ratios_b is present but contains no values
        b_ratios = result.get("contrast_ratios_b")
        assert b_ratios is None or not b_ratios   # None or empty dict/list

    def test_minimal_image_no_crash(self, tmp_path):
        data = np.random.default_rng(99).normal(500, 10, (128, 128)).astype(np.float32)
        ap_fits.writeto(str(tmp_path / "tiny.fits"), data, overwrite=True)
        img = AstroImage(str(tmp_path / "tiny.fits"), label="Tiny")
        img.load()
        img.estimate_background()
        result = SpatialDetailAnalyzer().analyze(img)
        assert isinstance(result, dict)

    def test_weber_contrast_a_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "weber_contrast_a" in result

    def test_single_image_weber_contrast_b_empty(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        # Single-image mode: weber_contrast_b is present but empty (same pattern as contrast_ratios_b)
        wc_b = result.get("weber_contrast_b")
        assert wc_b is None or not wc_b

    def test_weber_contrast_a_positive(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        for v in result.get("weber_contrast_a", {}).values():
            assert v >= 0.0

    def test_with_roi(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a,
                                                  roi=(50, 50, 450, 450))
        assert isinstance(result, dict)
