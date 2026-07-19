"""Unit tests for analysis/snr_analyzer.py."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as ap_fits

from analysis.snr_analyzer import SNRAnalyzer
from core.astro_image import AstroImage

_REQUIRED_KEYS = {
    "snr_global", "noise_median", "background_median",
    "pct_above_3", "pct_above_5", "pct_above_10", "pct_above_20",
    "gain_e_per_adu", "sky_bg_electrons",
}


@pytest.fixture(scope="module")
def snr_result(astro_image_a) -> dict:
    """Cached default-params SNRAnalyzer result, shared by TestAnalyze and
    TestGainGuard -- both exercise the same astro_image_a input."""
    return SNRAnalyzer().analyze(astro_image_a)


class TestAnalyze:
    def test_returns_dict(self, snr_result):
        assert isinstance(snr_result, dict)

    def test_required_keys_present(self, snr_result):
        assert _REQUIRED_KEYS.issubset(snr_result.keys())

    def test_snr_global_nonnegative(self, snr_result):
        assert snr_result["snr_global"] >= 0.0

    def test_noise_median_positive(self, snr_result):
        assert snr_result["noise_median"] > 0.0

    def test_background_median_positive(self, snr_result):
        bg = snr_result["background_median"]
        assert bg is None or bg > 0.0

    def test_percentages_in_valid_range(self, snr_result):
        for key in ("pct_above_3", "pct_above_5", "pct_above_10", "pct_above_20"):
            assert 0.0 <= snr_result[key] <= 100.0

    def test_percentages_monotone_decreasing(self, snr_result):
        assert snr_result["pct_above_3"] >= snr_result["pct_above_5"]
        assert snr_result["pct_above_5"] >= snr_result["pct_above_10"]
        assert snr_result["pct_above_10"] >= snr_result["pct_above_20"]

    def test_figures_key_present(self, snr_result):
        assert "figures" in snr_result


class TestGainGuard:
    """FITS GAIN=0 must be rejected; EGAIN=1.0 takes priority."""

    def test_egain_accepted_over_zero_gain(self, snr_result):
        # Test FITS has GAIN=0 and EGAIN=1.0; EGAIN wins
        assert snr_result["gain_e_per_adu"] == pytest.approx(1.0)

    def test_gain_zero_not_accepted(self, snr_result):
        assert snr_result["gain_e_per_adu"] != 0.0

    def test_sky_electrons_present_when_gain_known(self, snr_result):
        assert snr_result["sky_bg_electrons"] is not None

    def test_sky_electrons_none_without_any_gain_keyword(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 500
        ap_fits.writeto(str(tmp_path / "nogain.fits"), data, overwrite=True)
        img = AstroImage(str(tmp_path / "nogain.fits"), label="NoGain")
        img.load()
        result = SNRAnalyzer().analyze(img)
        assert result["sky_bg_electrons"] is None
        assert result["gain_e_per_adu"] is None

    def test_negative_gain_rejected(self, tmp_path):
        data = np.ones((128, 128), dtype=np.float32) * 500
        hdr = ap_fits.Header()
        hdr["EGAIN"] = -1.0
        ap_fits.writeto(str(tmp_path / "neggain.fits"), data, hdr, overwrite=True)
        img = AstroImage(str(tmp_path / "neggain.fits"), label="NegGain")
        img.load()
        result = SNRAnalyzer().analyze(img)
        assert result["gain_e_per_adu"] is None
