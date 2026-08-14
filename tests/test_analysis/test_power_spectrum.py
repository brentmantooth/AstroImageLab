"""Unit tests for analysis/power_spectrum.py."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.power_spectrum import PowerSpectrumAnalyzer
from core.models import POWER_SPECTRUM_NPIX

_RESULT_KEYS = {"mid_high_ratio", "power_spectrum_2d", "radial_power", "freq_axis",
                "spectral_mtf_curve", "spectral_mtf50_cycles_per_px"}


class TestAnalyze:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, astro_image_a):
        return PowerSpectrumAnalyzer().analyze(astro_image_a)

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, result):
        assert _RESULT_KEYS.issubset(result.keys())

    def test_mid_high_ratio_positive(self, result):
        ratio = result["mid_high_ratio"]
        if ratio is not None:
            assert ratio > 0.0

    def test_radial_power_nonnegative(self, result):
        rp = result["radial_power"]
        if rp is not None:
            assert np.all(np.asarray(rp) >= 0.0)

    def test_freq_axis_increasing(self, result):
        freq = result["freq_axis"]
        if freq is not None and len(freq) > 1:
            assert np.all(np.diff(freq) > 0)

    def test_freq_axis_in_nyquist_range(self, result):
        freq = result["freq_axis"]
        if freq is not None:
            assert float(freq[0]) >= 0.0
            assert float(freq[-1]) <= 0.5 + 1e-6   # Nyquist = 0.5 cycles/px

    def test_with_large_roi(self, astro_image_a):
        roi = (50, 50, 450, 450)
        result = PowerSpectrumAnalyzer().analyze(astro_image_a, roi=roi)
        assert isinstance(result, dict)
        # ROI is 400x400, larger than POWER_SPECTRUM_NPIX -> should produce results
        assert result["mid_high_ratio"] is not None

    def test_tiny_roi_zoomed_produces_result(self, astro_image_a):
        # Tiny ROI is zoomed up to POWER_SPECTRUM_NPIX internally — always produces a result
        roi = (100, 100, 116, 116)   # 16x16 px, will be zoomed to 256x256
        result = PowerSpectrumAnalyzer().analyze(astro_image_a, roi=roi)
        assert isinstance(result, dict)
        # Result may or may not be None depending on the region content, but must not crash
        assert "mid_high_ratio" in result


class TestSpectralMtf:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, astro_image_a):
        return PowerSpectrumAnalyzer().analyze(astro_image_a)

    def test_curve_nonnegative(self, result):
        curve = result["spectral_mtf_curve"]
        if curve is not None:
            assert np.all(np.asarray(curve) >= 0.0)

    def test_curve_same_length_as_freq_axis(self, result):
        curve = result["spectral_mtf_curve"]
        freq = result["freq_axis"]
        if curve is not None and freq is not None:
            assert len(curve) == len(freq)

    def test_mtf50_in_nyquist_range_when_defined(self, result):
        mtf50 = result["spectral_mtf50_cycles_per_px"]
        if mtf50 is not None:
            assert 0.0 < mtf50 <= 0.5 + 1e-6

    def test_missing_input_produces_none_gracefully(self, astro_image_a):
        analyzer = PowerSpectrumAnalyzer()
        curve, mtf50 = analyzer._spectral_mtf(np.array([]), np.array([]))
        assert curve is None
        assert mtf50 is None

    def test_zero_reference_band_power_returns_none(self):
        analyzer = PowerSpectrumAnalyzer()
        freq = np.linspace(0.0, 0.5, 20)
        radial = np.where(freq <= 0.10, 0.0, 1.0)   # low band entirely zero
        curve, mtf50 = analyzer._spectral_mtf(freq, radial)
        assert curve is None
        assert mtf50 is None

    def test_early_return_skeleton_has_new_keys(self, astro_image_a, monkeypatch):
        # Force the _normalise early-return path (an all-zero/negative region) and
        # confirm the skeleton dict still carries both new spectral-MTF keys, set
        # to None, matching every other scalar/array key in that skeleton.
        analyzer = PowerSpectrumAnalyzer()
        monkeypatch.setattr(analyzer, "_normalise", lambda region: None)
        result = analyzer.analyze(astro_image_a)
        assert _RESULT_KEYS.issubset(result.keys())
        assert result["spectral_mtf_curve"] is None
        assert result["spectral_mtf50_cycles_per_px"] is None
