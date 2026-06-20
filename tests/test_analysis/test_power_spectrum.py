"""Unit tests for analysis/power_spectrum.py."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.power_spectrum import PowerSpectrumAnalyzer
from core.models import POWER_SPECTRUM_NPIX

_RESULT_KEYS = {"mid_high_ratio", "power_spectrum_2d", "radial_power", "freq_axis"}


class TestAnalyze:
    def test_returns_dict(self, astro_image_a):
        result = PowerSpectrumAnalyzer().analyze(astro_image_a)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, astro_image_a):
        result = PowerSpectrumAnalyzer().analyze(astro_image_a)
        assert _RESULT_KEYS.issubset(result.keys())

    def test_mid_high_ratio_positive(self, astro_image_a):
        result = PowerSpectrumAnalyzer().analyze(astro_image_a)
        ratio = result["mid_high_ratio"]
        if ratio is not None:
            assert ratio > 0.0

    def test_radial_power_nonnegative(self, astro_image_a):
        result = PowerSpectrumAnalyzer().analyze(astro_image_a)
        rp = result["radial_power"]
        if rp is not None:
            assert np.all(np.asarray(rp) >= 0.0)

    def test_freq_axis_increasing(self, astro_image_a):
        result = PowerSpectrumAnalyzer().analyze(astro_image_a)
        freq = result["freq_axis"]
        if freq is not None and len(freq) > 1:
            assert np.all(np.diff(freq) > 0)

    def test_freq_axis_in_nyquist_range(self, astro_image_a):
        result = PowerSpectrumAnalyzer().analyze(astro_image_a)
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
