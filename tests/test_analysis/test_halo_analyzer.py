"""Unit tests for analysis/halo_analyzer.py."""
from __future__ import annotations

import pytest

from analysis.halo_analyzer import HaloAnalyzer

_REQUIRED_KEYS = {
    "halo_radius_px", "halo_to_core_ratio", "n_stars_fitted",
    "radial_profile", "radial_radii",
}


class TestAnalyze:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, astro_image_a):
        return HaloAnalyzer().analyze(astro_image_a)

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_required_keys_present(self, result):
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_n_stars_fitted_nonnegative(self, result):
        assert result["n_stars_fitted"] >= 0

    def test_halo_radius_positive_or_none(self, result):
        r = result["halo_radius_px"]
        assert r is None or r > 0.0

    def test_halo_to_core_ratio_positive_or_none(self, result):
        ratio = result["halo_to_core_ratio"]
        assert ratio is None or ratio >= 0.0

    def test_radial_profile_is_sequence(self, result):
        rp = result["radial_profile"]
        # May be None when no valid stars found
        assert rp is None or hasattr(rp, "__len__")

    def test_radial_radii_matches_profile_length(self, result):
        if result["radial_profile"] is not None and result["radial_radii"] is not None:
            assert len(result["radial_radii"]) == len(result["radial_profile"])
