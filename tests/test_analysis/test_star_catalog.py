"""Unit tests for analysis/star_catalog.py."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from analysis.star_catalog import StarCatalogBuilder
from core.models import MIN_STAR_SNR


class TestBuild:
    def test_returns_table(self, astro_image_a):
        cat = StarCatalogBuilder().build(astro_image_a)
        assert isinstance(cat, Table)

    def test_finds_stars(self, astro_image_a):
        cat = StarCatalogBuilder().build(astro_image_a)
        assert len(cat) > 0

    def test_sets_catalog_attribute(self, astro_image_a):
        builder = StarCatalogBuilder()
        builder.build(astro_image_a)
        assert hasattr(astro_image_a, "catalog")

    def test_has_required_columns(self, astro_image_a):
        cat = StarCatalogBuilder().build(astro_image_a)
        for col in ("x_centroid", "y_centroid", "peak"):
            assert col in cat.colnames

    def test_peaks_positive(self, astro_image_a):
        cat = StarCatalogBuilder().build(astro_image_a)
        assert np.all(np.array(cat["peak"]) > 0)


class TestFilterPsfStars:
    def _empty_catalog(self) -> Table:
        return Table(
            names=["x_centroid", "y_centroid", "peak", "flux",
                   "sharpness", "roundness1", "roundness2"],
            dtype=[float] * 7,
        )

    def test_empty_catalog_passthrough(self, astro_image_a):
        empty = self._empty_catalog()
        result = StarCatalogBuilder().filter_psf_stars(empty, astro_image_a, 4.0)
        assert len(result) == 0

    def test_no_border_stars_in_result(self, astro_image_a):
        builder = StarCatalogBuilder()
        cat = builder.build(astro_image_a)
        if len(cat) == 0:
            pytest.skip("no stars detected")
        filtered = builder.filter_psf_stars(cat, astro_image_a, 4.0, border_px=50)
        h, w = astro_image_a.data.shape
        for row in filtered:
            assert float(row["x_centroid"]) >= 50
            assert float(row["x_centroid"]) <= w - 50
            assert float(row["y_centroid"]) >= 50
            assert float(row["y_centroid"]) <= h - 50

    def test_result_is_subset_of_input(self, astro_image_a):
        builder = StarCatalogBuilder()
        cat = builder.build(astro_image_a)
        filtered = builder.filter_psf_stars(cat, astro_image_a, 4.0)
        assert len(filtered) <= len(cat)

    def test_high_snr_isolated_stars_survive(self, astro_image_a):
        builder = StarCatalogBuilder()
        cat = builder.build(astro_image_a)
        if len(cat) == 0:
            pytest.skip("no stars detected")
        filtered = builder.filter_psf_stars(cat, astro_image_a, 4.0, min_snr=MIN_STAR_SNR)
        assert len(filtered) > 0

    def test_saturated_stars_removed(self, astro_image_a):
        builder = StarCatalogBuilder()
        cat = builder.build(astro_image_a)
        if len(cat) == 0:
            pytest.skip("no stars detected")
        sat_thresh = astro_image_a.saturation_threshold()
        filtered = builder.filter_psf_stars(cat, astro_image_a, 4.0)
        peaks = np.array(filtered["peak"])
        assert np.all(peaks < sat_thresh)
