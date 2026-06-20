"""Unit tests for core/stretch.py — all functions are pure math, no fixtures needed."""
from __future__ import annotations

import numpy as np
import pytest

from core.stretch import (
    normalize_unit_interval,
    mtf,
    stf_stretch,
    stf_stretch_matched,
    normalize_for_display,
)


class TestNormalizeUnitInterval:
    def test_basic_range(self):
        arr = np.array([-10.0, 0.0, 5.0, 10.0])
        out = normalize_unit_interval(arr)
        assert float(out.min()) == pytest.approx(0.0)
        assert float(out.max()) == pytest.approx(1.0)

    def test_already_in_unit_interval(self):
        arr = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        out = normalize_unit_interval(arr)
        np.testing.assert_array_almost_equal(out, arr)

    def test_integer_dtype_scaled_by_max(self):
        arr = np.array([0, 32768, 65535], dtype=np.uint16)
        out = normalize_unit_interval(arr)
        assert float(out.min()) == pytest.approx(0.0)
        assert float(out.max()) == pytest.approx(1.0, abs=1e-4)

    def test_all_nan_returns_zeros(self):
        arr = np.array([np.nan, np.nan])
        out = normalize_unit_interval(arr)
        assert np.all(out == 0.0)

    def test_constant_array_returns_zeros(self):
        arr = np.full(10, 42.0)
        out = normalize_unit_interval(arr)
        assert np.all(out == 0.0)

    def test_output_dtype_float32(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = normalize_unit_interval(arr)
        assert out.dtype == np.float32

    def test_inf_ignored_in_min_max(self):
        arr = np.array([0.0, 1.0, np.inf, -np.inf])
        out = normalize_unit_interval(arr)
        # finite part [0,1] already in unit interval — should be returned unchanged
        assert np.isfinite(out[0]) and np.isfinite(out[1])


class TestMtf:
    @pytest.mark.parametrize("m", [0.1, 0.2, 0.3, 0.5])
    def test_midpoint_maps_to_half(self, m):
        result = float(mtf(np.array([m], dtype=np.float32), m)[0])
        assert result == pytest.approx(0.5, abs=1e-5)

    def test_zero_input_returns_zero(self):
        assert float(mtf(np.array([0.0]), 0.2)[0]) == pytest.approx(0.0)

    def test_one_input_returns_one(self):
        assert float(mtf(np.array([1.0]), 0.2)[0]) == pytest.approx(1.0)

    def test_output_clipped_to_unit(self):
        out = mtf(np.linspace(0, 1, 100, dtype=np.float32), 0.2)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_scalar_input(self):
        result = mtf(0.5, 0.5)
        assert float(result) == pytest.approx(0.5, abs=1e-5)

    def test_monotonically_increasing(self):
        x = np.linspace(0.0, 1.0, 50, dtype=np.float32)
        y = mtf(x, 0.2)
        diffs = np.diff(y)
        assert np.all(diffs >= 0)


class TestStfStretch:
    def test_output_in_unit_interval(self):
        sky = np.random.default_rng(1).normal(1000, 30, (64, 64)).astype(np.float64)
        out = stf_stretch(sky)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_sky_median_near_020(self):
        sky = np.random.default_rng(2).normal(1000, 30, (128, 128)).astype(np.float64)
        out = stf_stretch(sky)
        median = float(np.median(out))
        assert 0.12 < median < 0.30

    def test_all_nan_returns_zeros(self):
        arr = np.full((10, 10), np.nan)
        out = stf_stretch(arr)
        assert np.all(out == 0.0)

    def test_output_dtype_float32(self):
        arr = np.random.default_rng(3).normal(500, 10, (32, 32))
        out = stf_stretch(arr)
        assert out.dtype == np.float32

    def test_brighter_sky_shifts_median(self):
        dim  = np.random.default_rng(10).normal(100,  5, (64, 64))
        brt  = np.random.default_rng(10).normal(5000, 50, (64, 64))
        out_dim = stf_stretch(dim)
        out_brt = stf_stretch(brt)
        # Both should land near 0.2 (STF is scale-invariant), not systematically different
        assert abs(float(np.median(out_dim)) - 0.2) < 0.1
        assert abs(float(np.median(out_brt)) - 0.2) < 0.1


class TestStfStretchMatched:
    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(4)
        ref  = rng.normal(1000, 30, (64, 64)).astype(np.float64)
        data = rng.normal(800,  25, (64, 64)).astype(np.float64)
        out = stf_stretch_matched(data, ref)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_empty_ref_returns_zeros(self):
        ref  = np.full(5, np.nan)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = stf_stretch_matched(data, ref)
        assert np.all(out == 0.0)

    def test_identical_data_matches_ref(self):
        rng = np.random.default_rng(5)
        ref = rng.normal(1000, 30, (64, 64)).astype(np.float64)
        out_ref  = stf_stretch(ref)
        out_mat  = stf_stretch_matched(ref, ref)
        np.testing.assert_allclose(out_mat, out_ref, atol=0.01)


class TestNormalizeForDisplay:
    def test_output_uint8(self):
        arr = np.random.default_rng(5).normal(1000, 30, (64, 64))
        out = normalize_for_display(arr)
        assert out.dtype == np.uint8

    def test_output_range(self):
        arr = np.random.default_rng(6).normal(500, 20, (64, 64))
        out = normalize_for_display(arr)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_no_stretch_monotonic(self):
        arr = np.array([100.0, 200.0, 300.0, 400.0])
        out = normalize_for_display(arr, stretch=False)
        assert list(out) == sorted(out)

    def test_all_nan_returns_zeros(self):
        arr = np.full((5, 5), np.nan)
        out = normalize_for_display(arr)
        assert np.all(out == 0)

    def test_shape_preserved(self):
        arr = np.ones((32, 64))
        out = normalize_for_display(arr)
        assert out.shape == (32, 64)
