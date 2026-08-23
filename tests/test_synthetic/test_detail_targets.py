"""Unit tests for synthetic/detail_targets.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synthetic.detail_targets import (
    _center_crop_square,
    _peak_rescale,
    apply_camera_noise,
    apply_gaussian_blur,
    build_composite_target,
    build_siemens_target,
    load_fractal_source,
    load_zebra,
)

FRACTAL_DIR = Path(__file__).resolve().parents[2] / "AstroLabTestData" / "Fractal Source"


class TestPeakRescale:
    def test_hits_target_bounds(self):
        gray = np.array([[0.0, 240.0], [120.0, 60.0]])
        out = _peak_rescale(gray, target_peak_adu=1000.0, pedestal_adu=300.0)
        assert float(out.min()) == pytest.approx(300.0, abs=1e-3)
        assert float(out.max()) == pytest.approx(1300.0, abs=1e-3)

    def test_constant_input_returns_pedestal(self):
        gray = np.full((4, 4), 77.0)
        out = _peak_rescale(gray, target_peak_adu=500.0, pedestal_adu=100.0)
        assert np.all(out == 100.0)


class TestCenterCropSquare:
    def test_raises_when_source_too_small(self):
        with pytest.raises(ValueError):
            _center_crop_square(np.zeros((10, 10)), 20)

    def test_crops_to_requested_size(self):
        arr = np.arange(100 * 100, dtype=np.float64).reshape(100, 100)
        out = _center_crop_square(arr, 40)
        assert out.shape == (40, 40)


class TestSiemensTarget:
    def test_corners_are_near_pedestal(self):
        arr = build_siemens_target(peak_adu=8000.0, canvas_px=128, pedestal_adu=300.0)
        corner = arr[:5, :5]
        assert float(corner.max()) < 300.0 + 50.0

    def test_peak_matches_requested_adu(self):
        arr = build_siemens_target(peak_adu=8000.0, canvas_px=128, pedestal_adu=300.0)
        assert float(arr.max()) == pytest.approx(300.0 + 8000.0, rel=1e-3)

    def test_shape_matches_canvas(self):
        arr = build_siemens_target(peak_adu=1000.0, canvas_px=200)
        assert arr.shape == (200, 200)


class TestCompositeTarget:
    @pytest.fixture(scope="class")
    @classmethod
    def fractal_base(cls):
        return load_fractal_source(FRACTAL_DIR / "BigImage_00018.png",
                                   target_peak_adu=3000.0, canvas_px=256)

    def test_deterministic_with_fixed_seed(self, fractal_base):
        a = build_composite_target(fractal_base, n_accents=5,
                                   accent_peak_adu=15000.0, accent_size_px=41, seed=7)
        b = build_composite_target(fractal_base, n_accents=5,
                                   accent_peak_adu=15000.0, accent_size_px=41, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_more_accents_cover_more_area(self, fractal_base):
        sparse = build_composite_target(fractal_base, n_accents=3,
                                        accent_peak_adu=20000.0, accent_size_px=41, seed=1)
        dense = build_composite_target(fractal_base, n_accents=8,
                                       accent_peak_adu=20000.0, accent_size_px=41, seed=2)
        threshold = float(fractal_base.max()) + 5000.0
        assert np.count_nonzero(dense > threshold) > np.count_nonzero(sparse > threshold)

    def test_base_is_not_mutated(self, fractal_base):
        before = fractal_base.copy()
        build_composite_target(fractal_base, n_accents=3, accent_peak_adu=10000.0, seed=1)
        np.testing.assert_array_equal(fractal_base, before)


class TestGaussianBlur:
    def test_reduces_local_variance(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(1000, 200, (128, 128)).astype(np.float32)
        blurred = apply_gaussian_blur(arr, 3.0)
        assert float(np.var(blurred)) < float(np.var(arr))

    def test_zero_sigma_is_a_noop(self):
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        out = apply_gaussian_blur(arr, 0.0)
        np.testing.assert_array_equal(out, arr)


class TestCameraNoise:
    def test_stack_depth_none_is_unchanged(self):
        arr = np.random.default_rng(0).normal(1000, 50, (32, 32)).astype(np.float32)
        out = apply_camera_noise(arr, np.random.default_rng(1), stack_depth=None)
        np.testing.assert_array_equal(out, arr.astype(np.float32))

    def test_std_scales_as_inverse_sqrt_stack_depth(self):
        arr = np.full((128, 128), 5000.0, dtype=np.float32)
        rng = np.random.default_rng(0)
        std_1 = float(np.std(apply_camera_noise(arr, rng, stack_depth=1)))
        std_25 = float(np.std(apply_camera_noise(arr, rng, stack_depth=25)))
        ratio = std_1 / std_25
        assert 3.5 < ratio < 7.0   # sqrt(25) = 5x, generous tolerance for one draw

    def test_rejects_zero_stack_depth(self):
        arr = np.full((8, 8), 100.0, dtype=np.float32)
        with pytest.raises(ValueError):
            apply_camera_noise(arr, np.random.default_rng(0), stack_depth=0)


class TestFractalAndZebraLoaders:
    @pytest.mark.skipif(not (FRACTAL_DIR / "BigImage_00018.png").exists(),
                        reason="AstroLabTestData/Fractal Source not present")
    def test_load_fractal_source_shape_and_range(self):
        arr = load_fractal_source(FRACTAL_DIR / "BigImage_00018.png",
                                  target_peak_adu=5000.0, pedestal_adu=200.0,
                                  canvas_px=256)
        assert arr.shape == (256, 256)
        assert float(arr.min()) >= 200.0 - 1e-3
        assert float(arr.max()) <= 5200.0 + 1e-3

    @pytest.mark.skipif(not (FRACTAL_DIR / "Zebra.png").exists(),
                        reason="AstroLabTestData/Fractal Source not present")
    def test_load_zebra_shape_and_range(self):
        arr = load_zebra(FRACTAL_DIR / "Zebra.png",
                         target_peak_adu=5000.0, pedestal_adu=200.0, canvas_px=256)
        assert arr.shape == (256, 256)
        assert float(arr.min()) >= 200.0 - 1e-3
        assert float(arr.max()) <= 5200.0 + 1e-3
