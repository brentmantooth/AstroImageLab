"""Unit tests for analysis/background_mask.py — the Section 3e pixel-level
background/excluded classification (reproduces Background2D's own per-cell
sigma-clip, which photutils never exposes as a mask itself)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.stats import SigmaClip

from analysis.background_mask import classify_background_pixels


class TestBlobExclusion:
    def test_bright_blob_excluded_flat_noise_kept(self):
        rng = np.random.default_rng(0)
        data = rng.normal(100.0, 1.0, size=(256, 256)).astype(np.float32)
        # A clearly-outlier bright blob in one cell, well above 3 sigma.
        data[40:48, 40:48] += 500.0
        mask = classify_background_pixels(data, box_size=64)

        assert mask.shape == data.shape
        assert not mask[40:48, 40:48].any()
        # Far-away flat-noise region should be almost entirely kept.
        far_region = mask[200:256, 200:256]
        assert far_region.mean() > 0.95


class TestFlatNoiseKeptFraction:
    def test_kept_fraction_close_to_3sigma_expectation(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0.0, 1.0, size=(512, 512)).astype(np.float32)
        mask = classify_background_pixels(data, box_size=64)
        # ~99.7% expected for a symmetric 3-sigma clip on pure Gaussian noise;
        # iterative clipping can trim a bit further, so use a loose floor.
        assert mask.mean() > 0.98


class TestRaggedDimensions:
    @pytest.mark.parametrize("h,w,box_size", [(150, 200, 64), (65, 65, 64), (10, 200, 64)])
    def test_every_pixel_classified_no_gaps(self, h, w, box_size):
        rng = np.random.default_rng(2)
        data = rng.normal(50.0, 2.0, size=(h, w)).astype(np.float32)
        mask = classify_background_pixels(data, box_size=box_size)
        assert mask.shape == (h, w)
        assert mask.dtype == bool
        # No pixel should be left at the default-initialized False from an
        # uncovered gap in the interior+ragged tiling -- with pure Gaussian
        # noise essentially every cell should have a majority kept.
        assert mask.mean() > 0.9


class TestVectorizedMatchesNaiveLoop:
    def test_equivalence_on_small_array(self):
        rng = np.random.default_rng(3)
        box_size = 64
        h, w = 256, 256
        data = rng.normal(20.0, 3.0, size=(h, w)).astype(np.float32)
        data[10:15, 200:210] += 200.0   # one outlier blob to exercise real clipping

        vectorized = classify_background_pixels(data, box_size=box_size)

        clip = SigmaClip(sigma=3.0, maxiters=10)
        naive = np.zeros((h, w), dtype=bool)
        for r0 in range(0, h, box_size):
            r1 = min(r0 + box_size, h)
            for c0 in range(0, w, box_size):
                c1 = min(c0 + box_size, w)
                cell = data[r0:r1, c0:c1].ravel()
                clipped = clip(cell, masked=True)
                cell_mask = np.ma.getmaskarray(clipped)
                naive[r0:r1, c0:c1] = (~cell_mask).reshape(r1 - r0, c1 - c0)

        np.testing.assert_array_equal(vectorized, naive)
