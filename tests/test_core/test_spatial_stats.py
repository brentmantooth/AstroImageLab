"""Tests for core.spatial_stats.

The load-bearing checks here are the two with a known answer:

* on iid noise the block bootstrap SE must equal sigma/sqrt(N) and must be flat
  in block size -- if it is not, the estimator is wrong rather than merely
  imprecise;
* on a field with an *injected* correlation length the recovered length must
  track it, and the SE must plateau once the block exceeds it.

Everything else in this module is judgement; those two are arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from core.spatial_stats import (
    BootstrapResult,
    auto_block_size,
    block_bootstrap_ci,
    decorrelation_length,
    effective_sample_size,
    se_ladder,
    tile_reduce,
)


def iid_field(shape=(512, 512), sigma=1.0, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, shape)


def correlated_field(shape=(512, 512), smooth_sigma=8.0, seed=0):
    """Gaussian-smoothed noise: correlation length is set by smooth_sigma."""
    raw = np.random.default_rng(seed).normal(0.0, 1.0, shape)
    return gaussian_filter(raw, smooth_sigma)


class TestDecorrelationLength:
    def test_iid_field_is_essentially_uncorrelated(self):
        tau = decorrelation_length(iid_field())
        assert 0.5 <= tau <= 3.0, tau

    @pytest.mark.parametrize("smooth", [4.0, 8.0, 16.0])
    def test_recovers_injected_correlation_length(self, smooth):
        tau = decorrelation_length(correlated_field(smooth_sigma=smooth))
        # tau is an integral length, not the kernel sigma; it should scale with
        # the kernel and sit within a small multiple of it.
        assert smooth <= tau <= 12.0 * smooth, (smooth, tau)

    def test_monotone_in_smoothing(self):
        taus = [decorrelation_length(correlated_field(smooth_sigma=s))
                for s in (2.0, 8.0, 24.0)]
        assert taus[0] < taus[1] < taus[2], taus

    def test_constant_field_returns_one(self):
        assert decorrelation_length(np.full((64, 64), 3.0)) == 1.0

    def test_all_nan_field_returns_one(self):
        assert decorrelation_length(np.full((64, 64), np.nan)) == 1.0

    def test_empty_or_1d_returns_one(self):
        assert decorrelation_length(np.empty((0, 0))) == 1.0
        assert decorrelation_length(np.arange(10)) == 1.0


class TestTileReduce:
    def test_weighted_tile_mean_reproduces_global_mean(self):
        """The invariant the bootstrap's point estimate relies on."""
        f = iid_field((480, 480), seed=3) + 0.25
        values, weights = tile_reduce(f, None, 48)
        assert np.isclose(np.sum(values * weights) / np.sum(weights), f.mean())

    def test_tile_count_matches_geometry(self):
        values, _ = tile_reduce(iid_field((480, 480)), None, 48)
        assert values.size == 10 * 10

    def test_partial_tiles_are_cropped_not_padded(self):
        # 500 // 48 == 10, remainder discarded rather than forming a short tile.
        values, _ = tile_reduce(iid_field((500, 500)), None, 48)
        assert values.size == 10 * 10

    def test_mask_restricts_to_selected_pixels(self):
        f = np.zeros((128, 128))
        f[:64, :] = 1.0
        mask = np.zeros((128, 128), dtype=bool)
        mask[:64, :] = True
        values, weights = tile_reduce(f, mask, 32)
        assert np.allclose(values, 1.0)
        assert np.sum(weights) == 64 * 128

    def test_sparse_tiles_are_dropped(self):
        mask = np.zeros((128, 128), dtype=bool)
        mask[0, 0] = True          # one pixel in one tile: far below the floor
        values, _ = tile_reduce(np.ones((128, 128)), mask, 32)
        assert values.size == 0

    def test_a_sparse_mask_still_yields_tiles(self):
        """Section 8l's local-maxima mask selects ~5-12% of the frame. A flat
        fraction-of-tile-area rule rejects every tile at that density, which
        would silently disable the bootstrap for the population that best
        tracks perceived sharpness."""
        rng = np.random.default_rng(0)
        mask = rng.random((512, 512)) < 0.06
        values, _ = tile_reduce(iid_field((512, 512), seed=1), mask, 64)
        assert values.size >= 50, values.size

    def test_sparse_mask_mean_is_still_the_masked_mean(self):
        rng = np.random.default_rng(1)
        f = iid_field((512, 512), seed=2) + 0.4
        mask = rng.random((512, 512)) < 0.06
        values, weights = tile_reduce(f, mask, 64)
        assert (np.sum(values * weights) / np.sum(weights)
                == pytest.approx(f[mask].mean(), rel=1e-6))

    def test_absolute_floor_still_rejects_tiny_tiles(self):
        mask = np.zeros((256, 256), dtype=bool)
        mask[::64, ::64] = True     # ~1 px per 64x64 tile
        values, _ = tile_reduce(np.ones((256, 256)), mask, 64)
        assert values.size == 0

    def test_block_larger_than_image_is_clamped(self):
        values, _ = tile_reduce(iid_field((64, 64)), None, 4096)
        assert values.size == 1

    def test_custom_reducer_is_used(self):
        f = np.zeros((64, 64))
        f[0, 0] = 1000.0           # one outlier; median ignores it, mean does not
        med, _ = tile_reduce(f, None, 64, fn=np.nanmedian)
        avg, _ = tile_reduce(f, None, 64)
        assert med[0] == 0.0 and avg[0] > 0.0


class TestBlockBootstrapOnIidData:
    """On independent data the block bootstrap must reduce to the textbook SE."""

    def test_se_matches_analytic_sigma_over_sqrt_n(self):
        sigma, shape = 2.0, (512, 512)
        res = block_bootstrap_ci(iid_field(shape, sigma, seed=1), block_px=32, seed=1)
        analytic = sigma / np.sqrt(shape[0] * shape[1])
        assert res.se == pytest.approx(analytic, rel=0.25), (res.se, analytic)

    def test_se_is_flat_in_block_size(self):
        f = iid_field((512, 512), seed=2)
        ses = [block_bootstrap_ci(f, block_px=b, seed=2).se for b in (16, 32, 64)]
        assert max(ses) / min(ses) < 1.6, ses

    def test_reports_converged(self):
        assert block_bootstrap_ci(iid_field(seed=4), block_px=32, seed=4).converged

    def test_naive_se_is_not_understated_on_iid_data(self):
        res = block_bootstrap_ci(iid_field((512, 512), seed=5), block_px=32, seed=5)
        assert 0.6 < res.se_understatement < 1.7, res.se_understatement

    def test_ci_covers_the_true_zero_mean(self):
        res = block_bootstrap_ci(iid_field(seed=6), block_px=32, seed=6)
        assert res.lo < 0.0 < res.hi
        assert not res.excludes_zero

    def test_ci_excludes_zero_for_a_real_offset(self):
        res = block_bootstrap_ci(iid_field(seed=7) + 0.5, block_px=32, seed=7)
        assert res.excludes_zero


class TestBlockBootstrapOnCorrelatedData:
    def test_naive_se_is_understated(self):
        """The module's whole reason for existing."""
        res = block_bootstrap_ci(correlated_field(smooth_sigma=8.0, seed=8),
                                 block_px=64, seed=8)
        assert res.se_understatement > 3.0, res.se_understatement

    def test_se_plateaus_once_block_exceeds_correlation_length(self):
        f = correlated_field(smooth_sigma=4.0, seed=9)
        se_at = {b: block_bootstrap_ci(f, block_px=b, seed=9).se for b in (32, 64, 128)}
        assert se_at[128] / se_at[64] < 1.6, se_at

    def test_se_grows_while_block_is_below_correlation_length(self):
        f = correlated_field(smooth_sigma=16.0, seed=10)
        small = block_bootstrap_ci(f, block_px=4, seed=10).se
        large = block_bootstrap_ci(f, block_px=64, seed=10).se
        assert large > 2.0 * small, (small, large)


class TestBootstrapResult:
    def test_point_estimate_equals_the_masked_mean(self):
        f = correlated_field(seed=11) + 0.3
        res = block_bootstrap_ci(f, block_px=64, seed=11)
        assert res.point == pytest.approx(f.mean(), rel=1e-6)

    def test_ci_brackets_the_point_estimate(self):
        res = block_bootstrap_ci(correlated_field(seed=12) + 0.3, block_px=64, seed=12)
        assert res.lo <= res.point <= res.hi

    def test_wider_ci_for_lower_confidence_is_narrower(self):
        f = correlated_field(seed=13)
        c95 = block_bootstrap_ci(f, block_px=64, ci=95.0, seed=13)
        c68 = block_bootstrap_ci(f, block_px=64, ci=68.0, seed=13)
        assert (c68.hi - c68.lo) < (c95.hi - c95.lo)

    def test_records_block_geometry(self):
        res = block_bootstrap_ci(correlated_field((512, 512), seed=14),
                                 block_px=64, seed=14)
        assert res.block_px == 64
        assert res.n_blocks == 8 * 8

    def test_se_understatement_none_when_naive_se_is_zero(self):
        r = BootstrapResult(0, 0, 0, 1.0, 8, 4, True, None, 1.0, naive_se=0.0)
        assert r.se_understatement is None

    def test_deterministic_for_a_fixed_seed(self):
        f = correlated_field(seed=15)
        a = block_bootstrap_ci(f, block_px=64, seed=99)
        b = block_bootstrap_ci(f, block_px=64, seed=99)
        assert (a.lo, a.hi, a.se) == (b.lo, b.hi, b.se)


class TestAutoBlockSize:
    def test_block_grows_with_correlation_length(self):
        small, _, _ = auto_block_size(correlated_field(smooth_sigma=2.0, seed=16))
        large, _, _ = auto_block_size(correlated_field(smooth_sigma=24.0, seed=16))
        assert large > small

    def test_min_blocks_bound_is_respected(self):
        f = correlated_field((512, 512), smooth_sigma=64.0, seed=17)
        block, _, _ = auto_block_size(f, min_blocks=30)
        assert (512 // block) ** 2 >= 30

    def test_block_never_exceeds_image(self):
        block, _, _ = auto_block_size(correlated_field((64, 64), smooth_sigma=64.0))
        assert 1 <= block <= 64

    def test_iid_field_converges_at_the_smallest_rung(self):
        block, converged, ladder = auto_block_size(iid_field((512, 512), seed=18))
        assert converged
        assert block == min(ladder)

    def test_ladder_is_geometric_and_bounded(self):
        ladder = se_ladder(correlated_field((512, 512), seed=19))
        rungs = sorted(ladder)
        assert rungs[0] == 4
        assert all((512 // b) ** 2 >= 1 for b in rungs)

    def test_ladder_se_is_flat_for_iid_and_rising_for_correlated(self):
        flat = se_ladder(iid_field((512, 512), seed=20))
        rising = se_ladder(correlated_field((512, 512), smooth_sigma=16.0, seed=20))
        span = lambda d: max(d.values()) / min(d.values())   # noqa: E731
        assert span(flat) < 1.6, flat
        assert span(rising) > 3.0, rising


class TestLongRangeDependence:
    """A field with structure at every scale has no valid block size.

    This is not a corner case -- it is what the project's real Section 8
    difference maps do. Their integral autocorrelation length is ~7 px, yet the
    bootstrap SE keeps growing out to 224 px blocks with the growth per doubling
    *accelerating*. A single block-vs-double-block check reports such a field as
    converged; walking the ladder is what catches it.
    """

    @staticmethod
    def _fractal_field(shape=(512, 512), seed=0):
        """Sum of octaves: correlated at every scale, like a real sky gradient
        difference stacked on top of small-scale detail differences."""
        rng = np.random.default_rng(seed)
        out = np.zeros(shape)
        for smooth in (1.0, 4.0, 16.0, 64.0):
            out += gaussian_filter(rng.normal(0.0, 1.0, shape), smooth) * smooth
        return out

    def test_reported_as_not_converged(self):
        _, converged, _ = auto_block_size(self._fractal_field(seed=21))
        assert not converged

    def test_se_keeps_growing_across_the_whole_ladder(self):
        ladder = se_ladder(self._fractal_field(seed=22))
        rungs = sorted(ladder)
        assert ladder[rungs[-1]] / ladder[rungs[0]] > 4.0, ladder

    def test_single_doubling_check_would_have_missed_it(self):
        """Guards the reason the ladder exists rather than a cheap local check."""
        ladder = se_ladder(self._fractal_field(seed=23))
        rungs = sorted(ladder)
        first_ratio = ladder[rungs[1]] / ladder[rungs[0]]
        overall = ladder[rungs[-1]] / ladder[rungs[0]]
        assert first_ratio < overall, (first_ratio, overall)

    def test_bootstrap_marks_the_result_not_converged(self):
        res = block_bootstrap_ci(self._fractal_field(seed=24), seed=24)
        assert res is not None and not res.converged


class TestGuards:
    def test_returns_none_when_too_few_tiles_survive(self):
        mask = np.zeros((128, 128), dtype=bool)
        mask[0, 0] = True
        assert block_bootstrap_ci(np.ones((128, 128)), mask, block_px=32) is None

    def test_returns_none_for_non_2d_input(self):
        assert block_bootstrap_ci(np.arange(100)) is None

    def test_returns_none_for_empty_input(self):
        assert block_bootstrap_ci(np.empty((0, 0))) is None

    def test_nan_pixels_are_excluded_not_propagated(self):
        f = np.zeros((256, 256))
        f[::7, ::7] = np.nan
        res = block_bootstrap_ci(f + 1.0, block_px=32, seed=19)
        assert res is not None and np.isfinite(res.point)
        assert res.point == pytest.approx(1.0)


class TestEffectiveSampleSize:
    def test_iid_field_is_worth_roughly_its_pixel_count(self):
        n_eff = effective_sample_size(iid_field((256, 256), seed=20), block_px=32)
        assert 0.4 * 256 * 256 < n_eff < 2.5 * 256 * 256, n_eff

    def test_correlated_field_is_worth_far_less_than_its_pixel_count(self):
        f = correlated_field((512, 512), smooth_sigma=16.0, seed=21)
        n_eff = effective_sample_size(f, block_px=64)
        assert n_eff < 0.01 * f.size, n_eff

    def test_returns_none_when_bootstrap_fails(self):
        assert effective_sample_size(np.arange(10)) is None
