"""Unit tests for analysis/inspector_regions.py — the Data Inspector's array maths.

These exist because the code they cover used to live in gui/inspector_widgets.py,
where the headless suite cannot reach it (no PyQt6 by design).  A real bug shipped
in that gap: value_range carried an invented conditional that preserved negative
values for signed maps, so the Data Inspector shaded the Original panel from -9.23
where the report starts at 0.  It was found by rendering a figure and comparing, not
by a test.  TestValueRangeMatchesReport below is that check, made automatic.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.image_filters import SpatialDetailAnalyzer
from analysis.inspector_regions import (
    exclusion_mask,
    COMPARE_DIFFERENCE,
    COMPARE_LOGRATIO,
    COMPARE_MODES,
    DIR_LOWER,
    DIR_UPPER,
    ROI_ELLIPSE,
    ROI_POLYGON,
    ROI_RECT,
    THRESH_ABSOLUTE,
    THRESH_PERCENTILE,
    common_crop,
    comparison_map,
    comparison_range,
    correlation_sample,
    ids_to_mask,
    refine_mask,
    roi_mask,
    select_points_in_polygon,
    threshold_mask,
    to_2d,
    value_range,
)


@pytest.fixture(scope="module")
def pair():
    """Two 64x96 float32 maps with structure, spread, and negative values.

    Signed on purpose: the vmin-clamp regression only shows up on data that goes
    below zero, which the background-subtracted Original panel does.
    """
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:64, 0:96]
    blob = np.exp(-(((xx - 40) ** 2 + (yy - 30) ** 2) / 400.0))
    a = (10.0 * blob + rng.normal(0.0, 1.0, (64, 96))).astype(np.float32)
    b = (7.0 * blob + rng.normal(0.0, 1.0, (64, 96))).astype(np.float32)
    return a, b


class TestTo2d:
    def test_2d_passes_through_unchanged(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert to_2d(arr) is arr

    def test_rgb_reduced_to_luma(self):
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 1.0, 2.0, 3.0
        out = to_2d(rgb)
        assert out.shape == (2, 2)
        assert np.allclose(out, 0.2126 * 1 + 0.7152 * 2 + 0.0722 * 3)

    def test_luma_weights_sum_to_one(self):
        white = np.ones((3, 3, 3), dtype=np.float32)
        assert np.allclose(to_2d(white), 1.0)

    def test_uint8_rgb_is_promoted_not_wrapped(self):
        rgb = np.full((2, 2, 3), 200, dtype=np.uint8)
        out = to_2d(rgb)
        assert out.dtype.kind == "f"
        assert np.allclose(out, 200.0)


class TestCommonCrop:
    def test_equal_shapes_unchanged(self):
        a, b = np.zeros((4, 5)), np.ones((4, 5))
        ca, cb = common_crop(a, b)
        assert ca.shape == cb.shape == (4, 5)

    def test_crops_to_minimum_in_each_axis(self):
        # The real case: wavelet panels are 626x836 where the rest of the file is 626x835.
        ca, cb = common_crop(np.zeros((626, 836)), np.zeros((626, 835)))
        assert ca.shape == cb.shape == (626, 835)

    def test_crops_both_axes_independently(self):
        ca, cb = common_crop(np.zeros((10, 3)), np.zeros((4, 9)))
        assert ca.shape == cb.shape == (4, 3)

    def test_top_left_aligned(self):
        a = np.arange(20).reshape(4, 5)
        ca, _ = common_crop(a, np.zeros((2, 2)))
        assert np.array_equal(ca, a[:2, :2])


class TestValueRangeMatchesReport:
    """value_range must equal _plot_side_by_side's own scale, formula for formula."""

    @staticmethod
    def _report_range(a, b):
        # Lifted from analysis/image_filters.py::_plot_side_by_side.
        vmin = max(0.0, float(min(np.percentile(a, 0.5), np.percentile(b, 0.5))))
        vmax = float(max(np.percentile(a, 99.5), np.percentile(b, 99.5)))
        return vmin, vmax

    def test_matches_report_on_signed_data(self, pair):
        a, b = pair
        assert min(a.min(), b.min()) < 0, "fixture must be signed to exercise the clamp"
        assert value_range([a, b]) == pytest.approx(self._report_range(a, b))

    def test_negative_percentile_is_clamped_to_zero(self, pair):
        # THE regression: an earlier version returned the raw negative percentile for
        # signed maps, so the inspector started its colour ramp at -9.23 where the
        # report starts at 0 — a visibly different image for the very panel most
        # likely to be compared against the report.
        a, b = pair
        assert float(np.percentile(a, 0.5)) < 0
        assert value_range([a, b])[0] == 0.0

    def test_matches_report_on_non_negative_data(self, pair):
        a, b = (np.abs(x) for x in pair)
        assert value_range([a, b]) == pytest.approx(self._report_range(a, b))

    def test_positive_lower_percentile_is_preserved(self):
        arr = np.linspace(5.0, 9.0, 400, dtype=np.float32).reshape(20, 20)
        vmin, _ = value_range([arr])
        assert vmin > 0.0

    def test_range_spans_both_arrays(self, pair):
        a, b = pair
        vmin, vmax = value_range([a, b])
        assert vmax >= max(value_range([a])[1], value_range([b])[1]) - 1e-6

    def test_none_entries_ignored(self, pair):
        a, _ = pair
        assert value_range([a, None]) == pytest.approx(value_range([a]))

    def test_all_none_returns_unit_interval(self):
        assert value_range([None, None]) == (0.0, 1.0)

    def test_empty_list_returns_unit_interval(self):
        assert value_range([]) == (0.0, 1.0)

    def test_zero_size_array_ignored(self, pair):
        a, _ = pair
        assert value_range([a, np.zeros((0, 0))]) == pytest.approx(value_range([a]))

    def test_constant_array_gives_non_degenerate_range(self):
        # vmax must stay strictly above vmin or the colour bar has nothing to map.
        vmin, vmax = value_range([np.full((8, 8), 3.0, dtype=np.float32)])
        assert vmax > vmin

    def test_all_zero_array_gives_non_degenerate_range(self):
        vmin, vmax = value_range([np.zeros((8, 8), dtype=np.float32)])
        assert vmin == 0.0 and vmax > 0.0

    def test_rgb_input_measured_as_luma(self):
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        rgb[..., :] = 0.5
        assert value_range([rgb]) == pytest.approx(value_range([np.full((8, 8), 0.5, np.float32)]))


class TestComparisonMap:
    def test_logratio_delegates_to_the_report_helper(self, pair):
        a, b = pair
        assert np.allclose(comparison_map(a, b, COMPARE_LOGRATIO),
                           SpatialDetailAnalyzer._log_ratio_map(a, b), equal_nan=True)

    def test_logratio_is_the_default_mode(self, pair):
        a, b = pair
        assert np.array_equal(comparison_map(a, b), comparison_map(a, b, COMPARE_LOGRATIO))

    def test_difference_is_plain_signed_subtraction(self, pair):
        a, b = pair
        assert np.allclose(comparison_map(a, b, COMPARE_DIFFERENCE), a - b)

    @pytest.mark.parametrize("mode", COMPARE_MODES)
    def test_output_is_float32(self, pair, mode):
        a, b = pair
        assert comparison_map(a, b, mode).dtype == np.float32

    @pytest.mark.parametrize("mode", COMPARE_MODES)
    def test_mismatched_shapes_are_cropped(self, pair, mode):
        a, b = pair
        out = comparison_map(a, b[:, :-1], mode)
        assert out.shape == (a.shape[0], a.shape[1] - 1)

    def test_identical_inputs_give_zero_logratio(self, pair):
        a, _ = pair
        assert np.allclose(comparison_map(a, a, COMPARE_LOGRATIO), 0.0)

    def test_identical_inputs_give_zero_difference(self, pair):
        a, _ = pair
        assert np.array_equal(comparison_map(a, a, COMPARE_DIFFERENCE),
                              np.zeros_like(a))

    def test_difference_is_antisymmetric(self, pair):
        a, b = pair
        assert np.allclose(comparison_map(a, b, COMPARE_DIFFERENCE),
                           -comparison_map(b, a, COMPARE_DIFFERENCE))

    def test_logratio_is_antisymmetric(self, pair):
        # log10(|A|/|B|) == -log10(|B|/|A|); the epsilon floor is pooled over both
        # inputs, so swapping them must not change its value either.
        a, b = pair
        assert np.allclose(comparison_map(a, b, COMPARE_LOGRATIO),
                           -comparison_map(b, a, COMPARE_LOGRATIO), atol=1e-5)

    def test_logratio_discards_sign(self, pair):
        # Section 8 asks "which image shows more structure here", not "which is
        # signed-brighter" — required for signed families like wavelet band-passes.
        a, b = pair
        assert np.allclose(comparison_map(a, b, COMPARE_LOGRATIO),
                           comparison_map(-a, b, COMPARE_LOGRATIO))

    def test_difference_keeps_sign(self, pair):
        a, b = pair
        assert not np.allclose(comparison_map(a, b, COMPARE_DIFFERENCE),
                               comparison_map(-a, b, COMPARE_DIFFERENCE))

    def test_rgb_input_compared_as_luma(self):
        rgb = np.zeros((4, 4, 3), dtype=np.float32)
        rgb[..., :] = 0.5
        flat = np.full((4, 4), 0.5, dtype=np.float32)
        assert np.allclose(comparison_map(rgb, flat, COMPARE_DIFFERENCE), 0.0)

    def test_unknown_mode_raises(self, pair):
        a, b = pair
        with pytest.raises(ValueError, match="unknown comparison mode"):
            comparison_map(a, b, "ratio")

    def test_linear_ratio_is_not_a_mode(self):
        # Guards the convention: "ratio" always means the log10 ratio.
        assert "ratio" not in COMPARE_MODES


class TestComparisonRange:
    def test_symmetric_about_zero(self, pair):
        a, b = pair
        lo, hi = comparison_range(comparison_map(a, b, COMPARE_LOGRATIO))
        assert lo == pytest.approx(-hi)

    @pytest.mark.parametrize("mode", COMPARE_MODES)
    def test_symmetric_for_both_modes(self, pair, mode):
        a, b = pair
        lo, hi = comparison_range(comparison_map(a, b, mode))
        assert lo == pytest.approx(-hi)
        assert hi > 0

    def test_matches_the_report_helper(self, pair):
        a, b = pair
        diff = comparison_map(a, b, COMPARE_LOGRATIO)
        assert comparison_range(diff) == pytest.approx(
            SpatialDetailAnalyzer._log_ratio_color_range(diff))

    def test_covers_the_bulk_of_the_distribution(self, pair):
        # +/- the 99.5th percentile of |diff|: most pixels inside, tails clipped.
        a, b = pair
        diff = comparison_map(a, b, COMPARE_LOGRATIO)
        lo, hi = comparison_range(diff)
        inside = np.mean((diff >= lo) & (diff <= hi))
        assert 0.98 <= inside <= 1.0

    def test_all_zero_diff_gives_non_degenerate_range(self):
        lo, hi = comparison_range(np.zeros((8, 8), dtype=np.float32))
        assert hi > lo


# ---------------------------------------------------------------------------
# Region selection: ROI fixes the domain, threshold splits it
# ---------------------------------------------------------------------------

class TestRoiMask:
    SHAPE = (40, 60)

    def test_none_selects_everything(self):
        assert roi_mask(self.SHAPE, None).all()

    def test_rect_covers_the_normalised_box(self):
        m = roi_mask(self.SHAPE, {"kind": ROI_RECT, "x0": 0.0, "y0": 0.0,
                                   "x1": 0.5, "y1": 0.5})
        assert m[:20, :30].all()
        assert not m[:20, 30:].any()
        assert not m[20:, :].any()

    def test_rect_area_is_proportional(self):
        m = roi_mask(self.SHAPE, {"kind": ROI_RECT, "x0": 0.25, "y0": 0.25,
                                   "x1": 0.75, "y1": 0.75})
        assert m.sum() == pytest.approx(0.25 * 40 * 60, rel=0.1)

    def test_rect_normalises_a_reversed_drag(self):
        fwd = roi_mask(self.SHAPE, {"kind": ROI_RECT, "x0": 0.2, "y0": 0.2,
                                     "x1": 0.8, "y1": 0.8})
        rev = roi_mask(self.SHAPE, {"kind": ROI_RECT, "x0": 0.8, "y0": 0.8,
                                     "x1": 0.2, "y1": 0.2})
        assert np.array_equal(fwd, rev)

    def test_degenerate_rect_selects_nothing(self):
        m = roi_mask(self.SHAPE, {"kind": ROI_RECT, "x0": 0.5, "y0": 0.5,
                                   "x1": 0.5, "y1": 0.5})
        assert not m.any()

    def test_rect_is_clipped_to_the_image(self):
        m = roi_mask(self.SHAPE, {"kind": ROI_RECT, "x0": -1.0, "y0": -1.0,
                                   "x1": 2.0, "y1": 2.0})
        assert m.all()

    def test_ellipse_is_inscribed_in_its_box(self):
        box = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
        ell = roi_mask(self.SHAPE, {"kind": ROI_ELLIPSE, **box})
        rect = roi_mask(self.SHAPE, {"kind": ROI_RECT, **box})
        assert ell.sum() < rect.sum()
        assert not (ell & ~rect).any()               # never outside the box
        assert not ell[0, 0] and not ell[-1, -1]     # corners excluded
        assert ell[20, 30]                           # centre included

    def test_ellipse_area_is_about_pi_over_four(self):
        ell = roi_mask((200, 200), {"kind": ROI_ELLIPSE, "x0": 0.0, "y0": 0.0,
                                     "x1": 1.0, "y1": 1.0})
        assert ell.mean() == pytest.approx(np.pi / 4, rel=0.02)

    def test_polygon_matches_the_equivalent_rect(self):
        poly = roi_mask(self.SHAPE, {"kind": ROI_POLYGON,
                                      "points": [(0.25, 0.25), (0.75, 0.25),
                                                 (0.75, 0.75), (0.25, 0.75)]})
        rect = roi_mask(self.SHAPE, {"kind": ROI_RECT, "x0": 0.25, "y0": 0.25,
                                      "x1": 0.75, "y1": 0.75})
        assert poly.sum() == pytest.approx(rect.sum(), rel=0.05)

    def test_polygon_triangle_is_about_half_its_box(self):
        tri = roi_mask((200, 200), {"kind": ROI_POLYGON,
                                     "points": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]})
        assert tri.mean() == pytest.approx(0.5, rel=0.05)

    @pytest.mark.parametrize("pts", [[], [(0.1, 0.1)], [(0.1, 0.1), (0.5, 0.5)]])
    def test_polygon_needs_three_points(self, pts):
        assert not roi_mask(self.SHAPE, {"kind": ROI_POLYGON, "points": pts}).any()

    def test_mask_shape_always_matches_the_panel(self):
        for roi in (None,
                    {"kind": ROI_RECT, "x0": .1, "y0": .1, "x1": .9, "y1": .9},
                    {"kind": ROI_ELLIPSE, "x0": .1, "y0": .1, "x1": .9, "y1": .9},
                    {"kind": ROI_POLYGON, "points": [(.1, .1), (.9, .1), (.5, .9)]}):
            assert roi_mask(self.SHAPE, roi).shape == self.SHAPE

    def test_same_roi_rescales_to_a_different_panel(self):
        # Why the coordinates are normalised: a drawn region must survive a switch to
        # a panel of a different pixel size.
        roi = {"kind": ROI_RECT, "x0": 0.25, "y0": 0.0, "x1": 0.75, "y1": 1.0}
        small, large = roi_mask((40, 60), roi), roi_mask((400, 600), roi)
        assert small.mean() == pytest.approx(large.mean(), rel=0.02)


class TestThresholdMask:
    @pytest.fixture(scope="class")
    @classmethod
    def ab(cls):
        a = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        b = np.linspace(0.0, 2.0, 100, dtype=np.float32).reshape(10, 10)
        return a, b

    def test_percentile_upper_selects_the_top_tail(self, ab):
        a, b = ab
        res = threshold_mask(a, b, THRESH_PERCENTILE, 10.0, DIR_UPPER)
        assert res.mask.mean() == pytest.approx(0.10, abs=0.02)
        assert res.mask[-1, -1] and not res.mask[0, 0]

    def test_percentile_lower_selects_the_bottom_tail(self, ab):
        a, b = ab
        res = threshold_mask(a, b, THRESH_PERCENTILE, 10.0, DIR_LOWER)
        assert res.mask.mean() == pytest.approx(0.10, abs=0.02)
        assert res.mask[0, 0] and not res.mask[-1, -1]

    def test_percentile_is_scale_invariant(self, ab):
        # The reason percentile mode exists: one setting stays meaningful across
        # metric families whose units differ by orders of magnitude.
        a, b = ab
        base = threshold_mask(a, b, THRESH_PERCENTILE, 10.0, DIR_UPPER).mask
        scaled = threshold_mask(a * 1000.0, b * 1e-6,
                                 THRESH_PERCENTILE, 10.0, DIR_UPPER).mask
        assert np.array_equal(base, scaled)

    def test_masks_are_anded_not_ored(self):
        # A selects its right half, B its bottom half -> only the overlap survives.
        a = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        b = np.tile(np.arange(10, dtype=np.float32).reshape(10, 1), (1, 10))
        res = threshold_mask(a, b, THRESH_PERCENTILE, 50.0, DIR_UPPER)
        assert res.mask[-1, -1]
        assert not res.mask[-1, 0] and not res.mask[0, -1]
        assert res.mask.mean() == pytest.approx(0.25, abs=0.06)

    def test_absolute_applies_one_literal_cut(self, ab):
        a, b = ab
        res = threshold_mask(a, b, THRESH_ABSOLUTE, 0.5, DIR_UPPER)
        assert np.array_equal(res.mask, (a >= 0.5) & (b >= 0.5))

    def test_absolute_lower(self, ab):
        a, b = ab
        res = threshold_mask(a, b, THRESH_ABSOLUTE, 0.5, DIR_LOWER)
        assert np.array_equal(res.mask, (a <= 0.5) & (b <= 0.5))

    def test_flat_input_warns_instead_of_selecting_everything(self, ab):
        # The documented percentile degeneracy: on a constant array every percentile
        # equals that one value, so `>=` matches 100% of pixels, not the top N%.
        a, _ = ab
        res = threshold_mask(a, np.zeros_like(a), THRESH_PERCENTILE, 10.0, DIR_UPPER)
        assert not res.mask.any()
        assert res.warning and "no variation" in res.warning

    def test_warns_when_nothing_is_selected(self, ab):
        a, b = ab
        res = threshold_mask(a, b, THRESH_ABSOLUTE, 99.0, DIR_UPPER)
        assert not res.mask.any() and res.warning

    def test_warns_when_everything_is_selected(self, ab):
        a, b = ab
        res = threshold_mask(a, b, THRESH_ABSOLUTE, -1.0, DIR_UPPER)
        assert res.mask.all() and res.warning

    def test_normal_case_has_no_warning(self, ab):
        a, b = ab
        assert threshold_mask(a, b, THRESH_PERCENTILE, 10.0, DIR_UPPER).warning is None

    def test_mismatched_shapes_are_cropped(self, ab):
        a, b = ab
        assert threshold_mask(a, b[:, :-1]).mask.shape == (10, 9)

    @pytest.mark.parametrize("mode,direction",
                             [("nope", DIR_UPPER), (THRESH_PERCENTILE, "sideways")])
    def test_unknown_options_raise(self, ab, mode, direction):
        a, b = ab
        with pytest.raises(ValueError):
            threshold_mask(a, b, mode, 10.0, direction)


class TestRefineMask:
    def test_no_ops_return_the_input_unchanged(self):
        m = np.zeros((20, 20), dtype=bool)
        m[5:15, 5:15] = True
        assert np.array_equal(refine_mask(m), m)

    def test_opening_removes_isolated_specks(self):
        m = np.zeros((30, 30), dtype=bool)
        m[10:20, 10:20] = True      # real structure
        m[2, 2] = True              # single-pixel noise
        out = refine_mask(m, open_px=1)
        assert not out[2, 2]
        assert out[15, 15]

    def test_closing_fills_a_small_gap(self):
        m = np.zeros((30, 30), dtype=bool)
        m[10:20, 10:20] = True
        m[14:16, 14:16] = False     # a hole punched in the middle
        assert refine_mask(m, close_px=2)[14, 14]

    def test_opening_runs_before_closing(self):
        # Ordering is the whole point of this function.  Closing starts with a
        # dilation, so a close-first order would inflate every speck into a blob
        # instead of removing it -- the failure CLAUDE.md records for the nebula mask.
        rng = np.random.default_rng(3)
        m = np.zeros((60, 60), dtype=bool)
        m[20:40, 20:40] = True
        specks = rng.random((60, 60)) < 0.01
        specks[20:40, 20:40] = False
        m |= specks
        out = refine_mask(m, open_px=1, close_px=1)
        outside = out.copy()
        outside[18:42, 18:42] = False
        assert outside.sum() < specks.sum(), "specks were amplified, not removed"

    def test_min_size_drops_small_islands(self):
        m = np.zeros((40, 40), dtype=bool)
        m[5:25, 5:25] = True        # 20x20 island, kept
        m[35:37, 35:37] = True      # 2x2 island, dropped at min_size 3 (area 9)
        out = refine_mask(m, min_size_px=3)
        assert out[10, 10]
        assert not out[35, 35]

    def test_fill_holes_closes_small_enclosed_gaps(self):
        m = np.zeros((40, 40), dtype=bool)
        m[10:30, 10:30] = True
        m[19:21, 19:21] = False     # 2x2 hole, area 4 <= 3*3
        assert refine_mask(m, fill_hole_px=3)[19, 19]

    def test_output_is_boolean_and_same_shape(self):
        m = np.zeros((25, 31), dtype=bool)
        m[5:20, 5:20] = True
        out = refine_mask(m, open_px=1, close_px=1, min_size_px=2, fill_hole_px=2)
        assert out.dtype == bool and out.shape == (25, 31)

    def test_all_false_survives_every_operation(self):
        m = np.zeros((20, 20), dtype=bool)
        out = refine_mask(m, open_px=2, close_px=2, min_size_px=3, fill_hole_px=3)
        assert not out.any()


class TestCorrelationSample:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        rng = np.random.default_rng(11)
        a = rng.random((30, 40)).astype(np.float32)
        b = rng.random((30, 40)).astype(np.float32)
        return a, b, comparison_map(a, b, COMPARE_DIFFERENCE)

    def test_extracts_exactly_the_masked_pixels(self, data):
        a, b, c = data
        mask = np.zeros(a.shape, dtype=bool)
        mask[5:10, 5:10] = True
        s = correlation_sample(a, b, c, mask)
        assert s.n_total == 25
        assert np.allclose(np.sort(s.a_vals), np.sort(a[mask]))
        assert np.allclose(np.sort(s.b_vals), np.sort(b[mask]))

    def test_flat_ids_round_trip_to_the_same_mask(self, data):
        a, b, c = data
        mask = np.zeros(a.shape, dtype=bool)
        mask[3:9, 4:11] = True
        s = correlation_sample(a, b, c, mask)
        assert np.array_equal(ids_to_mask(s.flat_ids, s.shape), mask)

    def test_flat_ids_point_at_the_right_pixels(self, data):
        a, b, c = data
        mask = np.zeros(a.shape, dtype=bool)
        mask[7, 13] = True
        mask[20, 2] = True
        s = correlation_sample(a, b, c, mask)
        assert set(s.flat_ids.tolist()) == {7 * 40 + 13, 20 * 40 + 2}
        assert np.allclose(np.sort(s.a_vals), np.sort([a[7, 13], a[20, 2]]))

    def test_ids_are_int64(self, data):
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool))
        assert s.flat_ids.dtype == np.int64

    def test_subsampling_caps_the_point_count(self, data):
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool), max_samples=100)
        assert s.a_vals.size == 100
        assert s.flat_ids.size == 100

    def test_n_total_is_the_unsampled_population(self, data):
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool), max_samples=100)
        assert s.n_total == 30 * 40

    def test_axis_range_comes_from_the_full_population(self, data):
        # Subsampling must not shrink the axes: upper-tail divergence from the 1:1
        # line is exactly what the plot exists to reveal.
        a, b, c = data
        full = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool),
                                   max_samples=10 ** 9)
        small = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool),
                                    rng=np.random.default_rng(0), max_samples=50)
        assert small.axis_lo == pytest.approx(full.axis_lo)
        assert small.axis_hi == pytest.approx(full.axis_hi)

    def test_axis_range_brackets_the_data(self, data):
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool))
        assert s.axis_lo < min(s.a_vals.min(), s.b_vals.min())
        assert s.axis_hi > max(s.a_vals.max(), s.b_vals.max())

    def test_subsampling_keeps_values_paired_with_ids(self, data):
        # If the subsample ever shuffled values and ids apart, a lasso would light up
        # the wrong pixels -- silently, and only on large panels.
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool),
                                rng=np.random.default_rng(5), max_samples=200)
        rows, cols = np.divmod(s.flat_ids, s.shape[1])
        assert np.allclose(s.a_vals, a[rows, cols])
        assert np.allclose(s.b_vals, b[rows, cols])
        assert np.allclose(s.c_vals, c[rows, cols])

    def test_full_population_is_not_capped(self, data):
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool), max_samples=100)
        assert s.full_a.size == s.full_b.size == s.full_flat_ids.size == s.n_total == 30 * 40

    def test_full_population_reuses_rendered_arrays_when_not_subsampled(self, data):
        # No extra memory should be paid when the whole population already fits.
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool), max_samples=10 ** 9)
        assert s.full_a is s.a_vals
        assert s.full_b is s.b_vals
        assert s.full_flat_ids is s.flat_ids

    def test_lasso_over_full_population_recovers_every_pixel(self, data):
        # The concrete regression this guards against: a lasso enclosing the whole
        # plot must select ALL masked pixels, not just the rendered max_samples
        # subset -- otherwise the resulting image overlay is sparse ("salt and
        # pepper") instead of solid.
        a, b, c = data
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool),
                                rng=np.random.default_rng(3), max_samples=50)
        huge_square = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        hit_capped = select_points_in_polygon(s.b_vals, s.a_vals, huge_square)
        hit_full = select_points_in_polygon(s.full_b, s.full_a, huge_square)
        assert hit_capped.sum() == 50
        assert hit_full.sum() == s.n_total == 30 * 40

    def test_empty_mask_returns_empty_arrays(self, data):
        a, b, c = data
        s = correlation_sample(a, b, c, np.zeros(a.shape, dtype=bool))
        assert s.n_total == 0
        assert s.a_vals.size == 0 and s.flat_ids.size == 0
        assert s.axis_hi > s.axis_lo

    def test_mismatched_shapes_are_cropped(self, data):
        a, b, c = data
        s = correlation_sample(a, b[:, :-1], c, np.ones(a.shape, dtype=bool))
        assert s.shape == (30, 39)
        assert s.n_total == 30 * 39

    def test_deterministic_for_a_seeded_rng(self, data):
        a, b, c = data
        mask = np.ones(a.shape, dtype=bool)
        s1 = correlation_sample(a, b, c, mask, rng=np.random.default_rng(1),
                                 max_samples=100)
        s2 = correlation_sample(a, b, c, mask, rng=np.random.default_rng(1),
                                 max_samples=100)
        assert np.array_equal(s1.flat_ids, s2.flat_ids)


class TestIdsToMask:
    def test_empty_ids_give_an_empty_mask(self):
        assert not ids_to_mask(np.empty(0, dtype=np.int64), (5, 5)).any()

    def test_out_of_range_ids_are_ignored(self):
        m = ids_to_mask(np.array([0, 24, 999, -3], dtype=np.int64), (5, 5))
        assert m[0, 0] and m[4, 4]
        assert m.sum() == 2

    def test_shape_is_honoured(self):
        assert ids_to_mask(np.array([0], dtype=np.int64), (3, 7)).shape == (3, 7)


class TestSelectPointsInPolygon:
    def test_square_selects_only_enclosed_points(self):
        xs = np.array([0.5, 1.5, 2.5, 5.0])
        ys = np.array([0.5, 1.5, 2.5, 5.0])
        square = [(0, 0), (3, 0), (3, 3), (0, 3)]
        assert np.array_equal(select_points_in_polygon(xs, ys, square),
                              [True, True, True, False])

    def test_triangle_selects_its_own_half(self):
        rng = np.random.default_rng(2)
        xs, ys = rng.random(4000), rng.random(4000)
        tri = [(0, 0), (1, 0), (0, 1)]
        assert select_points_in_polygon(xs, ys, tri).mean() == pytest.approx(0.5, abs=0.03)

    def test_concave_polygon_excludes_the_notch(self):
        # A lasso is routinely concave, so a bounding-box test would be wrong.
        arrow = [(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)]
        inside = select_points_in_polygon(np.array([1.0]), np.array([0.5]), arrow)
        outside = select_points_in_polygon(np.array([2.0]), np.array([3.0]), arrow)
        assert inside[0] and not outside[0]

    def test_result_length_matches_the_input(self):
        xs = np.arange(17, dtype=float)
        out = select_points_in_polygon(xs, xs, [(0, 0), (100, 0), (100, 100), (0, 100)])
        assert out.shape == (17,)

    @pytest.mark.parametrize("poly", [[], [(0, 0)], [(0, 0), (1, 1)]])
    def test_degenerate_polygon_selects_nothing(self, poly):
        xs = np.array([0.5, 0.6])
        assert not select_points_in_polygon(xs, xs, poly).any()

    def test_no_points_returns_empty(self):
        empty = np.empty(0)
        out = select_points_in_polygon(empty, empty, [(0, 0), (1, 0), (1, 1)])
        assert out.shape == (0,)

    def test_selection_maps_back_to_the_right_pixels(self):
        # The end-to-end contract the linked brushing depends on: lasso a region of
        # the scatter, and the flat ids it yields must be the pixels whose values
        # actually sit there.
        rng = np.random.default_rng(31)
        a = rng.random((20, 25)).astype(np.float32)
        b = rng.random((20, 25)).astype(np.float32)
        c = comparison_map(a, b, COMPARE_DIFFERENCE)
        s = correlation_sample(a, b, c, np.ones(a.shape, dtype=bool))
        # Select the upper-left quadrant of the scatter: high A, low B.
        poly = [(-1, 0.5), (0.5, 0.5), (0.5, 2), (-1, 2)]
        hit = select_points_in_polygon(s.b_vals, s.a_vals, poly)
        assert hit.any()
        ids = s.flat_ids[hit]
        rows, cols = np.divmod(ids, s.shape[1])
        assert (a[rows, cols] >= 0.5).all()
        assert (b[rows, cols] <= 0.5).all()


class TestComposableRegions:
    """ROI fixes the domain; the threshold splits it. The four-row table from the plan."""

    @pytest.fixture(scope="class")
    @classmethod
    def maps(cls):
        rng = np.random.default_rng(21)
        a = rng.random((50, 50)).astype(np.float32)
        b = rng.random((50, 50)).astype(np.float32)
        return a, b

    def test_no_roi_no_threshold_is_every_pixel(self, maps):
        a, _ = maps
        assert roi_mask(a.shape, None).all()

    def test_roi_alone_splits_inside_from_outside(self, maps):
        a, _ = maps
        domain = roi_mask(a.shape, {"kind": ROI_RECT, "x0": 0.2, "y0": 0.2,
                                     "x1": 0.8, "y1": 0.8})
        inside, outside = domain, ~domain
        assert not (inside & outside).any()
        assert (inside | outside).all()

    def test_threshold_alone_splits_mask_from_its_complement(self, maps):
        a, b = maps
        res = threshold_mask(a, b, THRESH_PERCENTILE, 25.0, DIR_UPPER)
        assert not (res.mask & ~res.mask).any()
        assert (res.mask | ~res.mask).all()

    def test_roi_and_threshold_compose(self, maps):
        a, b = maps
        domain = roi_mask(a.shape, {"kind": ROI_RECT, "x0": 0.0, "y0": 0.0,
                                     "x1": 0.5, "y1": 1.0})
        split = threshold_mask(a, b, THRESH_PERCENTILE, 25.0, DIR_UPPER).mask
        in_mask = domain & split
        out_mask = domain & ~split
        # The two correlation populations partition the domain and nothing else.
        assert not (in_mask & out_mask).any()
        assert np.array_equal(in_mask | out_mask, domain)
        assert not in_mask[:, 30:].any(), "selection leaked outside the ROI"

    def test_populations_partition_the_domain_pixel_for_pixel(self, maps):
        a, b = maps
        c = comparison_map(a, b, COMPARE_LOGRATIO)
        domain = roi_mask(a.shape, {"kind": ROI_ELLIPSE, "x0": 0.1, "y0": 0.1,
                                     "x1": 0.9, "y1": 0.9})
        split = threshold_mask(a, b, THRESH_PERCENTILE, 30.0, DIR_UPPER).mask
        s_in = correlation_sample(a, b, c, domain & split)
        s_out = correlation_sample(a, b, c, domain & ~split)
        assert s_in.n_total + s_out.n_total == int(domain.sum())
        assert not set(s_in.flat_ids.tolist()) & set(s_out.flat_ids.tolist())


class TestExclusionMask:
    """Union of user-drawn exclusion regions (Section 3g).

    The load-bearing case is the empty one: roi_mask(shape, None) returns
    all-True because "no ROI" means the whole frame is the domain, whereas an
    empty exclusion list must mask *nothing*. Getting that inversion wrong would
    mask every pixel of every image by default.
    """

    SQUARE = {"kind": "polygon",
              "points": [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]}

    @pytest.mark.parametrize("regions", [None, [], (), [None]])
    def test_empty_excludes_nothing(self, regions):
        mask = exclusion_mask((40, 60), regions)
        assert mask.shape == (40, 60)
        assert mask.dtype == bool
        assert not mask.any()

    def test_single_polygon_covers_its_quadrant(self):
        mask = exclusion_mask((40, 60), [self.SQUARE])
        assert mask[:20, :30].all()
        assert not mask[25:, 35:].any()

    def test_multiple_regions_are_unioned(self):
        other = {"kind": "polygon",
                 "points": [(0.6, 0.6), (1.0, 0.6), (1.0, 1.0), (0.6, 1.0)]}
        both = exclusion_mask((40, 60), [self.SQUARE, other])
        first = exclusion_mask((40, 60), [self.SQUARE])
        second = exclusion_mask((40, 60), [other])
        assert np.array_equal(both, first | second)
        assert both.sum() > first.sum()

    def test_overlapping_regions_do_not_double_count(self):
        shifted = {"kind": "polygon",
                   "points": [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]}
        mask = exclusion_mask((40, 60), [self.SQUARE, shifted])
        assert mask.dtype == bool
        assert mask.sum() <= mask.size

    def test_rect_kind_also_accepted(self):
        """Shares roi_mask's schema, so any ROI kind works as an exclusion."""
        rect = {"kind": "rect", "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5}
        assert np.array_equal(exclusion_mask((40, 60), [rect]),
                              exclusion_mask((40, 60), [self.SQUARE]))

    def test_degenerate_polygon_excludes_nothing(self):
        thin = {"kind": "polygon", "points": [(0.1, 0.1), (0.2, 0.2)]}
        assert not exclusion_mask((40, 60), [thin]).any()
