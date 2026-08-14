"""Unit tests for analysis/source_mask.py — the Section 3g source-masked
background estimate.

The assertions here are deliberately *quantitative*: every frame comes from
tests/bg_frames.py with its exact true sky surface attached, so a recovered
background is checked against a known answer rather than against shape or
monotonicity properties. A background model can be smooth, plausible and badly
biased simultaneously — only a known truth separates those cases (CLAUDE.md's
edge-analyzer lesson).

Baselines quoted in the tolerances are the measured unmasked Background2D bias
on the same frames: ~0.01 sigma (star-only), ~0.08 (moderate), ~1.28 (heavy),
~1.48 (gradient+heavy).
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.stats import SigmaClip
from photutils.background import (Background2D, MADStdBackgroundRMS,
                                  SExtractorBackground)

from analysis.source_mask import (MaskedBackgroundResult, SourceMaskResult,
                                  _order_cap_for, _smoothed_sigma,
                                  build_source_mask, fit_sky_scaffold,
                                  masked_background_estimate,
                                  plane_quadric_agreement,
                                  source_masked_background)
from core.models import (SOURCEMASK_MAX_EXTRAPOLATION_RATIO,
                         SOURCEMASK_MAX_MODEL_DIVERGENCE_SIGMA,
                         SOURCEMASK_MIN_CELLS_CONSTANT, SOURCEMASK_MIN_CELLS_PLANE,
                         SOURCEMASK_MIN_CELLS_QUADRIC,
                         SOURCEMASK_MIN_SURFACE_BRIGHTNESS_SIGMA)
from tests.bg_frames import (bounded_nebula_frame, gradient_nebula_frame,
                             heavy_nebula_frame, make_bg_frame,
                             moderate_nebula_frame, star_only_frame,
                             vignetted_frame, vignetted_nebula_frame)


def _unmasked_background(data):
    """The current, unmasked estimate — the baseline every result is judged against."""
    return Background2D(
        data, box_size=64, filter_size=3,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
        bkg_estimator=SExtractorBackground(),
        bkg_rms_estimator=MADStdBackgroundRMS())


def _bias_sigma(surface, true_sky, sky_sigma):
    """Median (recovered - true) background offset, in sky-sigma units."""
    return float(np.median(np.asarray(surface, dtype=np.float64) - true_sky)) / sky_sigma


def _run(frame):
    """(result, unmasked_bias_sigma, masked_bias_sigma) for a bg_frames frame."""
    data, true_sky, truth = frame
    base = _unmasked_background(data)
    result = source_masked_background(
        data, box_size=64, fwhm_px=4.0,
        mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
    ss = truth["sky_sigma"]
    return (result,
            _bias_sigma(base.background, true_sky, ss),
            _bias_sigma(result.surface, true_sky, ss) if result.ok else float("nan"))


# Module-scoped: source_masked_background runs a full Background2D pass plus
# four convolution tiers, so identical-input calls are shared rather than
# repeated per test (CLAUDE.md's shared-fixture rule).
@pytest.fixture(scope="module")
def star_only():
    return _run(star_only_frame())


@pytest.fixture(scope="module")
def moderate():
    return _run(moderate_nebula_frame())


@pytest.fixture(scope="module")
def heavy():
    return _run(heavy_nebula_frame())


@pytest.fixture(scope="module")
def gradient_heavy():
    return _run(gradient_nebula_frame())


@pytest.fixture(scope="module")
def vignetted():
    """The frame whose curvature is real — the counter-example every
    curvature guard has to survive."""
    return _run(vignetted_frame())


class TestBackgroundRecoveryAccuracy:
    """The headline claim: the masked estimate is measurably closer to truth."""

    def test_star_only_degrades_gracefully(self, star_only):
        result, base_bias, masked_bias = star_only
        assert result.ok
        # No nebulosity to remove, so the masked estimate must not *introduce*
        # an error. This is the graceful-degradation requirement.
        assert abs(masked_bias) < 0.10
        assert abs(masked_bias - base_bias) < 0.10

    def test_moderate_nebula_improves(self, moderate):
        result, base_bias, masked_bias = moderate
        assert result.ok
        assert abs(masked_bias) < 0.15
        assert abs(masked_bias) < abs(base_bias)

    def test_heavy_nebula_removes_most_of_the_bias(self, heavy):
        result, base_bias, masked_bias = heavy
        assert result.ok
        # Baseline here is ~1.28 sigma; the point of the whole module.
        assert abs(base_bias) > 1.0, "fixture no longer reproduces the biased regime"
        # ~0.33 sigma, up from ~0.19 before the surface-brightness floor. The
        # floor deliberately leaves the faintest nebulosity unmasked, and on a
        # Gaussian blob whose wings never reach zero that unmasked remainder
        # contaminates the surviving cells. It is the right trade anyway: the
        # same floor takes bounded_nebula_frame from 0.105 to 0.015 sigma,
        # because real nebulosity does end and the retained cells are clean.
        assert abs(masked_bias) < 0.45
        assert abs(masked_bias) < 0.35 * abs(base_bias)

    def test_gradient_plus_heavy_nebula(self, gradient_heavy):
        result, base_bias, masked_bias = gradient_heavy
        assert result.ok
        assert abs(base_bias) > 1.0
        # The hardest of the four frames: heavy nebulosity *and* a 9-sigma sky
        # ramp, leaving few unmasked cells to constrain the plane. Measured at
        # ~0.34 sigma against a 1.48 sigma baseline.
        assert abs(masked_bias) < 0.5
        assert abs(masked_bias) < 0.35 * abs(base_bias)

    def test_error_is_bounded_across_the_whole_frame(self, heavy, gradient_heavy):
        """Median bias alone can hide a surface that is wrong at the edges.

        A fit through cells clustered away from the nebula can pass through the
        truth at frame centre while diverging badly at the corners, so the RMS
        and worst-case errors are checked too — that is exactly how an earlier
        revision looked correct (median +0.09 sigma) while carrying a 3.9 sigma
        corner error.
        """
        for frame, result in ((heavy_nebula_frame(), heavy[0]),
                              (gradient_nebula_frame(), gradient_heavy[0])):
            _, true_sky, truth = frame
            err = (np.asarray(result.surface, np.float64) - true_sky) / truth["sky_sigma"]
            assert float(np.sqrt(np.mean(err ** 2))) < 0.6
            assert float(np.max(np.abs(err))) < 1.0

    def test_gradient_is_recovered_not_absorbed(self, gradient_heavy):
        """The fitted surface must carry the real sky gradient, not flatten it.

        Measured on the surface itself rather than on fit["gradient"]: when BIC
        selects the quadric, the linear coefficients are the tangent slope at
        frame centre, not the frame-averaged tilt, so comparing them directly
        to the injected plane is not like-for-like. The overall tilt of the
        evaluated surface is what "the gradient was recovered" actually means.
        """
        result, _, _ = gradient_heavy
        h, w = result.surface.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        A = np.stack([np.ones(xx.size), xx.ravel(), yy.ravel()], axis=-1)
        _, gx, gy = np.linalg.lstsq(A, np.asarray(result.surface, np.float64).ravel(),
                                    rcond=None)[0]
        assert gx == pytest.approx(0.4, abs=0.1)
        assert gy == pytest.approx(0.15, abs=0.1)

    def test_bounded_nebula_is_where_the_floor_pays_off(self):
        """Realistic nebulosity that actually ends, leaving genuine blank sky.

        The Gaussian frames flatter an aggressive detector, because their wings
        never reach zero so masking more always helps. Here over-masking starved
        the fit to 15 clustered cells and cost 0.105 sigma; the floor keeps 23
        and lands at 0.015.
        """
        data, true_sky, truth = bounded_nebula_frame()
        base = _unmasked_background(data)
        result = source_masked_background(
            data, box_size=64, fwhm_px=4.0,
            mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        assert result.ok
        ss = truth["sky_sigma"]
        assert abs(_bias_sigma(result.surface, true_sky, ss)) < 0.10
        assert abs(_bias_sigma(result.surface, true_sky, ss)) \
            < abs(_bias_sigma(base.background, true_sky, ss))

    def test_vignetting_with_nebulosity(self):
        """The ambiguous case: same curvature sign from optics and from nebula."""
        data, true_sky, truth = vignetted_nebula_frame()
        base = _unmasked_background(data)
        result = source_masked_background(
            data, box_size=64, fwhm_px=4.0,
            mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        assert result.ok
        assert abs(_bias_sigma(result.surface, true_sky, truth["sky_sigma"])) < 0.35
        assert result.source_mask.coverage < 0.55

    def test_masked_rms_is_closer_to_true_sky_noise(self, heavy):
        """Nebula variance inflates MADStd; masking it should deflate the RMS."""
        result, _, _ = heavy
        data, _, truth = heavy_nebula_frame()
        base_rms = float(np.median(_unmasked_background(data).background_rms))
        true_sigma = truth["sky_sigma"]
        assert abs(result.rms_median - true_sigma) < abs(base_rms - true_sigma)


class TestSkyScaffold:
    """Stage 1 — the one-sided, lower-half-MAD robust fit."""

    def test_rejection_is_one_sided(self):
        """A cell far BELOW the sky must survive; the same excursion ABOVE must not.

        Symmetric clipping would reject both. Keeping the low cell is the whole
        mechanism that makes the scaffold robust to majority contamination.

        The mesh carries realistic scatter: a perfectly noiseless one makes
        every MAD-style scale estimate exactly zero, which is a degenerate case
        no real Background2D mesh produces.
        """
        rng = np.random.default_rng(0)
        mesh = 1000.0 + rng.normal(0.0, 10.0, (8, 8))
        rms = np.full((8, 8), 10.0)
        mesh[2, 2] = 1000.0 - 500.0   # deep low outlier
        mesh[5, 5] = 1000.0 + 500.0   # matching high outlier
        fit = fit_sky_scaffold(mesh, rms, 64, (512, 512))
        kept = fit["kept_cells"]
        assert kept[2, 2], "a low cell was rejected — the clip is not one-sided"
        assert not kept[5, 5], "a high cell survived — contamination is not rejected"

    def test_recovers_plane_under_majority_contamination(self):
        """More than half the cells contaminated, and the plane still comes back.

        The contamination is a centred blob, the shape real nebulosity takes.
        A half-plane whose edge runs parallel to the gradient is deliberately
        *not* used: that geometry is indistinguishable from a steeper gradient
        using only the mesh values, so no estimator could separate them, and a
        test built on it would be asserting the impossible rather than a
        regression.
        """
        n = 10
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[0:n, 0:n]
        cx = (xx + 0.5) * 64.0
        cy = (yy + 0.5) * 64.0
        truth = 1000.0 + 0.4 * cx + 0.15 * cy
        mesh = truth + rng.normal(0.0, 10.0, truth.shape)
        contaminated = np.hypot(xx - 4.5, yy - 4.5) < 4.37   # ~60% of cells
        mesh[contaminated] += 300.0
        assert contaminated.mean() > 0.5
        rms = np.full(mesh.shape, 10.0)
        fit = fit_sky_scaffold(mesh, rms, 64, (n * 64, n * 64))
        assert not np.any(fit["kept_cells"] & contaminated), \
            "contaminated cells survived the one-sided rejection"
        assert fit["gradient"]["magnitude"] == pytest.approx(np.hypot(0.4, 0.15), rel=0.15)

    def test_rejects_with_a_plane_but_may_finish_with_a_quadric(self):
        """Two-stage: reject at order 1, then let BIC add curvature if earned.

        Iterating with a quadric lets the scaffold swallow broad nebulosity;
        finishing with only a plane cannot represent vignetting. Rejecting with
        a plane and refitting the survivors gets both.
        """
        data, _, _ = heavy_nebula_frame()
        base = _unmasked_background(data)
        two_stage = fit_sky_scaffold(base.background_mesh, base.background_rms_mesh,
                                     64, data.shape[:2])
        plane_only = fit_sky_scaffold(base.background_mesh, base.background_rms_mesh,
                                      64, data.shape[:2], final_max_order=1)
        # Same rejection either way — the second stage must not change which
        # cells were kept, only the surface fitted through them.
        assert np.array_equal(two_stage["kept_cells"], plane_only["kept_cells"])
        assert plane_only["selected_order"] <= 1
        assert two_stage["selected_order"] <= 2

    def test_recovers_vignetting_the_plane_scaffold_could_not(self):
        """The second stage exists for this frame: real curvature, no nebulosity.

        A plane-only scaffold leaves the whole bowl in the residual, which the
        detector then masks as if it were source — measured at >50% coverage on
        a frame with no extended structure in it at all.
        """
        data, true_sky, truth = vignetted_frame()
        base = _unmasked_background(data)
        result = source_masked_background(
            data, box_size=64, fwhm_px=4.0,
            mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        assert result.ok
        assert result.scaffold["selected_order"] == 2, "curvature was not recovered"
        assert result.source_mask.coverage < 0.40, "still masking a source-free frame"
        assert abs(_bias_sigma(result.surface, true_sky, truth["sky_sigma"])) < 0.25


class TestDetectionThresholds:
    """Stage 2 — auditable, and correct against the smoothed noise."""

    def test_smoothed_sigma_matches_measurement(self):
        """sigma/sqrt(4*pi*sigma_k**2) must match a real convolution of white noise."""
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 30.0, (600, 600))
        from scipy.signal import fftconvolve

        from analysis.source_mask import _gaussian_kernel
        for k_sigma in (4.0, 8.0):
            conv = fftconvolve(noise, _gaussian_kernel(k_sigma), mode="same")
            # Trim the edges, where 'same' mode truncates the kernel.
            interior = conv[100:-100, 100:-100]
            assert float(np.std(interior)) == pytest.approx(
                _smoothed_sigma(30.0, k_sigma), rel=0.10)

    def test_every_tier_records_its_threshold(self, heavy):
        """Auditability: thresholds are reported, not hidden inside the mask."""
        result, _, _ = heavy
        tiers = result.source_mask.tiers
        assert len(tiers) == 4          # 1 point tier + 3 extended tiers
        for tier in tiers:
            assert tier["threshold_adu"] > 0
            assert {"kernel_sigma", "n_sigma", "npixels", "n_segments"} <= set(tier)
        assert tiers[0]["tier"] == "point"

    def test_recorded_threshold_is_the_larger_of_both_bounds(self, heavy):
        result, _, _ = heavy
        sky_sigma = result.source_mask.sky_sigma
        for tier in result.source_mask.tiers[1:]:
            statistical = tier["n_sigma"] * _smoothed_sigma(sky_sigma, tier["kernel_sigma"])
            floor = SOURCEMASK_MIN_SURFACE_BRIGHTNESS_SIGMA * sky_sigma
            assert tier["threshold_statistical_adu"] == pytest.approx(statistical, rel=1e-6)
            assert tier["threshold_floor_adu"] == pytest.approx(floor, rel=1e-6)
            assert tier["threshold_adu"] == pytest.approx(max(statistical, floor), rel=1e-6)

    def test_floor_binds_on_the_deep_tiers_and_is_reported(self, heavy):
        """The broadest kernels are exactly where statistical significance runs away.

        At sigma_k=16 px the noise drops ~57x, so the statistical bound alone
        triggered on structure at 3.5% of sky noise. The audit record must say
        which bound won, or the report cannot explain why a tier stopped.
        """
        result, _, _ = heavy
        deepest = max(result.source_mask.tiers[1:], key=lambda t: t["kernel_sigma"])
        assert deepest["threshold_source"] == "surface brightness"
        assert deepest["threshold_floor_adu"] > deepest["threshold_statistical_adu"]

    def test_floor_detects_strictly_less_than_significance_alone(self):
        """Compare the two thresholds on identical input — floor off vs on."""
        rng = np.random.default_rng(3)
        h = w = 512
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        # A broad, very faint plateau: real signal, but far too faint to matter.
        faint = 0.08 * 30.0 * np.exp(-0.5 * (np.hypot(yy - 256, xx - 256) / 150.0) ** 2)
        residual = faint + rng.normal(0.0, 30.0, (h, w))

        from analysis.source_mask import _detect_tier
        loose, rec_loose = _detect_tier(residual, 16.0, 2.0, 512, 30.0,
                                        min_surface_brightness=0.0)
        tight, rec_tight = _detect_tier(residual, 16.0, 2.0, 512, 30.0,
                                        min_surface_brightness=0.25)
        assert rec_tight["threshold_adu"] > rec_loose["threshold_adu"]
        assert rec_loose["threshold_source"] == "statistical"
        assert rec_tight["threshold_source"] == "surface brightness"
        assert tight.sum() < loose.sum(), "the floor did not reduce what was detected"
        # Everything the tight cut finds must also be found by the loose one.
        assert not np.any(tight & ~loose)

    def test_stars_do_not_dominate_the_extended_tiers(self, star_only):
        """A star-only frame must not come back mostly masked.

        Regression guard: convolved at 16 px a bright star still stands far
        above that tier's threshold and registers as a ~54 px-radius blob.
        Before the point sources were excised ahead of the extended tiers,
        40 ordinary stars alone masked 63% of this frame.
        """
        result, _, _ = star_only
        assert result.source_mask.coverage < 0.35
        assert result.source_mask.extended_mask.mean() < 0.10


class TestClassification:
    """Stage 3 — extent-based, and deliberately not eccentricity-based."""

    def test_star_and_blob_land_in_the_right_classes(self):
        data, _, _ = make_bg_frame(
            nebulae=((150.0, 150.0, 60.0, 90.0),), n_stars=25, seed=3)
        base = _unmasked_background(data)
        result = source_masked_background(
            data, box_size=64, fwhm_px=4.0,
            mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        assert result.source_mask.n_point > 0, "no point sources classified"
        assert result.source_mask.n_extended > 0, "the nebula was not classified extended"

    def test_extended_blob_is_actually_covered(self):
        """The mask must land on the nebula, not merely be the right size."""
        data, _, truth = make_bg_frame(
            nebulae=((256.0, 256.0, 80.0, 120.0),), n_stars=10, seed=4)
        base = _unmasked_background(data)
        result = source_masked_background(
            data, box_size=64, fwhm_px=4.0,
            mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        core = truth["nebula"] > 2.0 * truth["sky_sigma"]
        covered = float((result.source_mask.mask & core).sum()) / core.sum()
        assert covered > 0.95

    def test_classes_are_mutually_exclusive(self, heavy):
        result, _, _ = heavy
        sm = result.source_mask
        assert not np.any(sm.point_mask & sm.extended_mask)
        assert np.array_equal(sm.mask, sm.point_mask | sm.extended_mask)
        assert sm.coverage == pytest.approx(sm.mask.mean())

    def test_segment_records_report_shape_without_gating_on_it(self, heavy):
        """Eccentricity is recorded for the audit table but must not classify."""
        result, _, _ = heavy
        assert result.source_mask.segments
        for seg in result.source_mask.segments:
            assert {"area", "equivalent_radius", "eccentricity",
                    "classification", "tier"} <= set(seg)
            assert seg["classification"] in ("point", "extended")


class TestOrderCapAndFallback:
    """Stage 4 — the guard rails that keep a sparse mesh from extrapolating wildly."""

    @pytest.mark.parametrize("n_cells,expected", [
        (SOURCEMASK_MIN_CELLS_QUADRIC, 2),
        (SOURCEMASK_MIN_CELLS_QUADRIC - 1, 1),
        (SOURCEMASK_MIN_CELLS_PLANE, 1),
        (SOURCEMASK_MIN_CELLS_PLANE - 1, 0),
        (SOURCEMASK_MIN_CELLS_CONSTANT, 0),
        (SOURCEMASK_MIN_CELLS_CONSTANT - 1, None),
    ])
    def test_order_cap_ladder(self, n_cells, expected):
        assert _order_cap_for(n_cells) is expected

    def test_quadric_gate_is_relative_to_mesh_size(self):
        """The same surviving-cell count must mean different things on different
        image scales.

        SOURCEMASK_MIN_CELLS_QUADRIC alone is 54.7% of a 64-cell mesh but 1.2% of
        a 2925-cell one, so an absolute-only gate evaporates on exactly the large
        frames where a quadric has the most masked area to extrapolate across.
        """
        n = SOURCEMASK_MIN_CELLS_QUADRIC
        assert _order_cap_for(n, 64) == 2            # over half a small mesh: fine
        assert _order_cap_for(n, 2925) == 1          # a rounding error of a big one
        # 492 of 2925 is the real-data case that motivated this.
        assert _order_cap_for(492, 2925) == 1
        assert _order_cap_for(1500, 2925) == 2

    def test_absolute_floor_still_binds_on_a_small_mesh(self):
        """The fraction must never *loosen* the gate: a quarter of a 64-cell mesh
        is 16 cells, and 16 cells cannot honestly carry six quadric terms."""
        assert _order_cap_for(20, 64) == 1
        assert _order_cap_for(SOURCEMASK_MIN_CELLS_QUADRIC - 1, 64) == 1

    def test_default_total_reproduces_the_absolute_only_rule(self):
        """n_cells_total defaults to 0 so every single-argument call, including
        the ones in this file, keeps its pre-existing meaning."""
        for n in (3, 4, 8, 20, 34, 35, 100):
            assert _order_cap_for(n) == _order_cap_for(n, 0)

    def test_selected_order_never_exceeds_the_cap(self, heavy, gradient_heavy):
        for result, _, _ in (heavy, gradient_heavy):
            assert result.fit["selected_order"] <= result.order_cap

    def test_total_mask_falls_back_instead_of_raising(self):
        """photutils raises ValueError outright when everything is masked."""
        data, _, _ = star_only_frame()
        full_mask = np.ones(data.shape, dtype=bool)
        payload, reason = masked_background_estimate(data, full_mask, 64)
        assert payload is None
        assert reason and "covers" in reason

    def test_near_total_mask_falls_back(self):
        data, _, _ = star_only_frame()
        mask = np.ones(data.shape, dtype=bool)
        mask[:4, :4] = False        # a sliver of sky, far too little to fit
        payload, reason = masked_background_estimate(data, mask, 64)
        assert payload is None
        assert reason

    def test_collinear_survivors_drop_to_a_constant(self):
        """Cells surviving in a single strip cannot constrain a plane."""
        data, _, _ = star_only_frame()
        mask = np.ones(data.shape, dtype=bool)
        mask[:64, :] = False        # one mesh row of sky only
        payload, reason = masked_background_estimate(data, mask, 64)
        if payload is not None:
            assert payload["order_cap"] == 0
            assert payload["fit"]["selected_order"] == 0

    def test_fallback_result_is_well_formed(self):
        """An aborted estimate must still be a usable object for the report."""
        flat = np.full((256, 256), 1000.0, dtype=np.float32)
        result = source_masked_background(flat, box_size=64, fwhm_px=4.0, n_passes=1)
        assert isinstance(result, MaskedBackgroundResult)
        if not result.ok:
            assert result.surface is None
            assert result.fallback_reason


class TestPlaneQuadricAgreement:
    """Is a fitted quadric following real curvature, or extrapolating across the mask?"""

    def test_reported_whenever_a_quadric_was_on_the_table(self, vignetted):
        result, _, _ = vignetted
        assert result.order_cap == 2
        ag = result.agreement
        assert ag is not None
        assert ag["max_divergence_adu"] >= 0
        assert ag["max_divergence_measured_adu"] >= 0
        assert ag["verdict"] in ("agree", "diverge")

    def test_real_curvature_survives_despite_a_large_divergence(self, vignetted):
        """The load-bearing case. A vignetted frame's plane genuinely cannot fit,
        so the two surfaces differ hugely — several sky sigma — and the quadric is
        nonetheless correct. Divergence alone would demote it and make the
        estimate worse; the ratio is what keeps it.
        """
        ag = vignetted[0].agreement
        assert ag["max_divergence_sigma"] > SOURCEMASK_MAX_MODEL_DIVERGENCE_SIGMA
        assert ag["extrapolation_ratio"] <= SOURCEMASK_MAX_EXTRAPOLATION_RATIO
        assert ag["verdict"] == "agree"
        assert ag["demoted"] is False
        assert vignetted[0].fit["selected_order"] == 2

    def test_absent_when_no_quadric_was_considered(self, heavy):
        """The common healthy case: the cell-count gate already ruled curvature
        out, so there is nothing to second-guess and nothing to report."""
        result, _, _ = heavy
        assert result.order_cap < 2
        assert result.agreement is None

    def test_divergence_is_measured_over_the_whole_frame_not_just_the_cells(self):
        """The frame maximum can only ever be >= the maximum at the cells, since
        the cells are a subset of the frame. If that ever inverted, the ratio
        would read below 1 and the extrapolation test would be inert.
        """
        data, _, _ = gradient_nebula_frame()
        base = _unmasked_background(data)
        mesh = np.asarray(base.background_mesh, dtype=np.float64)
        rms = np.asarray(base.background_rms_mesh, dtype=np.float64)
        valid = np.isfinite(mesh) & np.isfinite(rms) & (rms > 0)
        ag = plane_quadric_agreement(mesh, rms, 64, data.shape[:2], valid,
                                     float(np.median(rms)))
        assert ag is not None
        assert ag["max_divergence_adu"] >= ag["max_divergence_measured_adu"]
        assert ag["extrapolation_ratio"] >= 1.0

    def test_returns_none_when_the_cells_cannot_support_both_fits(self):
        data, _, _ = star_only_frame()
        base = _unmasked_background(data)
        mesh = np.asarray(base.background_mesh, dtype=np.float64)
        rms = np.asarray(base.background_rms_mesh, dtype=np.float64)
        valid = np.zeros(mesh.shape, dtype=bool)
        valid.flat[:2] = True       # two cells: not even a plane
        assert plane_quadric_agreement(mesh, rms, 64, data.shape[:2], valid,
                                       float(np.median(rms))) is None


class TestOrchestrator:
    def test_result_fields_are_populated(self, heavy):
        result, _, _ = heavy
        assert result.ok
        assert result.surface.dtype == np.float32
        assert result.surface.shape == heavy_nebula_frame()[0].shape
        assert isinstance(result.source_mask, SourceMaskResult)
        assert result.scaffold is not None
        assert 0 < result.n_cells <= result.n_cells_total
        assert result.fallback_reason is None

    def test_deterministic(self):
        data, _, _ = moderate_nebula_frame()
        base = _unmasked_background(data)
        kwargs = dict(box_size=64, fwhm_px=4.0,
                      mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        first = source_masked_background(data, **kwargs)
        second = source_masked_background(data, **kwargs)
        assert np.array_equal(first.surface, second.surface)
        assert np.array_equal(first.source_mask.mask, second.source_mask.mask)

    def test_decimated_extended_tiers_agree_with_full_resolution(self):
        """step>1 is a cost optimisation, so it must not change the conclusion."""
        data, true_sky, truth = heavy_nebula_frame()
        base = _unmasked_background(data)
        kwargs = dict(box_size=64, fwhm_px=4.0,
                      mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        full = source_masked_background(data, step=1, **kwargs)
        dec = source_masked_background(data, step=2, **kwargs)
        assert full.ok and dec.ok
        ss = truth["sky_sigma"]
        assert abs(_bias_sigma(dec.surface, true_sky, ss)
                   - _bias_sigma(full.surface, true_sky, ss)) < 0.20

    def test_computes_its_own_mesh_when_not_supplied(self):
        data, true_sky, truth = moderate_nebula_frame()
        result = source_masked_background(data, box_size=64, fwhm_px=4.0)
        assert result.ok
        assert abs(_bias_sigma(result.surface, true_sky, truth["sky_sigma"])) < 0.15


class TestBuildSourceMask:
    def test_zero_sky_sigma_returns_an_empty_mask(self):
        """A degenerate noise estimate must not mask the whole frame."""
        data, _, _ = star_only_frame()
        pedestal = np.full(data.shape, 1000.0)
        result = build_source_mask(data, pedestal, 0.0, 4.0)
        assert result.coverage == 0.0
        assert not result.mask.any()

    def test_mask_shape_matches_input(self, heavy):
        result, _, _ = heavy
        assert result.source_mask.mask.shape == heavy_nebula_frame()[0].shape
        assert result.source_mask.mask.dtype == bool


class TestUserExclusionRegions:
    """Hand-drawn regions — for structure no threshold can identify.

    A smooth nebula is genuinely degenerate with a sky gradient, and a centred
    nebula with vignetting, so those cases cannot be resolved from pixel values
    alone however the detector is tuned.
    """

    @staticmethod
    def _corner_region():
        """A polygon over one corner, in the normalised schema the GUI stores."""
        return [{"kind": "polygon",
                 "points": [(0.02, 0.02), (0.30, 0.02), (0.30, 0.30), (0.02, 0.30)]}]

    @pytest.fixture(scope="class")
    @classmethod
    def with_region(cls):
        from analysis.inspector_regions import exclusion_mask
        data, true_sky, truth = moderate_nebula_frame()
        base = _unmasked_background(data)
        user = exclusion_mask(data.shape[:2], cls._corner_region())
        kwargs = dict(box_size=64, fwhm_px=4.0,
                      mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        plain = source_masked_background(data, **kwargs)
        drawn = source_masked_background(data, user_exclusion=user, **kwargs)
        return plain, drawn, user, data, true_sky, truth

    def test_default_none_is_unchanged(self, with_region):
        """No regions must reproduce the un-supervised result exactly."""
        plain, _, _, _, _, _ = with_region
        assert plain.source_mask.user_mask is None
        assert plain.source_mask.user_coverage == 0.0

    def test_drawn_pixels_are_always_masked(self, with_region):
        _, drawn, user, _, _, _ = with_region
        assert drawn.source_mask.user_mask is not None
        assert np.array_equal(drawn.source_mask.user_mask, user)
        # Every drawn pixel is in the final mask, whatever the detector thought.
        assert np.all(drawn.source_mask.mask[user])

    def test_user_class_is_exclusive_of_the_detector_classes(self, with_region):
        """Attribution matters: the report must not credit the algorithm."""
        _, drawn, user, _, _, _ = with_region
        sm = drawn.source_mask
        assert not np.any(sm.point_mask & user)
        assert not np.any(sm.extended_mask & user)
        assert np.array_equal(sm.mask, sm.point_mask | sm.extended_mask | user)
        assert sm.coverage == pytest.approx(sm.mask.mean())

    def test_drawing_over_sky_costs_cells(self, with_region):
        """A region over blank sky removes real measurements — it is not free."""
        plain, drawn, _, _, _, _ = with_region
        assert drawn.n_cells < plain.n_cells
        assert drawn.coverage > plain.coverage

    def test_scaffold_drops_the_excluded_cells(self):
        """Stage 1 injection: the pedestal must not be fitted through drawn cells."""
        from analysis.inspector_regions import exclusion_mask
        data, _, _ = moderate_nebula_frame()
        base = _unmasked_background(data)
        user = exclusion_mask(data.shape[:2], self._corner_region())
        result = source_masked_background(
            data, box_size=64, fwhm_px=4.0, user_exclusion=user,
            mesh=base.background_mesh, rms_mesh=base.background_rms_mesh)
        kept = result.scaffold["kept_cells"]
        from analysis.source_mask import _cell_unmasked_fraction
        frac = _cell_unmasked_fraction(user, kept.shape, 64)
        fully_drawn = frac < 0.5
        assert fully_drawn.any(), "region too small to cover a whole mesh cell"
        assert not np.any(kept & fully_drawn)

    def test_shape_mismatch_raises(self):
        data, _, _ = star_only_frame()
        with pytest.raises(ValueError, match="does not match"):
            source_masked_background(data, box_size=64,
                                     user_exclusion=np.zeros((10, 10), dtype=bool))

    def test_region_covering_everything_falls_back(self):
        """Must report a fallback, not fabricate a surface or raise."""
        data, _, _ = star_only_frame()
        result = source_masked_background(
            data, box_size=64, user_exclusion=np.ones(data.shape[:2], dtype=bool))
        assert not result.ok
        assert result.fallback_reason

    def test_honoured_even_when_sky_sigma_is_degenerate(self):
        """User regions do not depend on any threshold, so they still apply."""
        data, _, _ = star_only_frame()
        user = np.zeros(data.shape[:2], dtype=bool)
        user[10:60, 10:60] = True
        sm = build_source_mask(data, np.zeros(data.shape[:2]), 0.0, 4.0,
                               user_exclusion=user)
        assert np.array_equal(sm.mask, user)
        assert sm.coverage == pytest.approx(user.mean())
