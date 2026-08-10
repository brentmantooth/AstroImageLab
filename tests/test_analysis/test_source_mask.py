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
                                  source_masked_background)
from core.models import (SOURCEMASK_MIN_CELLS_CONSTANT, SOURCEMASK_MIN_CELLS_PLANE,
                         SOURCEMASK_MIN_CELLS_QUADRIC)
from tests.bg_frames import (gradient_nebula_frame, heavy_nebula_frame,
                             make_bg_frame, moderate_nebula_frame,
                             star_only_frame)


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
        assert abs(masked_bias) < 0.25
        assert abs(masked_bias) < 0.3 * abs(base_bias)

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

    def test_defaults_to_plane_order(self):
        """max_order=1 by default — a quadric scaffold can absorb broad nebulosity."""
        data, _, _ = heavy_nebula_frame()
        base = _unmasked_background(data)
        fit = fit_sky_scaffold(base.background_mesh, base.background_rms_mesh,
                               64, data.shape[:2])
        assert fit["selected_order"] <= 1


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

    def test_recorded_threshold_matches_the_formula(self, heavy):
        result, _, _ = heavy
        sky_sigma = result.source_mask.sky_sigma
        for tier in result.source_mask.tiers[1:]:
            expected = tier["n_sigma"] * _smoothed_sigma(sky_sigma, tier["kernel_sigma"])
            assert tier["threshold_adu"] == pytest.approx(expected, rel=1e-6)

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
