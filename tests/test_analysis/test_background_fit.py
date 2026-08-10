"""Unit tests for analysis/background_fit.py — the Section 3f gradient/
curvature polynomial fit to a Background2D mesh.

All tests build synthetic mesh/rms_mesh arrays directly (no FITS/AstroImage
needed) using the exact same cell-center reconstruction formula
fit_background_surface() itself uses, so injected truth values land at
precisely the coordinates the fit will recover them from.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.background_fit import (
    evaluate_surface_at,
    compare_fits,
    evaluate_surface,
    fit_background_surface,
)


def _synthetic_mesh(n_rows, n_cols, box_size, image_shape, truth_fn,
                     noise_std=0.0, rms_value=1.0, seed=0):
    """Build a (mesh, rms_mesh) pair from a truth function of centered native
    pixel offsets (dx, dy), evaluated at the same cell centers
    fit_background_surface() itself reconstructs from box_size."""
    yy, xx = np.mgrid[0:n_rows, 0:n_cols]
    x_native = (xx.astype(np.float64) + 0.5) * box_size
    y_native = (yy.astype(np.float64) + 0.5) * box_size
    h_img, w_img = image_shape
    dx = x_native - w_img / 2.0
    dy = y_native - h_img / 2.0
    mesh = truth_fn(dx, dy).astype(np.float64)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        mesh = mesh + rng.normal(0.0, noise_std, size=mesh.shape)
    rms_mesh = np.full(mesh.shape, rms_value, dtype=np.float64)
    return mesh.astype(np.float32), rms_mesh.astype(np.float32)


class TestGradientRecovery:
    def test_plane_recovered_within_a_few_se(self):
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)
        b_true, c_true = 0.02, -0.03

        def truth(dx, dy):
            return 100.0 + b_true * dx + c_true * dy

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.05, rms_value=0.05)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)

        assert not fit["insufficient_data"]
        assert fit["gradient"] is not None
        mag_true = float(np.hypot(b_true, c_true))
        se = fit["gradient"]["se"]
        assert abs(fit["gradient"]["magnitude"] - mag_true) < 5 * se
        assert fit["gradient"]["p_value"] < 0.05

    def test_exact_adu_range_is_corner_based_not_diagonal(self):
        # Noise-free plane: recovery should be numerically exact, so this
        # directly tests the adu_range formula, not fit precision.
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)
        b_true, c_true = 0.05, 0.01   # deliberately NOT diagonal-aligned

        def truth(dx, dy):
            return 50.0 + b_true * dx + c_true * dy

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.0, rms_value=1.0)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)
        h_img, w_img = image_shape
        expected_range = abs(b_true) * w_img + abs(c_true) * h_img
        wrong_diagonal_formula = np.hypot(b_true, c_true) * np.hypot(w_img, h_img)
        assert fit["gradient"]["adu_range"] == pytest.approx(expected_range, rel=1e-3)
        # The two formulas must differ meaningfully for this non-diagonal
        # case, or the test wouldn't distinguish them.
        assert abs(expected_range - wrong_diagonal_formula) > 0.1 * expected_range


class TestCurvatureRecovery:
    def test_isotropic_and_anisotropic_recovered(self):
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)
        d_true, e_true, f_true = 0.0012, 0.0006, 0.0018

        def truth(dx, dy):
            return 200.0 + d_true * dx ** 2 + e_true * dy ** 2 + f_true * dx * dy

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.02, rms_value=0.02)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)

        assert fit["selected_order"] == 2
        assert fit["isotropic_curvature"] is not None
        assert fit["anisotropic_curvature"] is not None
        iso_true = (d_true + e_true) / 2.0
        aniso_true = float(np.hypot((d_true - e_true) / 2.0, f_true / 2.0))
        iso = fit["isotropic_curvature"]
        aniso = fit["anisotropic_curvature"]
        assert abs(iso["magnitude"] - iso_true) < 5 * iso["se"]
        assert abs(aniso["magnitude"] - aniso_true) < 5 * aniso["se"]

    def test_anisotropic_magnitude_uses_f_over_2_not_raw_f(self):
        # d == e isolates the anisotropic term to be purely f-driven, so the
        # correct magnitude is exactly |f|/2 -- a formula using raw f instead
        # of f/2 would be off by a factor of 2, easily distinguished here.
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)
        d_true = e_true = 0.001
        f_true = 0.004

        def truth(dx, dy):
            return 200.0 + d_true * dx ** 2 + e_true * dy ** 2 + f_true * dx * dy

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.0, rms_value=1.0)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)

        aniso = fit["anisotropic_curvature"]
        assert aniso["magnitude"] == pytest.approx(abs(f_true) / 2.0, rel=1e-3)
        # A factor-of-2 bug (raw f instead of f/2) would land here instead:
        assert aniso["magnitude"] != pytest.approx(abs(f_true), rel=1e-2)
        # d == e means the traceless matrix's principal axis is at 45 deg.
        assert abs(abs(aniso["direction_deg"]) - 45.0) < 1.0


class TestModelSelection:
    def test_pure_noise_selects_order_zero(self):
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)

        def truth(dx, dy):
            return np.full(dx.shape, 500.0)

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.5, rms_value=0.5, seed=3)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)
        assert fit["selected_order"] == 0

    def test_strong_plane_signal_selects_order_at_least_one(self):
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)

        def truth(dx, dy):
            return 500.0 + 0.05 * dx - 0.04 * dy

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.1, rms_value=0.1, seed=4)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)
        assert fit["selected_order"] >= 1
        assert fit["bic"][1] < fit["bic"][0]

    def test_strong_quadric_signal_selects_order_two(self):
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)

        def truth(dx, dy):
            return 500.0 + 0.003 * dx ** 2 + 0.002 * dy ** 2 + 0.001 * dx * dy

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.05, rms_value=0.05, seed=5)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)
        assert fit["selected_order"] == 2
        assert fit["bic"][2] == min(fit["bic"].values())


class TestDegreesOfFreedomGuard:
    def test_four_cell_mesh_never_attempts_order_two(self):
        # Mirrors the real 128x128-minimum test-fixture case at box_size=64.
        box_size = 64
        n_rows = n_cols = 2
        image_shape = (n_rows * box_size, n_cols * box_size)
        mesh = np.full((n_rows, n_cols), 100.0, dtype=np.float32)
        rms_mesh = np.full((n_rows, n_cols), 1.0, dtype=np.float32)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)
        assert not fit["insufficient_data"]
        assert 2 not in fit["bic"]
        assert 1 in fit["bic"]
        assert 0 in fit["bic"]

    def test_single_cell_is_insufficient_data(self):
        mesh = np.array([[100.0]], dtype=np.float32)
        rms_mesh = np.array([[1.0]], dtype=np.float32)
        fit = fit_background_surface(mesh, rms_mesh, 64, (64, 64))
        assert fit["insufficient_data"] is True
        assert fit["gradient"] is None
        assert fit["isotropic_curvature"] is None
        assert fit["anisotropic_curvature"] is None
        assert fit["coefficients"] is None


class TestInvalidCellExclusion:
    def test_nonpositive_and_nan_rms_cells_excluded(self):
        box_size = 64
        n_rows = n_cols = 8
        image_shape = (n_rows * box_size, n_cols * box_size)

        def truth(dx, dy):
            return 100.0 + 0.01 * dx

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape,
                                          truth, noise_std=0.05, rms_value=0.05, seed=6)
        rms_mesh = rms_mesh.copy()
        rms_mesh[0, 0] = 0.0
        rms_mesh[1, 1] = -1.0
        rms_mesh[2, 2] = np.nan
        mesh = mesh.copy()
        mesh[3, 3] = np.nan

        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)
        assert fit["n_cells"] == n_rows * n_cols - 4
        assert not fit["insufficient_data"]


class TestEvaluateSurfaceShapeInvariant:
    @pytest.mark.parametrize("h,w,step", [(512, 512, 1), (512, 512, 5), (300, 700, 4), (129, 129, 3)])
    def test_matches_stride_slice_shape(self, h, w, step):
        box_size = 64
        n_rows, n_cols = max(2, h // box_size), max(2, w // box_size)
        image_shape = (n_rows * box_size, n_cols * box_size)

        def truth(dx, dy):
            return 10.0 + 0.001 * dx

        mesh, rms_mesh = _synthetic_mesh(n_rows, n_cols, box_size, image_shape, truth,
                                          noise_std=0.01, rms_value=0.01, seed=7)
        fit = fit_background_surface(mesh, rms_mesh, box_size, image_shape)

        dummy = np.zeros((h, w), dtype=np.float32)
        expected_shape = dummy[::step, ::step].shape
        result = evaluate_surface(fit, (h, w), step)
        assert result.shape == expected_shape


class TestCompareFits:
    @staticmethod
    def _fake_fit(gradient_mag=None, gradient_se=None):
        return {
            "gradient": (None if gradient_mag is None
                         else {"magnitude": gradient_mag, "se": gradient_se}),
            "isotropic_curvature": None,
            "anisotropic_curvature": None,
        }

    def test_well_separated_magnitudes_give_small_p(self):
        fit_a = self._fake_fit(10.0, 0.5)
        fit_b = self._fake_fit(2.0, 0.5)
        cmp = compare_fits(fit_a, fit_b)
        assert cmp["gradient"]["p"] < 0.01

    def test_indistinguishable_magnitudes_not_significant(self):
        fit_a = self._fake_fit(10.0, 5.0)
        fit_b = self._fake_fit(10.5, 5.0)
        cmp = compare_fits(fit_a, fit_b)
        assert cmp["gradient"]["p"] >= 0.05

    def test_none_guard_when_term_missing_on_one_side(self):
        fit_a = self._fake_fit(10.0, 0.5)
        fit_b = self._fake_fit(None, None)
        cmp = compare_fits(fit_a, fit_b)
        assert cmp["gradient"] == {"z": None, "p": None}
        assert cmp["isotropic_curvature"] == {"z": None, "p": None}
        assert cmp["anisotropic_curvature"] == {"z": None, "p": None}


class TestMaxOrderCap:
    """`max_order` gates which candidate orders BIC may choose from at all.

    Added for analysis/source_mask.py's stage 4: on a sparse, source-masked
    mesh BIC will happily select the quadric from a handful of surviving cells
    and then extrapolate wildly across the masked region.
    """

    @staticmethod
    def _quadric_mesh():
        # Curvature strong enough that unrestricted BIC prefers order 2.
        return _synthetic_mesh(
            8, 8, 64, (512, 512),
            lambda dx, dy: 1000.0 + 0.05 * dx + 4e-4 * dx ** 2 + 4e-4 * dy ** 2,
            noise_std=0.5, rms_value=1.0, seed=7)

    def test_default_is_unrestricted(self):
        mesh, rms = self._quadric_mesh()
        fit = fit_background_surface(mesh, rms, 64, (512, 512))
        assert fit["selected_order"] == 2, "fixture no longer prefers the quadric"

    @pytest.mark.parametrize("cap", [0, 1, 2])
    def test_selected_order_never_exceeds_cap(self, cap):
        mesh, rms = self._quadric_mesh()
        fit = fit_background_surface(mesh, rms, 64, (512, 512), max_order=cap)
        assert fit["selected_order"] <= cap
        assert all(order <= cap for order in fit["bic"])

    def test_capped_fit_drops_the_curvature_terms(self):
        mesh, rms = self._quadric_mesh()
        fit = fit_background_surface(mesh, rms, 64, (512, 512), max_order=1)
        assert fit["isotropic_curvature"] is None
        assert fit["anisotropic_curvature"] is None
        assert fit["gradient"] is not None

    def test_cap_zero_still_produces_a_usable_fit(self):
        mesh, rms = self._quadric_mesh()
        fit = fit_background_surface(mesh, rms, 64, (512, 512), max_order=0)
        assert not fit["insufficient_data"]
        assert fit["selected_order"] == 0
        assert fit["gradient"] is None
        surface = evaluate_surface(fit, (512, 512), 64)
        assert np.all(np.isfinite(surface))
        assert np.ptp(surface) == pytest.approx(0.0, abs=1e-6)


class TestDesignMatrixBroadcasting:
    """evaluate_surface_at must accept separable coordinate grids.

    analysis/source_mask.py evaluates the scaffold fit on mesh-cell centres
    built as a row vector of x's against a column vector of y's; before the
    broadcast, np.stack raised because the constant column alone was dx-shaped.
    """

    @staticmethod
    def _plane_fit():
        mesh, rms = _synthetic_mesh(8, 8, 64, (512, 512),
                                    lambda dx, dy: 1000.0 + 0.3 * dx + 0.1 * dy)
        return fit_background_surface(mesh, rms, 64, (512, 512))

    def test_separable_grids_broadcast(self):
        fit = self._plane_fit()
        xs = np.arange(0.0, 512.0, 64.0)[None, :]
        ys = np.arange(0.0, 512.0, 64.0)[:, None]
        out = evaluate_surface_at(fit, xs, ys)
        assert out.shape == (8, 8)

    def test_broadcast_matches_explicit_meshgrid(self):
        fit = self._plane_fit()
        xs = np.arange(0.0, 512.0, 64.0)
        ys = np.arange(0.0, 512.0, 64.0)
        separable = evaluate_surface_at(fit, xs[None, :], ys[:, None])
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        explicit = evaluate_surface_at(fit, xx, yy)
        assert np.allclose(separable, explicit)

    def test_scalar_inputs_still_work(self):
        fit = self._plane_fit()
        assert np.isfinite(float(evaluate_surface_at(fit, 100.0, 200.0)))
