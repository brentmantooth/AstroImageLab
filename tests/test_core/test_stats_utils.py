"""Unit tests for core/stats_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from core.stats_utils import combined_se_z_test, mannwhitney_effect


class TestMannwhitneyEffect:
    def test_minimum_sample_guard(self):
        assert mannwhitney_effect([1, 2], [1, 2, 3]) == (None, None)
        assert mannwhitney_effect([1, 2, 3], [1, 2]) == (None, None)

    def test_separable_samples_give_expected_sign_and_significance(self):
        va = [10.0, 11.0, 12.0, 13.0, 14.0]
        vb = [1.0, 2.0, 3.0, 4.0, 5.0]
        p, delta = mannwhitney_effect(va, vb)
        assert p is not None and p < 0.05
        assert delta is not None and delta > 0.9

        p2, delta2 = mannwhitney_effect(vb, va)
        assert p2 is not None and p2 < 0.05
        assert delta2 is not None and delta2 < -0.9

    def test_identical_distributions_not_significant(self):
        vals = [5.0, 6.0, 7.0, 8.0, 9.0]
        p, delta = mannwhitney_effect(vals, vals)
        assert p is not None and p >= 0.05
        assert delta is not None and abs(delta) < 1e-9

    def test_matches_pairwise_matrix_reference_on_small_sample(self):
        rng = np.random.default_rng(7)
        va = rng.normal(10, 2, size=17).tolist()
        vb = rng.normal(9, 3, size=23).tolist()

        p, delta = mannwhitney_effect(va, vb)

        # Brute-force reference: Cliff's delta via the O(n*m) pairwise sign
        # matrix, computed independently here (affordable at this small size)
        # to prove the O(n log n) identity in mannwhitney_effect is exact.
        arr_a = np.array(va)
        arr_b = np.array(vb)
        ref_delta = float(np.sign(arr_a[:, None] - arr_b[None, :]).sum()) / (len(va) * len(vb))

        assert p is not None
        assert delta is not None
        assert abs(delta - ref_delta) < 1e-9


class TestCombinedSeZTest:
    def test_none_propagation(self):
        assert combined_se_z_test(None, 1.0, 2.0, 1.0) == (None, None)
        assert combined_se_z_test(1.0, None, 2.0, 1.0) == (None, None)
        assert combined_se_z_test(1.0, 1.0, None, 1.0) == (None, None)
        assert combined_se_z_test(1.0, 1.0, 2.0, None) == (None, None)

    def test_zero_combined_se_guard(self):
        assert combined_se_z_test(1.0, 0.0, 2.0, 0.0) == (None, None)

    def test_known_value_round_trip(self):
        # z = (10-7)/sqrt(1**2+1**2) = 3/sqrt(2)
        z, p = combined_se_z_test(10.0, 1.0, 7.0, 1.0)
        assert z == pytest.approx(3.0 / (2 ** 0.5))
        assert p is not None and p < 0.05

    def test_symmetry(self):
        z1, p1 = combined_se_z_test(10.0, 1.5, 7.0, 2.0)
        z2, p2 = combined_se_z_test(7.0, 2.0, 10.0, 1.5)
        assert z1 == pytest.approx(-z2)
        assert p1 == pytest.approx(p2)

    def test_indistinguishable_estimates_not_significant(self):
        z, p = combined_se_z_test(10.0, 5.0, 10.5, 5.0)
        assert p is not None and p >= 0.05
