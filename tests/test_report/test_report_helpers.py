"""Unit tests for pure helper functions in report/report_builder.py."""
from __future__ import annotations

import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from report.report_builder import (
    _val,
    _val_pm,
    _info_box,
    _arr_to_b64_png,
    _fig_to_b64,
    _psf_stat_test,
)

EM_DASH = "—"


class TestVal:
    def test_none_returns_em_dash(self):
        assert _val(None, ".2f") == EM_DASH

    def test_float_formatted(self):
        assert _val(3.14159, ".2f") == "3.14"

    def test_zero_float(self):
        assert _val(0.0, ".3f") == "0.000"

    def test_custom_fallback(self):
        assert _val(None, ".2f", fallback="N/A") == "N/A"

    def test_non_float_returns_str(self):
        # integers go through str()
        assert _val(42, ".0f") == "42"

    def test_string_passthrough(self):
        assert _val("hello", ".2f") == "hello"

    def test_scientific_format(self):
        result = _val(0.000123, ".3g")
        assert "1.23e-04" in result or "0.000123" in result or "1.23" in result


class TestValPm:
    def test_value_and_spread(self):
        result = _val_pm(3.14, 0.05, ".2f")
        assert "3.14" in result
        assert "0.05" in result
        assert "±" in result or "±" in result

    def test_zero_spread_omits_pm(self):
        result = _val_pm(3.14, 0.0, ".2f")
        assert "±" not in result and "±" not in result

    def test_none_spread_omits_pm(self):
        result = _val_pm(3.14, None, ".2f")
        assert "±" not in result and "±" not in result

    def test_none_value_returns_dash(self):
        assert _val_pm(None, 0.5, ".2f") == EM_DASH


class TestInfoBox:
    def test_contains_title(self):
        html = _info_box("body text", title="My Section")
        assert "My Section" in html

    def test_has_details_tag(self):
        html = _info_box("body")
        assert "<details" in html
        assert "</details>" in html

    def test_has_summary_tag(self):
        html = _info_box("body", title="Title")
        assert "<summary>" in html
        assert "</summary>" in html

    def test_closed_by_default(self):
        html = _info_box("body", open=False)
        assert ' open' not in html

    def test_open_attribute_present(self):
        html = _info_box("body", open=True)
        assert ' open' in html

    def test_style_attribute(self):
        html = _info_box("body", style="color:red")
        assert 'style="color:red"' in html

    def test_no_style_no_attribute(self):
        html = _info_box("body", style="")
        assert 'style=' not in html

    def test_body_in_output(self):
        html = _info_box("unique body content XYZ")
        assert "unique body content XYZ" in html


class TestPsfStatTest:
    def test_too_few_values_returns_empty(self):
        html, p = _psf_stat_test([1.0, 2.0], [3.0, 4.0])
        assert html == ""
        assert p is None

    def test_exactly_three_each_accepted(self):
        _, p = _psf_stat_test([1.0, 1.1, 0.9], [4.0, 4.1, 3.9])
        assert p is not None

    def test_significant_difference(self):
        a = [2.0, 2.1, 1.9, 2.0, 2.05]
        b = [5.0, 5.1, 4.9, 5.0, 5.05]
        html, p = _psf_stat_test(a, b)
        assert p < 0.05
        assert isinstance(html, str) and len(html) > 0

    def test_p_value_in_unit_interval(self):
        a = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02]
        b = [2.0, 2.1, 1.9, 2.0, 2.05, 1.95, 2.02]
        html, p = _psf_stat_test(a, b)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_returns_html_string(self):
        a = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02]
        b = [2.0, 2.1, 1.9, 2.0, 2.05, 1.95, 2.02]
        html, _ = _psf_stat_test(a, b)
        assert isinstance(html, str)

    def test_identical_distributions_not_significant(self):
        a = [3.0, 3.1, 2.9, 3.0, 3.05]
        html, p = _psf_stat_test(a, a)
        # Identical lists have p=1.0
        assert p == pytest.approx(1.0, abs=0.01) or (p is not None and p > 0.05)


class TestArrToB64Png:
    def test_grayscale_returns_string(self):
        arr = np.zeros((16, 16), dtype=np.uint8)
        result = _arr_to_b64_png(arr)
        assert isinstance(result, str)

    def test_grayscale_valid_png(self):
        arr = np.full((16, 16), 128, dtype=np.uint8)
        decoded = base64.b64decode(_arr_to_b64_png(arr))
        assert decoded[:4] == b"\x89PNG"

    def test_rgb_returns_string(self):
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        result = _arr_to_b64_png(arr)
        assert isinstance(result, str)

    def test_rgb_valid_png(self):
        arr = np.ones((8, 8, 3), dtype=np.uint8) * 200
        decoded = base64.b64decode(_arr_to_b64_png(arr))
        assert decoded[:4] == b"\x89PNG"

    def test_nonempty_result(self):
        arr = np.zeros((32, 32), dtype=np.uint8)
        assert len(_arr_to_b64_png(arr)) > 50


class TestFigToB64:
    def _make_fig(self) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        return fig

    def test_returns_string(self):
        result = _fig_to_b64(self._make_fig())
        assert isinstance(result, str)

    def test_valid_png(self):
        decoded = base64.b64decode(_fig_to_b64(self._make_fig()))
        assert decoded[:4] == b"\x89PNG"

    def test_custom_dpi_changes_size(self):
        lo = _fig_to_b64(self._make_fig(), dpi=72)
        hi = _fig_to_b64(self._make_fig(), dpi=200)
        assert len(base64.b64decode(hi)) > len(base64.b64decode(lo))
