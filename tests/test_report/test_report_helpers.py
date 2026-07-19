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
    _format_significance_html,
    _sig_td,
    _power_ratio_db,
    _nc_ratio_rows,
    _panel_display_name,
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


class TestFormatSignificanceHtml:
    def test_not_significant(self):
        html = _format_significance_html(0.5, 0.05)
        assert "n.s." in html
        assert "p=0.500" in html

    def test_trivial_effect(self):
        html = _format_significance_html(0.01, 0.05)
        assert "trivial" in html

    def test_small_effect(self):
        html = _format_significance_html(0.01, 0.2)
        assert "small" in html

    def test_medium_effect(self):
        html = _format_significance_html(0.01, 0.4)
        assert "medium" in html

    def test_large_effect(self):
        html = _format_significance_html(0.01, 0.6)
        assert "large" in html

    def test_p_below_threshold_formatted(self):
        html = _format_significance_html(0.0001, 0.6)
        assert "p&lt;0.001" in html

    def test_delta_sign_shown(self):
        html = _format_significance_html(0.01, -0.6)
        assert "delta=-0.60" in html.replace("&delta;", "delta")


class TestSigTd:
    def test_none_p_plain_cell(self):
        assert _sig_td("hello", None) == "<td>hello</td>"

    def test_significant_p_blue_background(self):
        td = _sig_td("hello", 0.01)
        assert "background:#b3e5fc" in td

    def test_nonsignificant_p_grey_background(self):
        td = _sig_td("hello", 0.5)
        assert "background:#e0e0e0" in td


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


class TestPowerRatioDb:
    def test_known_ratio_in_db(self):
        freq = np.linspace(0.0, 0.5, 10)
        rp_b = np.full(10, 1.0)
        rp_a = rp_b * 2.0
        result = _power_ratio_db(freq, rp_a, freq, rp_b)
        assert result is not None
        _, ratio_db = result
        assert ratio_db == pytest.approx(10.0 * np.log10(2.0), abs=1e-6)

    def test_identical_curves_zero_db(self):
        freq = np.linspace(0.0, 0.5, 10)
        rp = np.linspace(1.0, 5.0, 10)
        _, ratio_db = _power_ratio_db(freq, rp, freq, rp)
        assert ratio_db == pytest.approx(0.0, abs=1e-6)

    def test_zero_bins_stay_finite(self):
        freq = np.linspace(0.0, 0.5, 5)
        rp_a = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
        rp_b = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
        _, ratio_db = _power_ratio_db(freq, rp_a, freq, rp_b)
        assert np.all(np.isfinite(ratio_db))

    def test_none_inputs_return_none(self):
        freq = np.linspace(0.0, 0.5, 5)
        rp = np.ones(5)
        assert _power_ratio_db(None, rp, freq, rp) is None
        assert _power_ratio_db(freq, None, freq, rp) is None
        assert _power_ratio_db(freq, rp, None, rp) is None
        assert _power_ratio_db(freq, rp, freq, None) is None

    def test_mismatched_length_returns_none(self):
        freq_a = np.linspace(0.0, 0.5, 5)
        freq_b = np.linspace(0.0, 0.5, 8)
        rp_a = np.ones(5)
        rp_b = np.ones(8)
        assert _power_ratio_db(freq_a, rp_a, freq_b, rp_b) is None

    def test_same_length_different_values_returns_none(self):
        freq_a = np.linspace(0.0, 0.5, 5)
        freq_b = np.linspace(0.0, 0.4, 5)
        rp = np.ones(5)
        assert _power_ratio_db(freq_a, rp, freq_b, rp) is None


class TestNcRatioRows:
    def _label(self, scale):
        return f"{scale} px"

    def test_known_ratio_formatted(self):
        rows = _nc_ratio_rows({3: 2.0}, {3: 1.0}, {3: 2.0}, self._label)
        assert "2.000" in rows
        assert "1.000" in rows

    def test_missing_entry_renders_em_dash(self):
        rows = _nc_ratio_rows({3: 1.5}, {}, {}, self._label)
        assert EM_DASH in rows

    def test_none_ratio_renders_em_dash(self):
        rows = _nc_ratio_rows({3: 1.5}, {3: None}, {3: None}, self._label)
        assert EM_DASH in rows

    def test_empty_inputs_produce_no_rows(self):
        rows = _nc_ratio_rows({}, {}, {}, self._label)
        assert rows == ""

    def test_scale_label_used(self):
        rows = _nc_ratio_rows({5: 1.0}, {5: 1.0}, {5: 1.0}, lambda s: f"σ = {s} px")
        assert "σ = 5 px" in rows

    def test_ratio_cell_colored_independently_of_ab_columns(self):
        # A/B columns: A < B -> "worse"/"better"; ratio itself (0.5) vs parity (1.0)
        # is colored via a SEPARATE _better_worse_class(vr, 1.0) call.
        rows = _nc_ratio_rows({3: 1.0}, {3: 2.0}, {3: 0.5}, self._label)
        assert "worse" in rows   # A's own value vs B's
        # the ratio 0.5 < 1.0 is also "worse" here, but computed independently
        # (not derived from the A/B column classes) - verify it renders at all
        assert "0.500" in rows

    def test_union_of_scale_keys(self):
        rows = _nc_ratio_rows({3: 1.0}, {5: 1.0}, {}, self._label)
        assert "3 px" in rows
        assert "5 px" in rows


class TestPanelDisplayName:
    @pytest.mark.parametrize("pkey,expected", [
        ("std_3px", "Std Dev 3 px"),
        ("std_10px", "Std Dev 10 px"),
        ("entropy_9px", "Entropy 9 px"),
        ("log_1.5", "LoG σ 1.5 px"),
        ("wavelet_2", "Wavelet level 2"),
        ("gradient_3.0", "Gradient σ 3.0 px"),
        ("nrm_log_1.5", "LoG σ 1.5 px (noise-normalized)"),
        ("nrm_std_3px", "Std Dev 3 px (noise-normalized)"),
        ("nrm_gradient_6.0", "Gradient σ 6.0 px (noise-normalized)"),
        ("original", "Original Image (ROI)"),
    ])
    def test_display_name(self, pkey, expected):
        assert _panel_display_name(pkey) == expected
