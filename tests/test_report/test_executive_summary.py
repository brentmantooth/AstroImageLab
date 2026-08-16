"""Tests for the unnumbered Executive Summary and the Section 9 dead-band.

The point of this section is that a reader learns whether a difference is worth
caring about before reading any of the detail. These tests pin the three things
that would silently break that: the axes being labelled at all, the detail axis
*not* being labelled (it has no valid reference), and the dead-band agreeing
with the label rather than drifting from it.
"""
from __future__ import annotations

import pytest

from core.models import AnalysisResult
from core.practical_significance import MATERIAL, NONE, NOTICEABLE, SUBTLE
from report.report_builder import ReportBuilder


def _result(label, fwhm_px=None, fwhm_mad=0.1, snr=None, fwhm_arcsec=None,
            localmax=None):
    r = AnalysisResult(label=label)
    r.psf_metrics = {"fwhm_px": fwhm_px, "fwhm_px_mad": fwhm_mad,
                     "fwhm_arcsec": fwhm_arcsec, "fwhm_arcsec_mad": 0.1}
    r.snr_metrics = {"snr_global": snr}
    r.spatial_metrics = {"localmax": localmax or {}}
    return r


def _builder(single=False):
    b = ReportBuilder()
    b._single_image = single
    return b


def _html(fwhm_a, fwhm_b, snr_a, snr_b, **kw):
    return _builder()._section_verdict(
        _result("A", fwhm_px=fwhm_a, snr=snr_a, **kw),
        _result("B", fwhm_px=fwhm_b, snr=snr_b, **kw))


class TestSectionRenders:
    def test_emits_an_unnumbered_heading(self):
        """Unnumbered on purpose: inserting a numbered section would renumber
        every section after it and invalidate the cross-references scattered
        through the report's own prose."""
        html = _html(2.0, 3.0, 10.0, 10.0)
        assert "<h2>Executive Summary</h2>" in html
        assert "<h2>1." not in html

    def test_absent_in_single_image_mode(self):
        html = _builder(single=True)._section_verdict(
            _result("A", fwhm_px=2.0, snr=10.0), _result("B"))
        assert html == ""

    def test_names_all_three_axes(self):
        html = _html(2.0, 2.1, 10.0, 10.2)
        for axis in ("Sharpness", "Noise", "Detail"):
            assert f"<strong>{axis}</strong>" in html

    def test_explains_the_labels(self):
        assert "How to read these labels" in _html(2.0, 2.1, 10.0, 10.2)


def _table_only(html):
    """The table and headline, excluding the explanatory info box — which
    legitimately names every label and so cannot be asserted against."""
    return html.split("<details")[0]


class TestAxisLabelling:
    def test_material_fwhm_change_is_labelled_material(self):
        assert MATERIAL in _table_only(_html(2.0, 3.0, 10.0, 10.0))

    def test_quiet_pair_is_labelled_no_visible_difference(self):
        table = _table_only(_html(2.000, 2.010, 10.0, 10.05))
        assert NONE in table
        assert MATERIAL not in table

    def test_headline_is_the_strongest_axis_not_an_average(self):
        """Sharpness flat, noise material -- the headline must follow the noise
        axis. Averaging would bury exactly the finding worth surfacing."""
        html = _html(2.0, 2.01, 10.0, 30.0)
        assert f"<strong>Overall: {MATERIAL}</strong>" in html

    def test_headline_ignores_the_unlabelled_detail_axis(self):
        html = _html(2.0, 2.01, 10.0, 10.02)
        assert f"<strong>Overall: {NONE}</strong>" in html

    def test_sharpness_uses_pixels_so_a_bad_pixel_scale_cannot_break_it(self):
        """The arcsec value depends on a FITS header some software writes
        wrongly; the percentage change is identical in either unit."""
        html = _builder()._section_verdict(
            _result("A", fwhm_px=2.0, fwhm_arcsec=1.0, snr=10.0),
            _result("B", fwhm_px=3.0, fwhm_arcsec=9999.0, snr=10.0),
            scale_known=False)
        assert MATERIAL in html
        assert "9999" not in html

    def test_arcsec_shown_as_context_when_the_scale_is_trustworthy(self):
        html = _builder()._section_verdict(
            _result("A", fwhm_px=2.0, fwhm_arcsec=1.0, snr=10.0),
            _result("B", fwhm_px=3.0, fwhm_arcsec=1.5, snr=10.0),
            scale_known=True)
        assert "arcsec" in html


class TestDetailAxisIsHonest:
    """Section 8's dead-band would need a null floor, and that floor does not
    transfer between datasets. The report must say so rather than borrow one."""

    def test_detail_axis_carries_no_label(self):
        html = _html(2.0, 2.01, 10.0, 10.02,
                     localmax={"std_3px": {"log_ratio_mean": 0.9}})
        assert "not established" in html

    def test_detail_magnitude_is_still_reported(self):
        html = _html(2.0, 2.01, 10.0, 10.02, localmax={
            "std_3px": {"log_ratio_mean": 0.9},
            "log_1.5": {"log_ratio_mean": -0.2}})
        assert "-0.200" in html and "+0.900" in html

    def test_a_huge_detail_difference_does_not_drive_the_headline(self):
        """The failure this guards: a large uncalibrated detail reading must not
        be silently promoted to a verdict."""
        html = _html(2.0, 2.005, 10.0, 10.01,
                     localmax={"std_3px": {"log_ratio_mean": 5.0}})
        assert f"<strong>Overall: {NONE}</strong>" in html

    def test_says_why_no_label_is_given(self):
        html = _html(2.0, 2.01, 10.0, 10.02)
        assert "null reference" in html or "does not transfer" in html


class TestMissingMetricsDegradeGracefully:
    def test_no_psf_metrics(self):
        html = _html(None, None, 10.0, 12.0)
        assert "<h2>Executive Summary</h2>" in html

    def test_no_metrics_at_all(self):
        html = _builder()._section_verdict(_result("A"), _result("B"))
        assert "<h2>Executive Summary</h2>" in html

    def test_starless_snr_preferred_when_both_sides_have_it(self):
        ra, rb = _result("A", fwhm_px=2.0, snr=9.0), _result("B", fwhm_px=2.0, snr=9.0)
        ra.snr_metrics["starless"] = {"snr_global": 4.0}
        rb.snr_metrics["starless"] = {"snr_global": 16.0}
        html = _builder()._section_verdict(ra, rb)
        assert "starless" in html
        assert MATERIAL in html          # 4 -> 16 is +12 dB


class TestSectionNineDeadBand:
    """The dead-band must be driven by the same verdict the summary shows, so a
    cell can never be green while the summary calls it invisible."""

    @staticmethod
    def _summary(fwhm_a, fwhm_b, snr_a, snr_b):
        return _builder()._section_summary(
            _result("A", fwhm_px=fwhm_a, snr=snr_a),
            _result("B", fwhm_px=fwhm_b, snr=snr_b),
            bw_differ=False)

    def test_sub_threshold_fwhm_renders_neutral(self):
        html = self._summary(2.000, 2.010, 10.0, 10.0)   # 0.5%, invisible
        assert "class='neutral'" in html

    def test_material_fwhm_still_renders_better_worse(self):
        html = self._summary(2.0, 3.0, 10.0, 10.0)
        assert "class='better'" in html or "class='worse'" in html

    def test_legend_explains_grey(self):
        html = self._summary(2.0, 2.01, 10.0, 10.0)
        assert "Grey cells" in html

    def test_legend_notes_ellipticity_and_eccentricity_are_one_statistic(self):
        html = self._summary(2.0, 2.5, 10.0, 11.0)
        assert "one measurement, not two" in html


class TestSection8lCiColumn:
    """Section 8l reports an honest interval instead of a saturated p-value."""

    @staticmethod
    def _rows(entry):
        from report.report_builder import _localmax_rows
        return _localmax_rows({"std_3px": entry}, [("std_3px", "Local sigma 3 px")])

    def test_ci_is_rendered(self):
        html = self._rows({"mean_a": 1.0, "mean_b": 2.0, "std_a": .1, "std_b": .1,
                           "log_ratio_mean": -0.3, "log_ratio_std": 0.2,
                           "ci_lo": -0.35, "ci_hi": -0.25, "ci_converged": True,
                           "n_px": 1000, "pct_area": 5.0})
        assert "[-0.3500, -0.2500]" in html

    def test_non_convergence_is_disclosed(self):
        """A still-widening interval is a lower bound on the uncertainty, and
        saying nothing would present it as settled."""
        html = self._rows({"mean_a": 1.0, "mean_b": 2.0, "std_a": .1, "std_b": .1,
                           "log_ratio_mean": -0.3, "log_ratio_std": 0.2,
                           "ci_lo": -0.5, "ci_hi": -0.1, "ci_converged": False,
                           "n_px": 1000, "pct_area": 5.0})
        assert "still widening" in html

    def test_missing_ci_degrades_to_a_dash(self):
        html = self._rows({"mean_a": 1.0, "mean_b": 2.0, "std_a": .1, "std_b": .1,
                           "log_ratio_mean": -0.3, "log_ratio_std": 0.2,
                           "ci_lo": None, "ci_hi": None, "ci_converged": None,
                           "n_px": 1000, "pct_area": 5.0})
        assert "<td>—</td>" in html

    def test_no_star_rating_or_p_value_in_the_row(self):
        html = self._rows({"mean_a": 1.0, "mean_b": 2.0, "std_a": .1, "std_b": .1,
                           "log_ratio_mean": -0.3, "log_ratio_std": 0.2,
                           "ci_lo": -0.35, "ci_hi": -0.25, "ci_converged": True,
                           "p_value": 1e-9, "cliffs_delta": 0.8,
                           "n_px": 1000, "pct_area": 5.0})
        assert "&#9733;" not in html and "p&lt;" not in html
