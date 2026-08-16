"""Tests for core.practical_significance.

The important tests here are TestAgainstKnownRealCases: the six comparisons in
AstroLabTestData/FilterCompare have expected answers supplied by the person who
took the images, so the label thresholds can be checked against reality rather
than against themselves. If those fail, the thresholds are wrong -- not the test.
"""
from __future__ import annotations

import json

import pytest

from core.practical_significance import (
    MATERIAL,
    NONE,
    NOTICEABLE,
    PRACTICAL_LABELS,
    SUBTLE,
    Calibration,
    PracticalVerdict,
    consensus_label,
    fwhm_change_pct,
    label_for_fwhm_change,
    label_for_snr_change_db,
    overall_label,
    snr_change_db,
    verdict_for_fwhm,
    verdict_for_metric,
    verdict_for_snr,
)


class TestFwhmLabels:
    @pytest.mark.parametrize("pct,expected", [
        (0.0, NONE), (2.0, NONE), (4.9, NONE),
        (5.0, SUBTLE), (10.0, SUBTLE), (14.9, SUBTLE),
        (15.0, NOTICEABLE), (25.0, NOTICEABLE), (29.9, NOTICEABLE),
        (30.0, MATERIAL), (176.0, MATERIAL),
    ])
    def test_bands(self, pct, expected):
        assert label_for_fwhm_change(pct) == expected

    def test_sign_does_not_change_the_label(self):
        """Sharper and blurrier by the same amount are equally visible."""
        assert label_for_fwhm_change(-42.0) == label_for_fwhm_change(42.0)

    def test_none_and_nan_return_none(self):
        assert label_for_fwhm_change(None) is None
        assert label_for_fwhm_change(float("nan")) is None


class TestSnrLabels:
    @pytest.mark.parametrize("db,expected", [
        (0.0, NONE), (0.17, NONE), (0.49, NONE),
        (0.5, SUBTLE), (0.83, SUBTLE), (1.49, SUBTLE),
        (1.5, NOTICEABLE), (2.14, NOTICEABLE),
        (3.0, MATERIAL), (12.0, MATERIAL),
    ])
    def test_bands(self, db, expected):
        assert label_for_snr_change_db(db) == expected

    def test_none_returns_none(self):
        assert label_for_snr_change_db(None) is None


class TestCurrencyConversions:
    def test_fwhm_change_is_a_percentage_from_a_to_b(self):
        assert fwhm_change_pct(2.0, 3.0) == pytest.approx(50.0)
        assert fwhm_change_pct(2.0, 1.0) == pytest.approx(-50.0)

    def test_fwhm_change_guards_bad_input(self):
        assert fwhm_change_pct(None, 2.0) is None
        assert fwhm_change_pct(0.0, 2.0) is None

    def test_snr_db_uses_the_amplitude_convention(self):
        """20*log10, not 10*log10 -- SNR is amplitude-like. A factor of 10 is
        20 dB; the power convention would give 10 and be silently 2x wrong."""
        assert snr_change_db(1.0, 10.0) == pytest.approx(20.0)

    def test_snr_db_sign(self):
        assert snr_change_db(10.0, 5.0) < 0

    def test_snr_db_guards_nonpositive(self):
        assert snr_change_db(0.0, 5.0) is None
        assert snr_change_db(5.0, -1.0) is None


class TestUncalibratedMetrics:
    def test_unknown_metric_gets_no_label_rather_than_a_guess(self):
        v = verdict_for_metric("std_3px", 0.5, Calibration.empty())
        assert v.label is None
        assert v.calibrated is False
        assert "calibrat" in v.note

    def test_magnitude_is_still_reported(self):
        v = verdict_for_metric("std_3px", 0.5, Calibration.empty())
        assert v.magnitude == 0.5

    def test_missing_measurement_is_distinguished_from_uncalibrated(self):
        v = verdict_for_metric("std_3px", None, Calibration.empty())
        assert v.note == "no measurement"

    def test_short_form_is_a_dash_when_unlabelled(self):
        assert verdict_for_metric("x", 0.5, Calibration.empty()).short == "—"

    def test_rank_sorts_below_every_real_label(self):
        assert verdict_for_metric("x", 0.5, Calibration.empty()).rank == -1


class TestCalibration:
    @staticmethod
    def _cal():
        return Calibration(metrics={
            "std_3px": {"abs_log_ratio": [0.0, 0.01, 0.2, 0.7],
                        "fwhm_pct": [0.0, 6.0, 16.0, 54.0]},
        })

    def test_interpolates_between_measured_points(self):
        pct = self._cal().equivalent_fwhm_pct("std_3px", 0.105)
        assert 6.0 < pct < 16.0

    def test_clamps_rather_than_extrapolating_past_the_curve(self):
        """A ratio beyond the largest calibrated point is at least that large;
        extrapolating a steep non-linear curve would invent a number."""
        assert self._cal().equivalent_fwhm_pct("std_3px", 99.0) == pytest.approx(54.0)

    def test_uses_absolute_ratio_so_sign_is_symmetric(self):
        c = self._cal()
        assert (c.equivalent_fwhm_pct("std_3px", -0.2)
                == c.equivalent_fwhm_pct("std_3px", 0.2))

    def test_calibrated_metric_gets_a_label(self):
        v = verdict_for_metric("std_3px", 0.7, self._cal())
        assert v.calibrated and v.label == MATERIAL
        assert v.currency == "fwhm_pct"

    def test_small_ratio_on_a_calibrated_metric_is_none_label(self):
        assert verdict_for_metric("std_3px", 0.005, self._cal()).label == NONE

    def test_unknown_key_within_a_populated_calibration(self):
        assert verdict_for_metric("entropy_9px", 0.5, self._cal()).label is None

    def test_malformed_entry_yields_no_conversion(self):
        c = Calibration(metrics={"m": {"abs_log_ratio": [0.1], "fwhm_pct": [1.0]}})
        assert c.equivalent_fwhm_pct("m", 0.1) is None

    def test_mismatched_lengths_yield_no_conversion(self):
        c = Calibration(metrics={"m": {"abs_log_ratio": [0, 1], "fwhm_pct": [1.0]}})
        assert c.equivalent_fwhm_pct("m", 0.5) is None


class TestCalibrationLoading:
    def test_absent_file_is_not_an_error(self, tmp_path):
        c = Calibration.load(tmp_path / "nope.json")
        assert c.metrics == {} and c.source == "absent"

    def test_malformed_file_is_not_an_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        assert Calibration.load(p).source == "unreadable"

    def test_round_trip(self, tmp_path):
        p = tmp_path / "cal.json"
        p.write_text(json.dumps({
            "source": "unit-test",
            "metrics": {"std_3px": {"abs_log_ratio": [0, 1], "fwhm_pct": [0, 50]}},
        }), encoding="utf-8")
        c = Calibration.load(p)
        assert c.has("std_3px")
        assert c.equivalent_fwhm_pct("std_3px", 0.5) == pytest.approx(25.0)
        assert c.source == "unit-test"


class TestNullFloor:
    """The screen that monotonicity and dynamic range cannot replace."""

    @staticmethod
    def _cal(floor):
        return Calibration(metrics={"m": {
            "abs_log_ratio": [0.0, 0.01, 0.1, 0.5],
            "fwhm_pct": [0.0, 5.0, 30.0, 100.0],
            "null_floor": floor,
        }})

    def test_reading_at_the_floor_is_no_visible_difference(self):
        v = verdict_for_metric("m", 0.02, self._cal(0.02))
        assert v.label == NONE and "floor" in v.note

    def test_reading_below_the_floor_is_no_visible_difference(self):
        assert verdict_for_metric("m", 0.005, self._cal(0.02)).label == NONE

    def test_floor_is_subtracted_not_merely_gated(self):
        """A bare gate leaves a barely-detected reading mapping through a steep
        curve to a large label -- the SCT wavelet_3 failure (0.0100 against a
        0.0068 floor read as 16% FWHM, i.e. 'noticeable')."""
        gated = verdict_for_metric("m", 0.03, self._cal(0.02))
        ungated = verdict_for_metric("m", 0.03, self._cal(None))
        assert gated.equivalent < ungated.equivalent

    def test_large_reading_is_barely_affected_by_the_floor(self):
        with_floor = verdict_for_metric("m", 0.5, self._cal(0.01))
        without = verdict_for_metric("m", 0.5, self._cal(None))
        assert abs(with_floor.equivalent - without.equivalent) < 0.1 * without.equivalent

    def test_absent_floor_is_not_an_error(self):
        assert verdict_for_metric("m", 0.1, self._cal(None)).label is not None

    def test_note_records_the_correction(self):
        assert "floor-corrected" in verdict_for_metric("m", 0.3, self._cal(0.01)).note


class TestConsensusLabel:
    """Correlated estimates of one quantity reduce by median, not max."""

    @staticmethod
    def _v(label):
        return PracticalVerdict(label, True, "fwhm_pct", 1.0, 1.0)

    def test_single_outlier_does_not_set_the_verdict(self):
        """Ten of eleven SCT metrics said NONE and one weak detection said
        NOTICEABLE; max made that one the report's headline."""
        vs = [self._v(NONE)] * 10 + [self._v(NOTICEABLE)]
        assert consensus_label(vs) == NONE

    def test_majority_carries(self):
        vs = [self._v(MATERIAL)] * 7 + [self._v(NONE)] * 3
        assert consensus_label(vs) == MATERIAL

    def test_ignores_unlabelled(self):
        vs = [self._v(NONE), self._v(NONE),
              verdict_for_metric("x", 1.0, Calibration.empty())]
        assert consensus_label(vs) == NONE

    def test_empty(self):
        assert consensus_label([]) is None

    def test_differs_from_max_on_a_skewed_set(self):
        vs = [self._v(NONE)] * 5 + [self._v(MATERIAL)]
        assert consensus_label(vs) == NONE
        assert overall_label(vs) == MATERIAL


class TestOverallLabel:
    def test_takes_the_strongest_not_the_average(self):
        vs = [verdict_for_fwhm(2.0, 2.02),      # ~1%, none
              verdict_for_fwhm(2.0, 2.02),
              verdict_for_fwhm(2.0, 3.0)]       # 50%, material
        assert overall_label(vs) == MATERIAL

    def test_ignores_uncalibrated_entries(self):
        vs = [verdict_for_metric("x", 99.0, Calibration.empty()),
              verdict_for_fwhm(2.0, 2.02)]
        assert overall_label(vs) == NONE

    def test_none_when_nothing_is_labelled(self):
        assert overall_label([verdict_for_metric("x", 1.0, Calibration.empty())]) is None

    def test_empty_input(self):
        assert overall_label([]) is None


class TestAgainstKnownRealCases:
    """Thresholds checked against the six comparisons whose answers are known.

    Values are the measured PSF FWHM (px) and global SNR (sigma) read out of the
    generated reports in AstroLabTestData/FilterCompare.
    """

    # name: (fwhm_a, fwhm_b, acceptable_labels)
    #
    # SCT is deliberately a two-label set. Its owner's reading is "there may or
    # may not be a difference", and it measures -5.2% FWHM -- a hair over the
    # 5% boundary. Either NONE or SUBTLE is a faithful report of that, and
    # pushing the threshold past 5.24% to force NONE would be fitting a constant
    # to a single point at the cost of every other case's band.
    FWHM_CASES = {
        "OIII CXB/Opt":       (2.229, 2.289, {NONE}),
        "Rosette CXB/SV":     (2.313, 2.375, {NONE}),
        "SCT Ha6/12 (null)":  (8.845, 8.381, {NONE, SUBTLE}),
        "HyperHa 6/12":       (3.566, 3.140, {SUBTLE}),
        "deconv sharpen":     (1.989, 1.173, {MATERIAL}),
        "gaussian blur":      (1.903, 5.179, {MATERIAL}),
    }

    @pytest.mark.parametrize("name", list(FWHM_CASES))
    def test_fwhm_label_matches_the_owners_reading(self, name):
        fa, fb, acceptable = self.FWHM_CASES[name]
        assert verdict_for_fwhm(fa, fb).label in acceptable, name

    def test_no_case_the_owner_calls_quiet_is_labelled_noticeable_or_worse(self):
        quiet = ["OIII CXB/Opt", "Rosette CXB/SV", "SCT Ha6/12 (null)", "HyperHa 6/12"]
        for name in quiet:
            fa, fb, _ = self.FWHM_CASES[name]
            assert verdict_for_fwhm(fa, fb).rank <= 1, name

    def test_the_two_deliberately_degraded_pairs_are_the_only_material_ones(self):
        material = [n for n, (fa, fb, _) in self.FWHM_CASES.items()
                    if verdict_for_fwhm(fa, fb).label == MATERIAL]
        assert sorted(material) == ["deconv sharpen", "gaussian blur"]

    def test_visible_sharpen_outranks_the_possibly_null_filter_pair(self):
        """The failure mode that motivated the whole exercise: on the Section 8
        log-ratio metrics the visibly-sharpened pair moves *less* than the SCT
        pair. On the FWHM currency it must not."""
        sharpen = verdict_for_fwhm(1.989, 1.173)
        sct = verdict_for_fwhm(8.845, 8.381)
        assert sharpen.rank > sct.rank

    # name: (snr_a, snr_b, expected_label)
    SNR_CASES = {
        "HyperHa 6/12":      (5.1025, 5.0015, NONE),
        "OIII CXB/Opt":      (9.9475, 9.3959, NONE),
        "SCT Ha6/12":        (4.7934, 4.3575, SUBTLE),
        "Rosette CXB/SV":    (11.2445, 14.3746, NOTICEABLE),
    }

    @pytest.mark.parametrize("name", list(SNR_CASES))
    def test_snr_label_matches_the_owners_reading(self, name):
        sa, sb, expected = self.SNR_CASES[name]
        assert verdict_for_snr(sa, sb).label == expected, name

    def test_rosette_snr_favours_sv_and_is_the_strongest_of_the_four(self):
        """Owner's reading: SV has better SNR, significant but not dramatic."""
        v = verdict_for_snr(11.2445, 14.3746)
        assert v.equivalent > 0                      # B (SV) better
        others = [verdict_for_snr(*self.SNR_CASES[n][:2]).rank
                  for n in self.SNR_CASES if n != "Rosette CXB/SV"]
        assert v.rank > max(others)


class TestVerdictDataclass:
    def test_labels_tuple_is_ordered_least_to_most(self):
        assert PRACTICAL_LABELS == (NONE, SUBTLE, NOTICEABLE, MATERIAL)

    def test_short_forms_exist_for_every_label(self):
        for label in PRACTICAL_LABELS:
            assert PracticalVerdict(label, True, "fwhm_pct", 1.0, 1.0).short != "—"
