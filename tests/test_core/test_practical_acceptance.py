"""End-to-end acceptance: the six real comparisons must be labelled the way
their owner reads them.

This is the test the whole practical-significance effort exists to pass. The
inputs are measurements read out of the six reports in
AstroLabTestData/FilterCompare; the expected column is what the person who took
the images says about each pair. If this fails, the labelling scheme is wrong,
not the expectations.

Three independent axes are combined with a max, because a difference that is
material on any one of them is material to the viewer:

    sharpness  -> PSF FWHM change (%)
    noise      -> global SNR change (dB)
    detail     -> Section 8 consensus over the top-5% population

Section 8 log-ratios here are the **top-5%** (local-maxima mask) values, not
whole-frame means. That distinction is load-bearing and was supplied by the
owner: perceived sharpness lives in the brightest structure, and on the bulk
mean a visibly-sharpened pair scores *below* a possibly-null filter pair.

Runs against the shipped resources/metric_calibration.json, so it also guards
that file -- regenerating the calibration must not silently break the six known
answers. Without a calibration only the FWHM/SNR currency is checked, which
needs none.
"""
from __future__ import annotations

import pytest

from core.practical_significance import (
    MATERIAL,
    NONE,
    NOTICEABLE,
    PRACTICAL_ORDER,
    SUBTLE,
    Calibration,
    consensus_label,
    verdict_for_fwhm,
    verdict_for_metric,
    verdict_for_snr,
)

# case -> (fwhm_a, fwhm_b, snr_a, snr_b, {metric: top5_log_ratio}, expected)
CASES = {
    "SCT Ha6/12 (owner: may or may not differ)": (
        8.989, 8.631, 4.7934, 4.3575,
        {"std_3px": -0.0019, "std_5px": 0.01193, "std_10px": 0.0021,
         "log_1.5": -0.1883, "log_3.0": 0.11885, "log_6.0": 0.02716,
         "gradient_1.5": 0.04347, "wavelet_2": 0.0145, "wavelet_3": 0.0606,
         "localgrad_1.5": 0.06971, "localgrad_3.0": 0.06709,
         "localgrad_6.0": 0.07163, "loclap_1.5": 0.0869,
         "loclap_3.0": 0.05195, "loclap_6.0": 0.05893},
        {NONE, SUBTLE}),
    "HyperHa 6/12": (
        3.698, 3.179, 5.1025, 5.0015,
        {"std_3px": 0.03078, "std_5px": 0.0262, "std_10px": 0.01897,
         "log_1.5": 0.02565, "log_3.0": -0.00143, "log_6.0": -0.01068,
         "gradient_1.5": -0.00775, "wavelet_2": 0.00819, "wavelet_3": 0.00823,
         "localgrad_1.5": 0.00847, "localgrad_3.0": -0.02074,
         "localgrad_6.0": -0.03, "loclap_1.5": 0.07106,
         "loclap_3.0": 0.04999, "loclap_6.0": -0.00956},
        {SUBTLE}),
    "Rosette CXB/SV (owner: SV better, how much?)": (
        2.296, 2.349, 11.2445, 14.3746,
        {"std_3px": 0.03239, "std_5px": 0.02075, "std_10px": 0.01614,
         "log_1.5": 0.02971, "log_3.0": 0.01257, "log_6.0": 0.01482,
         "gradient_1.5": 0.01591, "wavelet_2": 0.04984, "wavelet_3": 0.00649,
         "localgrad_1.5": 0.02941, "localgrad_3.0": 0.02672,
         "localgrad_6.0": 0.02628, "loclap_1.5": 0.04557,
         "loclap_3.0": 0.02411, "loclap_6.0": 0.02497},
        {SUBTLE, NOTICEABLE}),
    "OIII CXB/Opt": (
        2.208, 2.218, 9.9475, 9.3959,
        {"std_3px": -0.1262, "std_5px": -0.08882, "std_10px": -0.04888,
         "log_1.5": 0.00546, "log_3.0": -0.01797, "log_6.0": -0.01629,
         "gradient_1.5": -0.01529, "wavelet_2": -0.03995, "wavelet_3": -0.02634,
         "localgrad_1.5": -0.05095, "localgrad_3.0": -0.03577,
         "localgrad_6.0": -0.03804, "loclap_1.5": -0.14663,
         "loclap_3.0": -0.04918, "loclap_6.0": -0.02962},
        {NONE, SUBTLE}),
    "CXB crop: deconvolution sharpen (VISIBLE)": (
        1.919, 1.121, 9.7403, 4.8199,
        {"std_3px": -0.18334, "std_5px": -0.13523, "std_10px": -0.1014,
         "log_1.5": -0.09496, "log_3.0": -0.11425, "log_6.0": -0.05697,
         "gradient_1.5": -0.10186, "wavelet_2": -0.06883, "wavelet_3": -0.11993,
         "localgrad_1.5": -0.21964, "localgrad_3.0": -0.13162,
         "localgrad_6.0": -0.06683, "loclap_1.5": -0.24975,
         "loclap_3.0": -0.25421, "loclap_6.0": -0.15388},
        {MATERIAL}),
    "CXB crop: gaussian blur (VISIBLE)": (
        1.867, 5.151, 8.7668, 5.6251,
        {"std_3px": 0.52988, "std_5px": 0.25882, "std_10px": 0.05614,
         "log_1.5": 0.42554, "log_3.0": 0.00178, "log_6.0": -0.05017,
         "gradient_1.5": -0.01974, "wavelet_2": 0.79306, "wavelet_3": 0.19651,
         "localgrad_1.5": 0.04073, "localgrad_3.0": -0.08475,
         "localgrad_6.0": -0.13867, "loclap_1.5": 0.86938,
         "loclap_3.0": 0.12907, "loclap_6.0": -0.08011},
        {MATERIAL}),
}

VISIBLE = [n for n in CASES if "VISIBLE" in n]
QUIET = [n for n in CASES if "VISIBLE" not in n]


def _section8(ratios, cal):
    """Median over the top-5% population. Calibration is keyed per population,
    so a top-5% reading is never interpreted through a bulk-derived curve."""
    return consensus_label(
        [verdict_for_metric(f"{k}@top5", v, cal) for k, v in ratios.items()])


def _headline(fwhm_a, fwhm_b, snr_a, snr_b, ratios, cal):
    labels = [verdict_for_fwhm(fwhm_a, fwhm_b).label,
              verdict_for_snr(snr_a, snr_b).label,
              _section8(ratios, cal)]
    ranks = [PRACTICAL_ORDER[x] for x in labels if x in PRACTICAL_ORDER]
    if not ranks:
        return None
    return next(k for k, v in PRACTICAL_ORDER.items() if v == max(ranks))


@pytest.fixture(scope="module")
def cal():
    return Calibration.load()


def _need_cal(cal):
    if not cal.metrics:
        pytest.skip("no calibration; run tools/sensitivity_sweep.py blur-grid")


class TestHeadlineMatchesOwnerReading:
    @pytest.mark.parametrize("name", list(CASES))
    def test_case(self, name, cal):
        _need_cal(cal)
        fa, fb, sa, sb, ratios, acceptable = CASES[name]
        assert _headline(fa, fb, sa, sb, ratios, cal) in acceptable, name

    def test_only_the_deliberately_degraded_pairs_are_material(self, cal):
        _need_cal(cal)
        material = [n for n, v in CASES.items()
                    if _headline(*v[:5], cal) == MATERIAL]
        assert sorted(material) == sorted(VISIBLE), material

    def test_no_quiet_pair_reaches_material(self, cal):
        _need_cal(cal)
        for name in QUIET:
            h = _headline(*CASES[name][:5], cal)
            assert PRACTICAL_ORDER[h] < PRACTICAL_ORDER[MATERIAL], name


class TestAxisAttribution:
    """The headline must not merely be right -- it must be right for the right
    reason, or the executive summary will tell the user to look in the wrong
    place."""

    def test_rosette_advantage_is_attributed_to_snr_not_sharpness(self):
        """Owner's reading: SV is genuinely better, mostly in SNR. Sharpness is
        flat (+2.3% FWHM), so a headline built without an SNR axis reports this
        pair as no different at all."""
        fa, fb, sa, sb, _, _ = CASES["Rosette CXB/SV (owner: SV better, how much?)"]
        snr = verdict_for_snr(sa, sb)
        fwhm = verdict_for_fwhm(fa, fb)
        assert snr.rank > fwhm.rank
        assert snr.equivalent > 0        # B (SV) is the better one

    def test_blur_and_sharpen_are_material_on_the_sharpness_axis_alone(self):
        for name in VISIBLE:
            fa, fb = CASES[name][:2]
            assert verdict_for_fwhm(fa, fb).label == MATERIAL, name

    def test_hyperha_is_a_sharpness_difference_not_a_noise_one(self):
        fa, fb, sa, sb, _, _ = CASES["HyperHa 6/12"]
        assert verdict_for_fwhm(fa, fb).rank > verdict_for_snr(sa, sb).rank


class TestTopFivePercentIsWhatSeesSharpening:
    """The owner's finding, pinned.

    On whole-frame means the deconvolution-sharpened pair moves Section 8 *less*
    than the possibly-null SCT pair does on std_3px (-0.0163 vs -0.0198), so the
    detail axis reports nothing. Restricted to the brightest 5% it separates
    them by an order of magnitude and the detail axis sees the sharpening.
    """

    BULK_STD3 = {"sharpen": -0.0163, "sct": -0.0198}

    def test_bulk_mean_ranks_them_backwards(self):
        assert abs(self.BULK_STD3["sharpen"]) < abs(self.BULK_STD3["sct"])

    def test_top5_ranks_them_correctly(self):
        sharpen = abs(CASES["CXB crop: deconvolution sharpen (VISIBLE)"][4]["std_3px"])
        sct = abs(CASES["SCT Ha6/12 (owner: may or may not differ)"][4]["std_3px"])
        assert sharpen > 10 * sct, (sharpen, sct)

    def test_section8_now_sees_the_sharpening(self, cal):
        _need_cal(cal)
        ratios = CASES["CXB crop: deconvolution sharpen (VISIBLE)"][4]
        assert _section8(ratios, cal) == MATERIAL


class TestCurrencyOnlyPathNeedsNoCalibration:
    """FWHM and SNR label without any sweep having run, which is what makes a
    fresh checkout useful before calibration exists."""

    @pytest.mark.parametrize("name", list(CASES))
    def test_currency_alone_never_calls_a_quiet_pair_material(self, name):
        fa, fb, sa, sb, _, acceptable = CASES[name]
        if MATERIAL in acceptable:
            return
        for v in (verdict_for_fwhm(fa, fb), verdict_for_snr(sa, sb)):
            assert PRACTICAL_ORDER[v.label] < PRACTICAL_ORDER[MATERIAL], name
