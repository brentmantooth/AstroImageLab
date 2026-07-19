"""Unit tests for analysis/image_filters.py (SpatialDetailAnalyzer)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as ap_fits

from analysis.image_filters import SpatialDetailAnalyzer
from core.astro_image import AstroImage
from core.models import STD_KERNEL_SIZES, LOG_SIGMAS, ENTROPY_KERNEL_SIZES, WAVELET_LEVELS


@pytest.fixture(scope="module")
def single_image_result(astro_image_a) -> dict:
    """Cached default-params single-image SpatialDetailAnalyzer result. Shared by
    every test in this file that only inspects an already-computed single-image
    result rather than exercising analyze() with unique parameters (roi, crosshair,
    monkeypatches) -- analyze() runs LoG/wavelet/local-sigma/entropy/local-maxima
    across multiple scales and is too expensive to re-run per assertion."""
    return SpatialDetailAnalyzer().analyze(astro_image_a)


class TestAnalyze:
    def test_returns_dict(self, single_image_result):
        assert isinstance(single_image_result, dict)

    def test_contrast_ratios_a_present(self, single_image_result):
        assert "contrast_ratios_a" in single_image_result

    def test_wavelet_snr_a_present(self, single_image_result):
        assert "wavelet_snr_a" in single_image_result

    def test_panels_present(self, single_image_result):
        assert "panels" in single_image_result

    def test_original_panel_present_single_image(self, single_image_result):
        original = single_image_result["panels"]["original"]
        assert original["a"] is not None
        assert original["b"] is None
        assert original["diff"] is None
        assert "original" in single_image_result["figures"]
        assert "corr_original" not in single_image_result["figures"]

    def test_contrast_ratios_are_positive(self, single_image_result):
        ratios = single_image_result.get("contrast_ratios_a") or []
        for r in ratios:
            if r is not None:
                assert r >= 0.0

    def test_wavelet_snr_finite(self, single_image_result):
        snrs = single_image_result.get("wavelet_snr_a") or []
        for s in snrs:
            if s is not None:
                assert np.isfinite(s)

    def test_single_image_b_ratios_empty(self, single_image_result):
        # Single-image mode: contrast_ratios_b is present but contains no values
        b_ratios = single_image_result.get("contrast_ratios_b")
        assert b_ratios is None or not b_ratios   # None or empty dict/list

    def test_minimal_image_no_crash(self, tmp_path):
        data = np.random.default_rng(99).normal(500, 10, (128, 128)).astype(np.float32)
        ap_fits.writeto(str(tmp_path / "tiny.fits"), data, overwrite=True)
        img = AstroImage(str(tmp_path / "tiny.fits"), label="Tiny")
        img.load()
        img.estimate_background()
        result = SpatialDetailAnalyzer().analyze(img)
        assert isinstance(result, dict)

    def test_entropy_contrast_ratio_a_present(self, single_image_result):
        assert "entropy_contrast_ratio_a" in single_image_result

    def test_single_image_entropy_contrast_ratio_b_empty(self, single_image_result):
        # Single-image mode: entropy_contrast_ratio_b is present but empty (same pattern as contrast_ratios_b)
        ecr_b = single_image_result.get("entropy_contrast_ratio_b")
        assert ecr_b is None or not ecr_b

    def test_entropy_contrast_ratio_a_positive(self, single_image_result):
        for v in single_image_result.get("entropy_contrast_ratio_a", {}).values():
            if v is not None:
                assert v >= 0.0

    def test_with_roi(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a,
                                                  roi=(50, 50, 450, 450))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Noise-corrected multi-scale local contrast (two-image A/B comparison)
# ---------------------------------------------------------------------------

def _make_nc_test_fits(path, add_texture: bool, seed: int) -> None:
    """256x256 FITS with a smooth nebula blob (sigma=25, well above 2*rms after
    background subtraction). When add_texture, a fine checkerboard (period 6px,
    amplitude 8x sky noise) is added inside the blob so std/LoG/wavelet/entropy/
    gradient all detect meaningfully more local structure than the plain blob."""
    rng = np.random.default_rng(seed)
    h, w = 256, 256
    sky_level = 1000.0
    sky_noise = 20.0
    data = rng.normal(sky_level, sky_noise, (h, w)).astype(np.float64)

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    blob_sigma = 25.0
    blob_amp = 600.0
    blob = blob_amp * np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / blob_sigma ** 2))
    data += blob

    if add_texture:
        texture_amp = 8.0 * sky_noise
        checker = (((xx // 3).astype(int) + (yy // 3).astype(int)) % 2 == 0)
        blob_mask = blob > (0.3 * blob_amp)
        data += np.where(blob_mask & checker, texture_amp, 0.0)

    data = np.clip(data, 0, 65535).astype(np.float32)
    hdr = ap_fits.Header()
    hdr["EGAIN"] = 1.0
    hdr["GAIN"] = 1.0
    hdr["FOCALLEN"] = 500.0
    hdr["XPIXSZ"] = 3.76
    hdr["EXPTIME"] = 300.0
    hdr["INSTRUME"] = "TestCam"
    ap_fits.writeto(str(path), data, hdr, overwrite=True)


@pytest.fixture(scope="module")
def nc_image_pair(tmp_path_factory) -> tuple[AstroImage, AstroImage]:
    """Image A = blob + fine checkerboard texture; Image B = blob only.
    Both share the same blob location/extent, so their nebula masks overlap."""
    out = tmp_path_factory.mktemp("nc_fits")
    path_a = out / "nc_a.fits"
    path_b = out / "nc_b.fits"
    _make_nc_test_fits(path_a, add_texture=True, seed=1)
    _make_nc_test_fits(path_b, add_texture=False, seed=2)
    img_a = AstroImage(str(path_a), label="A")
    img_a.load()
    img_a.estimate_background()
    img_b = AstroImage(str(path_b), label="B")
    img_b.load()
    img_b.estimate_background()
    return img_a, img_b


@pytest.fixture(scope="module")
def nc_result(nc_image_pair) -> dict:
    img_a, img_b = nc_image_pair
    return SpatialDetailAnalyzer().analyze(img_a, img_b)


class TestNoiseCorrectedContrast:
    """A/B noise-corrected multi-scale local contrast: shared-nebula-ROI scoring,
    per-scale A/B ratios, and noise-normalised display panels."""

    def test_shared_nebula_pixels_positive(self, nc_result):
        assert nc_result["nc_shared_nebula_pixels"] > 0

    @pytest.mark.parametrize("prefix,scales", [
        ("std", STD_KERNEL_SIZES),
        ("log", LOG_SIGMAS),
        ("entropy", ENTROPY_KERNEL_SIZES),
        ("gm", LOG_SIGMAS),
    ])
    def test_nc_score_dict_keys_match_scales(self, nc_result, prefix, scales):
        assert set(nc_result[f"{prefix}_nc_score_a"].keys()) == set(scales)
        assert set(nc_result[f"{prefix}_nc_score_b"].keys()) == set(scales)

    def test_wavelet_nc_score_keys_match_levels(self, nc_result):
        expected = set(range(1, WAVELET_LEVELS + 1))
        assert set(nc_result["wavelet_nc_score_a"].keys()) == expected
        assert set(nc_result["wavelet_nc_score_b"].keys()) == expected

    @pytest.mark.parametrize("prefix", ["std", "log", "wavelet", "entropy", "gm"])
    def test_nc_noise_floor_positive_or_none(self, nc_result, prefix):
        for side in ("a", "b"):
            for v in nc_result[f"{prefix}_nc_noise_{side}"].values():
                if v is not None:
                    assert v > 0.0

    def test_std_nc_ratio_captures_finer_detail_in_a(self, nc_result):
        ratio = nc_result["std_nc_ratio"][min(STD_KERNEL_SIZES)]
        assert ratio is not None and ratio > 1.05

    def test_log_nc_ratio_captures_finer_detail_in_a(self, nc_result):
        ratio = nc_result["log_nc_ratio"][min(LOG_SIGMAS)]
        assert ratio is not None and ratio > 1.05

    def test_wavelet_nc_ratio_captures_finer_detail_in_a(self, nc_result):
        ratio = nc_result["wavelet_nc_ratio"][2]
        assert ratio is not None and ratio > 1.05

    def test_entropy_nc_ratio_captures_finer_detail_in_a(self, nc_result):
        ratio = nc_result["entropy_nc_ratio"][min(ENTROPY_KERNEL_SIZES)]
        assert ratio is not None and ratio > 1.0

    def test_gradient_nc_ratio_captures_finer_detail_in_a(self, nc_result):
        # Gradient magnitude's peak response scale for a given texture need not be
        # the finest sigma (Gaussian smoothing at small sigma can attenuate a very
        # fine checkerboard before the derivative is taken) — check the strongest
        # response across scales rather than pinning to one bin.
        values = [v for v in nc_result["gm_nc_ratio"].values() if v is not None]
        assert values and max(values) > 1.05

    def test_normalized_panels_present_two_image(self, nc_result):
        panels = nc_result["panels"]
        for ks in STD_KERNEL_SIZES:
            assert f"nrm_std_{ks}px" in panels
        for sigma in LOG_SIGMAS:
            assert f"nrm_log_{sigma}" in panels
            assert f"nrm_gradient_{sigma}" in panels
        for ks in ENTROPY_KERNEL_SIZES:
            assert f"nrm_entropy_{ks}px" in panels
        for lvl in (2, 3):
            assert f"nrm_wavelet_{lvl}" in panels

    def test_original_panel_present_two_image(self, nc_result):
        original = nc_result["panels"]["original"]
        assert original["a"] is not None
        assert original["b"] is not None
        assert original["diff"] is not None
        assert original["a"].shape == original["b"].shape
        assert "original" in nc_result["figures"]
        assert "corr_original" in nc_result["figures"]

    def test_normalized_panel_values_differ_from_raw(self, nc_result):
        panels = nc_result["panels"]
        ks = min(STD_KERNEL_SIZES)
        raw_a = panels[f"std_{ks}px"]["a"]
        nrm_a = panels[f"nrm_std_{ks}px"]["a"]
        assert not np.allclose(raw_a, nrm_a)

    def test_nc_ratio_overview_figure_present(self, nc_result):
        assert "nc_ratio_overview" in nc_result["figures"]

    def test_with_roi_two_image_no_crash(self, nc_image_pair):
        img_a, img_b = nc_image_pair
        result = SpatialDetailAnalyzer().analyze(
            img_a, img_b, roi=(20, 20, 236, 236))
        assert isinstance(result, dict)
        assert "std_nc_ratio" in result


class TestNoiseCorrectedContrastSingleImage:
    """Single-image mode: every new NC key must be empty, matching the existing
    contrast_ratios_b / entropy_contrast_ratio_b invariant (never None/absent, never
    populated with placeholder values on the A side either)."""

    @pytest.fixture(scope="class")
    @classmethod
    def single_result(cls, single_image_result):
        return single_image_result

    @pytest.mark.parametrize("key", [
        "std_nc_score_a", "std_nc_score_b", "std_nc_ratio",
        "log_nc_score_a", "log_nc_score_b", "log_nc_ratio",
        "wavelet_nc_score_a", "wavelet_nc_score_b", "wavelet_nc_ratio",
        "entropy_nc_score_a", "entropy_nc_score_b", "entropy_nc_ratio",
        "gm_nc_score_a", "gm_nc_score_b", "gm_nc_ratio",
    ])
    def test_nc_key_empty_in_single_image_mode(self, single_result, key):
        assert not single_result[key]

    def test_shared_nebula_pixels_zero(self, single_result):
        assert single_result["nc_shared_nebula_pixels"] == 0

    def test_no_normalized_panels_in_single_image_mode(self, single_result):
        assert not any(k.startswith("nrm_") for k in single_result["panels"])


class TestNcScoreHelper:
    """Direct unit tests of _nc_score's mask-emptiness contract — deterministic,
    unlike relying on two real images happening to produce non-overlapping
    nebula masks (background-threshold noise can create incidental overlap)."""

    def test_none_mask_neb_shared_returns_none(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.ones((20, 20), dtype=np.float32)
        bg_mask = np.ones((20, 20), dtype=bool)
        score, noise, neb_std = analyzer._nc_score(detail, None, bg_mask)
        assert score is None and noise is None and neb_std is None

    def test_empty_shared_nebula_mask_returns_none(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.ones((20, 20), dtype=np.float32)
        mask_neb_shared = np.zeros((20, 20), dtype=bool)   # no shared nebula pixels
        bg_mask = np.ones((20, 20), dtype=bool)
        score, noise, neb_std = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score is None and noise is None and neb_std is None

    def test_empty_bg_mask_returns_none(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.ones((20, 20), dtype=np.float32)
        mask_neb_shared = np.ones((20, 20), dtype=bool)
        bg_mask = np.zeros((20, 20), dtype=bool)   # no background pixels
        score, noise, neb_std = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score is None and noise is None and neb_std is None

    def test_zero_noise_floor_returns_none(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.zeros((20, 20), dtype=np.float32)
        detail[:10, :] = 5.0   # nebula half has signal, background half is exactly 0
        mask_neb_shared = np.zeros((20, 20), dtype=bool)
        mask_neb_shared[:10, :] = True
        bg_mask = np.zeros((20, 20), dtype=bool)
        bg_mask[10:, :] = True
        score, noise, neb_std = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score is None and noise is None and neb_std is None

    def test_valid_masks_return_ratio(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.zeros((20, 20), dtype=np.float32)
        detail[:10, :] = 10.0
        detail[10:, :] = 2.0
        mask_neb_shared = np.zeros((20, 20), dtype=bool)
        mask_neb_shared[:10, :] = True
        bg_mask = np.zeros((20, 20), dtype=bool)
        bg_mask[10:, :] = True
        score, noise, neb_std = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score == pytest.approx(5.0)
        assert neb_std == pytest.approx(0.0)   # nebula region is a constant 10.0 block
        assert noise == pytest.approx(2.0)


class TestComputeNcRatioErrors:
    """Direct unit tests of _compute_nc_ratio_errors's CV-propagation contract."""

    def test_known_values_hand_computed(self):
        # median_neb_a = score_a * noise_a = 5.0 * 2.0 = 10.0; cv_a = 1.0/10.0 = 0.1
        # median_neb_b = score_b * noise_b = 3.0 * 2.0 = 6.0;  cv_b = 0.6/6.0 = 0.1
        ratio = {1: 5.0 / 3.0}
        score_a, score_b = {1: 5.0}, {1: 3.0}
        noise_a, noise_b = {1: 2.0}, {1: 2.0}
        neb_std_a, neb_std_b = {1: 1.0}, {1: 0.6}
        out = SpatialDetailAnalyzer._compute_nc_ratio_errors(
            ratio, score_a, score_b, noise_a, noise_b, neb_std_a, neb_std_b)
        expected = (5.0 / 3.0) * (0.1 ** 2 + 0.1 ** 2) ** 0.5
        assert out[1] == pytest.approx(expected, rel=1e-9)

    def test_missing_input_gives_none(self):
        ratio = {1: 1.5}
        score_a, score_b = {1: 5.0}, {}   # score_b missing for scale 1
        noise_a, noise_b = {1: 2.0}, {1: 2.0}
        neb_std_a, neb_std_b = {1: 1.0}, {1: 0.6}
        out = SpatialDetailAnalyzer._compute_nc_ratio_errors(
            ratio, score_a, score_b, noise_a, noise_b, neb_std_a, neb_std_b)
        assert out[1] is None

    def test_zero_median_neb_guard(self):
        ratio = {1: 0.0}
        score_a, score_b = {1: 0.0}, {1: 3.0}   # median_neb_a = 0*2.0 = 0
        noise_a, noise_b = {1: 2.0}, {1: 2.0}
        neb_std_a, neb_std_b = {1: 1.0}, {1: 0.6}
        out = SpatialDetailAnalyzer._compute_nc_ratio_errors(
            ratio, score_a, score_b, noise_a, noise_b, neb_std_a, neb_std_b)
        assert out[1] is None

    def test_empty_ratio_dict_returns_empty(self):
        out = SpatialDetailAnalyzer._compute_nc_ratio_errors({}, {}, {}, {}, {}, {}, {})
        assert out == {}


class TestLogRatioHelper:
    """Direct unit tests of _log_ratio_map's epsilon-floor and sign-discard contract."""

    def test_equal_inputs_give_zero(self):
        a = np.full((10, 10), 5.0, dtype=np.float32)
        b = np.full((10, 10), 5.0, dtype=np.float32)
        result = SpatialDetailAnalyzer._log_ratio_map(a, b)
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_double_ratio_gives_log10_two(self):
        b = np.full((10, 10), 3.0, dtype=np.float32)
        a = 2.0 * b
        result = SpatialDetailAnalyzer._log_ratio_map(a, b)
        assert np.allclose(result, np.log10(2.0), atol=1e-5)


class TestFillSmallHoles:
    """Direct unit tests of _fill_small_holes's area-limited hole-filling contract."""

    def test_fills_hole_within_size_limit(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.ones((20, 20), dtype=bool)
        mask[5:8, 5:8] = False   # 3x3 hole, area 9 <= 5**2
        result = analyzer._fill_small_holes(mask, max_hole_px=5)
        assert result[5:8, 5:8].all()

    def test_does_not_fill_hole_above_size_limit(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.ones((30, 30), dtype=bool)
        mask[5:15, 5:15] = False   # 10x10 hole, area 100 > 5**2
        result = analyzer._fill_small_holes(mask, max_hole_px=5)
        assert not result[10, 10]

    def test_zero_max_hole_px_is_noop(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.ones((10, 10), dtype=bool)
        mask[3:5, 3:5] = False
        result = analyzer._fill_small_holes(mask, max_hole_px=0)
        assert np.array_equal(result, mask)

    def test_hole_touching_border_is_not_filled(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.ones((20, 20), dtype=bool)
        mask[0:3, 0:3] = False   # touches the array border -> not an enclosed hole
        result = analyzer._fill_small_holes(mask, max_hole_px=5)
        assert not result[0:3, 0:3].any()


class TestRemoveSmallObjects:
    """Direct unit tests of _remove_small_objects's area-limited speck-removal
    contract — the fix for dilation amplifying isolated noise-driven pixels into
    large blobs (see TestMakeMasksGrowth.test_dilation_does_not_amplify_noise_specks)."""

    def test_removes_object_within_size_limit(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:8, 5:8] = True   # 3x3 speck, area 9 <= 5**2
        result = analyzer._remove_small_objects(mask, max_size_px=5)
        assert not result.any()

    def test_keeps_object_above_size_limit(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:15, 5:15] = True   # 10x10 object, area 100 > 5**2
        result = analyzer._remove_small_objects(mask, max_size_px=5)
        assert result[10, 10]

    def test_zero_max_size_px_is_noop(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:5, 3:5] = True
        result = analyzer._remove_small_objects(mask, max_size_px=0)
        assert np.array_equal(result, mask)

    def test_mixed_sizes_keeps_only_large_object(self):
        analyzer = SpatialDetailAnalyzer()
        mask = np.zeros((30, 30), dtype=bool)
        mask[2, 2] = True                 # isolated 1px speck
        mask[20:28, 20:28] = True         # 8x8 real object, area 64 > 25
        result = analyzer._remove_small_objects(mask, max_size_px=5)
        assert not result[2, 2]
        assert result[24, 24]


class TestLocalMaximaMask:
    """Direct unit tests of _local_maxima_mask's scale-adaptive peak detection,
    region growth, and noise-suppression contract."""

    def test_finds_single_isolated_peak(self):
        data = np.zeros((30, 30), dtype=np.float32)
        data[15, 15] = 100.0
        mask = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=5, prominence_percentile=90.0, region_px=0)
        assert mask[15, 15]
        assert mask.sum() == 1   # no region growth requested

    def test_footprint_scaling_separates_vs_merges_nearby_peaks(self):
        data = np.zeros((40, 40), dtype=np.float32)
        data[15, 15] = 80.0
        data[15, 19] = 100.0   # 4 px away, slightly taller
        small = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=3, prominence_percentile=10.0, region_px=0)
        large = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=9, prominence_percentile=10.0, region_px=0)
        assert small[15, 15] and small[15, 19]     # both survive with a tight footprint
        assert not large[15, 15] and large[15, 19]  # large footprint suppresses the weaker peak

    def test_prominence_percentile_filters_low_peaks(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 10, size=(30, 30)).astype(np.float32)
        data[15, 15] = 100.0   # one dominant spike
        loose = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=3, prominence_percentile=50.0, region_px=0)
        strict = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=3, prominence_percentile=99.5, region_px=0)
        assert loose.sum() > strict.sum()
        assert strict[15, 15]   # the dominant spike always survives

    def test_region_px_grows_mask_around_peak(self):
        data = np.zeros((30, 30), dtype=np.float32)
        data[15, 15] = 100.0
        none = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=5, prominence_percentile=90.0, region_px=0)
        grown = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=5, prominence_percentile=90.0, region_px=3)
        assert np.count_nonzero(grown) > np.count_nonzero(none)
        assert np.all(grown[none])   # superset of the ungrown mask

    def test_presmooth_suppresses_noise_driven_detections(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 5.0, size=(60, 60)).astype(np.float32)
        # Broad "real" feature: a Gaussian bump, amplitude well above the noise floor.
        yy, xx = np.mgrid[0:60, 0:60]
        bump = 60.0 * np.exp(-(((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * 3.0 ** 2)))
        data += bump.astype(np.float32)
        unsmoothed = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=3, prominence_percentile=90.0, region_px=0, presmooth_sigma=0.0)
        smoothed = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=3, prominence_percentile=90.0, region_px=0, presmooth_sigma=2.0)
        assert smoothed.sum() < unsmoothed.sum()   # fewer spurious noise-driven peaks
        assert smoothed[40, 40]   # the genuine broad feature still survives

    def test_flat_image_returns_empty_mask(self):
        data = np.full((20, 20), 5.0, dtype=np.float32)
        mask = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=5, prominence_percentile=90.0, region_px=2)
        assert not mask.any()

    def test_empty_array_returns_empty_mask(self):
        data = np.zeros((0, 0), dtype=np.float32)
        mask = SpatialDetailAnalyzer._local_maxima_mask(
            data, footprint_px=5, prominence_percentile=90.0, region_px=2)
        assert mask.shape == (0, 0)

    def test_footprint_forced_odd_and_minimum_three(self):
        data = np.zeros((20, 20), dtype=np.float32)
        data[10, 10] = 50.0
        for fp in (0, 2, 4):
            mask = SpatialDetailAnalyzer._local_maxima_mask(
                data, footprint_px=fp, prominence_percentile=90.0, region_px=0)
            assert mask[10, 10]   # no crash, still detects the obvious peak


class TestLocalMaxStats:
    """Direct unit tests of _localmax_stats's masked-region aggregation contract."""

    def test_empty_mask_returns_none_stats(self):
        abs_a = np.ones((10, 10), dtype=np.float32)
        abs_b = np.ones((10, 10), dtype=np.float32)
        diff = np.zeros((10, 10), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        rng = np.random.default_rng(0)
        stats = SpatialDetailAnalyzer._localmax_stats(abs_a, abs_b, diff, mask, rng)
        vals_a, vals_b = stats.pop("vals_a"), stats.pop("vals_b")
        vals_log_ratio = stats.pop("vals_log_ratio")
        assert stats == {"mean_a": None, "mean_b": None, "std_a": None, "std_b": None,
                          "ratio": None, "log_ratio_mean": None, "log_ratio_std": None,
                          "p_value": None, "cliffs_delta": None,
                          "n_px": 0, "pct_area": 0.0}
        assert vals_a.size == 0
        assert vals_b.size == 0
        assert vals_log_ratio.size == 0

    def test_known_values_give_expected_means_and_ratio(self):
        abs_a = np.full((10, 10), 4.0, dtype=np.float32)
        abs_b = np.full((10, 10), 2.0, dtype=np.float32)
        diff = np.full((10, 10), np.log10(2.0), dtype=np.float32)   # log10(4/2)
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True   # 9 px
        rng = np.random.default_rng(0)
        stats = SpatialDetailAnalyzer._localmax_stats(abs_a, abs_b, diff, mask, rng)
        assert stats["n_px"] == 9
        assert np.isclose(stats["mean_a"], 4.0)
        assert np.isclose(stats["mean_b"], 2.0)
        assert stats["std_a"] == 0.0   # constant array within the mask
        assert stats["std_b"] == 0.0
        assert np.isclose(stats["ratio"], 2.0, rtol=1e-5)
        assert stats["log_ratio_mean"] == pytest.approx(np.log10(2.0), rel=1e-5)
        assert np.isclose(10.0 ** stats["log_ratio_mean"], stats["ratio"])
        assert stats["log_ratio_std"] == pytest.approx(0.0)   # diff is constant within the mask
        assert np.isclose(stats["pct_area"], 9.0)   # 9 of 100 px
        assert stats["vals_a"].size == 9
        assert stats["vals_b"].size == 9
        assert stats["vals_log_ratio"].size == 9
        # A is uniformly higher than B within the mask -> fully separable.
        assert stats["p_value"] is not None and stats["p_value"] < 0.05
        assert stats["cliffs_delta"] is not None and stats["cliffs_delta"] > 0.9

    def test_log_ratio_std_reflects_varying_diff_within_mask(self):
        abs_a = np.full((10, 10), 4.0, dtype=np.float32)
        abs_b = np.full((10, 10), 2.0, dtype=np.float32)
        diff = np.zeros((10, 10), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True   # 9 px
        # Known, varying log-ratio values within the mask -> hand-computable std.
        diff_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        diff[2:5, 2:5] = diff_vals.reshape(3, 3)
        rng = np.random.default_rng(0)
        stats = SpatialDetailAnalyzer._localmax_stats(abs_a, abs_b, diff, mask, rng)
        assert stats["log_ratio_std"] == pytest.approx(float(np.std(diff_vals)), rel=1e-5)
        assert stats["log_ratio_std"] > 0.0

    def test_mismatched_shapes_crop_to_common_size(self):
        abs_a = np.full((12, 12), 4.0, dtype=np.float32)
        abs_b = np.full((10, 10), 2.0, dtype=np.float32)
        diff = np.full((10, 10), np.log10(2.0), dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        rng = np.random.default_rng(0)
        stats = SpatialDetailAnalyzer._localmax_stats(abs_a, abs_b, diff, mask, rng)
        assert stats["n_px"] == 100

    def test_vals_capped_at_dist_max_samples(self, monkeypatch):
        import analysis.image_filters as image_filters_module
        monkeypatch.setattr(image_filters_module, "SECTION8_LOCALMAX_DIST_MAX_SAMPLES", 10)
        abs_a = np.full((20, 20), 4.0, dtype=np.float32)
        abs_b = np.full((20, 20), 2.0, dtype=np.float32)
        diff = np.full((20, 20), np.log10(2.0), dtype=np.float32)
        mask = np.ones((20, 20), dtype=bool)   # 400 px, well above the patched cap
        rng = np.random.default_rng(0)
        stats = SpatialDetailAnalyzer._localmax_stats(abs_a, abs_b, diff, mask, rng)
        assert stats["n_px"] == 400   # true, uncapped count
        assert stats["vals_a"].size == 10
        assert stats["vals_b"].size == 10
        assert stats["vals_log_ratio"].size == 10


class TestTopPercentMask:
    """Direct unit tests of _top_percent_mask's OR-brightness-threshold contract."""

    def test_top_percent_selects_expected_count(self):
        # abs_b == abs_a makes the OR a no-op, isolating abs_a's own contribution
        # (a genuinely flat abs_b, e.g. all-zero, is a degenerate case: its own
        # 90th-percentile threshold equals its only value, so "abs_b >= threshold"
        # trivially matches every pixel and would swamp this assertion).
        abs_a = np.arange(100, dtype=np.float32).reshape(10, 10)
        abs_b = abs_a.copy()
        mask = SpatialDetailAnalyzer._top_percent_mask(abs_a, abs_b, top_percent=10.0)
        # Top 10% of a 0..99 uniform ramp is roughly the top 10 values.
        assert mask.sum() == pytest.approx(10, abs=2)
        assert mask[9, 9]   # the single brightest pixel (value 99) is always included

    def test_or_across_both_images(self):
        abs_a = np.zeros((10, 10), dtype=np.float32)
        abs_b = np.zeros((10, 10), dtype=np.float32)
        abs_a[0, 0] = 100.0   # bright only in A
        abs_b[9, 9] = 100.0   # bright only in B
        mask = SpatialDetailAnalyzer._top_percent_mask(abs_a, abs_b, top_percent=5.0)
        assert mask[0, 0] and mask[9, 9]

    def test_larger_top_percent_selects_more_pixels(self):
        rng = np.random.default_rng(0)
        abs_a = rng.uniform(0, 1, size=(30, 30)).astype(np.float32)
        abs_b = rng.uniform(0, 1, size=(30, 30)).astype(np.float32)
        loose = SpatialDetailAnalyzer._top_percent_mask(abs_a, abs_b, top_percent=20.0)
        strict = SpatialDetailAnalyzer._top_percent_mask(abs_a, abs_b, top_percent=2.0)
        assert loose.sum() > strict.sum()


class TestCombinedLocalMaxMask:
    """_combined_localmax_mask: local-maxima peak mask (_local_maxima_mask) unioned
    (OR) with a top-percent brightness mask (_top_percent_mask) -- catches both
    sharp isolated peaks and broad bright plateaus that peak detection alone
    would miss. Used identically by _localmax_entry and the mask-grid builder."""

    def test_union_includes_peak_only_pixel(self):
        abs_a = np.zeros((30, 30), dtype=np.float32)
        abs_a[15, 15] = 100.0   # isolated sharp peak
        abs_b = np.zeros((30, 30), dtype=np.float32)
        analyzer = SpatialDetailAnalyzer()
        mask = analyzer._combined_localmax_mask(
            abs_a, abs_b, footprint_px=5, prominence_percentile=90.0,
            region_px=0, presmooth_sigma=0.0, top_percent=1.0)
        assert mask[15, 15]

    def test_union_includes_top_percent_only_pixel(self):
        # A broad plateau that never registers as an isolated local maximum
        # (every pixel ties for "the max of its own neighbourhood"), but is
        # well within the top-percent brightness threshold.
        abs_a = np.zeros((30, 30), dtype=np.float32)
        abs_a[10:20, 10:20] = 50.0   # flat plateau, no single dominant peak
        abs_b = np.zeros((30, 30), dtype=np.float32)
        analyzer = SpatialDetailAnalyzer()
        mask = analyzer._combined_localmax_mask(
            abs_a, abs_b, footprint_px=5, prominence_percentile=99.9,
            region_px=0, presmooth_sigma=0.0, top_percent=50.0)
        assert mask[15, 15]   # inside the plateau, selected via top-percent only


class TestLocalMaxTopPercentWiring:
    """Confirms localmax_top_percent is actually threaded from analyze() through
    to each family's mask, not just accepted and discarded."""

    def test_higher_top_percent_does_not_shrink_masked_pixel_count(self, nc_image_pair):
        img_a, img_b = nc_image_pair
        low = SpatialDetailAnalyzer().analyze(img_a, img_b, localmax_top_percent=1.0)
        high = SpatialDetailAnalyzer().analyze(img_a, img_b, localmax_top_percent=40.0)
        # A looser top-percent threshold can only add pixels to the OR'd mask,
        # never remove any -- every scale/metric's n_px must be non-decreasing,
        # and at least one must strictly grow (proves the parameter reaches
        # the mask, not just accepted and ignored).
        assert all(
            high["localmax"][key]["n_px"] >= low["localmax"][key]["n_px"]
            for key in low["localmax"]
        )
        assert any(
            high["localmax"][key]["n_px"] > low["localmax"][key]["n_px"]
            for key in low["localmax"]
        )


class _FakeMaskImage:
    """Minimal duck-typed stand-in for AstroImage, exposing only what
    _make_masks reads (background_rms, background_subtracted())."""

    def __init__(self, data: np.ndarray, rms: np.ndarray):
        self._data = data
        self.background_rms = rms

    def background_subtracted(self) -> np.ndarray:
        return self._data


class TestMakeMasksGrowth:
    """Direct unit tests of _make_masks's dilation and nebula-dominance contract."""

    def _bright_square_image(self, size=40, lo=15, hi=25, value=100.0):
        data = np.zeros((size, size), dtype=np.float32)
        data[lo:hi, lo:hi] = value
        rms = np.full((size, size), 1.0, dtype=np.float32)
        return _FakeMaskImage(data, rms)

    def test_dilation_grows_nebula_mask(self):
        analyzer = SpatialDetailAnalyzer()
        image = self._bright_square_image()
        neb_none, _ = analyzer._make_masks(image, nebula_sigma=1.7, dilation_px=0, max_hole_px=0)
        neb_grown, _ = analyzer._make_masks(image, nebula_sigma=1.7, dilation_px=3, max_hole_px=0)
        assert np.count_nonzero(neb_grown) > np.count_nonzero(neb_none)
        assert np.all(neb_grown[neb_none])   # superset of the ungrown mask

    def test_background_excludes_dilated_nebula_pixels(self):
        analyzer = SpatialDetailAnalyzer()
        image = self._bright_square_image()
        neb, bg = analyzer._make_masks(image, nebula_sigma=1.7, dilation_px=3, max_hole_px=0)
        assert not np.any(neb & bg)   # masks stay mutually exclusive after growth
        assert neb[14, 20]   # grown one row above the bright square's top edge
        assert not bg[14, 20]   # nebula dominates: excluded from background

    def test_zero_dilation_matches_undilated_threshold(self):
        analyzer = SpatialDetailAnalyzer()
        image = self._bright_square_image()
        neb, _ = analyzer._make_masks(image, nebula_sigma=1.7, dilation_px=0, max_hole_px=0)
        expected = image.background_subtracted() > 1.7 * 1.0
        assert np.array_equal(neb, expected)

    def test_dilation_does_not_amplify_isolated_noise_specks(self):
        """Regression test: dilation must not inflate scattered single-pixel
        threshold-crossings (expected at a loose ~1.7 sigma cut) into large blobs
        far from any real nebula structure. _make_masks strips small islands via
        _remove_small_objects before dilating, so an isolated speck should vanish
        entirely rather than grow."""
        analyzer = SpatialDetailAnalyzer()
        size = 60
        data = np.zeros((size, size), dtype=np.float32)
        data[25:35, 25:35] = 100.0   # real nebula core
        data[5, 5] = 100.0            # isolated single-pixel noise-like speck
        rms = np.full((size, size), 1.0, dtype=np.float32)
        image = _FakeMaskImage(data, rms)
        neb, _ = analyzer._make_masks(image, nebula_sigma=1.7, dilation_px=3, max_hole_px=5)
        assert not neb[2:9, 2:9].any()   # speck stripped before it could be dilated
        assert neb[30, 30]   # the real core is still classified as nebula


class TestSharedMaskCombination:
    """mask_neb_shared changed from two-image AND to OR; mask_bg_shared stays AND.
    _make_masks is monkeypatched to return controlled, disjoint per-image masks so
    the combination formula in analyze() is tested in isolation from noise/threshold
    behaviour, following the _auto_detect_top_rois monkeypatch precedent in
    test_edge_analyzer.py."""

    def test_nebula_union_counts_pixels_unique_to_either_image(self, astro_image_a, monkeypatch):
        shape = astro_image_a.background_subtracted().shape

        neb_a = np.zeros(shape, dtype=bool)
        neb_a[0:10, 0:10] = True
        bg_a = np.ones(shape, dtype=bool)
        bg_a[0:10, 0:10] = False

        neb_b = np.zeros(shape, dtype=bool)
        neb_b[20:30, 20:30] = True   # disjoint from neb_a
        bg_b = np.ones(shape, dtype=bool)
        bg_b[20:30, 20:30] = False

        calls = iter([(neb_a, bg_a), (neb_b, bg_b)])
        monkeypatch.setattr(
            SpatialDetailAnalyzer, "_make_masks",
            lambda self, image, nebula_sigma=None, dilation_px=None, max_hole_px=None: next(calls))

        result = SpatialDetailAnalyzer().analyze(astro_image_a, astro_image_a)
        # Union of two disjoint 10x10 regions: under the old AND semantics this
        # would be 0 (no true overlap); under OR it's the full 200-pixel footprint.
        assert result["nc_shared_nebula_pixels"] == 200

    def test_opposite_sign_equal_magnitude_gives_zero(self):
        a = np.full((10, 10), -5.0, dtype=np.float32)
        b = np.full((10, 10), 5.0, dtype=np.float32)
        result = SpatialDetailAnalyzer._log_ratio_map(a, b)
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_near_zero_pixel_amid_nonzero_data_stays_finite(self):
        a = np.full((10, 10), 5.0, dtype=np.float32)
        b = np.full((10, 10), 5.0, dtype=np.float32)
        b[0, 0] = 0.0
        result = SpatialDetailAnalyzer._log_ratio_map(a, b)
        assert np.all(np.isfinite(result))

    def test_mismatched_shapes_crop_to_common_size(self):
        a = np.full((10, 10), 5.0, dtype=np.float32)
        b = np.full((8, 9), 5.0, dtype=np.float32)
        result = SpatialDetailAnalyzer._log_ratio_map(a, b)
        assert result.shape == (8, 9)

    def test_all_zero_inputs_stay_finite(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((10, 10), dtype=np.float32)
        result = SpatialDetailAnalyzer._log_ratio_map(a, b)
        assert np.all(np.isfinite(result))


class TestMaskIllustrationFigure:
    def test_absent_in_single_image_mode(self, single_image_result):
        assert "mask_illustration" not in single_image_result["figures"]

    def test_present_in_two_image_mode(self, nc_result):
        assert "mask_illustration" in nc_result["figures"]
        assert isinstance(nc_result["figures"]["mask_illustration"], str)
        assert len(nc_result["figures"]["mask_illustration"]) > 0


class TestCorrelationScatterFigures:
    _CORR_KEYS = (
        ["corr_original"]
        + [f"corr_std_{ks}px" for ks in STD_KERNEL_SIZES]
        + [f"corr_log_{s}" for s in LOG_SIGMAS]
        + [f"corr_gradient_{s}" for s in LOG_SIGMAS]
        + ["corr_wavelet_2", "corr_wavelet_3"]
        + [f"corr_entropy_{ks}px" for ks in ENTROPY_KERNEL_SIZES]
    )

    @pytest.mark.parametrize("key", _CORR_KEYS)
    def test_present_in_two_image_mode(self, nc_result, key):
        assert key in nc_result["figures"]
        assert isinstance(nc_result["figures"][key], str)
        assert len(nc_result["figures"][key]) > 0

    @pytest.mark.parametrize("key", _CORR_KEYS)
    def test_absent_in_single_image_mode(self, single_image_result, key):
        assert key not in single_image_result["figures"]


class TestLocalMaxIntegration:
    """Section 8j: result["localmax"] population across all metric/scale
    combinations, mirroring the corr_* key set in TestCorrelationScatterFigures."""

    _LOCALMAX_KEYS = (
        [f"std_{ks}px" for ks in STD_KERNEL_SIZES]
        + [f"log_{s}" for s in LOG_SIGMAS]
        + [f"gradient_{s}" for s in LOG_SIGMAS]
        + ["wavelet_2", "wavelet_3"]
        + [f"entropy_{ks}px" for ks in ENTROPY_KERNEL_SIZES]
    )

    @pytest.mark.parametrize("key", _LOCALMAX_KEYS)
    def test_entry_present_with_valid_stats(self, nc_result, key):
        entry = nc_result["localmax"][key]
        assert entry["n_px"] >= 0
        assert 0.0 <= entry["pct_area"] <= 100.0
        assert entry["ratio"] is None or entry["ratio"] > 0
        assert entry["std_a"] is None or entry["std_a"] >= 0
        assert entry["std_b"] is None or entry["std_b"] >= 0
        assert entry["p_value"] is None or 0.0 <= entry["p_value"] <= 1.0
        assert entry["cliffs_delta"] is None or -1.0 <= entry["cliffs_delta"] <= 1.0
        assert entry["vals_a"].size <= entry["n_px"]
        assert entry["vals_b"].size <= entry["n_px"]
        assert entry["vals_log_ratio"].size <= entry["n_px"]
        if entry["ratio"] is not None:
            assert entry["log_ratio_mean"] is not None
            assert np.isclose(10.0 ** entry["log_ratio_mean"], entry["ratio"])

    def test_empty_in_single_image_mode(self, single_image_result):
        assert single_image_result["localmax"] == {}


class TestLocalMaxFigures:
    def test_ratio_overview_present_in_two_image_mode(self, nc_result):
        assert "localmax_ratio_overview" in nc_result["figures"]

    def test_ratio_overview_absent_in_single_image_mode(self, single_image_result):
        assert "localmax_ratio_overview" not in single_image_result["figures"]

    def test_mask_illustration_present_in_two_image_mode(self, nc_result):
        assert "localmax_mask_illustration" in nc_result["figures"]
        assert isinstance(nc_result["figures"]["localmax_mask_illustration"], str)
        assert len(nc_result["figures"]["localmax_mask_illustration"]) > 0

    def test_mask_illustration_absent_in_single_image_mode(self, single_image_result):
        assert "localmax_mask_illustration" not in single_image_result["figures"]

    def test_mask_illustration_is_a_multi_row_grid(self, nc_result):
        # Regression guard: the old single-scale illustration was one ~9-inch-tall
        # panel (~1350px at dpi=150); the grid is 5 rows tall (~2700px) -- decode
        # the PNG and check its height reflects the multi-row layout, not the old
        # single-panel figure.
        import base64
        import io
        from PIL import Image
        png_bytes = base64.b64decode(nc_result["figures"]["localmax_mask_illustration"])
        img = Image.open(io.BytesIO(png_bytes))
        assert img.height > 2000

    def test_localmax_entries_carry_vals_for_distribution_figure(self, nc_result):
        # result["figures"] never stores the A/B distribution violin plot itself
        # (that's rendered in report_builder.py from result["localmax"][key]["vals_a"/"vals_b"]),
        # so verify those raw-value arrays are actually present in two-image mode.
        any_present = any(
            entry.get("vals_a") is not None and entry["vals_a"].size > 0
            for entry in nc_result["localmax"].values()
        )
        assert any_present

    def test_localmax_entries_carry_vals_log_ratio_for_distribution_figure(self, nc_result):
        # Same contract as vals_a/vals_b above, for the log-ratio distribution
        # figure (report_builder.py::_localmax_log_ratio_distribution_figure),
        # which reads result["localmax"][key]["vals_log_ratio"].
        any_present = any(
            entry.get("vals_log_ratio") is not None and entry["vals_log_ratio"].size > 0
            for entry in nc_result["localmax"].values()
        )
        assert any_present

    def test_mask_grid_default_color_is_high_contrast(self):
        # Regression guard: the mask overlay used to default to "darkorange",
        # which is hard to distinguish against bright nebula regions in the
        # grayscale-stretched base image. Confirm the more contrasting default.
        import inspect
        sig = inspect.signature(SpatialDetailAnalyzer._plot_localmax_mask_grid)
        assert sig.parameters["color"].default == "magenta"


@pytest.fixture(scope="module")
def nc_result_with_crosshair(nc_image_pair) -> dict:
    img_a, img_b = nc_image_pair
    crosshair = {"x0": 0.1, "y0": 0.5, "x1": 0.9, "y1": 0.5}
    return SpatialDetailAnalyzer().analyze(img_a, img_b, crosshair=crosshair)


class TestCrosshairEmbeddedCrossSections:
    """Section 8 cross-sections are embedded panels inside the 2x2 map figures,
    not separate figures — no standalone xs_* keys should ever appear, and all
    5 families (including Local entropy) must still produce their usual figure
    keys with or without a crosshair, without crashing."""

    def test_no_crash_with_crosshair(self, nc_result_with_crosshair):
        assert isinstance(nc_result_with_crosshair, dict)

    def test_no_standalone_xs_keys_with_crosshair(self, nc_result_with_crosshair):
        figs = nc_result_with_crosshair["figures"]
        assert not any(k.startswith(("xs_std_", "xs_log_", "xs_wavelet_",
                                      "xs_gradient_", "xs_entropy_")) for k in figs)

    def test_no_standalone_xs_keys_without_crosshair(self, nc_result):
        figs = nc_result["figures"]
        assert not any(k.startswith(("xs_std_", "xs_log_", "xs_wavelet_",
                                      "xs_gradient_", "xs_entropy_")) for k in figs)

    @pytest.mark.parametrize("key_prefix,scales", [
        ("std_", STD_KERNEL_SIZES), ("entropy_", ENTROPY_KERNEL_SIZES),
    ])
    def test_family_figures_present_with_crosshair(self, nc_result_with_crosshair,
                                                     key_prefix, scales):
        figs = nc_result_with_crosshair["figures"]
        for scale in scales:
            assert f"{key_prefix}{scale}px" in figs

    def test_entropy_no_crash_without_crosshair(self, nc_result):
        # Regression: weber (entropy's predecessor family) previously had no
        # crosshair param at all.
        assert all(f"entropy_{ks}px" in nc_result["figures"] for ks in ENTROPY_KERNEL_SIZES)

    def test_entropy_nrm_figures_present_with_crosshair(self, nc_result_with_crosshair):
        figs = nc_result_with_crosshair["figures"]
        assert any(k.startswith("nrm_entropy_") for k in figs)

    def test_original_present_with_crosshair(self, nc_result_with_crosshair):
        assert "original" in nc_result_with_crosshair["figures"]


class TestCrosshairToCroppedPx:
    """Unit tests for the helper that overlays the user's cross-section line
    directly onto the Image A/B map panels — converts a normalised [0,1]
    crosshair into pixel coords matching _crop_border's own offset math."""

    def test_none_crosshair_returns_none(self):
        assert SpatialDetailAnalyzer._crosshair_to_cropped_px(None, (200, 200), 0.05) is None

    def test_full_diagonal_line_cropped(self):
        # 200x200 array, 5% crop -> n = 10 px removed from each edge.
        crosshair = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
        result = SpatialDetailAnalyzer._crosshair_to_cropped_px(crosshair, (200, 200), 0.05)
        assert result == pytest.approx((-10.0, -10.0, 190.0, 190.0))

    def test_center_point_unaffected_by_crop_offset(self):
        crosshair = {"x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}
        x0, y0, x1, y1 = SpatialDetailAnalyzer._crosshair_to_cropped_px(
            crosshair, (200, 200), 0.05)
        assert x0 == pytest.approx(90.0)
        assert y0 == pytest.approx(90.0)
        assert (x0, y0) == pytest.approx((x1, y1))

    def test_small_array_below_crop_threshold_no_offset(self):
        # _crop_border only crops when shape > 2n; a tiny array stays uncropped,
        # so the returned pixel coords must have zero offset applied.
        crosshair = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
        result = SpatialDetailAnalyzer._crosshair_to_cropped_px(crosshair, (10, 10), 0.5)
        assert result == pytest.approx((0.0, 0.0, 10.0, 10.0))


class TestSectionSpatialReportOrder:
    """Integration check on report/report_builder.py::_section_spatial's HTML
    output: the reorganized subsection order (8a Background, 8b Original Image,
    8c Mask Overview, 8d LoG, 8e Wavelet, 8f Gradient, 8g Local Std,
    8h Local Entropy, 8i NC overview, 8j Local-Maxima Masked Metrics), and the
    fix for the alphabetic-sort bug that put e.g. nrm_std_10px before nrm_std_3px/5px."""

    @pytest.fixture(scope="class")
    @classmethod
    def section_html(cls, nc_result):
        from core.models import AnalysisResult
        from report.report_builder import ReportBuilder

        ra = AnalysisResult(label="A", spatial_metrics=nc_result)
        rb = AnalysisResult(label="B", spatial_metrics=nc_result)
        return ReportBuilder()._section_spatial(ra, rb)

    def test_headings_present_in_order(self, section_html):
        import re
        expected = ["8a", "8b", "8c", "8d", "8e", "8f", "8g", "8h", "8i", "8j"]
        found = re.findall(r"<h3>(8[a-j])\.", section_html)
        assert found == expected

    def test_original_image_section_present(self, section_html):
        assert "8b. Original Image" in section_html
        assert 'alt="original"' in section_html or "original" in section_html

    def test_local_std_now_in_contrast_group_after_gradient(self, section_html):
        assert section_html.index("8f. Gradient Magnitude") < section_html.index(
            "8g. Local Standard Deviation Maps")

    def test_entropy_after_local_std(self, section_html):
        assert section_html.index("8g. Local Standard Deviation Maps") < section_html.index(
            "8h. Local Entropy Maps")

    def test_nrm_std_figures_in_ascending_kernel_order(self, section_html):
        # Regression test: figs_for()'s old lexicographic sorted(figs) rendered
        # nrm_std_10px before nrm_std_3px/5px. _family_nrm_figs must not.
        positions = {
            ks: section_html.find(f'"nrm_std_{ks}px"')
            for ks in STD_KERNEL_SIZES
        }
        assert all(p >= 0 for p in positions.values())
        ordered = sorted(STD_KERNEL_SIZES)
        for a, b in zip(ordered, ordered[1:]):
            assert positions[a] < positions[b]

    def test_no_stale_letter_references(self, section_html):
        import re
        # Strip embedded base64 image data first — long random-looking base64
        # blobs incidentally contain "8<letter>" substrings that aren't section
        # references at all.
        prose = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", section_html)
        # Every literal "8<letter>" reference must be one of the 10 valid
        # letters — a stray old letter (e.g. a missed "8e" -> "8h" rename)
        # would show up here.
        for m in re.finditer(r"\b8([a-z])\b", prose):
            assert m.group(1) in "abcdefghij", f"unexpected section letter: 8{m.group(1)}"

    def test_8c_mask_overview_survives_violin_removal(self, section_html):
        # Regression test: 8c's HTML block used to be gated on the (now-removed)
        # violin figure's own output, which meant the Nebula/Background mask
        # illustration silently disappeared too if that figure was ever empty.
        # The heading and mask illustration must render on their own gating.
        assert "8c. Mask Overview" in section_html
        assert "Nebula / Background Mask Regions" in section_html

    def test_old_violin_caption_removed(self, section_html):
        # Direct regression test for the removal request: the old 8c
        # Nebula-vs-Background log-ratio violin figure's distinctive caption
        # text must no longer appear anywhere in the section. "ratio
        # distributions" alone is no longer a safe substring to check on its
        # own -- the (legitimate, newer) 8j log-ratio distribution figure's
        # caption also contains that phrase -- so match the old figure's
        # full distinctive title instead.
        assert "Pixel-wise log" not in section_html
        assert "Pixel-wise log₁₀(A / B) ratio distributions" not in section_html

    def test_localmax_distribution_figure_present(self, section_html):
        assert 'alt="localmax_distributions"' in section_html

    def test_old_mask_grid_illustrative_caption_removed(self, section_html):
        # Regression test: the old single-scale illustration's "not literally
        # the source of every row's statistics" disclaimer no longer applies
        # now that the grid shows every row's actual mask.
        assert "Illustrative only" not in section_html
        assert "single representative scale" not in section_html
        assert "smallest" in section_html and "largest" in section_html

    @pytest.mark.parametrize("title", [
        "Show LoG maps & figures",
        "Show Wavelet maps & figures",
        "Show Gradient maps & figures",
        "Show Local σ maps & figures",
        "Show Local entropy maps & figures",
    ])
    def test_family_images_are_collapsed_by_default(self, section_html, title):
        # Each of the 5 metric families' map/correlation/noise-normalised figures
        # (8d-8h) must be wrapped in a closed-by-default <details> box, keeping
        # the report compact. The heading/methodology/table above each family
        # stay outside the box (checked implicitly: the table for that family
        # still appears earlier in section_html than this collapsible summary).
        idx = section_html.find(f"<summary>{title}</summary>")
        assert idx > 0, f"collapsible box with title {title!r} not found"
        # The <details> tag immediately preceding this <summary> must not carry
        # an "open" attribute -- default closed.
        details_start = section_html.rfind("<details", 0, idx)
        assert details_start > 0
        details_tag = section_html[details_start:idx]
        assert " open" not in details_tag

    def test_localmax_log_ratio_distribution_figure_present(self, section_html):
        assert 'alt="localmax_log_ratio_distributions"' in section_html

    def test_log_ratio_table_column_present(self, section_html):
        assert "log ratio A/B (geo. mean" in section_html
        assert "<th>Ratio A/B (geo. mean)</th>" not in section_html
