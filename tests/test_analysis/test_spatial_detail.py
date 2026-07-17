"""Unit tests for analysis/image_filters.py (SpatialDetailAnalyzer)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as ap_fits

from analysis.image_filters import SpatialDetailAnalyzer
from core.astro_image import AstroImage
from core.models import STD_KERNEL_SIZES, LOG_SIGMAS, WEBER_KERNEL_SIZES, WAVELET_LEVELS


class TestAnalyze:
    def test_returns_dict(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert isinstance(result, dict)

    def test_contrast_ratios_a_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "contrast_ratios_a" in result

    def test_wavelet_snr_a_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "wavelet_snr_a" in result

    def test_panels_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "panels" in result

    def test_original_panel_present_single_image(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        original = result["panels"]["original"]
        assert original["a"] is not None
        assert original["b"] is None
        assert original["diff"] is None
        assert "original" in result["figures"]
        assert "corr_original" not in result["figures"]

    def test_contrast_ratios_are_positive(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        ratios = result.get("contrast_ratios_a") or []
        for r in ratios:
            if r is not None:
                assert r >= 0.0

    def test_wavelet_snr_finite(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        snrs = result.get("wavelet_snr_a") or []
        for s in snrs:
            if s is not None:
                assert np.isfinite(s)

    def test_single_image_b_ratios_empty(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        # Single-image mode: contrast_ratios_b is present but contains no values
        b_ratios = result.get("contrast_ratios_b")
        assert b_ratios is None or not b_ratios   # None or empty dict/list

    def test_minimal_image_no_crash(self, tmp_path):
        data = np.random.default_rng(99).normal(500, 10, (128, 128)).astype(np.float32)
        ap_fits.writeto(str(tmp_path / "tiny.fits"), data, overwrite=True)
        img = AstroImage(str(tmp_path / "tiny.fits"), label="Tiny")
        img.load()
        img.estimate_background()
        result = SpatialDetailAnalyzer().analyze(img)
        assert isinstance(result, dict)

    def test_weber_contrast_a_present(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "weber_contrast_a" in result

    def test_single_image_weber_contrast_b_empty(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        # Single-image mode: weber_contrast_b is present but empty (same pattern as contrast_ratios_b)
        wc_b = result.get("weber_contrast_b")
        assert wc_b is None or not wc_b

    def test_weber_contrast_a_positive(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        for v in result.get("weber_contrast_a", {}).values():
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
    amplitude 8x sky noise) is added inside the blob so std/LoG/wavelet/Weber/
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
        ("weber", WEBER_KERNEL_SIZES),
        ("gm", LOG_SIGMAS),
    ])
    def test_nc_score_dict_keys_match_scales(self, nc_result, prefix, scales):
        assert set(nc_result[f"{prefix}_nc_score_a"].keys()) == set(scales)
        assert set(nc_result[f"{prefix}_nc_score_b"].keys()) == set(scales)

    def test_wavelet_nc_score_keys_match_levels(self, nc_result):
        expected = set(range(1, WAVELET_LEVELS + 1))
        assert set(nc_result["wavelet_nc_score_a"].keys()) == expected
        assert set(nc_result["wavelet_nc_score_b"].keys()) == expected

    @pytest.mark.parametrize("prefix", ["std", "log", "wavelet", "weber", "gm"])
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

    def test_weber_nc_ratio_captures_finer_detail_in_a(self, nc_result):
        ratio = nc_result["weber_nc_ratio"][min(WEBER_KERNEL_SIZES)]
        assert ratio is not None and ratio > 1.05

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
        for ks in WEBER_KERNEL_SIZES:
            assert f"nrm_weber_{ks}px" in panels
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
    contrast_ratios_b / weber_contrast_b invariant (never None/absent, never
    populated with placeholder values on the A side either)."""

    @pytest.fixture(scope="class")
    @classmethod
    def single_result(cls, astro_image_a):
        return SpatialDetailAnalyzer().analyze(astro_image_a)

    @pytest.mark.parametrize("key", [
        "std_nc_score_a", "std_nc_score_b", "std_nc_ratio",
        "log_nc_score_a", "log_nc_score_b", "log_nc_ratio",
        "wavelet_nc_score_a", "wavelet_nc_score_b", "wavelet_nc_ratio",
        "weber_nc_score_a", "weber_nc_score_b", "weber_nc_ratio",
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
        score, noise = analyzer._nc_score(detail, None, bg_mask)
        assert score is None and noise is None

    def test_empty_shared_nebula_mask_returns_none(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.ones((20, 20), dtype=np.float32)
        mask_neb_shared = np.zeros((20, 20), dtype=bool)   # no shared nebula pixels
        bg_mask = np.ones((20, 20), dtype=bool)
        score, noise = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score is None and noise is None

    def test_empty_bg_mask_returns_none(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.ones((20, 20), dtype=np.float32)
        mask_neb_shared = np.ones((20, 20), dtype=bool)
        bg_mask = np.zeros((20, 20), dtype=bool)   # no background pixels
        score, noise = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score is None and noise is None

    def test_zero_noise_floor_returns_none(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.zeros((20, 20), dtype=np.float32)
        detail[:10, :] = 5.0   # nebula half has signal, background half is exactly 0
        mask_neb_shared = np.zeros((20, 20), dtype=bool)
        mask_neb_shared[:10, :] = True
        bg_mask = np.zeros((20, 20), dtype=bool)
        bg_mask[10:, :] = True
        score, noise = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score is None and noise is None

    def test_valid_masks_return_ratio(self):
        analyzer = SpatialDetailAnalyzer()
        detail = np.zeros((20, 20), dtype=np.float32)
        detail[:10, :] = 10.0
        detail[10:, :] = 2.0
        mask_neb_shared = np.zeros((20, 20), dtype=bool)
        mask_neb_shared[:10, :] = True
        bg_mask = np.zeros((20, 20), dtype=bool)
        bg_mask[10:, :] = True
        score, noise = analyzer._nc_score(detail, mask_neb_shared, bg_mask)
        assert score == pytest.approx(5.0)
        assert noise == pytest.approx(2.0)


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
    def test_absent_in_single_image_mode(self, astro_image_a):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert "mask_illustration" not in result["figures"]

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
        + [f"corr_weber_{ks}px" for ks in WEBER_KERNEL_SIZES]
    )

    @pytest.mark.parametrize("key", _CORR_KEYS)
    def test_present_in_two_image_mode(self, nc_result, key):
        assert key in nc_result["figures"]
        assert isinstance(nc_result["figures"][key], str)
        assert len(nc_result["figures"][key]) > 0

    @pytest.mark.parametrize("key", _CORR_KEYS)
    def test_absent_in_single_image_mode(self, astro_image_a, key):
        result = SpatialDetailAnalyzer().analyze(astro_image_a)
        assert key not in result["figures"]


@pytest.fixture(scope="module")
def nc_result_with_crosshair(nc_image_pair) -> dict:
    img_a, img_b = nc_image_pair
    crosshair = {"x0": 0.1, "y0": 0.5, "x1": 0.9, "y1": 0.5}
    return SpatialDetailAnalyzer().analyze(img_a, img_b, crosshair=crosshair)


class TestCrosshairEmbeddedCrossSections:
    """Section 8 cross-sections are embedded panels inside the 2x2 map figures,
    not separate figures — no standalone xs_* keys should ever appear, and all
    5 families (including Weber, newly crosshair-capable) must still produce
    their usual figure keys with or without a crosshair, without crashing."""

    def test_no_crash_with_crosshair(self, nc_result_with_crosshair):
        assert isinstance(nc_result_with_crosshair, dict)

    def test_no_standalone_xs_keys_with_crosshair(self, nc_result_with_crosshair):
        figs = nc_result_with_crosshair["figures"]
        assert not any(k.startswith(("xs_std_", "xs_log_", "xs_wavelet_",
                                      "xs_gradient_", "xs_weber_")) for k in figs)

    def test_no_standalone_xs_keys_without_crosshair(self, nc_result):
        figs = nc_result["figures"]
        assert not any(k.startswith(("xs_std_", "xs_log_", "xs_wavelet_",
                                      "xs_gradient_", "xs_weber_")) for k in figs)

    @pytest.mark.parametrize("key_prefix,scales", [
        ("std_", STD_KERNEL_SIZES), ("weber_", WEBER_KERNEL_SIZES),
    ])
    def test_family_figures_present_with_crosshair(self, nc_result_with_crosshair,
                                                     key_prefix, scales):
        figs = nc_result_with_crosshair["figures"]
        for scale in scales:
            assert f"{key_prefix}{scale}px" in figs

    def test_weber_no_crash_without_crosshair(self, nc_result):
        # Regression: weber previously had no crosshair param at all.
        assert all(f"weber_{ks}px" in nc_result["figures"] for ks in WEBER_KERNEL_SIZES)

    def test_weber_nrm_figures_present_with_crosshair(self, nc_result_with_crosshair):
        figs = nc_result_with_crosshair["figures"]
        assert any(k.startswith("nrm_weber_") for k in figs)

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
    8c Log-Ratio Distribution, 8d LoG, 8e Wavelet, 8f Gradient, 8g Local Std,
    8h Weber, 8i NC overview), and the fix for the alphabetic-sort bug that put
    e.g. nrm_std_10px before nrm_std_3px/5px."""

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
        expected = ["8a", "8b", "8c", "8d", "8e", "8f", "8g", "8h", "8i"]
        found = re.findall(r"<h3>(8[a-i])\.", section_html)
        assert found == expected

    def test_original_image_section_present(self, section_html):
        assert "8b. Original Image" in section_html
        assert 'alt="original"' in section_html or "original" in section_html

    def test_local_std_now_in_contrast_group_after_gradient(self, section_html):
        assert section_html.index("8f. Gradient Magnitude") < section_html.index(
            "8g. Local Standard Deviation Maps")

    def test_weber_after_local_std(self, section_html):
        assert section_html.index("8g. Local Standard Deviation Maps") < section_html.index(
            "8h. Weber Fraction Contrast Maps")

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
        # Every literal "8<letter>" reference must be one of the 9 valid new
        # letters — a stray old letter (e.g. a missed "8e" -> "8h" rename)
        # would show up here.
        for m in re.finditer(r"\b8([a-z])\b", prose):
            assert m.group(1) in "abcdefghi", f"unexpected section letter: 8{m.group(1)}"
