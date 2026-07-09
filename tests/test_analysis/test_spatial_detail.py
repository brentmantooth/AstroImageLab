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
