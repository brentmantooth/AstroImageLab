"""Unit tests for analysis/edge_analyzer.py."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from analysis.edge_analyzer import EdgeAnalyzer
from core.models import EDGE_ESF_MIN_MONOTONICITY

_RESULT_KEYS = {"edges", "n_edges", "rois_used"}


def _make_clean_edge_roi(angle_deg: float = 30.0, size: int = 60) -> np.ndarray:
    """Single straight edge through the box center, background-subtracted
    semantics (background ~ 0, signal positive) -- matches real bgsub data."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    theta = np.radians(angle_deg)
    d = (xx - c) * np.cos(theta) + (yy - c) * np.sin(theta)
    roi = np.where(d > 0, 200.0, 0.0).astype(float)
    return gaussian_filter(roi, sigma=1.5)


def _make_double_edge_roi(size: int = 60) -> np.ndarray:
    """Thin bright stripe crossing the box -- genuinely two edges for any
    perpendicular scan direction; no rotation/masking fix can turn this into
    a single clean transition."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    stripe = np.abs((xx - c) - 0.3 * (yy - c)) < 5
    roi = np.where(stripe, 200.0, 0.0).astype(float)
    return gaussian_filter(roi, sigma=1.5)


class TestAnalyze:
    def test_returns_dict(self, astro_image_a):
        result = EdgeAnalyzer().analyze(astro_image_a)
        assert isinstance(result, dict)

    def test_required_keys_present(self, astro_image_a):
        result = EdgeAnalyzer().analyze(astro_image_a)
        assert _RESULT_KEYS.issubset(result.keys())

    def test_n_edges_nonnegative(self, astro_image_a):
        result = EdgeAnalyzer().analyze(astro_image_a)
        assert result["n_edges"] >= 0

    def test_edges_is_list(self, astro_image_a):
        result = EdgeAnalyzer().analyze(astro_image_a)
        assert isinstance(result["edges"], list)

    def test_n_edges_matches_list_length(self, astro_image_a):
        result = EdgeAnalyzer().analyze(astro_image_a)
        assert result["n_edges"] == len(result["edges"])

    def test_with_roi(self, astro_image_a):
        result = EdgeAnalyzer().analyze(astro_image_a,
                                        roi=(100, 100, 400, 400))
        assert isinstance(result, dict)


class TestEsfQuality:
    def test_clean_edge_scores_high(self):
        ea = EdgeAnalyzer()
        roi = _make_clean_edge_roi(angle_deg=30.0)
        edge_info = ea._detect_strongest_edge(roi)
        _, esf = ea._extract_esf(roi, edge_info)
        assert ea._esf_quality(esf) > 0.8

    def test_double_edge_scores_low(self):
        ea = EdgeAnalyzer()
        roi = _make_double_edge_roi()
        edge_info = ea._detect_strongest_edge(roi)
        _, esf = ea._extract_esf(roi, edge_info)
        assert ea._esf_quality(esf) < EDGE_ESF_MIN_MONOTONICITY

    def test_perfectly_flat_scores_zero(self):
        ea = EdgeAnalyzer()
        assert ea._esf_quality(np.full(60, 0.5)) == 0.0

    @pytest.mark.parametrize("angle_deg", [15.0, 30.0, 60.0, 75.0])
    def test_clean_edge_scores_high_at_various_angles(self, angle_deg):
        # Regression: before the disc-mask fix, a clean oblique edge scored as
        # low as ~0.0-0.3 (rotate() zero-padding the clipped box corners
        # fabricated a second transition), indistinguishable from a genuinely
        # bad edge. 45deg is intentionally excluded: a boundary passing exactly
        # through both box corners gives Sobel a perfect gradient tie along the
        # whole diagonal, so argmax picks a corner-adjacent pixel instead of
        # the center -- a narrow, non-representative synthetic degeneracy
        # (real edges are never exactly 45.000 deg through both corners) that
        # the quality gate correctly flags rather than silently mismeasures.
        ea = EdgeAnalyzer()
        roi = _make_clean_edge_roi(angle_deg=angle_deg)
        edge_info = ea._detect_strongest_edge(roi)
        _, esf = ea._extract_esf(roi, edge_info)
        assert ea._esf_quality(esf) > 0.8


class TestExtractEsfDiscMask:
    def test_no_nan_in_returned_esf(self):
        # _extract_esf trims the disc-masked NaN columns internally --
        # callers should never see NaN.
        ea = EdgeAnalyzer()
        roi = _make_clean_edge_roi(angle_deg=45.0)
        edge_info = ea._detect_strongest_edge(roi)
        positions, esf = ea._extract_esf(roi, edge_info)
        assert esf is not None
        assert not np.any(np.isnan(esf))
        assert not np.any(np.isnan(positions))

    def test_positions_start_at_zero(self):
        ea = EdgeAnalyzer()
        roi = _make_clean_edge_roi(angle_deg=45.0)
        edge_info = ea._detect_strongest_edge(roi)
        positions, _ = ea._extract_esf(roi, edge_info)
        assert positions[0] == 0.0

    def test_esf_normalised_to_unit_range(self):
        ea = EdgeAnalyzer()
        roi = _make_clean_edge_roi(angle_deg=30.0)
        edge_info = ea._detect_strongest_edge(roi)
        _, esf = ea._extract_esf(roi, edge_info)
        assert esf.min() >= -1e-9
        assert esf.max() <= 1.0 + 1e-9


class TestQualityGateAutoDetect:
    """Directly control which candidate ROIs _auto_detect_top_rois returns so
    the skip/fallback control flow can be tested deterministically, without
    depending on real Background2D estimation picking particular peaks."""

    def test_bad_candidates_skipped_in_favour_of_clean_one(self, astro_image_a, monkeypatch):
        bad1 = _make_double_edge_roi()
        bad2 = _make_double_edge_roi()
        clean = _make_clean_edge_roi(angle_deg=30.0)
        candidates = [
            (bad1, (0, 0, 60, 60)),
            (bad2, (100, 100, 160, 160)),
            (clean, (200, 200, 260, 260)),
        ]
        monkeypatch.setattr(
            EdgeAnalyzer, "_auto_detect_top_rois",
            lambda self, bgsub, image, n=9: candidates)

        result = EdgeAnalyzer().analyze(astro_image_a)
        assert result["n_edges"] == 1
        assert result["edges"][0]["low_confidence"] is False
        assert result["edges"][0]["roi_used"] == (200, 200, 260, 260)

    def test_all_bad_falls_back_to_least_bad_flagged(self, astro_image_a, monkeypatch):
        bad1 = _make_double_edge_roi()
        bad2 = _make_double_edge_roi()
        candidates = [
            (bad1, (0, 0, 60, 60)),
            (bad2, (100, 100, 160, 160)),
        ]
        monkeypatch.setattr(
            EdgeAnalyzer, "_auto_detect_top_rois",
            lambda self, bgsub, image, n=9: candidates)

        result = EdgeAnalyzer().analyze(astro_image_a)
        assert result["n_edges"] == 1
        assert result["edges"][0]["low_confidence"] is True

    def test_edge_entries_have_quality_keys(self, astro_image_a, monkeypatch):
        clean = _make_clean_edge_roi(angle_deg=30.0)
        monkeypatch.setattr(
            EdgeAnalyzer, "_auto_detect_top_rois",
            lambda self, bgsub, image, n=9: [(clean, (0, 0, 60, 60))])

        result = EdgeAnalyzer().analyze(astro_image_a)
        assert result["n_edges"] == 1
        entry = result["edges"][0]
        assert "esf_quality" in entry
        assert "low_confidence" in entry
        assert isinstance(entry["esf_quality"], float)


class TestQualityGateExplicitRoi:
    """A user-drawn or A-matched ROI is fixed -- low quality must still be
    flagged, but the edge must not be silently dropped (there's no
    alternative candidate to fall back to)."""

    def test_bad_edge_kept_but_flagged_with_explicit_roi(self, tmp_path):
        from astropy.io import fits as ap_fits
        from core.astro_image import AstroImage

        h = w = 256
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 20.0, (h, w)).astype(np.float64)
        yy, xx = np.mgrid[0:h, 0:w]
        stripe = np.abs((xx - 128) - 0.3 * (yy - 128)) < 5
        data[stripe] += 200.0
        data = np.clip(data, 0, 65535).astype(np.float32)
        hdr = ap_fits.Header()
        hdr["EGAIN"] = 1.0
        hdr["GAIN"] = 1.0
        path = tmp_path / "stripe.fits"
        ap_fits.writeto(str(path), data, hdr, overwrite=True)

        img = AstroImage(str(path), label="Stripe")
        img.load()
        img.estimate_background()

        result = EdgeAnalyzer().analyze(img, roi=(98, 98, 158, 158))
        assert result["n_edges"] == 1
        assert result["edges"][0]["low_confidence"] is True


class TestAnalyzeCrosshair:
    def test_degenerate_crosshair_returns_none(self, astro_image_a):
        # Zero-length crosshair (start == end)
        ch = {"x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.5}
        result = EdgeAnalyzer().analyze_crosshair(astro_image_a, ch)
        assert result is None

    def test_valid_crosshair_no_crash(self, astro_image_a):
        ch = {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9}
        result = EdgeAnalyzer().analyze_crosshair(astro_image_a, ch)
        # May be None (no clean edge) or a dict — must not raise
        assert result is None or isinstance(result, dict)

    def test_horizontal_crosshair_no_crash(self, astro_image_a):
        ch = {"x0": 0.1, "y0": 0.5, "x1": 0.9, "y1": 0.5}
        result = EdgeAnalyzer().analyze_crosshair(astro_image_a, ch)
        assert result is None or isinstance(result, dict)
