"""Unit tests for analysis/edge_analyzer.py."""
from __future__ import annotations

import pytest

from analysis.edge_analyzer import EdgeAnalyzer

_RESULT_KEYS = {"edges", "n_edges", "rois_used"}


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
