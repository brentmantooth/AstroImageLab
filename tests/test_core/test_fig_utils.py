"""Unit tests for core/fig_utils.py."""
from __future__ import annotations

import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from core.fig_utils import fig_to_b64, figs_to_b64


def _make_fig() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    return fig


class TestFigToB64:
    def test_returns_string(self):
        fig = _make_fig()
        result = fig_to_b64(fig)
        assert isinstance(result, str)

    def test_is_valid_base64_png(self):
        fig = _make_fig()
        result = fig_to_b64(fig)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"\x89PNG"

    def test_closes_figure(self):
        fig = _make_fig()
        num = fig.number
        fig_to_b64(fig)
        assert num not in plt.get_fignums()

    def test_nonempty_result(self):
        fig = _make_fig()
        result = fig_to_b64(fig)
        assert len(result) > 100

    def test_custom_dpi(self):
        fig_lo = _make_fig()
        fig_hi = _make_fig()
        lo = fig_to_b64(fig_lo, dpi=72)
        hi = fig_to_b64(fig_hi, dpi=200)
        # Higher DPI produces a larger PNG
        assert len(base64.b64decode(hi)) > len(base64.b64decode(lo))


class TestFigsToB64:
    def test_converts_figure_values(self):
        fig = _make_fig()
        d = {"my_fig": fig, "other": "already_a_string"}
        result = figs_to_b64(d)
        assert isinstance(result["my_fig"], str)
        assert base64.b64decode(result["my_fig"])[:4] == b"\x89PNG"

    def test_passthrough_non_figure_values(self):
        d = {"a": "hello", "b": 42, "c": None}
        result = figs_to_b64(d)
        assert result["a"] == "hello"
        assert result["b"] == 42
        assert result["c"] is None

    def test_empty_dict(self):
        assert figs_to_b64({}) == {}

    def test_mixed_dict(self):
        fig = _make_fig()
        d = {"fig": fig, "num": 3.14, "txt": "label"}
        result = figs_to_b64(d)
        assert isinstance(result["fig"], str)
        assert result["num"] == 3.14
        assert result["txt"] == "label"
