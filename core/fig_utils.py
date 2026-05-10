from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt


def fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib figure to a base64 PNG string and immediately close it.

    Call this in analyzer code instead of storing live Figure objects, so figures
    are released as soon as their pixels are captured.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def figs_to_b64(figures: dict, dpi: int = 120) -> dict:
    """Convert all Figure values in a dict to base64 strings in one call."""
    return {
        k: fig_to_b64(v, dpi=dpi) if isinstance(v, plt.Figure) else v
        for k, v in figures.items()
    }
