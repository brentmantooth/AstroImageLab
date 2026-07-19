from __future__ import annotations

import base64
import io
import threading

import matplotlib.pyplot as plt

# matplotlib's mathtext grammar (matplotlib/_mathtext.py) calls pyparsing's
# ParserElement.enable_packrat() at import time, turning on a process-wide,
# non-thread-safe memoization cache used by every mathtext parse (tick labels,
# titles, legends -- anything rendered through savefig()/draw()). Multiple
# analyzers render figures concurrently (SpatialDetailAnalyzer's own 5-way
# ThreadPoolExecutor, PowerSpectrumAnalyzer's per-image ThreadPoolExecutor, and
# analysis_thread.py's cross-analyzer parallel mode), so unsynchronized
# savefig() calls can corrupt that shared cache and raise a spurious
# mathtext ParseException on completely valid text (reproduced directly
# against matplotlib.mathtext.MathTextParser.parse() under thread
# concurrency). Serializing every savefig() through one process-wide lock
# eliminates the race.
_SAVEFIG_LOCK = threading.Lock()


def fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib figure to a base64 PNG string and immediately close it.

    Call this in analyzer code instead of storing live Figure objects, so figures
    are released as soon as their pixels are captured.
    """
    buf = io.BytesIO()
    with _SAVEFIG_LOCK:
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
