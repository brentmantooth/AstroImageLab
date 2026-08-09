from __future__ import annotations

import base64
import io
import threading

import matplotlib.pyplot as plt

# matplotlib's mathtext grammar (matplotlib/_mathtext.py) calls pyparsing's
# ParserElement.enable_packrat() at import time, turning on a process-wide,
# non-thread-safe memoization cache used by every mathtext parse (tick labels,
# titles, legends -- anything rendered through a draw pass). Multiple
# analyzers render figures concurrently (SpatialDetailAnalyzer's own 5-way
# ThreadPoolExecutor, PowerSpectrumAnalyzer's per-image ThreadPoolExecutor, and
# analysis_thread.py's cross-analyzer parallel mode), so unsynchronized calls
# can corrupt that shared cache and raise a spurious mathtext ParseException
# on completely valid text (reproduced directly against
# matplotlib.mathtext.MathTextParser.parse() under thread concurrency).
#
# savefig() is not the only trigger -- fig.tight_layout() also runs a full
# draw pass to measure text extents (titles, tick labels, legends), so it
# hits the same shared cache. Any figure-building code that can run
# concurrently with another figure-building call must route *both*
# tight_layout() and savefig() through this one process-wide lock; locking
# only savefig() leaves tight_layout() free to race and reproduces the same
# ParseException. Use finalize_layout() below instead of calling
# fig.tight_layout() directly.
#
# Nor is tight_layout() the last of it -- fig.colorbar() and ax.legend()
# (particularly loc="best"/auto-placement, which needs text-extent
# measurement to find a non-overlapping spot) can also hit the same cache
# during figure *construction*, before savefig() is ever called. Use
# locked_draw_call() below to wrap any such call in code that can run
# concurrently with other figure-building code.
_MPL_DRAW_LOCK = threading.Lock()


def finalize_layout(fig: plt.Figure, **kwargs) -> None:
    """Run fig.tight_layout() under the same lock as fig_to_b64()'s savefig().

    Call this instead of fig.tight_layout() in any analyzer/report figure
    builder that can execute concurrently with other figure-building code
    (see the _MPL_DRAW_LOCK comment above for why).
    """
    with _MPL_DRAW_LOCK:
        fig.tight_layout(**kwargs)


def locked_draw_call(fn, *args, **kwargs):
    """Run a matplotlib call that can trigger draw-adjacent text/layout
    measurement (colorbar placement, legend auto-placement, etc.) under the
    same process-wide lock as finalize_layout()/fig_to_b64(). Call this
    instead of calling fig.colorbar(...)/ax.legend(...) directly in any
    analyzer/report figure builder that can execute concurrently with other
    figure-building code (see the _MPL_DRAW_LOCK comment above for why).
    """
    with _MPL_DRAW_LOCK:
        return fn(*args, **kwargs)


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 120) -> bytes:
    """Render a matplotlib figure to raw PNG bytes and immediately close it.

    Call this in analyzer code instead of storing live Figure objects, so figures
    are released as soon as their pixels are captured.
    """
    buf = io.BytesIO()
    with _MPL_DRAW_LOCK:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    data = buf.getvalue()
    plt.close(fig)
    return data


def fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib figure to a base64 PNG string and immediately close it."""
    return base64.b64encode(fig_to_png_bytes(fig, dpi=dpi)).decode()


def figs_to_b64(figures: dict, dpi: int = 120) -> dict:
    """Convert all Figure values in a dict to base64 strings in one call."""
    return {
        k: fig_to_b64(v, dpi=dpi) if isinstance(v, plt.Figure) else v
        for k, v in figures.items()
    }
