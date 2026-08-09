"""Pure array maths behind the Data Inspector — no Qt, no pyqtgraph.

This module exists so the Data Inspector's arithmetic can be tested.  `gui/` is
untested by design (the headless suite has no PyQt6 — see CLAUDE.md), and that gap
is not theoretical: `value_range`'s report-matching `max(0.0, ...)` floor was once
written here as a conditional that preserved negatives for signed maps, which shaded
the Original panel from -9.23 where the report starts at 0.  Nothing could catch it
until the formula moved somewhere a test could reach.

Every colour-scale and comparison formula here is the report's own, imported from
`SpatialDetailAnalyzer` rather than reimplemented, so a panel means the same thing in
the Data Inspector as it does in the Section 8 figures.

Region selection is composable, matching the UI: the ROI (or the whole image when
none is drawn) fixes the pixel *domain*, and the threshold *splits* that domain into
an in-mask and an out-of-mask population, one per correlation plot.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from matplotlib.path import Path as _MplPath
from scipy.ndimage import binary_closing, binary_opening

from analysis.image_filters import SpatialDetailAnalyzer as _SDA
from core.models import SECTION8_SCATTER_MAX_SAMPLES

# _fill_small_holes / _remove_small_objects are instance methods that never touch
# self; one shared instance avoids reimplementing size-filtering that already exists.
_ANALYZER = _SDA()

# The two A-vs-B comparisons the inspector offers.  There is deliberately no linear
# A/B mode: Section 8's convention is the log10 ratio throughout, and a linear ratio
# is neither symmetric about its no-change point nor safe on a diverging colormap.
COMPARE_LOGRATIO = "logratio"
COMPARE_DIFFERENCE = "difference"
COMPARE_MODES = (COMPARE_LOGRATIO, COMPARE_DIFFERENCE)

# Rec.709 luma weights, matching gui/report_inspector.py's own RGB handling.
_LUMA = (0.2126, 0.7152, 0.0722)


def to_2d(arr: np.ndarray) -> np.ndarray:
    """Reduce an RGB array to Rec.709 luma; pass 2-D through unchanged.

    Defensive only: since the PSF Simulation panels became float32 values, nothing
    the report writes is RGB.  Kept so an older .npz (which still holds an RGB
    sim_diff) can be measured rather than rejected.
    """
    if arr.ndim == 3:
        a = arr.astype(np.float32)
        return _LUMA[0] * a[..., 0] + _LUMA[1] * a[..., 1] + _LUMA[2] * a[..., 2]
    return arr


def common_crop(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop both arrays to their common shape, top-left aligned.

    Mirrors the defensive crop `_log_ratio_map` already performs: panels of the same
    family can differ by a row or column (the wavelet reconstructions are 626x836
    where the rest of the file is 626x835), and two-image analysis can reach here
    with mismatched shapes when astroalign registration fails.
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def value_range(arrays: list[np.ndarray | None]) -> tuple[float, float]:
    """Shared percentile-clipped display range across several panels.

    Character-for-character the same formula as `_plot_side_by_side`'s shared A/B
    scale (`analysis/image_filters.py`), including the unconditional `max(0.0, ...)`
    floor, so a panel is shaded identically in the inspector and in the report.

    That floor does clip negative values to the bottom colour on signed panels (the
    background-subtracted Original, wavelet band-pass reconstructions).  It is kept
    deliberately rather than "improved" for signed data: matching the report is the
    point, and the inspector's colour bar is interactive, so the levels can be
    dragged to reveal the negative tail when that is what you want to look at.
    """
    finite = [to_2d(a) for a in arrays if a is not None and a.size]
    if not finite:
        return 0.0, 1.0
    vmin = max(0.0, min(float(np.percentile(a, 0.5)) for a in finite))
    vmax = max(float(np.percentile(a, 99.5)) for a in finite)
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return vmin, vmax


def comparison_map(arr_a: np.ndarray, arr_b: np.ndarray,
                   mode: str = COMPARE_LOGRATIO) -> np.ndarray:
    """A-vs-B comparison in the report's own terms, as float32.

    `COMPARE_LOGRATIO` delegates to `SpatialDetailAnalyzer._log_ratio_map`, which
    handles the common-shape crop and the percentile epsilon floor and discards sign
    via abs() so the result is defined for signed map families.  `COMPARE_DIFFERENCE`
    is the plain signed A - B on the same common crop.
    """
    a, b = to_2d(arr_a), to_2d(arr_b)
    if mode == COMPARE_DIFFERENCE:
        a, b = common_crop(a, b)
        return (a - b).astype(np.float32)
    if mode != COMPARE_LOGRATIO:
        raise ValueError(f"unknown comparison mode {mode!r}; expected one of {COMPARE_MODES}")
    return _SDA._log_ratio_map(a, b)


def comparison_range(diff: np.ndarray) -> tuple[float, float]:
    """Symmetric-about-zero colour range for a comparison map.

    Shared by the map, its histogram, and the correlation scatter's dot colouring, so
    one colour always means one value.  Works for either comparison mode: it is +/-
    the 99.5th percentile of |diff|, which is symmetric for any signed array, not
    something specific to the log ratio.
    """
    return _SDA._log_ratio_color_range(diff)


# ---------------------------------------------------------------------------
# Domain: the ROI
# ---------------------------------------------------------------------------

ROI_RECT = "rect"
ROI_ELLIPSE = "ellipse"
ROI_POLYGON = "polygon"
ROI_KINDS = (ROI_RECT, ROI_ELLIPSE, ROI_POLYGON)


def roi_mask(shape: tuple[int, int], roi: dict | None) -> np.ndarray:
    """Rasterise a normalised [0,1] ROI into a boolean mask of `shape`.

    `roi` is None (the whole image), or a dict with "kind" in ROI_KINDS:
      rect / ellipse: {"kind": ..., "x0","y0","x1","y1"}  — a normalised bounding box
      polygon:        {"kind": "polygon", "points": [(xn, yn), ...]}  — a lasso

    Coordinates are normalised rather than pixel-based so a drawn region survives a
    switch to a panel of a different size, the same convention the cross-section line
    and the view rectangle use.
    """
    h, w = shape[:2]
    if roi is None:
        return np.ones((h, w), dtype=bool)

    kind = roi.get("kind")
    if kind == ROI_POLYGON:
        return _polygon_mask(shape, roi.get("points") or [])

    x0, x1 = sorted((roi["x0"] * w, roi["x1"] * w))
    y0, y1 = sorted((roi["y0"] * h, roi["y1"] * h))
    c0, c1 = int(np.floor(max(0.0, x0))), int(np.ceil(min(float(w), x1)))
    r0, r1 = int(np.floor(max(0.0, y0))), int(np.ceil(min(float(h), y1)))

    mask = np.zeros((h, w), dtype=bool)
    if c1 <= c0 or r1 <= r0:
        return mask                       # degenerate drag; select nothing

    if kind == ROI_ELLIPSE:
        cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
        ry, rx = max(0.5 * (y1 - y0), 1e-9), max(0.5 * (x1 - x0), 1e-9)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        inside = (((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2) <= 1.0
        mask[r0:r1, c0:c1] = inside
    else:
        mask[r0:r1, c0:c1] = True
    return mask


def _polygon_mask(shape: tuple[int, int], points) -> np.ndarray:
    """Point-in-polygon over the polygon's bounding box only.

    Testing every pixel in a multi-megapixel panel against the path would dominate
    an interactive lasso drag; the bounding box is usually a small fraction of it.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask                       # not yet a closed region

    px, py = pts[:, 0] * w, pts[:, 1] * h
    c0, c1 = int(np.floor(max(0.0, px.min()))), int(np.ceil(min(float(w), px.max())))
    r0, r1 = int(np.floor(max(0.0, py.min()))), int(np.ceil(min(float(h), py.max())))
    if c1 <= c0 or r1 <= r0:
        return mask

    yy, xx = np.mgrid[r0:r1, c0:c1]
    # Pixel centres, so a pixel counts as inside when its middle is enclosed.
    coords = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    inside = _MplPath(np.column_stack((px, py))).contains_points(coords)
    mask[r0:r1, c0:c1] = inside.reshape(r1 - r0, c1 - c0)
    return mask


# ---------------------------------------------------------------------------
# Split: the threshold
# ---------------------------------------------------------------------------

THRESH_PERCENTILE = "percentile"
THRESH_ABSOLUTE = "absolute"
THRESH_MODES = (THRESH_PERCENTILE, THRESH_ABSOLUTE)

DIR_UPPER = "upper"
DIR_LOWER = "lower"
DIRECTIONS = (DIR_UPPER, DIR_LOWER)


class ThresholdResult(NamedTuple):
    mask: np.ndarray
    warning: str | None       # non-None when the cut is not meaningful; surface it


def threshold_mask(arr_a: np.ndarray, arr_b: np.ndarray,
                   mode: str = THRESH_PERCENTILE,
                   value: float = 10.0,
                   direction: str = DIR_UPPER) -> ThresholdResult:
    """Threshold A and B independently, then AND the two masks.

    Percentile mode cuts each image at its *own* distribution, so one setting stays
    meaningful across metric families whose units differ by orders of magnitude
    ("top 10%" means the same thing for Local sigma and for Entropy in bits).
    Absolute mode applies one literal value to both.

    `value` in percentile mode is the size of the selected tail, not the percentile
    itself: value=10, direction="upper" selects each image's top 10%.

    Returns a warning instead of a silently-wrong mask when an input has no spread:
    on a constant array every percentile equals that one value, so `arr >= threshold`
    matches 100% of pixels rather than the requested tail — the degenerate case
    CLAUDE.md documents for _top_percent_mask.
    """
    if mode not in THRESH_MODES:
        raise ValueError(f"unknown threshold mode {mode!r}; expected one of {THRESH_MODES}")
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; expected one of {DIRECTIONS}")

    a, b = common_crop(to_2d(arr_a), to_2d(arr_b))
    warning = None

    if mode == THRESH_PERCENTILE:
        flat = [x for x in (a, b) if x.size]
        if any(float(np.ptp(x)) == 0.0 for x in flat):
            return ThresholdResult(np.zeros(a.shape, dtype=bool),
                                    "One image has no variation, so a percentile cut "
                                    "would select every pixel; nothing selected instead.")
        q = (100.0 - value) if direction == DIR_UPPER else value
        q = float(np.clip(q, 0.0, 100.0))
        cut_a, cut_b = float(np.percentile(a, q)), float(np.percentile(b, q))
    else:
        cut_a = cut_b = float(value)

    if direction == DIR_UPPER:
        mask = (a >= cut_a) & (b >= cut_b)
    else:
        mask = (a <= cut_a) & (b <= cut_b)

    if not mask.any():
        warning = "The threshold selects no pixels."
    elif mask.all():
        warning = "The threshold selects every pixel."
    return ThresholdResult(mask, warning)


def refine_mask(mask: np.ndarray,
                open_px: int = 0, close_px: int = 0,
                min_size_px: int = 0, fill_hole_px: int = 0) -> np.ndarray:
    """Morphological cleanup of a threshold mask.

    Order is deliberate: **opening precedes closing**.  Closing starts with a
    dilation, and dilating a raw threshold mask inflates every isolated noise speck
    into a large blob — the failure CLAUDE.md records for the Section 8 nebula mask.
    Opening (erode-then-dilate) removes those specks first, so the closing that
    follows smooths real structure instead of amplifying noise.  Size-based cleanup
    runs last, reusing SpatialDetailAnalyzer's own helpers rather than a second
    implementation of the same idea.
    """
    out = np.asarray(mask, dtype=bool)
    if open_px > 0:
        out = binary_opening(out, iterations=int(open_px))
    if close_px > 0:
        out = binary_closing(out, iterations=int(close_px))
    if min_size_px > 0:
        out = _ANALYZER._remove_small_objects(out, int(min_size_px))
    if fill_hole_px > 0:
        out = _ANALYZER._fill_small_holes(out, int(fill_hole_px))
    return out


# ---------------------------------------------------------------------------
# Correlation sampling
# ---------------------------------------------------------------------------

class CorrelationSample(NamedTuple):
    a_vals: np.ndarray       # y axis of the scatter (rendered, capped at max_samples)
    b_vals: np.ndarray       # x axis (rendered, capped)
    c_vals: np.ndarray       # comparison value, drives dot colour (rendered, capped)
    flat_ids: np.ndarray     # int64 row*W + col in the common-cropped frame; matches
                             # a_vals/b_vals/c_vals
    full_a: np.ndarray       # a values for the FULL masked population, uncapped —
    full_b: np.ndarray       # for selection accuracy, never for rendering.  Equal to
    full_flat_ids: np.ndarray  # a_vals/b_vals/flat_ids (same objects) when n_total
                             # <= max_samples, i.e. no extra memory in the common case.
    n_total: int             # population size BEFORE subsampling == full_a.size
    axis_lo: float
    axis_hi: float
    shape: tuple[int, int]   # the frame flat_ids/full_flat_ids index into


def correlation_sample(map_a: np.ndarray, map_b: np.ndarray,
                       comparison: np.ndarray, mask: np.ndarray,
                       rng: np.random.Generator | None = None,
                       max_samples: int = SECTION8_SCATTER_MAX_SAMPLES) -> CorrelationSample:
    """Extract A-vs-B scatter points for the pixels selected by `mask`.

    `flat_ids` is the stable identity linking a plotted point back to its pixel: a
    flat index into the common-cropped (H, W) frame.  Selections are carried as these
    ids, never as positions within the subsampled arrays, which change whenever the
    sample is redrawn.

    Axis limits come from the FULL population, not the subsample — the subsample
    exists only to bound render cost, and upper-tail divergence from the 1:1 line is
    the signal the plot exists to show (same rule as _plot_metric_correlation).

    `full_a`/`full_b`/`full_flat_ids` carry the FULL masked population — a lasso
    selection must be tested against these, not the rendered `a_vals`/`b_vals`, or a
    "select everything" gesture can only ever select the random `max_samples` subset
    that happened to be plotted, producing a sparse mask instead of a solid one.
    """
    a, b = common_crop(to_2d(map_a), to_2d(map_b))
    c, _ = common_crop(to_2d(comparison), a)
    a, _ = common_crop(a, c)
    b, _ = common_crop(b, c)
    m, _ = common_crop(np.asarray(mask, dtype=bool), c)
    h, w = a.shape

    a_all, b_all, c_all = a[m], b[m], c[m]
    flat_all = np.flatnonzero(m.ravel()).astype(np.int64)
    n_total = int(a_all.size)
    if n_total == 0:
        empty = np.empty(0, dtype=np.float32)
        empty_i = np.empty(0, dtype=np.int64)
        return CorrelationSample(empty, empty, empty, empty_i,
                                  empty, empty, empty_i, 0, 0.0, 1.0, (h, w))

    lo = float(min(a_all.min(), b_all.min()))
    hi = float(max(a_all.max(), b_all.max()))
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    if n_total > max_samples:
        rng = rng if rng is not None else np.random.default_rng(42)
        idx = rng.choice(n_total, max_samples, replace=False)
        a_plot, b_plot, c_plot, flat_plot = a_all[idx], b_all[idx], c_all[idx], flat_all[idx]
    else:
        a_plot, b_plot, c_plot, flat_plot = a_all, b_all, c_all, flat_all

    return CorrelationSample(a_plot, b_plot, c_plot, flat_plot,
                             a_all, b_all, flat_all,
                             n_total, lo, hi, (h, w))


def select_points_in_polygon(xs: np.ndarray, ys: np.ndarray,
                              polygon) -> np.ndarray:
    """Boolean mask of which scatter points fall inside a lasso/rectangle polygon.

    Kept here rather than in the plot widget so the brushing maths is testable: an
    off-by-one between a point's plotted position and its flat pixel id lights up the
    wrong pixels, and it does so silently.  `polygon` is an (N, 2) sequence of
    (x, y) vertices in the scatter's own data coordinates.
    """
    pts = np.asarray(polygon, dtype=float)
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[0] < 3 or xs.size == 0:
        return np.zeros(xs.size, dtype=bool)
    return _MplPath(pts).contains_points(np.column_stack((xs, ys)))


def ids_to_mask(flat_ids: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of `shape` with the given flat pixel ids set — the inverse of
    correlation_sample's flat_ids, used to paint a scatter selection onto the images."""
    h, w = shape[:2]
    mask = np.zeros(h * w, dtype=bool)
    ids = np.asarray(flat_ids, dtype=np.int64)
    if ids.size:
        mask[ids[(ids >= 0) & (ids < h * w)]] = True
    return mask.reshape(h, w)
