from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
from astropy.modeling import fitting
from astropy.modeling.models import Moffat2D
import matplotlib
import matplotlib.patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QTextEdit, QWidget,
)

from core.astro_image import AstroImage
from core.stretch import normalize_for_display
from gui.image_panel import ZoomableImageLabel


_PROCESSING_NOTES_HTML = """
<b>Processing notes</b>
<p style="margin:2px 0 3px 0"><b>Image panels</b> show a raw (non-background-subtracted)
cutout. All quantitative analysis uses a global 2D background map (photutils
<i>Background2D</i>, 64×64 px mesh) subtracted from the full image first.</p>
<p style="margin:2px 0 3px 0"><b>PSF fit &amp; shape metrics</b> use a fixed 25×25 px
window regardless of sample radius. An additional local correction (median of the
outer 3-px border ring) is subtracted before fitting to remove nebula gradients.</p>
<p style="margin:2px 0 0px 0"><b>Background (ADU)</b> is the median of the outer
annulus (r ≥ 70% of sample radius) on the <i>background-subtracted</i> image —
a residual, not the raw sky level. It will differ from sky values in the main report.
SNR = peak (r &lt; 30%) ÷ σ(outer annulus).</p>
"""


def _moffat_fwhm(gamma: float, alpha: float) -> float:
    return 2.0 * gamma * math.sqrt(2.0 ** (1.0 / alpha) - 1.0)


# Metrics table rows: (display label, result key A, result key B, number format)
_METRIC_ROWS = [
    ("Peak (ADU)",       "peak_a",    "peak_b",    ".3g"),
    ("Background (ADU)", "bg_a",      "bg_b",      ".3g"),
    ("SNR",              "snr_a",     "snr_b",     ".1f"),
    ("FWHM (px)",        "fwhm_px_a", "fwhm_px_b", ".2f"),
    ("FWHM (arcsec)",    "fwhm_as_a", "fwhm_as_b", ".2f"),
    ("Moffat β",         "beta_a",    "beta_b",    ".2f"),
    ("Eccentricity",     "ecc_a",     "ecc_b",     ".3f"),
    ("Ellipticity",      "ell_a",     "ell_b",     ".3f"),
    ("Orientation (°)",  "orient_a",  "orient_b",  ".1f"),
]


# ---------------------------------------------------------------------------
# _StarImageLabel
# ---------------------------------------------------------------------------

class _StarImageLabel(ZoomableImageLabel):
    """ZoomableImageLabel extended with left-click star selection and a circle overlay."""

    star_clicked = pyqtSignal(float, float)   # emits full-resolution image x, y

    def __init__(self, parent=None):
        super().__init__(parent)
        self._star_circle: tuple[float, float, float] | None = None  # (xn, yn, rn) normalised

    def set_star_circle(self, xn: float, yn: float, rn: float) -> None:
        self._star_circle = (xn, yn, rn)
        self.update()

    def clear_star_circle(self) -> None:
        self._star_circle = None
        self.update()

    def set_image_array(self, arr: np.ndarray) -> None:
        """Override to store the full-resolution pixmap — no downsampling.

        The base class caps the stored pixmap at 1024 px (MAX_DISPLAY_PX) for
        speed in the main image panels.  Here we need native resolution so that
        zooming in stays sharp rather than upscaling a thumbnail.
        """
        self._full_image_shape = arr.shape[:2]
        self._roi_norm = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            self._display_arr = np.ascontiguousarray(arr.astype(np.uint8))
            qimg = QImage(self._display_arr.data, w, h, w,
                          QImage.Format.Format_Grayscale8)
        else:
            self._display_arr = np.ascontiguousarray(arr.astype(np.uint8))
            qimg = QImage(self._display_arr.data, w, h, w * 3,
                          QImage.Format.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qimg)
        self.update()

    def mousePressEvent(self, event) -> None:
        # Left click outside tool modes → star selection
        if (event.button() == Qt.MouseButton.LeftButton
                and not self._roi_mode and not self._line_mode):
            norm = self._widget_to_norm(event.pos())
            if norm and self._full_image_shape:
                H, W = self._full_image_shape
                self.star_clicked.emit(norm[0] * W, norm[1] * H)
            return
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._star_circle is None:
            return
        xn, yn, rn = self._star_circle
        r = self._view_rect()
        if r is None:
            return
        ox, oy, pw, ph = r
        cx = ox + xn * pw
        cy = oy + yn * ph
        # pw and ph have the same scale (aspect-locked), so circle stays circular
        r_widget = rn * pw
        p = QPainter(self)
        p.setPen(QPen(QColor("#17becf"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r_widget, r_widget)
        p.end()


# ---------------------------------------------------------------------------
# _DetectThread — one-shot star detection on dialog open
# ---------------------------------------------------------------------------

class _DetectThread(QThread):
    """Detect stars in Image A (and optionally B) without blocking the GUI."""

    # ndarray Nx3 (x,y,peak) for A; ndarray Nx3 or None for B
    finished = pyqtSignal(object, object)

    def __init__(self, img_a: AstroImage, img_b: AstroImage | None, parent=None):
        super().__init__(parent)
        self._img_a = img_a
        self._img_b = img_b

    def run(self) -> None:
        from analysis.star_catalog import StarCatalogBuilder
        builder = StarCatalogBuilder()
        arr_a = self._detect(builder, self._img_a)
        arr_b = self._detect(builder, self._img_b) if self._img_b is not None else None
        self.finished.emit(arr_a, arr_b)

    @staticmethod
    def _detect(builder, img: AstroImage) -> np.ndarray:
        try:
            img.estimate_background()
            cat = builder.build(img)
            if len(cat) > 0:
                return np.column_stack([
                    np.array(cat["x_centroid"], dtype=float),
                    np.array(cat["y_centroid"], dtype=float),
                    np.array(cat["peak"],       dtype=float),
                ])
        except Exception:
            pass
        return np.empty((0, 3), dtype=float)


# ---------------------------------------------------------------------------
# Shared analysis helpers
# ---------------------------------------------------------------------------

def _extract_cutout(data: np.ndarray, xc: float, yc: float, radius: int) -> np.ndarray:
    """Return a (2r × 2r) patch centred on (xc, yc), zero-padded at image edges."""
    h, w = data.shape
    r = radius
    y0r, y1r = int(round(yc)) - r, int(round(yc)) + r
    x0r, x1r = int(round(xc)) - r, int(round(xc)) + r
    y0c, y1c = max(0, y0r), min(h, y1r)
    x0c, x1c = max(0, x0r), min(w, x1r)
    out = np.zeros((2 * r, 2 * r), dtype=data.dtype)
    patch = data[y0c:y1c, x0c:x1c]
    out[y0c - y0r:y0c - y0r + patch.shape[0],
        x0c - x0r:x0c - x0r + patch.shape[1]] = patch
    return out


def _annular_rdf(log_data: np.ndarray, xc: float, yc: float,
                  radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """1-px annular mean/std in log10 space — mirrors HaloAnalyzer._annular_stats."""
    h, w = log_data.shape
    x0 = max(0, int(xc - radius))
    y0 = max(0, int(yc - radius))
    x1 = min(w, int(xc + radius + 1))
    y1 = min(h, int(yc + radius + 1))
    sub = log_data[y0:y1, x0:x1].astype(float)
    yg, xg = np.mgrid[y0:y1, x0:x1]
    r_map = np.sqrt((xg - xc) ** 2 + (yg - yc) ** 2)

    r_centers, means, stds = [], [], []
    for r_edge in np.arange(0.0, float(radius), 1.0):
        mask = (r_map >= r_edge) & (r_map < r_edge + 1.0)
        if mask.sum() > 2:
            vals = sub[mask]
            r_centers.append(r_edge + 0.5)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

    if len(r_centers) < 3:
        return None
    return np.array(r_centers), np.array(means), np.array(stds)


def _refine_star_center(data: np.ndarray, sat_threshold: float,
                        x: float, y: float,
                        search_r: int = 20) -> tuple[float, float]:
    """Return the true centroid of the star nearest to (x, y).

    Mirrors the logic in HaloAnalyzer._collect_saturated_stars:
    - Saturated stars: connected-component centroid of the blown-out pixel
      mask — handles flat-topped cores where the peak is ambiguous.
    - Unsaturated stars: centroid of the 85 %-of-peak cluster, which is
      more robust than a single peak pixel for slightly asymmetric PSFs.

    Returns the original (x, y) unchanged when the patch is empty or dark.
    """
    from scipy.ndimage import label as _label

    xi, yi = int(round(x)), int(round(y))
    y0 = max(0, yi - search_r)
    y1 = min(data.shape[0], yi + search_r + 1)
    x0 = max(0, xi - search_r)
    x1 = min(data.shape[1], xi + search_r + 1)
    patch = data[y0:y1, x0:x1]
    if patch.size == 0:
        return x, y

    # Saturated case: CC centroid of the clipped region (same as _collect_saturated_stars)
    sat_mask = patch >= sat_threshold
    if sat_mask.any():
        labeled, n_feat = _label(sat_mask)
        best_cx, best_cy, best_dist = x, y, float("inf")
        for region_idx in range(1, n_feat + 1):
            ys_r, xs_r = np.where(labeled == region_idx)
            if len(xs_r) < 4:   # skip hot pixels / cosmic rays
                continue
            cx_r = float(np.mean(xs_r)) + x0
            cy_r = float(np.mean(ys_r)) + y0
            dist  = float(np.hypot(cx_r - x, cy_r - y))
            if dist < best_dist:
                best_dist, best_cx, best_cy = dist, cx_r, cy_r
        if best_dist < float("inf"):
            return best_cx, best_cy

    # Unsaturated case: centroid of the 85 %-of-peak cluster
    peak = float(np.nanmax(patch))
    if peak <= 0:
        return x, y
    mask = patch >= 0.85 * peak
    ys, xs = np.where(mask)
    if ys.size == 0:
        return x, y
    return float(np.mean(xs)) + x0, float(np.mean(ys)) + y0


def _rdf_bg_level(rdf_m: np.ndarray) -> float | None:
    """Log10 background noise floor = mean of the outermost 20 % of the RDF profile."""
    if rdf_m is None or len(rdf_m) < 5:
        return None
    n_tail = max(3, len(rdf_m) // 5)
    return float(np.mean(rdf_m[-n_tail:]))


def _xs_bg_level(xs: np.ndarray) -> float | None:
    """Background level in a normalised cross-section = mean of the outer edge bins."""
    if xs is None or len(xs) < 6:
        return None
    n = max(2, len(xs) // 8)
    return float(np.mean(np.concatenate([xs[:n], xs[-n:]])))


def _compute_bg_ring_radius(rdf_r: np.ndarray,
                             rdf_m: np.ndarray) -> float | None:
    """Return the first pixel radius where the RDF drops to the background noise floor.

    The background level is estimated as the mean of the outermost 20 % of the
    profile (the plateau where the star's contribution is negligible).  The ring
    radius is the smallest r at which the normalised log10 profile first falls to
    that level — i.e. the inner edge of the noise floor.
    """
    bg_level = _rdf_bg_level(rdf_m)
    if bg_level is None or rdf_r is None:
        return None
    for r_val, m_val in zip(rdf_r, rdf_m):
        if m_val <= bg_level:
            return float(r_val)
    return None


# ---------------------------------------------------------------------------
# _AnalyzeThread — per-click PSF fit + RDF computation
# ---------------------------------------------------------------------------

class _AnalyzeThread(QThread):
    """Runs Moffat fit, morphological metrics, cross-section, and RDF for one star."""

    finished = pyqtSignal(object)   # emits result dict
    error    = pyqtSignal(str)

    def __init__(self,
                 img_a: AstroImage,
                 img_b: AstroImage | None,
                 star_xy: tuple[float, float],
                 secondary_stars: np.ndarray | None,
                 radius_px: int,
                 primary_is_a: bool = True,
                 parent=None):
        super().__init__(parent)
        self._img_a        = img_a
        self._img_b        = img_b
        self._xc_p, self._yc_p = star_xy
        self._secondary    = secondary_stars
        self._radius       = radius_px
        self._primary_is_a = primary_is_a

    def run(self) -> None:
        try:
            self.finished.emit(self._analyze())
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------

    def _analyze(self) -> dict:
        r = self._radius
        xc_p, yc_p = self._xc_p, self._yc_p

        if self._primary_is_a:
            res_a = self._analyze_star(self._img_a, xc_p, yc_p, r)
            res_b = None
            if (self._img_b is not None and self._secondary is not None
                    and len(self._secondary) > 0):
                dists = np.hypot(self._secondary[:, 0] - xc_p,
                                 self._secondary[:, 1] - yc_p)
                idx = int(np.argmin(dists))
                if dists[idx] <= 50.0:
                    xc_b_cat = float(self._secondary[idx, 0])
                    yc_b_cat = float(self._secondary[idx, 1])
                    # Refine — catalog centroid may be offset for saturated B star
                    xc_b, yc_b = _refine_star_center(
                        self._img_b.data,
                        float(self._img_b.saturation_threshold()),
                        xc_b_cat, yc_b_cat, search_r=max(15, r // 3))
                    res_b = self._analyze_star(self._img_b, xc_b, yc_b, r)
        else:
            # Clicked in Image B view; self._secondary holds Image A catalog
            res_b = self._analyze_star(self._img_b, xc_p, yc_p, r)
            res_a = None
            if self._secondary is not None and len(self._secondary) > 0:
                dists = np.hypot(self._secondary[:, 0] - xc_p,
                                 self._secondary[:, 1] - yc_p)
                idx = int(np.argmin(dists))
                if dists[idx] <= 50.0:
                    xc_a_cat = float(self._secondary[idx, 0])
                    yc_a_cat = float(self._secondary[idx, 1])
                    xc_a, yc_a = _refine_star_center(
                        self._img_a.data,
                        float(self._img_a.saturation_threshold()),
                        xc_a_cat, yc_a_cat, search_r=max(15, r // 3))
                    res_a = self._analyze_star(self._img_a, xc_a, yc_a, r)

        def _g(d, k):  return d[k]       if d else None
        def _gd(d, k): return d.get(k)   if d else None

        xs_a_raw = _g(res_a, "xs_raw")
        xs_b_raw = _g(res_b, "xs_raw")

        max_a = float(np.nanmax(xs_a_raw)) if xs_a_raw is not None and xs_a_raw.size > 0 else 1.0
        max_b = float(np.nanmax(xs_b_raw)) if xs_b_raw is not None and xs_b_raw.size > 0 else 0.0
        shared_max = max(max_a, max_b, 1.0)
        xs_a = np.clip(xs_a_raw / shared_max, 1e-6, None) if xs_a_raw is not None else None
        xs_b = np.clip(xs_b_raw / shared_max, 1e-6, None) if xs_b_raw is not None else None
        _ref = xs_a if xs_a is not None else xs_b
        px_offs = np.arange(len(_ref)) - len(_ref) // 2 if _ref is not None else np.array([])

        rdf_r_a = rdf_m_a = rdf_s_a = None
        rdf_a = _g(res_a, "rdf")
        if rdf_a is not None:
            rdf_r_a, m_a, rdf_s_a = rdf_a
            rdf_m_a = m_a - m_a[0]

        rdf_r_b = rdf_m_b = rdf_s_b = None
        rdf_b = _g(res_b, "rdf")
        if rdf_b is not None:
            rdf_r_b, m_b, rdf_s_b = rdf_b
            rdf_m_b = m_b - m_b[0]

        ps    = float(self._img_a.pixel_scale) if self._img_a.pixel_scale else 0.0
        fit_a = _g(res_a, "fit")
        fit_b = _g(res_b, "fit")
        # Saturated stars: _fit_moffat returns {"saturated": True} — show "Sat."
        # in table rows that depend on the core PSF fit; other metrics still valid
        sat_a = isinstance(fit_a, dict) and fit_a.get("saturated", False)
        sat_b = isinstance(fit_b, dict) and fit_b.get("saturated", False)
        fwhm_a = None if sat_a else _gd(fit_a, "fwhm")
        fwhm_b = None if sat_b else _gd(fit_b, "fwhm")
        beta_a = "Sat." if sat_a else _gd(fit_a, "alpha")
        beta_b = "Sat." if sat_b else _gd(fit_b, "alpha")
        fwhm_px_a = "Sat." if sat_a else fwhm_a
        fwhm_px_b = "Sat." if sat_b else fwhm_b
        fwhm_as_a = ("Sat." if sat_a
                     else (fwhm_a * ps if (fwhm_a is not None and ps > 0) else None))
        fwhm_as_b = ("Sat." if sat_b
                     else (fwhm_b * ps if (fwhm_b is not None and ps > 0) else None))

        return {
            "peak_a":    _g(res_a, "pk"),      "peak_b":    _g(res_b, "pk"),
            "bg_a":      _g(res_a, "bg"),      "bg_b":      _g(res_b, "bg"),
            "snr_a":     _g(res_a, "snr"),     "snr_b":     _g(res_b, "snr"),
            "fwhm_px_a": fwhm_px_a,            "fwhm_px_b": fwhm_px_b,
            "fwhm_as_a": fwhm_as_a,            "fwhm_as_b": fwhm_as_b,
            "beta_a":    beta_a,               "beta_b":    beta_b,
            "ecc_a":     _gd(_g(res_a, "shape"), "ecc"),
            "ecc_b":     _gd(_g(res_b, "shape"), "ecc"),
            "ell_a":     _gd(_g(res_a, "shape"), "ell"),
            "ell_b":     _gd(_g(res_b, "shape"), "ell"),
            "orient_a":  _gd(_g(res_a, "shape"), "orient"),
            "orient_b":  _gd(_g(res_b, "shape"), "orient"),
            "disp_a":    _g(res_a, "disp"),    "disp_b":    _g(res_b, "disp"),
            "cx_a": _g(res_a, "cx_cut"), "cy_a": _g(res_a, "cy_cut"),
            "cx_b": _g(res_b, "cx_cut"), "cy_b": _g(res_b, "cy_cut"),
            "xs_px":     px_offs,              "xs_a":      xs_a,     "xs_b":    xs_b,
            "rdf_r_a":   rdf_r_a,             "rdf_m_a":   rdf_m_a,  "rdf_s_a": rdf_s_a,
            "rdf_r_b":   rdf_r_b,             "rdf_m_b":   rdf_m_b,  "rdf_s_b": rdf_s_b,
            "pixel_scale": ps,
        }

    # ------------------------------------------------------------------

    def _analyze_star(self, img: AstroImage, xc: float, yc: float, r: int) -> dict:
        """Background subtraction, stats, Moffat fit, shape, and RDF for one star."""
        if img.background is None:
            img.estimate_background()
        bgsub    = img.background_subtracted()
        cut      = _extract_cutout(bgsub, xc, yc, r)
        bg, snr, pk = self._outer_stats(bgsub, xc, yc, r)
        fit      = self._fit_moffat(bgsub, xc, yc)
        shape    = self._shape_metrics(bgsub, xc, yc)
        xs_raw   = cut[cut.shape[0] // 2, :].astype(float)
        _pos     = bgsub[bgsub > 0]
        lf       = float(np.percentile(_pos, 1)) if _pos.size > 0 else 1.0
        log_data = np.log10(np.clip(bgsub, lf, None))
        rdf      = _annular_rdf(log_data, xc, yc, r)
        # Display cutout from raw data so the ROI shows the actual image appearance;
        # all analysis (fit, shape, RDF, cross-section) remains on bgsub
        raw_cut  = _extract_cutout(img.data, xc, yc, r)
        disp     = normalize_for_display(raw_cut.astype(np.float32))
        # Sub-pixel centroid position within the 2r×2r cutout (imshow data coords)
        cx_cut   = xc - (round(xc) - r)
        cy_cut   = yc - (round(yc) - r)
        return {
            "cut": cut, "bg": bg, "snr": snr, "pk": pk,
            "fit": fit, "shape": shape, "xs_raw": xs_raw, "rdf": rdf, "disp": disp,
            "cx_cut": cx_cut, "cy_cut": cy_cut,
        }

    # ------------------------------------------------------------------

    def _outer_stats(self, bgsub: np.ndarray, xc: float, yc: float,
                      r: int) -> tuple:
        """Background (median outer annulus), SNR, and peak of the inner core."""
        h, w = bgsub.shape
        x0 = max(0, int(xc - r))
        y0 = max(0, int(yc - r))
        x1 = min(w, int(xc + r + 1))
        y1 = min(h, int(yc + r + 1))
        sub  = bgsub[y0:y1, x0:x1].astype(float)
        yg, xg = np.mgrid[y0:y1, x0:x1]
        r_map = np.sqrt((xg - xc) ** 2 + (yg - yc) ** 2)

        outer = r_map >= r * 0.7
        if outer.sum() < 4:
            return None, None, None
        outer_vals = sub[outer]
        bg  = float(np.median(outer_vals))
        std = float(np.std(outer_vals))

        inner = r_map < r * 0.3
        if inner.sum() == 0:
            inner = np.ones(r_map.shape, dtype=bool)
        peak = float(np.nanmax(sub[inner]))
        snr  = peak / std if std > 0 else None
        return bg, snr, peak

    def _fit_moffat(self, bgsub: np.ndarray, xc: float, yc: float) -> dict | None:
        """Fit a 2D Moffat on the PSF core (25-px cutout, matches PSFAnalyzer size)."""
        half = 12
        h, w = bgsub.shape
        x0 = max(0, int(round(xc)) - half)
        y0 = max(0, int(round(yc)) - half)
        x1 = min(w, int(round(xc)) + half + 1)
        y1 = min(h, int(round(yc)) + half + 1)
        cut = bgsub[y0:y1, x0:x1].copy()
        if cut.size == 0:
            return None
        # Subtract median of outer 3-px border ring to remove galaxy/nebula gradient
        row_idx, col_idx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
        border = ((row_idx < 3) | (row_idx >= cut.shape[0] - 3) |
                  (col_idx < 3) | (col_idx >= cut.shape[1] - 3))
        if border.sum() > 0:
            cut = cut - float(np.median(cut[border]))
        cut = np.clip(cut, 0.0, None)
        amp = float(np.max(cut))
        if amp <= 0:
            return None
        # Detect flat-topped saturated core: if >25 % of the inner 7×7 region
        # is at ≥98 % of the peak the star is blown out; Moffat fit is not meaningful
        cy_c, cx_c = cut.shape[0] // 2, cut.shape[1] // 2
        r_i = 3   # 7×7 window
        inner = cut[max(0, cy_c - r_i):cy_c + r_i + 1,
                    max(0, cx_c - r_i):cx_c + r_i + 1]
        if inner.size > 0 and float(np.sum(inner >= 0.98 * amp)) / inner.size > 0.25:
            return {"saturated": True}
        cy, cx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
        model  = Moffat2D(amplitude=amp,
                          x_0=cut.shape[1] / 2.0, y_0=cut.shape[0] / 2.0,
                          gamma=2.0, alpha=2.5)
        fitter = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted = fitter(model, cx, cy, cut)
            except Exception:
                return None
        gamma = abs(fitted.gamma.value)
        alpha = abs(fitted.alpha.value)
        if not (0.1 <= alpha <= 50.0) or gamma < 0.05:
            return None
        fwhm = _moffat_fwhm(gamma, alpha)
        if not (0.2 <= fwhm <= 100.0):
            return None
        return {"fwhm": fwhm, "alpha": alpha, "gamma": gamma}

    def _shape_metrics(self, bgsub: np.ndarray, xc: float, yc: float) -> dict | None:
        """Eccentricity, ellipticity, orientation via photutils data_properties.

        Same approach as PSFAnalyzer._measure_ellipticity (psf_analyzer.py:332).
        """
        from photutils.morphology import data_properties
        half = 12
        h, w = bgsub.shape
        x0 = max(0, int(round(xc)) - half)
        y0 = max(0, int(round(yc)) - half)
        x1 = min(w, int(round(xc)) + half + 1)
        y1 = min(h, int(round(yc)) + half + 1)
        cut = np.clip(bgsub[y0:y1, x0:x1].copy(), 0, None)
        if cut.sum() == 0:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                props = data_properties(cut)
            return {
                "ecc":    float(props.eccentricity.value),
                "ell":    float(props.ellipticity.value),
                "orient": float(props.orientation.value) % 180,   # degrees in photutils 3.x; normalised to [0, 180)
            }
        except Exception:
            return None


# ---------------------------------------------------------------------------
# HaloAnalyzerDialog
# ---------------------------------------------------------------------------

class HaloAnalyzerDialog(QDialog):
    """Interactive halo/PSF characterisation tool.

    Load Image A, open via Tools → Halo Analyzer, click any star.
    A sample-radius circle is drawn on the image; the 4-panel figure and
    metrics table update automatically.
    """

    def __init__(self, img_a: AstroImage, img_b: AstroImage | None = None,
                 dark_mode: bool = False, parent=None):
        super().__init__(parent)
        self._img_a = img_a
        self._img_b = img_b
        self._dark_mode = dark_mode
        self._stars_a: np.ndarray | None = None
        self._stars_b: np.ndarray | None = None
        self._selected_xy: tuple[float, float] | None = None
        self._display_img: str = "A"
        self._detect_thread: _DetectThread | None = None
        self._analyze_thread: _AnalyzeThread | None = None
        self._last_result: dict | None = None

        self.setWindowTitle("Halo Analyzer")
        self.resize(1200, 850)
        self._build_ui()
        self._start_detection()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(4)

        # Image label created first so toolbar can connect to its slot
        self._star_label = _StarImageLabel()
        self._star_label.star_clicked.connect(self._on_star_clicked)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.setFixedWidth(90)
        reset_btn.clicked.connect(self._star_label.reset_zoom)
        toolbar.addWidget(reset_btn)
        toolbar.addSpacing(8)

        # Image A/B display selector
        toolbar.addWidget(QLabel("Display:"))
        self._img_selector = QComboBox()
        self._img_selector.addItem(f"Image A  ({Path(self._img_a.path).name})")
        if self._img_b is not None:
            self._img_selector.addItem(f"Image B  ({Path(self._img_b.path).name})")
        self._img_selector.currentIndexChanged.connect(self._on_display_changed)
        toolbar.addWidget(self._img_selector)
        toolbar.addSpacing(12)

        toolbar.addWidget(QLabel("Sample radius:"))
        self._radius_spin = QSpinBox()
        self._radius_spin.setRange(10, 500)
        self._radius_spin.setValue(50)
        self._radius_spin.setSuffix(" px")
        self._radius_spin.valueChanged.connect(self._on_radius_changed)
        toolbar.addWidget(self._radius_spin)
        toolbar.addSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)        # indeterminate / marquee
        self._progress_bar.setFixedWidth(120)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setVisible(False)
        toolbar.addWidget(self._progress_bar)
        toolbar.addSpacing(4)

        self._status_lbl = QLabel("Detecting stars…")
        toolbar.addWidget(self._status_lbl)
        toolbar.addSpacing(16)

        self._bg_ring_chk = QCheckBox("Draw Background Ring")
        self._bg_ring_chk.setChecked(True)
        self._bg_ring_chk.stateChanged.connect(self._on_bg_ring_toggled)
        toolbar.addWidget(self._bg_ring_chk)
        toolbar.addStretch()
        main.addLayout(toolbar)

        # --- Middle: image (left) + metrics table (right) ---
        mid_splitter = QSplitter(Qt.Orientation.Horizontal)

        img_container = QWidget()
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(2)
        self._img_header_lbl = QLabel(f"<b>{self._img_a.label}</b>")
        img_layout.addWidget(self._img_header_lbl)
        img_layout.addWidget(self._star_label, stretch=1)
        mid_splitter.addWidget(img_container)

        self._table = self._build_table()

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self._table, stretch=1)

        self._method_box = QTextEdit()
        self._method_box.setReadOnly(True)
        self._method_box.setHtml(_PROCESSING_NOTES_HTML)
        self._method_box.setMaximumHeight(150)
        right_layout.addWidget(self._method_box, stretch=0)

        mid_splitter.addWidget(right_widget)
        mid_splitter.setSizes([750, 350])

        # --- Vertical splitter: middle area above, matplotlib figure below ---
        self._fig = Figure(figsize=(12, 3))
        self._canvas = FigureCanvasQTAgg(self._fig)

        vert_splitter = QSplitter(Qt.Orientation.Vertical)
        vert_splitter.addWidget(mid_splitter)
        vert_splitter.addWidget(self._canvas)
        vert_splitter.setSizes([530, 280])
        main.addWidget(vert_splitter, stretch=1)

        # Display Image A immediately (background estimation runs in thread)
        try:
            self._star_label.set_image_array(self._img_a.display_image(stretch=True))
        except Exception:
            pass

    def _build_table(self) -> QTableWidget:
        fname_a = Path(self._img_a.path).name
        fname_b = Path(self._img_b.path).name if self._img_b else "Image B"

        table = QTableWidget(len(_METRIC_ROWS), 3)
        table.setHorizontalHeaderLabels(["", fname_a, fname_b])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.setMinimumWidth(280)

        for i, (label, _, _, _) in enumerate(_METRIC_ROWS):
            lbl_item = QTableWidgetItem(label)
            lbl_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            table.setItem(i, 0, lbl_item)
            for col in (1, 2):
                cell = QTableWidgetItem("—")
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table.setItem(i, col, cell)
        return table

    # ------------------------------------------------------------------
    # Star detection
    # ------------------------------------------------------------------

    def _start_detection(self) -> None:
        self._progress_bar.setVisible(True)
        self._detect_thread = _DetectThread(self._img_a, self._img_b, parent=self)
        self._detect_thread.finished.connect(self._on_detect_done)
        self._detect_thread.start()

    def _on_detect_done(self, arr_a: np.ndarray, arr_b: object) -> None:
        self._stars_a = arr_a
        self._stars_b = arr_b   # ndarray Nx3 or None
        n = len(arr_a) if arr_a is not None else 0
        if n > 0:
            self._status_lbl.setText(f"{n} stars found — click to select")
        else:
            self._status_lbl.setText("No stars detected")
        self._progress_bar.setVisible(False)
        self._detect_thread = None

    # ------------------------------------------------------------------
    # Star selection
    # ------------------------------------------------------------------

    def _on_star_clicked(self, x: float, y: float) -> None:
        img    = self._img_b if self._display_img == "B" else self._img_a
        stars  = self._stars_b if self._display_img == "B" else self._stars_a
        radius = self._radius_spin.value()
        search_r = max(15, radius // 3)
        sat = float(img.saturation_threshold())

        # Gate 1 (normal stars): nearest catalog entry within 2 × sample-radius
        near_catalog = False
        if stars is not None and len(stars) > 0:
            dists = np.hypot(stars[:, 0] - x, stars[:, 1] - y)
            near_catalog = float(dists[int(np.argmin(dists))]) <= 2 * radius

        # Gate 2 (saturated / catalog-missed stars): check local brightness
        if not near_catalog:
            xi, yi = int(round(x)), int(round(y))
            y0 = max(0, yi - search_r); y1 = min(img.data.shape[0], yi + search_r + 1)
            x0 = max(0, xi - search_r); x1 = min(img.data.shape[1], xi + search_r + 1)
            patch = img.data[y0:y1, x0:x1]
            if patch.size == 0:
                return
            local_peak = float(np.nanmax(patch))
            if local_peak < sat:
                # Not saturated — use robust sky+5σ threshold on a sparse image sample
                step  = max(1, img.data.size // 10000)
                samp  = img.data.flat[::step]
                sky   = float(np.nanpercentile(samp, 25))          # ~sky level
                sky_s = (float(np.nanpercentile(samp, 75)) - sky) / 0.674  # robust σ
                if local_peak < sky + 5.0 * sky_s:
                    return   # genuine empty-sky click

        # Refine to true star centroid — CC centroid for saturated flat tops,
        # 85%-of-peak cluster centroid for normal stars (mirrors _collect_saturated_stars)
        sx, sy = _refine_star_center(img.data, sat, x, y, search_r=search_r)

        self._selected_xy = (sx, sy)
        self._draw_circle(sx, sy, radius)
        self._run_analysis(sx, sy)

    def _on_radius_changed(self, value: int) -> None:
        if self._selected_xy is None:
            return
        sx, sy = self._selected_xy
        self._draw_circle(sx, sy, value)
        self._run_analysis(sx, sy)

    def _on_bg_ring_toggled(self) -> None:
        if self._last_result is not None:
            self._update_figure(self._last_result)

    def _on_display_changed(self, idx: int) -> None:
        self._display_img = "B" if idx == 1 else "A"
        img = self._img_b if self._display_img == "B" else self._img_a
        self._img_header_lbl.setText(f"<b>{img.label}</b>")
        self._star_label.clear_star_circle()
        self._selected_xy = None
        try:
            self._star_label.set_image_array(img.display_image(stretch=True))
        except Exception:
            pass

    def _draw_circle(self, sx: float, sy: float, radius: int) -> None:
        if self._star_label._full_image_shape is None:
            return
        H, W = self._star_label._full_image_shape
        self._star_label.set_star_circle(sx / W, sy / H, radius / W)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _run_analysis(self, xc: float, yc: float) -> None:
        # Cancel previous thread (disconnect signals so its result is discarded)
        if self._analyze_thread is not None and self._analyze_thread.isRunning():
            try:
                self._analyze_thread.finished.disconnect()
                self._analyze_thread.error.disconnect()
            except RuntimeError:
                pass
            self._analyze_thread.quit()

        self._status_lbl.setText("Analyzing…")
        self._progress_bar.setVisible(True)
        # When Image B is displayed, pass stars_a as the secondary catalog
        secondary = self._stars_a if self._display_img == "B" else self._stars_b
        self._analyze_thread = _AnalyzeThread(
            self._img_a, self._img_b,
            (xc, yc),
            secondary,
            self._radius_spin.value(),
            primary_is_a=(self._display_img == "A"),
            parent=self,
        )
        self._analyze_thread.finished.connect(self._on_analysis_done)
        self._analyze_thread.error.connect(self._on_analysis_error)
        self._analyze_thread.start()

    def _on_analysis_done(self, result: dict) -> None:
        self._analyze_thread = None
        self._last_result = result
        self._progress_bar.setVisible(False)
        n = len(self._stars_a) if self._stars_a is not None else 0
        self._status_lbl.setText(f"{n} stars found — click to select")
        self._update_table(result)
        self._update_figure(result)

    def _on_analysis_error(self, msg: str) -> None:
        self._analyze_thread = None
        self._progress_bar.setVisible(False)
        self._status_lbl.setText(f"Error: {msg[:80]}")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _update_table(self, result: dict) -> None:
        for row, (_, key_a, key_b, fmt) in enumerate(_METRIC_ROWS):
            for col, key in ((1, key_a), (2, key_b)):
                val = result.get(key)
                if val is None:
                    text = "—"
                else:
                    try:
                        text = format(float(val), fmt)
                    except (TypeError, ValueError):
                        text = str(val)
                item = self._table.item(row, col)
                if item:
                    item.setText(text)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _update_figure(self, result: dict) -> None:
        import matplotlib.pyplot as _plt
        _is_dark = self._dark_mode
        orig_color = "white" if _is_dark else "black"

        _saved_params = matplotlib.rcParams.copy()
        if _is_dark:
            _plt.style.use("dark_background")
        try:
            self._fig.clear()
            if _is_dark:
                bg = matplotlib.rcParams.get("figure.facecolor", "#121212")
                self._fig.patch.set_facecolor(bg)
                self._canvas.setStyleSheet(f"background-color: {bg};")
            else:
                self._fig.patch.set_facecolor("white")
                self._canvas.setStyleSheet("")
            axes = self._fig.subplots(1, 4)
            ax0, ax1, ax2, ax3 = axes

            # --- Cutout A ---
            disp_a = result.get("disp_a")
            cx_a = result.get("cx_a")
            cy_a = result.get("cy_a")
            if disp_a is not None:
                ax0.imshow(disp_a, origin="upper", cmap="turbo",
                           vmin=0, vmax=255, aspect="equal")
                if cx_a is not None and cy_a is not None:
                    ax0.plot(cx_a, cy_a, '+', color='magenta',
                             markersize=10, markeredgewidth=1.5, zorder=5)
            ax0.axis("off")
            ax0.set_title(self._img_a.label, fontsize=7, pad=2, color=orig_color)

            # --- Cutout B ---
            disp_b = result.get("disp_b")
            cx_b = result.get("cx_b")
            cy_b = result.get("cy_b")
            if disp_b is not None:
                ax1.imshow(disp_b, origin="upper", cmap="turbo",
                           vmin=0, vmax=255, aspect="equal")
                if cx_b is not None and cy_b is not None:
                    ax1.plot(cx_b, cy_b, '+', color='magenta',
                             markersize=10, markeredgewidth=1.5, zorder=5)
                b_lbl = self._img_b.label if self._img_b else "B"
                ax1.set_title(b_lbl, fontsize=7, pad=2, color=orig_color)
            else:
                msg = "No Image B" if self._img_b is None else "No match in B"
                ax1.text(0.5, 0.5, msg, ha="center", va="center",
                         transform=ax1.transAxes, fontsize=8, alpha=0.5,
                         color=orig_color)
            ax1.axis("off")

            # --- Background ring overlays ---
            if self._bg_ring_chk.isChecked():
                bg_r_a = _compute_bg_ring_radius(
                    result.get("rdf_r_a"), result.get("rdf_m_a"))
                if bg_r_a is not None and cx_a is not None and cy_a is not None:
                    ax0.add_patch(matplotlib.patches.Circle(
                        (cx_a, cy_a), bg_r_a,
                        fill=False, edgecolor="magenta", linewidth=1.5,
                        alpha=0.6, zorder=6))
                bg_r_b = _compute_bg_ring_radius(
                    result.get("rdf_r_b"), result.get("rdf_m_b"))
                if bg_r_b is not None and cx_b is not None and cy_b is not None:
                    ax1.add_patch(matplotlib.patches.Circle(
                        (cx_b, cy_b), bg_r_b,
                        fill=False, edgecolor="magenta", linewidth=1.5,
                        alpha=0.6, zorder=6))

            # --- Cross-section ---
            xs_px = result.get("xs_px")
            xs_a  = result.get("xs_a")
            xs_b  = result.get("xs_b")
            ps    = result.get("pixel_scale", 0.0)
            if xs_px is not None and xs_a is not None:
                ax2.semilogy(xs_px, xs_a, color="steelblue", linewidth=1.2,
                             label=self._img_a.label)
                if xs_b is not None:
                    b_lbl = self._img_b.label if self._img_b else "B"
                    ax2.semilogy(xs_px, xs_b, color="tomato", linewidth=1.2,
                                 label=b_lbl)
                # Background level lines
                for _xs in [xs_a, xs_b]:
                    _lvl = _xs_bg_level(_xs)
                    if _lvl is not None and _lvl > 0:
                        ax2.axhline(_lvl, color="magenta", linestyle="--",
                                    linewidth=1.0, alpha=0.6)
                ax2.set_xlabel("px from centre", fontsize=7, color=orig_color)
                ax2.set_ylabel("Norm. intensity", fontsize=7, color=orig_color)
                ax2.tick_params(labelsize=6, colors=orig_color)
                ax2.grid(True, alpha=0.3)
                if ps and ps > 0:
                    _ps = ps
                    ax2_top = ax2.secondary_xaxis(
                        "top",
                        functions=(lambda x, p=_ps: x * p, lambda x, p=_ps: x / p))
                    ax2_top.set_xlabel("arcsec from centre", fontsize=7, color=orig_color)
                    ax2_top.tick_params(labelsize=6, colors=orig_color)

            # --- RDF ---
            rdf_r_a = result.get("rdf_r_a")
            rdf_m_a = result.get("rdf_m_a")
            rdf_s_a = result.get("rdf_s_a")
            if rdf_r_a is not None and rdf_m_a is not None:
                y_a = 10 ** rdf_m_a
                ax3.semilogy(rdf_r_a, y_a, color="steelblue", linewidth=1.2,
                             label=self._img_a.label)
                if rdf_s_a is not None:
                    ax3.fill_between(rdf_r_a,
                                     10 ** (rdf_m_a - rdf_s_a),
                                     10 ** (rdf_m_a + rdf_s_a),
                                     color="steelblue", alpha=0.2)
                # Background level line for Image A
                _rdf_bg_a = _rdf_bg_level(rdf_m_a)
                if _rdf_bg_a is not None:
                    ax3.axhline(10 ** _rdf_bg_a, color="magenta", linestyle="--",
                                linewidth=1.0, alpha=0.6)

                rdf_r_b = result.get("rdf_r_b")
                rdf_m_b = result.get("rdf_m_b")
                rdf_s_b = result.get("rdf_s_b")
                if rdf_r_b is not None and rdf_m_b is not None:
                    y_b = 10 ** rdf_m_b
                    b_lbl = self._img_b.label if self._img_b else "B"
                    ax3.semilogy(rdf_r_b, y_b, color="tomato", linewidth=1.2,
                                 label=b_lbl)
                    if rdf_s_b is not None:
                        ax3.fill_between(rdf_r_b,
                                         10 ** (rdf_m_b - rdf_s_b),
                                         10 ** (rdf_m_b + rdf_s_b),
                                         color="tomato", alpha=0.2)
                    # Background level line for Image B
                    _rdf_bg_b = _rdf_bg_level(rdf_m_b)
                    if _rdf_bg_b is not None:
                        ax3.axhline(10 ** _rdf_bg_b, color="magenta", linestyle="--",
                                    linewidth=1.0, alpha=0.6)

                ax3.set_xlabel("Radius (px)", fontsize=7, color=orig_color)
                ax3.set_ylabel("Relative intensity", fontsize=7, color=orig_color)
                ax3.tick_params(labelsize=6, colors=orig_color)
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=6, loc="upper right")
                if ps and ps > 0:
                    _ps = ps
                    ax3_top = ax3.secondary_xaxis(
                        "top",
                        functions=(lambda x, p=_ps: x * p, lambda x, p=_ps: x / p))
                    ax3_top.set_xlabel("Radius (arcsec)", fontsize=7, color=orig_color)
                    ax3_top.tick_params(labelsize=6, colors=orig_color)

            self._fig.tight_layout(pad=1.0)
            self._canvas.draw_idle()
        finally:
            matplotlib.rcParams.update(_saved_params)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        for thread in (self._detect_thread, self._analyze_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(2000)
        super().closeEvent(event)
