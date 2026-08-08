"""Reusable pyqtgraph widgets for the Data Inspector.

Kept separate from gui/data_inspector.py so the window module stays about layout and
wiring while the drawing primitives stay independently readable.

Drawing only: every array formula this module needs — luma reduction, the shared
display range, the A-vs-B comparison and its colour range — lives in
analysis/inspector_regions.py, which is headless and therefore testable.  Nothing
here should recompute one of them; import it.
"""
from __future__ import annotations

import os

# Pin the Qt binding before pyqtgraph's shim probes for one.  Without this a frozen
# build that happens to see another binding on the path can bind to the wrong Qt.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from analysis.inspector_regions import (
    COMPARE_DIFFERENCE, COMPARE_LOGRATIO, ROI_ELLIPSE, ROI_POLYGON, ROI_RECT,
    comparison_map, comparison_range, select_points_in_polygon, to_2d, value_range,
)
from core.models import SECTION8_ANALYSIS_CMAP

# arr[row, col] -> (y, x) with no transposes anywhere in this module.
pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("antialias", False)   # large images/scatters; quality not needed

# Curve colours match the report's cross-section convention
# (analysis/image_filters.py::_draw_cross_section).
COLOR_A = "#4682b4"        # steelblue
COLOR_B = "#ff6347"        # tomato
COLOR_COMPARE = "#ffa500"  # orange — the opt-in right-axis comparison curve

# Default cross-section line in normalised [0,1] coords — a centred diagonal.
DEFAULT_LINE = (0.25, 0.5, 0.75, 0.5)

# Overlay defaults: magenta marks the pixels feeding the in-mask correlation plot,
# cyan marks pixels lassoed in a correlation plot.  Both user-changeable.
OVERLAY_MASK_COLOR = "#ff00ff"
OVERLAY_SELECTION_COLOR = "#00ffff"
OVERLAY_DEFAULT_ALPHA = 115          # 0-255; ~45%, matching the report's mask figure

_DARK_BG = "#1e1e1e"
_DARK_FG = "#dddddd"
_LIGHT_BG = "#ffffff"
_LIGHT_FG = "#202020"


def theme_colors(dark: bool) -> tuple[str, str]:
    """(background, foreground) for a widget.

    Deliberately returned per-call rather than set through
    pg.setConfigOption('background'/'foreground'), which is process-global state — the
    same mistake as the Report Inspector's import-time matplotlib rcParams mutation.
    """
    return (_DARK_BG, _DARK_FG) if dark else (_LIGHT_BG, _LIGHT_FG)


def style_plot_item(plot_item: pg.PlotItem, fg: str) -> None:
    """Apply the foreground colour to a PlotItem's axes, labels and title."""
    for name in ("left", "bottom", "right", "top"):
        ax = plot_item.getAxis(name)
        if ax is not None:
            ax.setPen(pg.mkPen(fg))
            ax.setTextPen(pg.mkPen(fg))
    plot_item.titleLabel.setAttr("color", fg)


# Human-readable names for the comparison modes.  The modes themselves and all the
# arithmetic behind them live in analysis/inspector_regions.py, where the headless
# suite can test them; only the display strings belong to the GUI.
COMPARE_LABELS = {
    COMPARE_LOGRATIO:   "Log ratio, log10(|A|/|B|)",
    COMPARE_DIFFERENCE: "Difference, A − B",
}


# ---------------------------------------------------------------------------
# Image view
# ---------------------------------------------------------------------------

class LinkedImageView(QWidget):
    """One image panel: base ImageItem, colour bar, and a cross-section line.

    Optionally also an A/B swipe overlay — a second ImageItem showing Image B to the
    right of a draggable divider, for before/after comparison in a single panel.

    The line and the divider are both stored and emitted in normalised [0,1] image
    coordinates — the same convention gui/image_panel.py uses — so they survive a
    switch to a panel of a different pixel size without ever falling out of bounds.

    Only one thing is draggable at a time, so a click is never ambiguous: the
    cross-section line's handles are enabled only while the cross-section tool is
    selected (see set_line_locked), and the divider exists only in swipe mode.
    """

    # x0n, y0n, x1n, y1n — normalised [0,1]
    line_changed = pyqtSignal(float, float, float, float)
    # divider position, normalised [0,1]
    swipe_changed = pyqtSignal(float)
    # ROI geometry dict in normalised coords (see analysis.inspector_regions.roi_mask)
    roi_changed = pyqtSignal(object)

    def __init__(self, title: str = "", dark: bool = False, parent=None):
        super().__init__(parent)
        bg, fg = theme_colors(dark)
        self._fg = fg
        self._arr: np.ndarray | None = None
        self._arr_swipe: np.ndarray | None = None
        self._swipe_pos: float = 0.5
        self._syncing_swipe = False

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(bg)

        self._plot = self._glw.addPlot(title=title)
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)          # row 0 at the top, matching origin="upper"
        self._plot.showAxes(False)
        self._plot.setMenuEnabled(False)
        style_plot_item(self._plot, fg)

        self._image = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._image)

        # A/B swipe overlay: Image B drawn on top of Image A, but only the strip to
        # the right of the divider.  Implemented by slicing B's own array and giving
        # the item a matching rect, so no compositing buffer is allocated per drag —
        # the slice is a numpy view.
        self._image_swipe = pg.ImageItem(axisOrder="row-major")
        self._image_swipe.setZValue(5)
        self._image_swipe.setVisible(False)
        self._plot.addItem(self._image_swipe)

        # Mask and selection overlays.  Each holds a uint8 {0,1} label image with a
        # 2-entry RGBA lookup table, so changing colour or opacity rewrites two LUT
        # rows instead of rebuilding an H x W x 4 buffer — and the overlay costs
        # 1 byte/px rather than 4.
        self._mask_overlay = pg.ImageItem(axisOrder="row-major")
        self._sel_overlay = pg.ImageItem(axisOrder="row-major")
        for item, z in ((self._mask_overlay, 8), (self._sel_overlay, 9)):
            item.setZValue(z)
            item.setVisible(False)
            self._plot.addItem(item)
        self._overlay_style = {
            "mask": (OVERLAY_MASK_COLOR, OVERLAY_DEFAULT_ALPHA),
            "selection": (OVERLAY_SELECTION_COLOR, OVERLAY_DEFAULT_ALPHA),
        }

        self._divider = pg.InfiniteLine(
            pos=0, angle=90, movable=True,
            pen=pg.mkPen("#ffffff", width=2),
            hoverPen=pg.mkPen("#ffe680", width=3),
        )
        self._divider.setZValue(25)
        self._divider.setVisible(False)
        self._plot.addItem(self._divider)
        self._divider.sigPositionChanged.connect(self._on_divider_dragged)

        self._colorbar = pg.ColorBarItem(width=12, interactive=True,
                                         colorMap=pg.colormap.get("viridis"))
        # Register BOTH image items: ColorBarItem drives every item in its img_list
        # from one set of levels and one colour map, so the two halves of an A/B wipe
        # can never drift onto different scales — including when the user drags the
        # bar's level handles.  (setImageItem replaces the list rather than appending,
        # so both must be passed in the same call.)
        self._colorbar.setImageItem([self._image, self._image_swipe],
                                    insert_in=self._plot)
        # Three of these sit side by side; a full-width bar would crowd the images.
        self._colorbar.getAxis("right").setStyle(tickTextWidth=28,
                                                 autoExpandTextSpace=False)

        # Cross-section line. Created once and repositioned, never rebuilt, so its
        # signal connections and drag state stay intact across panel changes.
        self._line = pg.LineSegmentROI(
            [[0, 0], [1, 1]],
            pen=pg.mkPen("#ff7f0e", width=2),
            hoverPen=pg.mkPen("#ffbb66", width=3),
            movable=True, rotatable=True, resizable=True, removable=False,
        )
        self._line.setZValue(20)
        self._plot.addItem(self._line)
        self._line.sigRegionChanged.connect(self._on_line_dragged)

        # ROI shape.  Only one exists at a time; changing kind rebuilds it, because
        # pyqtgraph's rect/ellipse/polygon ROIs are different classes.
        self._roi_item: pg.ROI | None = None
        self._roi_kind: str | None = None
        self._syncing_roi = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._glw)

    # -- data -------------------------------------------------------------

    def set_image(self, arr: np.ndarray | None, cmap_name: str,
                  levels: tuple[float, float] | None) -> None:
        """Show `arr`. A None array clears the panel (single-image mode)."""
        had_shape = self.shape is not None
        self._arr = arr
        if arr is None:
            self._image.clear()
            self._line.setVisible(False)
            self.set_swipe_image(None)
            return
        self._line.setVisible(True)
        if not had_shape:
            # First panel in this view: place the line on a default diagonal rather
            # than leaving it at its degenerate 1-px construction position.
            self.set_line_normalised(*DEFAULT_LINE)
        disp = to_2d(arr) if arr.ndim == 3 else arr
        self._image.setImage(np.asarray(disp), autoLevels=False)
        cm = _get_colormap(cmap_name)
        if levels is None:
            lo = float(np.nanmin(disp))
            hi = float(np.nanmax(disp))
            if hi <= lo:
                hi = lo + 1e-9
            levels = (lo, hi)
        self._colorbar.setColorMap(cm)
        self._colorbar.setLevels(levels)
        if self._arr_swipe is not None:
            # New panel, possibly a new shape: re-place the overlay so its rect and
            # slice match the image now underneath it.  Levels/colour map come from
            # the shared colour bar, which setLevels above has already updated.
            self.set_swipe_normalised(self._swipe_pos)

    def set_title(self, title: str) -> None:
        self._plot.setTitle(title, color=self._fg, size="9pt")

    @property
    def view_box(self) -> pg.ViewBox:
        return self._plot.getViewBox()

    @property
    def shape(self) -> tuple[int, int] | None:
        return None if self._arr is None else self._arr.shape[:2]

    # -- cross-section line -----------------------------------------------

    def set_line_normalised(self, x0n: float, y0n: float,
                            x1n: float, y1n: float) -> None:
        """Place the line from normalised coords, without re-emitting."""
        shape = self.shape
        if shape is None:
            return
        h, w = shape
        self._line.blockSignals(True)
        try:
            handles = self._line.getHandles()
            self._line.setPos((0, 0))
            handles[0].setPos(x0n * w, y0n * h)
            handles[1].setPos(x1n * w, y1n * h)
        finally:
            self._line.blockSignals(False)

    def line_normalised(self) -> tuple[float, float, float, float] | None:
        shape = self.shape
        if shape is None:
            return None
        h, w = shape
        pts = [self._line.mapToParent(hd.pos()) for hd in self._line.getHandles()]
        return (pts[0].x() / w, pts[0].y() / h, pts[1].x() / w, pts[1].y() / h)

    def set_line_visible(self, visible: bool) -> None:
        self._line.setVisible(visible)

    def set_line_locked(self, locked: bool) -> None:
        """Show the line but make it un-grabbable.

        With the cross-section tool inactive, the line stays on screen so you can see
        where the profile is taken, but its handles neither render nor accept mouse
        events — so a click in the panel unambiguously means pan (or, in swipe mode,
        the divider) rather than "did I just nudge the profile?".
        """
        self._line.translatable = not locked
        self._line.resizable = not locked
        self._line.rotatable = not locked
        for handle in self._line.getHandles():
            handle.setVisible(not locked)
            handle.setEnabled(not locked)
        self._line.setAcceptedMouseButtons(
            Qt.MouseButton.NoButton if locked else Qt.MouseButton.LeftButton)

    def _on_line_dragged(self) -> None:
        coords = self.line_normalised()
        if coords is not None:
            self.line_changed.emit(*coords)

    # -- A/B swipe --------------------------------------------------------

    def set_swipe_image(self, arr_b: np.ndarray | None) -> None:
        """Enable (arr_b given) or disable (None) the before/after overlay.

        No levels or colour map are passed: the shared ColorBarItem already owns both
        for this item (see __init__), which is what guarantees the two halves of the
        wipe stay on one scale.
        """
        self._arr_swipe = None if arr_b is None else to_2d(arr_b)
        active = self._arr_swipe is not None
        self._image_swipe.setVisible(active)
        self._divider.setVisible(active)
        if not active:
            self._image_swipe.clear()
            return
        self.set_swipe_normalised(self._swipe_pos)

    def set_swipe_normalised(self, pos_n: float) -> None:
        """Place the divider from a normalised [0,1] x position, without emitting."""
        self._swipe_pos = float(np.clip(pos_n, 0.0, 1.0))
        shape = self.shape
        if shape is None or self._arr_swipe is None:
            return
        h, w = shape
        col = int(round(self._swipe_pos * w))
        col = max(0, min(w, col))

        self._syncing_swipe = True
        try:
            self._divider.setPos(col)
        finally:
            self._syncing_swipe = False

        hb, wb = self._arr_swipe.shape[:2]
        colb = max(0, min(wb, int(round(self._swipe_pos * wb))))
        if colb >= wb:
            self._image_swipe.clear()
            return
        # Slice is a view, not a copy; the rect places it back at the right x offset.
        self._image_swipe.setImage(self._arr_swipe[:, colb:], autoLevels=False)
        self._image_swipe.setRect(QRectF(float(col), 0.0,
                                          float(w - col), float(h)))

    def swipe_normalised(self) -> float:
        return self._swipe_pos

    # -- overlays ---------------------------------------------------------

    def _overlay_item(self, which: str) -> pg.ImageItem:
        return self._mask_overlay if which == "mask" else self._sel_overlay

    def set_overlay(self, which: str, mask: np.ndarray | None) -> None:
        """Show `mask` (a boolean array) as a translucent layer, or clear it."""
        item = self._overlay_item(which)
        if mask is None or not np.any(mask):
            item.clear()
            item.setVisible(False)
            return
        item.setImage(np.ascontiguousarray(mask, dtype=np.uint8),
                      autoLevels=False, levels=(0, 1))
        item.setLookupTable(self._overlay_lut(which))
        item.setVisible(True)

    def set_overlay_style(self, which: str, color: str | None = None,
                          alpha: int | None = None) -> None:
        """Recolour an overlay without touching its mask — a 2-row LUT rewrite."""
        cur_color, cur_alpha = self._overlay_style[which]
        self._overlay_style[which] = (color or cur_color,
                                      cur_alpha if alpha is None else int(alpha))
        item = self._overlay_item(which)
        if item.image is not None:
            item.setLookupTable(self._overlay_lut(which))

    def set_overlay_visible(self, which: str, visible: bool) -> None:
        item = self._overlay_item(which)
        item.setVisible(bool(visible) and item.image is not None)

    def _overlay_lut(self, which: str) -> np.ndarray:
        color, alpha = self._overlay_style[which]
        c = pg.mkColor(color)
        return np.array([[0, 0, 0, 0],
                         [c.red(), c.green(), c.blue(), int(alpha)]], dtype=np.ubyte)

    # -- ROI --------------------------------------------------------------

    def set_roi(self, roi: dict | None) -> None:
        """Draw the shared ROI, in normalised coords, without re-emitting.

        A rect and an ellipse are different pyqtgraph classes, so switching kind
        replaces the item; moving the same kind only repositions it, which keeps
        drag state intact while the user is working.
        """
        shape = self.shape
        if shape is None:
            return
        h, w = shape
        kind = None if roi is None else roi.get("kind")

        if kind != self._roi_kind:
            if self._roi_item is not None:
                self._plot.removeItem(self._roi_item)
                self._roi_item = None
            self._roi_kind = kind
            if kind is not None:
                self._roi_item = self._make_roi_item(kind)
                self._roi_item.setZValue(15)
                self._plot.addItem(self._roi_item)
                self._roi_item.sigRegionChanged.connect(self._on_roi_dragged)
        if self._roi_item is None:
            return

        self._syncing_roi = True
        try:
            if kind == ROI_POLYGON:
                pts = [(x * w, y * h) for x, y in (roi.get("points") or [])]
                if pts:
                    self._roi_item.setPoints(pts, closed=True)
            else:
                x0, x1 = sorted((roi["x0"] * w, roi["x1"] * w))
                y0, y1 = sorted((roi["y0"] * h, roi["y1"] * h))
                self._roi_item.setPos((x0, y0), finish=False)
                self._roi_item.setSize((max(x1 - x0, 1.0), max(y1 - y0, 1.0)),
                                       finish=False)
        finally:
            self._syncing_roi = False

    def _make_roi_item(self, kind: str) -> pg.ROI:
        pen = pg.mkPen(OVERLAY_MASK_COLOR, width=2)
        hover = pg.mkPen("#ffffff", width=2)
        common = dict(pen=pen, hoverPen=hover, removable=False)
        if kind == ROI_ELLIPSE:
            return pg.EllipseROI([0, 0], [10, 10], **common)
        if kind == ROI_POLYGON:
            return pg.PolyLineROI([[0, 0], [10, 0], [10, 10]], closed=True, **common)
        return pg.RectROI([0, 0], [10, 10], **common)

    def roi_normalised(self) -> dict | None:
        """Current ROI geometry in normalised [0,1] coords, or None."""
        shape = self.shape
        if shape is None or self._roi_item is None or self._roi_kind is None:
            return None
        h, w = shape
        if self._roi_kind == ROI_POLYGON:
            pts = [self._roi_item.mapToParent(hd.pos())
                   for hd in self._roi_item.getHandles()]
            return {"kind": ROI_POLYGON,
                    "points": [(p.x() / w, p.y() / h) for p in pts]}
        pos, size = self._roi_item.pos(), self._roi_item.size()
        return {"kind": self._roi_kind,
                "x0": pos.x() / w, "y0": pos.y() / h,
                "x1": (pos.x() + size.x()) / w, "y1": (pos.y() + size.y()) / h}

    def set_roi_locked(self, locked: bool) -> None:
        """Same treatment as the cross-section line: visible but un-grabbable when
        its tool is not the active one, so a click is never ambiguous."""
        if self._roi_item is None:
            return
        self._roi_item.translatable = not locked
        for handle in self._roi_item.getHandles():
            handle.setVisible(not locked)
            handle.setEnabled(not locked)
        self._roi_item.setAcceptedMouseButtons(
            Qt.MouseButton.NoButton if locked else Qt.MouseButton.LeftButton)

    def _on_roi_dragged(self) -> None:
        if self._syncing_roi:
            return
        roi = self.roi_normalised()
        if roi is not None:
            self.roi_changed.emit(roi)

    def _on_divider_dragged(self) -> None:
        if self._syncing_swipe or self._arr_swipe is None:
            return
        shape = self.shape
        if shape is None:
            return
        h, w = shape
        pos_n = float(np.clip(self._divider.value() / max(w, 1), 0.0, 1.0))
        self.set_swipe_normalised(pos_n)
        self.swipe_changed.emit(pos_n)


def _get_colormap(name: str) -> pg.ColorMap:
    """Colormap by name, preferring pyqtgraph's own then falling back to matplotlib.

    "bwr" (the log-ratio / difference diverging map) only exists on the matplotlib
    side; "viridis" and "grey" ship with pyqtgraph.
    """
    try:
        cm = pg.colormap.get(name)
        if cm is not None:
            return cm
    except Exception:
        pass
    return pg.colormap.get(name, source="matplotlib")


def cmap_for_panel(npz_key: str) -> str:
    """Greyscale for source imagery, viridis for derived metric maps.

    Mirrors _section_spatial's own choice: the Original panel is drawn with
    cmap="gray" because it is the input, not a metric.
    """
    if npz_key.startswith(("display_", "linear_", "sim_")):
        return "grey"
    if npz_key.startswith("sp_original") or npz_key.startswith("sp_nrm_original"):
        return "grey"
    if npz_key.startswith("epsf_"):
        return SECTION8_ANALYSIS_CMAP
    return SECTION8_ANALYSIS_CMAP


# ---------------------------------------------------------------------------
# Cross-section plot
# ---------------------------------------------------------------------------

class CrossSectionPlot(QWidget):
    """A and B profiles along the drawn line, plus an opt-in comparison curve.

    The comparison sits on its own right-hand axis because a log ratio and a raw map
    value share no principled vertical scale.  It is off by default, its axis
    auto-ranges symmetric about zero, and a dashed zero line is drawn on it — so
    "above/below the line" reads correctly no matter what the left axis is doing.
    """

    def __init__(self, dark: bool = False, parent=None):
        super().__init__(parent)
        bg, fg = theme_colors(dark)
        self._fg = fg
        self._show_compare = False
        self._compare_label = COMPARE_LABELS["logratio"]

        self._pw = pg.PlotWidget(background=bg)
        self._plot: pg.PlotItem = self._pw.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "Position along line (px)")
        self._plot.setLabel("left", "Map value")
        self._plot.addLegend(offset=(10, 10), labelTextColor=fg)
        style_plot_item(self._plot, fg)

        self._curve_a = self._plot.plot([], [], pen=pg.mkPen(COLOR_A, width=1.5), name="A")
        self._curve_b = self._plot.plot([], [], pen=pg.mkPen(COLOR_B, width=1.5), name="B")

        # Second ViewBox for the comparison curve, sharing the x axis.
        self._vb2 = pg.ViewBox()
        self._plot.showAxis("right")
        self._plot.scene().addItem(self._vb2)
        self._plot.getAxis("right").linkToView(self._vb2)
        self._vb2.setXLink(self._plot.getViewBox())
        right_axis = self._plot.getAxis("right")
        right_axis.setLabel(self._compare_label, color=COLOR_COMPARE)
        right_axis.setPen(pg.mkPen(COLOR_COMPARE))
        right_axis.setTextPen(pg.mkPen(COLOR_COMPARE))
        # A log ratio is already dimensionless and small; an SI prefix turns "0.5"
        # into "500" plus a "(x0.001)" suffix on the label, which reads as a
        # different quantity entirely.
        right_axis.enableAutoSIPrefix(False)
        self._plot.getAxis("left").enableAutoSIPrefix(False)

        self._curve_c = pg.PlotDataItem([], [], pen=pg.mkPen(COLOR_COMPARE, width=1.2))
        self._zero_line = pg.InfiniteLine(pos=0.0, angle=0,
                                          pen=pg.mkPen(COLOR_COMPARE, width=1,
                                                       style=Qt.PenStyle.DashLine))
        self._vb2.addItem(self._curve_c)
        self._vb2.addItem(self._zero_line)

        self._plot.getViewBox().sigResized.connect(self._sync_vb2_geometry)
        self.set_comparison_visible(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._pw)

    def _sync_vb2_geometry(self) -> None:
        self._vb2.setGeometry(self._plot.getViewBox().sceneBoundingRect())
        self._vb2.linkedViewChanged(self._plot.getViewBox(), self._vb2.XAxis)

    def set_comparison_visible(self, visible: bool) -> None:
        self._show_compare = visible
        self._curve_c.setVisible(visible)
        self._zero_line.setVisible(visible)
        self._plot.showAxis("right", visible)
        self._sync_vb2_geometry()

    def set_comparison_label(self, label: str) -> None:
        self._compare_label = label
        self._plot.getAxis("right").setLabel(label, color=COLOR_COMPARE)

    def set_labels(self, label_a: str, label_b: str) -> None:
        legend = self._plot.legend
        if legend is not None:
            legend.clear()
            legend.addItem(self._curve_a, label_a)
            legend.addItem(self._curve_b, label_b)

    def update_profiles(self, pos: np.ndarray | None,
                        prof_a: np.ndarray | None,
                        prof_b: np.ndarray | None,
                        prof_c: np.ndarray | None) -> None:
        if pos is None or prof_a is None:
            self._curve_a.setData([], [])
            self._curve_b.setData([], [])
            self._curve_c.setData([], [])
            return
        self._curve_a.setData(pos, prof_a)
        if prof_b is None:
            self._curve_b.setData([], [])
        else:
            n = min(len(pos), len(prof_b))
            self._curve_b.setData(pos[:n], prof_b[:n])

        if prof_c is None or not self._show_compare:
            self._curve_c.setData([], [])
            return
        n = min(len(pos), len(prof_c))
        self._curve_c.setData(pos[:n], prof_c[:n])
        # Symmetric about zero so the dashed zero line always sits mid-axis and
        # "A above B" reads the same way regardless of the left axis' range.
        m = float(np.nanmax(np.abs(prof_c[:n]))) if n else 0.0
        m = m if m > 0 else 1.0
        self._vb2.setYRange(-m * 1.1, m * 1.1, padding=0)


# ---------------------------------------------------------------------------
# Comparison histogram + box-whisker
# ---------------------------------------------------------------------------

class RatioHistogramPlot(QWidget):
    """Log-Y histogram of the comparison map, with an IQR box-whisker strip above.

    Bars are coloured through the same diverging colormap and the same symmetric
    range as the comparison image, so a bar's colour and a pixel's colour mean the
    same value.  Box/whisker styling follows _draw_boxwhisker (median magenta,
    IQR cyan).
    """

    _MEDIAN_COLOR = "magenta"
    _BOX_COLOR = "#00e5ff"
    # Bar baseline in log10(count) space: slightly below zero so single-count bins
    # still render as a visible stub instead of a zero-height bar.
    _Y_FLOOR = -0.2

    def __init__(self, dark: bool = False, parent=None):
        super().__init__(parent)
        bg, fg = theme_colors(dark)
        self._fg = fg

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(bg)

        self._box_plot = self._glw.addPlot(row=0, col=0)
        self._box_plot.setMaximumHeight(46)
        self._box_plot.hideAxis("left")
        self._box_plot.hideAxis("bottom")
        self._box_plot.setMouseEnabled(x=True, y=False)
        self._box_plot.setYRange(-1, 1, padding=0)
        self._box_plot.setMenuEnabled(False)
        style_plot_item(self._box_plot, fg)

        self._hist_plot = self._glw.addPlot(row=1, col=0)
        self._hist_plot.setLogMode(x=False, y=True)
        self._hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self._hist_plot.setLabel("left", "Pixel count")
        self._hist_plot.setLabel("bottom", COMPARE_LABELS["logratio"])
        self._hist_plot.setMenuEnabled(False)
        # A log ratio is dimensionless and often small (±0.15 on the raw input
        # images); without this pyqtgraph relabels the axis "±800 (x0.001)", which
        # reads as a completely different quantity.
        self._hist_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self._hist_plot.getAxis("left").enableAutoSIPrefix(False)
        style_plot_item(self._hist_plot, fg)
        self._box_plot.setXLink(self._hist_plot)

        self._bars: pg.BarGraphItem | None = None
        self._box_items: list = []
        for plot in (self._hist_plot, self._box_plot):
            zero = pg.InfiniteLine(pos=0.0, angle=90,
                                   pen=pg.mkPen(fg, width=1, style=Qt.PenStyle.DashLine))
            plot.addItem(zero)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._glw)

    def set_x_label(self, label: str) -> None:
        self._hist_plot.setLabel("bottom", label)

    def clear(self) -> None:
        if self._bars is not None:
            self._hist_plot.removeItem(self._bars)
            self._bars = None
        for item in self._box_items:
            self._box_plot.removeItem(item)
        self._box_items = []

    def update_histogram(self, values: np.ndarray | None,
                         color_range: tuple[float, float],
                         cmap_name: str = "bwr", n_bins: int = 60) -> None:
        self.clear()
        if values is None or values.size == 0:
            return
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return

        counts, edges = np.histogram(vals, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        # BarGraphItem does NOT honour PlotItem.setLogMode — that only transforms
        # PlotDataItems — so with a log y-axis the bars must carry log10 heights
        # themselves, in the ViewBox's own (already-log) coordinates.  Empty bins are
        # dropped rather than plotted at log10(0) = -inf.
        keep = counts > 0
        if not keep.any():
            return
        centers, widths = centers[keep], widths[keep]
        heights = np.log10(counts[keep].astype(float))

        cm = _get_colormap(cmap_name)
        vmin, vmax = color_range
        span = (vmax - vmin) or 1.0
        # Clip to the colour range, so extreme-tail bins saturate to the same end
        # colours the image itself uses for its own out-of-range pixels.
        frac = np.clip((centers - vmin) / span, 0.0, 1.0)
        brushes = [pg.mkBrush(cm.map(float(f), mode="qcolor")) for f in frac]

        self._bars = pg.BarGraphItem(x=centers, y0=self._Y_FLOOR,
                                     height=heights - self._Y_FLOOR,
                                     width=widths * 0.98, brushes=brushes, pen=None)
        self._hist_plot.addItem(self._bars)
        self._hist_plot.setYRange(self._Y_FLOOR, float(heights.max()) * 1.05 + 0.1,
                                  padding=0)
        # Symmetric x-range so the zero line stays centred, matching the image's own
        # symmetric colour scale.
        xmax = float(np.abs(edges).max()) or 1.0
        self._hist_plot.setXRange(-xmax, xmax, padding=0.02)

        self._draw_boxwhisker(vals)

    def _draw_boxwhisker(self, vals: np.ndarray) -> None:
        """Horizontal IQR box with 1.5-IQR whiskers and a median tick."""
        q1, med, q3 = (float(v) for v in np.percentile(vals, [25, 50, 75]))
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inside = vals[(vals >= lo_fence) & (vals <= hi_fence)]
        w_lo = float(inside.min()) if inside.size else q1
        w_hi = float(inside.max()) if inside.size else q3

        box_pen = pg.mkPen(self._BOX_COLOR, width=1.5)
        med_pen = pg.mkPen(self._MEDIAN_COLOR, width=2.0)
        half = 0.35

        segments = [
            ([q1, q3, q3, q1, q1], [-half, -half, half, half, -half], box_pen),   # box
            ([w_lo, q1], [0, 0], box_pen),                                        # left whisker
            ([q3, w_hi], [0, 0], box_pen),                                        # right whisker
            ([w_lo, w_lo], [-half, half], box_pen),                               # left cap
            ([w_hi, w_hi], [-half, half], box_pen),                               # right cap
            ([med, med], [-half, half], med_pen),                                 # median
        ]
        for xs, ys, pen in segments:
            item = pg.PlotDataItem(xs, ys, pen=pen)
            self._box_plot.addItem(item)
            self._box_items.append(item)


# ---------------------------------------------------------------------------
# Correlation scatter with lasso / rectangle brushing
# ---------------------------------------------------------------------------

BRUSH_OFF = "off"
BRUSH_RECT = "rect"
BRUSH_LASSO = "lasso"


class _BrushViewBox(pg.ViewBox):
    """ViewBox that can capture a rectangle or freehand polygon instead of panning.

    While dragging, only the outline is drawn — the point-in-polygon test runs once,
    on release.  Testing every point on every mouse-move is what makes a lasso feel
    heavy on a 50 000-point scatter.
    """

    polygon_drawn = pyqtSignal(object)     # (N, 2) vertices in data coords

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._brush_mode = BRUSH_OFF
        self._points: list[tuple[float, float]] = []
        self._outline = pg.PlotDataItem(
            [], [], pen=pg.mkPen(OVERLAY_SELECTION_COLOR, width=1.5,
                                 style=Qt.PenStyle.DashLine))
        self._outline.setZValue(50)
        self.addItem(self._outline, ignoreBounds=True)

    def set_brush_mode(self, mode: str) -> None:
        self._brush_mode = mode
        if mode == BRUSH_OFF:
            self._clear_outline()

    def _clear_outline(self) -> None:
        self._points = []
        self._outline.setData([], [])

    def mouseDragEvent(self, ev, axis=None):
        if self._brush_mode == BRUSH_OFF or ev.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis=axis)
            return
        ev.accept()
        pos = self.mapToView(ev.pos())
        if ev.isStart():
            self._points = [(pos.x(), pos.y())]
            return
        if self._brush_mode == BRUSH_RECT:
            x0, y0 = self._points[0]
            self._points = [(x0, y0), (pos.x(), y0), (pos.x(), pos.y()), (x0, pos.y())]
        else:
            self._points.append((pos.x(), pos.y()))
        pts = self._points + self._points[:1]        # close the outline visually
        self._outline.setData([p[0] for p in pts], [p[1] for p in pts])
        if ev.isFinish():
            poly = np.asarray(self._points, dtype=float)
            self._clear_outline()
            if poly.shape[0] >= 3:
                self.polygon_drawn.emit(poly)


class CorrelationPlot(QWidget):
    """A-vs-B scatter for one pixel population, with a 1:1 line and brushing.

    Three layers, per the linked-brushing model: an all-points base layer coloured by
    each pixel's comparison value, a high-contrast selected layer, and the reference
    line.  Selection never restyles the base layer — it only refills the second one.
    """

    # np.int64 array of flat pixel ids
    selection_changed = pyqtSignal(object)

    _N_BRUSH_BINS = 64

    def __init__(self, title: str = "", dark: bool = False, parent=None):
        super().__init__(parent)
        bg, fg = theme_colors(dark)
        self._fg = fg
        self._sample = None
        self._brush_cache: list | None = None

        self._vb = _BrushViewBox()
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(bg)
        self._plot = self._glw.addPlot(viewBox=self._vb, title=title)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setMenuEnabled(False)
        self._plot.setAspectLocked(True)
        style_plot_item(self._plot, fg)
        for name in ("bottom", "left"):
            self._plot.getAxis(name).enableAutoSIPrefix(False)

        self._base = pg.ScatterPlotItem(pen=None, size=3, useCache=True)
        self._selected = pg.ScatterPlotItem(
            pen=None, size=4, brush=pg.mkBrush(OVERLAY_SELECTION_COLOR))
        self._selected.setZValue(10)
        # Slope-1 reference: where A == B. Divergence from it is the whole point.
        self._unity = pg.InfiniteLine(pos=(0, 0), angle=45,
                                      pen=pg.mkPen(fg, width=1.2,
                                                   style=Qt.PenStyle.DashLine))
        for item in (self._base, self._selected, self._unity):
            self._plot.addItem(item)

        self._vb.polygon_drawn.connect(self._on_polygon)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._glw)

    # -- data -------------------------------------------------------------

    def set_sample(self, sample, color_range: tuple[float, float],
                   label_a: str, label_b: str, region: str) -> None:
        """Show one CorrelationSample (from analysis.inspector_regions)."""
        self._sample = sample
        self._selected.setData([], [])
        if sample is None or sample.n_total == 0:
            self._base.setData([], [])
            self._plot.setTitle(f"{region} (no pixels)", color=self._fg, size="9pt")
            return

        self._base.setData(sample.b_vals, sample.a_vals,
                           brush=self._brushes(sample.c_vals, color_range))
        self._plot.setLabel("bottom", f"{label_b} (x)")
        self._plot.setLabel("left", f"{label_a} (y)")
        shown = sample.a_vals.size
        suffix = f", showing {shown:,}" if shown < sample.n_total else ""
        self._plot.setTitle(f"{region} (n={sample.n_total:,}{suffix})",
                            color=self._fg, size="9pt")
        self._plot.setXRange(sample.axis_lo, sample.axis_hi, padding=0)
        self._plot.setYRange(sample.axis_lo, sample.axis_hi, padding=0)

    def _brushes(self, c_vals: np.ndarray, color_range: tuple[float, float]) -> list:
        """One shared QBrush per colour bin, indexed per point.

        Building 50 000 individual QBrush objects dominates the update; quantising the
        comparison value into a fixed number of bins makes it a lookup instead.  The
        colormap and range are the map's own, so a dot's colour means what the same
        colour means in the comparison image.
        """
        if self._brush_cache is None:
            cm = _get_colormap("bwr")
            self._brush_cache = [
                pg.mkBrush(cm.map(i / (self._N_BRUSH_BINS - 1), mode="qcolor"))
                for i in range(self._N_BRUSH_BINS)
            ]
        lo, hi = color_range
        span = (hi - lo) or 1.0
        idx = np.clip(((c_vals - lo) / span * (self._N_BRUSH_BINS - 1)).astype(int),
                      0, self._N_BRUSH_BINS - 1)
        cache = self._brush_cache
        return [cache[i] for i in idx]

    # -- selection --------------------------------------------------------

    def set_brush_mode(self, mode: str) -> None:
        self._vb.set_brush_mode(mode)

    def set_selected_ids(self, ids: np.ndarray | None) -> None:
        """Highlight the points whose flat pixel ids are in `ids`.

        Matching on ids rather than positions is what lets a selection made in one
        plot survive a resample, and lets the sibling plot show the same pixels.
        """
        s = self._sample
        if s is None or ids is None or len(ids) == 0 or s.n_total == 0:
            self._selected.setData([], [])
            return
        hit = np.isin(s.flat_ids, np.asarray(ids, dtype=np.int64))
        self._selected.setData(s.b_vals[hit], s.a_vals[hit])

    def _on_polygon(self, poly: np.ndarray) -> None:
        s = self._sample
        if s is None or s.n_total == 0:
            return
        hit = select_points_in_polygon(s.b_vals, s.a_vals, poly)
        self.selection_changed.emit(s.flat_ids[hit])
