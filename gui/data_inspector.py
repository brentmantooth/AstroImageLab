"""Data Inspector — interactive pyqtgraph explorer for a report's `_inspector.npz`.

A more interactive successor to gui/report_inspector.py, which it will eventually
replace.  Both are kept while this one is developed and tested; the Report Inspector
still opens automatically after a run, this one is opened from the File menu.

Phase 1 (this file): the data-selection model carried over from the Report Inspector
(Section -> Image set -> A / B), three pan/zoom-synchronised image panels
(A | B | comparison), the comparison histogram with its box-whisker strip, and a
cross-section line that can be dragged in any of the three panels.

Phase 2 adds ROI / threshold region selection, mask overlays, the A-vs-B correlation
plots, and lasso brushing from a correlation plot back onto image pixels.
"""
from __future__ import annotations

import os
from pathlib import Path

# Must precede the pyqtgraph import (see gui/inspector_widgets.py).
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QPushButton, QSplitter, QFrame, QSizePolicy, QSpinBox,
    QDoubleSpinBox, QColorDialog,
)

from analysis.image_filters import SpatialDetailAnalyzer as _SDA
from analysis.inspector_regions import (
    DIR_LOWER, DIR_UPPER, ROI_ELLIPSE, ROI_POLYGON, ROI_RECT,
    THRESH_ABSOLUTE, THRESH_PERCENTILE, comparison_map, comparison_range,
    ids_to_mask, to_2d, value_range,
)
from core.inspector_catalog import InspectorData, load_inspector_data
from gui.inspector_widgets import (
    BRUSH_LASSO, BRUSH_OFF, COMPARE_LABELS, DEFAULT_LINE,
    OVERLAY_DEFAULT_ALPHA, OVERLAY_MASK_COLOR, OVERLAY_SELECTION_COLOR,
    CorrelationPlot, CrossSectionPlot, LinkedImageView,
    RatioHistogramPlot, cmap_for_panel, theme_colors,
)

# Normalised [0,1] view rectangle covering the whole image.
_FULL_VIEW = (0.0, 1.0, 0.0, 1.0)

# What a left-click/drag in an image panel does.  Exactly one is active at a time so
# a click is never ambiguous — the inactive tools' handles are locked (see
# LinkedImageView.set_line_locked).  Phase 2 adds ROI and threshold entries here.
TOOL_PAN = "pan"
TOOL_LINE = "line"
TOOL_ROI = "roi"
TOOL_BRUSH = "brush"

# Per (view mode, tool) hint shown under the toolbar, so the current click behaviour
# is always stated rather than something the user has to discover.
_HINTS = {
    ("side", TOOL_PAN): (
        "Left-drag to pan, wheel to zoom — all three panels move together. "
        "The orange cross-section line is locked; choose the Cross-section tool to move it."),
    ("side", TOOL_LINE): (
        "Drag the orange line or its square handles to move the cross-section. "
        "Wheel still zooms; right-drag still scales."),
    ("swipe", TOOL_PAN): (
        "Drag the white divider to wipe between Image A (left) and Image B (right). "
        "Left-drag elsewhere pans, wheel zooms. The cross-section line is locked."),
    ("swipe", TOOL_LINE): (
        "Drag the orange line to move the cross-section; drag the white divider to "
        "wipe between A and B. Both are live — the divider is the white vertical line."),
    ("side", TOOL_ROI): (
        "Drag the magenta ROI or its handles to move/resize the region. "
        "The cross-section line is locked while this tool is active."),
    ("swipe", TOOL_ROI): (
        "Drag the magenta ROI to move/resize the region; the white divider still "
        "wipes between A and B. The cross-section line is locked."),
    ("side", TOOL_BRUSH): (
        "Drag inside a correlation plot to lasso points — the matching pixels light "
        "up in cyan on all three images. Image panels are locked to pan/zoom."),
    ("swipe", TOOL_BRUSH): (
        "Drag inside a correlation plot to lasso points; the matching pixels light up "
        "in cyan. The white divider still wipes between A and B."),
}


def _get_array(npz, key: str) -> np.ndarray | None:
    try:
        return npz[key]
    except KeyError:
        return None


class _RegionThread(QThread):
    """Builds the masks, correlation samples and histogram off the GUI thread.

    Everything here is pure array work from analysis.inspector_regions; the request
    carries its own copies of the parameters so a control changing mid-run cannot
    alter what this run is computing.  `token` is echoed back so the window can drop
    a result that a newer request has already superseded.
    """

    done = pyqtSignal(object)

    def __init__(self, req: dict, parent=None):
        super().__init__(parent)
        self._req = req

    def run(self) -> None:
        from analysis.inspector_regions import (
            correlation_sample, refine_mask, roi_mask, threshold_mask,
        )
        r = self._req
        try:
            a, b, comp = r["a"], r["b"], r["comparison"]
            shape = comp.shape[:2]
            domain = roi_mask(shape, r["roi"])

            warning = None
            if r["threshold_on"]:
                res = threshold_mask(a, b, r["thresh_mode"], r["thresh_value"],
                                     r["direction"])
                split, warning = res.mask, res.warning
                split = refine_mask(split, open_px=r["open_px"], close_px=r["close_px"],
                                    min_size_px=r["min_size_px"],
                                    fill_hole_px=r["fill_hole_px"])
                hs = min(domain.shape[0], split.shape[0])
                ws = min(domain.shape[1], split.shape[1])
                domain, split = domain[:hs, :ws], split[:hs, :ws]
                # A threshold partitions the DOMAIN: selected vs the rest of it.
                in_mask, out_mask = domain & split, domain & ~split
            else:
                # No threshold, so the ROI itself is the split: inside vs outside.
                # With no ROI either, domain is everything and plot 2 is empty —
                # the "whole image" case.
                in_mask, out_mask = domain, ~domain

            hs, ws = in_mask.shape
            rng = np.random.default_rng(42)   # reproducible subsampling
            self.done.emit({
                "token": r["token"],
                "in_mask": in_mask,
                "out_mask": out_mask,
                "domain": domain,
                "warning": warning,
                "sample_in": correlation_sample(a, b, comp, in_mask, rng=rng),
                "sample_out": correlation_sample(a, b, comp, out_mask, rng=rng),
                # The histogram follows the domain, so drawing an ROI narrows the
                # distribution to that region rather than always showing the frame.
                "hist_values": comp[:hs, :ws][domain[:hs, :ws]],
            })
        except Exception as exc:                     # never take the window down
            self.done.emit({"token": r["token"], "error": str(exc)})


class DataInspector(QMainWindow):
    """Interactive explorer for one `<stem>_inspector.npz`."""

    def __init__(self, npz_path: str | Path, dark_mode: bool = False, parent=None):
        super().__init__(parent)
        self._npz = np.load(str(npz_path), allow_pickle=False)
        self._data: InspectorData = load_inspector_data(self._npz)
        self._dark = dark_mode
        self._single_image = self._data.single_image

        # Currently displayed arrays, cached so dragging the cross-section line does
        # not re-decompress them out of the .npz on every mouse move.
        self._arr_a: np.ndarray | None = None
        self._arr_b: np.ndarray | None = None
        self._compare: np.ndarray | None = None
        self._cmap_a: str = "grey"

        # Persistent geometry, all in normalised [0,1] image coordinates so it
        # survives a switch to a panel of a different pixel size without ever
        # falling out of bounds (same convention as gui/image_panel.py).
        self._line: tuple[float, float, float, float] = DEFAULT_LINE
        self._view: tuple[float, float, float, float] = _FULL_VIEW
        self._swipe: float = 0.5
        self._roi: dict | None = None
        self._syncing = False          # re-entrancy guard for the 3-way line/ROI sync
        self._did_first_fit = False    # see showEvent
        self._view_mode = "side"       # "side" | "swipe"
        self._tool = TOOL_PAN

        # Region-selection state.  _selected_ids are flat pixel indices into
        # _region_shape — the stable identity linking scatter points to pixels.
        self._selected_ids: np.ndarray | None = None
        self._region_shape: tuple[int, int] | None = None
        self._mask_for_overlay: np.ndarray | None = None
        self._region_token = 0
        self._region_thread: _RegionThread | None = None

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._refit)

        self._region_timer = QTimer(self)
        self._region_timer.setSingleShot(True)
        self._region_timer.timeout.connect(self._start_region_job)

        self.setWindowTitle("Data Inspector")
        self.resize(1500, 980)
        self._build_ui()
        self._connect_signals()
        self._populate_sections()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        bg, fg = theme_colors(self._dark)
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # --- Row 1: data selection ------------------------------------
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self._section_combo = QComboBox()
        self._image_set_combo = QComboBox()
        self._left_combo = QComboBox()
        self._right_combo = QComboBox()
        for combo, width in ((self._section_combo, 150), (self._image_set_combo, 260),
                             (self._left_combo, 150), (self._right_combo, 150)):
            combo.setMinimumWidth(width)

        self._compare_combo = QComboBox()
        # "Ratio" means the log10 ratio throughout, following the report's Section 8
        # convention — there is deliberately no linear A/B mode.
        self._compare_combo.addItem("Log ratio  log₁₀(|A|/|B|)", "logratio")
        self._compare_combo.addItem("Difference  A − B", "difference")
        self._compare_combo.setMinimumWidth(190)

        bar.addWidget(QLabel("Section:"));    bar.addWidget(self._section_combo)
        bar.addWidget(QLabel("Image set:"));  bar.addWidget(self._image_set_combo)
        bar.addWidget(self._vline())
        bar.addWidget(QLabel("A:"));          bar.addWidget(self._left_combo)
        bar.addWidget(QLabel("B:"));          bar.addWidget(self._right_combo)
        bar.addWidget(self._vline())
        bar.addWidget(QLabel("Compare:"));    bar.addWidget(self._compare_combo)
        bar.addStretch()
        root.addLayout(bar)

        # --- Row 1b: how the panels are shown, and what a click does ---
        bar2 = QHBoxLayout()
        bar2.setSpacing(6)

        self._view_combo = QComboBox()
        self._view_combo.addItem("Side by side", "side")
        self._view_combo.addItem("A / B swipe", "swipe")
        self._view_combo.setMinimumWidth(140)

        self._tool_combo = QComboBox()
        self._tool_combo.addItem("Pan / Zoom", TOOL_PAN)
        self._tool_combo.addItem("Cross-section line", TOOL_LINE)
        self._tool_combo.addItem("Move / resize region", TOOL_ROI)
        self._tool_combo.addItem("Lasso correlation points", TOOL_BRUSH)
        self._tool_combo.setMinimumWidth(190)

        bar2.addWidget(QLabel("View:"));  bar2.addWidget(self._view_combo)
        bar2.addWidget(QLabel("Tool:"));  bar2.addWidget(self._tool_combo)
        bar2.addWidget(self._vline())

        self._reset_line_btn = QPushButton("Reset Line")
        self._reset_zoom_btn = QPushButton("Reset Zoom")
        for btn in (self._reset_line_btn, self._reset_zoom_btn):
            btn.setFixedWidth(96)
            bar2.addWidget(btn)
        bar2.addSpacing(10)

        self._hint_lbl = QLabel()
        self._hint_lbl.setStyleSheet("color: #b0b0b0; font-size: 8pt;")
        bar2.addWidget(self._hint_lbl, stretch=1)
        root.addLayout(bar2)

        root.addLayout(self._build_region_bar())

        # --- Row 2: filenames -----------------------------------------
        if self._single_image:
            fn_text = f"Single Image Analysis — A: {self._data.filename_a}"
        else:
            fn_text = (f"A: {self._data.filename_a}"
                       f"    │    B: {self._data.filename_b}")
        self._filename_lbl = QLabel(fn_text)
        self._filename_lbl.setStyleSheet("color: #999999; font-size: 8pt;")
        root.addWidget(self._filename_lbl)

        # --- Row 3: metric description --------------------------------
        self._concept_lbl = QLabel()
        self._concept_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._concept_lbl.setWordWrap(True)
        self._concept_lbl.setStyleSheet("color: #6cabdb; font-size: 9pt;")
        self._concept_lbl.setVisible(False)
        root.addWidget(self._concept_lbl)

        # --- Body: three stacked, independently resizable rows ---------
        self._view_a = LinkedImageView("Image A", dark=self._dark)
        self._view_b = LinkedImageView("Image B", dark=self._dark)
        self._view_c = LinkedImageView("Comparison", dark=self._dark)
        self._views = (self._view_a, self._view_b, self._view_c)

        images_row = QSplitter(Qt.Orientation.Horizontal)
        for view in self._views:
            images_row.addWidget(view)
        images_row.setSizes([500, 500, 500])

        # Pan/zoom is shared: acting on any panel moves all three.
        for view in (self._view_b, self._view_c):
            view.view_box.setXLink(self._view_a.view_box)
            view.view_box.setYLink(self._view_a.view_box)

        self._xsec = CrossSectionPlot(dark=self._dark)
        self._hist = RatioHistogramPlot(dark=self._dark)
        self._compare_on_xsec = QCheckBox("Show comparison on right axis")

        xsec_panel = QWidget()
        xsec_layout = QVBoxLayout(xsec_panel)
        xsec_layout.setContentsMargins(0, 0, 0, 0)
        xsec_layout.setSpacing(2)
        xsec_layout.addWidget(self._xsec, stretch=1)
        xsec_layout.addWidget(self._compare_on_xsec, stretch=0)

        plots_row = QSplitter(Qt.Orientation.Horizontal)
        plots_row.addWidget(xsec_panel)
        plots_row.addWidget(self._hist)
        plots_row.setSizes([750, 750])

        self._corr_in = CorrelationPlot("In mask", dark=self._dark)
        self._corr_out = CorrelationPlot("Outside mask", dark=self._dark)
        corr_row = QSplitter(Qt.Orientation.Horizontal)
        corr_row.addWidget(self._corr_in)
        corr_row.addWidget(self._corr_out)
        corr_row.setSizes([750, 750])

        body = QSplitter(Qt.Orientation.Vertical)
        body.addWidget(images_row)
        body.addWidget(plots_row)
        body.addWidget(corr_row)
        # The correlation plots are aspect-locked squares (so the 1:1 line reads at
        # 45 degrees, as in the report), which means they need height to be usable —
        # hence a generous default share.  All three rows stay splitter-adjustable.
        body.setSizes([420, 250, 380])
        root.addWidget(body, stretch=1)

        self.setCentralWidget(central)

    def _build_region_bar(self) -> QHBoxLayout:
        """Domain / split / refine / overlay controls.

        Composable by design: the ROI picks the pixel DOMAIN, the threshold SPLITS
        that domain into the two correlation populations.  With neither set, plot 1
        gets every pixel and plot 2 is empty.
        """
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self._roi_combo = QComboBox()
        self._roi_combo.addItem("Whole image", None)
        self._roi_combo.addItem("Rectangle", ROI_RECT)
        self._roi_combo.addItem("Ellipse", ROI_ELLIPSE)
        self._roi_combo.addItem("Lasso", ROI_POLYGON)
        bar.addWidget(QLabel("Region:"))
        bar.addWidget(self._roi_combo)
        bar.addWidget(self._vline())

        self._thresh_chk = QCheckBox("Threshold")
        self._thresh_mode = QComboBox()
        self._thresh_mode.addItem("Percentile", THRESH_PERCENTILE)
        self._thresh_mode.addItem("Absolute", THRESH_ABSOLUTE)
        self._thresh_value = QDoubleSpinBox()
        self._thresh_value.setDecimals(3)
        self._thresh_value.setRange(-1e9, 1e9)
        self._thresh_value.setValue(10.0)
        self._thresh_value.setSuffix(" %")
        self._thresh_value.setFixedWidth(96)
        self._thresh_dir = QComboBox()
        self._thresh_dir.addItem("Upper", DIR_UPPER)
        self._thresh_dir.addItem("Lower", DIR_LOWER)
        bar.addWidget(QLabel("Split:"))
        for wdg in (self._thresh_chk, self._thresh_mode, self._thresh_value,
                    self._thresh_dir):
            bar.addWidget(wdg)
        bar.addWidget(self._vline())

        bar.addWidget(QLabel("Refine:"))
        self._open_spin, self._close_spin, self._minsize_spin = (
            self._small_spin("open"), self._small_spin("close"), self._small_spin("min"))
        for label, spin in (("open", self._open_spin), ("close", self._close_spin),
                            ("min size", self._minsize_spin)):
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 8pt;")
            bar.addWidget(lbl)
            bar.addWidget(spin)
        bar.addWidget(self._vline())

        bar.addWidget(QLabel("Overlay:"))
        self._mask_chk = QCheckBox("Mask")
        self._mask_chk.setChecked(True)
        self._mask_color_btn = self._color_button(OVERLAY_MASK_COLOR)
        self._sel_chk = QCheckBox("Selection")
        self._sel_chk.setChecked(True)
        self._sel_color_btn = self._color_button(OVERLAY_SELECTION_COLOR)
        self._clear_sel_btn = QPushButton("Clear selection")
        self._clear_sel_btn.setFixedWidth(110)
        for wdg in (self._mask_chk, self._mask_color_btn, self._sel_chk,
                    self._sel_color_btn, self._clear_sel_btn):
            bar.addWidget(wdg)

        self._warn_lbl = QLabel()
        self._warn_lbl.setStyleSheet("color: #e0a030; font-size: 8pt;")
        bar.addWidget(self._warn_lbl, stretch=1)
        return bar

    def _small_spin(self, name: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 50)
        spin.setValue(0)
        spin.setFixedWidth(52)
        spin.setToolTip(f"{name} (px); 0 disables")
        return spin

    def _color_button(self, color: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(22, 22)
        btn.setStyleSheet(f"background-color: {color}; border: 1px solid #888;")
        btn.setProperty("color", color)
        return btn

    def _vline(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _connect_signals(self) -> None:
        self._section_combo.currentIndexChanged.connect(self._on_section_changed)
        self._image_set_combo.currentIndexChanged.connect(self._on_image_set_changed)
        self._left_combo.currentIndexChanged.connect(self._on_panel_changed)
        self._right_combo.currentIndexChanged.connect(self._on_panel_changed)
        self._compare_combo.currentIndexChanged.connect(self._on_compare_mode_changed)
        self._compare_on_xsec.toggled.connect(self._on_compare_curve_toggled)
        self._view_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        self._tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        self._reset_line_btn.clicked.connect(self._on_reset_line)
        self._reset_zoom_btn.clicked.connect(self._on_reset_zoom)
        for view in self._views:
            view.line_changed.connect(self._on_line_changed)
            view.swipe_changed.connect(self._on_swipe_changed)
            view.roi_changed.connect(self._on_roi_changed)
        # Views B and C are x/y-linked to A, so watching A alone covers all three.
        self._view_a.view_box.sigResized.connect(self._on_viewbox_resized)

        self._roi_combo.currentIndexChanged.connect(self._on_roi_kind_changed)
        self._thresh_chk.toggled.connect(self._on_threshold_toggled)
        self._thresh_mode.currentIndexChanged.connect(self._on_threshold_mode_changed)
        for wdg in (self._thresh_value, self._open_spin, self._close_spin,
                    self._minsize_spin):
            wdg.valueChanged.connect(self._request_regions)
        self._thresh_dir.currentIndexChanged.connect(self._request_regions)

        self._mask_chk.toggled.connect(
            lambda on: self._set_overlay_visible("mask", on))
        self._sel_chk.toggled.connect(
            lambda on: self._set_overlay_visible("selection", on))
        self._mask_color_btn.clicked.connect(lambda: self._pick_color("mask"))
        self._sel_color_btn.clicked.connect(lambda: self._pick_color("selection"))
        self._clear_sel_btn.clicked.connect(self._clear_selection)

        for plot in (self._corr_in, self._corr_out):
            plot.selection_changed.connect(self._on_scatter_selection)

        self._apply_tool()

    # ------------------------------------------------------------------
    # Combo population — mirrors the Report Inspector's selection model
    # ------------------------------------------------------------------

    def _populate_sections(self) -> None:
        self._section_combo.blockSignals(True)
        self._section_combo.clear()
        for section in self._data.sections:
            self._section_combo.addItem(section)
        self._section_combo.blockSignals(False)
        if self._data.sections:
            self._section_combo.setCurrentIndex(0)
            self._on_section_changed(0)

    def _current_entries(self) -> list:
        return self._data.sections.get(self._section_combo.currentText(), [])

    def _on_section_changed(self, _idx: int) -> None:
        entries = self._current_entries()
        self._image_set_combo.blockSignals(True)
        self._image_set_combo.clear()
        for entry in entries:
            self._image_set_combo.addItem(entry.name)
        self._image_set_combo.blockSignals(False)
        self._on_image_set_changed(0)

    def _on_image_set_changed(self, idx: int) -> None:
        entries = self._current_entries()
        if idx < 0 or idx >= len(entries):
            return
        entry = entries[idx]

        concept = entry.concept
        self._concept_lbl.setText(concept)
        self._concept_lbl.setVisible(bool(concept))

        labels = list(entry.options.keys())
        self._left_combo.blockSignals(True)
        self._right_combo.blockSignals(True)
        self._left_combo.clear()
        self._right_combo.clear()
        self._left_combo.addItems(labels)
        self._right_combo.addItems(labels)
        if len(labels) >= 2:
            self._right_combo.setCurrentIndex(1)
        # No Image B to compare against in single-image mode.
        self._right_combo.setEnabled(not self._single_image and len(labels) >= 2)
        self._left_combo.blockSignals(False)
        self._right_combo.blockSignals(False)

        self._load_panels()

    def _on_panel_changed(self, _idx: int) -> None:
        self._load_panels()

    # ------------------------------------------------------------------
    # Panel loading
    # ------------------------------------------------------------------

    def _load_panels(self) -> None:
        entries = self._current_entries()
        idx = self._image_set_combo.currentIndex()
        if idx < 0 or idx >= len(entries):
            return
        options = entries[idx].options

        key_a = options.get(self._left_combo.currentText())
        key_b = options.get(self._right_combo.currentText())
        if key_a is None:
            return

        # Preserve the current zoom before the panels change under it.
        self._capture_view()

        self._arr_a = _get_array(self._npz, key_a)
        self._arr_b = None if (self._single_image or key_b is None or not self._right_combo.isEnabled()) \
            else _get_array(self._npz, key_b)
        if self._arr_a is None:
            return

        levels = value_range([self._arr_a, self._arr_b])
        cmap = cmap_for_panel(key_a)
        self._cmap_a = cmap
        self._view_a.set_image(self._arr_a, cmap, levels)
        self._view_a.set_title(f"{self._data.label_a} — {self._left_combo.currentText()}")

        if self._arr_b is None:
            self._view_b.set_image(None, cmap, None)
            self._view_b.set_title("No Image B")
            self._view_c.set_image(None, "bwr", None)
            self._view_c.set_title("No comparison")
            self._compare = None
        else:
            self._view_b.set_image(self._arr_b, cmap_for_panel(key_b), levels)
            self._view_b.set_title(f"{self._data.label_b} — {self._right_combo.currentText()}")
            self._update_comparison()

        # Swipe mode has no Image B to fall back on in single-image mode, so let
        # _apply_view_mode decide what the A panel shows and what it is called.
        self._apply_view_mode()
        self._apply_line_to_views()
        for view in self._views:
            view.set_roi(self._roi)
        self._apply_tool()
        self._apply_view()
        self._update_profiles()
        # A new panel means new pixel values, so masks and both scatters are stale.
        # Drop the selection too: its flat ids index a frame that may have resized.
        self._selected_ids = None
        self._clear_selection()
        self._request_regions()

    def _compare_mode(self) -> str:
        return self._compare_combo.currentData() or "logratio"

    def _update_comparison(self) -> None:
        """Recompute the comparison map, its colour scale, and its histogram."""
        if self._arr_a is None or self._arr_b is None:
            self._compare = None
            self._hist.clear()
            return
        mode = self._compare_mode()
        label = COMPARE_LABELS[mode]
        self._compare = comparison_map(self._arr_a, self._arr_b, mode)
        crange = comparison_range(self._compare)
        self._view_c.set_image(self._compare, "bwr", crange)
        self._view_c.set_title(label)
        self._hist.set_x_label(label)
        self._hist.update_histogram(self._compare.ravel(), crange)
        self._xsec.set_comparison_label(label)

    def _on_compare_mode_changed(self, _idx: int) -> None:
        self._update_comparison()
        self._apply_line_to_views()
        self._update_profiles()
        self._request_regions()   # dot colours and the histogram both follow the mode

    # ------------------------------------------------------------------
    # Cross-section line — one shared normalised geometry, three views
    # ------------------------------------------------------------------

    def _on_line_changed(self, x0n: float, y0n: float, x1n: float, y1n: float) -> None:
        if self._syncing:
            return
        self._line = (x0n, y0n, x1n, y1n)
        self._apply_line_to_views(exclude=self.sender())
        self._update_profiles()

    def _apply_line_to_views(self, exclude=None) -> None:
        """Mirror the shared line into every view except the one being dragged."""
        self._syncing = True
        try:
            for view in self._views:
                if view is not exclude:
                    view.set_line_normalised(*self._line)
        finally:
            self._syncing = False

    def _on_reset_line(self) -> None:
        self._line = DEFAULT_LINE
        self._apply_line_to_views()
        self._update_profiles()

    # ------------------------------------------------------------------
    # View mode (side-by-side vs A/B swipe) and active tool
    # ------------------------------------------------------------------

    def _on_view_mode_changed(self, _idx: int) -> None:
        self._view_mode = self._view_combo.currentData() or "side"
        self._apply_view_mode()
        self._apply_tool()
        # Showing/hiding Image B re-lays out the splitter; refit once Qt has done so.
        QTimer.singleShot(0, self._refit)

    def _apply_view_mode(self) -> None:
        """Swipe mode folds B into the A panel and hides the now-redundant B panel.

        Giving the swipe panel the freed width is the point of the mode — a
        before/after wipe is only readable when the image is large.  The comparison
        panel stays: it answers a different question than the wipe does.
        """
        swipe = self._view_mode == "swipe"
        self._view_b.setVisible(not swipe)

        if swipe and self._arr_a is not None and self._arr_b is not None:
            self._view_a.set_swipe_image(self._arr_b)
            self._view_a.set_swipe_normalised(self._swipe)
            self._view_a.set_title(
                f"{self._data.label_a}  ←|→  {self._data.label_b}")
        else:
            self._view_a.set_swipe_image(None)
            if self._arr_a is not None:
                self._view_a.set_title(
                    f"{self._data.label_a} — {self._left_combo.currentText()}")

    def _on_swipe_changed(self, pos_n: float) -> None:
        self._swipe = pos_n

    def _on_tool_changed(self, _idx: int) -> None:
        self._tool = self._tool_combo.currentData() or TOOL_PAN
        self._apply_tool()

    def _apply_tool(self) -> None:
        """Only the active tool accepts the mouse, and the hint says which one.

        Everything else stays visible but un-grabbable, so a click in a panel always
        means exactly one thing.
        """
        for view in self._views:
            view.set_line_locked(self._tool != TOOL_LINE)
            view.set_roi_locked(self._tool != TOOL_ROI)
        brush = BRUSH_LASSO if self._tool == TOOL_BRUSH else BRUSH_OFF
        for plot in (self._corr_in, self._corr_out):
            plot.set_brush_mode(brush)
        self._hint_lbl.setText(_HINTS.get((self._view_mode, self._tool), ""))

    # ------------------------------------------------------------------
    # Region selection: ROI (domain) + threshold (split)
    # ------------------------------------------------------------------

    def _on_roi_kind_changed(self, _idx: int) -> None:
        kind = self._roi_combo.currentData()
        if kind is None:
            self._roi = None
        elif kind == ROI_POLYGON:
            # Seed a triangle the user can drag into shape; pyqtgraph's PolyLineROI
            # has no click-to-draw mode, so it starts from a visible default.
            self._roi = {"kind": kind,
                         "points": [(0.3, 0.3), (0.7, 0.35), (0.5, 0.7)]}
        else:
            self._roi = {"kind": kind, "x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75}
        for view in self._views:
            view.set_roi(self._roi)
        # Selecting a region is only useful if you can then move it.
        if kind is not None and self._tool != TOOL_ROI:
            self._tool_combo.setCurrentIndex(
                self._tool_combo.findData(TOOL_ROI))
        self._apply_tool()
        self._request_regions()

    def _on_roi_changed(self, roi: dict) -> None:
        if self._syncing:
            return
        self._roi = roi
        self._syncing = True
        try:
            for view in self._views:
                if view is not self.sender():
                    view.set_roi(roi)
        finally:
            self._syncing = False
        self._request_regions()

    def _on_threshold_toggled(self, _on: bool) -> None:
        self._request_regions()

    def _on_threshold_mode_changed(self, _idx: int) -> None:
        """Re-seed the spin box when switching units, so the value stays meaningful."""
        mode = self._thresh_mode.currentData()
        if mode == THRESH_PERCENTILE:
            self._thresh_value.setSuffix(" %")
            self._thresh_value.setRange(0.0, 100.0)
            self._thresh_value.setValue(10.0)
        else:
            self._thresh_value.setSuffix("")
            self._thresh_value.setRange(-1e9, 1e9)
            if self._arr_a is not None:
                arr = to_2d(self._arr_a)
                self._thresh_value.setValue(float(np.percentile(arr, 90.0)))
        self._request_regions()

    def _request_regions(self, *_args) -> None:
        """Queue a recompute. Debounced — an ROI drag emits a continuous stream."""
        self._region_timer.start(120)

    def _start_region_job(self) -> None:
        if self._arr_a is None or self._compare is None:
            self._clear_region_outputs()
            return
        self._region_token += 1
        req = {
            "token": self._region_token,
            "a": to_2d(self._arr_a),
            "b": to_2d(self._arr_b) if self._arr_b is not None else to_2d(self._arr_a),
            "comparison": self._compare,
            "roi": self._roi,
            "threshold_on": self._thresh_chk.isChecked(),
            "thresh_mode": self._thresh_mode.currentData(),
            "thresh_value": float(self._thresh_value.value()),
            "direction": self._thresh_dir.currentData(),
            "open_px": self._open_spin.value(),
            "close_px": self._close_spin.value(),
            "min_size_px": self._minsize_spin.value(),
            "fill_hole_px": self._minsize_spin.value(),
        }
        old = self._region_thread
        if old is not None and old.isRunning():
            # Supersede: drop the old result rather than block the GUI waiting for it.
            try:
                old.done.disconnect()
            except TypeError:
                pass
            old.quit()
        self._region_thread = _RegionThread(req, parent=self)
        self._region_thread.done.connect(self._on_regions_ready)
        self._region_thread.start()

    def _on_regions_ready(self, result: dict) -> None:
        # A slow run must never overwrite newer UI state.
        if result.get("token") != self._region_token:
            return
        if result.get("error"):
            self._warn_lbl.setText(f"Region error: {result['error']}")
            return

        self._warn_lbl.setText(result.get("warning") or "")
        self._region_shape = result["sample_in"].shape

        self._mask_for_overlay = result["in_mask"]
        self._set_overlay_visible("mask", self._mask_chk.isChecked())

        crange = comparison_range(self._compare) if self._compare is not None else (-1, 1)
        label = COMPARE_LABELS[self._compare_mode()]
        self._hist.set_x_label(label)
        self._hist.update_histogram(result["hist_values"], crange)

        for plot, sample, region in (
            (self._corr_in, result["sample_in"], "In mask"),
            (self._corr_out, result["sample_out"], "Outside mask"),
        ):
            plot.set_sample(sample, crange, self._data.label_a,
                            self._data.label_b, region)
            plot.set_selected_ids(self._selected_ids)

    def _clear_region_outputs(self) -> None:
        for plot in (self._corr_in, self._corr_out):
            plot.set_sample(None, (-1.0, 1.0), self._data.label_a,
                            self._data.label_b, "")
        self._mask_for_overlay = None
        for view in self._views:
            view.set_overlay("mask", None)

    # ------------------------------------------------------------------
    # Overlays and scatter selection
    # ------------------------------------------------------------------

    def _set_overlay_visible(self, which: str, visible: bool) -> None:
        source = (self._mask_for_overlay if which == "mask"
                  else self._selection_mask())
        for view in self._views:
            view.set_overlay(which, source if visible else None)

    def _selection_mask(self) -> np.ndarray | None:
        if self._selected_ids is None or self._region_shape is None:
            return None
        return ids_to_mask(self._selected_ids, self._region_shape)

    def _pick_color(self, which: str) -> None:
        btn = self._mask_color_btn if which == "mask" else self._sel_color_btn
        initial = QColor(btn.property("color"))
        chosen = QColorDialog.getColor(initial, self, f"{which.title()} overlay colour")
        if not chosen.isValid():
            return
        name = chosen.name()
        btn.setProperty("color", name)
        btn.setStyleSheet(f"background-color: {name}; border: 1px solid #888;")
        for view in self._views:
            view.set_overlay_style(which, color=name)

    def _on_scatter_selection(self, ids) -> None:
        self._selected_ids = np.asarray(ids, dtype=np.int64)
        # Both plots highlight the same ids, so a selection made in one is visible in
        # the other; the images get the matching pixels.
        for plot in (self._corr_in, self._corr_out):
            plot.set_selected_ids(self._selected_ids)
        self._set_overlay_visible("selection", self._sel_chk.isChecked())

    def _clear_selection(self) -> None:
        self._selected_ids = None
        for plot in (self._corr_in, self._corr_out):
            plot.set_selected_ids(None)
        for view in self._views:
            view.set_overlay("selection", None)

    def _update_profiles(self) -> None:
        if self._arr_a is None:
            self._xsec.update_profiles(None, None, None, None)
            return
        x0n, y0n, x1n, y1n = self._line
        pos, prof_a = _SDA._sample_line(to_2d(self._arr_a), x0n, y0n, x1n, y1n)
        prof_b = None
        if self._arr_b is not None:
            _, prof_b = _SDA._sample_line(to_2d(self._arr_b), x0n, y0n, x1n, y1n)
        prof_c = None
        if self._compare is not None:
            _, prof_c = _SDA._sample_line(self._compare, x0n, y0n, x1n, y1n)
        self._xsec.set_labels(self._data.label_a, self._data.label_b)
        self._xsec.update_profiles(pos, prof_a, prof_b, prof_c)

    def _on_compare_curve_toggled(self, checked: bool) -> None:
        self._xsec.set_comparison_visible(checked)
        self._update_profiles()

    # ------------------------------------------------------------------
    # Zoom / pan state, also normalised so it survives a panel change
    # ------------------------------------------------------------------

    def _capture_view(self) -> None:
        shape = self._view_a.shape
        if shape is None:
            return
        h, w = shape
        (x0, x1), (y0, y1) = self._view_a.view_box.viewRange()
        # Clamp to the image before storing.  An aspect-locked ViewBox always
        # *expands* the axis with spare room, so storing the expanded range and
        # re-applying it would let the other axis expand in turn — the view would
        # zoom out a little further on every panel change.  Clamping to the image
        # bounds makes capture/apply idempotent, and "which part of the image am I
        # looking at" is the only part worth carrying to a different-sized panel.
        self._view = (max(0.0, x0 / w), min(1.0, x1 / w),
                      max(0.0, y0 / h), min(1.0, y1 / h))

    def _apply_view(self) -> None:
        shape = self._view_a.shape
        if shape is None:
            return
        h, w = shape
        x0n, x1n, y0n, y1n = self._view
        self._view_a.view_box.setRange(xRange=(x0n * w, x1n * w),
                                       yRange=(y0n * h, y1n * h), padding=0)

    def _on_reset_zoom(self) -> None:
        self._view = _FULL_VIEW
        self._apply_view()

    def _refit(self) -> None:
        """Re-fit the current region into whatever size the panel now has.

        An aspect-locked ViewBox answers a resize by *preserving scale* rather than
        refitting, so any layout change leaves the image at the wrong size until the
        range is re-applied — this was the initial-open bug, and hiding Image B for
        swipe mode reproduced it exactly.  Capture-then-apply is safe to call at any
        time: _capture_view clamps to the image bounds, so it round-trips without
        drift and it preserves a zoom the user set with the wheel (which never went
        through _view).
        """
        self._capture_view()
        self._apply_view()

    def showEvent(self, event) -> None:
        """Re-fit once the window has its real geometry.

        _load_panels() runs during __init__, when the ViewBox still has its
        placeholder size, so the first panel opened at roughly a third of the space
        it could fill.  singleShot(0) runs once the layout has settled; the flag
        keeps a later hide/show from disturbing the user's own zoom.
        """
        super().showEvent(event)
        if not self._did_first_fit:
            self._did_first_fit = True
            QTimer.singleShot(0, self._apply_view)

    def _on_viewbox_resized(self) -> None:
        """Re-fit whenever the image panel changes size, from any cause.

        Reacting to the ViewBox's own sigResized rather than the window's
        resizeEvent is what makes this reliable: hiding Image B for swipe mode
        resizes the panel without resizing the window, and an aspect-locked ViewBox
        answers that resize by blowing the range up (measured: x went to
        [-2228, 3063] for an 835-px-wide image).  Qt also finishes the splitter
        re-layout *after* the handler that triggered it returns, so a refit issued
        at that moment is immediately undone — only the resize signal fires late
        enough to be authoritative.
        """
        # Debounced: a drag-resize emits a continuous stream of these.
        self._resize_timer.start(60)

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._region_timer.stop()
        if self._region_thread is not None and self._region_thread.isRunning():
            self._region_thread.quit()
            self._region_thread.wait(2000)
        self._corr_in.shutdown()
        self._corr_out.shutdown()
        try:
            self._npz.close()
        except Exception:
            pass
        super().closeEvent(event)
