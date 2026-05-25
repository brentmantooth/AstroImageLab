"""Report Inspector — interactive image viewer with cross-section tool.

Opened automatically after analysis completes; reads from a companion _inspector.npz
file written alongside each HTML report so it works for past runs too.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as _PIL

import matplotlib
matplotlib.rcParams.update({
    "figure.facecolor": "#1e1e1e",
    "axes.facecolor":   "#1e1e1e",
    "axes.edgecolor":   "#555555",
    "text.color":       "#dddddd",
    "axes.labelcolor":  "#dddddd",
    "xtick.color":      "#aaaaaa",
    "ytick.color":      "#aaaaaa",
    "grid.color":       "#333333",
    "legend.facecolor": "#2a2a2a",
    "legend.edgecolor": "#555555",
})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QFrame, QSizePolicy,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _bilinear_sample(data: np.ndarray, y: float, x: float) -> float:
    h, w = data.shape[:2]
    x0 = max(0, min(int(x), w - 1))
    y0 = max(0, min(int(y), h - 1))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0;      fy = y - y0
    return float(
        data[y0, x0] * (1 - fx) * (1 - fy) +
        data[y0, x1] *      fx  * (1 - fy) +
        data[y1, x0] * (1 - fx) *      fy  +
        data[y1, x1] *      fx  *      fy
    )


def _sample_line(data: np.ndarray, p0: tuple, p1: tuple,
                  n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear cross-section. Returns (distance_px, values normalized to [0,1])."""
    x0, y0 = p0
    x1, y1 = p1
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist < 1.0:
        return np.array([0.0]), np.array([0.0])
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    fdata = data.astype(np.float32)
    if fdata.ndim == 3:
        fdata = (0.2126 * fdata[:, :, 0] +
                 0.7152 * fdata[:, :, 1] +
                 0.0722 * fdata[:, :, 2])
    profile = np.array([_bilinear_sample(fdata, float(y), float(x))
                        for x, y in zip(xs, ys)], dtype=np.float32)
    peak = float(profile.max())
    if peak > 0:
        profile /= peak
    return np.linspace(0.0, dist, n), profile


def _to_uint8_display(arr: np.ndarray) -> np.ndarray:
    """Normalize any float array to uint8 via 1st–99th percentile clip."""
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    name:    str
    options: dict[str, str]   # display_label → npz_key; ordered


@dataclass
class _InspectorData:
    label_a: str
    label_b: str
    sim_label_ref: str
    sections: dict[str, list[_Entry]] = field(default_factory=dict)


def _load_inspector_data(npz: "np.lib.npyio.NpzFile") -> _InspectorData:
    json_bytes = npz["catalog_json"].tobytes()
    cat = json.loads(json_bytes.decode("utf-8"))
    data = _InspectorData(
        label_a=cat.get("label_a", "Image A"),
        label_b=cat.get("label_b", "Image B"),
        sim_label_ref=cat.get("sim_label_ref", ""),
    )
    npz_keys = set(npz.files)
    for section, entries in cat.get("sections", {}).items():
        parsed = []
        for e in entries:
            # Support both new {options: {...}} and legacy {key_a, key_b} formats
            if "options" in e:
                opts = {k: v for k, v in e["options"].items() if v in npz_keys}
            else:
                opts = {}
                if e.get("key_a") in npz_keys:
                    opts["Image A"] = e["key_a"]
                if e.get("key_b") in npz_keys:
                    opts["Image B"] = e["key_b"]
            if opts:
                parsed.append(_Entry(name=e["name"], options=opts))
        if parsed:
            data.sections[section] = parsed
    return data


def _get_array(npz: "np.lib.npyio.NpzFile", key: str) -> np.ndarray | None:
    try:
        return npz[key]
    except KeyError:
        return None


def _prepare_for_display(arr: np.ndarray) -> np.ndarray:
    """Convert any array to uint8 suitable for imshow."""
    if arr.dtype == np.uint8:
        return arr
    return _to_uint8_display(arr)


# ---------------------------------------------------------------------------
# Image canvas widget
# ---------------------------------------------------------------------------

class InspectorImageCanvas(QWidget):
    """Matplotlib canvas with two-click cross-section line tool and slider mode."""

    line_updated   = pyqtSignal(object, object)   # (p0, p1) tuples or (None, None)
    reveal_changed = pyqtSignal(float)             # [0, 1]

    def __init__(self, label_a: str, label_b: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._label_a = label_a
        self._label_b = label_b

        self._arr_a: np.ndarray | None = None
        self._arr_b: np.ndarray | None = None
        self._mode = "side_by_side"
        self._reveal = 0.5
        self._state = "idle"      # "idle" | "drawing" | "fixed"
        self._p0: tuple | None = None
        self._p1: tuple | None = None
        self._drag_divider = False

        self._fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._ax_a: Any = None
        self._ax_b: Any = None
        self._cid_press   = self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._cid_motion  = self._canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_entry(self, arr_a: np.ndarray, arr_b: np.ndarray,
                   keep_line: bool = False) -> None:
        old_shape = self._arr_a.shape[:2] if self._arr_a is not None else None
        new_shape = arr_a.shape[:2]
        self._arr_a = arr_a
        self._arr_b = arr_b
        if not (keep_line and old_shape is not None and old_shape == new_shape):
            self._state = "idle"
            self._p0 = self._p1 = None
            self.line_updated.emit(None, None)
        elif self._p0 is not None and self._p1 is not None:
            # Arrays replaced but line retained — re-emit so profile resamples from new data
            self.line_updated.emit(self._p0, self._p1)
        self._rebuild_axes()
        self._redraw()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._rebuild_axes()
        self._redraw()

    def set_reveal(self, frac: float) -> None:
        self._reveal = max(0.01, min(0.99, frac))
        if self._mode == "slider":
            self._redraw()

    def set_labels(self, la: str, lb: str) -> None:
        self._label_a = la
        self._label_b = lb
        self._redraw()

    def clear_line(self) -> None:
        self._state = "idle"
        self._p0 = self._p1 = None
        self._redraw()
        self.line_updated.emit(None, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_axes(self) -> None:
        self._fig.clear()
        if self._mode == "side_by_side":
            axes = self._fig.subplots(1, 2)
            self._ax_a, self._ax_b = axes[0], axes[1]
        else:
            self._ax_a = self._fig.add_subplot(1, 1, 1)
            self._ax_b = None

    def _cmap(self, arr: np.ndarray) -> str | None:
        return "gray" if arr.ndim == 2 else None

    def _redraw(self) -> None:
        if self._arr_a is None:
            self._canvas.draw_idle()
            return

        if self._mode == "side_by_side":
            self._ax_a.cla()
            self._ax_b.cla()
            self._ax_a.imshow(self._arr_a, cmap=self._cmap(self._arr_a),
                              origin="upper", aspect="equal",
                              vmin=0, vmax=255 if self._arr_a.dtype == np.uint8 else None)
            self._ax_b.imshow(self._arr_b, cmap=self._cmap(self._arr_b),
                              origin="upper", aspect="equal",
                              vmin=0, vmax=255 if self._arr_b.dtype == np.uint8 else None)
            self._ax_a.set_title(self._label_a, fontsize=9, pad=3)
            self._ax_b.set_title(self._label_b, fontsize=9, pad=3)
            for ax in (self._ax_a, self._ax_b):
                ax.set_xticks([]);  ax.set_yticks([])
            if self._p0 is not None and self._p1 is not None:
                for ax in (self._ax_a, self._ax_b):
                    ax.plot([self._p0[0], self._p1[0]],
                            [self._p0[1], self._p1[1]],
                            color="cyan", lw=1.5, solid_capstyle="round")
                    ax.plot(*self._p0, "+", color="cyan", ms=10, mew=1.5)
                    ax.plot(*self._p1, "+", color="cyan", ms=10, mew=1.5)
        else:
            # Slider / composite mode
            arr_a = self._arr_a
            arr_b = self._arr_b
            W = arr_a.shape[1]
            split = int(W * self._reveal)
            if arr_a.ndim == 2:
                composite = arr_a.copy()
                composite[:, split:] = arr_b[:, split:]
            else:
                composite = arr_a.copy()
                composite[:, split:, :] = arr_b[:, split:, :]
            self._ax_a.cla()
            self._ax_a.imshow(composite, cmap=self._cmap(arr_a),
                              origin="upper", aspect="equal",
                              vmin=0, vmax=255 if composite.dtype == np.uint8 else None)
            self._ax_a.axvline(split, color="white", lw=1.5, ls="--", alpha=0.8)
            self._ax_a.set_title(
                f"{self._label_a}  ←|→  {self._label_b}", fontsize=9, pad=3)
            self._ax_a.set_xticks([]);  self._ax_a.set_yticks([])
            if self._p0 is not None and self._p1 is not None:
                self._ax_a.plot([self._p0[0], self._p1[0]],
                                [self._p0[1], self._p1[1]],
                                color="yellow", lw=1.5, solid_capstyle="round")
                self._ax_a.plot(*self._p0, "+", color="yellow", ms=10, mew=1.5)
                self._ax_a.plot(*self._p1, "+", color="yellow", ms=10, mew=1.5)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def _clamp_pt(self, x: float, y: float) -> tuple:
        if self._arr_a is None:
            return (x, y)
        H, W = self._arr_a.shape[:2]
        return (max(0.0, min(float(x), W - 1)),
                max(0.0, min(float(y), H - 1)))

    def _split_col(self) -> int:
        if self._arr_a is None:
            return 0
        return int(self._arr_a.shape[1] * self._reveal)

    def _on_press(self, event) -> None:
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if self._arr_a is None:
            return
        # Determine which axes received the event
        active_ax = self._ax_a
        if self._mode == "side_by_side" and event.inaxes == self._ax_b:
            active_ax = self._ax_b
        if event.inaxes not in (self._ax_a, self._ax_b):
            return

        pt = self._clamp_pt(event.xdata, event.ydata)

        # Slider mode: check if click is on/near the divider line
        if self._mode == "slider":
            split = self._split_col()
            # Convert 12 display pixels to data coords
            try:
                inv = self._ax_a.transData.inverted()
                disp_pt = self._ax_a.transData.transform((split, 0))
                data_pt = inv.transform((disp_pt[0] + 12, 0))
                tol = abs(data_pt[0] - split)
            except Exception:
                tol = 12.0
            if abs(pt[0] - split) <= tol:
                self._drag_divider = True
                return

        # Cross-section click
        if self._state in ("idle", "fixed"):
            self._p0 = pt
            self._p1 = pt
            self._state = "drawing"
            self._redraw()
            self.line_updated.emit(None, None)
        elif self._state == "drawing":
            self._p1 = pt
            self._state = "fixed"
            self._redraw()
            self.line_updated.emit(self._p0, self._p1)

    def _on_motion(self, event) -> None:
        if self._arr_a is None:
            return
        if self._drag_divider and self._mode == "slider" and event.xdata is not None:
            W = self._arr_a.shape[1]
            self._reveal = max(0.01, min(0.99, event.xdata / W))
            self._redraw()
            self.reveal_changed.emit(self._reveal)
            return
        if self._state == "drawing" and event.xdata is not None and event.ydata is not None:
            self._p1 = self._clamp_pt(event.xdata, event.ydata)
            self._redraw()
            if self._p0 is not None and self._p1 is not None:
                self.line_updated.emit(self._p0, self._p1)

    def _on_release(self, event) -> None:
        self._drag_divider = False


# ---------------------------------------------------------------------------
# Profile canvas widget
# ---------------------------------------------------------------------------

class _ProfileCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(200)
        self._fig = plt.figure(figsize=(10, 2.2), constrained_layout=True)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._ax.set_xlabel("Distance (px)", fontsize=8)
        self._ax.set_ylabel("Normalized intensity", fontsize=8)
        self._ax.grid(True, alpha=0.25)
        self._ax.tick_params(labelsize=7)
        self._canvas.draw_idle()

    def update_profiles(self, dist_a: np.ndarray, prof_a: np.ndarray,
                         dist_b: np.ndarray, prof_b: np.ndarray,
                         label_a: str, label_b: str) -> None:
        self._ax.cla()
        self._ax.plot(dist_a, prof_a, color="steelblue", lw=1.2, label=label_a)
        self._ax.plot(dist_b, prof_b, color="tomato",    lw=1.2, label=label_b)
        self._ax.set_xlabel("Distance (px)", fontsize=8)
        self._ax.set_ylabel("Normalized intensity", fontsize=8)
        self._ax.legend(fontsize=7, loc="upper right")
        self._ax.grid(True, alpha=0.25)
        self._ax.tick_params(labelsize=7)
        self._canvas.draw_idle()

    def clear(self) -> None:
        self._ax.cla()
        self._ax.set_xlabel("Distance (px)", fontsize=8)
        self._ax.set_ylabel("Normalized intensity", fontsize=8)
        self._ax.grid(True, alpha=0.25)
        self._ax.tick_params(labelsize=7)
        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main inspector window
# ---------------------------------------------------------------------------

class ReportInspector(QMainWindow):
    """Standalone window for interactive image comparison and cross-section analysis."""

    def __init__(self, npz_path: str | Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Astro Image Lab — Report Inspector")
        self.setMinimumSize(1000, 750)

        self._npz = np.load(str(npz_path), allow_pickle=False)
        self._data = _load_inspector_data(self._npz)
        self._current_arr_a: np.ndarray | None = None
        self._current_arr_b: np.ndarray | None = None
        self._current_shape: tuple | None = None

        self._build_ui()
        self._populate_section_combo()
        if self._data.sections:
            self._section_combo.setCurrentIndex(0)
            self._on_section_changed(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Top toolbar row ────────────────────────────────────────────
        top_row = QHBoxLayout()

        top_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Side by Side", "Before/After Slider"])
        self._mode_combo.setFixedWidth(160)
        top_row.addWidget(self._mode_combo)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        top_row.addWidget(sep)

        top_row.addWidget(QLabel("Section:"))
        self._section_combo = QComboBox()
        self._section_combo.setMinimumWidth(200)
        top_row.addWidget(self._section_combo)

        top_row.addWidget(QLabel("Image set:"))
        self._image_set_combo = QComboBox()
        self._image_set_combo.setMinimumWidth(160)
        top_row.addWidget(self._image_set_combo)

        top_row.addWidget(QLabel("Left:"))
        self._left_combo = QComboBox()
        self._left_combo.setMinimumWidth(130)
        top_row.addWidget(self._left_combo)

        top_row.addWidget(QLabel("Right:"))
        self._right_combo = QComboBox()
        self._right_combo.setMinimumWidth(130)
        top_row.addWidget(self._right_combo)

        top_row.addStretch()
        root.addLayout(top_row)

        # ── Slider reveal row (hidden in side-by-side mode) ────────────
        self._slider_row_widget = QWidget()
        slider_layout = QHBoxLayout(self._slider_row_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(4)
        # Left spacer — dynamically sized to match the axes left margin so the
        # slider track lines up with the left edge of the displayed image.
        self._slider_left_spacer = QWidget()
        self._slider_left_spacer.setFixedWidth(0)
        slider_layout.addWidget(self._slider_left_spacer)
        self._reveal_slider = QSlider(Qt.Orientation.Horizontal)
        self._reveal_slider.setRange(0, 100)
        self._reveal_slider.setValue(50)
        slider_layout.addWidget(self._reveal_slider, stretch=1)
        # Right spacer — mirrors the axes right margin.
        self._slider_right_spacer = QWidget()
        self._slider_right_spacer.setFixedWidth(0)
        slider_layout.addWidget(self._slider_right_spacer)
        self._reveal_pct_label = QLabel("50 %")
        self._reveal_pct_label.setFixedWidth(40)
        slider_layout.addWidget(self._reveal_pct_label)
        self._slider_row_widget.setVisible(False)
        root.addWidget(self._slider_row_widget)

        # ── Image canvas ────────────────────────────────────────────────
        self._image_canvas = InspectorImageCanvas(
            self._data.label_a, self._data.label_b, self)
        root.addWidget(self._image_canvas, stretch=3)

        # ── Profile graph ───────────────────────────────────────────────
        self._profile_canvas = _ProfileCanvas(self)
        root.addWidget(self._profile_canvas)

        # ── Signal wiring ───────────────────────────────────────────────
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._section_combo.currentIndexChanged.connect(self._on_section_changed)
        self._image_set_combo.currentIndexChanged.connect(self._on_image_set_changed)
        self._left_combo.currentIndexChanged.connect(self._on_left_changed)
        self._right_combo.currentIndexChanged.connect(self._on_right_changed)
        self._reveal_slider.valueChanged.connect(self._on_reveal_slider_changed)
        self._image_canvas.line_updated.connect(self._on_line_updated)
        self._image_canvas.reveal_changed.connect(self._on_reveal_changed)
        # Sync slider margins after every matplotlib draw so spacers stay aligned
        # with the axes bounds (which shift slightly on resize or content change).
        self._image_canvas._canvas.mpl_connect("draw_event", self._on_canvas_drawn)

    # ------------------------------------------------------------------
    # Combo population
    # ------------------------------------------------------------------

    def _populate_section_combo(self) -> None:
        self._section_combo.blockSignals(True)
        self._section_combo.clear()
        for section in self._data.sections:
            self._section_combo.addItem(section)
        self._section_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, idx: int) -> None:
        if idx == 0:
            self._slider_row_widget.setVisible(False)
            self._image_canvas.set_mode("side_by_side")
        else:
            self._slider_row_widget.setVisible(True)
            self._image_canvas.set_mode("slider")
            self._sync_slider_margins()

    def _on_canvas_drawn(self, event) -> None:
        if self._mode_combo.currentIndex() == 1:
            self._sync_slider_margins()

    def _sync_slider_margins(self) -> None:
        """Resize the slider's left/right spacers to align its track with the image axes."""
        ax = self._image_canvas._ax_a
        if ax is None:
            return
        pos = ax.get_position()   # axes bbox as fractions of the figure [0, 1]
        w = self._image_canvas._canvas.width()
        new_left  = max(0, int(pos.x0 * w))
        new_right = max(0, int((1.0 - pos.x1) * w))
        if (self._slider_left_spacer.width()  != new_left or
                self._slider_right_spacer.width() != new_right):
            self._slider_left_spacer.setFixedWidth(new_left)
            self._slider_right_spacer.setFixedWidth(new_right)

    def _on_section_changed(self, idx: int) -> None:
        section = self._section_combo.currentText()
        entries = self._data.sections.get(section, [])
        self._image_set_combo.blockSignals(True)
        self._image_set_combo.clear()
        for entry in entries:
            self._image_set_combo.addItem(entry.name)
        self._image_set_combo.blockSignals(False)
        self._on_image_set_changed(0)

    def _on_image_set_changed(self, idx: int) -> None:
        section = self._section_combo.currentText()
        entries = self._data.sections.get(section, [])
        if idx < 0 or idx >= len(entries):
            return
        option_labels = list(entries[idx].options.keys())
        self._left_combo.blockSignals(True)
        self._right_combo.blockSignals(True)
        self._left_combo.clear()
        self._right_combo.clear()
        self._left_combo.addItems(option_labels)
        self._right_combo.addItems(option_labels)
        if len(option_labels) >= 2:
            self._right_combo.setCurrentIndex(1)
        self._left_combo.blockSignals(False)
        self._right_combo.blockSignals(False)
        self._load_current_panels()

    def _on_left_changed(self, idx: int) -> None:
        self._load_current_panels()

    def _on_right_changed(self, idx: int) -> None:
        self._load_current_panels()

    def _on_reveal_slider_changed(self, val: int) -> None:
        self._reveal_pct_label.setText(f"{val} %")
        self._image_canvas.set_reveal(val / 100.0)

    def _on_reveal_changed(self, frac: float) -> None:
        self._reveal_slider.blockSignals(True)
        self._reveal_slider.setValue(int(frac * 100))
        self._reveal_pct_label.setText(f"{int(frac * 100)} %")
        self._reveal_slider.blockSignals(False)

    def _on_line_updated(self, p0, p1) -> None:
        if p0 is None or self._current_arr_a is None:
            self._profile_canvas.clear()
            return
        dist_a, prof_a = _sample_line(self._current_arr_a, p0, p1)
        # Scale line coordinates to the right panel's pixel space if sizes differ.
        # p0/p1 are in the left panel's (arr_a) pixel coordinate system.
        ha, wa = self._current_arr_a.shape[:2]
        hb, wb = self._current_arr_b.shape[:2]
        if (ha, wa) != (hb, wb):
            p0b = (p0[0] * wb / wa, p0[1] * hb / ha)
            p1b = (p1[0] * wb / wa, p1[1] * hb / ha)
        else:
            p0b, p1b = p0, p1
        dist_b, prof_b = _sample_line(self._current_arr_b, p0b, p1b)
        self._profile_canvas.update_profiles(
            dist_a, prof_a, dist_b, prof_b,
            self._left_combo.currentText(),
            self._right_combo.currentText())

    # ------------------------------------------------------------------
    # Panel loading
    # ------------------------------------------------------------------

    def _load_current_panels(self) -> None:
        section = self._section_combo.currentText()
        img_set_idx = self._image_set_combo.currentIndex()
        entries = self._data.sections.get(section, [])
        if img_set_idx < 0 or img_set_idx >= len(entries):
            return
        entry = entries[img_set_idx]

        left_label  = self._left_combo.currentText()
        right_label = self._right_combo.currentText()
        if not left_label or not right_label:
            return
        key_left  = entry.options.get(left_label)
        key_right = entry.options.get(right_label)
        if key_left is None or key_right is None:
            return

        raw_left  = _get_array(self._npz, key_left)
        raw_right = _get_array(self._npz, key_right)
        if raw_left is None or raw_right is None:
            return

        arr_left  = _prepare_for_display(raw_left)
        arr_right = _prepare_for_display(raw_right)

        old_shape = self._current_shape
        new_shape = arr_left.shape[:2]
        keep_line = (old_shape is not None and old_shape == new_shape)

        self._current_arr_a = raw_left.astype(np.float32)
        self._current_arr_b = raw_right.astype(np.float32)
        if self._current_arr_a.max() > 1.5:
            self._current_arr_a /= 255.0
        if self._current_arr_b.max() > 1.5:
            self._current_arr_b /= 255.0
        self._current_shape = new_shape

        self._image_canvas.set_labels(left_label, right_label)
        self._image_canvas.load_entry(arr_left, arr_right, keep_line=keep_line)
        if not keep_line:
            self._profile_canvas.clear()
