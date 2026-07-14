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

from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QFrame, QSizePolicy, QCheckBox, QToolTip,
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
                  n: int = 512, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear cross-section.

    Returns (distance_px, profile). When normalize=True each profile is divided
    by its own peak so it maps to [0, 1]. When normalize=False the raw array
    values are returned, keeping both profiles on a shared absolute scale.
    """
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
    if normalize:
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
    concept: str = ""         # optional explanatory text (Spatial Detail section)


@dataclass
class _InspectorData:
    label_a: str
    label_b: str
    sim_label_ref: str
    filename_a: str = ""
    filename_b: str = ""
    single_image: bool = False
    sections: dict[str, list[_Entry]] = field(default_factory=dict)


def _load_inspector_data(npz: "np.lib.npyio.NpzFile") -> _InspectorData:
    json_bytes = npz["catalog_json"].tobytes()
    cat = json.loads(json_bytes.decode("utf-8"))
    data = _InspectorData(
        label_a=cat.get("label_a", "Image A"),
        label_b=cat.get("label_b", "Image B"),
        sim_label_ref=cat.get("sim_label_ref", ""),
        filename_a=cat.get("filename_a", ""),
        filename_b=cat.get("filename_b", ""),
        single_image=cat.get("single_image", False),
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
                parsed.append(_Entry(name=e["name"], options=opts,
                                      concept=e.get("concept", "")))
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

        # Pan state (right-click drag)
        self._pan_drag   = False
        self._pan_last_x = 0.0   # display-coord x at last pan motion event
        self._pan_last_y = 0.0
        self._pan_ax     = None  # axes being panned

        # Saved zoom/pan: keyed "a"/"b" → (xlim, ylim). Empty = full-image view.
        self._view_state: dict = {}

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
        self._cid_scroll  = self._canvas.mpl_connect("scroll_event",         self._on_scroll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_entry(self, arr_a: np.ndarray, arr_b: np.ndarray | None,
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

    def _save_view_state(self) -> None:
        """Snapshot the current xlim/ylim of active axes for restore after _redraw."""
        state: dict = {}
        if self._ax_a is not None:
            state["a"] = (self._ax_a.get_xlim(), self._ax_a.get_ylim())
        if self._ax_b is not None:
            state["b"] = (self._ax_b.get_xlim(), self._ax_b.get_ylim())
        self._view_state = state

    def _rebuild_axes(self) -> None:
        self._fig.clear()
        self._view_state = {}
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
            self._ax_a.set_title(self._label_a, fontsize=9, pad=3)
            self._ax_a.set_xticks([]);  self._ax_a.set_yticks([])
            if self._arr_b is not None:
                self._ax_b.imshow(self._arr_b, cmap=self._cmap(self._arr_b),
                                  origin="upper", aspect="equal",
                                  vmin=0, vmax=255 if self._arr_b.dtype == np.uint8 else None)
                self._ax_b.set_title(self._label_b, fontsize=9, pad=3)
                self._ax_b.set_xticks([]);  self._ax_b.set_yticks([])
            else:
                # Single-image mode: grey placeholder in right panel
                H, W = self._arr_a.shape[:2]
                self._ax_b.imshow(np.full((H, W), 40, dtype=np.uint8),
                                  cmap="gray", origin="upper", aspect="equal",
                                  vmin=0, vmax=255)
                self._ax_b.text(W / 2, H / 2, "No Image B",
                                ha="center", va="center",
                                color="#aaaaaa", fontsize=13)
                self._ax_b.set_title(self._label_b, fontsize=9, pad=3)
                self._ax_b.set_xticks([]);  self._ax_b.set_yticks([])
            if self._p0 is not None and self._p1 is not None:
                draw_axes = ([self._ax_a, self._ax_b] if self._arr_b is not None
                             else [self._ax_a])
                for ax in draw_axes:
                    ax.plot([self._p0[0], self._p1[0]],
                            [self._p0[1], self._p1[1]],
                            color="cyan", lw=1.5, solid_capstyle="round")
                    ax.plot(*self._p0, "+", color="cyan", ms=10, mew=1.5)
                    ax.plot(*self._p1, "+", color="cyan", ms=10, mew=1.5)
        else:
            # Slider / composite mode
            arr_a = self._arr_a
            arr_b = self._arr_b if self._arr_b is not None else arr_a
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

        # Restore zoom/pan state saved before this redraw so that cross-section
        # clicks and other _redraw() triggers don't reset the view.
        if self._view_state:
            if "a" in self._view_state and self._ax_a is not None:
                self._ax_a.set_xlim(self._view_state["a"][0])
                self._ax_a.set_ylim(self._view_state["a"][1])
            if "b" in self._view_state and self._ax_b is not None:
                self._ax_b.set_xlim(self._view_state["b"][0])
                self._ax_b.set_ylim(self._view_state["b"][1])

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
        if event.button == 3 and event.inaxes in (self._ax_a, self._ax_b):
            self._pan_drag   = True
            self._pan_last_x = event.x
            self._pan_last_y = event.y
            self._pan_ax     = event.inaxes
            return
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
        if self._pan_drag and self._pan_ax is not None:
            try:
                inv       = self._pan_ax.transData.inverted()
                last_data = inv.transform((self._pan_last_x, self._pan_last_y))
                curr_data = inv.transform((event.x,          event.y))
                dx = last_data[0] - curr_data[0]
                dy = last_data[1] - curr_data[1]
                xl = self._pan_ax.get_xlim()
                yl = self._pan_ax.get_ylim()
                self._pan_ax.set_xlim(xl[0] + dx, xl[1] + dx)
                self._pan_ax.set_ylim(yl[0] + dy, yl[1] + dy)
                self._sync_view(self._pan_ax)
                self._pan_last_x = event.x
                self._pan_last_y = event.y
                self._save_view_state()
                self._canvas.draw_idle()
            except Exception:
                pass
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
        if event.button == 3:
            self._pan_drag = False
            self._pan_ax   = None
            return
        self._drag_divider = False

    def _sync_view(self, source_ax) -> None:
        """Copy the current normalized view of source_ax to the other axis.

        Uses fractional image coordinates so images with different pixel dimensions
        stay aligned to the same relative region.
        """
        if self._mode != "side_by_side":
            return
        if source_ax is self._ax_a:
            other_ax, source_arr, other_arr = self._ax_b, self._arr_a, self._arr_b
        else:
            other_ax, source_arr, other_arr = self._ax_a, self._arr_b, self._arr_a
        if other_ax is None or source_arr is None or other_arr is None:
            return
        H_s, W_s = source_arr.shape[:2]
        H_o, W_o = other_arr.shape[:2]
        xl = source_ax.get_xlim()
        yl = source_ax.get_ylim()
        # Normalise to [0, 1] fractions of the source image, then scale to other image
        other_ax.set_xlim((xl[0] + 0.5) / W_s * W_o - 0.5,
                           (xl[1] + 0.5) / W_s * W_o - 0.5)
        other_ax.set_ylim((yl[0] + 0.5) / H_s * H_o - 0.5,
                           (yl[1] + 0.5) / H_s * H_o - 0.5)

    def _on_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        active = [ax for ax in (self._ax_a, self._ax_b) if ax is not None]
        if event.inaxes not in active:
            return
        factor = 0.8 if event.button == "up" else (1.0 / 0.8)
        ax = event.inaxes
        xc, yc = event.xdata, event.ydata
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(xc + (xl[0] - xc) * factor, xc + (xl[1] - xc) * factor)
        ax.set_ylim(yc + (yl[0] - yc) * factor, yc + (yl[1] - yc) * factor)
        self._sync_view(ax)
        self._save_view_state()
        self._canvas.draw_idle()

    def reset_zoom(self) -> None:
        """Restore axes to the full-image view."""
        self._view_state = {}
        for ax, arr in [(self._ax_a, self._arr_a), (self._ax_b, self._arr_b)]:
            if ax is None or arr is None:
                continue
            H, W = arr.shape[:2]
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)   # upper origin (y axis flipped for imshow)
        self._canvas.draw_idle()


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

        # Cached profile data for hover lookups — set by update_profiles(), cleared by clear()
        self._dist_a: np.ndarray | None = None
        self._prof_a: np.ndarray | None = None
        self._dist_b: np.ndarray | None = None
        self._prof_b: np.ndarray | None = None
        self._label_a: str = ""
        self._label_b: str = ""
        self._add_hover_line()

        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("axes_leave_event", self._on_leave)
        self._canvas.draw_idle()

    def _add_hover_line(self) -> None:
        # ax.cla() (in update_profiles/clear) detaches any prior line artist, so this
        # must be re-created after every cla() — a fresh Line2D reference each time.
        self._hover_line = self._ax.axvline(
            0, color="#dddddd", lw=1.0, ls=":", alpha=0.8, visible=False)

    def update_profiles(self, dist_a: np.ndarray, prof_a: np.ndarray,
                         dist_b: np.ndarray | None, prof_b: np.ndarray | None,
                         label_a: str, label_b: str,
                         normalized: bool = True,
                         subtract_median: bool = False,
                         is_linear: bool = False) -> None:
        self._dist_a, self._prof_a = dist_a, prof_a
        self._dist_b, self._prof_b = dist_b, prof_b
        self._label_a, self._label_b = label_a, label_b

        self._ax.cla()
        self._ax.plot(dist_a, prof_a, color="steelblue", lw=1.2, label=label_a)
        if dist_b is not None and prof_b is not None:
            self._ax.plot(dist_b, prof_b, color="tomato", lw=1.2, label=label_b)
        self._ax.set_xlabel("Distance (px)", fontsize=8)
        lin = " (linear)" if is_linear else ""
        if normalized and subtract_median:
            ylabel = f"Normalized, bg-subtracted{lin}"
        elif normalized:
            ylabel = f"Normalized{lin}"
        elif subtract_median:
            ylabel = f"Background-subtracted{lin}"
        else:
            ylabel = f"Linear value [0–1]" if is_linear else "Value"
        self._ax.set_ylabel(ylabel, fontsize=8)
        self._ax.legend(fontsize=7, loc="upper right")
        self._ax.grid(True, alpha=0.25)
        self._ax.tick_params(labelsize=7)
        self._add_hover_line()
        self._canvas.draw_idle()

    def clear(self) -> None:
        self._dist_a = self._prof_a = self._dist_b = self._prof_b = None
        self._label_a = self._label_b = ""

        self._ax.cla()
        self._ax.set_xlabel("Distance (px)", fontsize=8)
        self._ax.set_ylabel("Normalized intensity", fontsize=8)
        self._ax.grid(True, alpha=0.25)
        self._ax.tick_params(labelsize=7)
        self._add_hover_line()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Hover tooltip + crosshair
    # ------------------------------------------------------------------

    def _on_motion(self, event) -> None:
        if event.inaxes is not self._ax or self._dist_a is None:
            self._on_leave(event)
            return

        x = float(event.xdata)
        val_a = float(np.interp(x, self._dist_a, self._prof_a))
        lines = [f"x = {x:.4g} px", f"{self._label_a}: {val_a:.4g}"]

        if self._dist_b is not None and self._prof_b is not None:
            val_b = float(np.interp(x, self._dist_b, self._prof_b))
            ratio = val_a / val_b if val_b != 0 else float("nan")
            ratio_str = f"{ratio:.4g}" if np.isfinite(ratio) else "—"
            lines.append(f"{self._label_b}: {val_b:.4g}")
            lines.append(f"Ratio A/B: {ratio_str}")

        self._hover_line.set_xdata([x, x])
        self._hover_line.set_visible(True)
        self._canvas.draw_idle()

        # Use the native Qt cursor position rather than converting matplotlib's
        # event.x/event.y — those are device-pixel coordinates and mapToGlobal
        # expects logical pixels, so on a HiDPI display the tooltip would land
        # far from the cursor. Offset up-right so it doesn't sit on top of the
        # crosshair/data point it's describing.
        pos = QCursor.pos()
        QToolTip.showText(pos + QPoint(16, -16), "\n".join(lines), self._canvas)

    def _on_leave(self, event) -> None:
        if self._hover_line.get_visible():
            self._hover_line.set_visible(False)
            self._canvas.draw_idle()
        QToolTip.hideText()


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
        self._single_image = self._data.single_image
        self._current_arr_a: np.ndarray | None = None
        self._current_arr_b: np.ndarray | None = None
        self._current_shape: tuple | None = None
        self._current_is_linear: bool = False

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

        from PyQt6.QtWidgets import QPushButton
        self._reset_zoom_btn = QPushButton("Reset Zoom")
        self._reset_zoom_btn.setFixedWidth(90)
        top_row.addWidget(self._reset_zoom_btn)

        top_row.addStretch()
        root.addLayout(top_row)

        # ── Filename info row (unobtrusive) ────────────────────────────
        fn_a = self._data.filename_a or self._data.label_a
        fn_b = self._data.filename_b or self._data.label_b
        fn_row = QHBoxLayout()
        if self._single_image:
            fn_text = f"Single Image Analysis — A: {fn_a}"
        elif fn_a or fn_b:
            fn_text = f"A: {fn_a}    │    B: {fn_b}"
        else:
            fn_text = ""
        if fn_text:
            fn_lbl = QLabel(fn_text)
            fn_lbl.setStyleSheet("color: #888888; font-size: 8pt;")
            fn_row.addWidget(fn_lbl)
            fn_row.addStretch()
            root.addLayout(fn_row)

        # ── Concept info box (populated for Spatial Detail entries only) ──
        self._concept_lbl = QLabel()
        self._concept_lbl.setWordWrap(True)
        self._concept_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._concept_lbl.setStyleSheet("color: #6cabdb; font-size: 9pt; padding: 2px 0;")
        self._concept_lbl.setVisible(False)
        root.addWidget(self._concept_lbl)

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

        # ── Profile options row ─────────────────────────────────────────
        prof_opts = QHBoxLayout()
        self._normalize_check = QCheckBox("Normalize to peak")
        self._normalize_check.setChecked(False)   # default: absolute values
        self._normalize_check.setToolTip(
            "When checked, each profile is divided by its own peak → [0, 1].\n"
            "When unchecked, raw array values are shown on a shared scale,\n"
            "preserving absolute differences between the two images.")
        prof_opts.addWidget(self._normalize_check)
        self._subtract_median_check = QCheckBox("Subtract image median")
        self._subtract_median_check.setChecked(False)
        self._subtract_median_check.setToolTip(
            "Subtract each image's global median from its cross-section profile.\n"
            "Removes the background pedestal so profiles start near zero.\n"
            "Applied before 'Normalize to peak' when both are checked.")
        prof_opts.addWidget(self._subtract_median_check)
        prof_opts.addStretch()
        root.addLayout(prof_opts)

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
        self._normalize_check.stateChanged.connect(self._on_normalize_changed)
        self._subtract_median_check.stateChanged.connect(self._on_normalize_changed)
        self._reset_zoom_btn.clicked.connect(self._image_canvas.reset_zoom)

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
            QTimer.singleShot(0, self._sync_slider_margins)

    def _on_canvas_drawn(self, event) -> None:
        if self._mode_combo.currentIndex() == 1:
            self._sync_slider_margins()

    def _sync_slider_margins(self) -> None:
        """Resize the slider's left/right spacers to align its track with the image edges.

        Uses transData transforms to find the actual on-screen pixel positions of the image
        left/right edges (accounting for aspect="equal" pillarboxing inside the axes), then
        offsets for the slider layout's inter-widget spacings and the percentage label.
        """
        ax  = self._image_canvas._ax_a
        arr = self._image_canvas._arr_a
        if ax is None or arr is None:
            return
        W = arr.shape[1]
        try:
            # Image spans [-0.5, W-0.5] in data coords (imshow default pixel-centre convention)
            disp_left  = ax.transData.transform((-0.5,    0))[0]
            disp_right = ax.transData.transform((W - 0.5, 0))[0]
            # Scale renderer pixels → canvas logical pixels
            fig_w_disp = self._image_canvas._fig.get_window_extent().width
            canvas_w   = self._image_canvas._canvas.width()
            if fig_w_disp <= 0:
                return
            scale     = canvas_w / fig_w_disp
            img_left  = disp_left  * scale
            img_right = disp_right * scale
        except Exception:
            return
        # Slider layout (spacing=4 between every adjacent pair):
        #   [left_spacer] [4] [slider_track] [4] [right_spacer] [4] [pct_label(40)]
        # For track-left  == img_left:  left_spacer  = img_left - 4
        # For track-right == img_right: right_spacer = canvas_w - img_right - 48
        new_left  = max(0, int(img_left)  - 4)
        new_right = max(0, int(canvas_w - img_right) - 48)
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
        concept = entries[idx].concept
        if concept:
            self._concept_lbl.setText(concept)
            self._concept_lbl.setVisible(True)
        else:
            self._concept_lbl.clear()
            self._concept_lbl.setVisible(False)
        option_labels = list(entries[idx].options.keys())
        self._left_combo.blockSignals(True)
        self._right_combo.blockSignals(True)
        self._left_combo.clear()
        self._right_combo.clear()
        self._left_combo.addItems(option_labels)
        self._right_combo.addItems(option_labels)
        if len(option_labels) >= 2:
            self._right_combo.setCurrentIndex(1)
        # In single-image mode the right panel shows a placeholder; disable right combo
        self._right_combo.setEnabled(not self._single_image)
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
        normalized      = self._normalize_check.isChecked()
        subtract_median = self._subtract_median_check.isChecked()
        is_linear       = self._current_is_linear

        dist_a, prof_a = _sample_line(self._current_arr_a, p0, p1, normalize=False)
        if subtract_median:
            prof_a = prof_a - float(np.median(self._current_arr_a))
        if normalized:
            peak = float(prof_a.max())
            if peak > 0:
                prof_a = prof_a / peak

        if self._current_arr_b is None:
            self._profile_canvas.update_profiles(
                dist_a, prof_a, None, None,
                self._left_combo.currentText(), "",
                normalized=normalized, subtract_median=subtract_median,
                is_linear=is_linear)
            return

        # Scale line coordinates to the right panel's pixel space if sizes differ.
        # p0/p1 are in the left panel's (arr_a) pixel coordinate system.
        ha, wa = self._current_arr_a.shape[:2]
        hb, wb = self._current_arr_b.shape[:2]
        if (ha, wa) != (hb, wb):
            p0b = (p0[0] * wb / wa, p0[1] * hb / ha)
            p1b = (p1[0] * wb / wa, p1[1] * hb / ha)
        else:
            p0b, p1b = p0, p1

        dist_b, prof_b = _sample_line(self._current_arr_b, p0b, p1b, normalize=False)
        if subtract_median:
            prof_b = prof_b - float(np.median(self._current_arr_b))
        if normalized:
            peak = float(prof_b.max())
            if peak > 0:
                prof_b = prof_b / peak

        self._profile_canvas.update_profiles(
            dist_a, prof_a, dist_b, prof_b,
            self._left_combo.currentText(),
            self._right_combo.currentText(),
            normalized=normalized, subtract_median=subtract_median,
            is_linear=is_linear)

    def _on_normalize_changed(self, _state: int) -> None:
        if self._image_canvas._p0 is not None and self._image_canvas._p1 is not None:
            self._on_line_updated(self._image_canvas._p0, self._image_canvas._p1)
        else:
            self._profile_canvas.clear()

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

        self._current_is_linear = key_left.startswith("linear_")
        raw_left = _get_array(self._npz, key_left)
        if raw_left is None:
            return

        arr_left = _prepare_for_display(raw_left)

        old_shape = self._current_shape
        new_shape = arr_left.shape[:2]
        keep_line = (old_shape is not None and old_shape == new_shape)

        self._current_arr_a = raw_left.astype(np.float32)
        if self._current_arr_a.max() > 1.5:
            self._current_arr_a /= 255.0
        self._current_shape = new_shape

        if self._single_image:
            self._current_arr_b = None
            self._image_canvas.set_labels(left_label, "No Image B")
            self._image_canvas.load_entry(arr_left, None, keep_line=keep_line)
        else:
            raw_right = _get_array(self._npz, key_right)
            if raw_right is None:
                return
            arr_right = _prepare_for_display(raw_right)
            self._current_arr_b = raw_right.astype(np.float32)
            if self._current_arr_b.max() > 1.5:
                self._current_arr_b /= 255.0
            self._image_canvas.set_labels(left_label, right_label)
            self._image_canvas.load_entry(arr_left, arr_right, keep_line=keep_line)

        if not keep_line:
            self._profile_canvas.clear()
