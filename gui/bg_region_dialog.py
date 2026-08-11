"""Background exclusion regions — draw areas to keep out of the background fit.

Section 3g detects sources automatically, but two situations are genuinely
unresolvable from pixel values alone, however the thresholds are tuned:

  * smooth extended nebulosity is degenerate with a sky gradient — a nebula edge
    running parallel to the gradient looks exactly like a steeper gradient;
  * a centred nebula is degenerate with vignetting — both are radially
    symmetric with the same curvature sign.

Only the observer knows which is which, so this window lets them say. Regions
are also useful for anything no detector targets: satellite trails, dust motes,
a bright galaxy the user would rather exclude by hand.

The preview shows what the algorithm will actually do *before* committing, so
drawing is informed rather than blind.

All array maths lives in analysis/ (`exclusion_mask`, `source_masked_background`)
— gui/ has no test coverage by design, so anything computed here would be
unverifiable.
"""
from __future__ import annotations

import os

# Pin the Qt binding before pyqtgraph probes for one: a frozen build that sees
# another binding on the path can otherwise bind to the wrong Qt.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget,
                             QMainWindow, QMessageBox, QPushButton, QSplitter,
                             QVBoxLayout, QWidget)

from analysis.inspector_regions import exclusion_mask
from core.models import BACKGROUND_DISPLAY_MAX_DIM
from gui.inspector_widgets import LinkedImageView, theme_colors

TOOL_PAN = "pan"
TOOL_DRAW = "draw"

_HINTS = {
    TOOL_PAN: "Drag to pan, scroll to zoom.",
    TOOL_DRAW: "Drag to draw an exclusion region around structure to keep out "
               "of the background fit.",
}


class _PreviewThread(QThread):
    """Runs the Section 3g estimate off the GUI thread.

    Pure array work from analysis/; the request carries its own copies so a
    control changing mid-run cannot alter what this run is computing, and
    `token` is echoed back so a superseded result can be dropped.
    """

    done = pyqtSignal(object)

    def __init__(self, req: dict, parent=None):
        super().__init__(parent)
        self._req = req

    def run(self) -> None:
        from analysis.source_mask import source_masked_background
        r = self._req
        try:
            image = r["image"]
            user = exclusion_mask(image.data.shape[:2], r["regions"])
            result = source_masked_background(
                image.data,
                box_size=int(image.background.box_size[0]),
                fwhm_px=r["fwhm_px"],
                user_exclusion=user if user.any() else None,
                mesh=image.background.background_mesh,
                rms_mesh=image.background.background_rms_mesh)
            self.done.emit({"token": r["token"], "result": result, "user": user})
        except Exception as exc:                      # noqa: BLE001 - surfaced in the UI
            self.done.emit({"token": r["token"], "error": str(exc)})


class BackgroundRegionDialog(QMainWindow):
    """Draw, list and preview background exclusion regions."""

    regions_changed = pyqtSignal(object)     # list of normalised region dicts

    def __init__(self, image_a, image_b=None, regions=None,
                 dark_mode: bool = False, fwhm_px: float = 4.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Background Exclusion Regions")
        self.resize(1180, 760)

        self._images = [(img, lbl) for img, lbl in
                        ((image_a, getattr(image_a, "label", "Image A")),
                         (image_b, getattr(image_b, "label", "Image B")))
                        if img is not None]
        self._regions: list[dict] = [dict(r) for r in (regions or [])]
        self._dark = dark_mode
        self._fwhm_px = float(fwhm_px)
        self._tool = TOOL_PAN
        self._token = 0
        self._thread: _PreviewThread | None = None

        self._build_ui()
        self._refresh_list()
        self._apply_tool()
        self._show_images()
        self._request_preview()

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        tools = QHBoxLayout()
        tools.addWidget(QLabel("Tool:"))
        self._tool_combo = QComboBox()
        self._tool_combo.addItem("Pan / zoom", TOOL_PAN)
        self._tool_combo.addItem("Draw exclusion region", TOOL_DRAW)
        self._tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        tools.addWidget(self._tool_combo)
        self._hint = QLabel(_HINTS[TOOL_PAN])
        self._hint.setStyleSheet("color: #888;")
        tools.addWidget(self._hint)
        tools.addStretch(1)
        root.addLayout(tools)

        split = QSplitter(Qt.Orientation.Horizontal)
        panels = QWidget()
        panel_row = QHBoxLayout(panels)
        panel_row.setContentsMargins(0, 0, 0, 0)
        self._views: list[LinkedImageView] = []
        for _img, label in self._images:
            view = LinkedImageView(label, dark=self._dark)
            view.polygon_drawn.connect(self._on_polygon_drawn)
            # The line and ROI tools of LinkedImageView are unused here; lock
            # them so a click can only ever mean "draw" or "pan".
            view.set_line_locked(True)
            view.set_roi_locked(True)
            panel_row.addWidget(view)
            self._views.append(view)
        split.addWidget(panels)

        side = QWidget()
        side_col = QVBoxLayout(side)
        side_col.addWidget(QLabel("Exclusion regions"))
        self._list = QListWidget()
        side_col.addWidget(self._list)

        btns = QHBoxLayout()
        self._del_btn = QPushButton("Delete selected")
        self._del_btn.clicked.connect(self._delete_selected)
        btns.addWidget(self._del_btn)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.clicked.connect(self._clear_all)
        btns.addWidget(self._clear_btn)
        side_col.addLayout(btns)

        self._stats = QLabel("")
        self._stats.setWordWrap(True)
        side_col.addWidget(self._stats)
        side_col.addStretch(1)

        done = QPushButton("Use these regions")
        done.clicked.connect(self._accept)
        side_col.addWidget(done)
        split.addWidget(side)
        split.setSizes([880, 300])
        root.addWidget(split, 1)

        self.setCentralWidget(central)

        # Debounced: the estimate costs ~0.2 s at 512 px and ~13 s at 24 MP, far
        # too slow to run on every stroke. Drawn regions repaint instantly; only
        # the detector re-run waits.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start_preview)

    # -- display ----------------------------------------------------------

    def _show_images(self) -> None:
        for view, (img, _label) in zip(self._views, self._images):
            disp = img.display_image(stretch=True)
            step = max(1, max(disp.shape[:2]) // BACKGROUND_DISPLAY_MAX_DIM)
            view.set_image(disp[::step, ::step].astype(np.float32), "gray", None)
        self._repaint_regions()

    def _repaint_regions(self) -> None:
        """Rasterise the drawn regions onto each panel — cheap, so not threaded."""
        for view in self._views:
            shape = view.shape
            if shape is None:
                continue
            mask = exclusion_mask(shape, self._regions)
            view.set_overlay("exclusion", mask if mask.any() else None)

    def _refresh_list(self) -> None:
        self._list.clear()
        for i, region in enumerate(self._regions, start=1):
            n = len(region.get("points") or [])
            self._list.addItem(f"Region {i} — {n} points")
        self._del_btn.setEnabled(bool(self._regions))
        self._clear_btn.setEnabled(bool(self._regions))

    # -- tools ------------------------------------------------------------

    def _on_tool_changed(self, *_a) -> None:
        self._tool = self._tool_combo.currentData()
        self._apply_tool()

    def _apply_tool(self) -> None:
        for view in self._views:
            view.set_lasso_locked(self._tool != TOOL_DRAW)
        self._hint.setText(_HINTS.get(self._tool, ""))

    # -- editing ----------------------------------------------------------

    def _on_polygon_drawn(self, region: dict) -> None:
        if not region or len(region.get("points") or []) < 3:
            return
        self._regions.append(region)
        self._refresh_list()
        self._repaint_regions()
        self._request_preview()

    def _delete_selected(self) -> None:
        row = self._list.currentRow()
        if 0 <= row < len(self._regions):
            del self._regions[row]
            self._refresh_list()
            self._repaint_regions()
            self._request_preview()

    def _clear_all(self) -> None:
        if not self._regions:
            return
        self._regions = []
        self._refresh_list()
        self._repaint_regions()
        self._request_preview()

    # -- preview ----------------------------------------------------------

    def _request_preview(self) -> None:
        self._stats.setText("Updating preview…")
        self._timer.start(300)

    def _start_preview(self) -> None:
        if not self._images:
            return
        image = self._images[0][0]
        if getattr(image, "background", None) is None:
            self._stats.setText("Background not estimated for this image.")
            return
        self._token += 1
        req = {"token": self._token, "image": image,
               "regions": [dict(r) for r in self._regions],
               "fwhm_px": self._fwhm_px}
        old = self._thread
        if old is not None and old.isRunning():
            # Supersede rather than wait: never block the GUI thread.
            try:
                old.done.disconnect()
            except TypeError:
                pass
            old.quit()
        self._thread = _PreviewThread(req, parent=self)
        self._thread.done.connect(self._on_preview_ready)
        self._thread.start()

    def _on_preview_ready(self, payload: dict) -> None:
        if payload.get("token") != self._token:
            return                       # superseded by a newer request
        if payload.get("error"):
            self._stats.setText(f"Preview failed: {payload['error']}")
            return
        result = payload["result"]
        if not result.ok:
            self._stats.setText(
                f"Estimate unavailable — {result.fallback_reason}")
            self._set_detector_overlay(None)
            return

        sm = result.source_mask
        # Show what the detector found on its own, so the drawn regions can be
        # judged against it rather than in isolation.
        auto = sm.point_mask | sm.extended_mask
        self._set_detector_overlay(auto)
        order = {0: "constant", 1: "plane", 2: "quadric"}.get(
            result.fit.get("selected_order"), "—")
        self._stats.setText(
            f"<b>Preview — {self._images[0][1]}</b><br>"
            f"Masked: {result.coverage:.0%} of frame<br>"
            f"&nbsp;&nbsp;detector: {auto.mean():.0%}<br>"
            f"&nbsp;&nbsp;your regions: {sm.user_coverage:.0%}<br>"
            f"Mesh cells used: {result.n_cells} / {result.n_cells_total}<br>"
            f"Fitted model: {order}<br>"
            f"Background: {result.background_median:.6g} ADU")

    def _set_detector_overlay(self, mask) -> None:
        for view in self._views:
            shape = view.shape
            if shape is None or mask is None:
                view.set_overlay("mask", None)
                continue
            h, w = shape
            step = max(1, max(mask.shape[:2]) // max(h, w))
            small = mask[::step, ::step][:h, :w]
            if small.shape != (h, w):
                padded = np.zeros((h, w), dtype=bool)
                padded[:small.shape[0], :small.shape[1]] = small
                small = padded
            view.set_overlay("mask", small)

    # -- result -----------------------------------------------------------

    @property
    def regions(self) -> list[dict]:
        return [dict(r) for r in self._regions]

    def _accept(self) -> None:
        self.regions_changed.emit(self.regions)
        self.close()

    def closeEvent(self, event):
        thread = self._thread
        if thread is not None and thread.isRunning():
            try:
                thread.done.disconnect()
            except TypeError:
                pass
            thread.quit()
        super().closeEvent(event)
