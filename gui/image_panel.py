from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, QRectF, QPointF, QSettings, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QGroupBox, QFormLayout, QLineEdit, QFileDialog, QScrollArea,
    QSizePolicy, QRubberBand,
)

from core.astro_image import AstroImage

MAX_DISPLAY_PX = 1024   # max dimension for on-screen display (downsampled for speed)


def _downsample_for_display(arr: np.ndarray) -> np.ndarray:
    """Stride-sample arr so its longest dimension is ≤ MAX_DISPLAY_PX."""
    max_dim = max(arr.shape[:2])
    if max_dim <= MAX_DISPLAY_PX:
        return arr
    step = max_dim // MAX_DISPLAY_PX + 1
    return arr[::step, ::step]


class ZoomableImageLabel(QLabel):
    """QLabel with scroll-wheel zoom, right-click pan, ROI and cross-section line tools."""

    roi_selected  = pyqtSignal(int, int, int, int)       # x0, y0, x1, y1 in full-res image coords
    line_selected = pyqtSignal(float, float, float, float)  # x0n, y0n, x1n, y1n normalised [0,1]
    view_changed  = pyqtSignal(float, float, float)      # zoom, pan_nx, pan_ny

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin = QPoint()
        self._pixmap_orig: QPixmap | None = None
        self._display_arr: np.ndarray | None = None   # keeps buffer alive for QImage
        self._full_image_shape: tuple | None = None   # (H, W) of the original full-res image
        self._roi_mode = False
        self._roi_norm: tuple | None = None  # (x0n, y0n, x1n, y1n) normalised [0,1] persistent overlay
        self._line_mode: bool = False
        self._line_state: str = "idle"  # "idle" | "drawing" | "fixed"
        self._line_n0: tuple | None = None  # (xn, yn) normalised start
        self._line_n1: tuple | None = None  # (xn, yn) normalised end
        # Zoom / pan state (1.0 = fit-to-widget; pan in widget pixels relative to centred position)
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._pan_dragging: bool = False
        self._pan_last_pos: QPoint = QPoint()

    # ------------------------------------------------------------------
    # Zoom / pan helpers
    # ------------------------------------------------------------------

    def _fit_size(self) -> tuple[float, float] | None:
        """Return (pw, ph) — pixmap size when scaled to fit widget at zoom=1, aspect-locked."""
        if self._pixmap_orig is None:
            return None
        pw0, ph0 = float(self._pixmap_orig.width()), float(self._pixmap_orig.height())
        if pw0 == 0 or ph0 == 0:
            return None
        scale = min(self.width() / pw0, self.height() / ph0)
        return pw0 * scale, ph0 * scale

    def _view_rect(self) -> tuple[float, float, float, float] | None:
        """Return (ox, oy, pw, ph) — image top-left and size in widget pixels."""
        fs = self._fit_size()
        if fs is None:
            return None
        pw_fit, ph_fit = fs
        pw = pw_fit * self._zoom
        ph = ph_fit * self._zoom
        ox = (self.width()  - pw) / 2 + self._pan_x
        oy = (self.height() - ph) / 2 + self._pan_y
        return ox, oy, pw, ph

    def _emit_view_changed(self) -> None:
        fs = self._fit_size()
        if fs is None:
            return
        pw_fit, ph_fit = fs
        pan_nx = self._pan_x / pw_fit if pw_fit > 0 else 0.0
        pan_ny = self._pan_y / ph_fit if ph_fit > 0 else 0.0
        self.view_changed.emit(self._zoom, pan_nx, pan_ny)

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()
        self._emit_view_changed()

    def apply_view(self, zoom: float, pan_nx: float, pan_ny: float) -> None:
        """Apply synchronized view from the partner panel (does not re-emit)."""
        self._zoom = zoom
        fs = self._fit_size()
        if fs:
            self._pan_x = pan_nx * fs[0]
            self._pan_y = pan_ny * fs[1]
        self.update()

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def clear_roi_overlay(self) -> None:
        self._roi_norm = None
        self.update()

    def set_roi_mode(self, enabled: bool) -> None:
        self._roi_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if (enabled or self._line_mode)
                       else Qt.CursorShape.ArrowCursor)

    def set_line_mode(self, enabled: bool) -> None:
        self._line_mode = enabled
        if not enabled and self._line_state == "drawing":
            self._line_state = "idle"
            self._line_n0 = None
            self._line_n1 = None
        self.setCursor(Qt.CursorShape.CrossCursor if (enabled or self._roi_mode)
                       else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_line_normalised(self, x0n: float, y0n: float,
                             x1n: float, y1n: float) -> None:
        self._line_n0 = (x0n, y0n)
        self._line_n1 = (x1n, y1n)
        self._line_state = "fixed"
        self.update()

    def _widget_to_norm(self, pt: QPoint) -> tuple[float, float] | None:
        r = self._view_rect()
        if r is None:
            return None
        ox, oy, pw, ph = r
        if pw == 0 or ph == 0:
            return None
        xn = (pt.x() - ox) / pw
        yn = (pt.y() - oy) / ph
        return float(max(0.0, min(1.0, xn))), float(max(0.0, min(1.0, yn)))

    def _norm_to_widget(self, xn: float, yn: float) -> QPoint | None:
        r = self._view_rect()
        if r is None:
            return None
        ox, oy, pw, ph = r
        return QPoint(int(ox + xn * pw), int(oy + yn * ph))

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def set_image_array(self, arr: np.ndarray) -> None:
        self._full_image_shape = arr.shape[:2]
        self._roi_norm = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        arr = _downsample_for_display(arr)
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            self._display_arr = np.ascontiguousarray(arr.astype(np.uint8))
            qimg = QImage(self._display_arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            self._display_arr = np.ascontiguousarray(arr.astype(np.uint8))
            qimg = QImage(self._display_arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qimg)
        self.update()

    def _update_display(self) -> None:
        self.update()

    # ------------------------------------------------------------------
    # Qt events — rendering
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        r = self._view_rect()
        if r is not None and self._pixmap_orig is not None:
            ox, oy, pw, ph = r
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            src = QRectF(0, 0, self._pixmap_orig.width(), self._pixmap_orig.height())
            p.drawPixmap(QRectF(ox, oy, pw, ph), self._pixmap_orig, src)
            # ROI overlay
            if self._roi_norm:
                x0n, y0n, x1n, y1n = self._roi_norm
                p.setPen(QPen(QColor("#17becf"), 2))
                p.setBrush(QColor(23, 190, 207, 35))
                p.drawRect(QRectF(ox + x0n * pw, oy + y0n * ph,
                                  (x1n - x0n) * pw, (y1n - y0n) * ph))
            # Line overlay
            if self._line_n0 and self._line_n1 and self._line_state in ("drawing", "fixed"):
                p.setPen(QPen(QColor("#17becf"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(QPointF(ox + self._line_n0[0] * pw, oy + self._line_n0[1] * ph),
                           QPointF(ox + self._line_n1[0] * pw, oy + self._line_n1[1] * ph))
        p.end()

    # ------------------------------------------------------------------
    # Qt events — zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        if self._pixmap_orig is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else (1.0 / 1.25)
        new_zoom = max(0.1, min(50.0, self._zoom * factor))
        fs = self._fit_size()
        if fs is None:
            return
        r = self._view_rect()
        if r is None:
            return
        ox, oy, pw, ph = r
        pw_fit, ph_fit = fs
        cx = float(event.position().x())
        cy = float(event.position().y())
        xn = (cx - ox) / pw
        yn = (cy - oy) / ph
        pw_new = pw_fit * new_zoom
        ph_new = ph_fit * new_zoom
        self._pan_x = cx - xn * pw_new - (self.width()  - pw_new) / 2
        self._pan_y = cy - yn * ph_new - (self.height() - ph_new) / 2
        self._zoom = new_zoom
        self.update()
        self._emit_view_changed()

    # ------------------------------------------------------------------
    # Qt events — mouse (tool interactions + pan)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._pan_dragging = True
            self._pan_last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._line_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._line_state in ("idle", "fixed"):
                coords = self._widget_to_norm(event.pos())
                if coords:
                    self._line_n0 = coords
                    self._line_n1 = coords
                self._line_state = "drawing"
            elif self._line_state == "drawing":
                coords = self._widget_to_norm(event.pos())
                if coords:
                    self._line_n1 = coords
                self._line_state = "fixed"
                if self._line_n0 and self._line_n1:
                    self.line_selected.emit(
                        self._line_n0[0], self._line_n0[1],
                        self._line_n1[0], self._line_n1[1])
            self.update()
            return
        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._rubber_band.setGeometry(QRect(self._origin, self._origin))
            self._rubber_band.show()

    def mouseMoveEvent(self, event):
        if self._pan_dragging:
            delta = event.pos() - self._pan_last_pos
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._pan_last_pos = event.pos()
            self.update()
            self._emit_view_changed()
            return
        if self._line_mode and self._line_state == "drawing":
            coords = self._widget_to_norm(event.pos())
            if coords:
                self._line_n1 = coords
            self.update()
            return
        if self._roi_mode and not self._origin.isNull():
            self._rubber_band.setGeometry(
                QRect(self._origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._pan_dragging:
            self._pan_dragging = False
            self.setCursor(Qt.CursorShape.CrossCursor if (self._roi_mode or self._line_mode)
                           else Qt.CursorShape.ArrowCursor)
            return
        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            self._rubber_band.hide()
            rect = QRect(self._origin, event.pos()).normalized()
            img_rect = self._image_coords(rect)
            if img_rect:
                self.roi_selected.emit(*img_rect)
                r = self._view_rect()
                if r is not None:
                    ox, oy, pw, ph = r
                    self._roi_norm = (
                        max(0.0, min(1.0, (rect.left()   - ox) / pw)),
                        max(0.0, min(1.0, (rect.top()    - oy) / ph)),
                        max(0.0, min(1.0, (rect.right()  - ox) / pw)),
                        max(0.0, min(1.0, (rect.bottom() - oy) / ph)),
                    )
                self.update()
            self._origin = QPoint()

    def _image_coords(self, widget_rect: QRect) -> tuple[int, int, int, int] | None:
        if self._pixmap_orig is None:
            return None
        r = self._view_rect()
        if r is None:
            return None
        ox, oy, pw, ph = r
        if self._full_image_shape is not None:
            orig_h, orig_w = self._full_image_shape
        else:
            orig_w = float(self._pixmap_orig.width())
            orig_h = float(self._pixmap_orig.height())
        scale_x = orig_w / pw if pw > 0 else 1.0
        scale_y = orig_h / ph if ph > 0 else 1.0
        x0 = int((widget_rect.left()   - ox) * scale_x)
        y0 = int((widget_rect.top()    - oy) * scale_y)
        x1 = int((widget_rect.right()  - ox) * scale_x)
        y1 = int((widget_rect.bottom() - oy) * scale_y)
        x0 = max(0, min(x0, int(orig_w)))
        y0 = max(0, min(y0, int(orig_h)))
        x1 = max(0, min(x1, int(orig_w)))
        y1 = max(0, min(y1, int(orig_h)))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)


class ImagePanel(QWidget):
    """Left or right image panel: file loading, display, metadata, bandwidth."""

    image_loaded = pyqtSignal(object)   # emits AstroImage
    roi_selected = pyqtSignal(int, int, int, int)
    line_selected = pyqtSignal(float, float, float, float)
    view_changed  = pyqtSignal(float, float, float)  # zoom, pan_nx, pan_ny

    def __init__(self, title: str = "Image", parent=None):
        super().__init__(parent)
        self._image: AstroImage | None = None
        self._starless_image: AstroImage | None = None
        self._build_ui(title)

    def _build_ui(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title + stretch checkbox + open button
        top = QHBoxLayout()
        self._header_layout = top   # exposed for external widget insertion
        top.addWidget(QLabel(f"<b>{title}</b>"))
        top.addStretch()
        self._stretch_cb = QCheckBox("Stretch")
        self._stretch_cb.setChecked(True)
        self._stretch_cb.setToolTip("Auto-stretch display (statistical MTF stretch).\n"
                                    "Uncheck for linear 0.1–99.9% percentile view.")
        self._stretch_cb.toggled.connect(self._refresh_display)
        top.addWidget(self._stretch_cb)
        self._btn_open = QPushButton("Open FITS / XISF…")
        self._btn_open.clicked.connect(self._open_file)
        top.addWidget(self._btn_open)
        layout.addLayout(top)

        # Image display
        self._img_label = ZoomableImageLabel()
        self._img_label.roi_selected.connect(self.roi_selected)
        self._img_label.line_selected.connect(self.line_selected)
        self._img_label.view_changed.connect(self.view_changed)
        layout.addWidget(self._img_label, stretch=1)

        # Metadata group — two side-by-side columns to reduce height
        meta_box = QGroupBox("Image info")
        meta_outer = QVBoxLayout(meta_box)
        meta_outer.setContentsMargins(4, 4, 4, 4)

        cols = QHBoxLayout()
        left_form = QFormLayout()
        left_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        right_form = QFormLayout()
        right_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._meta_fields: dict[str, QLabel] = {}
        for key in ["File", "Telescope", "Camera", "Filter", "Exposure"]:
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            left_form.addRow(f"{key}:", lbl)
            self._meta_fields[key] = lbl
        for key in ["Bit depth", "Gain", "Date", "Pixel scale", "Binning"]:
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            right_form.addRow(f"{key}:", lbl)
            self._meta_fields[key] = lbl

        cols.addLayout(left_form)
        cols.addLayout(right_form)
        meta_outer.addLayout(cols)

        # Bandwidth field (editable) — full width below the two columns
        self._bw_edit = QLineEdit()
        self._bw_edit.setPlaceholderText("e.g. 3")
        self._bw_edit.setMaximumWidth(80)
        bw_row = QHBoxLayout()
        bw_row.addWidget(QLabel("Bandwidth:"))
        bw_row.addWidget(self._bw_edit)
        bw_row.addWidget(QLabel("nm"))
        bw_row.addStretch()
        meta_outer.addLayout(bw_row)

        # Filter thickness field (editable)
        self._thickness_edit = QLineEdit()
        self._thickness_edit.setText("1")
        self._thickness_edit.setMaximumWidth(80)
        thickness_row = QHBoxLayout()
        thickness_row.addWidget(QLabel("Filter thickness:"))
        thickness_row.addWidget(self._thickness_edit)
        thickness_row.addWidget(QLabel("mm"))
        thickness_row.addStretch()
        meta_outer.addLayout(thickness_row)

        # Starless filename label
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("Starless:"))
        self._starless_lbl = QLabel("—")
        self._starless_lbl.setStyleSheet("color: #666;")
        sl_row.addWidget(self._starless_lbl)
        sl_row.addStretch()
        meta_outer.addLayout(sl_row)

        layout.addWidget(meta_box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def image(self) -> AstroImage | None:
        return self._image

    @property
    def starless_image(self) -> AstroImage | None:
        return self._starless_image

    def set_roi_mode(self, enabled: bool) -> None:
        self._img_label.set_roi_mode(enabled)

    def clear_roi_overlay(self) -> None:
        self._img_label.clear_roi_overlay()

    def set_line_mode(self, enabled: bool) -> None:
        self._img_label.set_line_mode(enabled)

    def reset_zoom(self) -> None:
        self._img_label.reset_zoom()

    def apply_view(self, zoom: float, pan_nx: float, pan_ny: float) -> None:
        self._img_label.apply_view(zoom, pan_nx, pan_ny)

    def _open_file(self) -> None:
        settings = QSettings("FilterImageComparator", "FilterImageComparator")
        last_dir = settings.value("last_data_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image",
            last_dir,
            "Astronomical images (*.fits *.fit *.fts *.xisf *.tiff *.tif);;All files (*.*)",
        )
        if not path:
            return
        settings.setValue("last_data_dir", str(Path(path).parent))
        img = AstroImage(path)
        try:
            img.load()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load error", str(e))
            return

        self._image = img
        self._starless_image = None
        self._starless_lbl.setText("—")
        self._starless_lbl.setStyleSheet("color: #666;")
        self._populate_metadata(img)
        self._refresh_display()
        self.image_loaded.emit(img)
        self._ask_about_starless(img)

    def load_path(self, path: str) -> None:
        """Load an image directly from a file path (no file dialog, no starless prompt)."""
        img = AstroImage(path)
        try:
            img.load()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load error", str(e))
            return
        self._image = img
        self._starless_image = None
        self._starless_lbl.setText("—")
        self._starless_lbl.setStyleSheet("color: #666;")
        self._populate_metadata(img)
        self._refresh_display()
        self.image_loaded.emit(img)

    def set_starless_path(self, path: str) -> None:
        """Attach a pre-generated starless image to the currently loaded main image."""
        if not path or self._image is None:
            return
        sl = AstroImage(path)
        try:
            sl.load()
        except Exception:
            return
        self._starless_image = sl
        self._image.starless_image = sl      # mirrors _ask_about_starless behaviour
        self._starless_lbl.setText(Path(path).name)
        self._starless_lbl.setStyleSheet("color: #155724;")

    def _ask_about_starless(self, img: AstroImage) -> None:
        from PyQt6.QtWidgets import QMessageBox
        prefix = ""
        if img.is_color:
            prefix = ("This is a color (RGB) image. It has been converted to "
                      "luminance for analysis.\n\n")
        answer = QMessageBox.question(
            self, "Starless image",
            prefix + "Do you have a starless version of this image?\n\n"
            "If yes, it will be used for power spectrum and spatial detail "
            "analysis to reduce star contamination.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        settings = QSettings("FilterImageComparator", "FilterImageComparator")
        last_dir = settings.value("last_data_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open starless image", last_dir,
            "Astronomical images (*.fits *.fit *.fts *.xisf *.tiff *.tif);;All files (*.*)",
        )
        if not path:
            return
        sl = AstroImage(path)
        try:
            sl.load()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox as _MB
            _MB.critical(self, "Load error", f"Starless load failed:\n{e}")
            return
        self._starless_image = sl
        img.starless_image = sl
        self._starless_lbl.setText(sl.path.name)
        self._starless_lbl.setStyleSheet("color: #155724;")

    def apply_bandwidth_from_field(self) -> None:
        """Push the bandwidth QLineEdit value into the AstroImage."""
        if self._image is None:
            return
        txt = self._bw_edit.text().strip()
        if txt:
            try:
                self._image.bandwidth_nm = float(txt)
            except ValueError:
                pass

    def apply_filter_thickness_from_field(self) -> None:
        """Push the filter thickness QLineEdit value into the AstroImage."""
        if self._image is None:
            return
        txt = self._thickness_edit.text().strip()
        if txt:
            try:
                val = float(txt)
                if val > 0:
                    self._image.filter_thickness_mm = val
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _populate_metadata(self, img: AstroImage) -> None:
        fname = Path(img.path).name
        if img.is_color:
            fname += " (color→lum)"
        self._meta_fields["File"].setText(fname)
        for key in ["Bit depth", "Telescope", "Camera", "Filter", "Exposure",
                    "Gain", "Date", "Binning"]:
            val = img.meta.get(key, "—")
            self._meta_fields[key].setText(val)

        scale_txt = f"{img.pixel_scale:.3f} \"/px"
        if img.pixel_scale_is_estimated:
            scale_txt += " (estimated)"
        self._meta_fields["Pixel scale"].setText(scale_txt)

        if img.bandwidth_nm is not None:
            self._bw_edit.setText(str(img.bandwidth_nm))

    def _refresh_display(self) -> None:
        if self._image is None:
            return
        try:
            stretch = self._stretch_cb.isChecked()
            display = self._image.display_image(stretch=stretch)
            self._img_label.set_image_array(display)
        except Exception as exc:
            self._img_label.setText(f"Display error:\n{exc}")
