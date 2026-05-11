from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, rotate
from scipy.interpolate import interp1d

from core.astro_image import AstroImage
from core.fig_utils import figs_to_b64
from core.models import EDGE_ROI_HALF_WIDTH

EDGE_DISPLAY_HALF_WIDTH = 250   # half-side of the context window shown in the report figure


class EdgeAnalyzer:
    """Extract edge spread function (ESF) and line spread function (LSF)
    from a nebula edge to measure local contrast and resolution."""

    def analyze(self, image: AstroImage,
                roi: tuple[int, int, int, int] | None = None) -> dict:
        image.estimate_background()
        bgsub = image.background_subtracted()

        result: dict = {
            "edge_width_10_90_px": None,
            "edge_width_10_90_arcsec": None,
            "gradient_magnitude": None,
            "edge_contrast_ratio": None,
            "esf": None,
            "lsf": None,
        }

        # Select region of interest
        if roi is not None:
            x0, y0, x1, y1 = roi
            roi_data = bgsub[y0:y1, x0:x1]
        else:
            roi_data, roi = self._auto_detect_roi(bgsub, image)

        result["roi_used"] = roi   # expose for caller to reuse across both images

        if roi_data is None or roi_data.size == 0:
            return result

        edge_info = self._detect_strongest_edge(roi_data)
        if edge_info is None:
            return result

        result["gradient_magnitude"] = edge_info["gradient_magnitude"]

        positions, esf = self._extract_esf(roi_data, edge_info)
        if esf is None or len(esf) < 5:
            return result

        lsf = self._compute_lsf(positions, esf)
        width = self._measure_edge_width(positions, esf)

        result["esf"] = esf
        result["lsf"] = lsf
        result["edge_width_10_90_px"] = width
        if width is not None:
            result["edge_width_10_90_arcsec"] = width * image.pixel_scale

        # Edge contrast ratio (bandwidth-sensitive — flagged in report)
        ecr = self._measure_edge_contrast_ratio(roi_data, edge_info)
        result["edge_contrast_ratio"] = ecr

        # Build 300×300 display context centred on the gradient peak
        x0, y0, x1, y1 = roi
        xc_full = x0 + edge_info["center_x"]
        yc_full = y0 + edge_info["center_y"]
        dw = EDGE_DISPLAY_HALF_WIDTH
        dx0 = max(0, xc_full - dw)
        dy0 = max(0, yc_full - dw)
        dx1 = min(bgsub.shape[1], xc_full + dw)
        dy1 = min(bgsub.shape[0], yc_full + dw)
        display_roi = bgsub[dy0:dy1, dx0:dx1]
        analysis_rect = (x0 - dx0, y0 - dy0, x1 - dx0, y1 - dy0)
        edge_info_display = dict(edge_info)
        edge_info_display["center_x"] = xc_full - dx0
        edge_info_display["center_y"] = yc_full - dy0

        result["figures"] = figs_to_b64({
            "edge": self._plot_results(roi_data, display_roi, analysis_rect,
                                       positions, esf, lsf, width, image.label,
                                       edge_info_display)
        })

        return result

    # ------------------------------------------------------------------
    # ROI auto-detection
    # ------------------------------------------------------------------

    def _auto_detect_roi(self, bgsub: np.ndarray, image: AstroImage
                          ) -> tuple[np.ndarray | None, tuple | None]:
        """Find a patch centred on the strongest gradient in the image."""
        from core.stretch import stf_stretch
        # STF stretch amplifies faint nebula edges before gradient detection;
        # bgsub (linear) is still returned as roi_data for accurate ESF measurement.
        stretched = stf_stretch(bgsub).astype(np.float64)
        sx = sobel(stretched, axis=1)
        sy = sobel(stretched, axis=0)
        gm = np.sqrt(sx**2 + sy**2)

        # Avoid borders
        margin = EDGE_ROI_HALF_WIDTH + 5
        gm[:margin, :] = 0
        gm[-margin:, :] = 0
        gm[:, :margin] = 0
        gm[:, -margin:] = 0

        peak_idx = np.unravel_index(np.argmax(gm), gm.shape)
        yc, xc = peak_idx
        hw = EDGE_ROI_HALF_WIDTH
        h, w = bgsub.shape
        x0 = max(0, xc - hw)
        y0 = max(0, yc - hw)
        x1 = min(w, xc + hw)
        y1 = min(h, yc + hw)
        roi = (x0, y0, x1, y1)
        return bgsub[y0:y1, x0:x1], roi

    # ------------------------------------------------------------------
    # Edge detection within ROI
    # ------------------------------------------------------------------

    def _detect_strongest_edge(self, roi_data: np.ndarray) -> dict | None:
        sx = sobel(roi_data, axis=1).astype(float)
        sy = sobel(roi_data, axis=0).astype(float)
        gm = np.sqrt(sx**2 + sy**2)

        peak_idx = np.unravel_index(np.argmax(gm), gm.shape)
        yc, xc = peak_idx
        angle_rad = float(np.arctan2(sy[yc, xc], sx[yc, xc]))

        return {
            "center_x": xc,
            "center_y": yc,
            "angle_rad": angle_rad,
            "gradient_magnitude": float(gm[yc, xc]),
        }

    # ------------------------------------------------------------------
    # ESF extraction
    # ------------------------------------------------------------------

    def _extract_esf(self, roi_data: np.ndarray,
                      edge_info: dict) -> tuple[np.ndarray, np.ndarray | None]:
        angle_deg = np.degrees(edge_info["angle_rad"])
        # Rotate so edge normal points horizontally (edge runs vertically)
        rotation_angle = -(90.0 - angle_deg)
        rotated = rotate(roi_data, rotation_angle, reshape=False, order=3)

        # Column means across all rows gives the ESF
        esf_raw = np.mean(rotated, axis=0)
        positions = np.arange(len(esf_raw), dtype=float)

        # Normalise to [0, 1]
        lo, hi = esf_raw.min(), esf_raw.max()
        if hi - lo < 1e-12:
            return positions, None
        esf = (esf_raw - lo) / (hi - lo)

        # Ensure ESF goes low→high (flip if descending)
        if esf[0] > esf[-1]:
            esf = 1.0 - esf

        return positions, esf

    # ------------------------------------------------------------------
    # LSF and width
    # ------------------------------------------------------------------

    def _compute_lsf(self, positions: np.ndarray,
                      esf: np.ndarray) -> np.ndarray:
        return np.gradient(esf, positions)

    def _measure_edge_width(self, positions: np.ndarray,
                             esf: np.ndarray) -> float | None:
        try:
            interp = interp1d(esf, positions, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
            p10 = float(interp(0.10))
            p90 = float(interp(0.90))
            width = abs(p90 - p10)
            return width if np.isfinite(width) else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Edge contrast ratio (bandwidth-sensitive)
    # ------------------------------------------------------------------

    def _measure_edge_contrast_ratio(self, roi_data: np.ndarray,
                                      edge_info: dict) -> float | None:
        xc = edge_info["center_x"]
        half = max(5, roi_data.shape[1] // 4)
        bright_side = roi_data[:, max(0, xc - half):xc]
        dark_side = roi_data[:, xc:min(roi_data.shape[1], xc + half)]

        bright_mean = float(np.mean(bright_side)) if bright_side.size > 0 else None
        dark_mean = float(np.mean(dark_side)) if dark_side.size > 0 else None

        if bright_mean is None or dark_mean is None:
            return None
        if bright_mean < dark_mean:
            bright_mean, dark_mean = dark_mean, bright_mean
        if dark_mean <= 0:
            return None
        return bright_mean / dark_mean

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _plot_results(self, roi_data: np.ndarray,
                       display_roi: np.ndarray,
                       analysis_rect: tuple,
                       positions: np.ndarray,
                       esf: np.ndarray,
                       lsf: np.ndarray,
                       width: float | None,
                       label: str,
                       edge_info: dict | None = None) -> plt.Figure:
        from matplotlib.patches import Rectangle

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # ROI context image (500×500 display region)
        h_disp, w_disp = display_roi.shape
        axes[0].imshow(display_roi, origin="lower", cmap="gray",
                       aspect="equal", interpolation="nearest",
                       extent=[0, w_disp, 0, h_disp])
        axes[0].set_title(f"Edge ROI — {label}")
        axes[0].set_xlabel("X (px)")
        axes[0].set_ylabel("Y (px)")

        # Dashed lime rectangle marking the actual 60×60 analysis region
        bx0, by0, bx1, by1 = analysis_rect
        rect = Rectangle(
            (bx0, by0), bx1 - bx0, by1 - by0,
            linewidth=1.5, edgecolor="lime", facecolor="none",
            linestyle="--", zorder=5, label="Analysis region",
        )
        axes[0].add_patch(rect)

        # Overlay lines showing the edge location and ESF scan direction
        if edge_info is not None:
            xc = edge_info["center_x"]
            yc = edge_info["center_y"]
            angle = edge_info["angle_rad"]
            h, w = display_roi.shape
            t = max(h, w)

            def _clipped_line(cx, cy, ang):
                """Return line endpoints clipped to the image bounds [0,w] x [0,h]."""
                x0 = np.clip(cx - t * np.cos(ang), 0, w_disp)
                x1 = np.clip(cx + t * np.cos(ang), 0, w_disp)
                y0 = np.clip(cy - t * np.sin(ang), 0, h_disp)
                y1 = np.clip(cy + t * np.sin(ang), 0, h_disp)
                return [x0, x1], [y0, y1]

            xs, ys = _clipped_line(xc, yc, angle)
            axes[0].plot(xs, ys, color="cyan", linewidth=1.5, alpha=0.85,
                         label="ESF scan direction")

            perp = angle + np.pi / 2
            xs, ys = _clipped_line(xc, yc, perp)
            axes[0].plot(xs, ys, color="yellow", linewidth=1.2, linestyle="--",
                         alpha=0.75, label="Edge orientation")

            axes[0].legend(fontsize=7, loc="lower right")

        # Pin axis to the image data — must come after all plot calls
        axes[0].set_xlim(0, w_disp)
        axes[0].set_ylim(0, h_disp)

        # ESF
        axes[1].plot(positions, esf, "b-", linewidth=1.5)
        axes[1].axhline(0.10, color="gray", linestyle="--", linewidth=0.8)
        axes[1].axhline(0.90, color="gray", linestyle="--", linewidth=0.8)
        if width is not None:
            axes[1].set_title(f"ESF — 10-90% width = {width:.2f} px")
        else:
            axes[1].set_title("ESF")
        axes[1].set_xlabel("Position (px)")
        axes[1].set_ylabel("Normalised intensity")
        axes[1].grid(True, alpha=0.3)

        # LSF
        axes[2].plot(positions, lsf, "r-", linewidth=1.5)
        axes[2].set_title("LSF (derivative of ESF)")
        axes[2].set_xlabel("Position (px)")
        axes[2].set_ylabel("d(ESF)/dx")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
