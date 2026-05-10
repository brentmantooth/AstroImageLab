from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, rotate
from scipy.interpolate import interp1d

from core.astro_image import AstroImage
from core.models import EDGE_ROI_HALF_WIDTH


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

        result["figures"] = {
            "edge": self._plot_results(roi_data, positions, esf, lsf, width,
                                       image.label, edge_info)
        }

        return result

    # ------------------------------------------------------------------
    # ROI auto-detection
    # ------------------------------------------------------------------

    def _auto_detect_roi(self, bgsub: np.ndarray, image: AstroImage
                          ) -> tuple[np.ndarray | None, tuple | None]:
        """Find a patch centred on the strongest gradient in the image."""
        sx = sobel(bgsub, axis=1)
        sy = sobel(bgsub, axis=0)
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
                       positions: np.ndarray,
                       esf: np.ndarray,
                       lsf: np.ndarray,
                       width: float | None,
                       label: str,
                       edge_info: dict | None = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # ROI image
        axes[0].imshow(roi_data, origin="lower", cmap="gray",
                       aspect="auto", interpolation="nearest")
        axes[0].set_title(f"Edge ROI — {label}")
        axes[0].set_xlabel("X (px)")
        axes[0].set_ylabel("Y (px)")

        # Overlay lines showing the edge location and ESF scan direction
        if edge_info is not None:
            xc = edge_info["center_x"]
            yc = edge_info["center_y"]
            angle = edge_info["angle_rad"]
            h, w = roi_data.shape
            t = max(h, w)

            # Cyan line: scan direction (perpendicular to edge, i.e. gradient direction)
            axes[0].plot(
                [xc - t * np.cos(angle), xc + t * np.cos(angle)],
                [yc - t * np.sin(angle), yc + t * np.sin(angle)],
                color="cyan", linewidth=1.5, alpha=0.85, label="ESF scan direction",
            )
            # Yellow dashed line: edge orientation (along the edge)
            perp = angle + np.pi / 2
            axes[0].plot(
                [xc - t * np.cos(perp), xc + t * np.cos(perp)],
                [yc - t * np.sin(perp), yc + t * np.sin(perp)],
                color="yellow", linewidth=1.2, linestyle="--", alpha=0.75,
                label="Edge orientation",
            )
            axes[0].legend(fontsize=7, loc="lower right")

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
