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
N_TOP_EDGES = 3                  # number of gradient peaks to auto-detect


class EdgeAnalyzer:
    """Extract edge spread function (ESF) and line spread function (LSF)
    from nebula edges to measure local contrast and resolution.

    When roi is None, the top N_TOP_EDGES well-separated gradient peaks are
    located automatically.  For each, a separate ESF/LSF measurement is run.
    The strongest edge's metrics are promoted to top-level keys for the summary
    table; all per-edge results are available in result["edges"].
    """

    def analyze(self, image: AstroImage,
                roi=None,
                forced_angles: list[float] | None = None) -> dict:
        """
        Parameters
        ----------
        roi : None | tuple | list[tuple]
            None            — auto-detect top-N gradient peaks across the image.
            single tuple    — analyse that one user-drawn region only.
            list of tuples  — use the supplied ROI coordinates (from image A)
                              so that image B covers the identical pixel regions.
        forced_angles : list[float] | None
            If provided, override the per-ROI angle detection with the supplied
            angle_rad values (one per ROI, in order).  Used so the weaker image
            adopts the stronger image's scan direction.
        """
        image.estimate_background()
        bgsub = image.background_subtracted()

        # Build list of (roi_data_array, roi_tuple) pairs
        if roi is None:
            roi_pairs = self._auto_detect_top_rois(bgsub, image)
        elif isinstance(roi, list):
            h, w = bgsub.shape
            roi_pairs = []
            for r in roi:
                x0, y0, x1, y1 = r
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                d = bgsub[y0:y1, x0:x1]
                if d.size > 0:
                    roi_pairs.append((d, r))
        else:
            x0, y0, x1, y1 = roi
            roi_pairs = [(bgsub[y0:y1, x0:x1], roi)]

        rois_used = [r for _, r in roi_pairs]

        edges: list[dict] = []
        for i, (roi_data, roi_tuple) in enumerate(roi_pairs):
            if roi_data is None or roi_data.size == 0:
                continue

            edge_info = self._detect_strongest_edge(roi_data)
            if edge_info is None:
                continue

            if forced_angles is not None and i < len(forced_angles):
                edge_info = dict(edge_info)
                edge_info["angle_rad"] = forced_angles[i]

            positions, esf = self._extract_esf(roi_data, edge_info)
            if esf is None or len(esf) < 5:
                continue

            lsf   = self._compute_lsf(positions, esf)
            width = self._measure_edge_width(positions, esf)
            ecr   = self._measure_edge_contrast_ratio(roi_data, edge_info)

            # 500×500 display context
            x0, y0, x1, y1 = roi_tuple
            xc_full = x0 + edge_info["center_x"]
            yc_full = y0 + edge_info["center_y"]
            dw  = EDGE_DISPLAY_HALF_WIDTH
            dx0 = max(0, xc_full - dw)
            dy0 = max(0, yc_full - dw)
            dx1 = min(bgsub.shape[1], xc_full + dw)
            dy1 = min(bgsub.shape[0], yc_full + dw)
            display_roi  = bgsub[dy0:dy1, dx0:dx1]
            analysis_rect = (x0 - dx0, y0 - dy0, x1 - dx0, y1 - dy0)
            edge_info_display = dict(edge_info)
            edge_info_display["center_x"] = xc_full - dx0
            edge_info_display["center_y"] = yc_full - dy0

            edges.append({
                "roi_used":               roi_tuple,
                "gradient_magnitude":     edge_info["gradient_magnitude"],
                "angle_rad":              edge_info["angle_rad"],
                "edge_width_10_90_px":    width,
                "edge_width_10_90_arcsec": (width * image.pixel_scale
                                            if width is not None else None),
                "edge_contrast_ratio":    ecr,
                "esf":                    esf,
                "lsf":                    lsf,
                "figures": figs_to_b64({
                    "edge": self._plot_results(
                        roi_data, display_roi, analysis_rect,
                        positions, esf, lsf, width, image.label,
                        edge_info_display, edge_num=i + 1,
                    )
                }),
            })

        # Top-level keys populated from the strongest edge for summary-table compat
        result: dict = {
            "edges":                 edges,
            "rois_used":             rois_used,
            "n_edges":               len(edges),
            "edge_width_10_90_px":   None,
            "edge_width_10_90_arcsec": None,
            "gradient_magnitude":    None,
            "edge_contrast_ratio":   None,
            "esf":                   None,
            "lsf":                   None,
            "roi_used":              rois_used[0] if rois_used else None,
        }
        if edges:
            best = max(edges, key=lambda e: e.get("gradient_magnitude") or 0)
            for k in ("edge_width_10_90_px", "edge_width_10_90_arcsec",
                      "gradient_magnitude", "edge_contrast_ratio", "esf", "lsf"):
                result[k] = best[k]

        return result

    # ------------------------------------------------------------------
    # ROI auto-detection
    # ------------------------------------------------------------------

    def _auto_detect_top_rois(self, bgsub: np.ndarray, image: AstroImage,
                               n: int = N_TOP_EDGES
                               ) -> list[tuple[np.ndarray, tuple]]:
        """Find n well-separated patches centred on the strongest gradients."""
        from core.stretch import stf_stretch
        stretched = stf_stretch(bgsub).astype(np.float64)
        gx = sobel(stretched, axis=1)
        gy = sobel(stretched, axis=0)
        gm = np.sqrt(gx ** 2 + gy ** 2)

        margin = EDGE_ROI_HALF_WIDTH + 5
        gm[:margin, :] = 0
        gm[-margin:, :] = 0
        gm[:, :margin] = 0
        gm[:, -margin:] = 0

        # 3× half-width separation keeps ROIs from overlapping
        min_sep = EDGE_ROI_HALF_WIDTH * 3
        h, w = bgsub.shape
        hw = EDGE_ROI_HALF_WIDTH

        results: list[tuple[np.ndarray, tuple]] = []
        gm_work = gm.copy()

        for _ in range(n):
            if gm_work.max() == 0:
                break
            yc, xc = np.unravel_index(np.argmax(gm_work), gm_work.shape)
            x0 = max(0, xc - hw);  y0 = max(0, yc - hw)
            x1 = min(w, xc + hw);  y1 = min(h, yc + hw)
            roi_tuple = (x0, y0, x1, y1)
            results.append((bgsub[y0:y1, x0:x1], roi_tuple))
            # Suppress neighbourhood so the next peak is in a different region
            s_y0 = max(0, yc - min_sep);  s_y1 = min(h, yc + min_sep)
            s_x0 = max(0, xc - min_sep);  s_x1 = min(w, xc + min_sep)
            gm_work[s_y0:s_y1, s_x0:s_x1] = 0

        return results

    # ------------------------------------------------------------------
    # Edge detection within ROI
    # ------------------------------------------------------------------

    def _detect_strongest_edge(self, roi_data: np.ndarray) -> dict | None:
        sx = sobel(roi_data, axis=1).astype(float)
        sy = sobel(roi_data, axis=0).astype(float)
        gm = np.sqrt(sx ** 2 + sy ** 2)

        yc, xc = np.unravel_index(np.argmax(gm), gm.shape)
        angle_rad = float(np.arctan2(sy[yc, xc], sx[yc, xc]))

        return {
            "center_x":          xc,
            "center_y":          yc,
            "angle_rad":         angle_rad,
            "gradient_magnitude": float(gm[yc, xc]),
        }

    # ------------------------------------------------------------------
    # ESF extraction
    # ------------------------------------------------------------------

    def _extract_esf(self, roi_data: np.ndarray,
                      edge_info: dict) -> tuple[np.ndarray, np.ndarray | None]:
        angle_deg = np.degrees(edge_info["angle_rad"])
        rotation_angle = -(90.0 - angle_deg)
        rotated = rotate(roi_data, rotation_angle, reshape=False, order=3)

        esf_raw   = np.mean(rotated, axis=0)
        positions = np.arange(len(esf_raw), dtype=float)

        lo, hi = esf_raw.min(), esf_raw.max()
        if hi - lo < 1e-12:
            return positions, None
        esf = (esf_raw - lo) / (hi - lo)

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
        xc   = edge_info["center_x"]
        half = max(5, roi_data.shape[1] // 4)
        bright_side = roi_data[:, max(0, xc - half):xc]
        dark_side   = roi_data[:, xc:min(roi_data.shape[1], xc + half)]

        bright_mean = float(np.mean(bright_side)) if bright_side.size > 0 else None
        dark_mean   = float(np.mean(dark_side))   if dark_side.size   > 0 else None

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
                       edge_info: dict | None = None,
                       edge_num: int = 1) -> plt.Figure:
        from matplotlib.patches import Rectangle

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        h_disp, w_disp = display_roi.shape
        axes[0].imshow(display_roi, origin="lower", cmap="gray",
                       aspect="equal", interpolation="nearest",
                       extent=[0, w_disp, 0, h_disp])
        axes[0].set_title(f"Edge #{edge_num} ROI — {label}")
        axes[0].set_xlabel("X (px)")
        axes[0].set_ylabel("Y (px)")

        bx0, by0, bx1, by1 = analysis_rect
        rect = Rectangle(
            (bx0, by0), bx1 - bx0, by1 - by0,
            linewidth=1.5, edgecolor="lime", facecolor="none",
            linestyle="--", zorder=5, label="Analysis region",
        )
        axes[0].add_patch(rect)

        if edge_info is not None:
            xc    = edge_info["center_x"]
            yc    = edge_info["center_y"]
            angle = edge_info["angle_rad"]
            t     = max(bx1 - bx0, by1 - by0) * 2

            def _clipped_line(cx, cy, ang):
                """Line endpoints clipped to the 60×60 analysis region."""
                x0_ = np.clip(cx - t * np.cos(ang), bx0, bx1)
                x1_ = np.clip(cx + t * np.cos(ang), bx0, bx1)
                y0_ = np.clip(cy - t * np.sin(ang), by0, by1)
                y1_ = np.clip(cy + t * np.sin(ang), by0, by1)
                return [x0_, x1_], [y0_, y1_]

            xs, ys = _clipped_line(xc, yc, angle)
            axes[0].plot(xs, ys, color="cyan", linewidth=1.5, alpha=0.85,
                         label="ESF scan direction")

            perp = angle + np.pi / 2
            xs, ys = _clipped_line(xc, yc, perp)
            axes[0].plot(xs, ys, color="yellow", linewidth=1.2, linestyle="--",
                         alpha=0.75, label="Edge orientation")

            axes[0].legend(fontsize=7, loc="lower right")

        axes[0].set_xlim(0, w_disp)
        axes[0].set_ylim(0, h_disp)

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

        axes[2].plot(positions, lsf, "r-", linewidth=1.5)
        axes[2].set_title("LSF (derivative of ESF)")
        axes[2].set_xlabel("Position (px)")
        axes[2].set_ylabel("d(ESF)/dx")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
