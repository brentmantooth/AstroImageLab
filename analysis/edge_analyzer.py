from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, rotate, gaussian_gradient_magnitude, median_filter, uniform_filter1d
from scipy.interpolate import interp1d

from core.astro_image import AstroImage
from core.fig_utils import figs_to_b64, finalize_layout
from core.models import (EDGE_ROI_HALF_WIDTH, EDGE_ROI_MAP_INDICATOR_PX,
                          SECTION8_BORDER_CROP_FRACTION, EDGE_ESF_MIN_MONOTONICITY,
                          EDGE_N_TOP_EDGES)

EDGE_DISPLAY_HALF_WIDTH = 250   # half-side of the context window shown in the report figure
N_CANDIDATE_EDGES = EDGE_N_TOP_EDGES * 3   # extra auto-detect candidates so low-quality ones can be skipped
_ESF_DISC_MARGIN_PX = 2.0        # safety margin (px) subtracted from the inscribed-circle radius
_ESF_QUALITY_SMOOTH_FRAC = 0.20  # smoothing window as a fraction of ESF length, for the quality metric only


def _gradient_sigma(pixel_scale: float) -> float:
    """Gaussian sigma in pixels giving ~1.5 arcsec equivalent smoothing, capped 1–8 px.

    Long-focal-length images have small pixel scales (e.g. 0.3 "/px) and their
    nebula-edge gradients span many pixels; a larger sigma avoids missing them.
    Short-focal-length images (e.g. 2 "/px) keep sigma near 1 px.
    """
    return float(np.clip(1.5 / max(pixel_scale, 0.01), 1.0, 8.0))


class EdgeAnalyzer:
    """Extract edge spread function (ESF) and line spread function (LSF)
    from nebula edges to measure local contrast and resolution.

    When roi is None, the top EDGE_N_TOP_EDGES well-separated gradient peaks are
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

        # Build list of (roi_data_array, roi_tuple) pairs. Auto-detect mode
        # searches more candidates than EDGE_N_TOP_EDGES so low-quality ones (ESF
        # crossed more than one physical edge) can be skipped in favour of the
        # next-strongest gradient peak; user-drawn/A-matched ROIs are fixed
        # and can't be swapped for an alternative, so quality is only flagged
        # for those, never used to drop them.
        allow_skip = roi is None
        if roi is None:
            roi_pairs = self._auto_detect_top_rois(bgsub, image, n=N_CANDIDATE_EDGES)
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
        rejected: list[dict] = []   # low-quality auto-detect candidates, kept only as a fallback
        for i, (roi_data, roi_tuple) in enumerate(roi_pairs):
            if allow_skip and len(edges) >= EDGE_N_TOP_EDGES:
                break
            if roi_data is None or roi_data.size == 0:
                continue

            edge_info = self._detect_strongest_edge(roi_data)
            if edge_info is None:
                continue

            if forced_angles is not None and i < len(forced_angles):
                edge_info = dict(edge_info)
                edge_info["angle_rad"] = forced_angles[i]

            positions, esf, start_xy = self._extract_esf(roi_data, edge_info)
            if esf is None or len(esf) < 5:
                continue

            quality = self._esf_quality(esf)
            low_confidence = quality < EDGE_ESF_MIN_MONOTONICITY
            entry = self._build_edge_entry(
                image, bgsub, roi_data, roi_tuple, edge_info,
                positions, esf, start_xy, i, quality, low_confidence)

            if allow_skip and low_confidence:
                rejected.append(entry)
                continue
            edges.append(entry)

        if allow_skip and not edges and rejected:
            # Every candidate crossed more than one edge -- surface the least
            # bad one rather than showing nothing, clearly flagged.
            rejected.sort(key=lambda e: e["esf_quality"], reverse=True)
            edges.append(rejected[0])

        # Overview map should only mark the edges actually analyzed, not every
        # searched candidate (auto-detect searches N_CANDIDATE_EDGES > len(edges)
        # so low-quality candidates can be skipped in favour of the next peak).
        rois_used = [e["roi_used"] for e in edges]

        # Full-image gradient map for report visualisation
        from core.stretch import stf_stretch
        sigma = _gradient_sigma(image.pixel_scale)
        gm_full = gaussian_gradient_magnitude(
            stf_stretch(bgsub), sigma=sigma
        )
        gradient_fig = self._plot_gradient_map(gm_full, image.label, rois_used)

        # Store downsampled display array for shared-scale rendering in the report
        h_gm, w_gm = gm_full.shape
        step_gm = max(1, max(h_gm, w_gm) // 800)
        gm_display = gm_full[::step_gm, ::step_gm].astype(np.float32)
        gm_display = median_filter(gm_display, size=3)
        gm_vmax = float(np.percentile(gm_display, 99)) or 1.0

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
            "gm_display":            gm_display,
            "gm_vmax":               gm_vmax,
            "gm_shape":              (h_gm, w_gm),
            "figures":               figs_to_b64({"gradient_map": gradient_fig}),
        }
        if edges:
            # Prefer a confident measurement over a merely-stronger-gradient one
            # (a corner/knot can have higher raw gradient than a clean edge).
            best = max(edges, key=lambda e: (not e.get("low_confidence", False),
                                              e.get("gradient_magnitude") or 0))
            for k in ("edge_width_10_90_px", "edge_width_10_90_arcsec",
                      "gradient_magnitude", "edge_contrast_ratio", "esf", "lsf"):
                result[k] = best[k]

        return result

    # ------------------------------------------------------------------
    # Edge-result assembly
    # ------------------------------------------------------------------

    def _build_edge_entry(self, image: AstroImage, bgsub: np.ndarray,
                           roi_data: np.ndarray, roi_tuple: tuple,
                           edge_info: dict, positions: np.ndarray, esf: np.ndarray,
                           start_xy: tuple | None,
                           edge_num: int, quality: float, low_confidence: bool) -> dict:
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

        # The ESF scan/edge-orientation guide lines must pivot on the ROI's own
        # geometric center, not the Sobel-detected gradient peak (edge_info's
        # center_x/center_y) -- scipy.ndimage.rotate() in _extract_esf always
        # rotates roi_data about its own array center ((h-1)/2, (w-1)/2), and
        # start_xy (the "Scan start" marker) is derived by inverse-rotating a
        # point on that same pivot. Drawing the guide lines through the
        # gradient-peak point instead left the red marker floating off the
        # cyan line whenever the peak wasn't exactly at the ROI's center.
        h_roi, w_roi = roi_data.shape
        roi_cx_full = x0 + (w_roi - 1) / 2.0
        roi_cy_full = y0 + (h_roi - 1) / 2.0
        edge_info_display = dict(edge_info)
        edge_info_display["center_x"] = roi_cx_full - dx0
        edge_info_display["center_y"] = roi_cy_full - dy0

        # ESF start point (position index 0), in the same display-frame
        # coordinates as edge_info_display, for the directional marker.
        start_xy_display = None
        if start_xy is not None:
            start_x_full = x0 + start_xy[0]
            start_y_full = y0 + start_xy[1]
            start_xy_display = (start_x_full - dx0, start_y_full - dy0)

        return {
            "roi_used":               roi_tuple,
            "gradient_magnitude":     edge_info["gradient_magnitude"],
            "angle_rad":              edge_info["angle_rad"],
            "edge_width_10_90_px":    width,
            "edge_width_10_90_arcsec": (width * image.pixel_scale
                                        if width is not None else None),
            "edge_contrast_ratio":    ecr,
            "esf":                    esf,
            "lsf":                    lsf,
            "positions":              positions.tolist(),
            "esf_quality":            quality,
            "low_confidence":         low_confidence,
            "figures": figs_to_b64({
                "edge": self._plot_results(
                    roi_data, display_roi, analysis_rect,
                    positions, esf, lsf, width, image.label,
                    edge_info_display, edge_num=edge_num + 1,
                    low_confidence=low_confidence,
                    start_xy_display=start_xy_display,
                )
            }),
        }

    # ------------------------------------------------------------------
    # ROI auto-detection
    # ------------------------------------------------------------------

    def _auto_detect_top_rois(self, bgsub: np.ndarray, image: AstroImage,
                               n: int = EDGE_N_TOP_EDGES
                               ) -> list[tuple[np.ndarray, tuple]]:
        """Find n well-separated patches centred on the strongest gradients."""
        from core.stretch import stf_stretch
        stretched = stf_stretch(bgsub)
        sigma = _gradient_sigma(image.pixel_scale)
        gm = gaussian_gradient_magnitude(stretched, sigma=sigma)

        h, w = gm.shape
        margin = max(EDGE_ROI_HALF_WIDTH + 5, int(min(h, w) * SECTION8_BORDER_CROP_FRACTION))
        gm[:margin, :] = 0
        gm[-margin:, :] = 0
        gm[:, :margin] = 0
        gm[:, -margin:] = 0

        # 3× half-width separation keeps ROIs from overlapping
        min_sep = EDGE_ROI_HALF_WIDTH * 3
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
                      edge_info: dict) -> tuple[np.ndarray, np.ndarray | None, tuple | None]:
        angle_deg = np.degrees(edge_info["angle_rad"])
        # Empirically verified (not hand-derived -- see the impulse-trick
        # comment below for why that matters with rotate()'s sign/handedness):
        # rendering rotate(roi_data, angle, ...) for known synthetic edge
        # angles and inspecting the result directly shows rotation_angle =
        # angle_deg aligns the edge vertically, exactly as this function
        # needs (esf_raw below averages DOWN COLUMNS, so the edge must run
        # vertically for that average to stay on one side of the transition
        # per column). A previous formula, -(90.0 - angle_deg), looked like
        # a more "natural" complement but actually aligned the edge
        # HORIZONTALLY instead -- averaging down columns then integrated
        # straight across the transition, canceling out nearly all of the
        # real signal and leaving only a disc-boundary/interpolation
        # artifact (still smooth and monotonic, so _esf_quality's gate never
        # caught it).
        rotation_angle = angle_deg
        # cval=nan (not the default 0.0) marks pixels that rotate() had to
        # invent because the source square doesn't cover that output pixel at
        # this angle -- see the disc-mask comment below for why this matters.
        rotated = rotate(roi_data, rotation_angle, reshape=False, order=3, cval=np.nan)

        # Rotating a square ROI about its own center clips its corners for any
        # angle other than 0/90 deg (reshape=False keeps the output the same
        # size as the input, so content that rotates outside that frame is
        # lost). Averaging over the full box (the previous behaviour) mixes
        # this invented/zero-filled corner content into the profile, which can
        # fabricate a second, spurious transition -- worst at 45 deg, where
        # ~30% of the box area is affected. The disc inscribed in the ROI
        # square (radius = half the side length) is the largest region
        # guaranteed to contain only genuine data at ANY rotation angle, since
        # a rotation about the center never moves points closer to the center
        # outside the original square. Restrict averaging to that disc (minus
        # a small margin for cubic-spline interpolation at the boundary).
        h, w = rotated.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.mgrid[0:h, 0:w]
        radius = min(h, w) / 2.0 - _ESF_DISC_MARGIN_PX
        in_disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
        masked = np.where(in_disc & ~np.isnan(rotated), rotated, np.nan)

        # Columns beyond the disc radius are entirely NaN by construction
        # (trimmed out below) -- nanmean's "empty slice" warning for those is
        # expected, not a sign of a problem.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            esf_raw = np.nanmean(masked, axis=0)
        positions = np.arange(len(esf_raw), dtype=float)

        valid_idx = np.where(~np.isnan(esf_raw))[0]
        if len(valid_idx) < 5:
            return positions, None, None
        lo = float(np.nanmin(esf_raw))
        hi = float(np.nanmax(esf_raw))
        if hi - lo < 1e-12:
            return positions, None, None
        esf = (esf_raw - lo) / (hi - lo)

        if esf[valid_idx[0]] > esf[valid_idx[-1]]:
            esf = 1.0 - esf

        # Trim to the contiguous valid range (only the disc-excluded columns
        # near the two far ends can be NaN) so downstream LSF/width code never
        # has to handle missing values.
        i0, i1 = valid_idx[0], valid_idx[-1] + 1

        # Locate the ROI-local (pre-rotation) pixel corresponding to the ESF's
        # start (position index 0), for a directional marker. Uses rotate()
        # itself in reverse -- an impulse at the same rotated-frame column,
        # inverse-rotated back -- rather than hand-deriving rotate()'s angle-
        # sign convention, so this stays correct regardless of its handedness.
        impulse = np.zeros((h, w), dtype=float)
        impulse[int(round(cy)), i0] = 1.0
        impulse_orig = rotate(impulse, -rotation_angle, reshape=False, order=1, cval=0.0)
        start_ry, start_rx = np.unravel_index(np.argmax(impulse_orig), impulse_orig.shape)
        start_xy = (int(start_rx), int(start_ry))

        return positions[i0:i1] - positions[i0], esf[i0:i1], start_xy

    @staticmethod
    def _esf_quality(esf: np.ndarray) -> float:
        """Monotonicity ratio of a lightly-smoothed ESF: net variation over
        total variation. 1.0 = a clean single transition; low values mean the
        profile doubles back on itself (median(|detail|)-style contamination
        from a second edge -- a corner, filament, or nearby knot crossed by
        the same scan line)."""
        n = len(esf)
        win = max(3, int(round(n * _ESF_QUALITY_SMOOTH_FRAC)) | 1)
        smoothed = uniform_filter1d(esf.astype(float), size=win, mode="nearest")
        diffs = np.diff(smoothed)
        total_variation = float(np.sum(np.abs(diffs)))
        if total_variation < 1e-9:
            return 0.0
        net_variation = float(abs(smoothed[-1] - smoothed[0]))
        return net_variation / total_variation

    # ------------------------------------------------------------------
    # LSF and width
    # ------------------------------------------------------------------

    def _compute_lsf(self, positions: np.ndarray,
                      esf: np.ndarray) -> np.ndarray:
        from scipy.signal import savgol_filter
        n = len(esf)
        # Window: ~18% of ESF length, minimum 5, always odd, must be < n
        raw_w = max(5, round(0.18 * n))
        window = raw_w if raw_w % 2 == 1 else raw_w + 1
        window = min(window, n if n % 2 == 1 else n - 1)
        if window < 5 or window >= n:
            return np.gradient(esf, positions)
        delta = float(positions[1] - positions[0]) if len(positions) > 1 else 1.0
        return savgol_filter(esf, window_length=window, polyorder=3, deriv=1, delta=delta)

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
    # Gradient map visualisation
    # ------------------------------------------------------------------

    def _plot_gradient_map(self, gm: np.ndarray, label: str,
                            rois_used: list) -> plt.Figure:
        """Full-image gradient magnitude (border-cropped) with detected ROI rectangles overlaid."""
        from matplotlib.patches import Rectangle
        h, w = gm.shape
        nb = int(min(h, w) * SECTION8_BORDER_CROP_FRACTION)
        gm_crop = gm[nb:h - nb, nb:w - nb]
        hc, wc = gm_crop.shape
        aspect = hc / max(wc, 1)
        fig_w = min(8.0, max(4.0, wc / 400))
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
        step = max(1, max(hc, wc) // 800)
        disp = gm_crop[::step, ::step]
        disp = median_filter(disp, size=3)
        vmax = float(np.percentile(disp, 99)) or 1.0
        ax.imshow(disp, origin="upper", cmap="inferno",
                  vmin=0, vmax=vmax,
                  extent=[0, wc, hc, 0], aspect="equal", interpolation="nearest")
        half = EDGE_ROI_MAP_INDICATOR_PX // 2
        for (x0, y0, x1, y1) in rois_used:
            cx = (x0 + x1) / 2.0 - nb
            cy = (y0 + y1) / 2.0 - nb
            ix0 = max(0, cx - half)
            iy0 = max(0, cy - half)
            ix1 = min(wc, cx + half)
            iy1 = min(hc, cy + half)
            rect = Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0,
                              linewidth=2.0, edgecolor="cyan", facecolor="none",
                              linestyle="-")
            ax.add_patch(rect)
        ax.set_title(f"Gradient magnitude — {label}")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        finalize_layout(fig)
        return fig

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
                       edge_num: int = 1,
                       low_confidence: bool = False,
                       start_xy_display: tuple | None = None) -> plt.Figure:
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(5, 5))

        h_disp, w_disp = display_roi.shape
        ax.imshow(display_roi, origin="upper", cmap="gray",
                  aspect="equal", interpolation="nearest",
                  extent=[0, w_disp, h_disp, 0])
        title = f"Edge #{edge_num} ROI — {label}"
        if low_confidence:
            title += "  ⚠ low confidence"
            ax.set_title(title, color="#c0392b")
        else:
            ax.set_title(title)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

        bx0, by0, bx1, by1 = analysis_rect
        rect = Rectangle(
            (bx0, by0), bx1 - bx0, by1 - by0,
            linewidth=1.5, edgecolor="lime", facecolor="none",
            linestyle="--", zorder=5, label="Analysis region",
        )
        ax.add_patch(rect)

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
            ax.plot(xs, ys, color="cyan", linewidth=1.5, alpha=0.85,
                    label="ESF scan direction")

            perp = angle + np.pi / 2
            xs, ys = _clipped_line(xc, yc, perp)
            ax.plot(xs, ys, color="yellow", linewidth=1.2, linestyle="--",
                    alpha=0.75, label="Edge orientation")

            if start_xy_display is not None:
                ax.plot(start_xy_display[0], start_xy_display[1], marker="s",
                        markersize=5, color="red", zorder=6,
                        label="Scan start")

            ax.legend(fontsize=7, loc="lower right")

        ax.set_xlim(0, w_disp)
        ax.set_ylim(0, h_disp)

        finalize_layout(fig)
        return fig

    # ------------------------------------------------------------------
    # Cross-section edge analysis along a user-drawn line
    # ------------------------------------------------------------------

    def analyze_crosshair(self, image: AstroImage, crosshair: dict) -> dict | None:
        """Extract ESF + LSF along the user-drawn cross-section line.

        crosshair has keys x0, y0, x1, y1 in normalised [0,1] full-image coords.
        Returns an edge dict (same structure as entries in result["edges"]) or None.
        """
        from scipy.ndimage import map_coordinates
        image.estimate_background()
        bgsub = image.background_subtracted()
        H, W = bgsub.shape
        c0, r0 = crosshair["x0"] * W, crosshair["y0"] * H
        c1, r1 = crosshair["x1"] * W, crosshair["y1"] * H
        length = float(np.hypot(c1 - c0, r1 - r0))
        if length < 5:
            return None
        n = max(20, int(length))
        cols = np.linspace(c0, c1, n)
        rows = np.linspace(r0, r1, n)
        profile = map_coordinates(bgsub.astype(np.float32), [rows, cols],
                                  order=1, mode="nearest")
        positions = np.linspace(0.0, length, n)
        lo, hi = profile.min(), profile.max()
        if hi - lo < 1e-12:
            return None
        esf = (profile - lo) / (hi - lo)
        lsf = self._compute_lsf(positions, esf)
        width = self._measure_edge_width(positions, esf)
        return {
            "roi_used":                None,
            "gradient_magnitude":      float(hi - lo),
            "angle_rad":               float(np.arctan2(r1 - r0, c1 - c0)),
            "edge_width_10_90_px":     width,
            "edge_width_10_90_arcsec": (width * image.pixel_scale if width else None),
            "edge_contrast_ratio":     None,
            "esf":                     esf.tolist(),
            "lsf":                     lsf.tolist() if lsf is not None else [],
            "positions":               positions.tolist(),
            "is_crosshair":            True,
            "figures":                 {},
        }
