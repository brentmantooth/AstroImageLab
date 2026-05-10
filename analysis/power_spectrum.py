from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from core.astro_image import AstroImage
from core.models import POWER_SPECTRUM_NPIX

LOW_FREQ_MAX = 0.10    # cycles/px boundary between low and mid+high
NYQUIST = 0.50         # cycles/px


class PowerSpectrumAnalyzer:
    """Compute 2D power spectrum of a star-free nebula region.

    All processing uses mean-signal-normalised data so the result is
    dimensionless and comparable across filters with different bandwidths.
    """

    def analyze(self, image: AstroImage,
                roi: tuple[int, int, int, int] | None = None) -> dict:
        image.estimate_background()
        bgsub = image.background_subtracted()

        result: dict = {
            "mid_high_ratio": None,
            "power_spectrum_2d": None,
            "radial_power": None,
            "freq_axis": None,
        }

        region = self._extract_roi(bgsub, image, roi)
        if region is None:
            return result

        normalised = self._normalise(region)
        if normalised is None:
            return result

        windowed = self._apply_window(normalised)
        ps2d = self._compute_ps2d(windowed)
        freq, radial = self._radial_average(ps2d)
        ratio = self._compute_mid_high_ratio(freq, radial)

        result.update({
            "mid_high_ratio": ratio,
            "power_spectrum_2d": ps2d,
            "radial_power": radial,
            "freq_axis": freq,
        })
        result["figures"] = {
            "power_spectrum": self._plot_results(ps2d, freq, radial, ratio, image.label)
        }
        return result

    # ------------------------------------------------------------------
    # ROI selection
    # ------------------------------------------------------------------

    def _extract_roi(self, bgsub: np.ndarray,
                      image: AstroImage,
                      roi: tuple | None) -> np.ndarray | None:
        N = POWER_SPECTRUM_NPIX
        h, w = bgsub.shape

        if roi is not None:
            x0, y0, x1, y1 = roi
            sub = bgsub[y0:y1, x0:x1]
            # Resize to NxN if needed
            from scipy.ndimage import zoom
            if sub.shape != (N, N):
                zy = N / sub.shape[0]
                zx = N / sub.shape[1]
                sub = zoom(sub, (zy, zx), order=1)
            return sub[:N, :N]

        # Auto-select: find a star-free NxN region
        catalog = getattr(image, "catalog", None)
        if catalog is not None and len(catalog) > 0:
            star_xs = np.array(catalog["xcentroid"])
            star_ys = np.array(catalog["ycentroid"])
        else:
            star_xs, star_ys = np.array([]), np.array([])

        # Try candidate positions on a grid
        step = N // 2
        best_pos = None
        best_score = -np.inf
        for yc in range(N // 2, h - N // 2, step):
            for xc in range(N // 2, w - N // 2, step):
                x0 = xc - N // 2
                y0 = yc - N // 2
                if len(star_xs) > 0:
                    dists = np.sqrt((star_xs - xc)**2 + (star_ys - yc)**2)
                    if dists.min() < N // 2:
                        continue
                patch = bgsub[y0:y0 + N, x0:x0 + N]
                score = float(np.mean(patch))
                if score > best_score:
                    best_score = score
                    best_pos = (x0, y0)

        if best_pos is None:
            # Fallback: centre of image
            x0 = w // 2 - N // 2
            y0 = h // 2 - N // 2
            best_pos = (x0, y0)

        x0, y0 = best_pos
        region = bgsub[y0:y0 + N, x0:x0 + N].copy()

        # Sigma-clip to remove unmasked faint stars
        clipped = sigma_clip(region, sigma=3.0, maxiters=3)
        region[clipped.mask] = float(np.ma.median(clipped))
        return region

    # ------------------------------------------------------------------
    # Normalisation (bandwidth-independent)
    # ------------------------------------------------------------------

    def _normalise(self, region: np.ndarray) -> np.ndarray | None:
        mean_signal = float(np.mean(region[region > 0])) if np.any(region > 0) else 0.0
        if mean_signal <= 0:
            return None
        return region / mean_signal

    # ------------------------------------------------------------------
    # Window and FFT
    # ------------------------------------------------------------------

    def _apply_window(self, data: np.ndarray) -> np.ndarray:
        N = data.shape[0]
        win = np.outer(np.hanning(N), np.hanning(N))
        return (data - data.mean()) * win

    def _compute_ps2d(self, windowed: np.ndarray) -> np.ndarray:
        N = windowed.shape[0]
        fft2d = np.fft.fftshift(np.fft.fft2(windowed))
        return (np.abs(fft2d) ** 2) / (N ** 2)

    # ------------------------------------------------------------------
    # Radial average
    # ------------------------------------------------------------------

    def _radial_average(self, ps2d: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        N = ps2d.shape[0]
        cx, cy = N // 2, N // 2
        y_idx, x_idx = np.mgrid[0:N, 0:N]
        r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)

        max_r = N // 2
        nbins = max_r
        edges = np.linspace(0, max_r, nbins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        freq_axis = centers / max_r * NYQUIST

        radial = np.array([
            np.mean(ps2d[(r >= edges[i]) & (r < edges[i + 1])])
            if np.any((r >= edges[i]) & (r < edges[i + 1])) else 0.0
            for i in range(nbins)
        ])
        return freq_axis, radial

    def _compute_mid_high_ratio(self, freq: np.ndarray,
                                  radial: np.ndarray) -> float | None:
        low_mask = freq <= LOW_FREQ_MAX
        high_mask = freq > LOW_FREQ_MAX
        low_sum = float(np.sum(radial[low_mask]))
        high_sum = float(np.sum(radial[high_mask]))
        if low_sum <= 0:
            return None
        return high_sum / low_sum

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _plot_results(self, ps2d: np.ndarray,
                       freq: np.ndarray,
                       radial: np.ndarray,
                       ratio: float | None,
                       label: str) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im = axes[0].imshow(
            np.log10(ps2d + ps2d[ps2d > 0].min() * 0.01),
            origin="lower", cmap="inferno",
            extent=[-NYQUIST, NYQUIST, -NYQUIST, NYQUIST],
        )
        axes[0].set_title(f"2D Power Spectrum (log) — {label}")
        axes[0].set_xlabel("Freq X (cyc/px)")
        axes[0].set_ylabel("Freq Y (cyc/px)")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].semilogy(freq, radial, linewidth=1.5)
        axes[1].axvline(LOW_FREQ_MAX, color="gray", linestyle="--",
                         linewidth=0.8, label=f"Low/high boundary ({LOW_FREQ_MAX} cyc/px)")
        title = f"Radial power — {label}"
        if ratio is not None:
            title += f"\nMid/high ratio = {ratio:.4f}"
        axes[1].set_title(title)
        axes[1].set_xlabel("Spatial frequency (cyc/px)")
        axes[1].set_ylabel("Normalised power (dimensionless)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
