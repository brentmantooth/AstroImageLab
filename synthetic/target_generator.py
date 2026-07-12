"""Spatial test target generator for spatial-detail metric calibration.

Generates a 4×3 grid of calibrated test patterns with known spatial frequencies
and waveform types.  Load the clean reference into Image A and the degraded
companion into Image B to calibrate the app's spatial-detail metrics against
known inputs.

The four column frequencies are deliberately chosen to align with the four
wavelet decomposition band centres (levels 4→1, coarse→fine):
    f=0.04 c/px → λ=25 px → wavelet level 4 (~16 px scale)
    f=0.08 c/px → λ=12 px → wavelet level 3 (~8 px scale)
    f=0.16 c/px → λ=6 px  → wavelet level 2 (~4 px scale)
    f=0.32 c/px → λ=3 px  → wavelet level 1 (~2 px / near Nyquist)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from synthetic.generator import _moffat_2d, _fwhm_to_alpha, _siemens_star


# (row, col) → (waveform_type, frequency_cycles_per_pixel)
# Row 0: horizontal sine sweep — cleanest MTF measurement
# Row 1: horizontal square-wave sweep — tests harmonic/ringing behavior
# Row 2: direction variants + special targets
_ZONE_SPEC: dict[tuple[int, int], tuple[str, float]] = {
    (0, 0): ("sine_h",   0.04),
    (0, 1): ("sine_h",   0.08),
    (0, 2): ("sine_h",   0.16),
    (0, 3): ("sine_h",   0.32),
    (1, 0): ("square_h", 0.04),
    (1, 1): ("square_h", 0.08),
    (1, 2): ("square_h", 0.16),
    (1, 3): ("square_h", 0.32),
    (2, 0): ("sine_v",   0.08),
    (2, 1): ("sine_45",  0.08),
    (2, 2): ("siemens",  0.00),
    (2, 3): ("edge",     0.00),
}

N_ROWS = 3
N_COLS = 4

# Human-readable zone descriptions exported for use in the dialog info panel
ZONE_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "Sine H  f=0.04 c/px  (λ≈25 px, wavelet L4)",
    (0, 1): "Sine H  f=0.08 c/px  (λ≈12 px, wavelet L3)",
    (0, 2): "Sine H  f=0.16 c/px  (λ≈6 px,  wavelet L2)",
    (0, 3): "Sine H  f=0.32 c/px  (λ≈3 px,  wavelet L1 / near Nyquist)",
    (1, 0): "Square H  f=0.04 c/px",
    (1, 1): "Square H  f=0.08 c/px",
    (1, 2): "Square H  f=0.16 c/px",
    (1, 3): "Square H  f=0.32 c/px",
    (2, 0): "Sine V  f=0.08 c/px  (vertical direction)",
    (2, 1): "Sine 45°  f=0.08 c/px  (diagonal direction)",
    (2, 2): "Siemens star  (all-angle radial chirp)",
    (2, 3): "Slant edge ~5°  (ISO 12233-style ESF input)",
}


class SpatialTargetGenerator:
    """Generate a calibrated spatial test target FITS image pair.

    Always produces two FITS files:
      • clean reference  — pure pattern, no PSF, no noise
      • degraded companion — same pattern with optional PSF blur and/or noise
    Load clean → Image A and degraded → Image B, then run the analysis to see
    how each metric responds to known spatial-frequency content.
    """

    def generate(self, params: dict,
                 preview: bool = False) -> tuple[str, str] | np.ndarray:
        """Generate the target pair.

        preview=True  → returns degraded ndarray (float32) for live display
        preview=False → returns (clean_path, degraded_path) as FITS paths
        """
        width        = int(params["width"])
        height       = int(params["height"])
        contrast_min = float(params.get("contrast_min", params.get("contrast", 0.02)))
        contrast_max = float(params.get("contrast_max", params.get("contrast", 0.50)))
        sky_adu      = float(params["sky_adu"])
        fwhm_px      = float(params.get("fwhm_px", 0.0))
        beta         = float(params.get("moffat_beta", 4.77))
        rn_adu       = float(params.get("read_noise_adu", 10.0))

        if preview:
            scale  = min(640 / width, 480 / height, 1.0)
            width  = max(64, int(width  * scale) & ~1)
            height = max(64, int(height * scale) & ~1)

        clean    = self._build_pattern(width, height, contrast_min, contrast_max, sky_adu)
        degraded = clean.copy()

        if params.get("apply_psf") and fwhm_px > 0:
            degraded = self._apply_psf(degraded, fwhm_px, beta)
        if params.get("apply_noise"):
            degraded = self._add_noise(degraded, sky_adu, rn_adu,
                                       seed=int(params.get("seed", 42)))

        clean    = np.clip(clean,    0.0, 65535.0).astype(np.float32)
        degraded = np.clip(degraded, 0.0, 65535.0).astype(np.float32)

        if preview:
            return degraded

        clean_path    = self._write_fits(clean,    params, is_clean=True)
        degraded_path = self._write_fits(degraded, params, is_clean=False)
        return (clean_path, degraded_path)

    # ------------------------------------------------------------------
    # Pattern construction
    # ------------------------------------------------------------------

    def _build_pattern(self, W: int, H: int,
                       contrast_min: float, contrast_max: float,
                       dc: float) -> np.ndarray:
        """Assemble the N_ROWS × N_COLS zone grid as a float64 array."""
        image        = np.full((H, W), dc, dtype=np.float64)
        zw           = W // N_COLS
        zh           = H // N_ROWS
        amplitude_min = dc * contrast_min
        amplitude_max = dc * contrast_max

        for (row, col), (waveform, freq) in _ZONE_SPEC.items():
            x0 = col * zw
            y0 = row * zh
            x1 = x0 + zw if col < N_COLS - 1 else W
            y1 = y0 + zh if row < N_ROWS - 1 else H
            zW, zH = x1 - x0, y1 - y0
            image[y0:y1, x0:x1] = self._make_zone(
                waveform, freq, zW, zH, amplitude_min, amplitude_max, dc)
        return image

    def _make_zone(self, waveform: str, freq: float,
                   W: int, H: int,
                   amplitude_min: float, amplitude_max: float,
                   dc: float) -> np.ndarray:
        """Return an (H × W) float64 zone with contrast ramping top→bottom.

        Contrast (Michelson) increases linearly from amplitude_min/dc at the
        top edge to amplitude_max/dc at the bottom edge of the zone.  At any
        horizontal row across all N_COLS zones the contrast is identical,
        enabling direct cross-frequency comparison at the same contrast level.
        """
        y_idx, x_idx = np.mgrid[0:H, 0:W].astype(np.float64)
        # Y-ramp: 0 at top of zone, 1 at bottom — independent of absolute image position
        y_rel         = y_idx / max(H - 1, 1)
        amplitude_map = amplitude_min + (amplitude_max - amplitude_min) * y_rel

        if waveform == "sine_h":
            pattern = np.sin(2.0 * np.pi * freq * x_idx)

        elif waveform == "square_h":
            # Hard square wave — preserve edge sharpness to test Gibbs behavior
            pattern = np.where(np.sin(2.0 * np.pi * freq * x_idx) >= 0, 1.0, -1.0)

        elif waveform == "sine_v":
            pattern = np.sin(2.0 * np.pi * freq * y_idx)

        elif waveform == "sine_45":
            # Diagonal at 45°: spatial frequency along (1,1) direction
            d = (x_idx + y_idx) / np.sqrt(2.0)
            pattern = np.sin(2.0 * np.pi * freq * d)

        elif waveform == "siemens":
            size = min(W, H)
            if size % 2 == 0:
                size -= 1   # odd so the star centre falls on a pixel
            ss = _siemens_star(size)   # values 0–~0.5, tapered to 0 at centre/edge
            # Centre at 0 and normalise to [-1, 1] so the zone DC = dc
            ss_c   = ss - ss.mean()
            ss_max = np.abs(ss_c).max()
            if ss_max > 0:
                ss_c /= ss_max
            pat = np.zeros((H, W), dtype=np.float64)
            sx = (W - size) // 2;  sy = (H - size) // 2
            tx = max(0, sx);        ty = max(0, sy)
            ox = max(0, -sx);       oy = max(0, -sy)
            cw = min(size - ox, W - tx)
            ch = min(size - oy, H - ty)
            if cw > 0 and ch > 0:
                pat[ty:ty+ch, tx:tx+cw] = ss_c[oy:oy+ch, ox:ox+cw]
            pattern = pat

        elif waveform == "edge":
            # Slant edge at ~5° from vertical — hard step for ISO 12233-style ESF
            angle_rad = np.radians(5.0)
            cx, cy    = W / 2.0, H / 2.0
            # Signed distance from the slant line (positive = right side)
            dist    = ((x_idx - cx) * np.cos(angle_rad)
                       - (y_idx - cy) * np.sin(angle_rad))
            pattern = np.where(dist >= 0, 1.0, -1.0)

        else:
            pattern = np.zeros((H, W), dtype=np.float64)

        return dc + amplitude_map * pattern

    # ------------------------------------------------------------------
    # Degradation
    # ------------------------------------------------------------------

    def _apply_psf(self, image: np.ndarray,
                   fwhm_px: float, beta: float) -> np.ndarray:
        """Convolve with an isotropic Moffat PSF (field-centre, no aberrations)."""
        alpha = _fwhm_to_alpha(fwhm_px, beta)
        stamp = max(5, int(fwhm_px * 6))
        if stamp % 2 == 0:
            stamp += 1
        kern  = _moffat_2d(stamp, alpha, beta)
        kern /= kern.sum()
        result = fftconvolve(image, kern, mode="same")
        return np.maximum(result, 0.0)

    def _add_noise(self, image: np.ndarray, sky_adu: float,
                   read_noise_adu: float, seed: int = 42) -> np.ndarray:
        """Add Poisson sky shot noise + Gaussian read noise (gain=1 assumed)."""
        rng     = np.random.default_rng(seed)
        sky_e   = max(0.1, sky_adu)
        poisson = rng.poisson(sky_e, size=image.shape).astype(np.float64)
        rdnoise = rng.normal(0.0, read_noise_adu, size=image.shape)
        return image + poisson + rdnoise

    # ------------------------------------------------------------------
    # FITS output
    # ------------------------------------------------------------------

    def _write_fits(self, image: np.ndarray,
                    params: dict, is_clean: bool) -> str:
        from astropy.io import fits

        W            = image.shape[1]
        H            = image.shape[0]
        contrast_min = float(params.get("contrast_min", params.get("contrast", 0.02)))
        contrast_max = float(params.get("contrast_max", params.get("contrast", 0.50)))
        sky_adu      = float(params["sky_adu"])

        hdr = fits.Header()
        hdr["SIMPLE"]   = True
        hdr["BITPIX"]   = -32
        hdr["NAXIS"]    = 2
        hdr["NAXIS1"]   = W
        hdr["NAXIS2"]   = H
        hdr["INSTRUME"] = "SpatialTarget"
        hdr["EGAIN"]    = (1.0, "e-/ADU gain=1 assumed")
        hdr["GAIN"]     = (1.0, "e-/ADU gain=1 assumed")
        # Grid metadata — allows future auto-detection of zone layout in analysis
        hdr["TGT_TYPE"] = ("spatial_grid",  "generator: spatial test target")
        hdr["TGT_ROWS"] = (N_ROWS,          "grid rows")
        hdr["TGT_COLS"] = (N_COLS,          "grid columns")
        hdr["TGT_CMIN"] = (contrast_min,     "Michelson contrast at zone top (min)")
        hdr["TGT_CMAX"] = (contrast_max,     "Michelson contrast at zone bottom (max)")
        hdr["TGT_SKY"]  = (sky_adu,          "DC sky level ADU")
        hdr["TGT_CLEN"] = (is_clean,         "True=clean reference image")
        if params.get("apply_psf") and not is_clean:
            hdr["TGT_FWHM"] = (float(params.get("fwhm_px",      0.0)),  "PSF FWHM px")
            hdr["TGT_BETA"] = (float(params.get("moffat_beta", 4.77)),  "Moffat beta")
        if params.get("apply_noise") and not is_clean:
            hdr["TGT_RN"]   = (float(params.get("read_noise_adu", 10.0)), "read noise ADU")
        hdr["TGT_NSED"] = (int(params.get("seed", 42)), "noise RNG seed")
        # Per-zone frequency and waveform type for future per-zone analysis
        for (row, col), (waveform, freq) in _ZONE_SPEC.items():
            hdr[f"TGT_{row}{col}F"] = (freq,           f"r{row}c{col} freq c/px")
            hdr[f"TGT_{row}{col}W"] = (waveform[:8],   f"r{row}c{col} waveform")

        hdu  = fits.PrimaryHDU(image, header=hdr)
        hdul = fits.HDUList([hdu])

        tag  = "clean" if is_clean else "degraded"
        stem = f"target_Cmin{contrast_min:.2f}_Cmax{contrast_max:.2f}_SKY{sky_adu:.0f}"
        if params.get("apply_psf") and not is_clean:
            stem += f"_PSF{params.get('fwhm_px', 0.0):.1f}px"
        if params.get("apply_noise") and not is_clean:
            stem += f"_RN{params.get('read_noise_adu', 10.0):.0f}"
        out_path = Path(params["output_dir"]) / f"{stem}_{tag}.fits"
        hdul.writeto(str(out_path), overwrite=True)
        return str(out_path)
