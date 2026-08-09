from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from photutils.background import Background2D, SExtractorBackground, MADStdBackgroundRMS

from core.models import (BACKGROUND_DISPLAY_MAX_DIM, DEFAULT_PIXEL_SCALE,
                          FILTER_THICKNESS_MM)

# Background2D's per-cell outlier rejection. Named constants (rather than
# inline literals in estimate_background()) so analysis/background_mask.py's
# classify_background_pixels() can reproduce the exact same clip when
# recovering a pixel-level accept/reject mask that Background2D itself never
# exposes -- see CLAUDE.md's Background Model diagnostics section.
BACKGROUND_SIGCLIP_SIGMA = 3.0
BACKGROUND_SIGCLIP_MAXITERS = 10


def decimation_step(h: int, w: int, max_dim: int = BACKGROUND_DISPLAY_MAX_DIM) -> int:
    """Stride for downsampling an (h, w) array so its longer side is <= max_dim.

    Shared by AstroImage.background_display() and report_builder.py's Section 3f/3g
    residual/overlay figures so arrays built at different call sites land on the
    exact same pixel grid -- see CLAUDE.md's background-diagnostics note on why
    this one formula is factored out instead of duplicated per file.
    """
    return max(1, max(h, w) // max_dim)


# FITS keywords tried in priority order for pixel scale derivation.
# (keyword, multiplier to arcsec/px, apply_abs) — CDELT1/CD1_1 are in degrees/px
# (abs() strips the sign some WCS conventions use to indicate axis-flip direction);
# PIXSCALE/SCALE are already arcsec/px.
_PIXEL_SCALE_KEYWORDS: list[tuple[str, float, bool]] = [
    ("CDELT1", 3600.0, True),
    ("CD1_1", 3600.0, True),
    ("PIXSCALE", 1.0, False),
    ("SCALE", 1.0, False),
]


def _resolve_gain(header: fits.Header | None) -> float | None:
    """Resolve the physical e-/ADU gain from FITS header keywords.

    EGAIN is preferred: it is the FITS standard for the actual e⁻/ADU conversion
    factor. GAIN is ambiguous — many cameras write the gain mode index (0, 100,
    200 …) there, not the physical conversion factor. Values <= 0 are always
    invalid for e⁻/ADU and are skipped so a zero gain mode index doesn't
    produce bogus 0.0 electron values.
    """
    if header is None:
        return None
    for kw in ("EGAIN", "GAIN", "CCDGAIN", "GAINDB"):
        v = header.get(kw)
        if v is not None:
            try:
                g = float(v)
                if g > 0:
                    return g
            except (TypeError, ValueError):
                pass
    return None

_DTYPE_LABELS: dict[str, str] = {
    "uint8":   "8-bit unsigned int",
    "int16":   "16-bit signed int",
    "uint16":  "16-bit unsigned int",
    "int32":   "32-bit signed int",
    "uint32":  "32-bit unsigned int",
    "float32": "32-bit float",
    "float64": "64-bit float",
}


def _dtype_label(dtype: np.dtype) -> str:
    return _DTYPE_LABELS.get(dtype.name, str(dtype))



class AstroImage:
    """Wraps a single FITS or XISF file for analysis."""

    def __init__(self, path: str, label: str = ""):
        self.path = Path(path)
        self.label = label or self.path.stem
        self.data: np.ndarray | None = None
        self.header: fits.Header | None = None
        self.pixel_scale: float = DEFAULT_PIXEL_SCALE
        self.pixel_scale_is_estimated: bool = False
        self.bandwidth_nm: float | None = None
        self.filter_thickness_mm: float = FILTER_THICKNESS_MM
        self.original_dtype: np.dtype | None = None   # dtype before float32 conversion
        self.background: Background2D | None = None
        self.background_rms: np.ndarray | None = None
        self.is_color: bool = False                    # True when RGB file was converted to luminance
        self.starless_image: AstroImage | None = None  # Set by ImagePanel when starless is loaded

        # Extracted metadata for display
        self.meta: dict = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        suffix = self.path.suffix.lower()
        if suffix in (".fits", ".fit", ".fts"):
            self._load_fits()
        elif suffix == ".xisf":
            self._load_xisf()
        elif suffix in (".tiff", ".tif"):
            self._load_tiff()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        if self.data is not None:
            self.original_dtype = self.data.dtype   # capture before float32 conversion
            self.data = self.data.astype(np.float32)
            self.pixel_scale = self._extract_pixel_scale()
            self.bandwidth_nm = self._extract_bandwidth()
            self._extract_metadata()

    def _load_fits(self) -> None:
        with fits.open(self.path) as hdul:
            for hdu in hdul:
                if hdu.data is None or max(hdu.data.shape) <= 100:
                    continue
                d = hdu.data
                if d.ndim == 2:
                    self.data = d.copy()
                    self.header = hdu.header.copy()
                    return
                if d.ndim == 3:
                    header = hdu.header
                    # Require explicit header evidence before treating as RGB.
                    # Monochrome FITS files can be 3D (e.g. multiple integration planes).
                    colortyp = str(header.get("COLORTYP", "")).upper()
                    is_rgb = ("RGB" in colortyp) or ("BAYERPAT" in header)
                    if is_rgb and d.shape[0] >= 3:
                        self.data = (0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]).copy()
                        self.is_color = True
                    else:
                        self.data = d[0].copy()
                    self.header = header.copy()
                    return
        raise ValueError(f"No valid image data found in {self.path.name}")

    def _load_xisf(self) -> None:
        try:
            import xisf as xisf_lib  # type: ignore
        except ImportError:
            raise ImportError("xisf package not installed. Run: pip install xisf")
        x = xisf_lib.XISF(str(self.path))
        # Check colorSpace metadata before reading pixel data
        is_rgb = False
        try:
            meta_list = x.get_images_metadata()
        except Exception:
            meta_list = []
        if meta_list:
            cs = str(meta_list[0].get("colorSpace", "Gray")).upper()
            is_rgb = "RGB" in cs

        img = x.read_image(0)
        if img is None:
            raise ValueError(f"No image data in {self.path.name}")
        # xisf returns (H, W) or (H, W, C)
        if img.ndim == 3:
            if is_rgb and img.shape[2] >= 3:
                img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
                self.is_color = True
            else:
                img = img[:, :, 0]
        self.data = img   # float32 conversion happens in load() after dtype is captured
        # Build a minimal header-like dict from XISF metadata
        if meta_list:
            self.header = fits.Header()
            raw = meta_list[0]
            # Map common XISF FITSKeyword entries into the header
            fk = raw.get("FITSKeywords", {})
            for key, entries in fk.items():
                if entries:
                    self.header[key] = entries[0].get("value", "")

    def _load_tiff(self) -> None:
        from PIL import Image as _PILImage
        img = _PILImage.open(self.path)
        arr = np.asarray(img).copy()
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                arr = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
                self.is_color = True
            else:
                arr = arr[:, :, 0]
        self.data = arr
        self.header = None  # TIFF carries no FITS keywords

    # ------------------------------------------------------------------
    # Pixel scale
    # ------------------------------------------------------------------

    def _extract_pixel_scale(self) -> float:
        if self.header is None:
            self.pixel_scale_is_estimated = True
            return DEFAULT_PIXEL_SCALE

        for kw, factor, use_abs in _PIXEL_SCALE_KEYWORDS:
            if kw in self.header:
                try:
                    v = float(self.header[kw])
                    return (abs(v) if use_abs else v) * factor
                except (ValueError, TypeError):
                    pass

        # Derive from focal length + pixel size
        if "FOCALLEN" in self.header and "XPIXSZ" in self.header:
            try:
                focallen_mm = float(self.header["FOCALLEN"])
                xpixsz_um = float(self.header["XPIXSZ"])
                if focallen_mm > 0:
                    return (xpixsz_um / focallen_mm) * 206.265
            except (ValueError, TypeError):
                pass

        self.pixel_scale_is_estimated = True
        return DEFAULT_PIXEL_SCALE

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------

    def _extract_bandwidth(self) -> float | None:
        if self.header is None:
            return None
        for kw in ("BANDWID", "FWHM", "BANDWIDTH"):
            if kw in self.header:
                try:
                    return float(self.header[kw])
                except (ValueError, TypeError):
                    pass
        return None

    # ------------------------------------------------------------------
    # Metadata for GUI display
    # ------------------------------------------------------------------

    def _extract_metadata(self) -> None:
        # Bit depth — available regardless of header
        if self.original_dtype is not None:
            self.meta["Bit depth"] = _dtype_label(self.original_dtype)

        if self.header is None:
            return
        mapping = {
            "Telescope":    ["TELESCOP"],
            "Camera":       ["INSTRUME"],
            "Filter":       ["FILTER"],
            "Focal length": ["FOCALLEN"],
            "Pixel size":   ["XPIXSZ"],
            "Exposure":     ["EXPTIME", "EXPOSURE"],
            "Date":         ["DATE-OBS"],
            "Binning":      ["XBINNING"],
            "Bandwidth":    ["BANDWID", "FWHM", "BANDWIDTH"],
        }
        for display_key, keywords in mapping.items():
            for kw in keywords:
                if kw in self.header:
                    self.meta[display_key] = str(self.header[kw]).strip()
                    break

        # Gain — prefer the resolved physical e-/ADU value (EGAIN priority order,
        # see _resolve_gain); fall back to the raw GAIN string (often a camera mode
        # index) only when no keyword resolves to a valid value, so the display
        # still shows *something* for files with only an ambiguous GAIN keyword.
        gain = _resolve_gain(self.header)
        if gain is not None:
            self.meta["Gain"] = f"{gain:.3g} e⁻/ADU"
        elif "GAIN" in self.header:
            self.meta["Gain"] = str(self.header["GAIN"]).strip()

        # Focal ratio — prefer explicit keyword; fall back to FOCALLEN / APTDIA
        fr: float | None = None
        for kw in ("FOCRATIO", "FRATIO", "FNUMBER"):
            if kw in self.header:
                try:
                    fr = float(self.header[kw])
                    break
                except (TypeError, ValueError):
                    pass
        if fr is None:
            fl = self.header.get("FOCALLEN")
            ap = self.header.get("APTDIA")
            if fl is not None and ap is not None:
                try:
                    fl_f, ap_f = float(fl), float(ap)
                    if ap_f > 0:
                        fr = fl_f / ap_f
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        if fr is not None:
            self.meta["Focal ratio"] = f"f/{fr:.1f}"

        # CCD sensor temperature
        for kw in ("CCD-TEMP", "CCDTEMP"):
            if kw in self.header:
                try:
                    self.meta["CCD temperature"] = f"{float(self.header[kw]):.1f} °C"
                    break
                except (TypeError, ValueError):
                    pass

        # Cooling set-point temperature
        for kw in ("SET-TEMP", "SETTEMP"):
            if kw in self.header:
                try:
                    self.meta["Cooling set-point"] = f"{float(self.header[kw]):.1f} °C"
                    break
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # Background estimation
    # ------------------------------------------------------------------

    def estimate_background(self, box_size: int = 64) -> None:
        if self.data is None:
            raise RuntimeError("Image not loaded")
        if self.background is not None:
            return   # already computed for this instance's data; self.data never changes post-load
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=AstropyUserWarning)
            self.background = Background2D(
                self.data,
                box_size=box_size,
                filter_size=3,
                sigma_clip=SigmaClip(sigma=BACKGROUND_SIGCLIP_SIGMA,
                                      maxiters=BACKGROUND_SIGCLIP_MAXITERS),
                bkg_estimator=SExtractorBackground(),
                bkg_rms_estimator=MADStdBackgroundRMS(),
            )
        self.background_rms = self.background.background_rms

    def background_subtracted(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Image not loaded")
        if self.background is None:
            return self.data.copy()
        return (self.data - self.background.background).astype(np.float32)

    def background_display(self, kind: str = "model",
                            max_dim: int = BACKGROUND_DISPLAY_MAX_DIM) -> np.ndarray:
        """Stride-decimated float32 copy of a Background2D-derived array, for
        report figures / Data Inspector display.

        kind="model" -> self.background.background (full-res interpolated sky level)
        kind="rms"   -> self.background_rms         (full-res noise floor)

        Stride decimation (not the LANCZOS resize used for photographic display
        images) so the smoothly-varying field keeps its exact computed values.
        """
        if self.background is None:
            raise RuntimeError("Background not estimated; call estimate_background() first")
        if kind == "model":
            arr = self.background.background
        elif kind == "rms":
            arr = self.background_rms
        else:
            raise ValueError(f"unknown kind {kind!r}; expected 'model' or 'rms'")
        h, w = arr.shape[:2]
        step = decimation_step(h, w, max_dim)
        return arr[::step, ::step].astype(np.float32)

    def saturation_threshold(self) -> float:
        if self.data is None:
            return 65535.0
        if self.header is not None and "DATAMAX" in self.header:
            try:
                return float(self.header["DATAMAX"]) * 0.90
            except (ValueError, TypeError):
                pass
        return float(np.max(self.data)) * 0.90

    def nddata(self) -> NDData:
        """Return NDData with uncertainty plane for photutils PSF tools."""
        bgsub = self.background_subtracted()
        if self.background_rms is not None:
            uncertainty = StdDevUncertainty(self.background_rms)
        else:
            uncertainty = None
        return NDData(bgsub, uncertainty=uncertainty)

    # ------------------------------------------------------------------
    # Display stretch
    # ------------------------------------------------------------------

    def display_image(self, stretch: bool = True) -> np.ndarray:
        """Return uint8 array suitable for Qt display."""
        if self.data is None:
            raise RuntimeError("Image not loaded")
        from core.stretch import normalize_for_display
        return normalize_for_display(self.data, stretch=stretch)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        shape = self.data.shape if self.data is not None else "not loaded"
        return f"AstroImage({self.label!r}, shape={shape}, scale={self.pixel_scale:.3f}\"/px)"
