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

# Fallback star FWHM for the source mask's point-source tier when no better hint
# is available (an AstroImage built outside the GUI, e.g. by a test or a tool).
SOURCEMASK_DEFAULT_FWHM_PX = 4.0


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
        # The UNMASKED Background2D. Kept under this name and with this meaning
        # even after source-masking is adopted: stage 1's sky scaffold requires an
        # unmasked mesh by design, Section 3f reads box_size/mesh from it, and
        # Section 3g's "before" column is the comparison that justifies the change.
        self.background: Background2D | None = None
        # The noise floor actually used everywhere (SNR, star detection, Section 8).
        # The masked pass's RMS map once source-masking succeeds, otherwise the
        # unmasked one -- removing nebulosity lowers the measured noise as well as
        # the level, so pairing a masked background with an unmasked RMS would mix
        # two different estimates of the same sky.
        self.background_rms: np.ndarray | None = None
        # The sky model actually subtracted. Its own attribute rather than
        # `background.background`, because the adopted model is a fitted polynomial
        # surface, not a Background2D product. Feeding the source mask straight into
        # Background2D(mask=...) instead is the known-bad configuration: at high
        # coverage most cells are excluded and photutils back-fills them by
        # interpolation, which measured worse than the bias being removed.
        self.background_model: np.ndarray | None = None
        self.source_mask_result = None      # analysis.source_mask.MaskedBackgroundResult | None
        self.is_color: bool = False                    # True when RGB file was converted to luminance
        self.starless_image: AstroImage | None = None  # Set by ImagePanel when starless is loaded
        # Normalised polygon dicts the user drew in the Background Regions dialog,
        # populated by AnalysisThread from the settings. Kept alongside the
        # rasterised mask below because the report describes the regions (how many,
        # what fraction) while estimate_background() needs the pixels.
        self.bg_exclusion_regions: list[dict] = []
        # Full-resolution bool mask of those regions, True where excluded. Set from
        # outside via set_background_exclusion_mask() rather than rasterised here:
        # analysis.inspector_regions.exclusion_mask reaches analysis.image_filters,
        # which imports this module, so importing it would close a cycle. Same
        # externally-populated-slot convention as `starless_image`.
        self.bg_exclusion_mask: np.ndarray | None = None
        # Whether estimate_background() runs the Section 3g source-masked pass and
        # adopts its result. Set by AnalysisThread from the control panel.
        self.use_source_masked_background: bool = True
        # Star FWHM for the source mask's point-source tier. Only a hint: PSF
        # analysis cannot supply a measured value because it needs the background
        # first, so AnalysisThread derives one from the reference-seeing setting.
        self.psf_fwhm_hint_px: float = SOURCEMASK_DEFAULT_FWHM_PX

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

        # Image dimensions — from the loaded array, not the header, so it's
        # always available regardless of what the file's keywords contain.
        if self.data is not None:
            h, w = self.data.shape[:2]
            self.meta["Image dimensions"] = f"{w} × {h} px"

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

        # Frames combined — present only for an integrated/stacked image. No
        # single de facto keyword across stacking tools, so try the IRAF-derived
        # convention (NCOMBINE, widely honoured) alongside common tool-specific
        # keywords (Siril's STACKCNT) in priority order.
        for kw in ("NCOMBINE", "STACKCNT", "NSTACKED", "NFRAMES", "NIMAGES"):
            if kw in self.header:
                try:
                    n = int(float(self.header[kw]))
                    if n > 0:
                        self.meta["Frames combined"] = str(n)
                        break
                except (TypeError, ValueError):
                    pass

        # Camera's native ADC bit depth — distinct from "Bit depth" above, which
        # is the file's own storage dtype and is commonly 32-bit float after
        # calibration/stacking regardless of what the sensor originally captured.
        for kw in ("BITDEPTH", "CAMBITS", "ADCBITS", "CCDBITS"):
            if kw in self.header:
                try:
                    bits = int(float(self.header[kw]))
                    if bits > 0:
                        self.meta["Camera bit depth"] = f"{bits}-bit"
                        break
                except (TypeError, ValueError):
                    pass

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

    def set_background_exclusion_mask(self, mask: np.ndarray | None) -> None:
        """Attach the user's hand-drawn exclusion mask.

        Must be called BEFORE estimate_background(). It invalidates any background
        already computed, because the mask changes the answer and the idempotency
        guard below would otherwise hand back a background estimated without it --
        a silent wrong result rather than an error.

        Rasterisation happens in the caller (analysis.inspector_regions.
        exclusion_mask); importing that here would close a cycle back through
        analysis.image_filters.
        """
        self.bg_exclusion_mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.background = None
        self.background_model = None
        self.background_rms = None
        self.source_mask_result = None

    def estimate_background(self, box_size: int = 64) -> None:
        """Estimate the sky background, in two phases.

        Phase 1 is the plain per-cell-sigma-clipped Background2D, kept on
        `self.background` because later stages and Section 3f/3g all need the
        unmasked mesh. Phase 2 re-estimates it with stars and extended nebulosity
        masked out (analysis/source_mask.py) and adopts that result as the model
        actually subtracted, since the phase-1 clip assumes every cell is
        sky-dominated and biases upward wherever that fails.

        Falls back to phase 1 silently whenever phase 2 cannot produce a surface
        (too little sky left, too few surviving cells); `source_mask_result
        .fallback_reason` carries the explanation for the report.
        """
        if self.data is None:
            raise RuntimeError("Image not loaded")
        if self.background_model is not None:
            return   # already computed; self.data and the mask both fixed by now
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
        # Adopt the unmasked result first, so every exit path below leaves a usable
        # model behind even if phase 2 raises or declines.
        self.background_model = self.background.background
        self.background_rms = self.background.background_rms

        if not self.use_source_masked_background:
            return

        # Imported here, not at module scope: analysis.source_mask is safe to import
        # from core (it reaches only analysis.background_fit + numpy/scipy/photutils)
        # but keeping it local means a caller that never enables source-masking does
        # not pay photutils.segmentation's import cost.
        from analysis.source_mask import source_masked_background

        user = self.bg_exclusion_mask
        res = source_masked_background(
            self.data, box_size=box_size,
            fwhm_px=float(self.psf_fwhm_hint_px),
            user_exclusion=user if (user is not None and user.any()) else None,
            mesh=self.background.background_mesh,
            rms_mesh=self.background.background_rms_mesh)
        self.source_mask_result = res
        if res.ok:
            self.background_model = np.asarray(res.surface, dtype=np.float32)
            if res.rms_map is not None:
                self.background_rms = np.asarray(res.rms_map, dtype=np.float32)

    def background_subtracted(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("Image not loaded")
        if self.background_model is None:
            return self.data.copy()
        return (self.data - self.background_model).astype(np.float32)

    def background_display(self, kind: str = "model",
                            max_dim: int = BACKGROUND_DISPLAY_MAX_DIM) -> np.ndarray:
        """Stride-decimated float32 copy of a Background2D-derived array, for
        report figures / Data Inspector display.

        kind="model"          -> self.background_model      (the sky level actually subtracted)
        kind="rms"            -> self.background_rms        (the noise floor actually used)
        kind="unmasked_model" -> self.background.background (phase 1, for Section 3g's
                                 before/after comparison)

        Stride decimation (not the LANCZOS resize used for photographic display
        images) so the smoothly-varying field keeps its exact computed values.
        """
        if self.background is None or self.background_model is None:
            raise RuntimeError("Background not estimated; call estimate_background() first")
        if kind == "model":
            arr = self.background_model
        elif kind == "rms":
            arr = self.background_rms
        elif kind == "unmasked_model":
            arr = self.background.background
        else:
            raise ValueError(
                f"unknown kind {kind!r}; expected 'model', 'rms' or 'unmasked_model'")
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
