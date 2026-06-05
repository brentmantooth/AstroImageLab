"""Synthetic astronomical image generator for AstroImageLab metric validation.

All aberrations are modelled as kernel-based PSF modifications applied per-star at
each field position.  This is a deliberate approximation (not full wavefront optics)
intended to produce realistic-looking images whose metrics can be compared against
known input parameters.  The info dialog in the GUI explains each model's limitations.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve

from synthetic.cameras import CAMERAS

# ---------------------------------------------------------------------------
# Photometric reference
# ---------------------------------------------------------------------------
# A magnitude-0 star at f/5, 1-second exposure, with a fiducial 100% QE
# detector produces REF_FLUX_E electrons.  All other star brightnesses are
# scaled from this reference.  The exact value is arbitrary — what matters is
# the relative scale between magnitude values and a plausible ADU range.
_REF_FLUX_E = 2_000_000.0
_REF_FRATIO = 5.0

# Bortle class → approximate sky surface brightness (mag / arcsec²)
_BORTLE_SURFACE_MAG = {
    1: 22.0, 2: 21.5, 3: 21.0, 4: 20.5, 5: 19.5,
    6: 18.5, 7: 18.0, 8: 17.5, 9: 17.0,
}

# Nebula patch placement centres as (fx, fy) fractions of (width, height)
_NEBULA_CENTRES = [
    (0.50, 0.50),                          # image centre
    (0.15, 0.15), (0.85, 0.15),           # top corners
    (0.15, 0.85), (0.85, 0.85),           # bottom corners
]


# ---------------------------------------------------------------------------
# Low-level kernel helpers
# ---------------------------------------------------------------------------

def _gaussian_2d(size: int, sigma_x: float, sigma_y: float | None = None,
                 cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Unnormalised 2D Gaussian on a (size × size) grid.

    cx, cy: centre offset from stamp centre in pixels (positive = right/down).
    """
    if sigma_y is None:
        sigma_y = sigma_x
    half = size // 2
    y, x = np.mgrid[-half: half + 1, -half: half + 1]
    x = x - cx
    y = y - cy
    g = np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
    return g[:size, :size]


def _moffat_2d(size: int, alpha: float, beta: float = 4.0,
               cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Unnormalised 2D Moffat profile on a (size × size) grid.

    I(r) = (1 + r²/α²)^(-β)
    FWHM = 2α × sqrt(2^(1/β) − 1)
    α from FWHM: α = FWHM / (2 × sqrt(2^(1/β) − 1))

    cx, cy: centre offset in pixels (positive = right/down).
    """
    half = size // 2
    y, x = np.mgrid[-half: half + 1, -half: half + 1]
    x = (x - cx).astype(np.float64)
    y = (y - cy).astype(np.float64)
    r2 = (x ** 2 + y ** 2) / (alpha ** 2)
    g = (1.0 + r2) ** (-beta)
    return g[:size, :size]


def _fwhm_to_alpha(fwhm_px: float, beta: float = 4.0) -> float:
    """Convert FWHM in pixels to Moffat α parameter."""
    return fwhm_px / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))


def _ring_kernel(radius_px: float, stamp_size: int, thickness: float = 2.0) -> np.ndarray:
    """Thin annulus kernel for out-of-focus / defocus simulation."""
    half = stamp_size // 2
    y, x = np.mgrid[-half: half + 1, -half: half + 1]
    r = np.sqrt(x ** 2 + y ** 2)
    k = np.exp(-0.5 * ((r - radius_px) / thickness) ** 2)
    k = k[:stamp_size, :stamp_size]
    total = k.sum()
    return k / total if total > 0 else k


def _motion_blur_kernel(length_px: float, angle_deg: float,
                        stamp_size: int) -> np.ndarray:
    """Linear motion-blur kernel of given length and angle."""
    k = np.zeros((stamp_size, stamp_size), dtype=np.float64)
    half = stamp_size // 2
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    n = max(1, int(length_px))
    for i in range(-n // 2, n // 2 + 1):
        xi = int(half + i * dx + 0.5)
        yi = int(half + i * dy + 0.5)
        if 0 <= xi < stamp_size and 0 <= yi < stamp_size:
            k[yi, xi] = 1.0
    total = k.sum()
    return k / total if total > 0 else k


def _siemens_star(size_px: int, n_sectors: int = 36) -> np.ndarray:
    """Sinusoidal Siemens-star resolution test pattern on a (size_px × size_px) grid.

    Radial sectors alternate smoothly between 0 and 1.  Tapered to zero at the
    central singularity and at the edge so it stamps cleanly into an image.
    """
    half = size_px // 2
    y, x = (v.astype(np.float64) for v in
             np.mgrid[-half: half + 1, -half: half + 1])
    angle = np.arctan2(y, x)
    pattern = 0.5 + 0.5 * np.sin(n_sectors * angle)
    r_norm = np.sqrt(x ** 2 + y ** 2) / max(half, 1)
    # taper: zero at centre singularity and at edge
    taper = np.clip(r_norm * 6, 0.0, 1.0) * np.clip(1.0 - r_norm, 0.0, 1.0) ** 2
    return (pattern * taper)[:size_px, :size_px]


# ---------------------------------------------------------------------------
# Per-star PSF builder
# ---------------------------------------------------------------------------

def _star_psf(params: dict, nx: float, ny: float,
              plate_scale: float, stamp_size: int) -> np.ndarray:
    """Build a normalised PSF stamp for one star at normalised field position (nx, ny).

    nx, ny ∈ [-1, 1] where (0, 0) is the image centre.
    plate_scale: arcsec per pixel.
    Returns a float64 array of shape (stamp_size, stamp_size) that sums to 1.
    """
    r = math.sqrt(nx ** 2 + ny ** 2)   # 0 (centre) … ~1.41 (corner)
    theta = math.atan2(ny, nx)

    # ── Base FWHM: user's seeing/star size (minimum 4.7 px so stars are
    #    visible in the app's downsampled display panel)  ─────────────────
    MOFFAT_BETA = float(params.get("moffat_beta", 4.77))
    fwhm_px = max(params["fwhm_arcsec"] / plate_scale, 4.7)

    # ── Field curvature: FWHM grows with r² ─────────────────────────────
    fc = params["field_curvature"]
    fwhm_fc = fwhm_px * (1.0 + fc * r ** 2 * 2.0)

    # ── Astigmatism: elliptical PSF with orientation varying with angle ──
    ast = params["astigmatism"]
    if ast > 0:
        twist = ast * r * 1.5
        fwhm_x = fwhm_fc * (1.0 + twist * abs(math.cos(2 * theta)))
        fwhm_y = fwhm_fc * (1.0 + twist * abs(math.sin(2 * theta)))
        if math.cos(2 * theta) < 0:
            fwhm_x, fwhm_y = fwhm_y, fwhm_x
    else:
        fwhm_x = fwhm_y = fwhm_fc

    fwhm_x = max(fwhm_x, 4.7)
    fwhm_y = max(fwhm_y, 4.7)

    # ── Core Moffat stamp (matches the app's Moffat PSF fitting) ─────────
    alpha_x = _fwhm_to_alpha(fwhm_x, MOFFAT_BETA)
    alpha_y = _fwhm_to_alpha(fwhm_y, MOFFAT_BETA)
    # Approximate elliptical Moffat by scaling the radial coordinate
    half = stamp_size // 2
    y_g, x_g = np.mgrid[-half: half + 1, -half: half + 1]
    r2_ell = (x_g / alpha_x) ** 2 + (y_g / alpha_y) ** 2
    stamp = (1.0 + r2_ell).astype(np.float64) ** (-MOFFAT_BETA)
    stamp = stamp[:stamp_size, :stamp_size]

    # use alpha_x for secondary lobe sizing (average alpha)
    alpha_avg = (alpha_x + alpha_y) / 2.0
    sigma_equiv = alpha_avg  # Gaussian sigma ≈ Moffat alpha for secondary lobes

    # ── Spherical aberration: blend with wide Moffat halo ────────────────
    sph = params["spherical"]
    if sph > 0:
        wide_alpha = _fwhm_to_alpha(fwhm_fc * 3.5, MOFFAT_BETA)
        r2_wide = (x_g ** 2 + y_g ** 2) / (wide_alpha ** 2)
        wide = (1.0 + r2_wide[:stamp_size, :stamp_size]).astype(np.float64) ** (-MOFFAT_BETA)
        wide /= (wide.sum() + 1e-12)
        core_weight = 1.0 - sph * 0.65
        stamp = core_weight * stamp + sph * 0.65 * wide

    # ── Halo: very wide Gaussian (filter-reflection / scatter artefact) ──
    halo_str = params["halo"]
    if halo_str > 0:
        halo_sigma = max(halo_str * 55.0, 3.0)
        halo_kern = _gaussian_2d(stamp_size, halo_sigma)
        s = halo_kern.sum()
        if s > 0:
            halo_kern /= s
        stamp = stamp + halo_str * 0.08 * halo_kern

    # ── Coma: radially-offset secondary lobe, magnitude ∝ r ─────────────
    coma = params["coma"]
    if coma > 0 and r > 0.05:
        offset_px = coma * r * 10.0
        ox = offset_px * math.cos(theta)
        oy = offset_px * math.sin(theta)
        secondary = _gaussian_2d(stamp_size, sigma_equiv * 1.4, cx=ox, cy=oy)
        stamp = stamp + coma * 0.45 * secondary

    # ── Collimation: fixed-direction lobe (independent of field position) ─
    coll = params["collimation"]
    if coll > 0:
        coll_offset = coll * 7.0
        secondary = _gaussian_2d(stamp_size, sigma_equiv * 1.4, cx=coll_offset, cy=0.0)
        stamp = stamp + coll * 0.35 * secondary

    # ── Defocus: poor_focus + magnitude of backfocus ─────────────────────
    defocus = params["poor_focus"] + abs(params["backfocus"])
    if defocus > 0:
        ring_r = defocus * 12.0
        ring_r = max(ring_r, 1.0)
        ring = _ring_kernel(ring_r, stamp_size, thickness=max(2.0, ring_r * 0.25))
        # Blend: light defocus preserves core; severe defocus → full ring
        blend = min(defocus, 1.0)
        stamp = (1.0 - blend) * stamp + blend * ring

    # ── Normalise ─────────────────────────────────────────────────────────
    total = stamp.sum()
    if total > 0:
        stamp /= total
    return stamp.astype(np.float64)


# ---------------------------------------------------------------------------
# Photometric helpers
# ---------------------------------------------------------------------------

def _mag_to_electrons(mag: float, fratio: float, exposure_s: float) -> float:
    """Convert AB magnitude to electrons (approximate, for relative calibration)."""
    flux_ratio = 10.0 ** (-0.4 * mag)
    fratio_corr = (_REF_FRATIO / fratio) ** 2
    return _REF_FLUX_E * flux_ratio * fratio_corr * exposure_s


def _bortle_to_sky_adu(bortle: int, plate_scale: float,
                       gain_e_per_adu: float, exposure_s: float,
                       fratio: float) -> float:
    """Return sky background in ADU per pixel for the given Bortle class."""
    surf_mag = _BORTLE_SURFACE_MAG.get(int(bortle), 19.5)
    # Flux per arcsec² relative to mag-0 star
    flux_per_arcsec2 = 10.0 ** (-0.4 * surf_mag)
    # Scale to pixels
    flux_per_px = flux_per_arcsec2 * plate_scale ** 2
    electrons = _REF_FLUX_E * flux_per_px * ((_REF_FRATIO / fratio) ** 2) * exposure_s
    return electrons / gain_e_per_adu


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    """Generate a synthetic FITS image from a parameter dict."""

    def generate(self, params: dict,
                 preview: bool = False) -> tuple[str, str] | np.ndarray:
        """Generate the image.

        preview=True returns a numpy array (full_image, for live preview).
        Non-preview mode saves two FITS files and returns (main_path, starless_path).
        """
        cam = CAMERAS[params["camera"]]
        focal_mm = float(params["focal_length_mm"])
        aperture_mm = float(params["aperture_mm"])
        fratio = focal_mm / aperture_mm if aperture_mm > 0 else 5.0
        pixel_um = cam["pixel_um"]
        # arcsec / px — unchanged between preview and full-res
        plate_scale = 206.265 * pixel_um / focal_mm

        if preview:
            # 4× camera downsample — same star count, faithful scene representation
            width  = max(64, cam["width_px"]  // 4)
            height = max(64, cam["height_px"] // 4)
        else:
            width  = cam["width_px"]
            height = cam["height_px"]
        n_stars = int(params["n_stars"])

        # Star positions and magnitudes: seeded from n_stars so identical across
        # all runs sharing the same count, regardless of other parameter changes.
        star_rng  = np.random.default_rng(int(n_stars))
        # Noise (guiding angle, sky Poisson, read noise): independent seed.
        noise_rng = np.random.default_rng(params.get("seed", 42))

        # ── Common geometry ────────────────────────────────────────────────
        exposure_s      = float(params["exposure_s"])
        gain_e_per_adu  = float(params["gain_e_per_adu"])
        full_well_adu   = cam["full_well_e"] / gain_e_per_adu
        cx_img          = width  / 2.0
        cy_img          = height / 2.0
        # Normalise field position to [-1, 1] using half-diagonal
        half_diag = math.sqrt(cx_img ** 2 + cy_img ** 2)

        # ── Sky background level ───────────────────────────────────────────
        sky_adu = _bortle_to_sky_adu(
            int(params["bortle"]), plate_scale, gain_e_per_adu, exposure_s, fratio
        )

        # ── Nebula layer (Siemens-star patches, PSF-convolved per position) ─
        nebula_layer = np.zeros((height, width), dtype=np.float64)
        if params.get("nebula_enabled"):
            nebula_size_frac = float(params.get("nebula_size", 0.15))
            nebula_size_px   = max(11, int(nebula_size_frac * min(width, height)))
            if nebula_size_px % 2 == 0:
                nebula_size_px += 1
            nebula_peak_adu  = sky_adu * float(params.get("nebula_brightness", 1.0))
            # Halo excluded — halos are a point-source optical effect only
            params_no_halo   = dict(params)
            params_no_halo["halo"] = 0.0
            for fx, fy in _NEBULA_CENTRES:
                cx_neb = int(fx * width)
                cy_neb = int(fy * height)
                nx = (cx_neb - cx_img) / half_diag
                ny = (cy_neb - cy_img) / half_diag
                # Same field-position-dependent PSF as stars (coma, astigmatism, etc.)
                psf = _star_psf(params_no_halo, nx, ny, plate_scale, 61)
                patch = _siemens_star(nebula_size_px)
                convolved = fftconvolve(patch, psf, mode="same").astype(np.float64)
                peak = convolved.max()
                if peak > 0:
                    convolved *= nebula_peak_adu / peak
                half_neb = nebula_size_px // 2
                x0 = cx_neb - half_neb;  y0 = cy_neb - half_neb
                x1 = x0 + nebula_size_px; y1 = y0 + nebula_size_px
                px0 = max(0, x0);  py0 = max(0, y0)
                px1 = min(width, x1); py1 = min(height, y1)
                kx0 = px0 - x0;  ky0 = py0 - y0
                kx1 = kx0 + (px1 - px0); ky1 = ky0 + (py1 - py0)
                if px1 > px0 and py1 > py0:
                    nebula_layer[py0:py1, px0:px1] += convolved[ky0:ky1, kx0:kx1]

        # ── Star positions ─────────────────────────────────────────────────
        n_edge  = max(1, n_stars // 5)
        n_inner = n_stars - n_edge
        xs_inner = star_rng.uniform(0.05 * width, 0.95 * width, n_inner)
        ys_inner = star_rng.uniform(0.05 * height, 0.95 * height, n_inner)
        margin  = 0.10
        xs_edge = star_rng.uniform(0, width, n_edge)
        ys_edge = star_rng.choice(
            np.concatenate([
                star_rng.uniform(0, margin * height, n_edge // 2 + 1),
                star_rng.uniform((1 - margin) * height, height, n_edge // 2 + 1),
            ]),
            size=n_edge, replace=False,
        )
        star_xs = np.concatenate([xs_inner, xs_edge])
        star_ys = np.concatenate([ys_inner, ys_edge])

        # ── Star magnitudes ────────────────────────────────────────────────
        mags = star_rng.uniform(float(params["mag_min"]), float(params["mag_max"]), n_stars)

        # ── Star layer ────────────────────────────────────────────────────
        halo_str   = float(params["halo"])
        stamp_size = int(min(max(61, halo_str * 220 + 61), 301))
        if stamp_size % 2 == 0:
            stamp_size += 1
        half_stamp = stamp_size // 2

        star_layer = np.zeros((height, width), dtype=np.float64)
        for i in range(n_stars):
            sx, sy = star_xs[i], star_ys[i]
            nx = (sx - cx_img) / half_diag
            ny = (sy - cy_img) / half_diag
            psf     = _star_psf(params, nx, ny, plate_scale, stamp_size)
            star_e  = _mag_to_electrons(mags[i], fratio, exposure_s)
            star_adu = star_e / gain_e_per_adu
            x0 = int(sx) - half_stamp;  y0 = int(sy) - half_stamp
            x1, y1 = x0 + stamp_size, y0 + stamp_size
            px0 = max(0, x0);  py0 = max(0, y0)
            px1 = min(width, x1); py1 = min(height, y1)
            kx0 = px0 - x0;  ky0 = py0 - y0
            kx1 = kx0 + (px1 - px0); ky1 = ky0 + (py1 - py0)
            if px1 > px0 and py1 > py0:
                star_layer[py0:py1, px0:px1] += star_adu * psf[ky0:ky1, kx0:kx1]

        # ── Compose: full image = stars + nebula; starless = nebula only ───
        full_image   = star_layer + nebula_layer
        starless_arr = nebula_layer.copy()

        # ── Guiding error: applied to full image only (stars trail) ────────
        guiding = float(params["guiding"])
        if guiding > 0:
            trail_px  = guiding * 20.0
            angle_deg = float(noise_rng.uniform(0, 360))
            k_size    = max(3, int(trail_px * 2 + 3) | 1)
            k = _motion_blur_kernel(trail_px, angle_deg, k_size)
            full_image = convolve(full_image, k, mode="reflect")

        # ── Sky + read noise — added independently to each image ────────────
        read_noise_adu = cam["read_noise_e"] / gain_e_per_adu
        sky_e          = sky_adu * gain_e_per_adu
        for arr in (full_image, starless_arr):
            sky_noise = noise_rng.poisson(max(0.1, sky_e), size=arr.shape).astype(float)
            arr += sky_noise / gain_e_per_adu
            arr += noise_rng.normal(0.0, read_noise_adu, size=arr.shape)

        full_image   = np.clip(full_image,   0.0, full_well_adu).astype(np.float32)
        starless_arr = np.clip(starless_arr, 0.0, full_well_adu).astype(np.float32)

        if preview:
            return full_image

        # ── FITS output ────────────────────────────────────────────────────
        main_path = self._write_fits(full_image, params, cam, plate_scale, fratio,
                                     sky_adu, mags)
        starless_path = (Path(main_path).parent /
                         (Path(main_path).stem + "_starless.fits"))
        self._write_fits(starless_arr, params, cam, plate_scale, fratio,
                         sky_adu, mags,
                         out_path_override=starless_path,
                         extra_keywords={"SYN_STRL": (True, "starless companion")})
        return (main_path, str(starless_path))

    # ------------------------------------------------------------------
    def _write_fits(self, image: np.ndarray, params: dict, cam: dict,
                    plate_scale: float, fratio: float,
                    sky_adu: float, mags: np.ndarray, *,
                    out_path_override=None,
                    extra_keywords: dict | None = None) -> str:
        from astropy.io import fits

        hdr = fits.Header()
        hdr["SIMPLE"]   = True
        hdr["BITPIX"]   = -32
        hdr["NAXIS"]    = 2
        hdr["NAXIS1"]   = image.shape[1]
        hdr["NAXIS2"]   = image.shape[0]
        hdr["INSTRUME"] = "Synthetic"
        hdr["TELESCOP"] = (f"FL{params['focal_length_mm']:.0f}mm "
                           f"f/{fratio:.1f}")
        hdr["FOCALLEN"] = (float(params["focal_length_mm"]), "mm")
        hdr["APTDIA"]   = (float(params["aperture_mm"]), "mm")
        hdr["FOCRATIO"] = (round(fratio, 2), "f/ratio")
        hdr["XPIXSZ"]   = (cam["pixel_um"], "micron")
        hdr["EGAIN"]    = (float(params["gain_e_per_adu"]), "e-/ADU")
        hdr["GAIN"]     = (float(params["gain_e_per_adu"]), "e-/ADU")
        hdr["EXPTIME"]  = (float(params["exposure_s"]), "seconds")
        hdr["CCD-TEMP"] = (-20.0, "C simulated")
        hdr["SET-TEMP"] = (-20.0, "C simulated")
        # Synthetic-specific keywords for traceability
        hdr["SYN_FWHM"] = (float(params["fwhm_arcsec"]),           "arcsec target FWHM")
        hdr["SYN_BETA"] = (float(params.get("moffat_beta", 4.77)), "Moffat beta")
        hdr["SYN_STAR"] = (int(params["n_stars"]),                 "stars generated")
        hdr["SYN_BRTL"] = (int(params["bortle"]),                  "Bortle class")
        hdr["SYN_COMA"] = (float(params["coma"]),            "coma slider 0-1")
        hdr["SYN_FC"]   = (float(params["field_curvature"]), "field curv slider 0-1")
        hdr["SYN_AST"]  = (float(params["astigmatism"]),     "astigmatism slider 0-1")
        hdr["SYN_SPH"]  = (float(params["spherical"]),       "spherical slider 0-1")
        hdr["SYN_COLL"] = (float(params["collimation"]),     "collimation slider 0-1")
        hdr["SYN_BF"]   = (float(params["backfocus"]),       "backfocus slider -1 to 1")
        hdr["SYN_GUID"] = (float(params["guiding"]),         "guiding slider 0-1")
        hdr["SYN_FOC"]  = (float(params["poor_focus"]),      "poor focus slider 0-1")
        hdr["SYN_HALO"] = (float(params["halo"]),            "halo slider 0-1")
        hdr["SYN_SKYA"] = (round(sky_adu, 4),               "sky background ADU/px")
        hdr["SYN_MAGN"] = (f"{mags.min():.1f}-{mags.max():.1f}", "mag range")
        hdr["SYN_SSED"] = (int(params["n_stars"]),           "star position seed (= n_stars)")
        hdr["SYN_NSED"] = (int(params.get("seed", 42)),      "noise seed")

        if extra_keywords:
            for key, val in extra_keywords.items():
                hdr[key] = val

        hdu = fits.PrimaryHDU(image, header=hdr)
        hdul = fits.HDUList([hdu])

        if out_path_override is not None:
            out_path = Path(out_path_override)
        else:
            # ── Auto-generate filename from params ─────────────────────────
            cam_short = re.sub(r"[^A-Za-z0-9]", "",
                               params["camera"].split("—")[-1].strip())
            fname = (
                f"synth_{cam_short}"
                f"_N{params['n_stars']}"
                f"_FWHM{params['fwhm_arcsec']:.1f}"
                f"_B{params['bortle']}"
            )
            for key, tag in [
                ("coma", "C"), ("field_curvature", "FC"),
                ("astigmatism", "AS"), ("spherical", "SP"), ("collimation", "CO"),
                ("guiding", "G"), ("poor_focus", "PF"), ("halo", "H"),
            ]:
                v = float(params[key])
                if v > 0.01:
                    fname += f"_{tag}{v:.2f}"
            bf = float(params["backfocus"])
            if abs(bf) > 0.01:
                fname += f"_BF{bf:+.2f}"
            if params.get("nebula_enabled"):
                fname += "_NEB"
            fname += ".fits"
            out_path = Path(params["output_dir"]) / fname

        hdul.writeto(str(out_path), overwrite=True)
        return str(out_path)
