"""Synthetic star-free targets for the spatial-detail metric screening study.

Pure array-in/array-out module: no core.astro_image import, no FITS writing.
Every function here takes/returns plain numpy arrays so it can be unit-tested
without a display and composed freely by tools/spatial_detail_screen.py. This
mirrors analysis/source_mask.py's array-only convention, not the two GUI
generators (synthetic/generator.py, synthetic/target_generator.py), which
must not be modified and are not touched here.

Two things are reused, read-only, from those protected generators without
modifying them: `synthetic.generator._siemens_star` (a free-standing
function with no class-state dependency) for both the pure Siemens-star
target and the composite target's sharp accents, and
`synthetic.cameras.CAMERAS` for physically-plausible gain/read-noise when
simulating shot noise.

Noise is deliberately NOT compensated the way
tools/sensitivity_sweep.py::_blur_preserving_noise compensates it: that
function holds noise *fixed* while blur sweeps, so blur's own noise
suppression does not confound a blur-only measurement. Here noise level is
an independent sweep axis by design -- the whole point is separating how a
metric responds to "less structure" from how it responds to "more noise" --
so compensating for it would reintroduce exactly the coupling this study
exists to keep apart. Blur is applied to a clean signal only; noise is drawn
independently afterward at whatever stack depth the caller asks for.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from synthetic.cameras import CAMERAS, DEFAULT_CAMERA
from synthetic.generator import _siemens_star

CANVAS_PX = 640          # square canvas; comfortably larger than the 17px
                          # entropy kernel / 10px std kernel, small enough to
                          # keep a make_figures=False sweep cell cheap
PEDESTAL_ADU = 6000.0     # a genuinely light-polluted sky floor -- was 300.0,
                          # then 2500.0. This alone raises Poisson shot-noise
                          # sigma by ~4.5x (sqrt(6000/300)) at every stack
                          # depth versus the original value; combined with
                          # apply_camera_noise's lowered gain_e_per_adu
                          # default (1.0 -> 0.25), total noise is substantially
                          # higher than the first two passes at this. The
                          # target peak levels in tools/spatial_detail_screen.py
                          # were lowered to match (lower contrast against this
                          # brighter sky).


def _peak_rescale(gray: np.ndarray, target_peak_adu: float,
                   pedestal_adu: float = PEDESTAL_ADU) -> np.ndarray:
    """Linearly rescale gray's own [min, max] to [pedestal, pedestal+peak].

    This is a plain dynamic-range rescale, not a photometric de-stretch --
    there is no way to recover the true tone curve a source PNG went
    through. It preserves relative spatial structure, which is what a
    blur/noise response study actually probes; absolute SNR read off a
    fractal/zebra-derived target should be treated as illustrative, not
    photometrically exact.
    """
    gray = gray.astype(np.float64)
    lo, hi = float(gray.min()), float(gray.max())
    span = hi - lo
    if span <= 0:
        return np.full(gray.shape, pedestal_adu, dtype=np.float32)
    norm = (gray - lo) / span
    return (pedestal_adu + norm * target_peak_adu).astype(np.float32)


def _center_crop_square(arr: np.ndarray, size_px: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < size_px or w < size_px:
        raise ValueError(f"source {h}x{w} smaller than requested canvas {size_px}px")
    y0 = (h - size_px) // 2
    x0 = (w - size_px) // 2
    return arr[y0:y0 + size_px, x0:x0 + size_px]


def _load_gray_png(path: Path) -> np.ndarray:
    """First channel of a PNG, as float64.

    Every source PNG used here stores R=G=B (confirmed: the fractal
    BigImage_*.png files are RGB, Zebra.png/ZebraSmall.png are RGBA), so
    channel 0 already carries the full grayscale content regardless of mode.
    """
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr[:, :, 0].astype(np.float64)


def load_fractal_source(path: Path, target_peak_adu: float,
                         pedestal_adu: float = PEDESTAL_ADU,
                         canvas_px: int = CANVAS_PX) -> np.ndarray:
    """A fractal-noise PNG (AstroLabTestData/Fractal Source/BigImage_*.png),
    centre-cropped to canvas_px and rescaled to [pedestal, pedestal+peak] ADU.
    Stands in for nebula-like diffuse structure.
    """
    gray = _center_crop_square(_load_gray_png(path), canvas_px)
    return _peak_rescale(gray, target_peak_adu, pedestal_adu)


def load_zebra(path: Path, target_peak_adu: float,
               pedestal_adu: float = PEDESTAL_ADU,
               canvas_px: int = CANVAS_PX) -> np.ndarray:
    """The high-contrast Zebra stripe pattern
    (AstroLabTestData/Fractal Source/Zebra.png), centre-cropped and
    rescaled. A sharp, high-contrast comparison target, unlike the
    smoothly-varying fractal/Siemens targets.
    """
    gray = _center_crop_square(_load_gray_png(path), canvas_px)
    return _peak_rescale(gray, target_peak_adu, pedestal_adu)


def build_siemens_target(peak_adu: float, canvas_px: int = CANVAS_PX,
                          pedestal_adu: float = PEDESTAL_ADU,
                          n_sectors: int = 36) -> np.ndarray:
    """A single Siemens-star resolution target filling the canvas, peaking
    at peak_adu against a pedestal_adu sky floor.

    `_siemens_star` (read-only import from synthetic.generator) tapers to
    exactly zero at the canvas edge, so this target carries a genuine
    flat-sky ring for free -- no separate border logic is needed for
    background estimation.
    """
    star = _siemens_star(canvas_px, n_sectors=n_sectors)
    peak = float(star.max())
    if peak <= 0:
        raise ValueError("degenerate Siemens star pattern")
    return (pedestal_adu + star / peak * peak_adu).astype(np.float32)


def build_composite_target(fractal_base: np.ndarray, n_accents: int,
                            accent_peak_adu: float, accent_size_px: int = 81,
                            seed: int | None = None) -> np.ndarray:
    """A diffuse fractal base with a few sharp Siemens-star-like accents
    stamped on top -- the "extended structure plus a few sharp features"
    shape of a representative nebula-like target.

    fractal_base must already be loaded/rescaled/cropped to the working
    canvas size (see load_fractal_source). Accents are added (not blended)
    on top, since real superimposed sources add flux; `_siemens_star`'s own
    taper to zero at the accent's edges means this never introduces a hard
    rectangular discontinuity against the base.

    Placement uses this project's two-RNG convention: deterministic
    placement seeded from a count by default (mirroring
    synthetic/generator.py's `star_rng = np.random.default_rng(int(n_stars))`),
    independent of whatever noise seed the caller is using that run. Simple
    rejection sampling keeps accents away from the canvas edge and from each
    other.
    """
    canvas = fractal_base.astype(np.float32).copy()
    h, w = canvas.shape
    if accent_size_px % 2 == 0:
        accent_size_px += 1
    half = accent_size_px // 2
    if h <= accent_size_px or w <= accent_size_px:
        raise ValueError("canvas too small for the requested accent size")

    accent = _siemens_star(accent_size_px)
    peak = float(accent.max())
    if peak > 0:
        accent = (accent / peak * accent_peak_adu).astype(np.float32)

    placement_rng = np.random.default_rng(seed if seed is not None else n_accents)
    min_sep = 1.5 * accent_size_px
    max_tries = 200 * max(1, n_accents)
    placed: list[tuple[int, int]] = []
    tries = 0
    while len(placed) < n_accents and tries < max_tries:
        tries += 1
        cx = int(placement_rng.integers(half, w - half))
        cy = int(placement_rng.integers(half, h - half))
        if all((cx - px) ** 2 + (cy - py) ** 2 >= min_sep ** 2 for px, py in placed):
            placed.append((cx, cy))

    for cx, cy in placed:
        y0, y1 = cy - half, cy - half + accent_size_px
        x0, x1 = cx - half, cx - half + accent_size_px
        canvas[y0:y1, x0:x1] += accent

    return canvas


def apply_gaussian_blur(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-blur the clean signal only. Noise is added afterward and
    independently by apply_camera_noise -- see the module docstring for why
    this deliberately does not compensate for the noise the blur removes.
    sigma_px<=0 returns arr unchanged rather than a degenerate near-zero blur.
    """
    if sigma_px <= 0:
        return arr.astype(np.float32)
    return gaussian_filter(arr, sigma_px).astype(np.float32)


def apply_camera_noise(arr_adu: np.ndarray, rng: np.random.Generator,
                        stack_depth: int | None,
                        camera: str = DEFAULT_CAMERA,
                        gain_e_per_adu: float = 0.25) -> np.ndarray:
    """Add camera-realistic shot + read noise at a given equivalent stack
    depth. stack_depth=None returns arr_adu unchanged -- the noiseless
    reference level (kept for diagnostic/CLI-override use; the default
    sweep ladders no longer include it -- see tools/spatial_detail_screen.py).

    Exact, not approximate: the sum of N iid Poisson(lambda) draws is itself
    Poisson(N*lambda), so Poisson(N*lambda)/N is exactly distributed as the
    mean of N independently-acquired subs -- there is no need to actually
    draw and average N frames. Read-noise variance scales the same way
    (sigma / sqrt(N)). The Poisson floor (`np.maximum(0.1, ...)`) mirrors
    synthetic/generator.py's own convention so a literal zero-signal pixel
    never hands rng.poisson a zero lambda.

    gain_e_per_adu default was 1.0, raised noise by lowering it to 0.25 --
    this matches the real ZWO ASI2600MM Pro's own measured EGAIN at gain=100
    (0.243 e-/ADU, read from actual FITS headers in this project's real test
    data). Fewer electrons per ADU means the same nominal ADU signal
    corresponds to fewer photons, so shot noise is a larger fraction of it
    in ADU units -- physically realistic, not an arbitrary noise dial.
    """
    if stack_depth is None:
        return arr_adu.astype(np.float32)
    if stack_depth < 1:
        raise ValueError("stack_depth must be >= 1 or None")
    cam = CAMERAS[camera]
    signal_e = np.maximum(arr_adu, 0.0) * gain_e_per_adu
    shot_e = rng.poisson(np.maximum(0.1, signal_e * stack_depth)) / stack_depth
    read_sigma_e = cam["read_noise_e"] / math.sqrt(stack_depth)
    out_e = shot_e + rng.normal(0.0, read_sigma_e, size=arr_adu.shape)
    return (out_e / gain_e_per_adu).astype(np.float32)
