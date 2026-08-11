"""Ground-truth synthetic frames for background-estimation tests.

Every frame is returned alongside the *exact* sky surface that was used to
build it, so background-recovery tests can assert on measured accuracy
(|recovered - true| < tolerance) rather than on shape/monotonicity properties.

That distinction matters: this codebase has already shipped a metric that was
confidently, monotonically, consistently wrong through a suite that only
checked structural properties (see CLAUDE.md's edge-analyzer rotation note).
A recovered background can be smooth, plausible and badly biased at the same
time, and only a known truth catches it.
"""
from __future__ import annotations

import numpy as np

# Defaults chosen to sit in the regime the Section 3g diagnostic targets:
# a sky level and noise typical of a stacked broadband frame, on an image
# comfortably larger than the 64 px background mesh box.
DEFAULT_SHAPE = (512, 512)
DEFAULT_SKY = 1000.0
DEFAULT_SIGMA = 30.0


def make_bg_frame(*, shape: tuple[int, int] = DEFAULT_SHAPE,
                  sky_level: float = DEFAULT_SKY,
                  sky_sigma: float = DEFAULT_SIGMA,
                  gradient: tuple[float, float] = (0.0, 0.0),
                  vignetting: float = 0.0,
                  nebulae: tuple[tuple[float, float, float, float], ...] = (),
                  nebula_profile: str = "gaussian",
                  n_stars: int = 0,
                  star_peak: tuple[float, float] = (3000.0, 30000.0),
                  star_fwhm_px: float = 4.0,
                  seed: int = 0,
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a frame with a known sky surface.

    gradient       : (dx, dy) in ADU per pixel, added to `sky_level`.
    vignetting     : corner darkening as a fraction of `sky_level`, applied as a
                     radial bowl. Part of the *true sky*, so a correct estimator
                     must reproduce it rather than mask it.
    nebulae        : tuple of (cy, cx, radius_px, amplitude_adu) blobs.
    nebula_profile : "gaussian" (wings extend forever) or "bounded" (compact,
                     reaches exactly zero at `radius`). Both are worth testing:
                     a Gaussian never leaves genuinely blank sky, so masking
                     more always helps it, which flatters any aggressive
                     detector. Real targets look more like "bounded".
    n_stars        : Gaussian point sources at random positions.

    Returns (data, true_sky, truth) where `true_sky` is the noiseless sky
    surface (level + gradient + vignetting, excluding nebulae and stars) and
    `truth` holds the component layers plus convenience masks.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    gx, gy = gradient
    true_sky = sky_level + gx * xx + gy * yy
    if vignetting > 0:
        r2 = ((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) / ((w / 2.0) ** 2 + (h / 2.0) ** 2)
        true_sky = true_sky - vignetting * sky_level * r2

    nebula = np.zeros((h, w), dtype=np.float64)
    for cy, cx, radius, amp in nebulae:
        d = np.hypot(yy - cy, xx - cx)
        if nebula_profile == "bounded":
            nebula += amp * np.clip(1.0 - (d / radius) ** 2, 0.0, None) ** 2
        else:
            nebula += amp * np.exp(-0.5 * (d / radius) ** 2)

    stars = np.zeros((h, w), dtype=np.float64)
    star_positions = np.zeros((0, 2))
    if n_stars > 0:
        sigma = star_fwhm_px / 2.3548200450309493
        margin = max(10.0, 4.0 * sigma)
        # Seeded from n_stars, matching the generator's two-RNG convention:
        # star positions stay reproducible independently of the noise draw.
        star_rng = np.random.default_rng(int(n_stars))
        star_positions = np.stack([
            star_rng.uniform(margin, h - margin, n_stars),
            star_rng.uniform(margin, w - margin, n_stars)], axis=-1)
        peaks = star_rng.uniform(star_peak[0], star_peak[1], n_stars)
        # Stamp into a bounded patch per star. Evaluating exp() over the whole
        # frame for each star is O(n_stars * H * W) and becomes the dominant
        # cost of the entire test at realistic sensor sizes (600 stars on a
        # 24 MP frame is 14 billion operations); the profile beyond 4 sigma is
        # numerically zero anyway.
        half = max(1, int(np.ceil(4.0 * sigma)))
        for (sy, sx), peak in zip(star_positions, peaks):
            y0, y1 = max(0, int(sy) - half), min(h, int(sy) + half + 1)
            x0, x1 = max(0, int(sx) - half), min(w, int(sx) + half + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            d2 = ((yy[y0:y1, x0:x1] - sy) ** 2 + (xx[y0:y1, x0:x1] - sx) ** 2)
            stars[y0:y1, x0:x1] += peak * np.exp(-0.5 * d2 / sigma ** 2)

    noise = rng.normal(0.0, sky_sigma, (h, w))
    data = (true_sky + nebula + stars + noise).astype(np.float32)

    truth = {
        "nebula": nebula,
        "stars": stars,
        "noise": noise,
        "sky_sigma": float(sky_sigma),
        "gradient": (float(gx), float(gy)),
        "vignetting": float(vignetting),
        "star_positions": star_positions,
        # Coverage cut at 1 sigma is the "visible nebulosity" figure quoted in
        # the design notes; the 0.15 sigma cut is the generous extent a source
        # mask should be reaching for.
        "nebula_mask": nebula > sky_sigma,
        "nebula_mask_faint": nebula > 0.15 * sky_sigma,
    }
    return data, true_sky.astype(np.float64), truth


def nebula_coverage(truth: dict, faint: bool = False) -> float:
    """Fraction of the frame covered by nebulosity above the chosen cut."""
    key = "nebula_mask_faint" if faint else "nebula_mask"
    return float(truth[key].mean())


# --- Named frames used across the source-mask tests ------------------------
# Kept here (not in conftest) so a scratch script can import and reproduce the
# exact frames a failing assertion refers to.

def star_only_frame(seed: int = 0):
    """No nebulosity at all — the graceful-degradation case."""
    return make_bg_frame(n_stars=40, seed=seed)


def moderate_nebula_frame(seed: int = 0):
    """~15% coverage at 1 sigma: the regime Background2D still handles tolerably."""
    return make_bg_frame(
        nebulae=((256.0, 256.0, 70.0, 90.0),),
        n_stars=40, seed=seed)


def heavy_nebula_frame(seed: int = 0):
    """>50% coverage at 1 sigma: where the per-cell sigma-clip assumption breaks."""
    return make_bg_frame(
        nebulae=((220.0, 190.0, 110.0, 90.0),
                 (300.0, 320.0, 90.0, 70.0),
                 (140.0, 380.0, 70.0, 60.0)),
        n_stars=40, seed=seed)


def gradient_nebula_frame(seed: int = 0):
    """Heavy nebulosity superimposed on a strong linear sky gradient.

    The gradient is deliberately non-diagonal (dx != dy) so a direction bug
    cannot pass by symmetry.
    """
    return make_bg_frame(
        gradient=(0.4, 0.15),
        nebulae=((220.0, 190.0, 110.0, 90.0),
                 (300.0, 320.0, 90.0, 70.0),
                 (140.0, 380.0, 70.0, 60.0)),
        n_stars=40, seed=seed)


def bounded_nebula_frame(seed: int = 5):
    """Compact, sharply-bounded nebulosity on a gradient, leaving real blank sky.

    Closer to a real target than the Gaussian frames: those have wings that
    never reach zero, so over-masking always *improves* their measured accuracy
    and a too-aggressive detector looks good. This frame penalises it honestly.
    """
    return make_bg_frame(
        gradient=(0.4, 0.15),
        nebulae=((230.0, 230.0, 154.0, 90.0),),
        nebula_profile="bounded", n_stars=40, seed=seed)


def vignetted_frame(seed: int = 5):
    """Radial vignetting, no nebulosity at all.

    Regression guard for a real bug: with a plane-only sky scaffold the
    residual kept the whole bowl, the detector masked it as if it were source,
    and the estimate came out worse than doing nothing while masking >50% of a
    frame containing no extended structure whatsoever.
    """
    return make_bg_frame(vignetting=0.30, n_stars=40, seed=seed)


def vignetted_nebula_frame(seed: int = 5):
    """Vignetting *and* nebulosity — the genuinely ambiguous case.

    Both are radially symmetric with the same curvature sign, so no threshold
    can separate "the corners are dark because of the optics" from "the centre
    is bright because of a nebula". This is the case user-drawn exclusion
    regions exist to resolve.
    """
    return make_bg_frame(
        vignetting=0.30,
        nebulae=((230.0, 230.0, 154.0, 90.0),),
        nebula_profile="bounded", n_stars=40, seed=seed)
