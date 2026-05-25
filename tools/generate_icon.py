"""Generate resources/icon.ico for AstroImageLab.

A/B comparison theme: dark background, split circle, left half = broad blurry
star (Image A), right half = sharp bright star (Image B), white dividing line.

Run once from the repo root:
    python tools/generate_icon.py
"""

from __future__ import annotations

import pathlib
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).parent.parent
OUTPUT = ROOT / "resources" / "icon.ico"


def _gaussian(size: int, cx: float, cy: float, sigma: float,
              peak: float) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    return peak * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))


def make_frame(size: int) -> Image.Image:
    bg = np.array([13, 17, 23], dtype=np.float32)  # #0D1117 dark navy

    # Channels R G B
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:] = bg

    cx = size / 2.0
    cy = size / 2.0

    # Left half — Image A: broad, muted blue-white star
    sigma_a = size * 0.10
    blob_a = _gaussian(size, cx * 0.58, cy, sigma_a, peak=1.0)
    # Muted blue-white colour (#8BBFFF) at 45% brightness
    img[:, :, 0] += blob_a * 0.396 * 255  # R
    img[:, :, 1] += blob_a * 0.497 * 255  # G
    img[:, :, 2] += blob_a * 0.710 * 255  # B

    # Right half — Image B: tight, bright warm-white star
    sigma_b = size * 0.055
    blob_b = _gaussian(size, cx * 1.42, cy, sigma_b, peak=1.0)
    # Warm white (#FFFCE0) at 95% brightness
    img[:, :, 0] += blob_b * 1.000 * 255  # R
    img[:, :, 1] += blob_b * 0.988 * 255  # G
    img[:, :, 2] += blob_b * 0.878 * 255  # B

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Mask to a circle
    alpha = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x - cx)**2 + (y - cy)**2 <= (size / 2 - 1)**2:
                alpha[y, x] = 255

    # Vertical dividing line (2px wide, white)
    half = size // 2
    line_w = max(1, size // 64)
    img[alpha > 0, :] = img[alpha > 0, :]  # ensure circle
    for dx in range(-line_w, line_w + 1):
        col = half + dx
        if 0 <= col < size:
            img[:, col] = [220, 220, 220]

    # Re-apply circle mask on divider too
    rgba = np.dstack([img, alpha])
    return Image.fromarray(rgba, mode="RGBA")


SIZES = [256, 128, 64, 48, 32, 16]

frames = [make_frame(s) for s in SIZES]
frames[0].save(
    OUTPUT,
    format="ICO",
    sizes=[(s, s) for s in SIZES],
    append_images=frames[1:],
)
print(f"Wrote {OUTPUT}")