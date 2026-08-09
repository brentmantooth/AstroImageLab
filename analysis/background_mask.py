"""Reproduces photutils Background2D's per-cell sigma-clip as a pixel-level
accept/reject mask.

Background2D itself never exposes which raw pixels within each mesh cell
survived its internal SigmaClip -- only the resulting per-cell scalar
background/RMS values (background_mesh/background_rms_mesh). This module
re-runs the identical clip (same sigma/maxiters, same box_size cells) purely
to recover that mask, for the Section 3e "which pixels count as background"
diagnostic. See CLAUDE.md's background-diagnostics notes.
"""
from __future__ import annotations

import numpy as np
from astropy.stats import SigmaClip

from core.astro_image import BACKGROUND_SIGCLIP_MAXITERS, BACKGROUND_SIGCLIP_SIGMA


def classify_background_pixels(data: np.ndarray, box_size: int,
                                sigma: float = BACKGROUND_SIGCLIP_SIGMA,
                                maxiters: int = BACKGROUND_SIGCLIP_MAXITERS) -> np.ndarray:
    """Bool array, same shape as `data`. True = kept as background by the
    same per-cell sigma-clip Background2D runs internally on raw pixel
    values (before any background subtraction).

    Cells are box_size x box_size, truncated (not padded) at the image's
    bottom/right edge if the dimensions aren't evenly divisible -- a
    documented approximation of Background2D's own internal edge handling,
    acceptable for a diagnostic pixel classification.

    The uniform-size interior cells are clipped in one vectorized call
    (reshaped into (n_cells, box_size**2), SigmaClip's own axis=1 per-row
    clipping); only the ragged bottom/right-edge cells fall back to a
    per-cell loop, since there are few of them.
    """
    b = int(box_size)
    h, w = data.shape[:2]
    n_rows_full = h // b
    n_cols_full = w // b
    clip = SigmaClip(sigma=sigma, maxiters=maxiters)
    kept = np.zeros((h, w), dtype=bool)

    if n_rows_full > 0 and n_cols_full > 0:
        core = data[:n_rows_full * b, :n_cols_full * b]
        cells = (core.reshape(n_rows_full, b, n_cols_full, b)
                     .transpose(0, 2, 1, 3)
                     .reshape(n_rows_full * n_cols_full, b * b))
        clipped = clip(cells, axis=1, masked=True)
        # nomask (scalar False) when nothing was clipped anywhere -- must use
        # getmaskarray, not .mask directly, or a boolean-indexed assignment
        # below would silently only touch the first row (CLAUDE.md pitfall:
        # "sigma_clip mask is scalar False when nothing is clipped").
        cell_kept = ~np.ma.getmaskarray(clipped)
        kept[:n_rows_full * b, :n_cols_full * b] = (
            cell_kept.reshape(n_rows_full, n_cols_full, b, b)
                     .transpose(0, 2, 1, 3)
                     .reshape(n_rows_full * b, n_cols_full * b)
        )

    for r0 in range(0, h, b):
        r1 = min(r0 + b, h)
        row_ragged = (r1 - r0) < b
        for c0 in range(0, w, b):
            c1 = min(c0 + b, w)
            col_ragged = (c1 - c0) < b
            if not (row_ragged or col_ragged):
                continue   # already handled by the vectorized core above
            cell = data[r0:r1, c0:c1].ravel()
            clipped_cell = clip(cell, masked=True)
            cell_mask = np.ma.getmaskarray(clipped_cell)
            kept[r0:r1, c0:c1] = (~cell_mask).reshape(r1 - r0, c1 - c0)

    return kept
