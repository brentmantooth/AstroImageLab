from __future__ import annotations

import warnings

import numpy as np
from astropy.modeling import fitting
from astropy.modeling.models import Moffat2D
from astropy.utils.exceptions import AstropyUserWarning


def moffat_fwhm(gamma: float, alpha: float) -> float:
    """FWHM from astropy Moffat2D gamma/alpha parameters."""
    return 2.0 * gamma * np.sqrt(2.0 ** (1.0 / alpha) - 1.0)


def fit_moffat2d_core(
    cutout: np.ndarray,
    *,
    alpha_bounds: tuple[float, float],
    gamma_min: float,
    fwhm_bounds: tuple[float, float],
) -> dict | None:
    """Fit a 2D Moffat profile to a star cutout centred in the frame.

    `alpha_bounds`/`gamma_min` constrain the fit itself (TRFLSQFitter supports
    bounded parameters, unlike the legacy LevMarLSQFitter); `fwhm_bounds` is an
    additional post-fit plausibility check on the derived FWHM. Returns
    `{"fwhm", "alpha", "gamma", "peak"}` in pixel units, or `None` if the cutout
    is empty, the fit raises, or the fitted parameters fall outside the
    caller-supplied bounds.
    """
    if cutout.size == 0:
        return None

    cy, cx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
    amp = float(np.max(cutout))
    cx0 = cutout.shape[1] / 2.0
    cy0 = cutout.shape[0] / 2.0

    model = Moffat2D(amplitude=amp, x_0=cx0, y_0=cy0, gamma=2.0, alpha=2.5)
    model.gamma.bounds = (gamma_min, None)
    model.alpha.bounds = alpha_bounds

    fitter = fitting.TRFLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        try:
            fitted = fitter(model, cx, cy, cutout)
        except Exception:
            return None

    gamma = abs(fitted.gamma.value)
    alpha = abs(fitted.alpha.value)
    if not (alpha_bounds[0] <= alpha <= alpha_bounds[1]) or gamma < gamma_min:
        return None
    fwhm = moffat_fwhm(gamma, alpha)
    if not (fwhm_bounds[0] <= fwhm <= fwhm_bounds[1]):
        return None
    return {"fwhm": fwhm, "alpha": alpha, "gamma": gamma, "peak": amp}
