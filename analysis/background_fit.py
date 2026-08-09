"""Weighted-least-squares polynomial fit to a photutils Background2D mesh.

Quantifies any large-scale gradient (candidate cause: light pollution / moon
glow near the horizon) or curvature (candidate cause: vignetting) present in
an image's background model, with BIC-based model-order selection and a
physically-labeled decomposition (gradient / isotropic curvature / anisotropic
curvature) rather than raw polynomial coefficients. See CLAUDE.md's
background-diagnostics notes for the full statistical design rationale.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats

# Kass & Raftery (1995) evidence thresholds for delta-BIC, most specific first.
_BIC_EVIDENCE_THRESHOLDS = ((10.0, "very strong"), (6.0, "strong"), (2.0, "positive"))

# Fixed polynomial basis for each candidate order. Order matters: it must
# match _design_matrix()'s column order exactly.
_ORDER_TERMS: dict[int, tuple[str, ...]] = {
    0: ("const",),
    1: ("const", "x", "y"),
    2: ("const", "x", "y", "x2", "y2", "xy"),
}
# Power of `scale` each term's normalized-coordinate coefficient must be
# divided by to convert to native ADU/px units (diagonal rescale only --
# valid because normalization centers on the native origin, it doesn't shift
# it, so no cross-term Taylor re-expansion is needed).
_TERM_SCALE_POWER = {"const": 0, "x": 1, "y": 1, "x2": 2, "y2": 2, "xy": 2}


def _design_matrix(dx: np.ndarray, dy: np.ndarray, order: int) -> np.ndarray:
    """Design matrix for `order` (0/1/2) at native-unit offsets (dx, dy)."""
    cols = [np.ones_like(dx)]
    if order >= 1:
        cols += [dx, dy]
    if order >= 2:
        cols += [dx ** 2, dy ** 2, dx * dy]
    return np.stack(cols, axis=-1)


def _pvalue(stat: float, df: float) -> float:
    if not np.isfinite(stat) or df <= 0:
        return float("nan")
    return float(2.0 * _stats.t.sf(abs(stat), df=df))


def _mag_se(v1: float, v2: float, var1: float, var2: float, cov12: float) -> float:
    """Delta-method SE of sqrt(v1**2+v2**2) given Var(v1), Var(v2), Cov(v1,v2)."""
    mag = float(np.hypot(v1, v2))
    if mag <= 0:
        return float("nan")
    var_mag = (v1 ** 2 * var1 + v2 ** 2 * var2 + 2.0 * v1 * v2 * cov12) / mag ** 2
    return float(np.sqrt(max(var_mag, 0.0)))


def _bic_evidence_label(delta_bic: float) -> str:
    for threshold, label in _BIC_EVIDENCE_THRESHOLDS:
        if delta_bic > threshold:
            return label
    return "weak"


def fit_background_surface(mesh: np.ndarray, rms_mesh: np.ndarray,
                            box_size: int, image_shape: tuple[int, int]) -> dict:
    """Fit nested constant/plane/quadric models to a Background2D mesh.

    Fits weighted least squares (inverse-variance weights from rms_mesh) for
    each of 3 nested polynomial orders that have enough valid cells to
    support their parameter count, selects the minimum-BIC order, and
    decomposes the selected model into physically-labeled quantities:
    gradient (linear terms), isotropic curvature (vignetting-like radially
    symmetric term), and anisotropic curvature (tilt/off-center/asymmetric
    curvature). mesh/rms_mesh are Background2D's own low-resolution,
    independently-measured cell grids -- not the full interpolated
    background image, which is not independent-pixel data.

    Returns a dict; see module docstring / CLAUDE.md for full field list.
    `insufficient_data` is True (all other fields None/empty) when fewer
    than 2 mesh cells have finite, positive rms.
    """
    h_img, w_img = image_shape
    n_rows, n_cols = mesh.shape

    valid = np.isfinite(mesh) & np.isfinite(rms_mesh) & (rms_mesh > 0)
    n_valid = int(valid.sum())

    result: dict = {
        "selected_order": None, "bic": {}, "delta_bic": None, "bic_evidence": None,
        "n_cells": n_valid, "r_squared": None, "coefficients": None, "covariance": None,
        "param_names": None, "normalize": None, "box_size": box_size,
        "gradient": None, "isotropic_curvature": None, "anisotropic_curvature": None,
        "insufficient_data": n_valid < 2,
    }
    if result["insufficient_data"]:
        return result

    # Mesh-cell-center coordinates, reconstructed from box_size (Background2D
    # doesn't expose these directly). Approximate at a padded/truncated edge
    # cell -- acceptable for a low-order diagnostic fit.
    yy, xx = np.mgrid[0:n_rows, 0:n_cols]
    x_native = (xx.astype(np.float64) + 0.5) * box_size
    y_native = (yy.astype(np.float64) + 0.5) * box_size

    # Normalize about the frame center for conditioning; back-transformed to
    # native units below via a diagonal rescale.
    cx, cy = w_img / 2.0, h_img / 2.0
    scale = max(h_img, w_img) / 2.0 or 1.0

    x_norm = ((x_native - cx) / scale)[valid]
    y_norm = ((y_native - cy) / scale)[valid]
    # Deliberate float64 exception to CLAUDE.md's "always float32" rule: this
    # matrix solve is tiny (tens-hundreds of mesh cells, never image-scale),
    # so there's no memory/perf cost, and precision matters for the SE/
    # p-value inference downstream -- same justification as the astroalign
    # float64 exception in gui/analysis_thread.py.
    z = mesh[valid].astype(np.float64)
    w = 1.0 / (rms_mesh[valid].astype(np.float64) ** 2)
    sw = np.sqrt(w)

    fits_by_order: dict[int, dict] = {}
    for order, terms in _ORDER_TERMS.items():
        k = len(terms)
        if n_valid <= k:
            continue
        X = _design_matrix(x_norm, y_norm, order)
        Xw = X * sw[:, None]
        yw = z * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = z - X @ beta
        wrss = float(np.sum(w * resid ** 2))
        df = n_valid - k
        sigma_sq = wrss / df
        cov = sigma_sq * np.linalg.pinv(Xw.T @ Xw)
        weighted_mean = float(np.average(z, weights=w))
        wtss = float(np.sum(w * (z - weighted_mean) ** 2))
        r2 = 1.0 - wrss / wtss if wtss > 0 else float("nan")
        bic = n_valid * np.log(max(wrss / n_valid, 1e-300)) + k * np.log(n_valid)
        fits_by_order[order] = {
            "terms": terms, "k": k, "beta": beta, "cov": cov,
            "df": df, "r_squared": r2, "bic": float(bic),
        }

    result["bic"] = {o: f["bic"] for o, f in fits_by_order.items()}
    selected_order = min(fits_by_order, key=lambda o: fits_by_order[o]["bic"])
    bics_sorted = sorted(result["bic"].values())
    delta_bic = (bics_sorted[1] - bics_sorted[0]) if len(bics_sorted) > 1 else float("inf")
    result["selected_order"] = selected_order
    result["delta_bic"] = delta_bic
    result["bic_evidence"] = _bic_evidence_label(delta_bic)

    sel = fits_by_order[selected_order]
    terms = sel["terms"]
    scale_vec = np.array([scale ** (-_TERM_SCALE_POWER[t]) for t in terms])
    coefficients = sel["beta"] * scale_vec
    covariance = sel["cov"] * np.outer(scale_vec, scale_vec)
    df = sel["df"]
    idx = {t: i for i, t in enumerate(terms)}

    result["r_squared"] = sel["r_squared"]
    result["coefficients"] = coefficients
    result["covariance"] = covariance
    result["param_names"] = terms
    result["normalize"] = {"cx": cx, "cy": cy, "scale": scale}

    if "x" in idx and "y" in idx:
        bi, ci = idx["x"], idx["y"]
        b, c = float(coefficients[bi]), float(coefficients[ci])
        var_b, var_c = float(covariance[bi, bi]), float(covariance[ci, ci])
        cov_bc = float(covariance[bi, ci])
        magnitude = float(np.hypot(b, c))
        se = _mag_se(b, c, var_b, var_c, cov_bc)
        p_value = _pvalue(magnitude / se, df) if se > 0 else float("nan")
        # Exact corner-to-corner swing of the fitted plane (max - min over the
        # 4 image corners) -- NOT magnitude * diagonal_length, which
        # overstates the true range whenever the gradient direction isn't
        # aligned with the diagonal (Cauchy-Schwarz).
        half_w, half_h = w_img / 2.0, h_img / 2.0
        corner_vals = [b * sx * half_w + c * sy * half_h
                       for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        adu_range = float(max(corner_vals) - min(corner_vals))
        result["gradient"] = {
            "magnitude": magnitude, "se": se,
            "direction_deg": float(np.degrees(np.arctan2(c, b))),
            "p_value": p_value, "adu_range": adu_range,
        }

    if "x2" in idx and "y2" in idx and "xy" in idx:
        di, ei, fi = idx["x2"], idx["y2"], idx["xy"]
        d, e, f = float(coefficients[di]), float(coefficients[ei]), float(coefficients[fi])
        var_d, var_e, var_f = (float(covariance[di, di]), float(covariance[ei, ei]),
                                float(covariance[fi, fi]))
        cov_de, cov_df, cov_ef = (float(covariance[di, ei]), float(covariance[di, fi]),
                                   float(covariance[ei, fi]))

        # Isotropic (vignetting-like): trace/2 of the quadratic form's matrix.
        iso_mag = (d + e) / 2.0
        iso_se = float(np.sqrt(max((var_d + var_e + 2.0 * cov_de) / 4.0, 0.0)))
        iso_p = _pvalue(iso_mag / iso_se, df) if iso_se > 0 else float("nan")
        r_corner_sq = (w_img / 2.0) ** 2 + (h_img / 2.0) ** 2
        result["isotropic_curvature"] = {
            "magnitude": float(iso_mag), "se": iso_se, "p_value": iso_p,
            "center_to_corner_adu": float(iso_mag * r_corner_sq),
        }

        # Anisotropic (astigmatism-like): traceless part's eigenvalue
        # magnitude. The quadratic form's matrix is [[d, f/2], [f/2, e]] (the
        # xy coefficient f is twice the off-diagonal matrix entry, since both
        # dx*dy and dy*dx contribute) -- p/q below use f/2, not raw f.
        p_ = (d - e) / 2.0
        q_ = f / 2.0
        var_p = (var_d + var_e - 2.0 * cov_de) / 4.0
        var_q = var_f / 4.0
        cov_pq = (cov_df - cov_ef) / 4.0
        aniso_mag = float(np.hypot(p_, q_))
        aniso_se = _mag_se(p_, q_, var_p, var_q, cov_pq)
        aniso_p = _pvalue(aniso_mag / aniso_se, df) if aniso_se > 0 else float("nan")
        # atan2 is scale-invariant, so the raw (d-e, f) pair gives the same
        # angle as (p_, q_) would -- the /2 factors cancel out of the ratio.
        result["anisotropic_curvature"] = {
            "magnitude": aniso_mag, "se": aniso_se,
            "direction_deg": float(0.5 * np.degrees(np.arctan2(f, d - e))),
            "p_value": aniso_p,
        }

    return result


def evaluate_surface_at(fit_result: dict, x_native, y_native) -> np.ndarray:
    """Evaluate a fit_background_surface() result at native pixel coordinates."""
    x_native = np.asarray(x_native, dtype=np.float64)
    y_native = np.asarray(y_native, dtype=np.float64)
    if fit_result.get("insufficient_data") or fit_result.get("coefficients") is None:
        return np.full(np.broadcast(x_native, y_native).shape, np.nan, dtype=np.float32)
    norm = fit_result["normalize"]
    dx = x_native - norm["cx"]
    dy = y_native - norm["cy"]
    X = _design_matrix(dx, dy, fit_result["selected_order"])
    return (X @ fit_result["coefficients"]).astype(np.float32)


def evaluate_surface(fit_result: dict, image_shape: tuple[int, int], step: int) -> np.ndarray:
    """Evaluate a fit at exactly the pixel grid `arr[::step, ::step]` samples
    for a native (h, w) = image_shape array -- so the result is pixel-for-
    pixel aligned with AstroImage.background_display()'s output."""
    h, w = image_shape
    y_idx = np.arange(0, h, step, dtype=np.float64)
    x_idx = np.arange(0, w, step, dtype=np.float64)
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
    return evaluate_surface_at(fit_result, xx, yy)


def compare_fits(fit_a: dict, fit_b: dict) -> dict:
    """Approximate combined-SE z-test comparing two independent fits' gradient/
    isotropic/anisotropic magnitudes. Returns {"z": None, "p": None} for a
    term missing (None) on either side."""
    from core.stats_utils import combined_se_z_test
    result: dict = {}
    for key in ("gradient", "isotropic_curvature", "anisotropic_curvature"):
        a, b = fit_a.get(key), fit_b.get(key)
        if a is None or b is None:
            result[key] = {"z": None, "p": None}
            continue
        z, p = combined_se_z_test(a["magnitude"], a["se"], b["magnitude"], b["se"])
        result[key] = {"z": z, "p": p}
    return result
