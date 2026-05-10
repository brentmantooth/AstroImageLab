from __future__ import annotations

import numpy as np


def normalize_unit_interval(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    dtype = data.dtype
    data = data.astype(np.float32, copy=False)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    if min_val >= 0.0 and max_val <= 1.0:
        return data
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scale = float(info.max) if info.max > 0 else None
        if scale:
            return data / scale
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return np.zeros_like(data, dtype=np.float32)
    return (data - min_val) / (max_val - min_val)


def mtf(x: np.ndarray | float, m: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.clip(m, 1e-6, 1.0 - 1e-6))
    y = np.zeros_like(x)
    mask = (x > 0.0) & (x < 1.0)
    if np.any(mask):
        num = (m - 1.0) * x[mask]
        den = (2.0 * m - 1.0) * x[mask] - m
        y[mask] = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    y = np.where(x >= 1.0, 1.0, y)
    return np.clip(y, 0.0, 1.0)


def stf_stretch(data: np.ndarray,
                shadow_clip: float = -2.8,
                target_bkg: float = 0.2) -> np.ndarray:
    norm = normalize_unit_interval(data)
    finite = norm[np.isfinite(norm)]
    if finite.size == 0:
        return np.zeros_like(norm, dtype=np.float32)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    c = med + shadow_clip * 1.4826 * mad
    c = float(np.clip(c, 0.0, 1.0))
    if not np.isfinite(c):
        return np.zeros_like(norm, dtype=np.float32)
    if c >= 1.0:
        c = 1.0 - 1e-6
    denom = 1.0 - c
    if denom <= 0.0:
        return np.zeros_like(norm, dtype=np.float32)
    scaled = (norm - c) / denom
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = np.clip(scaled, 0.0, 1.0)
    med_c = float(np.clip(med - c, 0.0, 1.0))
    midtones = float(mtf(med_c, target_bkg))
    if not np.isfinite(midtones) or midtones <= 0.0 or midtones >= 1.0:
        return scaled
    return mtf(scaled, midtones)


def normalize_for_display(data: np.ndarray,
                           stretch: bool = True) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    if stretch:
        scaled = stf_stretch(data)
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        scaled = (data - lo) / (hi - lo)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)
