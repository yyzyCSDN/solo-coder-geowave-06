from __future__ import annotations
import numpy as np

def _interp_columns(sec: np.ndarray, live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = sec.shape[0]
    x = np.arange(n)
    prev_idx = np.maximum.accumulate(np.where(live, x, -1))
    next_idx = np.minimum.accumulate(np.where(live, x, n)[::-1])[::-1]
    interior = ~live & (prev_idx >= 0) & (next_idx < n)
    out = np.nan_to_num(sec)
    if interior.any():
        p = prev_idx[interior]
        q = next_idx[interior]
        w = ((x[interior] - p) / (q - p))[:, None]
        out[interior] = (1.0 - w) * sec[p] + w * sec[q]
    return out, interior

def regularize_cube(cube: np.ndarray, live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ns, nr, nt = cube.shape
    out = np.zeros_like(cube)
    valid = np.zeros_like(live)
    for o in range(-(nr - 1), ns):
        lo = max(0, -o)
        hi = min(ns, nr - o)
        if hi - lo < 2:
            idx = np.arange(lo, hi)
            out[idx, idx + o] = np.nan_to_num(cube[idx, idx + o])
            valid[idx, idx + o] = live[idx, idx + o]
            continue
        idx = np.arange(lo, hi)
        sec = cube[idx, idx + o]
        lv = live[idx, idx + o]
        if int(lv.sum()) < 2:
            out[idx, idx + o] = np.nan_to_num(sec)
            valid[idx, idx + o] = lv
            continue
        filled, interior = _interp_columns(sec, lv)
        out[idx, idx + o] = filled
        valid[idx, idx + o] = lv | interior
    return out, valid

def _vrms_function(vrms) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vrms, float)
    if arr.ndim == 0:
        return np.array([0.0, 1.0]), np.array([float(arr), float(arr)])
    return arr[:, 0], arr[:, 1]

def extrapolate_near_offsets(cube: np.ndarray, live: np.ndarray, spacing_m: float, dt_s: float, n_fill: int, vrms=1500.0) -> tuple[np.ndarray, np.ndarray]:
    ns, nr, nt = cube.shape
    out = cube.copy()
    filled = np.zeros_like(live)
    t = np.arange(nt) * dt_s
    vt, vv = _vrms_function(vrms)
    v = np.interp(t, vt, vv)
    for i in range(ns):
        js = np.where(live[i])[0]
        if not js.size:
            continue
        nearest = int(js[np.argmin(np.abs(js - i))])
        x0 = abs(nearest - i) * spacing_m
        for j in range(ns):
            if live[i, j] or abs(j - i) * spacing_m >= x0 or abs(j - i) > n_fill:
                continue
            x1 = abs(j - i) * spacing_m
            t_src = np.sqrt(np.maximum(t ** 2 + (x0 ** 2 - x1 ** 2) / v ** 2, 0.0))
            out[i, j] = np.interp(t_src, t, out[i, nearest], left=0.0, right=0.0)
            filled[i, j] = True
    return out, filled
