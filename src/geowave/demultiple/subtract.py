from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_toeplitz

@dataclass(frozen=True)
class WindowReport:
    t0_s: float
    t1_s: float
    offset_bin: int
    removal: float
    damped: int
    gamma_min: float

@dataclass
class SubtractionResult:
    primary: np.ndarray
    removed: np.ndarray
    windows: list[WindowReport]

def _window_starts(nt: int, n_win: int, step: int) -> list[int]:
    if n_win >= nt:
        return [0]
    starts = list(range(0, nt - n_win + 1, step))
    if starts[-1] != nt - n_win:
        starts.append(nt - n_win)
    return starts

def _taper(n_win: int, ramp: int, left: bool, right: bool) -> np.ndarray:
    w = np.ones(n_win)
    r = min(ramp, n_win // 2)
    if r > 0:
        c = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
        if left:
            w[:r] = c
        if right:
            w[-r:] = c[::-1]
    return w

def _offset_bins(offsets: np.ndarray, n_bins: int) -> list[np.ndarray]:
    idx = np.arange(offsets.size)
    if n_bins <= 1:
        return [idx]
    edges = np.quantile(offsets, np.linspace(0, 1, n_bins + 1))
    bins = []
    for b in range(n_bins):
        sel = (offsets >= edges[b]) & (offsets <= edges[b + 1] if b == n_bins - 1 else offsets < edges[b + 1])
        if sel.any():
            bins.append(idx[sel])
    return bins

def _estimate_filter(mw: np.ndarray, dw: np.ndarray, taper: np.ndarray, n_f: int, lag: int, eps_abs: float) -> np.ndarray:
    n = mw.shape[1]
    n_f = min(n_f, n - lag)
    if n_f <= 0:
        return np.zeros(0)
    r = np.zeros(n_f)
    g = np.zeros(n_f)
    ed = 0.0
    for m, d in zip(mw, dw):
        mt = m * taper
        dt = d * taper
        r += np.correlate(mt, mt, 'full')[n - 1:n - 1 + n_f]
        g += np.correlate(dt, mt, 'full')[n - 1 + lag:n - 1 + lag + n_f]
        ed += float(dt @ dt)
    if r[0] <= 1e-9 * ed:
        return np.zeros(n_f)
    r[0] += eps_abs
    try:
        return solve_toeplitz((r, r), g)
    except np.linalg.LinAlgError:
        return np.zeros(n_f)

def _apply_filter(m: np.ndarray, f: np.ndarray, lag: int) -> np.ndarray:
    if not f.size:
        return np.zeros(m.size)
    full = np.convolve(m, f)
    if lag >= 0:
        return np.concatenate([np.zeros(lag), full])[:m.size]
    out = full[-lag:m.size - lag]
    return np.concatenate([out, np.zeros(m.size - out.size)])

def _guard_gamma(d: np.ndarray, s: np.ndarray, max_removal: float) -> tuple[float, float]:
    ed = float(d @ d)
    if ed <= 0:
        return 1.0, 0.0
    r = d - s
    removal = 1.0 - float(r @ r) / ed
    if removal <= max_removal:
        return 1.0, removal
    ss = float(s @ s)
    ds = float(d @ s)
    if ss <= 0:
        return 0.0, removal
    disc = ds * ds - ss * max_removal * ed
    gamma = (ds - np.sqrt(max(disc, 0.0))) / ss
    return float(np.clip(gamma, 0.0, 1.0)), removal

def adaptive_subtract(data: np.ndarray, model: np.ndarray, dt_s: float, spacing_m: float, window_s: float=0.8, overlap: float=0.5, offset_bins: int=1, filter_length_s: float=0.1, filter_lag_s: float=0.0, prewhiten: float=0.02, max_removal: float=0.9, valid: np.ndarray | None=None) -> SubtractionResult:
    ns, nr, nt = data.shape
    if valid is None:
        valid = np.ones((ns, nr), bool)
    n_win = max(8, int(round(window_s / dt_s)))
    step = max(1, int(round(n_win * (1.0 - overlap))))
    ramp = n_win - step
    n_f = max(1, int(round(filter_length_s / dt_s)))
    lag = max(0, int(round(filter_lag_s / dt_s)))
    starts = _window_starts(nt, n_win, step)
    removed = np.zeros_like(data)
    den = np.zeros(nt)
    agg: dict[tuple[int, int], list[float]] = {}
    for w, s0 in enumerate(starts):
        s1 = min(s0 + n_win, nt)
        taper = _taper(s1 - s0, ramp, left=w > 0, right=s1 < nt)
        den[s0:s1] += taper
    den = np.maximum(den, 1e-12)
    for i in range(ns):
        offs = np.abs(np.arange(nr) - i) * spacing_m
        bins = _offset_bins(offs, offset_bins)
        eps_abs = prewhiten * float(np.mean(np.sum(model[i, valid[i]] ** 2, axis=1))) * (n_win / nt) if valid[i].any() else 0.0
        for w, s0 in enumerate(starts):
            s1 = min(s0 + n_win, nt)
            taper = _taper(s1 - s0, ramp, left=w > 0, right=s1 < nt)
            for b, idx in enumerate(bins):
                idx = idx[valid[i, idx]]
                if not idx.size:
                    continue
                mw = model[i, idx][:, s0:s1]
                dw = data[i, idx][:, s0:s1]
                f = _estimate_filter(mw, dw, taper, n_f, lag, eps_abs)
                s = np.stack([_apply_filter(model[i, j, s0:s1], f, lag) for j in idx])
                d_cat = (dw * taper).ravel()
                s_cat = (s * taper).ravel()
                gamma, removal = _guard_gamma(d_cat, s_cat, max_removal)
                if gamma < 1.0:
                    s = s * gamma
                    s_cat = s_cat * gamma
                for jj, j in enumerate(idx):
                    removed[i, j, s0:s1] += s[jj] * taper
                key = (w, b)
                if key not in agg:
                    agg[key] = [0.0, 0.0, 0.0, 1.0]
                rec = agg[key]
                res = d_cat - s_cat
                rec[0] += float(d_cat @ d_cat)
                rec[1] += float(res @ res)
                rec[2] += 1.0 if gamma < 1.0 else 0.0
                rec[3] = min(rec[3], gamma)
    removed = removed / den
    windows = []
    for (w, b), rec in sorted(agg.items()):
        s0 = starts[w]
        s1 = min(s0 + n_win, nt)
        removal = 1.0 - rec[1] / rec[0] if rec[0] > 0 else 0.0
        windows.append(WindowReport(s0 * dt_s, s1 * dt_s, b, removal, int(rec[2]), rec[3]))
    return SubtractionResult(data - removed, removed, windows)
