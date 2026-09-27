from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.multiple.metrics import energy_leakage

@dataclass(frozen=True)
class SubtractionResult:
    primary: np.ndarray             # 保留信号（一次波估计）
    matched_multiples: np.ndarray   # 实际减去的多次波（匹配滤波后）
    leakage: np.ndarray             # (n_win_shot, n_win_time) 分窗能量泄漏率
    window_starts: tuple[tuple[int, ...], tuple[int, ...]]
    removed_fraction: float         # 相减能量占数据能量比
    guarded_windows: int            # 触发泄漏保护并回退的窗口数

def matched_filter(model: np.ndarray, data: np.ndarray, n_filter: int,
                   lag0: int | None=None, damping: float=1e-2) -> np.ndarray:
    '''最小二乘匹配滤波：min ||d - f*m||^2，Tikhonov 预白化 damping·tr(R)/L。
    model/data: (n_trace, n_t)；f 覆盖 lag ∈ [-lag0, n_filter-lag0)，
    允许模型相对数据有时移/相位差。正规方程由移位视图精确构造，
    与 apply_filter 的边界截断严格对偶（不用 Toeplitz 近似）。'''
    if lag0 is None:
        lag0 = n_filter // 2
    nt = model.shape[-1]
    S = np.zeros((n_filter,) + model.shape)
    for k in range(n_filter):
        lag = k - lag0
        if lag >= 0:
            S[k, ..., lag:] = model[..., :nt - lag]
        else:
            S[k, ..., :nt + lag] = model[..., -lag:]
    S2 = S.reshape(n_filter, -1)
    R = S2 @ S2.T
    rhs = S2 @ np.asarray(data, float).reshape(-1)
    R.flat[::n_filter + 1] += damping * np.trace(R) / n_filter
    return np.linalg.solve(R, rhs)

def apply_filter(model: np.ndarray, f: np.ndarray, lag0: int) -> np.ndarray:
    'y[t] = sum_k f[k] * m[t - (k - lag0)]，支持任意前导轴。'
    n = model.shape[-1]
    out = np.zeros_like(model)
    for k, fk in enumerate(f):
        lag = k - lag0
        if lag >= 0:
            out[..., lag:] += fk * model[..., :n - lag]
        else:
            out[..., :n + lag] += fk * model[..., -lag:]
    return out

def _starts(n: int, win: int, stride: int) -> tuple[int, ...]:
    s = sorted(set(list(range(0, max(n - win, 0) + 1, stride)) + [max(n - win, 0)]))
    return tuple(s)

def _taper(n: int) -> np.ndarray:
    return np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2

def adaptive_subtract(data: np.ndarray, model: np.ndarray, n_filter: int=31,
                      lag0: int | None=None, damping: float=1e-2,
                      win_shots: int=8, win_time: int=128, overlap: float=0.5,
                      leak_thresh: float=0.2, kappa: float=1.05,
                      guard: bool=True) -> SubtractionResult:
    '''分窗自适应相减：每个 (炮 × 时间) 窗估计一个匹配滤波器，重叠区锥化加权叠加。

    泄漏保护：逐窗计算能量泄漏率（见 metrics.energy_leakage），超过
    leak_thresh 时先加大阻尼重估，仍超则退化为单标量拟合（其相减能量
    结构上不超过模型可解释能量），从而阻止滤波器吃掉一次波。'''
    data = np.asarray(data, float)
    model = np.asarray(model, float)
    ns, nr, nt = data.shape
    if lag0 is None:
        lag0 = n_filter // 2
    ws, wt = min(win_shots, ns), min(win_time, nt)
    ss = _starts(ns, ws, max(1, round(ws * (1 - overlap))))
    ts = _starts(nt, wt, max(1, round(wt * (1 - overlap))))
    taper = np.outer(_taper(ws), _taper(wt))[:, None, :]
    matched = np.zeros_like(data)
    weight = np.zeros((ns, 1, nt))
    leakage = np.zeros((len(ss), len(ts)))
    guarded = 0
    for i, s0 in enumerate(ss):
        for j, t0 in enumerate(ts):
            sl = np.s_[s0:s0 + ws, :, t0:t0 + wt]
            m = model[sl].reshape(ws * nr, wt)
            d = data[sl].reshape(ws * nr, wt)
            f = matched_filter(m, d, n_filter, lag0, damping)
            y = apply_filter(m, f, lag0)
            leak = energy_leakage(d, y, m, kappa)
            if guard and leak > leak_thresh:
                guarded += 1
                f2 = matched_filter(m, d, n_filter, lag0, damping * 100.0)
                y2 = apply_filter(m, f2, lag0)
                if energy_leakage(d, y2, m, kappa) > leak_thresh:
                    mm = float((m * m).sum())
                    alpha = float((d * m).sum() / mm) if mm > 0 else 0.0
                    y2 = alpha * m
                y = y2
                leak = energy_leakage(d, y, m, kappa)
            leakage[i, j] = leak
            matched[sl] += taper * y.reshape(ws, nr, wt)
            weight[sl] += taper
    matched /= np.maximum(weight, 1e-6)
    ed = float((data * data).sum())
    removed = float((matched * matched).sum() / max(ed, 1e-12))
    return SubtractionResult(data - matched, matched, leakage, (ss, ts), removed, guarded)
