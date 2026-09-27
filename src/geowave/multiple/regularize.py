from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d

def infer_mask(cube: np.ndarray) -> np.ndarray:
    '全零道视为缺失道，返回 (n_shot, n_receiver) 掩码，True=观测。'
    return np.any(np.asarray(cube) != 0.0, axis=-1)

def _fill_axis(cube: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    out = cube
    other = 1 - axis
    for i in range(cube.shape[other]):
        good = mask[:, i] if axis == 0 else mask[i, :]
        if good.all() or not good.any():
            continue
        g = np.where(good)[0]
        b = np.where(~good)[0]
        sub = out[:, i, :] if axis == 0 else out[i, :, :]
        filled = interp1d(g, sub[g], axis=0, kind='linear', fill_value='extrapolate')(b)
        if axis == 0:
            out[b, i, :] = filled
        else:
            out[i, b, :] = filled
    return out

def linear_fill(cube: np.ndarray, mask: np.ndarray, passes: int=2) -> np.ndarray:
    '沿炮/道轴交替线性插值，快速兜底，也可作为 POCS 初值。'
    out = np.asarray(cube, float).copy()
    for _ in range(max(1, passes)):
        for axis in (0, 1):
            out = _fill_axis(out, mask, axis)
    return out

def pocs_fill(cube: np.ndarray, mask: np.ndarray, n_iter: int=60,
              tau_end: float=0.02, init: np.ndarray | None=None) -> np.ndarray:
    '''POCS 规则化（Abma & Kabir 2006）：(炮, 道) 二维谱域逐次阈值收缩，
    每轮把观测样点复原，仅重填缺失位置。可处理随机缺道、整炮/整缆缺失。'''
    keep = mask[..., None]
    obs = np.where(keep, cube, 0.0)
    x = obs if init is None else np.where(keep, obs, init)
    spec = np.abs(np.fft.fft2(obs, axes=(0, 1)))
    tau_max = np.percentile(spec, 99.9, axis=(0, 1), keepdims=True)
    for frac in np.linspace(1.0, tau_end, max(1, n_iter)):
        X = np.fft.fft2(x, axes=(0, 1))
        X = np.where(np.abs(X) >= frac * tau_max, X, 0.0)
        y = np.fft.ifft2(X, axes=(0, 1)).real
        x = np.where(keep, obs, y)
    return x

def regularize_cube(cube: np.ndarray, mask: np.ndarray | None=None,
                    n_iter: int=60, tau_end: float=0.02) -> tuple[np.ndarray, np.ndarray]:
    '缺炮缺道规则化入口：返回 (规则化数据体, 观测掩码)。观测样点保持原值。'
    cube = np.asarray(cube, float)
    if mask is None:
        mask = infer_mask(cube)
    mask = np.asarray(mask, bool)
    if mask.all():
        return cube.copy(), mask
    init = linear_fill(cube, mask)
    return pocs_fill(cube, mask, n_iter, tau_end, init), mask
