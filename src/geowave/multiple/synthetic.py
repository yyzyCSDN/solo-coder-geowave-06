from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from geowave.multiple.predict import convolve_cubes

def ricker(f0_hz: float, dt_s: float, n: int) -> np.ndarray:
    t = (np.arange(n) - n // 2) * dt_s
    a = (np.pi * f0_hz * t) ** 2
    return (1 - 2 * a) * np.exp(-a)

def synthetic_primaries(n_shot: int=48, n_receiver: int=48, n_t: int=500,
                        dt_s: float=0.004, dx_m: float=25.0, f0_hz: float=30.0,
                        wavelet_scale: float=1.0, seed: int=7) -> np.ndarray:
    '''1.5D 合成一次波：水平强水层底（0.4 s）+ 微倾斜弱目标层（0.8 s）+
    随机散射体，双曲时移 + Ricker 子波。目标层与水层底一次多次波同时到达。'''
    rng = np.random.default_rng(seed)
    t0 = np.arange(n_t) * dt_s
    nxm = n_shot + n_receiver - 1
    R = np.zeros((nxm, n_t))
    xm = np.arange(nxm) * dx_m / 2

    def horizon(t_at_xm: np.ndarray, amp: float) -> None:
        ti = np.round(t_at_xm / dt_s).astype(int)
        ok = (ti >= 0) & (ti < n_t)
        R[np.arange(nxm)[ok], ti[ok]] += amp

    horizon(np.full(nxm, 0.4), 0.4)                 # 水层底，强反射
    horizon(0.8 + 1.5e-4 * xm, 0.08)                # 目标层，弱、微倾
    horizon(1.2 + 1.0e-4 * xm, 0.06)
    speckle = 0.03 * (rng.random((nxm, n_t)) < 0.01) * rng.standard_normal((nxm, n_t))
    R += gaussian_filter1d(speckle, 2.0, axis=0)
    v_rms = 1500.0 + 500.0 * t0
    w = wavelet_scale * ricker(f0_hz, dt_s, 41)
    P = np.zeros((n_shot, n_receiver, n_t))
    for s in range(n_shot):
        for r in range(n_receiver):
            offset = abs(s - r) * dx_m
            t = np.sqrt(t0 ** 2 + (offset / v_rms) ** 2)
            P[s, r] = np.convolve(np.interp(t, t0, R[s + r]), w, mode='same')
    return P

def forward_surface_multiples(primaries: np.ndarray, dt_s: float, dx_m: float,
                              n_iter: int=6) -> np.ndarray:
    '正演反馈模型：D = P - P·D 定点迭代，返回含各阶自由表面多次波的总场。'
    D = primaries.copy()
    for _ in range(n_iter):
        D = primaries - convolve_cubes(primaries, D, dt_s, dx_m)
    return D

def calibrate_wavelet_scale(primaries_unit: np.ndarray, dt_s: float, dx_m: float,
                            target_ratio: float=1.2,
                            window: tuple[float, float]=(0.7, 0.9)) -> float:
    '''多次波强度随波形幅值线性增长（M∝γ², P∝γ）。按目标窗内
    多次波/一次波能量比反推子波幅值，使合成数据多次波足够强。'''
    P1 = primaries_unit
    M1 = convolve_cubes(P1, P1, dt_s, dx_m)
    dt_s_ = dt_s
    i0, i1 = (int(w / dt_s_) for w in window)
    e_p = float((P1[..., i0:i1] ** 2).sum())
    e_m = float((M1[..., i0:i1] ** 2).sum())
    if e_p <= 0 or e_m <= 0:
        return 1.0
    return float(target_ratio / np.sqrt(e_m / e_p))

def damage_cube(cube: np.ndarray, random_frac: float=0.12,
                missing_shots: tuple[int, ...]=(10, 26),
                missing_receivers: tuple[int, ...]=(30,), seed: int=3
                ) -> tuple[np.ndarray, np.ndarray]:
    '模拟野外缺炮缺道：随机丢道 + 整炮缺失 + 整缆缺失。返回 (损坏数据, 观测掩码)。'
    rng = np.random.default_rng(seed)
    ns, nr, _ = cube.shape
    mask = np.ones((ns, nr), bool)
    mask[rng.random((ns, nr)) < random_frac] = False
    for s in missing_shots:
        if s < ns:
            mask[s, :] = False
    for r in missing_receivers:
        if r < nr:
            mask[:, r] = False
    return np.where(mask[..., None], cube, 0.0), mask
