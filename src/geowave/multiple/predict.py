from __future__ import annotations
import numpy as np

def convolve_cubes(a: np.ndarray, b: np.ndarray, dt_s: float, dx_m: float,
                   fmin_hz: float=0.0, fmax_hz: float | None=None) -> np.ndarray:
    '''表面褶积 C(ω) = dt·dx · A(ω) @ B(ω)：沿时间褶积、沿表面炮检对求和。
    a/b: (n_shot, n_receiver, n_t)，炮与检波点须位于同一表面规则网格。'''
    nt = a.shape[-1]
    nfft = 1 << (2 * nt - 1).bit_length()
    A = np.fft.rfft(a, nfft, axis=-1)
    B = np.fft.rfft(b, nfft, axis=-1)
    freqs = np.fft.rfftfreq(nfft, dt_s)
    if fmax_hz is None:
        fmax_hz = 0.8 * freqs.max()
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    C = np.zeros_like(A)
    C[..., band] = dt_s * dx_m * np.einsum('skf,krf->srf', A[..., band], B[..., band])
    return np.fft.irfft(C, nfft, axis=-1)[..., :nt]

def predict_multiples(cube: np.ndarray, dt_s: float, dx_m: float=1.0, n_iter: int=2,
                      fmin_hz: float=3.0, fmax_hz: float | None=None) -> np.ndarray:
    '''自由表面相关多次波预测（反馈模型，Verschuur et al. 1992）。

    自由表面反射系数取 -1：M = -P·D，迭代 P ← D + P·D 收敛到 P = D(I-D)^{-1}，
    逐次迭代把高阶多次波纳入模型。输入须先规则化（缺炮会在模型中产生空洞）。
    返回与 cube 同形的多次波模型（含各阶）；与真实多次波之间差一个源子波/
    尺度因子，由后续自适应相减吸收。'''
    D = np.asarray(cube, float)
    P = D.copy()
    for _ in range(max(1, n_iter)):
        P = D + convolve_cubes(P, D, dt_s, dx_m, fmin_hz, fmax_hz)
    return D - P
