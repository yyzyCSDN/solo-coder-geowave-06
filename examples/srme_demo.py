#!/usr/bin/env python3
'''海上自由表面多次波压制演示：目标层被水层底多次波覆盖。

流程：缺炮缺道规则化(POCS) → 反馈模型多次波预测 → 分窗自适应相减(泄漏保护)。
对比简单带通：多次波与一次波同频带，带通要么压不住多次波、要么伤及一次波。

运行：python3 examples/srme_demo.py
'''
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt
from geowave.multiple.pipeline import SRMEConfig, run_srme
from geowave.multiple.metrics import nrmse
from geowave.multiple.synthetic import (synthetic_primaries, forward_surface_multiples,
                                        calibrate_wavelet_scale, damage_cube)

DT, DX = 0.004, 25.0

def best_bandpass_nrmse(data: np.ndarray, reference: np.ndarray, dt: float) -> tuple[float, float]:
    '扫描低截频的最优零相位带通，作为“简单滤波”基线。'
    best = (np.inf, 0.0)
    for lo in [2, 5, 8, 12, 16, 20, 25, 30]:
        b, a = butter(4, [lo, 0.9 / (2 * dt)], btype='band', fs=1 / dt)
        err = nrmse(filtfilt(b, a, data, axis=-1), reference)
        if err < best[0]:
            best = (err, float(lo))
    return best

def main() -> None:
    # 1. 合成海上数据：强水层底(0.4 s)的一次多次波正好压在 0.8 s 弱目标层上
    P1 = synthetic_primaries(48, 48, 500, DT, DX, wavelet_scale=1.0, seed=7)
    gamma = calibrate_wavelet_scale(P1, DT, DX, target_ratio=1.2, window=(0.7, 0.9))
    P = P1 * gamma
    D = forward_surface_multiples(P, DT, DX)
    rng = np.random.default_rng(11)
    noise = 0.02 * np.sqrt(np.mean(D ** 2))   # 2% 随机噪声
    D_noisy = D + noise * rng.standard_normal(D.shape)

    # 2. 模拟野外缺损：12% 随机丢道 + 2 整炮 + 1 条缆
    damaged, mask = damage_cube(D_noisy, random_frac=0.12,
                                missing_shots=(10, 26), missing_receivers=(30,))

    # 3. SRME 流程
    cfg = SRMEConfig(DT, DX, n_iter=2, pocs_iter=60, n_filter=31,
                     win_shots=8, win_time=125, leak_thresh=0.2)
    res = run_srme(damaged, cfg, mask=mask, reference_primary=P)

    # 4. 基线：最优带通
    bp_err, bp_lo = best_bandpass_nrmse(res.regularized, P, DT)

    print('=' * 64)
    print('海上自由表面多次波压制 (SRME) 演示')
    print('=' * 64)
    print(f'数据: 48炮 x 48道 x 500样点, dt=4ms, dx=25m')
    print(f'缺损: {res.diagnostics["missing_traces"]} 道缺失 '
          f'({100 * res.diagnostics["missing_traces"] / mask.size:.1f}%), 含整炮/整缆')
    print(f'多次波/目标层能量比(0.7-0.9s): 约 1.2 (目标被覆盖)')
    print('-' * 64)
    print(f'{"原始数据 NRMS":<28}{res.diagnostics["raw_nrmse"]:.3f}')
    print(f'{"最优带通 NRMS":<28}{bp_err:.3f}  (低截 {bp_lo:g} Hz, 伤一次波)')
    print(f'{"SRME 一次波 NRMS":<28}{res.diagnostics["primary_nrmse"]:.3f}')
    print('-' * 64)
    print(f'相减能量占比:            {res.removed_fraction:.3f}')
    print(f'泄漏率 中位/最大:        {res.diagnostics["leakage_median"]:.4f} / '
          f'{res.diagnostics["leakage_max"]:.4f}')
    print(f'触发泄漏保护的窗口数:    {res.diagnostics["guarded_windows"]}')
    print(f'一次波-多次波模型相关:   {res.diagnostics["primary_model_corr"]:.4f}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        s = 24
        panels = [('损坏数据', damaged[s]), ('规则化数据', res.regularized[s]),
                  ('预测多次波', res.multiples_model[s]),
                  ('匹配后多次波(减去部分)', res.multiples_matched[s]),
                  ('SRME 一次波(保留信号)', res.primary[s]), ('真实一次波', P[s])]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        for ax, (title, panel) in zip(axes.ravel(), panels):
            v = np.percentile(np.abs(panel), 99)
            ax.imshow(panel.T, aspect='auto', cmap='seismic', vmin=-v, vmax=v,
                      extent=[0, panel.shape[0], panel.shape[1] * DT, 0])
            ax.set_title(title)
        axes[1, 0].set_xlabel('receiver')
        fig.suptitle(f'SRME demo: primary NRMS {res.diagnostics["primary_nrmse"]:.3f} '
                     f'vs bandpass {bp_err:.3f}')
        fig.tight_layout()
        out = 'examples/srme_demo.png'
        fig.savefig(out, dpi=110)
        print(f'图件已保存: {out}')
    except ImportError:
        print('(未安装 matplotlib，跳过图件)')

if __name__ == '__main__':
    main()
