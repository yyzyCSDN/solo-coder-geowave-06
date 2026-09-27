from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.multiple.regularize import regularize_cube
from geowave.multiple.predict import predict_multiples
from geowave.multiple.subtract import adaptive_subtract
from geowave.multiple.metrics import nrmse, normalized_xcorr

@dataclass(frozen=True)
class SRMEConfig:
    dt_s: float
    dx_m: float = 25.0          # 表面网格间距（炮检同网格）
    n_iter: int = 2             # 反馈迭代次数，>=2 可压制高阶多次波
    fmin_hz: float = 3.0
    fmax_hz: float | None = None
    pocs_iter: int = 60         # 规则化迭代次数
    n_filter: int = 31          # 匹配滤波长度（样点）
    damping: float = 1e-2       # 相对预白化
    win_shots: int = 8
    win_time: int = 128
    overlap: float = 0.5
    leak_thresh: float = 0.2    # 分窗能量泄漏率上限，超过即回退
    kappa: float = 1.05         # 波形整形余量

@dataclass(frozen=True)
class SRMEResult:
    primary: np.ndarray             # 保留信号（一次波估计）
    multiples_model: np.ndarray     # 原始预测多次波（未匹配）
    multiples_matched: np.ndarray   # 实际减去的多次波
    regularized: np.ndarray         # 规则化后的数据
    mask: np.ndarray                # (n_shot, n_receiver) 观测掩码
    leakage: np.ndarray             # 分窗能量泄漏率
    removed_fraction: float
    diagnostics: dict

def run_srme(cube: np.ndarray, config: SRMEConfig, mask: np.ndarray | None=None,
             reference_primary: np.ndarray | None=None) -> SRMEResult:
    '''海上自由表面多次波压制流程：规则化 → 预测 → 分窗自适应相减。

    cube: (n_shot, n_receiver, n_t)，缺失炮/道填零或用 mask 指定。
    输出严格区分预测多次波（multiples_model / multiples_matched）与
    保留信号（primary），并用分窗能量泄漏率阻止过度相减。'''
    cube = np.asarray(cube, float)
    filled, obs_mask = regularize_cube(cube, mask, n_iter=config.pocs_iter)
    model = predict_multiples(filled, config.dt_s, config.dx_m, config.n_iter,
                              config.fmin_hz, config.fmax_hz)
    sub = adaptive_subtract(filled, model, config.n_filter, damping=config.damping,
                            win_shots=config.win_shots, win_time=config.win_time,
                            overlap=config.overlap, leak_thresh=config.leak_thresh,
                            kappa=config.kappa)
    diag = {
        'observed_traces': int(obs_mask.sum()),
        'missing_traces': int(obs_mask.size - obs_mask.sum()),
        'leakage_median': float(np.median(sub.leakage)),
        'leakage_max': float(sub.leakage.max()),
        'guarded_windows': sub.guarded_windows,
        'primary_model_corr': normalized_xcorr(sub.primary, model),
    }
    if reference_primary is not None:
        diag['primary_nrmse'] = nrmse(sub.primary, reference_primary)
        diag['raw_nrmse'] = nrmse(filled, reference_primary)
    return SRMEResult(sub.primary, model, sub.matched_multiples, filled, obs_mask,
                      sub.leakage, sub.removed_fraction, diag)
