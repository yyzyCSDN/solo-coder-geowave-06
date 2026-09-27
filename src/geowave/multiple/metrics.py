from __future__ import annotations
import numpy as np

def nrmse(estimate: np.ndarray, reference: np.ndarray, eps: float=1e-12) -> float:
    '能量归一化均方根误差：||est-ref|| / ||ref||。'
    e = np.asarray(estimate, float).ravel() - np.asarray(reference, float).ravel()
    r = np.asarray(reference, float).ravel()
    return float(np.sqrt(e @ e / max(r @ r, eps)))

def normalized_xcorr(a: np.ndarray, b: np.ndarray, eps: float=1e-12) -> float:
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    return float((x @ y) / max(np.sqrt((x @ x) * (y @ y)), eps))

def energy_leakage(data: np.ndarray, removed: np.ndarray, model: np.ndarray,
                   kappa: float=1.05, eps: float=1e-12) -> float:
    '''能量泄漏率，用于阻止过度相减。

    E_removed = ||d||^2 - ||d-removed||^2  为实际从数据中拿掉的能量；
    多次波模型在单标量最优拟合下最多可解释 E_model = alpha^2 ||m||^2。
    匹配滤波允许少量波形整形余量 kappa，超出部分视为吃掉了一次波能量：

        leak = max(0, E_removed - kappa * E_model) / ||d||^2

    leak 接近 0 表示相减能量均可由模型解释；显著大于 0 表示滤波器在拟合一次波。
    '''
    d = np.asarray(data, float).ravel()
    r = np.asarray(removed, float).ravel()
    m = np.asarray(model, float).ravel()
    ed = d @ d
    if ed <= eps:
        return 0.0
    e_removed = 2.0 * (d @ r) - r @ r
    mm = m @ m
    alpha = (d @ m) / mm if mm > eps else 0.0
    e_model = alpha * alpha * mm
    return float(max(0.0, e_removed - kappa * e_model) / ed)
