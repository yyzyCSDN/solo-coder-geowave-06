from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import uniform_filter1d
from geowave.demultiple.subtract import WindowReport

@dataclass(frozen=True)
class LeakageReport:
    removal_ratio: float
    energy_ratio: float
    similarity_mean: float
    similarity_p95: float
    similarity_max: float
    damped_windows: int
    over_subtracted: bool
    windows: tuple[WindowReport, ...] = field(default_factory=tuple)

def energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, float) ** 2))

def removal_ratio(data: np.ndarray, primary: np.ndarray) -> float:
    ed = energy(data)
    return 1.0 - energy(primary) / ed if ed > 0 else 0.0

def local_similarity(a: np.ndarray, b: np.ndarray, half_win: int) -> np.ndarray:
    n = max(1, 2 * half_win + 1)
    aa = uniform_filter1d(a * a, n, axis=-1, mode='nearest')
    bb = uniform_filter1d(b * b, n, axis=-1, mode='nearest')
    ab = uniform_filter1d(a * b, n, axis=-1, mode='nearest')
    ea = 1e-6 * float(np.mean(aa))
    eb = 1e-6 * float(np.mean(bb))
    return ab / np.sqrt((aa + ea) * (bb + eb) + 1e-30)

def leakage_report(data: np.ndarray, primary: np.ndarray, removed: np.ndarray, dt_s: float, window_s: float=0.2, similarity_limit: float=0.5, windows: tuple[WindowReport, ...]=()) -> LeakageReport:
    ed = energy(data)
    ep = energy(primary)
    sim = np.abs(local_similarity(primary, removed, max(1, int(0.5 * window_s / dt_s))))
    sim_mean = float(sim.mean())
    sim_p95 = float(np.quantile(sim, 0.95))
    sim_max = float(sim.max())
    damped = sum(w.damped for w in windows)
    return LeakageReport(
        removal_ratio=1.0 - ep / ed if ed > 0 else 0.0,
        energy_ratio=ep / ed if ed > 0 else 0.0,
        similarity_mean=sim_mean,
        similarity_p95=sim_p95,
        similarity_max=sim_max,
        damped_windows=damped,
        over_subtracted=sim_p95 > similarity_limit,
        windows=windows)
