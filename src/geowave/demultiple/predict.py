from __future__ import annotations
import numpy as np

def predict_multiples(cube: np.ndarray, dt_s: float, fmin_hz: float=0.0, fmax_hz: float | None=None) -> np.ndarray:
    ns, nr, nt = cube.shape
    nf = 2 * nt
    fmax_hz = float(fmax_hz) if fmax_hz else 0.5 / dt_s
    F = np.fft.rfft(np.nan_to_num(cube), n=nf, axis=-1)
    freqs = np.fft.rfftfreq(nf, dt_s)
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    M = np.zeros_like(F)
    Fb = F[:, :, band]
    M[:, :, band] = np.einsum('ikf,kjf->ijf', Fb, Fb) * dt_s
    return np.fft.irfft(M, n=nf, axis=-1)[:, :, :nt]
