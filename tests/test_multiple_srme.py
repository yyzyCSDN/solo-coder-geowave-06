from __future__ import annotations
import numpy as np
import pytest
from geowave.multiple.regularize import regularize_cube, infer_mask
from geowave.multiple.predict import predict_multiples, convolve_cubes
from geowave.multiple.subtract import matched_filter, apply_filter, adaptive_subtract
from geowave.multiple.metrics import energy_leakage, nrmse
from geowave.multiple.pipeline import SRMEConfig, run_srme
from geowave.multiple.synthetic import (synthetic_primaries, forward_surface_multiples,
                                        calibrate_wavelet_scale, damage_cube)

DT, DX = 0.004, 25.0

@pytest.fixture(scope='module')
def earth():
    P1 = synthetic_primaries(24, 24, 256, DT, DX, wavelet_scale=1.0, seed=7)
    gamma = calibrate_wavelet_scale(P1, DT, DX, target_ratio=1.2, window=(0.7, 0.9))
    P = P1 * gamma
    D = forward_surface_multiples(P, DT, DX)
    return P, D

def test_infer_mask_detects_zero_traces():
    cube = np.ones((3, 4, 8))
    cube[1, 2] = 0.0
    mask = infer_mask(cube)
    assert mask.shape == (3, 4) and not mask[1, 2] and mask.sum() == 11

def test_regularize_recovers_decimated_data(earth):
    P, _ = earth
    damaged, mask = damage_cube(P, random_frac=0.15, missing_shots=(5,), missing_receivers=(9,))
    filled, _ = regularize_cube(damaged, mask, n_iter=40)
    assert np.allclose(filled[mask], P[mask])          # 观测样点不被改动
    err = nrmse(filled[~mask], P[~mask])
    assert err < 0.45, f'缺道插值误差过大: {err:.3f}'

def test_prediction_matches_forward_model(earth):
    P, D = earth
    model = predict_multiples(D, DT, DX, n_iter=3)
    assert nrmse(model, D - P) < 0.15

def test_matched_filter_recovers_known_filter():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((20, 200))
    f_true = np.sin(np.linspace(0, 3 * np.pi, 25)) * np.hanning(25)
    d = apply_filter(m, f_true, lag0=12)
    f_est = matched_filter(m, d, 25, lag0=12, damping=1e-6)
    assert nrmse(f_est, f_true) < 1e-3

def test_subtraction_guard_protects_pure_primary(earth):
    P, _ = earth
    junk = convolve_cubes(P[::-1], P, DT, DX)         # 与数据不匹配的伪模型
    res = adaptive_subtract(P, junk, n_filter=25, win_shots=6, win_time=96,
                            leak_thresh=0.1)
    assert res.removed_fraction < 0.05                 # 几乎不减能量
    assert nrmse(res.primary, P) < 0.25                # 一次波基本保留

def test_pipeline_end_to_end(earth):
    P, D = earth
    damaged, mask = damage_cube(D, random_frac=0.12, missing_shots=(10,), missing_receivers=(14,))
    cfg = SRMEConfig(DT, DX, n_iter=2, pocs_iter=40, n_filter=25,
                     win_shots=6, win_time=96, leak_thresh=0.2)
    res = run_srme(damaged, cfg, mask=mask, reference_primary=P)
    assert np.allclose(res.primary + res.multiples_matched, res.regularized)  # 输出严格可分
    assert res.diagnostics['primary_nrmse'] < 0.4
    assert res.diagnostics['primary_nrmse'] < res.diagnostics['raw_nrmse']
    assert res.diagnostics['leakage_median'] < 0.2
