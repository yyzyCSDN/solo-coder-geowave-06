import numpy as np
import pytest
from geowave.model.traces import TraceHeader, Trace, TraceSet
from geowave.demultiple.grid import assemble_surface_grid
from geowave.demultiple.regularize import regularize_cube, extrapolate_near_offsets
from geowave.demultiple.predict import predict_multiples
from geowave.demultiple.subtract import adaptive_subtract
from geowave.demultiple.metrics import removal_ratio, local_similarity, energy
from geowave.demultiple.pipeline import SrmeConfig, run_srme

DT = 0.004
NT = 625
DX = 25.0
N_STA = 80
NEAR = 6
N_REC = 28
N_SHOTS = N_STA - NEAR - N_REC + 1
REFLECTORS = [(0.5, 1500.0, 0.9), (0.9, 1700.0, -0.7), (1.4, 2000.0, 0.8), (1.8, 2300.0, -0.7), (2.2, 2500.0, 0.6)]
VRMS_TREND = tuple((t0, v) for t0, v, _ in REFLECTORS)

def _ricker(f_hz, dt, n):
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f_hz * t) ** 2
    return (1 - 2 * a) * np.exp(-a)

def _primary_trace(offset_m, wave):
    refl = np.zeros(NT)
    for t0, v, amp in REFLECTORS:
        t_ev = np.sqrt(t0 ** 2 + (offset_m / v) ** 2)
        refl[int(round(t_ev / DT))] += amp
    return np.convolve(refl, wave)[:NT]

def _primary_cube(wave):
    cube = np.zeros((N_STA, N_STA, NT))
    for i in range(N_STA):
        for j in range(N_STA):
            cube[i, j] = _primary_trace(abs(j - i) * DX, wave)
    return cube

def _corr(a, b):
    return float(np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))

def _recorded_cells():
    for i in range(N_SHOTS):
        for k in range(N_REC):
            yield i, i + NEAR + k

def _surface_data(prim, orders=6):
    d = prim.copy()
    for _ in range(orders):
        d = prim - predict_multiples(d, DT, 0.0, 0.5 / DT)
    return d

@pytest.fixture(scope='module')
def synthetic():
    wave = _ricker(22.0, DT, 51)
    prim = _primary_cube(wave)
    data = _surface_data(prim)
    mult = data - prim
    rng = np.random.default_rng(7)
    data = data + 0.02 * rng.standard_normal(prim.shape)
    kill_shots = {14, 15, 16}
    traces = []
    for i, j in _recorded_cells():
        if i in kill_shots or rng.random() < 0.12:
            continue
        h = TraceHeader(f'g{i}-{j}', f's{i}', f'r{j}', i * DX, 0.0, j * DX, 0.0, DT, 0.0)
        traces.append(Trace(h, data[i, j]))
    return TraceSet(traces, 'line'), prim, mult

def test_prediction_matches_time_convolution():
    wave = _ricker(22.0, DT, 51)
    prim = _primary_cube(wave)
    pred = predict_multiples(prim, DT, 0.0, 0.5 / DT)
    s, r = 20, 50
    ref = np.zeros(2 * NT)
    for k in range(N_STA):
        ref[:2 * NT - 1] += np.convolve(prim[s, k], prim[k, r]) * DT
    assert _corr(pred[s, r], ref[:NT]) > 0.99

def test_regularization_fills_missing(synthetic):
    dataset, prim, mult = synthetic
    grid = assemble_surface_grid(dataset)
    live = np.isfinite(grid.cube[:, :, 0])
    assert not live.all()
    cube, filled = extrapolate_near_offsets(grid.cube, live, grid.spacing_m, grid.dt_s, NEAR - 1, VRMS_TREND)
    reg, valid = regularize_cube(cube, live | filled)
    assert np.isfinite(reg).all()
    truth = prim + mult
    killed = [(15, j) for j in range(15 + NEAR, 15 + NEAR + N_REC)]
    got = np.array([reg[i, j] for i, j in killed])
    ref = np.array([truth[i, j] for i, j in killed])
    assert _corr(got, ref) > 0.6
    assert np.allclose(reg[live], grid.cube[live])
    assert valid[live].all()

def test_pipeline_attenuates_multiples(synthetic):
    dataset, prim, _ = synthetic
    cfg = SrmeConfig(fill_near_offsets=NEAR - 1, near_offset_vrms_m_s=VRMS_TREND, fmin_hz=3.0, fmax_hz=90.0, offset_bins=2)
    out = run_srme(dataset, cfg)
    ids_in = dataset.trace_ids()
    assert out.primary.trace_ids() == ids_in
    assert out.multiples_model.trace_ids() == ids_in
    assert out.removed.trace_ids() == ids_in
    cell = {}
    for tr in dataset.traces:
        cell[tr.header.trace_id] = (int(tr.header.source_x / DX), int(tr.header.receiver_x / DX))
    prim_out = {t.header.trace_id: t.samples for t in out.primary.traces}
    d_obs = []
    o_obs = []
    p_obs = []
    for tr in dataset.traces:
        tid = tr.header.trace_id
        i, j = cell[tid]
        if not 10 <= i <= 40:
            continue
        d_obs.append(tr.samples[:NT])
        o_obs.append(prim_out[tid])
        p_obs.append(prim[i, j])
    d_obs = np.array(d_obs)
    o_obs = np.array(o_obs)
    p_obs = np.array(p_obs)
    assert _corr(o_obs, p_obs) > _corr(d_obs, p_obs) + 0.005

    def win_energy(x, wins):
        return sum(energy(x[:, int(a / DT):int(b / DT)]) for a, b in wins)

    mult_windows = [(1.25, 1.35), (1.65, 1.75), (2.05, 2.15)]
    prim_windows = [(0.85, 0.95), (1.35, 1.45), (1.75, 1.85)]
    assert win_energy(o_obs, mult_windows) < 0.5 * win_energy(d_obs, mult_windows)
    assert win_energy(o_obs, prim_windows) > 0.9 * win_energy(d_obs, prim_windows)
    for tr_in, tr_p, tr_r in zip(dataset.traces, out.primary.traces, out.removed.traces):
        assert np.allclose(tr_p.samples + tr_r.samples, tr_in.samples[:NT], atol=1e-8)
    rep = out.report
    assert np.isfinite(rep.removal_ratio) and np.isfinite(rep.similarity_p95)
    assert 0.0 < rep.removal_ratio < 1.0
    assert not rep.over_subtracted
    assert rep.similarity_p95 < 0.6

def test_leakage_guard_caps_removal():
    rng = np.random.default_rng(3)
    g = rng.standard_normal((3, 6, 400))
    guarded = adaptive_subtract(g, g.copy(), DT, DX, window_s=0.2, filter_length_s=0.05, max_removal=0.8)
    assert removal_ratio(g, guarded.primary) <= 0.85
    assert any(w.damped for w in guarded.windows)
    free = adaptive_subtract(g, g.copy(), DT, DX, window_s=0.2, filter_length_s=0.05, max_removal=1.0)
    assert removal_ratio(g, free.primary) > 0.9
    assert removal_ratio(g, guarded.primary) < removal_ratio(g, free.primary)

def test_local_similarity_detects_leakage():
    rng = np.random.default_rng(5)
    p = rng.standard_normal((2, 500))
    m = rng.standard_normal((2, 500))
    clean = np.abs(local_similarity(p, m, 25)).mean()
    assert clean < 0.2
    leaked = m + 0.5 * p
    dirty = np.abs(local_similarity(p, leaked, 25)).mean()
    assert dirty > 0.4
    assert np.abs(local_similarity(p, np.zeros_like(p), 25)).max() == 0.0

def test_executor_registers_srme(synthetic):
    from geowave.workflow.operations import build_default_executor
    dataset, _, _ = synthetic
    ex = build_default_executor()
    assert 'srme' in ex.operations
    out = ex.operations['srme'](dataset, {'window_s': 0.5, 'offset_bins': 1})
    assert isinstance(out, TraceSet)
    assert out.trace_ids() == dataset.trace_ids()
