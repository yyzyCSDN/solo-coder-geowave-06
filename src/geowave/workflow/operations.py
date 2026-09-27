from __future__ import annotations
from geowave.model.traces import TraceSet
from geowave.signal.detrend import remove_mean, remove_linear, robust_detrend
from geowave.signal.filters import bandpass, lowpass, notch
from geowave.signal.agc import rms_agc
from geowave.signal.despike import repair_spikes
from geowave.signal.normalize import peak_normalize, rms_normalize, surface_consistent_scale
from geowave.signal.q_compensation import inverse_q_trace
from geowave.signal.mute import top_mute, bottom_mute, linear_offset_mute
from geowave.signal.deconvolution import spiking_decon, predictive_decon
from geowave.demultiple.pipeline import SrmeConfig, run_srme
from .executor import Executor

def _srme(dataset: TraceSet, p: dict[str, object]) -> TraceSet:
    fields = {k: v for k, v in p.items() if k in SrmeConfig.__dataclass_fields__}
    typed = {k: (bool(v) if k == 'use_reciprocity' else float(v) if k in ('station_spacing_m', 'fmin_hz', 'fmax_hz', 'window_s', 'window_overlap', 'filter_length_s', 'filter_lag_s', 'prewhiten', 'max_removal', 'similarity_limit', 'near_offset_vrms_m_s') else int(v)) for k, v in fields.items()}
    return run_srme(dataset, SrmeConfig(**typed)).primary

def _map(dataset: TraceSet, fn, tag: str) -> TraceSet:
    return TraceSet([fn(t) for t in dataset.traces], dataset.name + '-' + tag)

def build_default_executor() -> Executor:
    ex = Executor()
    ex.register('remove_mean', lambda ds, p: _map(ds, remove_mean, 'mean'))
    ex.register('remove_linear', lambda ds, p: _map(ds, remove_linear, 'linear'))
    ex.register('robust_detrend', lambda ds, p: _map(ds, lambda t: robust_detrend(t, int(p.get('segments', 8))), 'robust'))
    ex.register('bandpass', lambda ds, p: _map(ds, lambda t: bandpass(t, float(p['low_hz']), float(p['high_hz']), int(p.get('order', 4))), 'bandpass'))
    ex.register('lowpass', lambda ds, p: _map(ds, lambda t: lowpass(t, float(p['cut_hz']), int(p.get('order', 4))), 'lowpass'))
    ex.register('notch', lambda ds, p: _map(ds, lambda t: notch(t, float(p['freq_hz']), float(p.get('q', 30))), 'notch'))
    ex.register('rms_agc', lambda ds, p: _map(ds, lambda t: rms_agc(t, float(p['window_s'])), 'agc'))
    ex.register('despike', lambda ds, p: _map(ds, lambda t: repair_spikes(t, int(p.get('window', 11)), float(p.get('threshold', 8))), 'despike'))
    ex.register('peak_normalize', lambda ds, p: _map(ds, lambda t: peak_normalize(t, float(p.get('target', 1))), 'peak'))
    ex.register('rms_normalize', lambda ds, p: _map(ds, lambda t: rms_normalize(t, float(p.get('target', 1))), 'rms'))
    ex.register('surface_consistent_scale', lambda ds, p: surface_consistent_scale(ds))
    ex.register('inverse_q', lambda ds, p: _map(ds, lambda t: inverse_q_trace(t, float(p['q']), gain_limit=float(p.get('gain_limit', 20))), 'q'))
    ex.register('top_mute', lambda ds, p: _map(ds, lambda t: top_mute(t, float(p['cut_s']), float(p.get('taper_s', 0.02))), 'topmute'))
    ex.register('bottom_mute', lambda ds, p: _map(ds, lambda t: bottom_mute(t, float(p['cut_s']), float(p.get('taper_s', 0.02))), 'bottommute'))
    ex.register('linear_mute', lambda ds, p: _map(ds, lambda t: linear_offset_mute(t, float(p['intercept_s']), float(p['slope_s_per_m']), float(p.get('taper_s', 0.02))), 'linmute'))
    ex.register('spiking_decon', lambda ds, p: _map(ds, lambda t: spiking_decon(t, int(p.get('operator_length', 32)), float(p.get('prewhiten', 0.01))), 'spikedecon'))
    ex.register('predictive_decon', lambda ds, p: _map(ds, lambda t: predictive_decon(t, int(p['prediction_lag']), int(p.get('operator_length', 32)), float(p.get('prewhiten', 0.01))), 'preddecon'))
    ex.register('srme', _srme)
    return ex
