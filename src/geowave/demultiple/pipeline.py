from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet
from geowave.demultiple.grid import assemble_surface_grid, observed_traceset, full_traceset
from geowave.demultiple.regularize import regularize_cube, extrapolate_near_offsets
from geowave.demultiple.predict import predict_multiples
from geowave.demultiple.subtract import adaptive_subtract
from geowave.demultiple.metrics import LeakageReport, leakage_report

@dataclass(frozen=True)
class SrmeConfig:
    station_spacing_m: float | None = None
    use_reciprocity: bool = True
    fill_near_offsets: int = 0
    near_offset_vrms_m_s: object = 1500.0
    fmin_hz: float = 0.0
    fmax_hz: float | None = None
    window_s: float = 0.8
    window_overlap: float = 0.5
    offset_bins: int = 1
    filter_length_s: float = 0.1
    filter_lag_s: float = 0.0
    prewhiten: float = 0.02
    max_removal: float = 0.9
    similarity_limit: float = 0.5

@dataclass
class SrmeOutput:
    primary: TraceSet
    multiples_model: TraceSet
    removed: TraceSet
    regularized: TraceSet
    report: LeakageReport

def run_srme(dataset: TraceSet, config: SrmeConfig=SrmeConfig()) -> SrmeOutput:
    grid = assemble_surface_grid(dataset, config.station_spacing_m, config.use_reciprocity)
    live = np.isfinite(grid.cube[:, :, 0])
    cube = grid.cube
    if config.fill_near_offsets > 0:
        cube, filled = extrapolate_near_offsets(cube, live, grid.spacing_m, grid.dt_s, config.fill_near_offsets, config.near_offset_vrms_m_s)
        live = live | filled
    cube, valid = regularize_cube(cube, live)
    model = predict_multiples(cube, grid.dt_s, config.fmin_hz, config.fmax_hz)
    sub = adaptive_subtract(cube, model, grid.dt_s, grid.spacing_m, config.window_s, config.window_overlap, config.offset_bins, config.filter_length_s, config.filter_lag_s, config.prewhiten, config.max_removal, valid)
    keep = valid[:, :, None]
    report = leakage_report(np.where(keep, cube, 0.0), np.where(keep, sub.primary, 0.0), np.where(keep, sub.removed, 0.0), grid.dt_s, config.window_s, config.similarity_limit, tuple(sub.windows))
    name = dataset.name + '-srme'
    return SrmeOutput(
        primary=observed_traceset(grid, dataset, sub.primary, name + '-primary', 'srme-primary'),
        multiples_model=observed_traceset(grid, dataset, model, name + '-model', 'srme-model'),
        removed=observed_traceset(grid, dataset, sub.removed, name + '-removed', 'srme-removed'),
        regularized=full_traceset(grid, cube, name + '-regular', 'srme-regularized'),
        report=report)
