from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from geowave.model.traces import Trace, TraceHeader, TraceSet

@dataclass
class SurfaceGrid:
    stations_m: np.ndarray
    dt_s: float
    nt: int
    cube: np.ndarray
    observed: np.ndarray
    cells: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def spacing_m(self) -> float:
        return float(self.stations_m[1] - self.stations_m[0]) if self.stations_m.size > 1 else 1.0

def _station_spacing(xs: np.ndarray, xr: np.ndarray) -> float:
    uniq = np.unique(np.concatenate([xs, xr]))
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 1e-6]
    return float(diffs.min()) if diffs.size else 25.0

def assemble_surface_grid(dataset: TraceSet, station_spacing_m: float | None=None, use_reciprocity: bool=True) -> SurfaceGrid:
    dt = dataset.validate_sampling()
    if not dataset.traces:
        raise ValueError('empty dataset')
    xs = np.array([t.header.source_x for t in dataset.traces])
    xr = np.array([t.header.receiver_x for t in dataset.traces])
    spacing = float(station_spacing_m) if station_spacing_m else _station_spacing(xs, xr)
    x0 = float(min(xs.min(), xr.min()))
    n_sta = int(round((max(xs.max(), xr.max()) - x0) / spacing)) + 1
    stations = x0 + np.arange(n_sta) * spacing
    nt = min(t.samples.size for t in dataset.traces)
    cube = np.full((n_sta, n_sta, nt), np.nan)
    observed = np.zeros((n_sta, n_sta), bool)
    cells: dict[tuple[int, int], int] = {}
    for k, tr in enumerate(dataset.traces):
        i = int(round((tr.header.source_x - x0) / spacing))
        j = int(round((tr.header.receiver_x - x0) / spacing))
        if (i, j) in cells:
            continue
        cells[(i, j)] = k
        cube[i, j] = tr.samples[:nt]
        observed[i, j] = True
    if use_reciprocity:
        for i in range(n_sta):
            for j in range(i + 1, n_sta):
                if not observed[i, j] and observed[j, i]:
                    cube[i, j] = cube[j, i]
                elif not observed[j, i] and observed[i, j]:
                    cube[j, i] = cube[i, j]
    return SurfaceGrid(stations, dt, nt, cube, observed, cells)

def observed_traceset(grid: SurfaceGrid, dataset: TraceSet, cube: np.ndarray, name: str, tag: str) -> TraceSet:
    traces = []
    for (i, j), k in sorted(grid.cells.items(), key=lambda kv: kv[1]):
        traces.append(dataset.traces[k].copy_with(cube[i, j], step=tag))
    return TraceSet(traces, name)

def full_traceset(grid: SurfaceGrid, cube: np.ndarray, name: str, tag: str) -> TraceSet:
    traces = []
    n_sta = grid.stations_m.size
    for i in range(n_sta):
        for j in range(n_sta):
            if not np.isfinite(cube[i, j]).all():
                continue
            h = TraceHeader(f'{name}:{i}:{j}', f'{name}-s{i}', f'{name}-r{j}', float(grid.stations_m[i]), 0.0, float(grid.stations_m[j]), 0.0, grid.dt_s, 0.0)
            traces.append(Trace(h, cube[i, j], {'interpolated': 'false' if grid.observed[i, j] else 'true', 'step': tag}))
    return TraceSet(traces, name)
