from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from parallel_extractor_model import run_parallel_extractors_and_blend


DEFAULT_PARAMS = {
    'K_multiplier': 1.00,
    'C_entrain': 0.015,
    'k_precip': 0.50,
    'E_murphree': 0.70,
    'delta_crit': 2.50,
    'alpha_density': 3.00,
}

PROFILE_DIR = Path(__file__).resolve().parent / 'calibration_profiles'

PARAM_BOUNDS = {
    'K_multiplier': (0.30, 3.50),
    'C_entrain': (0.001, 0.10),
    'k_precip': (0.05, 5.00),
    'E_murphree': (0.30, 1.00),
    'delta_crit': (0.50, 8.00),
    'alpha_density': (1.00, 7.00),
}


@dataclass
class ParallelPlantDataPoint:
    timestamp: str = ''
    feed_density_kg_m3: float = 1028.0
    feed_CCR_wt_pct: float = 22.8
    feed_visc_135_cst: Optional[float] = None
    feed_flow_a_m3hr: float = 0.0
    feed_flow_b_m3hr: float = 0.0
    feed_temp_a_C: float = 75.0
    feed_temp_b_C: float = 75.0
    top_temp_a_C: float = 85.0
    top_temp_b_C: float = 85.0
    mid_temp_a_C: float = 75.0
    mid_temp_b_C: float = 75.0
    bottom_temp_a_C: float = 70.0
    bottom_temp_b_C: float = 70.0
    primary_prop_a: float = 0.0
    primary_prop_b: float = 0.0
    secondary_prop_a: float = 0.0
    secondary_prop_b: float = 0.0
    dao_yield_vol_pct: Optional[float] = None
    dao_visc_100_cst: Optional[float] = None
    dao_ccr_wt_pct: Optional[float] = None
    dao_asphaltene_wt_pct: Optional[float] = None

    @property
    def has_measurements(self) -> bool:
        return any(v is not None for v in [
            self.dao_yield_vol_pct,
            self.dao_visc_100_cst,
            self.dao_ccr_wt_pct,
            self.dao_asphaltene_wt_pct,
        ])


@dataclass
class ParallelCalibrationWeights:
    DAO_yield: float = 1.00
    DAO_viscosity: float = 0.20
    DAO_CCR: float = 0.50
    DAO_asphaltene: float = 5.00
    non_convergence_penalty: float = 25.00

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParallelCalibrationResult:
    success: bool
    active_params: list[str]
    calibrated_params: dict
    initial_params: dict
    cost_initial: float
    cost_final: float
    improvement_pct: float
    n_operating_points: int
    n_function_evals: int
    metrics: dict
    point_results: list
    n_converged_points: int
    n_nonconverged_points: int
    message: str
    elapsed_s: float


def points_from_normalized_dataframe(df: pd.DataFrame, row_flag: str = 'usable_core_row') -> list[ParallelPlantDataPoint]:
    sub = df[df[row_flag]].copy() if row_flag in df.columns else df.copy()
    points: list[ParallelPlantDataPoint] = []
    for _, row in sub.iterrows():
        points.append(ParallelPlantDataPoint(
            timestamp=str(row['event_ts']),
            feed_density_kg_m3=float(row['feed_density_kg_m3']),
            feed_CCR_wt_pct=float(row['feed_CCR_wt_pct']),
            feed_visc_135_cst=None if pd.isna(row['feed_visc_135_cst']) else float(row['feed_visc_135_cst']),
            feed_flow_a_m3hr=float(row['feed_flow_a_m3hr']),
            feed_flow_b_m3hr=float(row['feed_flow_b_m3hr']),
            feed_temp_a_C=float(row['feed_temp_a_C']),
            feed_temp_b_C=float(row['feed_temp_b_C']),
            top_temp_a_C=float(row['top_temp_a_C']),
            top_temp_b_C=float(row['top_temp_b_C']),
            mid_temp_a_C=float(row['mid_temp_a_C']),
            mid_temp_b_C=float(row['mid_temp_b_C']),
            bottom_temp_a_C=float(row['bottom_temp_a_C']),
            bottom_temp_b_C=float(row['bottom_temp_b_C']),
            primary_prop_a=float(row['primary_prop_a']),
            primary_prop_b=float(row['primary_prop_b']),
            secondary_prop_a=float(row['secondary_prop_a']),
            secondary_prop_b=float(row['secondary_prop_b']),
            dao_yield_vol_pct=None if pd.isna(row['dao_yield_vol_pct']) else float(row['dao_yield_vol_pct']),
            dao_visc_100_cst=None if pd.isna(row['dao_visc_100_cst']) else float(row['dao_visc_100_cst']),
            dao_ccr_wt_pct=None if pd.isna(row.get('dao_ccr_wt_pct', np.nan)) else float(row['dao_ccr_wt_pct']),
            dao_asphaltene_wt_pct=None if pd.isna(row.get('dao_asphaltene_wt_pct', np.nan)) else float(row['dao_asphaltene_wt_pct']),
        ))
    return points


def simulate_parallel_point(
    pt: ParallelPlantDataPoint,
    params: dict,
    solvent_name: str = 'propane',
    pressure_bar: float = 40.0,
    n_stages: int = 4,
) -> dict:
    row = pd.Series({
        'feed_density_kg_m3': pt.feed_density_kg_m3,
        'feed_CCR_wt_pct': pt.feed_CCR_wt_pct,
        'feed_visc_135_cst': pt.feed_visc_135_cst,
        'feed_flow_a_m3hr': pt.feed_flow_a_m3hr,
        'feed_flow_b_m3hr': pt.feed_flow_b_m3hr,
        'feed_temp_a_C': pt.feed_temp_a_C,
        'feed_temp_b_C': pt.feed_temp_b_C,
        'top_temp_a_C': pt.top_temp_a_C,
        'top_temp_b_C': pt.top_temp_b_C,
        'mid_temp_a_C': pt.mid_temp_a_C,
        'mid_temp_b_C': pt.mid_temp_b_C,
        'bottom_temp_a_C': pt.bottom_temp_a_C,
        'bottom_temp_b_C': pt.bottom_temp_b_C,
        'primary_prop_a': pt.primary_prop_a,
        'primary_prop_b': pt.primary_prop_b,
        'secondary_prop_a': pt.secondary_prop_a,
        'secondary_prop_b': pt.secondary_prop_b,
        'propane_temp_C': np.nan,
    })
    pred = run_parallel_extractors_and_blend(
        row=row,
        params=params,
        solvent_name=solvent_name,
        pressure_bar=pressure_bar,
        n_stages=n_stages,
    )
    return {
        'DAO_yield': pred.DAO_yield_vol_pct_pred,
        'DAO_viscosity': pred.DAO_viscosity_pred,
        'DAO_CCR': pred.DAO_CCR_pred,
        'DAO_asphaltene': pred.DAO_asphaltene_pred,
        'train_a_converged': pred.train_a_converged,
        'train_b_converged': pred.train_b_converged,
        'converged': pred.train_a_converged and pred.train_b_converged,
        'raw_prediction': pred,
    }


def build_parallel_residuals(
    param_vec: np.ndarray,
    dataset: list[ParallelPlantDataPoint],
    weights: ParallelCalibrationWeights,
    history: list,
    active_params: Optional[list[str]] = None,
) -> np.ndarray:
    param_names = active_params or list(DEFAULT_PARAMS.keys())
    params = DEFAULT_PARAMS.copy()
    params.update(dict(zip(param_names, param_vec)))
    residuals = []
    cost = 0.0
    for pt in dataset:
        pred = simulate_parallel_point(pt, params)
        if not pred['converged']:
            penalty = float(weights.non_convergence_penalty)
            residuals.append(penalty)
            cost += penalty ** 2
            continue
        if pt.dao_yield_vol_pct is not None:
            r = weights.DAO_yield * (pred['DAO_yield'] - pt.dao_yield_vol_pct)
            residuals.append(r)
            cost += r ** 2
        if pt.dao_visc_100_cst is not None:
            r = weights.DAO_viscosity * (pred['DAO_viscosity'] - pt.dao_visc_100_cst)
            residuals.append(r)
            cost += r ** 2
        if pt.dao_ccr_wt_pct is not None:
            r = weights.DAO_CCR * (pred['DAO_CCR'] - pt.dao_ccr_wt_pct)
            residuals.append(r)
            cost += r ** 2
        if pt.dao_asphaltene_wt_pct is not None:
            r = weights.DAO_asphaltene * (pred['DAO_asphaltene'] - pt.dao_asphaltene_wt_pct)
            residuals.append(r)
            cost += r ** 2
    history.append({'params': params.copy(), 'cost': float(cost)})
    return np.array(residuals, dtype=float)


def compute_parallel_metrics(
    x_vec: np.ndarray,
    dataset: list[ParallelPlantDataPoint],
    active_params: Optional[list[str]] = None,
) -> tuple[dict, list]:
    param_names = active_params or list(DEFAULT_PARAMS.keys())
    params = DEFAULT_PARAMS.copy()
    params.update(dict(zip(param_names, x_vec)))
    errors = {'DAO_yield': [], 'DAO_viscosity': [], 'DAO_CCR': [], 'DAO_asphaltene': []}
    results = []
    n_converged = 0
    for pt in dataset:
        pred = simulate_parallel_point(pt, params)
        row = {
            'timestamp': pt.timestamp,
            'sim': {},
            'plant': {},
            'error': {},
            'converged': bool(pred['converged']),
            'train_a_converged': bool(pred.get('train_a_converged', False)),
            'train_b_converged': bool(pred.get('train_b_converged', False)),
            'included_in_metrics': bool(pred['converged']),
        }
        if pred['converged']:
            n_converged += 1
        mappings = [
            ('DAO_yield', pt.dao_yield_vol_pct),
            ('DAO_viscosity', pt.dao_visc_100_cst),
            ('DAO_CCR', pt.dao_ccr_wt_pct),
            ('DAO_asphaltene', pt.dao_asphaltene_wt_pct),
        ]
        for key, plant_val in mappings:
            if plant_val is not None and pred['converged']:
                sim_val = float(pred[key])
                err = sim_val - float(plant_val)
                errors[key].append(err)
                row['sim'][key] = round(sim_val, 4)
                row['plant'][key] = round(float(plant_val), 4)
                row['error'][key] = round(float(err), 4)
        results.append(row)
    metrics = {}
    for key, errs in errors.items():
        if errs:
            arr = np.array(errs, dtype=float)
            metrics[key] = {
                'MAE': round(float(np.mean(np.abs(arr))), 4),
                'RMSE': round(float(np.sqrt(np.mean(arr ** 2))), 4),
                'bias': round(float(np.mean(arr)), 4),
                'n': int(len(arr)),
            }
    metrics['convergence'] = {
        'n_points': int(len(dataset)),
        'converged_points': int(n_converged),
        'nonconverged_points': int(len(dataset) - n_converged),
        'convergence_pct': round(100.0 * n_converged / max(len(dataset), 1), 2),
    }
    return metrics, results


def run_parallel_calibration(
    dataset: list[ParallelPlantDataPoint],
    init_params: Optional[dict] = None,
    weights: Optional[ParallelCalibrationWeights] = None,
    active_params: Optional[list[str]] = None,
    max_nfev: int = 200,
    ftol: float = 1e-4,
    verbose: bool = True,
) -> ParallelCalibrationResult:
    t0 = time.time()
    if init_params is None:
        init_params = DEFAULT_PARAMS.copy()
    if weights is None:
        weights = ParallelCalibrationWeights()
    if active_params is None:
        active_params = list(DEFAULT_PARAMS.keys())
    invalid = [p for p in active_params if p not in DEFAULT_PARAMS]
    if invalid:
        raise ValueError(f'Unknown active calibration params: {invalid}')
    valid = [pt for pt in dataset if pt.has_measurements]
    if not valid:
        raise ValueError('No valid parallel calibration points with measurements found.')
    param_names = list(active_params)
    x0 = np.array([init_params.get(p, DEFAULT_PARAMS[p]) for p in param_names], dtype=float)
    lo = np.array([PARAM_BOUNDS[p][0] for p in param_names], dtype=float)
    hi = np.array([PARAM_BOUNDS[p][1] for p in param_names], dtype=float)
    history = []
    res0 = build_parallel_residuals(x0, valid, weights, history, param_names)
    cost0 = float(np.sum(res0 ** 2))
    if verbose:
        print('=' * 60)
        print('Parallel PDA Calibration')
        print(f'Operating points : {len(valid)}')
        print(f'Initial cost     : {cost0:.4f}')
        print('=' * 60)
    opt = least_squares(
        fun=build_parallel_residuals,
        x0=x0,
        bounds=(lo, hi),
        args=(valid, weights, history, param_names),
        method='trf',
        ftol=ftol,
        xtol=1e-5,
        gtol=1e-5,
        max_nfev=max_nfev,
        verbose=0,
    )
    x_opt = opt.x
    cost_f = float(np.sum(opt.fun ** 2))
    improvement = 100.0 * (cost0 - cost_f) / max(cost0, 1e-10)
    cal_params = DEFAULT_PARAMS.copy()
    cal_params.update({p: round(float(v), 6) for p, v in zip(param_names, x_opt)})
    metrics, point_results = compute_parallel_metrics(x_opt, valid, param_names)
    elapsed = time.time() - t0
    if verbose:
        print(f'Final cost       : {cost_f:.4f}')
        print(f'Improvement      : {improvement:.2f}%')
        print(f'Evaluations      : {opt.nfev}')
        print(f'Message          : {opt.message}')
        for k, m in metrics.items():
            if {'MAE', 'RMSE', 'bias'}.issubset(m):
                print(f"  {k:16s} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} bias={m['bias']:+.4f}")
            else:
                print(
                    f"  {k:16s} converged={m.get('converged_points', 0)}/"
                    f"{m.get('n_points', 0)} ({m.get('convergence_pct', 0.0):.2f}%)"
                )
    return ParallelCalibrationResult(
        success=bool(opt.success or (cost_f < cost0)),
        active_params=param_names,
        calibrated_params=cal_params,
        initial_params=init_params.copy(),
        cost_initial=round(cost0, 4),
        cost_final=round(cost_f, 4),
        improvement_pct=round(improvement, 2),
        n_operating_points=len(valid),
        n_function_evals=int(opt.nfev),
        metrics=metrics,
        point_results=point_results,
        n_converged_points=int(metrics['convergence']['converged_points']),
        n_nonconverged_points=int(metrics['convergence']['nonconverged_points']),
        message=str(opt.message),
        elapsed_s=round(elapsed, 2),
    )


def run_parallel_calibration_from_workbooks(
    lims_path: str,
    extractor_path: str,
    row_flag: str = 'usable_core_row',
    **kwargs,
) -> tuple[ParallelCalibrationResult, pd.DataFrame, object]:
    from data_ingestion import build_normalized_parallel_dataset

    df, summary = build_normalized_parallel_dataset(lims_path, extractor_path)
    points = points_from_normalized_dataframe(df, row_flag=row_flag)
    result = run_parallel_calibration(points, **kwargs)
    return result, df, summary


def load_parallel_calibration_profile(profile_name_or_path: str) -> dict:
    candidate = Path(profile_name_or_path)
    search_paths = []
    if candidate.exists():
        search_paths.append(candidate)
    else:
        search_paths.append(PROFILE_DIR / profile_name_or_path)
        if candidate.suffix != '.json':
            search_paths.append(PROFILE_DIR / f'{profile_name_or_path}.json')
    for path in search_paths:
        if path.exists():
            with path.open() as f:
                profile = json.load(f)
            profile.setdefault('active_params', list(DEFAULT_PARAMS.keys()))
            profile.setdefault('parameters', DEFAULT_PARAMS.copy())
            profile.setdefault('row_flag', 'usable_core_row')
            profile['_profile_path'] = str(path)
            return profile
    raise FileNotFoundError(f'Parallel calibration profile not found: {profile_name_or_path}')


def run_parallel_calibration_from_profile(
    lims_path: str,
    extractor_path: str,
    profile_name_or_path: str,
    row_flag: Optional[str] = None,
    **kwargs,
) -> tuple[ParallelCalibrationResult, pd.DataFrame, object, dict]:
    profile = load_parallel_calibration_profile(profile_name_or_path)
    result, df, summary = run_parallel_calibration_from_workbooks(
        lims_path,
        extractor_path,
        row_flag=row_flag or profile.get('row_flag', 'usable_core_row'),
        init_params=kwargs.pop('init_params', profile.get('parameters', DEFAULT_PARAMS.copy())),
        active_params=kwargs.pop('active_params', profile.get('active_params', list(DEFAULT_PARAMS.keys()))),
        **kwargs,
    )
    return result, df, summary, profile
