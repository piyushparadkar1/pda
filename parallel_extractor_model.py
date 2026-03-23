from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from residue_distribution import HPCL_FEEDS, build_residue_distribution, estimate_sara_from_properties, api_from_density
from hunter_nash_extractor import run_extractor
from asphaltene_kinetics import KineticParams
from stage_efficiency import StageEfficiency
from entrainment_model import EntrainmentParams
from run_simulation import build_T_profile
from quality_model import predict_dao_viscosity


@dataclass
class TrainCase:
    train_id: str
    feed_density_kg_m3: float
    feed_CCR_wt_pct: float
    feed_visc_135_cst: Optional[float]
    feed_flow_m3hr: float
    feed_mass_basis_kg_hr: float
    so_ratio: float
    predilution_frac: float
    feed_temp_C: float
    T_top_C: float
    T_mid_C: float
    T_bottom_C: float
    propane_temp_C: Optional[float] = None
    solvent_name: str = 'propane'
    pressure_bar: float = 40.0
    n_stages: int = 4


@dataclass
class BlendedPrediction:
    DAO_yield_vol_pct_pred: float
    DAO_yield_wt_pct_pred: float
    DAO_viscosity_pred: float
    DAO_CCR_pred: float
    DAO_asphaltene_pred: float
    DAO_density_pred: float
    MW_DAO_pred: float
    SARA_DAO_pred: dict
    DAO_mass_flow_pred_kg_hr: float
    DAO_volume_flow_pred_m3hr: float
    train_a_converged: bool
    train_b_converged: bool


def convert_visc_135_to_visc_100(visc_135_cst: float | None) -> float | None:
    """Estimate kinematic viscosity at 100°C from a 135°C lab result.

    The repo already contains two design-feed anchor pairs in ``HPCL_FEEDS``:
    - Basra/Kuwait mix: 230 cSt @135°C -> 1621 cSt @100°C
    - Basra Light:      177 cSt @135°C -> 1137 cSt @100°C

    Rather than using a fixed multiplier, interpolate the implied ratio in log
    space so the conversion remains positive, monotonic, and exact at the known
    plant-feed anchors.
    """
    if visc_135_cst is None or pd.isna(visc_135_cst):
        return None

    visc_135 = float(visc_135_cst)
    if visc_135 <= 0.0:
        return None

    anchor_pairs = sorted(
        (float(feed['visc_135']), float(feed['visc_100']) / float(feed['visc_135']))
        for feed in HPCL_FEEDS.values()
        if float(feed['visc_135']) > 0.0 and float(feed['visc_100']) > 0.0
    )
    anchor_visc_135 = np.array([p[0] for p in anchor_pairs], dtype=float)
    anchor_ratio = np.array([p[1] for p in anchor_pairs], dtype=float)

    ratio = float(np.exp(np.interp(
        np.log(visc_135),
        np.log(anchor_visc_135),
        np.log(anchor_ratio),
        left=np.log(anchor_ratio[0]),
        right=np.log(anchor_ratio[-1]),
    )))
    return visc_135 * ratio


def build_train_case(
    row: pd.Series,
    train: str,
    solvent_name: str = 'propane',
    pressure_bar: float = 40.0,
    n_stages: int = 4,
) -> TrainCase:
    train = str(train).upper()
    if train not in {'A', 'B'}:
        raise ValueError("train must be 'A' or 'B'")
    if train == 'A':
        feed_flow = float(row['feed_flow_a_m3hr'])
        feed_temp = float(row['feed_temp_a_C'])
        T_top = float(row['top_temp_a_C'])
        T_mid = float(row['mid_temp_a_C'])
        T_bottom = float(row['bottom_temp_a_C'])
        primary = float(row['primary_prop_a'])
        secondary = float(row['secondary_prop_a'])
    else:
        feed_flow = float(row['feed_flow_b_m3hr'])
        feed_temp = float(row['feed_temp_b_C'])
        T_top = float(row['top_temp_b_C'])
        T_mid = float(row['mid_temp_b_C'])
        T_bottom = float(row['bottom_temp_b_C'])
        primary = float(row['primary_prop_b'])
        secondary = float(row['secondary_prop_b'])
    total_prop = primary + secondary
    so_ratio = total_prop / feed_flow if feed_flow > 0 else np.nan
    pred_frac = secondary / total_prop if total_prop > 0 else 0.0
    feed_density = float(row['feed_density_kg_m3'])
    feed_ccr = float(row['feed_CCR_wt_pct'])
    feed_visc_135 = row.get('feed_visc_135_cst', np.nan)
    return TrainCase(
        train_id=train,
        feed_density_kg_m3=feed_density,
        feed_CCR_wt_pct=feed_ccr,
        feed_visc_135_cst=None if pd.isna(feed_visc_135) else float(feed_visc_135),
        feed_flow_m3hr=feed_flow,
        feed_mass_basis_kg_hr=feed_flow * feed_density,
        so_ratio=so_ratio,
        predilution_frac=pred_frac,
        feed_temp_C=feed_temp,
        T_top_C=T_top,
        T_mid_C=T_mid,
        T_bottom_C=T_bottom,
        propane_temp_C=float(row['propane_temp_C']) if 'propane_temp_C' in row and pd.notna(row['propane_temp_C']) else None,
        solvent_name=solvent_name,
        pressure_bar=pressure_bar,
        n_stages=n_stages,
    )


def build_train_components(case: TrainCase, n_comp: int = 20) -> list:
    feed_visc_100 = convert_visc_135_to_visc_100(case.feed_visc_135_cst)
    sara = estimate_sara_from_properties(
        density_kg_m3=case.feed_density_kg_m3,
        CCR=case.feed_CCR_wt_pct,
        visc_100=feed_visc_100,
    )
    sg = case.feed_density_kg_m3 / 1000.0
    F_precip = float(np.clip(0.28 + (sara['asphaltenes'] - 10.0) / 80.0, 0.15, 0.50))
    custom_feed = {
        'SARA': sara,
        'MW_heavy_cut': 750.0,
        'F_precip': F_precip,
        'SG_15': sg,
        'API': api_from_density(case.feed_density_kg_m3),
        'CCR_wt': case.feed_CCR_wt_pct,
        'visc_100': feed_visc_100,
    }
    return build_residue_distribution(custom_feed=custom_feed, n_comp=n_comp, solvent_name=case.solvent_name)


def run_single_train(
    case: TrainCase,
    params: dict,
    n_comp: int = 20,
    thermo_mode: str = 'kvalue',
) -> dict:
    comps = build_train_components(case, n_comp=n_comp)
    T_profile = build_T_profile(case.T_bottom_C, case.T_top_C, case.n_stages)
    return run_extractor(
        components=comps,
        solvent_name=case.solvent_name,
        solvent_ratio=float(case.so_ratio),
        N_stages=int(case.n_stages),
        T_profile=T_profile,
        P=float(case.pressure_bar) * 1e5,
        kinetics=KineticParams(float(params.get('k_precip', 0.5)), 10.0),
        efficiency=StageEfficiency(float(params.get('E_murphree', 0.70))),
        entrainment=EntrainmentParams(float(params.get('C_entrain', 0.015)), 1.20),
        K_multiplier=float(params.get('K_multiplier', 1.0)),
        delta_crit=float(params.get('delta_crit', 2.5)),
        predilution_frac=float(case.predilution_frac),
        alpha_density=float(params.get('alpha_density', 3.0)),
        thermo_mode=thermo_mode,
        feed_basis=float(case.feed_mass_basis_kg_hr),
    )


def compute_stream_density(mass_vector: np.ndarray, components: list) -> float:
    total = mass_vector.sum()
    if total <= 1e-14:
        return 0.0
    rho = np.array([c.density for c in components], dtype=float)
    return float(np.dot(mass_vector / total, rho))


def compute_stream_mw(mass_vector: np.ndarray, components: list) -> float:
    total = mass_vector.sum()
    if total <= 1e-14:
        return 0.0
    mw_arr = np.array([c.MW for c in components], dtype=float)
    return float(np.dot(mass_vector / total, mw_arr))


def compute_stream_sara(mass_vector: np.ndarray, components: list) -> dict:
    total = mass_vector.sum()
    if total <= 1e-14:
        return {}
    out: dict[str, float] = {}
    for c, m in zip(components, mass_vector):
        out[c.sara_class] = out.get(c.sara_class, 0.0) + float(m)
    return {k: 100.0 * v / total for k, v in out.items()}


def compute_stream_asphaltene_pct(mass_vector: np.ndarray, components: list) -> float:
    total = mass_vector.sum()
    if total <= 1e-14:
        return 0.0
    asp = sum(float(m) for c, m in zip(components, mass_vector) if c.sara_class == 'asphaltenes')
    return 100.0 * asp / total


def compute_stream_ccr_from_sara(sara_dao: dict) -> float:
    f_res = sara_dao.get('resins', 0.0) / 100.0
    f_asp = sara_dao.get('asphaltenes', 0.0) / 100.0
    f_aro = sara_dao.get('aromatics', 0.0) / 100.0
    return float(max((0.10 * f_asp + 0.12 * f_res + 0.005 * f_aro) * 100.0, 0.01))


def blend_train_results(result_a: dict, result_b: dict, case_a: TrainCase, case_b: TrainCase) -> BlendedPrediction:
    components = result_a['components']
    mass_DAO_total = result_a['mass_DAO'] + result_b['mass_DAO']
    mass_asphalt_total = result_a['mass_asphalt'] + result_b['mass_asphalt']
    feed_mass_total = case_a.feed_mass_basis_kg_hr + case_b.feed_mass_basis_kg_hr
    feed_vol_total = case_a.feed_flow_m3hr + case_b.feed_flow_m3hr
    dao_mass_flow = float(mass_DAO_total.sum())
    dao_yield_wt_pct = 100.0 * dao_mass_flow / max(feed_mass_total, 1e-14)
    density_dao = compute_stream_density(mass_DAO_total, components)
    mw_dao = compute_stream_mw(mass_DAO_total, components)
    sara_dao = compute_stream_sara(mass_DAO_total, components)
    asph_pct = compute_stream_asphaltene_pct(mass_DAO_total, components)
    dao_ccr = compute_stream_ccr_from_sara(sara_dao)
    dao_volume_flow = dao_mass_flow / max(density_dao * 1000.0, 1e-14)
    dao_yield_vol_pct = 100.0 * dao_volume_flow / max(feed_vol_total, 1e-14)
    dao_visc = predict_dao_viscosity(mw_dao, density_dao, sara_dao)
    return BlendedPrediction(
        DAO_yield_vol_pct_pred=float(dao_yield_vol_pct),
        DAO_yield_wt_pct_pred=float(dao_yield_wt_pct),
        DAO_viscosity_pred=float(dao_visc),
        DAO_CCR_pred=float(dao_ccr),
        DAO_asphaltene_pred=float(asph_pct),
        DAO_density_pred=float(density_dao),
        MW_DAO_pred=float(mw_dao),
        SARA_DAO_pred=sara_dao,
        DAO_mass_flow_pred_kg_hr=float(dao_mass_flow),
        DAO_volume_flow_pred_m3hr=float(dao_volume_flow),
        train_a_converged=bool(result_a.get('converged', False)),
        train_b_converged=bool(result_b.get('converged', False)),
    )


def run_parallel_extractors_and_blend(
    row: pd.Series,
    params: dict,
    solvent_name: str = 'propane',
    pressure_bar: float = 40.0,
    n_stages: int = 4,
    n_comp: int = 20,
    thermo_mode: str = 'kvalue',
) -> BlendedPrediction:
    case_a = build_train_case(row, 'A', solvent_name=solvent_name, pressure_bar=pressure_bar, n_stages=n_stages)
    case_b = build_train_case(row, 'B', solvent_name=solvent_name, pressure_bar=pressure_bar, n_stages=n_stages)
    result_a = run_single_train(case_a, params, n_comp=n_comp, thermo_mode=thermo_mode)
    result_b = run_single_train(case_b, params, n_comp=n_comp, thermo_mode=thermo_mode)
    return blend_train_results(result_a, result_b, case_a, case_b)
