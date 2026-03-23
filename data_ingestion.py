from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

import numpy as np
import pandas as pd


@dataclass
class JoinToleranceConfig:
    feed_density: pd.Timedelta = pd.Timedelta(days=14)
    feed_ccr: pd.Timedelta = pd.Timedelta(days=14)
    feed_visc_135: pd.Timedelta = pd.Timedelta(days=2)
    dao_response: pd.Timedelta = pd.Timedelta(minutes=30)
    dao_lag: pd.Timedelta = pd.Timedelta(hours=3)


@dataclass
class NormalizedDatasetSummary:
    n_rows_total: int
    n_rows_core: int
    n_rows_ccr: int
    n_rows_asph: int
    min_event_ts: pd.Timestamp | None
    max_event_ts: pd.Timestamp | None


@dataclass
class NormalizedDatasetDiagnostics:
    usable_core_rows: int
    usable_ccr_rows: int
    usable_asph_rows: int
    feed_density_missing_rows: int
    feed_ccr_missing_rows: int
    feed_visc_missing_rows: int
    dao_visc_missing_rows: int
    dao_ccr_missing_rows: int
    dao_asph_missing_rows: int
    max_feed_density_age_hr: float | None
    max_feed_ccr_age_hr: float | None
    max_feed_visc_age_hr: float | None
    max_dao_visc_match_age_hr: float | None
    max_dao_ccr_match_age_hr: float | None
    max_dao_asph_match_age_hr: float | None


def clean_excel_headers(raw_headers: Iterable[object]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for h in raw_headers:
        if h is None or pd.isna(h):
            name = 'blank'
        else:
            name = str(h).strip() or 'blank'
        count = seen.get(name, 0)
        headers.append(name if count == 0 else f'{name}_{count}')
        seen[name] = count + 1
    return headers


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def coerce_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    if column in out.columns:
        out[column] = pd.to_datetime(out[column], errors='coerce')
    return out


def load_lims_workbook(path: str) -> dict[str, pd.DataFrame]:
    return {
        'feed_density_raw': pd.read_excel(path, sheet_name='feed desity at 15degC', header=None),
        'feed_ccr_raw': pd.read_excel(path, sheet_name='feed ccr wt% ', header=None),
        'feed_visc_135_raw': pd.read_excel(path, sheet_name='feed kin viscosity at 135 degC', header=None),
        'dao_visc_100_raw': pd.read_excel(path, sheet_name='dao viscosity', header=None),
        'lims_common_raw': pd.read_excel(path, sheet_name='lims pda common ', header=None),
    }


def load_extractor_workbook(path: str) -> dict[str, pd.DataFrame]:
    return {
        'pda_data_raw': pd.read_excel(path, sheet_name='pda_data', header=None),
        'tag_description_raw': pd.read_excel(path, sheet_name='tag_description', header=None),
    }


def parse_lims_common_sample_time(id_text: str) -> pd.Timestamp | pd.NaT:
    if id_text is None:
        return pd.NaT
    s = str(id_text).upper().strip()
    m = re.search(r'([RN])-?(\d{1,2})-([A-Z]{3})-(\d{2})\s+(\d{2}):(\d{2})', s)
    if not m:
        return pd.NaT
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
    }
    mon = month_map.get(m.group(3))
    if mon is None:
        return pd.NaT
    return pd.Timestamp(2000 + int(m.group(4)), mon, int(m.group(2)), int(m.group(5)), int(m.group(6)))


def _normalize_sheet_with_header_row(raw: pd.DataFrame, header_row: int = 1) -> pd.DataFrame:
    headers = clean_excel_headers(raw.iloc[header_row].tolist())
    body = raw.iloc[header_row + 1:].copy()
    body.columns = headers
    return body.reset_index(drop=True)


def normalize_simple_lims_sheet(
    df: pd.DataFrame,
    sample_ts_col: str,
    value_col: str,
    property_name: str,
    source_sheet: str,
    id_col: str | None = None,
    authorized_ts_col: str | None = None,
) -> pd.DataFrame:
    cols = [sample_ts_col, value_col]
    if id_col and id_col in df.columns:
        cols.append(id_col)
    if authorized_ts_col and authorized_ts_col in df.columns:
        cols.append(authorized_ts_col)
    out = df[cols].copy()
    out['sample_ts'] = pd.to_datetime(out[sample_ts_col], errors='coerce')
    out['value'] = pd.to_numeric(out[value_col], errors='coerce')
    out['property_name'] = property_name
    out['source_sheet'] = source_sheet
    out['raw_id_text'] = out[id_col] if id_col and id_col in out.columns else pd.Series([None] * len(out), index=out.index)
    out['authorized_ts'] = pd.to_datetime(out[authorized_ts_col], errors='coerce') if authorized_ts_col and authorized_ts_col in out.columns else pd.NaT
    out = out[['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts']]
    return out.dropna(subset=['sample_ts', 'value']).sort_values('sample_ts').reset_index(drop=True)


def normalize_lims_common(df: pd.DataFrame) -> pd.DataFrame:
    body = _normalize_sheet_with_header_row(df, header_row=1)
    rename_map = {
        'Id text': 'id_text',
        'Product': 'product',
        'Sample name': 'sample_name',
        'Sample category': 'sample_category',
        'Component name': 'component_name',
        'Result text': 'value_raw',
        'Date result authorised': 'authorized_ts',
        'Units': 'units',
    }
    body = body.rename(columns={k: v for k, v in rename_map.items() if k in body.columns})
    body['sample_ts'] = body['id_text'].map(parse_lims_common_sample_time)
    body['authorized_ts'] = pd.to_datetime(body.get('authorized_ts'), errors='coerce')
    body['value_num'] = pd.to_numeric(body.get('value_raw'), errors='coerce')
    keep = [
        'product', 'sample_name', 'sample_category', 'component_name',
        'value_raw', 'value_num', 'units', 'id_text', 'sample_ts', 'authorized_ts',
    ]
    keep = [c for c in keep if c in body.columns]
    return body[keep].dropna(subset=['sample_ts']).sort_values('sample_ts').reset_index(drop=True)


def normalize_pda_data(df: pd.DataFrame) -> pd.DataFrame:
    body = _normalize_sheet_with_header_row(df, header_row=1)
    rename_map = {
        'Timestamp': 'event_ts',
        'fc4101a.PV - Average': 'feed_flow_a_m3hr',
        'fc4101b.PV - Average': 'feed_flow_b_m3hr',
        '41ti41132.PV - Average': 'feed_temp_a_C',
        '41ti41133.PV - Average': 'feed_temp_b_C',
        '41ti41101.PV - Average': 'propane_temp_C',
        'tc4102a.pv - Average': 'top_temp_a_C',
        'tc4102b.pv - Average': 'top_temp_b_C',
        'ti4108a.pv - Average': 'mid_temp_a_C',
        'ti4108b.pv - Average': 'mid_temp_b_C',
        'ti4110a.pv - Average': 'bottom_temp_a_C',
        'ti4110b.pv - Average': 'bottom_temp_b_C',
        '41fic4103a.pv - Average': 'primary_prop_a',
        '41fic4103b.pv - Average': 'primary_prop_b',
        '41fic41113.pv - Average': 'secondary_prop_a',
        '41fic41114.pv - Average': 'secondary_prop_b',
        '41fic41110.pv - Average': 'dao_flow_m3hr',
        '41fi4119.pv - Average': 'asphalt_flow_m3hr',
    }
    body = body.rename(columns={k: v for k, v in rename_map.items() if k in body.columns})
    body['event_ts'] = pd.to_datetime(body['event_ts'], errors='coerce')
    numeric_cols = [v for v in rename_map.values() if v in body.columns and v != 'event_ts']
    body = coerce_numeric_columns(body, numeric_cols)
    keep = ['event_ts'] + numeric_cols
    return body[keep].dropna(subset=['event_ts']).sort_values('event_ts').reset_index(drop=True)


def build_hourly_process_table(pda_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_pda_data(pda_df)
    df['feed_flow_total_m3hr'] = df['feed_flow_a_m3hr'] + df['feed_flow_b_m3hr']
    df['primary_prop_total'] = df['primary_prop_a'] + df['primary_prop_b']
    df['secondary_prop_total'] = df['secondary_prop_a'] + df['secondary_prop_b']
    df['prop_total'] = df['primary_prop_total'] + df['secondary_prop_total']
    df['so_ratio_a'] = np.where(df['feed_flow_a_m3hr'] > 0, (df['primary_prop_a'] + df['secondary_prop_a']) / df['feed_flow_a_m3hr'], np.nan)
    df['so_ratio_b'] = np.where(df['feed_flow_b_m3hr'] > 0, (df['primary_prop_b'] + df['secondary_prop_b']) / df['feed_flow_b_m3hr'], np.nan)
    df['predilution_frac_a'] = np.where((df['primary_prop_a'] + df['secondary_prop_a']) > 0, df['secondary_prop_a'] / (df['primary_prop_a'] + df['secondary_prop_a']), np.nan)
    df['predilution_frac_b'] = np.where((df['primary_prop_b'] + df['secondary_prop_b']) > 0, df['secondary_prop_b'] / (df['primary_prop_b'] + df['secondary_prop_b']), np.nan)
    df['valid_train_a'] = df[['feed_flow_a_m3hr', 'so_ratio_a', 'top_temp_a_C', 'bottom_temp_a_C']].notna().all(axis=1)
    df['valid_train_b'] = df[['feed_flow_b_m3hr', 'so_ratio_b', 'top_temp_b_C', 'bottom_temp_b_C']].notna().all(axis=1)
    return df


def extract_feed_density_table(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    body = _normalize_sheet_with_header_row(raw_tables['feed_density_raw'], header_row=1)
    return normalize_simple_lims_sheet(body, 'Sampled Date', 'Result Value', 'feed_density_kg_m3', 'feed desity at 15degC')


def extract_feed_ccr_table(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    body = _normalize_sheet_with_header_row(raw_tables['feed_ccr_raw'], header_row=1)
    preferred = normalize_simple_lims_sheet(body, 'Sampled Date', 'Result Value', 'feed_ccr_wt_pct', 'feed ccr wt%')
    common = normalize_lims_common(raw_tables['lims_common_raw'])
    fallback = common[
        (common['product'] == 'D_PDA_FEED') &
        (common['component_name'] == 'Ramsbottom Carbon Residue') &
        (common['value_num'].notna())
    ].copy()
    if not fallback.empty:
        fallback = fallback.rename(columns={'value_num': 'value', 'id_text': 'raw_id_text'})
        fallback['property_name'] = 'feed_ccr_wt_pct'
        fallback['source_sheet'] = 'lims pda common'
        fallback = fallback[['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts']]
    combined = pd.concat([preferred, fallback], ignore_index=True)
    return combined.sort_values('sample_ts').drop_duplicates(subset=['sample_ts', 'value'], keep='first').reset_index(drop=True)


def extract_feed_visc_135_table(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    body = _normalize_sheet_with_header_row(raw_tables['feed_visc_135_raw'], header_row=1)
    value_col = next(c for c in body.columns if 'viscosity' in c.lower())
    preferred = normalize_simple_lims_sheet(body, 'Sampled Date', value_col, 'feed_visc_135_cst', 'feed kin viscosity at 135 degC')
    common = normalize_lims_common(raw_tables['lims_common_raw'])
    fallback = common[
        (common['product'] == 'D_PDA_FEED') &
        (common['component_name'] == 'Viscosity Kinematic @135 deg.C') &
        (common['value_num'].notna())
    ].copy()
    if not fallback.empty:
        fallback = fallback.rename(columns={'value_num': 'value', 'id_text': 'raw_id_text'})
        fallback['property_name'] = 'feed_visc_135_cst'
        fallback['source_sheet'] = 'lims pda common'
        fallback = fallback[['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts']]
    combined = pd.concat([preferred, fallback], ignore_index=True)
    return combined.sort_values('sample_ts').drop_duplicates(subset=['sample_ts', 'value'], keep='first').reset_index(drop=True)


def extract_dao_visc_100_table(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    body = _normalize_sheet_with_header_row(raw_tables['dao_visc_100_raw'], header_row=1)
    value_col = next(c for c in body.columns if 'viscosity' in c.lower())
    preferred = normalize_simple_lims_sheet(body, 'Sampled Date', value_col, 'dao_visc_100_cst', 'dao viscosity')
    common = normalize_lims_common(raw_tables['lims_common_raw'])
    fallback = common[
        (common['product'] == 'D_DAO') &
        (common['component_name'] == 'Viscosity Kinematic @ 100 deg.C') &
        (common['value_num'].notna())
    ].copy()
    if not fallback.empty:
        fallback = fallback.rename(columns={'value_num': 'value', 'id_text': 'raw_id_text'})
        fallback['property_name'] = 'dao_visc_100_cst'
        fallback['source_sheet'] = 'lims pda common'
        fallback = fallback[['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts']]
    combined = pd.concat([preferred, fallback], ignore_index=True)
    return combined.sort_values('sample_ts').drop_duplicates(subset=['sample_ts', 'value'], keep='first').reset_index(drop=True)


def extract_dao_ccr_table(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = normalize_lims_common(raw_tables['lims_common_raw'])
    priority = [
        'Carbon Residue, Micro method',
        'Carbon residue (Ramsbottom) on whole sam',
        'Carbon Residue, Ramsbottom',
    ]
    pieces = []
    for rank, comp in enumerate(priority):
        sub = common[
            (common['product'] == 'D_DAO') &
            (common['component_name'] == comp) &
            (common['value_num'].notna())
        ].copy()
        sub['rank'] = rank
        pieces.append(sub)
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if combined.empty:
        return pd.DataFrame(columns=['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts'])
    combined = combined.sort_values(['sample_ts', 'rank']).drop_duplicates(subset=['sample_ts'], keep='first')
    combined = combined.rename(columns={'value_num': 'value', 'id_text': 'raw_id_text'})
    combined['property_name'] = 'dao_ccr_wt_pct'
    combined['source_sheet'] = 'lims pda common'
    return combined[['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts']].reset_index(drop=True)


def extract_dao_asph_table(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = normalize_lims_common(raw_tables['lims_common_raw'])
    sub = common[
        (common['product'] == 'D_DAO') &
        (common['component_name'] == 'Asphaltene') &
        (common['value_num'].notna())
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts'])
    sub = sub.rename(columns={'value_num': 'value', 'id_text': 'raw_id_text'})
    sub['property_name'] = 'dao_asphaltene_wt_pct'
    sub['source_sheet'] = 'lims pda common'
    return sub[['sample_ts', 'value', 'property_name', 'source_sheet', 'raw_id_text', 'authorized_ts']].reset_index(drop=True)


def attach_feed_lab_properties(
    process_df: pd.DataFrame,
    feed_density_df: pd.DataFrame,
    feed_ccr_df: pd.DataFrame,
    feed_visc_135_df: pd.DataFrame,
    tolerances: JoinToleranceConfig,
) -> pd.DataFrame:
    df = process_df.sort_values('event_ts').copy()

    density = feed_density_df.rename(columns={'sample_ts': 'feed_density_sample_ts', 'value': 'feed_density_kg_m3'})
    df = pd.merge_asof(df, density[['feed_density_sample_ts', 'feed_density_kg_m3']].sort_values('feed_density_sample_ts'), left_on='event_ts', right_on='feed_density_sample_ts', direction='nearest', tolerance=tolerances.feed_density)
    df['feed_density_age_hr'] = (df['event_ts'] - df['feed_density_sample_ts']).abs().dt.total_seconds() / 3600.0
    df['feed_density_is_stale'] = df['feed_density_age_hr'] > (tolerances.feed_density.total_seconds() / 3600.0)

    ccr = feed_ccr_df.rename(columns={'sample_ts': 'feed_ccr_sample_ts', 'value': 'feed_CCR_wt_pct'})
    df = pd.merge_asof(df, ccr[['feed_ccr_sample_ts', 'feed_CCR_wt_pct']].sort_values('feed_ccr_sample_ts'), left_on='event_ts', right_on='feed_ccr_sample_ts', direction='nearest', tolerance=tolerances.feed_ccr)
    df['feed_ccr_age_hr'] = (df['event_ts'] - df['feed_ccr_sample_ts']).abs().dt.total_seconds() / 3600.0
    df['feed_ccr_is_stale'] = df['feed_ccr_age_hr'] > (tolerances.feed_ccr.total_seconds() / 3600.0)

    visc = feed_visc_135_df.rename(columns={'sample_ts': 'feed_visc_135_sample_ts', 'value': 'feed_visc_135_cst'})
    df = pd.merge_asof(df, visc[['feed_visc_135_sample_ts', 'feed_visc_135_cst']].sort_values('feed_visc_135_sample_ts'), left_on='event_ts', right_on='feed_visc_135_sample_ts', direction='nearest', tolerance=tolerances.feed_visc_135)
    df['feed_visc_135_age_hr'] = (df['event_ts'] - df['feed_visc_135_sample_ts']).abs().dt.total_seconds() / 3600.0
    df['feed_visc_135_is_stale'] = df['feed_visc_135_age_hr'] > (tolerances.feed_visc_135.total_seconds() / 3600.0)

    return df


def _attach_one_dao_property(process_df: pd.DataFrame, prop_df: pd.DataFrame, value_name: str, sample_name: str, auth_name: str, tolerance: pd.Timedelta, lag: pd.Timedelta) -> pd.DataFrame:
    match_age_name = f'{value_name}_match_age_hr'
    matched_name = f'{value_name}_matched'
    if prop_df.empty:
        out = process_df.copy()
        out[value_name] = np.nan
        out[sample_name] = pd.NaT
        out[auth_name] = pd.NaT
        out[match_age_name] = np.nan
        out[matched_name] = False
        return out
    tmp = prop_df.copy()
    tmp['effective_event_ts'] = tmp['sample_ts'] - lag
    tmp = tmp.rename(columns={'value': value_name, 'sample_ts': sample_name, 'authorized_ts': auth_name})
    merged = pd.merge_asof(
        process_df.sort_values('event_ts'),
        tmp[['effective_event_ts', value_name, sample_name, auth_name]].sort_values('effective_event_ts'),
        left_on='event_ts', right_on='effective_event_ts', direction='nearest', tolerance=tolerance,
    )
    merged[match_age_name] = (merged['event_ts'] - merged['effective_event_ts']).abs().dt.total_seconds() / 3600.0
    merged[matched_name] = merged[value_name].notna()
    return merged.drop(columns=['effective_event_ts'])


def attach_dao_lab_properties(
    process_df: pd.DataFrame,
    dao_visc_df: pd.DataFrame,
    dao_ccr_df: pd.DataFrame,
    dao_asph_df: pd.DataFrame,
    tolerances: JoinToleranceConfig,
) -> pd.DataFrame:
    df = _attach_one_dao_property(process_df, dao_visc_df, 'dao_visc_100_cst', 'dao_visc_sample_ts', 'dao_visc_authorized_ts', tolerances.dao_response, tolerances.dao_lag)
    df = _attach_one_dao_property(df, dao_ccr_df, 'dao_ccr_wt_pct', 'dao_ccr_sample_ts', 'dao_ccr_authorized_ts', tolerances.dao_response, tolerances.dao_lag)
    df = _attach_one_dao_property(df, dao_asph_df, 'dao_asphaltene_wt_pct', 'dao_asph_sample_ts', 'dao_asph_authorized_ts', tolerances.dao_response, tolerances.dao_lag)
    return df


def compute_internal_yield_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['dao_yield_vol_pct'] = np.where(out['feed_flow_total_m3hr'] > 0, 100.0 * out['dao_flow_m3hr'] / out['feed_flow_total_m3hr'], np.nan)
    out['yield_basis_flag'] = 'volume'
    return out


def build_row_usability_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['has_feed_density'] = out['feed_density_kg_m3'].notna()
    out['has_feed_ccr'] = out['feed_CCR_wt_pct'].notna()
    out['has_feed_visc_135'] = out['feed_visc_135_cst'].notna()
    out['has_dao_yield'] = out['dao_yield_vol_pct'].notna()
    out['has_dao_visc'] = out['dao_visc_100_cst'].notna()
    out['has_dao_ccr'] = out['dao_ccr_wt_pct'].notna()
    out['has_dao_asph'] = out['dao_asphaltene_wt_pct'].notna()
    out['usable_core_row'] = (
        out['valid_train_a'] & out['valid_train_b'] &
        out['has_feed_density'] & out['has_feed_ccr'] & out['has_feed_visc_135'] &
        out['has_dao_yield'] & out['has_dao_visc'] &
        (~out['feed_density_is_stale']) & (~out['feed_ccr_is_stale']) & (~out['feed_visc_135_is_stale'])
    )
    out['usable_ccr_row'] = out['usable_core_row'] & out['has_dao_ccr']
    out['usable_asph_row'] = out['usable_core_row'] & out['has_dao_asph']
    return out


def compute_normalized_dataset_diagnostics(df: pd.DataFrame) -> NormalizedDatasetDiagnostics:
    def _max_or_none(series: pd.Series) -> float | None:
        clean = series.dropna()
        return None if clean.empty else float(clean.max())

    return NormalizedDatasetDiagnostics(
        usable_core_rows=int(df['usable_core_row'].sum()),
        usable_ccr_rows=int(df['usable_ccr_row'].sum()),
        usable_asph_rows=int(df['usable_asph_row'].sum()),
        feed_density_missing_rows=int(df['feed_density_kg_m3'].isna().sum()),
        feed_ccr_missing_rows=int(df['feed_CCR_wt_pct'].isna().sum()),
        feed_visc_missing_rows=int(df['feed_visc_135_cst'].isna().sum()),
        dao_visc_missing_rows=int(df['dao_visc_100_cst'].isna().sum()),
        dao_ccr_missing_rows=int(df['dao_ccr_wt_pct'].isna().sum()),
        dao_asph_missing_rows=int(df['dao_asphaltene_wt_pct'].isna().sum()),
        max_feed_density_age_hr=_max_or_none(df['feed_density_age_hr']),
        max_feed_ccr_age_hr=_max_or_none(df['feed_ccr_age_hr']),
        max_feed_visc_age_hr=_max_or_none(df['feed_visc_135_age_hr']),
        max_dao_visc_match_age_hr=_max_or_none(df['dao_visc_100_cst_match_age_hr']),
        max_dao_ccr_match_age_hr=_max_or_none(df['dao_ccr_wt_pct_match_age_hr']),
        max_dao_asph_match_age_hr=_max_or_none(df['dao_asphaltene_wt_pct_match_age_hr']),
    )


def build_normalized_parallel_dataset(
    lims_path: str,
    extractor_path: str,
    tolerances: JoinToleranceConfig | None = None,
) -> tuple[pd.DataFrame, NormalizedDatasetSummary]:
    if tolerances is None:
        tolerances = JoinToleranceConfig()
    lims_raw = load_lims_workbook(lims_path)
    ext_raw = load_extractor_workbook(extractor_path)
    process = build_hourly_process_table(ext_raw['pda_data_raw'])
    feed_density = extract_feed_density_table(lims_raw)
    feed_ccr = extract_feed_ccr_table(lims_raw)
    feed_visc_135 = extract_feed_visc_135_table(lims_raw)
    dao_visc = extract_dao_visc_100_table(lims_raw)
    dao_ccr = extract_dao_ccr_table(lims_raw)
    dao_asph = extract_dao_asph_table(lims_raw)
    df = attach_feed_lab_properties(process, feed_density, feed_ccr, feed_visc_135, tolerances)
    df = attach_dao_lab_properties(df, dao_visc, dao_ccr, dao_asph, tolerances)
    df = compute_internal_yield_targets(df)
    df = build_row_usability_flags(df)
    summary = NormalizedDatasetSummary(
        n_rows_total=len(df),
        n_rows_core=int(df['usable_core_row'].sum()),
        n_rows_ccr=int(df['usable_ccr_row'].sum()),
        n_rows_asph=int(df['usable_asph_row'].sum()),
        min_event_ts=df['event_ts'].min() if not df.empty else None,
        max_event_ts=df['event_ts'].max() if not df.empty else None,
    )
    return df, summary
