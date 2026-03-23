import unittest
import pandas as pd
import numpy as np

from data_ingestion import (
    JoinToleranceConfig,
    build_hourly_process_table,
    attach_dao_lab_properties,
    build_row_usability_flags,
    compute_normalized_dataset_diagnostics,
    parse_lims_common_sample_time,
)


class DataIngestionTests(unittest.TestCase):
    def test_parse_lims_common_sample_time(self):
        ts = parse_lims_common_sample_time('R-22-MAR-26 17:00-PDA-R030308')
        self.assertEqual(ts, pd.Timestamp(2026, 3, 22, 17, 0))
        self.assertTrue(pd.isna(parse_lims_common_sample_time('bad-format')))

    def test_build_hourly_process_table_derives_train_features(self):
        raw = pd.DataFrame([
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 'Timestamp', 'fc4101a.PV - Average', 'fc4101b.PV - Average', '41ti41132.PV - Average', '41ti41133.PV - Average', '41ti41101.PV - Average', 'tc4102a.pv - Average', 'tc4102b.pv - Average', 'ti4108a.pv - Average', 'ti4108b.pv - Average', 'ti4110a.pv - Average', 'ti4110b.pv - Average', '41fic4103a.pv - Average', '41fic4103b.pv - Average', '41fic41113.pv - Average', '41fic41114.pv - Average', '41fic41110.pv - Average', '41fi4119.pv - Average'],
            [np.nan, '2026-01-01 00:00:00', 40, 20, 85, 83, 60, 90, 89, 78, 77, 70, 69, 200, 100, 40, 20, 10, 50],
        ])
        out = build_hourly_process_table(raw)
        self.assertAlmostEqual(out.loc[0, 'feed_flow_total_m3hr'], 60.0)
        self.assertAlmostEqual(out.loc[0, 'so_ratio_a'], 6.0)
        self.assertAlmostEqual(out.loc[0, 'so_ratio_b'], 6.0)
        self.assertAlmostEqual(out.loc[0, 'predilution_frac_a'], 40 / 240)
        self.assertAlmostEqual(out.loc[0, 'predilution_frac_b'], 20 / 120)
        self.assertTrue(bool(out.loc[0, 'valid_train_a']))
        self.assertTrue(bool(out.loc[0, 'valid_train_b']))

    def test_attach_dao_lab_properties_applies_three_hour_lag(self):
        process = pd.DataFrame({'event_ts': pd.to_datetime(['2026-01-01 00:00:00', '2026-01-01 01:00:00'])})
        dao_visc = pd.DataFrame({
            'sample_ts': pd.to_datetime(['2026-01-01 03:00:00']),
            'value': [25.0],
            'authorized_ts': pd.to_datetime(['2026-01-01 05:00:00']),
        })
        empty = pd.DataFrame(columns=['sample_ts', 'value', 'authorized_ts'])
        out = attach_dao_lab_properties(process, dao_visc, empty, empty, JoinToleranceConfig())
        self.assertEqual(out.loc[0, 'dao_visc_100_cst'], 25.0)
        self.assertTrue(pd.isna(out.loc[1, 'dao_visc_100_cst']))
        self.assertEqual(out.loc[0, 'dao_visc_sample_ts'], pd.Timestamp('2026-01-01 03:00:00'))
        self.assertEqual(out.loc[0, 'dao_visc_100_cst_match_age_hr'], 0.0)
        self.assertTrue(bool(out.loc[0, 'dao_visc_100_cst_matched']))

    def test_build_row_usability_flags_and_diagnostics(self):
        df = pd.DataFrame({
            'valid_train_a': [True, True],
            'valid_train_b': [True, True],
            'feed_density_kg_m3': [1028.0, np.nan],
            'feed_CCR_wt_pct': [22.8, 22.8],
            'feed_visc_135_cst': [220.0, 220.0],
            'dao_yield_vol_pct': [15.0, 15.0],
            'dao_visc_100_cst': [30.0, np.nan],
            'dao_ccr_wt_pct': [np.nan, np.nan],
            'dao_asphaltene_wt_pct': [np.nan, np.nan],
            'feed_density_is_stale': [False, False],
            'feed_ccr_is_stale': [False, False],
            'feed_visc_135_is_stale': [False, False],
            'feed_density_age_hr': [5.0, np.nan],
            'feed_ccr_age_hr': [6.0, 6.0],
            'feed_visc_135_age_hr': [4.0, 4.0],
            'dao_visc_100_cst_match_age_hr': [0.0, np.nan],
            'dao_ccr_wt_pct_match_age_hr': [np.nan, np.nan],
            'dao_asphaltene_wt_pct_match_age_hr': [np.nan, np.nan],
        })
        flagged = build_row_usability_flags(df)
        self.assertTrue(bool(flagged.loc[0, 'usable_core_row']))
        self.assertFalse(bool(flagged.loc[1, 'usable_core_row']))
        diag = compute_normalized_dataset_diagnostics(flagged)
        self.assertEqual(diag.usable_core_rows, 1)
        self.assertEqual(diag.feed_density_missing_rows, 1)
        self.assertEqual(diag.dao_visc_missing_rows, 1)
        self.assertEqual(diag.max_dao_visc_match_age_hr, 0.0)


if __name__ == '__main__':
    unittest.main()
