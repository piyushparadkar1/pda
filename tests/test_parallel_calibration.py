import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from parallel_calibration import (
    DEFAULT_PARAMS,
    ParallelCalibrationWeights,
    build_parallel_residuals,
    compute_parallel_metrics,
    load_parallel_calibration_profile,
    points_from_normalized_dataframe,
    run_parallel_calibration,
)


class ParallelCalibrationTests(unittest.TestCase):
    def test_points_from_normalized_dataframe_filters_usable_rows(self):
        df = pd.DataFrame([
            {
                'event_ts': pd.Timestamp('2026-01-01 00:00:00'),
                'usable_core_row': True,
                'feed_density_kg_m3': 1028.0,
                'feed_CCR_wt_pct': 22.8,
                'feed_visc_135_cst': 230.0,
                'feed_flow_a_m3hr': 40.0,
                'feed_flow_b_m3hr': 35.0,
                'feed_temp_a_C': 84.0,
                'feed_temp_b_C': 83.0,
                'top_temp_a_C': 92.0,
                'top_temp_b_C': 91.0,
                'mid_temp_a_C': 80.0,
                'mid_temp_b_C': 79.0,
                'bottom_temp_a_C': 72.0,
                'bottom_temp_b_C': 71.0,
                'primary_prop_a': 180.0,
                'primary_prop_b': 155.0,
                'secondary_prop_a': 45.0,
                'secondary_prop_b': 40.0,
                'dao_yield_vol_pct': 18.0,
                'dao_visc_100_cst': 32.0,
                'dao_ccr_wt_pct': np.nan,
                'dao_asphaltene_wt_pct': np.nan,
            },
            {
                'event_ts': pd.Timestamp('2026-01-01 01:00:00'),
                'usable_core_row': False,
                'feed_density_kg_m3': 1028.0,
                'feed_CCR_wt_pct': 22.8,
                'feed_visc_135_cst': 230.0,
                'feed_flow_a_m3hr': 40.0,
                'feed_flow_b_m3hr': 35.0,
                'feed_temp_a_C': 84.0,
                'feed_temp_b_C': 83.0,
                'top_temp_a_C': 92.0,
                'top_temp_b_C': 91.0,
                'mid_temp_a_C': 80.0,
                'mid_temp_b_C': 79.0,
                'bottom_temp_a_C': 72.0,
                'bottom_temp_b_C': 71.0,
                'primary_prop_a': 180.0,
                'primary_prop_b': 155.0,
                'secondary_prop_a': 45.0,
                'secondary_prop_b': 40.0,
                'dao_yield_vol_pct': 20.0,
                'dao_visc_100_cst': 35.0,
                'dao_ccr_wt_pct': np.nan,
                'dao_asphaltene_wt_pct': np.nan,
            },
        ])
        pts = points_from_normalized_dataframe(df)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0].timestamp, '2026-01-01 00:00:00')
        self.assertEqual(pts[0].dao_visc_100_cst, 32.0)

    def test_build_parallel_residuals_and_metrics_with_stubbed_predictions(self):
        pts = points_from_normalized_dataframe(pd.DataFrame([
            {
                'event_ts': pd.Timestamp('2026-01-01 00:00:00'),
                'usable_core_row': True,
                'feed_density_kg_m3': 1028.0,
                'feed_CCR_wt_pct': 22.8,
                'feed_visc_135_cst': 230.0,
                'feed_flow_a_m3hr': 40.0,
                'feed_flow_b_m3hr': 35.0,
                'feed_temp_a_C': 84.0,
                'feed_temp_b_C': 83.0,
                'top_temp_a_C': 92.0,
                'top_temp_b_C': 91.0,
                'mid_temp_a_C': 80.0,
                'mid_temp_b_C': 79.0,
                'bottom_temp_a_C': 72.0,
                'bottom_temp_b_C': 71.0,
                'primary_prop_a': 180.0,
                'primary_prop_b': 155.0,
                'secondary_prop_a': 45.0,
                'secondary_prop_b': 40.0,
                'dao_yield_vol_pct': 18.0,
                'dao_visc_100_cst': 32.0,
                'dao_ccr_wt_pct': 1.5,
                'dao_asphaltene_wt_pct': 0.2,
            },
        ]))
        weights = ParallelCalibrationWeights(DAO_yield=1.0, DAO_viscosity=0.5, DAO_CCR=2.0, DAO_asphaltene=3.0)
        history = []
        with patch('parallel_calibration.simulate_parallel_point', return_value={
            'DAO_yield': 20.0,
            'DAO_viscosity': 30.0,
            'DAO_CCR': 1.0,
            'DAO_asphaltene': 0.1,
            'train_a_converged': True,
            'train_b_converged': True,
            'converged': True,
            'raw_prediction': None,
        }):
            residuals = build_parallel_residuals(
                np.array(list(DEFAULT_PARAMS.values()), dtype=float),
                pts,
                weights,
                history,
            )
            metrics, results = compute_parallel_metrics(np.array(list(DEFAULT_PARAMS.values()), dtype=float), pts)
        np.testing.assert_allclose(residuals, np.array([2.0, -1.0, -1.0, -0.3]))
        self.assertEqual(len(history), 1)
        self.assertEqual(metrics['DAO_yield']['MAE'], 2.0)
        self.assertEqual(metrics['DAO_viscosity']['bias'], -2.0)
        self.assertEqual(results[0]['error']['DAO_CCR'], -0.5)
        self.assertEqual(metrics['convergence']['converged_points'], 1)
        self.assertTrue(results[0]['included_in_metrics'])

    def test_build_parallel_residuals_penalizes_nonconverged_predictions(self):
        pts = points_from_normalized_dataframe(pd.DataFrame([
            {
                'event_ts': pd.Timestamp('2026-01-01 00:00:00'),
                'usable_core_row': True,
                'feed_density_kg_m3': 1028.0,
                'feed_CCR_wt_pct': 22.8,
                'feed_visc_135_cst': 230.0,
                'feed_flow_a_m3hr': 40.0,
                'feed_flow_b_m3hr': 35.0,
                'feed_temp_a_C': 84.0,
                'feed_temp_b_C': 83.0,
                'top_temp_a_C': 92.0,
                'top_temp_b_C': 91.0,
                'mid_temp_a_C': 80.0,
                'mid_temp_b_C': 79.0,
                'bottom_temp_a_C': 72.0,
                'bottom_temp_b_C': 71.0,
                'primary_prop_a': 180.0,
                'primary_prop_b': 155.0,
                'secondary_prop_a': 45.0,
                'secondary_prop_b': 40.0,
                'dao_yield_vol_pct': 18.0,
                'dao_visc_100_cst': 32.0,
                'dao_ccr_wt_pct': np.nan,
                'dao_asphaltene_wt_pct': np.nan,
            },
        ]))
        weights = ParallelCalibrationWeights(non_convergence_penalty=17.5)
        history = []
        with patch('parallel_calibration.simulate_parallel_point', return_value={
            'DAO_yield': 999.0,
            'DAO_viscosity': 999.0,
            'DAO_CCR': 999.0,
            'DAO_asphaltene': 999.0,
            'train_a_converged': False,
            'train_b_converged': True,
            'converged': False,
            'raw_prediction': None,
        }):
            residuals = build_parallel_residuals(
                np.array(list(DEFAULT_PARAMS.values()), dtype=float),
                pts,
                weights,
                history,
            )
            metrics, results = compute_parallel_metrics(np.array(list(DEFAULT_PARAMS.values()), dtype=float), pts)
        np.testing.assert_allclose(residuals, np.array([17.5]))
        self.assertEqual(history[0]['cost'], 17.5 ** 2)
        self.assertEqual(metrics['convergence']['nonconverged_points'], 1)
        self.assertFalse(results[0]['included_in_metrics'])
        self.assertFalse(results[0]['train_a_converged'])

    def test_run_parallel_calibration_smoke_with_stubbed_simulator(self):
        pts = points_from_normalized_dataframe(pd.DataFrame([
            {
                'event_ts': pd.Timestamp('2026-01-01 00:00:00'),
                'usable_core_row': True,
                'feed_density_kg_m3': 1028.0,
                'feed_CCR_wt_pct': 22.8,
                'feed_visc_135_cst': 230.0,
                'feed_flow_a_m3hr': 40.0,
                'feed_flow_b_m3hr': 35.0,
                'feed_temp_a_C': 84.0,
                'feed_temp_b_C': 83.0,
                'top_temp_a_C': 92.0,
                'top_temp_b_C': 91.0,
                'mid_temp_a_C': 80.0,
                'mid_temp_b_C': 79.0,
                'bottom_temp_a_C': 72.0,
                'bottom_temp_b_C': 71.0,
                'primary_prop_a': 180.0,
                'primary_prop_b': 155.0,
                'secondary_prop_a': 45.0,
                'secondary_prop_b': 40.0,
                'dao_yield_vol_pct': 18.0,
                'dao_visc_100_cst': 32.0,
                'dao_ccr_wt_pct': np.nan,
                'dao_asphaltene_wt_pct': np.nan,
            },
        ]))

        def fake_sim(pt, params, solvent_name='propane', pressure_bar=40.0, n_stages=4):
            offset = params['K_multiplier'] - 1.5
            return {
                'DAO_yield': 18.0 + offset,
                'DAO_viscosity': 32.0 + 2.0 * offset,
                'DAO_CCR': 0.0,
                'DAO_asphaltene': 0.0,
                'train_a_converged': True,
                'train_b_converged': True,
                'converged': True,
                'raw_prediction': None,
            }

        with patch('parallel_calibration.simulate_parallel_point', side_effect=fake_sim):
            result = run_parallel_calibration(pts, max_nfev=5, verbose=False)
        self.assertTrue(result.success)
        self.assertIn('K_multiplier', result.calibrated_params)
        self.assertGreaterEqual(result.n_function_evals, 1)
        self.assertIn('DAO_yield', result.metrics)
        self.assertEqual(result.n_nonconverged_points, 0)

    def test_run_parallel_calibration_with_active_params_only_updates_subset(self):
        pts = points_from_normalized_dataframe(pd.DataFrame([
            {
                'event_ts': pd.Timestamp('2026-01-01 00:00:00'),
                'usable_core_row': True,
                'feed_density_kg_m3': 1028.0,
                'feed_CCR_wt_pct': 22.8,
                'feed_visc_135_cst': 230.0,
                'feed_flow_a_m3hr': 40.0,
                'feed_flow_b_m3hr': 35.0,
                'feed_temp_a_C': 84.0,
                'feed_temp_b_C': 83.0,
                'top_temp_a_C': 92.0,
                'top_temp_b_C': 91.0,
                'mid_temp_a_C': 80.0,
                'mid_temp_b_C': 79.0,
                'bottom_temp_a_C': 72.0,
                'bottom_temp_b_C': 71.0,
                'primary_prop_a': 180.0,
                'primary_prop_b': 155.0,
                'secondary_prop_a': 45.0,
                'secondary_prop_b': 40.0,
                'dao_yield_vol_pct': 18.0,
                'dao_visc_100_cst': 32.0,
                'dao_ccr_wt_pct': np.nan,
                'dao_asphaltene_wt_pct': np.nan,
            },
        ]))

        def fake_sim(pt, params, solvent_name='propane', pressure_bar=40.0, n_stages=4):
            offset = params['K_multiplier'] - 1.25
            return {
                'DAO_yield': 18.0 + offset,
                'DAO_viscosity': 32.0 + offset,
                'DAO_CCR': 0.0,
                'DAO_asphaltene': 0.0,
                'train_a_converged': True,
                'train_b_converged': True,
                'converged': True,
                'raw_prediction': None,
            }

        with patch('parallel_calibration.simulate_parallel_point', side_effect=fake_sim):
            result = run_parallel_calibration(
                pts,
                active_params=['K_multiplier'],
                max_nfev=5,
                verbose=False,
            )
        self.assertEqual(result.active_params, ['K_multiplier'])
        self.assertIn('alpha_density', result.calibrated_params)
        self.assertAlmostEqual(result.calibrated_params['alpha_density'], DEFAULT_PARAMS['alpha_density'])

    def test_load_parallel_calibration_profile_by_name(self):
        profile = load_parallel_calibration_profile('parallel_usable_core_v1')
        self.assertEqual(profile['profile_name'], 'parallel_usable_core_v1')
        self.assertEqual(profile['row_flag'], 'usable_core_row')
        self.assertIn('K_multiplier', profile['parameters'])


if __name__ == '__main__':
    unittest.main()
