import math
import unittest

import numpy as np
import pandas as pd

from parallel_extractor_model import (
    blend_train_results,
    build_train_case,
    convert_visc_135_to_visc_100,
    run_parallel_extractors_and_blend,
)
from residue_distribution import PseudoComponent


class ParallelExtractorModelTests(unittest.TestCase):
    def test_convert_visc_135_to_visc_100_matches_repo_feed_anchors(self):
        self.assertAlmostEqual(convert_visc_135_to_visc_100(230.0), 1621.0, places=6)
        self.assertAlmostEqual(convert_visc_135_to_visc_100(177.0), 1137.0, places=6)
        self.assertIsNone(convert_visc_135_to_visc_100(None))
        self.assertIsNone(convert_visc_135_to_visc_100(0.0))

    def test_build_train_case_computes_train_specific_inputs(self):
        row = pd.Series({
            'feed_density_kg_m3': 1028.0,
            'feed_CCR_wt_pct': 22.8,
            'feed_visc_135_cst': 230.0,
            'feed_flow_a_m3hr': 40.0,
            'feed_flow_b_m3hr': 20.0,
            'feed_temp_a_C': 82.0,
            'feed_temp_b_C': 80.0,
            'top_temp_a_C': 90.0,
            'top_temp_b_C': 88.0,
            'mid_temp_a_C': 78.0,
            'mid_temp_b_C': 76.0,
            'bottom_temp_a_C': 70.0,
            'bottom_temp_b_C': 68.0,
            'primary_prop_a': 180.0,
            'primary_prop_b': 90.0,
            'secondary_prop_a': 60.0,
            'secondary_prop_b': 30.0,
            'propane_temp_C': 55.0,
        })
        case = build_train_case(row, 'A', pressure_bar=41.5, n_stages=5)
        self.assertEqual(case.train_id, 'A')
        self.assertAlmostEqual(case.feed_mass_basis_kg_hr, 40.0 * 1028.0)
        self.assertAlmostEqual(case.so_ratio, (180.0 + 60.0) / 40.0)
        self.assertAlmostEqual(case.predilution_frac, 60.0 / 240.0)
        self.assertEqual(case.pressure_bar, 41.5)
        self.assertEqual(case.n_stages, 5)
        self.assertEqual(case.propane_temp_C, 55.0)

    def test_blend_train_results_mass_and_flags(self):
        components = [
            PseudoComponent(index=0, MW=450.0, z=0.4, density=0.88, delta=10.0, sara_class='saturates'),
            PseudoComponent(index=1, MW=600.0, z=0.4, density=0.93, delta=11.0, sara_class='aromatics'),
            PseudoComponent(index=2, MW=900.0, z=0.2, density=1.02, delta=12.0, sara_class='resins'),
        ]
        case_a = type('Case', (), {'feed_mass_basis_kg_hr': 1000.0, 'feed_flow_m3hr': 1.0})()
        case_b = type('Case', (), {'feed_mass_basis_kg_hr': 500.0, 'feed_flow_m3hr': 0.5})()
        result_a = {
            'components': components,
            'mass_DAO': np.array([120.0, 180.0, 60.0]),
            'mass_asphalt': np.array([80.0, 300.0, 260.0]),
            'converged': True,
        }
        result_b = {
            'components': components,
            'mass_DAO': np.array([60.0, 90.0, 30.0]),
            'mass_asphalt': np.array([40.0, 150.0, 130.0]),
            'converged': False,
        }
        pred = blend_train_results(result_a, result_b, case_a, case_b)
        self.assertAlmostEqual(pred.DAO_mass_flow_pred_kg_hr, 540.0)
        self.assertAlmostEqual(pred.DAO_yield_wt_pct_pred, 36.0)
        self.assertTrue(pred.train_a_converged)
        self.assertFalse(pred.train_b_converged)
        self.assertGreater(pred.DAO_volume_flow_pred_m3hr, 0.0)
        self.assertGreater(pred.DAO_viscosity_pred, 0.0)
        self.assertTrue(math.isfinite(pred.DAO_CCR_pred))

    def test_run_parallel_extractors_and_blend_smoke(self):
        row = pd.Series({
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
            'propane_temp_C': 55.0,
        })
        params = {
            'K_multiplier': 1.0,
            'C_entrain': 0.015,
            'k_precip': 0.5,
            'E_murphree': 0.70,
            'delta_crit': 2.5,
            'alpha_density': 3.0,
        }
        pred = run_parallel_extractors_and_blend(row, params, n_stages=3, n_comp=12)
        self.assertTrue(math.isfinite(pred.DAO_yield_vol_pct_pred))
        self.assertTrue(math.isfinite(pred.DAO_viscosity_pred))
        self.assertGreater(pred.DAO_mass_flow_pred_kg_hr, 0.0)
        self.assertIn('aromatics', pred.SARA_DAO_pred)
        self.assertIsInstance(pred.train_a_converged, bool)
        self.assertIsInstance(pred.train_b_converged, bool)


if __name__ == '__main__':
    unittest.main()
