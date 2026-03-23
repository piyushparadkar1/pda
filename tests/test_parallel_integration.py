import math
from pathlib import Path
import unittest

from data_ingestion import build_normalized_parallel_dataset
from parallel_calibration import (
    points_from_normalized_dataframe,
    run_parallel_calibration,
    run_parallel_calibration_from_workbooks,
    run_parallel_calibration_from_profile,
)
from parallel_extractor_model import run_parallel_extractors_and_blend


class ParallelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lims = Path('lims.xlsx')
        extractor = Path('extractor_parameters.xlsx')
        if not lims.exists() or not extractor.exists():
            raise unittest.SkipTest('workbook fixtures are not available')
        df, _ = build_normalized_parallel_dataset(str(lims), str(extractor))
        usable = df[df['usable_core_row']].reset_index(drop=True)
        if usable.empty:
            raise unittest.SkipTest('no usable core rows available in workbook fixtures')
        cls.row = usable.iloc[0]
        cls.points = points_from_normalized_dataframe(usable.head(1))
        cls.params = {
            'K_multiplier': 1.0,
            'C_entrain': 0.015,
            'k_precip': 0.5,
            'E_murphree': 0.70,
            'delta_crit': 2.5,
            'alpha_density': 3.0,
        }

    def test_real_workbook_parallel_simulation_smoke(self):
        pred = run_parallel_extractors_and_blend(self.row, self.params, n_stages=4, n_comp=20)
        self.assertTrue(math.isfinite(pred.DAO_yield_vol_pct_pred))
        self.assertTrue(math.isfinite(pred.DAO_viscosity_pred))
        self.assertGreater(pred.DAO_mass_flow_pred_kg_hr, 0.0)
        self.assertTrue(pred.train_a_converged)
        self.assertTrue(pred.train_b_converged)

    def test_real_workbook_parallel_calibration_smoke(self):
        result = run_parallel_calibration(self.points, max_nfev=1, verbose=False)
        self.assertEqual(result.n_operating_points, 1)
        self.assertIn('convergence', result.metrics)
        self.assertEqual(result.n_converged_points + result.n_nonconverged_points, 1)
        self.assertEqual(result.n_converged_points, 1)
        self.assertEqual(len(result.point_results), 1)
        self.assertIn('converged', result.point_results[0])

    def test_real_workbook_parallel_calibration_from_workbooks_smoke(self):
        result, df, summary = run_parallel_calibration_from_workbooks(
            'lims.xlsx',
            'extractor_parameters.xlsx',
            max_nfev=1,
            verbose=False,
        )
        self.assertGreater(len(df), 0)
        self.assertGreater(summary.n_rows_total, 0)
        self.assertGreater(result.n_operating_points, 0)

    def test_real_workbook_parallel_calibration_from_profile_smoke(self):
        result, df, summary, profile = run_parallel_calibration_from_profile(
            'lims.xlsx',
            'extractor_parameters.xlsx',
            'parallel_usable_core_v1',
            max_nfev=1,
            verbose=False,
        )
        self.assertEqual(profile['profile_name'], 'parallel_usable_core_v1')
        self.assertGreater(len(df), 0)
        self.assertGreater(summary.n_rows_total, 0)
        self.assertGreater(result.n_operating_points, 0)


if __name__ == '__main__':
    unittest.main()
