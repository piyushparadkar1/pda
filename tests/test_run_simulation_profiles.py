import unittest

from run_simulation import (
    calibration_profile_kind,
    list_ui_calibration_profiles,
    load_ui_calibration_profile,
)


class RunSimulationProfileTests(unittest.TestCase):
    def test_calibration_profile_kind_detects_parallel_metadata(self):
        self.assertEqual(
            calibration_profile_kind('parallel_usable_core_v1', {'row_flag': 'usable_core_row'}),
            'parallel',
        )
        self.assertEqual(
            calibration_profile_kind('sda_default', {'parameters': {'K_multiplier': 1.0}}),
            'legacy',
        )

    def test_list_ui_calibration_profiles_includes_parallel_metadata(self):
        profiles = list_ui_calibration_profiles()
        by_name = {profile['name']: profile for profile in profiles}
        self.assertIn('parallel_usable_core_v1', by_name)
        self.assertEqual(by_name['parallel_usable_core_v1']['kind'], 'parallel')
        self.assertIn('sda_default', by_name)
        self.assertEqual(by_name['sda_default']['kind'], 'legacy')

    def test_load_ui_calibration_profile_dispatches_parallel_loader(self):
        loaded = load_ui_calibration_profile('parallel_usable_core_v1')
        self.assertEqual(loaded['kind'], 'parallel')
        self.assertEqual(loaded['profile']['profile_name'], 'parallel_usable_core_v1')
        self.assertIn('alpha_density', loaded['params'])


if __name__ == '__main__':
    unittest.main()
