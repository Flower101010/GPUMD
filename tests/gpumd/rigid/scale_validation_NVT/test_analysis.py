"""Host checks for NVT acceptance, including deliberately incorrect thermostat behavior."""
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from check import kinetic_partitions
from analyze import select_window, summarize_windows
from generate_cases import make
from nvt_statistics import assess_temperature
from scale_common import (AMU_ANG2_PER_FS2_TO_EV as CONVERSION, KB_GPUMD_EV_PER_K as KB,
                          body_ids, read_initial_xyz, total_ke)


class NVTAnalysisTests(unittest.TestCase):
    def test_offline_window_excludes_warmup_and_includes_endpoint(self):
        np.testing.assert_array_equal(select_window([1., 2., 3.], 1., 3.), [False, True, True])
        with self.assertRaises(ValueError):
            select_window([1., 2., 3.], 1., 4.)
        with self.assertRaises(ValueError):
            select_window([1., 2., 3.], 3., 3.)

    def test_window_comparison_reports_cooling_without_selecting_a_pass(self):
        times = np.arange(1, 101) / 10
        temperatures = 500 - 10 * times
        rows = summarize_windows(times, temperatures, [0., 8.], 10.)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0]['mean_K'], rows[1]['mean_K'])
        self.assertAlmostEqual(rows[1]['slope_K_per_ps'], -10)
        self.assertNotIn('passed', rows[1])

    def test_invalid_absolute_equilibration_rejected_before_md(self):
        script = str(Path(__file__).with_name('run_tests.py'))
        result = subprocess.run([sys.executable, script, '--long', '--steps', '200',
                                 '--equilibration-ps', '1'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Equilibration must end before', result.stderr)

    def assess(self, total=None, partitions=None, **kwargs):
        # A balanced, stationary deterministic signal with the canonical T variance.
        phase = np.linspace(0, 80 * np.pi, 1000, endpoint=False)
        if total is None:
            total = 300 + np.sqrt(4 * 300**2 / 9900) * np.sin(phase)
        if partitions is None:
            partitions = np.full((80, 3), 300.0)
        params = dict(target=300.0, dof=9900, equilibration_fs=2000,
                      production_fs=8000, tau_fs=100)
        params.update(kwargs)
        return assess_temperature(total, partitions, **params)

    def test_stationary_target_passes(self):
        result = self.assess()
        self.assertTrue(all(result['checks'].values()))
        self.assertAlmostEqual(result['variance_ratio'], 1000 / 999)

    def test_clamped_temperature_is_not_nvt_acceptance(self):
        result = self.assess(total=np.full(1000, 300.0))
        self.assertTrue(result['checks']['Mean temperature at target'])
        self.assertFalse(result['checks']['Temperature fluctuation sanity'])

    def test_wrong_temperature_fails(self):
        result = self.assess(total=np.full(1000, 450.0))
        self.assertFalse(result['checks']['Mean temperature at target'])

    def test_correct_total_can_hide_bad_equipartition(self):
        # Body translation is cold and body rotation hot: equal DOF counts
        # cancel in total T, so total temperature alone must not pass this case.
        result = self.assess(partitions=np.tile([300., 150., 450.], (80, 1)))
        self.assertTrue(result['checks']['Mean temperature at target'])
        self.assertFalse(result['checks']['Body rotation temperature at target'])
        self.assertFalse(result['checks']['Body translation temperature at target'])

    def test_drift_and_short_sampling_fail(self):
        result = self.assess(total=np.linspace(250, 350, 1000))
        self.assertFalse(result['checks']['No unresolved temperature drift'])
        short = self.assess(production_fs=80, equilibration_fs=20)
        self.assertFalse(short['checks']['Sufficient equilibration duration'])
        self.assertFalse(short['checks']['Sufficient production duration'])
        self.assertFalse(short['checks']['Block duration at least tau'])

    def test_kinetic_partition_against_analytic_rotation(self):
        mass = np.array([2., 2., 2., 3.])
        relative_positions = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 2., 0.]])
        center_velocity = np.array([.01, -.02, .03])
        free_velocity = np.array([.04, .01, -.01])
        omega = .02
        velocity = np.vstack((center_velocity + np.cross([0, 0, omega], relative_positions), free_velocity))
        actual = kinetic_partitions(mass, velocity, np.array([[0, 1, 2]]), np.array([False, False, False, True]))
        expected = .5 * CONVERSION * np.array([
            3 * np.dot(free_velocity, free_velocity),
            6 * np.dot(center_velocity, center_velocity), 16 * omega**2])
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
        self.assertAlmostEqual(actual.sum(), total_ke(mass, velocity))

    def test_fixture_membership_temperature_and_zero_momentum(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make('mixed_2700_300', Path(directory), 150)
            _, _, _, mass, vel, groups = read_initial_xyz(Path(directory) / 'model.xyz')
            ids, nfree = body_ids(groups)
            self.assertEqual((len(ids), nfree, len(mass)), (300, 2700, 3600))
            self.assertEqual(config['thermal_dof'], 9900)
            self.assertAlmostEqual(2 * total_ke(mass, vel) / (9900 * KB), 150, places=10)
            np.testing.assert_allclose((mass[:, None] * vel).sum(axis=0), 0, atol=1e-12)
            with self.assertRaises(ValueError):
                make('unsupported_case', Path(directory))


if __name__ == '__main__':
    unittest.main()
