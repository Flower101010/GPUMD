"""Engineering acceptance for a constant-target NVT run, with explicit sampling gates."""
from __future__ import annotations
import numpy as np

THRESHOLDS = {
    'mean_temperature_relative': 0.05,
    'partition_temperature_relative': 0.10,
    'temperature_drift_relative': 0.05,
    'block_mean_standard_error_relative': 0.02,
    'minimum_equilibration_tau': 10.0,
    'minimum_production_tau': 20.0,
    'minimum_temperature_samples': 100,
    'minimum_partition_samples': 20,
    'minimum_blocks': 10,
    'variance_ratio_minimum': 0.2,
    'variance_ratio_maximum': 5.0,
}


def assess_temperature(temperatures, partition_temperatures, *, target, dof,
                       equilibration_fs, production_fs, tau_fs, blocks=20):
    """Inputs are already restricted to the production window.

    Block errors quantify observed uncertainty, not independent-sample confidence
    intervals. A broad variance guard catches clamped/explosive temperature;
    this is not a formal canonical-distribution test on correlated samples.
    """
    values = np.asarray(temperatures, dtype=float)
    partitions = np.asarray(partition_temperatures, dtype=float)
    if values.ndim != 1 or not len(values) or partitions.ndim != 2 or partitions.shape[1] != 3:
        raise ValueError('Temperature analysis requires samples and three kinetic partitions')
    if not len(partitions) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(partitions)):
        raise ValueError('Missing or nonfinite production temperatures')
    if not all(np.isfinite(x) and x > 0 for x in (target, dof, production_fs, tau_fs)):
        raise ValueError('Invalid temperature analysis parameters')
    if not np.isfinite(equilibration_fs) or equilibration_fs < 0 or blocks < 2:
        raise ValueError('Invalid equilibration duration or block count')
    nblocks = min(blocks, len(values))
    block_means = np.array([np.mean(chunk) for chunk in np.array_split(values, nblocks)])
    sem = float(np.std(block_means, ddof=1) / np.sqrt(nblocks)) if nblocks > 1 else None
    quarter = max(1, len(values) // 4)
    drift = float(abs(np.mean(values[-quarter:]) - np.mean(values[:quarter])) / target)
    means = np.mean(partitions, axis=0)
    mean = float(np.mean(values))
    relative = abs(mean / target - 1)
    variance = float(np.var(values, ddof=1)) if len(values) > 1 else None
    expected_variance = 2 * target**2 / dof
    variance_ratio = variance / expected_variance if variance is not None else None
    checks = {
        'Sufficient equilibration duration': equilibration_fs / tau_fs >= THRESHOLDS['minimum_equilibration_tau'],
        'Sufficient production duration': production_fs / tau_fs >= THRESHOLDS['minimum_production_tau'],
        'Sufficient temperature samples': len(values) >= THRESHOLDS['minimum_temperature_samples'],
        'Sufficient kinetic partition samples': len(partitions) >= THRESHOLDS['minimum_partition_samples'],
        'Sufficient blocks': nblocks >= THRESHOLDS['minimum_blocks'],
        'Block duration at least tau': production_fs / nblocks >= tau_fs,
        'Mean temperature at target': relative <= THRESHOLDS['mean_temperature_relative'],
        'No unresolved temperature drift': drift <= THRESHOLDS['temperature_drift_relative'],
        'Block uncertainty resolved': sem is not None and sem / target <= THRESHOLDS['block_mean_standard_error_relative'],
        'Temperature fluctuation sanity': variance_ratio is not None and
            THRESHOLDS['variance_ratio_minimum'] <= variance_ratio <= THRESHOLDS['variance_ratio_maximum'],
    }
    for label, value in zip(('Free particles', 'Body translation', 'Body rotation'), means):
        checks[f'{label} temperature at target'] = abs(value / target - 1) <= THRESHOLDS['partition_temperature_relative']
    return {
        'target_kelvin': target, 'mean_kelvin': mean, 'relative_mean_error': relative,
        'samples': len(values), 'partition_samples': len(partitions),
        'blocks': nblocks, 'block_means_kelvin': block_means.tolist(),
        'block_duration_fs': production_fs / nblocks,
        'block_mean_standard_error_kelvin': sem, 'relative_temperature_drift': drift,
        'partition_mean_kelvin': dict(zip(('free', 'body_translation', 'body_rotation'), means.tolist())),
        'variance_kelvin2': variance,
        'canonical_reference_variance_kelvin2': expected_variance,
        'variance_ratio': variance_ratio,
        'equilibration_tau': equilibration_fs / tau_fs, 'production_tau': production_fs / tau_fs,
        'thresholds': dict(THRESHOLDS), 'checks': {k: bool(v) for k, v in checks.items()},
    }
