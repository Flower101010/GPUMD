"""NVT acceptance: rigid constraints, thermo consistency and resolved temperature control.

Physical-system energy is exchanged with the bath. E=K+U is diagnostic only;
no NVE energy-drift or nonzero-momentum conservation test is applied here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from nvt_statistics import assess_temperature
from scale_common import (
    KB_GPUMD_EV_PER_K as KB, AMU_ANG2_PER_FS2_TO_EV as CONVERSION,
    body_ids, iter_xyz, mic, read_initial_xyz, total_ke,
)


def kinetic_partitions(mass, velocity, ids, free):
    body_mass = mass[ids]
    center_velocity = np.sum(body_mass[:, :, None] * velocity[ids], axis=1) / body_mass.sum(axis=1)[:, None]
    free_ke = total_ke(mass[free], velocity[free])
    translation_ke = total_ke(body_mass.sum(axis=1), center_velocity)
    relative = velocity[ids] - center_velocity[:, None, :]
    rotation_ke = 0.5 * CONVERSION * np.sum(body_mass[:, :, None] * relative**2)
    return np.array([free_ke, translation_ke, rotation_ke])


def verify(folder):
    folder = Path(folder)
    config = json.loads((folder / 'manifest.json').read_text())
    if config['ensemble'] != 'rigid_nvt' or config['case'] != 'mixed_2700_300':
        raise ValueError('Expected a mixed rigid NVT manifest')
    steps, dt = config['steps'], config['dt_fs']
    ti, xi = config['thermo_interval'], config['xyz_interval']
    target = config['target_temperature_kelvin']
    cutoff_fs = config['equilibration_steps'] * dt
    total_fs = steps * dt
    lattice, species, r0, mass, v0, groups = read_initial_xyz(folder / 'model.xyz')
    lengths = np.diag(lattice)
    if not np.allclose(lattice, np.diag(lengths)) or not np.all(lengths > 0):
        raise ValueError('Expected positive orthorhombic box lengths')
    ids, nfree = body_ids(groups)
    nb, natoms = len(ids), len(mass)
    assert (nb, nfree, natoms) == (300, 2700, 3600), 'Wrong fixture size'
    dof = 6 * nb + 3 * nfree
    assert config['thermal_dof'] == dof and config['atoms'] == natoms
    assert all(s == 'C' for s in species)
    assert np.unique(groups).size == nb + nfree and groups.min() >= 0
    free = np.ones(natoms, dtype=bool)
    free[ids.ravel()] = False
    initial_temperature = 2 * total_ke(mass, v0) / (dof * KB)
    assert abs(initial_temperature / config['initial_temperature_kelvin'] - 1) < 1e-10
    # A common NHC rescales P, so arbitrary nonzero P is not conserved.
    # This fixture starts at P=0, which remains zero under that scaling.
    momentum_scale = float(np.sum(mass) * np.sqrt(np.sum(mass[:, None] * v0**2) / np.sum(mass)))
    assert np.linalg.norm(np.sum(mass[:, None] * v0, axis=0)) / momentum_scale < 1e-12
    ref_dists = [np.linalg.norm(mic(r0[ids[:, a]] - r0[ids[:, b]], lengths), axis=1)
                 for a, b in ((0, 1), (0, 2), (1, 2))]
    header = (folder / 'thermo.out').read_text()
    assert '# ensemble rigid_nvt' in header and f'# thermal_dof {dof}\n' in header
    assert 'Rigid NVT initialized: 300 bodies, 2700 free particles, 9900 thermal DOF.' in (folder / 'gpumd.log').read_text()
    data = np.atleast_2d(np.loadtxt(folder / 'thermo.out'))
    assert data.shape == (steps // ti, 18), f'Wrong thermo shape: {data.shape}'
    assert np.all(np.isfinite(data)), 'Nonfinite thermo data'
    times_fs = np.arange(1, len(data) + 1) * dt * ti
    assert np.all(data[:, 0] > 0), 'Nonpositive instantaneous temperature'
    assert np.allclose(data[:, 1], 0.5 * dof * KB * data[:, 0], rtol=1e-8)
    metrics = dict(distance=0.0, velocity_constraint=0.0, zero_com_residual=0.0,
                   ke_relative=0.0, temperature_relative=0.0, pe_per_bead=0.0, force=0.0,
                   partition_ke_relative=0.0)
    partition_trace, snapshot_times = [], []
    for frame in iter_xyz(folder / 'trajectory.xyz'):
        assert len(frame.pos) == natoms and np.array_equal(frame.group, groups)
        assert frame.species == species
        assert np.allclose(frame.mass, mass, atol=1e-12, rtol=1e-12)
        assert np.allclose(frame.lattice, lattice, atol=1e-8, rtol=0), 'Box changed in fixed-box NVT'
        assert all(frame.pbc)
        for array in (frame.pos, frame.vel, frame.forces, frame.energy_atom):
            assert np.all(np.isfinite(array)), f'Nonfinite trajectory at {frame.time_fs} fs'
        assert np.all(frame.pos >= -1e-6) and np.all(frame.pos < lengths + 1e-6)
        rr, vv = frame.pos[ids], frame.vel[ids]
        for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            dr = mic(rr[:, a] - rr[:, b], lengths)
            dv = vv[:, a] - vv[:, b]
            metrics['distance'] = max(metrics['distance'], float(np.max(np.abs(np.linalg.norm(dr, axis=1) - ref_dists[k]))))
            metrics['velocity_constraint'] = max(metrics['velocity_constraint'], float(np.max(np.abs(np.einsum('ij,ij->i', dr, dv)))))
        p = np.sum(mass[:, None] * frame.vel, axis=0)
        metrics['zero_com_residual'] = max(metrics['zero_com_residual'], float(np.linalg.norm(p) / momentum_scale))
        metrics['force'] = max(metrics['force'], float(np.max(np.linalg.norm(frame.forces, axis=1))))
        k = round(frame.time_fs / (dt * ti)) - 1
        assert 0 <= k < len(data) and abs(frame.time_fs - times_fs[k]) <= 1e-4
        kinetic = total_ke(mass, frame.vel)
        temperature = 2 * kinetic / (dof * KB)
        metrics['ke_relative'] = max(metrics['ke_relative'], abs(kinetic - data[k, 1]) / max(1, abs(kinetic)))
        metrics['temperature_relative'] = max(metrics['temperature_relative'], abs(temperature - data[k, 0]) / max(1, abs(temperature)))
        metrics['pe_per_bead'] = max(metrics['pe_per_bead'], abs(float(frame.energy_atom.sum()) - data[k, 2]) / natoms)
        partition_ke = kinetic_partitions(mass, frame.vel, ids, free)
        metrics['partition_ke_relative'] = max(metrics['partition_ke_relative'], abs(float(partition_ke.sum()) - kinetic) / max(1, kinetic))
        partition_t = 2 * partition_ke / (KB * np.array([3 * nfree, 3 * nb, 3 * nb]))
        partition_trace.append([frame.time_fs, *partition_ke, *partition_t])
        snapshot_times.append(frame.time_fs)
    assert len(snapshot_times) == steps // xi
    assert np.allclose(snapshot_times, np.arange(1, steps // xi + 1) * dt * xi, atol=1e-4, rtol=0)
    partition_trace = np.array(partition_trace)
    production_mask = times_fs > cutoff_fs
    production_partitions = partition_trace[partition_trace[:, 0] > cutoff_fs, 4:7]
    stats = assess_temperature(
        data[production_mask, 0], production_partitions, target=target, dof=dof,
        equilibration_fs=cutoff_fs, production_fs=total_fs - cutoff_fs, tau_fs=config['tau_fs'])
    thresholds = {'distance': 2e-6, 'velocity_constraint': 1e-7, 'zero_com_residual': 2e-4,
                  'ke_relative': 1e-4, 'temperature_relative': 1e-4, 'pe_per_bead': 5e-5,
                  'partition_ke_relative': 1e-10}
    checks = {name: bool(metrics[name] < limit) for name, limit in thresholds.items()}
    checks['Nonzero NEP forces'] = metrics['force'] > 1e-5
    if config['strict_statistics']:
        checks.update(stats['checks'])
    energy = data[:, 1] + data[:, 2]
    energy_diagnostics = {'endpoint_change_eV_per_bead': float((energy[-1] - energy[0]) / natoms),
                          'range_eV_per_bead': float(np.ptp(energy) / natoms),
                          'acceptance_criterion': False}
    np.savetxt(folder / 'nvt_trace.csv', np.column_stack((times_fs / 1000, data[:, 0],
               np.full(len(data), target), data[:, 1:3], energy)), delimiter=',', comments='',
               header='time_ps,temperature_K,target_K,kinetic_energy_eV,potential_energy_eV,total_energy_eV', fmt='%.14e')
    np.savetxt(folder / 'kinetic_partitions.csv', partition_trace, delimiter=',', comments='',
               header='time_fs,K_free_eV,K_body_translation_eV,K_body_rotation_eV,T_free_K,T_body_translation_K,T_body_rotation_K', fmt='%.14e')
    report = {
        'case': config['case'], 'ensemble': 'rigid_nvt', 'n_atoms': natoms, 'n_bodies': nb,
        'n_free': nfree, 'thermal_dof': dof, 'simulated_ps': total_fs / 1000,
        'acceptance_scope': 'long_nvt' if config['strict_statistics'] else 'smoke_only',
        'strict_statistics': config['strict_statistics'], 'metrics': metrics, 'thresholds': thresholds,
        'equilibrium_acceptance': ('passed' if all(stats['checks'].values()) else 'failed')
            if config['strict_statistics'] else 'not_evaluated_smoke',
        'temperature_statistics': stats, 'physical_energy_diagnostic_only': energy_diagnostics,
        'checks': checks, 'passed': all(checks.values()),
    }
    (folder / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    plot(folder, times_fs, data, partition_trace, target, cutoff_fs)
    print(f'\nNVT: {natoms} beads, {nb} bodies, {nfree} free particles, {dof} DOFs')
    print(f'Production mean T={stats["mean_kelvin"]:.4f} K; target={target:g} K')
    print('Partition mean temperatures: ' + str(stats['partition_mean_kelvin']))
    print(f'Observed temperature variance / canonical reference: {stats["variance_ratio"]:.4g}')
    for name, passed in checks.items():
        print(f'  {name}: {"PASS" if passed else "FAIL"}')
    if not config['strict_statistics']:
        print('Temperature statistics are informational in smoke; no equilibrium acceptance is claimed.')
    return report['passed']


def plot(folder, times_fs, data, partition_trace, target, cutoff_fs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plot skipped: Matplotlib is unavailable. Use analyze.py with a Python that has Matplotlib.')
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(times_fs / 1000, data[:, 0], label='Instantaneous total T')
    axes[0].axhline(target, color='black', ls='--', label='Target')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].legend()
    for i, label in enumerate(('Free particles', 'Body translation', 'Body rotation')):
        axes[1].plot(partition_trace[:, 0] / 1000, partition_trace[:, 4 + i], label=label)
    axes[1].axhline(target, color='black', ls='--')
    axes[1].set_ylabel('Partition T (K)')
    axes[1].legend()
    energy = data[:, 1] + data[:, 2]
    axes[2].plot(times_fs / 1000, (energy - energy[0]) / 3600)
    axes[2].set_ylabel('Physical energy change\n(eV / bead; diagnostic only)')
    axes[2].set_xlabel('Time (ps)')
    for axis in axes:
        axis.axvline(cutoff_fs / 1000, color='grey', ls=':', label='Production start')
        axis.grid(alpha=0.2)
    fig.suptitle('Mixed rigid NVT diagnostics')
    fig.tight_layout()
    fig.savefig(folder / 'nvt_diagnostics.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Result directory containing manifest.json')
    args = parser.parse_args()
    if not verify(args.folder):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
