"""Build a fresh rigid-ensemble initial state from an existing NVT/NPT snapshot.

Coordinates/velocities are reused; thermostat and piston history is not continued.
No force evaluation or dynamics is performed here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scale_common import (body_ids, iter_xyz, mic, total_ke,
                          KB_GPUMD_EV_PER_K as KB)
from static_compare import write_static_model
from generate_cases import SIDE_A


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def from_equilibrated_nvt(source, output, initial_temperature, target_temperature, nep):
    return _from_snapshot(source, output, initial_temperature, target_temperature, nep, 'rigid_nvt')


def from_completed_npt(source, output, initial_temperature, target_temperature, nep):
    return _from_snapshot(source, output, initial_temperature, target_temperature, nep, 'rigid_npt')


def _from_snapshot(source, output, initial_temperature, target_temperature, nep, expected_ensemble):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Destination and source directories must be separate')
    config = json.loads((source / 'manifest.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    validation = json.loads((source / 'validation.json').read_text())
    if (config.get('ensemble') != expected_ensemble or config.get('case') != 'mixed_2700_300'
            or not config.get('strict_statistics')):
        raise ValueError('Expected a completed long mixed rigid-ensemble source run')
    if expected_ensemble == 'rigid_nvt':
        if (not summary.get('passed') or summary.get('equilibrium_acceptance') != 'passed'
                or not validation.get('passed') or validation.get('equilibrium_acceptance') != 'passed'):
            raise ValueError('NVT source must have passed long-run acceptance')
    else:
        required = ('distance', 'velocity_constraint', 'zero_com_residual', 'ke_relative',
                    'temperature_relative', 'pe_per_bead', 'partition_ke_relative')
        if (validation.get('ensemble') != 'rigid_npt'
                or not all(validation.get('checks', {}).get(key) is True for key in required)):
            raise ValueError('NPT diagnostic source must pass geometry and thermo-consistency checks')
    if config['target_temperature_kelvin'] != target_temperature:
        raise ValueError('Source target must match the requested target temperature')
    if digest(Path(nep)) != config['nep_sha256'] or digest(source / 'nep.txt') != config['nep_sha256']:
        raise ValueError('Source and requested ensemble must use the same NEP potential')
    if not np.isfinite(initial_temperature) or initial_temperature <= 0:
        raise ValueError('Initial temperature must be positive and finite')
    last = None
    count = 0
    for last in iter_xyz(source / 'trajectory.xyz'):
        count += 1
    expected_time = config['steps'] * config['dt_fs']
    if (last is None or count != config['steps'] // config['xyz_interval']
            or abs(last.time_fs - expected_time) > 1e-4):
        raise ValueError('Source trajectory is incomplete or does not match its manifest')
    if len(last.mass) != 3600 or not all(last.pbc) or last.species != ['C'] * 3600:
        raise ValueError('Wrong source fixture or boundary conditions')
    for array in (last.lattice, last.pos, last.mass, last.vel):
        if not np.all(np.isfinite(array)):
            raise ValueError('Nonfinite source snapshot')
    lengths = np.diag(last.lattice)
    if np.any(lengths <= 0) or not np.allclose(last.lattice, np.diag(lengths), atol=1e-9, rtol=0):
        raise ValueError('Source must have a positive orthorhombic box')
    ids, nfree = body_ids(last.group)
    if len(ids) != 300 or nfree != 2700 or np.any(last.mass <= 0):
        raise ValueError('Wrong source membership/masses')
    body_mask = np.zeros(3600, dtype=bool)
    body_mask[ids.ravel()] = True
    if not np.all(last.mass[body_mask] == 12) or not np.all(last.mass[~body_mask] == 18):
        raise ValueError('Unexpected source fixture masses')
    for a, b in ((0, 1), (0, 2), (1, 2)):
        dr = mic(last.pos[ids[:, a]] - last.pos[ids[:, b]], lengths)
        dv = last.vel[ids[:, a]] - last.vel[ids[:, b]]
        if (np.max(np.abs(np.linalg.norm(dr, axis=1) - SIDE_A)) >= 2e-6
                or np.max(np.abs(np.einsum('ij,ij->i', dr, dv))) >= 1e-7):
            raise ValueError('Source rigid geometry/velocities do not match the fixture')
    # Translation removal and a common scalar preserve every tangent constraint.
    source_temperature = 2 * total_ke(last.mass, last.vel) / (9900 * KB)
    last.vel -= np.average(last.vel, axis=0, weights=last.mass)
    measured = 2 * total_ke(last.mass, last.vel) / (9900 * KB)
    if measured <= 0:
        raise ValueError('Source has zero kinetic temperature')
    last.vel *= np.sqrt(initial_temperature / measured)
    last.pos %= lengths
    output.mkdir(parents=True, exist_ok=True)
    write_static_model(last, output / 'model.xyz')
    label = 'accepted NVT' if expected_ensemble == 'rigid_nvt' else 'completed NPT (diagnostic source)'
    print(f'Initial state: {label} frame at {last.time_fs / 1000:g} ps; '
          f'source T={source_temperature:.3f} K; initial T={initial_temperature:g} K', flush=True)
    return dict(case='mixed_2700_300', atoms=3600, bodies=300, free_beads=2700,
        groups=3000, thermal_dof=9900, box_lengths_angstrom=lengths.tolist(),
        seed=config.get('seed'), initial_temperature_kelvin=initial_temperature,
        bead_symbol='C', rigid_bead_mass_amu=12., free_water_bead_mass_amu=18.,
        initial_state='equilibrated_nvt' if expected_ensemble == 'rigid_nvt' else 'completed_npt_snapshot',
        initial_state_source=dict(directory=str(source), time_fs=last.time_fs,
            ensemble=expected_ensemble, equilibrium_acceptance=validation.get('equilibrium_acceptance'),
            temperature_kelvin=source_temperature, gpumd_sha256=config.get('gpumd_sha256'),
            hashes={name: digest(source / name) for name in (
                'manifest.json', 'summary.json', 'validation.json', 'trajectory.xyz', 'nep.txt')},
            continuation=False))
