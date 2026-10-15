"""Copy the exact model.xyz used to start the NVT timestep comparison."""
from pathlib import Path
import json
import re
import shutil
import numpy as np
from equilibrated_state import digest
from generate_cases import SIDE_A
from scale_common import (read_initial_xyz, body_ids, mic, total_ke,
                          KB_GPUMD_EV_PER_K as KB)


def from_nvt_initial_model(source, output, initial_temperature, target_temperature, nep):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('NVT source and NPT output directories must be separate')
    config = json.loads((source / 'manifest.json').read_text())
    if config.get('ensemble') != 'rigid_nvt' or config.get('case') != 'mixed_2700_300':
        raise ValueError('Expected a mixed rigid NVT source manifest')
    if config['target_temperature_kelvin'] != target_temperature:
        raise ValueError('NVT source target must match the NPT target')
    if digest(Path(nep)) != config['nep_sha256'] or digest(source / 'nep.txt') != config['nep_sha256']:
        raise ValueError('NVT source and NPT must use the same potential')
    model = source / 'model.xyz'
    lattice, species, pos, mass, vel, groups = read_initial_xyz(model)
    with model.open() as stream:
        stream.readline()
        header = stream.readline()
    pbc = re.search(r'\bpbc="([^"]+)"', header)
    if pbc is None or pbc.group(1).upper().split() != ['T', 'T', 'T']:
        raise ValueError('Expected full periodicity')
    for array in (lattice, pos, mass, vel):
        if not np.all(np.isfinite(array)):
            raise ValueError('Nonfinite source model')
    lengths = np.diag(lattice)
    if np.any(lengths <= 0) or not np.allclose(lattice, np.diag(lengths), atol=1e-9, rtol=0):
        raise ValueError('Expected a positive orthorhombic box')
    ids, nfree = body_ids(groups)
    if species != ['C'] * 3600 or len(ids) != 300 or nfree != 2700:
        raise ValueError('Wrong fixture membership')
    free = np.ones(3600, dtype=bool)
    free[ids.ravel()] = False
    if not np.all(mass[ids] == 12) or not np.all(mass[free] == 18):
        raise ValueError('Unexpected source masses')
    for a, b in ((0, 1), (0, 2), (1, 2)):
        dr = mic(pos[ids[:, a]] - pos[ids[:, b]], lengths)
        dv = vel[ids[:, a]] - vel[ids[:, b]]
        if (np.max(abs(np.linalg.norm(dr, axis=1) - SIDE_A)) >= 2e-6
                or np.max(abs(np.einsum('ij,ij->i', dr, dv))) >= 1e-7):
            raise ValueError('Invalid rigid geometry or tangent velocities')
    measured_temperature = 2 * total_ke(mass, vel) / (9900 * KB)
    if not np.isclose(measured_temperature, initial_temperature, rtol=1e-10, atol=1e-10):
        raise ValueError('Requested initial temperature differs from the NVT input; '
                         'exact reuse does not rescale velocities')
    if not np.isclose(measured_temperature, config['initial_temperature_kelvin'], rtol=1e-10):
        raise ValueError('Source model and manifest initial temperatures differ')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, output / 'model.xyz')
    result = {key: config[key] for key in ('case', 'atoms', 'bodies', 'free_beads', 'groups',
              'thermal_dof', 'seed', 'bead_symbol', 'rigid_bead_mass_amu', 'free_water_bead_mass_amu')}
    result.update(initial_state='same_initial_model_as_nvt',
        initial_temperature_kelvin=measured_temperature, box_lengths_angstrom=lengths.tolist(),
        initial_state_source=dict(directory=str(source), frame='initial_model', time_fs=0.,
            temperature_kelvin=measured_temperature, gpumd_sha256=config.get('gpumd_sha256'),
            byte_identical_model=True, velocity_rescaled=False, continuation=False,
            hashes={name: digest(source / name) for name in ('model.xyz', 'manifest.json', 'nep.txt')}))
    print(f'Initial state: exact NVT input model from {source}; '
          f'T={measured_temperature:g} K, box={lengths.tolist()} A', flush=True)
    return result
