"""Reuse a low-generalized-force snapshot from an existing validated NVE run.

This reads existing output only. Resetting velocities is an explicit preparation
operation, not NVE continuation or evidence of low-temperature equilibration.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from generate_cases import SIDE_A
from scale_common import body_ids, iter_xyz, mic, total_ke, KB_GPUMD_EV_PER_K as KB
from static_compare import write_static_model


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def force_score(frame, ids):
    """Sum F_free^2/m + F_COM^2/M + torque^T I^-1 torque.

    Internal forces cancelled by the rigid constraints do not drive the ranking.
    The score is in eV^2/(amu A^2); it does not establish timestep stability.
    """
    free = np.ones(len(frame.mass), dtype=bool)
    free[ids.ravel()] = False
    mass = frame.mass[ids]
    r = mic(frame.pos[ids] - frame.pos[ids[:, :1]], np.diag(frame.lattice))
    r -= (np.sum(mass[:, :, None] * r, axis=1) / mass.sum(axis=1)[:, None])[:, None, :]
    forces = frame.forces[ids]
    torque = np.cross(r, forces).sum(axis=1)
    inertia = np.sum(mass[:, :, None, None] * (
        np.sum(r*r, axis=2)[:, :, None, None] * np.eye(3)
        - r[:, :, :, None] * r[:, :, None, :]), axis=1)
    return float(np.sum(frame.forces[free]**2 / frame.mass[free, None])
                 + np.sum(forces.sum(axis=1)**2 / mass.sum(axis=1)[:, None])
                 + np.sum(torque * np.linalg.solve(inertia, torque[:, :, None])[:, :, 0]))


def expand_centers(frame, factor):
    """Dilate the periodic box and molecular centers, preserving rigid geometry."""
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('Box scale must be finite and at least 1')
    lengths = np.diag(frame.lattice).copy()
    ids, _ = body_ids(frame.group)
    mass = frame.mass[ids]
    anchor = frame.pos[ids[:, 0]].copy()
    relative = mic(frame.pos[ids] - anchor[:, None, :], lengths)
    offset = np.sum(mass[:, :, None] * relative, axis=1) / mass.sum(axis=1)[:, None]
    centers = (anchor + offset) % lengths
    arms = relative - offset[:, None, :]
    frame.pos *= factor
    frame.pos[ids] = factor * centers[:, None, :] + arms
    frame.lattice *= factor
    frame.pos %= lengths * factor


def from_existing_nve(source, output, initial_temperature, nep, *, box_scale=1.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('NVT output and NVE source directories must be separate')
    config = json.loads((source / 'manifest.json').read_text())
    validation = json.loads((source / 'validation.json').read_text())
    if (config.get('case') != 'mixed_2700_300' or not validation.get('passed')
            or config.get('profile') != 'long'):
        raise ValueError('Source must be a validated long mixed NVE run')
    with (source / 'thermo.out').open() as stream:
        header = ''.join(stream.readline() for _ in range(8))
    if '# ensemble rigid_nve' not in header or '# thermal_dof 9900' not in header:
        raise ValueError('Source output is not the expected rigid NVE ensemble')
    if digest(Path(nep)) != config['nep_sha256'] or digest(source / 'nep.txt') != config['nep_sha256']:
        raise ValueError('NVE source and requested NVT must use the same potential')
    if not np.isfinite(initial_temperature) or initial_temperature <= 0:
        raise ValueError('Initial temperature must be positive and finite')
    if not np.isfinite(box_scale) or box_scale < 1:
        raise ValueError('Box scale must be finite and at least 1')
    expected_end = config['steps'] * config['dt_fs']
    selected = None
    best = float('inf')
    candidates = []
    count = 0
    last_time = None
    for frame in iter_xyz(source / 'trajectory.xyz'):
        count += 1
        last_time = frame.time_fs
        if frame.time_fs < expected_end / 2 - 1e-4:
            continue
        if len(frame.mass) != 3600 or frame.species != ['C'] * 3600 or not all(frame.pbc):
            raise ValueError('Wrong NVE source fixture')
        for array in (frame.pos, frame.vel, frame.mass, frame.forces, frame.lattice):
            if not np.all(np.isfinite(array)):
                raise ValueError('Nonfinite NVE snapshot')
        lengths = np.diag(frame.lattice)
        if np.any(lengths <= 0) or not np.allclose(frame.lattice, np.diag(lengths), atol=1e-9, rtol=0):
            raise ValueError('Expected an orthorhombic NVE snapshot')
        ids, nfree = body_ids(frame.group)
        free = np.ones(3600, dtype=bool)
        free[ids.ravel()] = False
        if (len(ids) != 300 or nfree != 2700 or not np.all(frame.mass[ids] == 12)
                or not np.all(frame.mass[free] == 18)):
            raise ValueError('Wrong source membership/masses')
        score = force_score(frame, ids)
        candidates.append(dict(time_fs=frame.time_fs, generalized_force_score=score))
        if score < best:
            best, selected = score, frame
    if (selected is None or count != config['steps'] // config['xyz_interval']
            or abs(last_time - expected_end) > 1e-4):
        raise ValueError('NVE trajectory incomplete or inconsistent with manifest')
    ids, _ = body_ids(selected.group)
    lengths = np.diag(selected.lattice)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        dr = mic(selected.pos[ids[:, a]] - selected.pos[ids[:, b]], lengths)
        dv = selected.vel[ids[:, a]] - selected.vel[ids[:, b]]
        if (np.max(abs(np.linalg.norm(dr, axis=1) - SIDE_A)) >= 2e-6
                or np.max(abs(np.einsum('ij,ij->i', dr, dv))) >= 1e-7):
            raise ValueError('Selected NVE frame violates rigid constraints')
    source_temperature = 2 * total_ke(selected.mass, selected.vel) / (9900 * KB)
    selected.vel -= np.average(selected.vel, axis=0, weights=selected.mass)
    measured = 2 * total_ke(selected.mass, selected.vel) / (9900 * KB)
    if not np.isfinite(measured) or measured <= 0:
        raise ValueError('Cannot rescale source velocities')
    scale = float(np.sqrt(initial_temperature / measured))
    selected.vel *= scale
    source_lengths = lengths.copy()
    expand_centers(selected, box_scale)
    lengths = np.diag(selected.lattice).copy()
    output.mkdir(parents=True, exist_ok=True)
    write_static_model(selected, output / 'model.xyz')
    result = {key: config[key] for key in ('case', 'atoms', 'bodies', 'free_beads',
        'groups', 'thermal_dof', 'seed', 'bead_symbol', 'rigid_bead_mass_amu', 'free_water_bead_mass_amu')}
    result.update(initial_state='existing_nve_frame_then_velocity_rescaled',
        initial_temperature_kelvin=initial_temperature, box_lengths_angstrom=lengths.tolist(),
        box_scale=box_scale, source_box_lengths_angstrom=source_lengths.tolist(),
        box_scaling='Molecular centers and free particles; rigid internal geometry unchanged',
        initial_state_source=dict(directory=str(source), time_fs=selected.time_fs,
            temperature_kelvin=source_temperature, velocity_scale=scale,
            selection='Lowest generalized force score among saved frames in the latter half',
            candidates=candidates, remove_com_motion=True, continuation=False,
            hashes={name: digest(source / name) for name in
                    ('manifest.json', 'validation.json', 'trajectory.xyz', 'nep.txt')}))
    print(f'Initial state: existing NVE frame at {selected.time_fs / 1000:g} ps; '
          f'source T={source_temperature:.3f} K; explicit velocity reset to {initial_temperature:g} K', flush=True)
    print(f'Box lengths scaled by {box_scale:g}: {lengths.tolist()} A; '
          f'volume factor={box_scale**3:g}; rigid internal distances unchanged.', flush=True)
    return result
