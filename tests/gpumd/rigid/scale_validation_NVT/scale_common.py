"""Shared parsing and constants for synthetic NEP/rigid scalability verification."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# GPUMD source constants: src/utilities/common.cuh
KB_GPUMD_EV_PER_K = 8.617343e-5
TIME_UNIT_FS = 10.18051
AMU_ANG2_PER_FS2_TO_EV = TIME_UNIT_FS**2


@dataclass
class Frame:
    time_fs: float
    lattice: np.ndarray
    pbc: tuple[bool, bool, bool]
    species: list[str]
    pos: np.ndarray
    mass: np.ndarray
    vel: np.ndarray
    forces: np.ndarray
    energy_atom: np.ndarray
    group: np.ndarray


def _schema(header):
    prop_match = re.search(r'\b[Pp]roperties=([^\s]+)', header)
    if not prop_match:
        raise ValueError('Extended XYZ missing Properties')
    fields = prop_match.group(1).split(':')
    if len(fields) % 3:
        raise ValueError('Invalid Properties schema')
    schema, offset = {}, 0
    for i in range(0, len(fields), 3):
        n = int(fields[i + 2])
        schema[fields[i]] = slice(offset, offset + n)
        offset += n
    return schema


def iter_xyz(path: Path):
    with open(path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                return
            if not line.strip():
                continue
            n = int(line)
            header = fp.readline()
            fields = _schema(header)
            tm = re.search(r'\bTime=([^\s]+)', header)
            box = re.search(r'\b[Ll]attice="([^"]+)"', header)
            pm = re.search(r'\bpbc="([^"]+)"', header)
            if tm is None or box is None or pm is None:
                raise ValueError('Trajectory frame lacks Time/Lattice/pbc')
            lattice = np.asarray([float(x) for x in box.group(1).split()]).reshape(3, 3)
            pbc = tuple(x.upper() == 'T' for x in pm.group(1).split())
            if len(pbc) != 3:
                raise ValueError('Invalid pbc field')
            for name in ('species', 'pos', 'mass', 'vel', 'forces', 'energy_atom', 'group'):
                if name not in fields:
                    raise ValueError(f'Trajectory lacks field {name}. Check dump_xyz options.')
            # Assemble arrays per frame; do not load an entire long simulation into RAM.
            xyz, mass, vel, force, e, label, species = [], [], [], [], [], [], []
            for _ in range(n):
                row = fp.readline().split()
                if not row:
                    raise ValueError('Unexpected end of frame')
                species.append(row[fields['species'].start])
                xyz.append([float(v) for v in row[fields['pos']]])
                mass.append(float(row[fields['mass'].start]))
                vel.append([float(v) for v in row[fields['vel']]])
                force.append([float(v) for v in row[fields['forces']]])
                e.append(float(row[fields['energy_atom'].start]))
                label.append(int(row[fields['group'].start]))
            yield Frame(
                time_fs=float(tm.group(1)), lattice=lattice, pbc=pbc,
                species=species, pos=np.asarray(xyz), mass=np.asarray(mass),
                vel=np.asarray(vel), forces=np.asarray(force),
                energy_atom=np.asarray(e), group=np.asarray(label, dtype=np.int64),
            )


def read_initial_xyz(path):
    with open(path, encoding='utf-8') as fp:
        n = int(fp.readline())
        header = fp.readline()
        schema = _schema(header)
        latmatch = re.search(r'\b[Ll]attice="([^"]+)"', header)
        if not latmatch:
            raise ValueError('Missing lattice')
        lat = np.array([float(s) for s in latmatch.group(1).split()]).reshape(3, 3)
        p, m, v, g, spec = [], [], [], [], []
        for _ in range(n):
            row = fp.readline().split()
            spec.append(row[schema['species'].start])
            p.append([float(x) for x in row[schema['pos']]])
            m.append(float(row[schema['mass'].start]))
            v.append([float(x) for x in row[schema['vel']]])
            g.append(int(row[schema['group'].start]))
        return lat, spec, np.array(p), np.array(m), np.array(v), np.array(g)


def mic(dr, box_lengths):
    return dr - box_lengths * np.rint(dr / box_lengths)


def total_ke(mass, v):
    return 0.5 * AMU_ANG2_PER_FS2_TO_EV * float(np.sum(mass[:, None] * v * v))


def body_ids(groups):
    counts = np.bincount(groups)
    bodies = [np.flatnonzero(groups == label) for label in np.flatnonzero(counts >= 2)]
    if any(len(ids) != 3 for ids in bodies):
        raise AssertionError('Some nonsingleton group is not a three-bead body')
    return np.asarray(bodies, dtype=np.int64), int(np.count_nonzero(counts == 1))
