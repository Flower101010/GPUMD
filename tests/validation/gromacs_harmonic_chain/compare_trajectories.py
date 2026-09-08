#!/usr/bin/env python3

"""Compare deterministic GPUMD and GROMACS trajectories for a bonded chain."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


EV_TO_KJ_PER_MOL = 96.4853321233
NUMBER_OF_ATOMS = 8
SAMPLE_INTERVAL_FS = 1.0


def read_gpumd_xyz(path: Path) -> list[list[tuple[float, float, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[list[tuple[float, float, float]]] = []
    cursor = 0
    while cursor < len(lines):
        count = int(lines[cursor])
        if count != NUMBER_OF_ATOMS:
            raise ValueError(f"{path}: expected {NUMBER_OF_ATOMS} atoms, got {count}")
        cursor += 2
        frame = []
        for line in lines[cursor : cursor + count]:
            fields = line.split()
            frame.append(tuple(float(value) for value in fields[1:4]))
        frames.append(frame)
        cursor += count
    return frames


def read_gromacs_positions(path: Path) -> list[tuple[float, list[tuple[float, float, float]]]]:
    frames = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", "@")):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 1 + 3 * NUMBER_OF_ATOMS:
            raise ValueError(f"{path}:{line_number}: unexpected column count {len(values)}")
        time_fs = values[0] * 1000.0
        positions = []
        for atom in range(NUMBER_OF_ATOMS):
            offset = 1 + 3 * atom
            positions.append(tuple(10.0 * values[offset + axis] for axis in range(3)))
        frames.append((time_fs, positions))
    return frames


def read_gpumd_energy(path: Path) -> list[tuple[float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        rows.append((values[2], values[1] + values[2]))
    return rows


def read_gromacs_energy(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", "@")):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 4:
            raise ValueError(f"{path}:{line_number}: expected time and three energies")
        rows.append(
            (
                values[0] * 1000.0,
                values[1] / EV_TO_KJ_PER_MOL,
                values[3] / EV_TO_KJ_PER_MOL,
            )
        )
    return rows


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in range(3)))


def bond_lengths(frame: list[tuple[float, float, float]]) -> list[float]:
    return [distance(frame[index], frame[index + 1]) for index in range(NUMBER_OF_ATOMS - 1)]


def pair_distances(frame: list[tuple[float, float, float]]) -> list[float]:
    return [
        distance(frame[first], frame[second])
        for first in range(NUMBER_OF_ATOMS)
        for second in range(first + 1, NUMBER_OF_ATOMS)
    ]


def subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a[axis] - b[axis] for axis in range(3))


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(a[axis] * b[axis] for axis in range(3))


def cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(dot(vector, vector))


def angles(frame: list[tuple[float, float, float]]) -> list[float]:
    values = []
    for center in range(1, NUMBER_OF_ATOMS - 1):
        left = subtract(frame[center - 1], frame[center])
        right = subtract(frame[center + 1], frame[center])
        cosine = dot(left, right) / (norm(left) * norm(right))
        values.append(math.acos(max(-1.0, min(1.0, cosine))))
    return values


def dihedrals(frame: list[tuple[float, float, float]]) -> list[float]:
    values = []
    for first in range(NUMBER_OF_ATOMS - 3):
        b1 = subtract(frame[first + 1], frame[first])
        b2 = subtract(frame[first + 2], frame[first + 1])
        b3 = subtract(frame[first + 3], frame[first + 2])
        n1 = cross(b1, b2)
        n2 = cross(b2, b3)
        values.append(math.atan2(norm(b2) * dot(b1, n2), dot(n1, n2)))
    return values


def periodic_difference(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def radius_of_gyration(frame: list[tuple[float, float, float]]) -> float:
    center = [sum(position[axis] for position in frame) / NUMBER_OF_ATOMS for axis in range(3)]
    return math.sqrt(
        sum(sum((position[axis] - center[axis]) ** 2 for axis in range(3)) for position in frame)
        / NUMBER_OF_ATOMS
    )


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpumd-xyz", type=Path, required=True)
    parser.add_argument("--gromacs-positions", type=Path, required=True)
    parser.add_argument("--gpumd-thermo", type=Path, required=True)
    parser.add_argument("--gromacs-energy", type=Path, required=True)
    parser.add_argument("--case-name", default="harmonic chain")
    parser.add_argument("--compare-angles", action="store_true")
    parser.add_argument("--compare-dihedrals", action="store_true")
    args = parser.parse_args()

    gpumd_frames = read_gpumd_xyz(args.gpumd_xyz)
    gromacs_with_time = read_gromacs_positions(args.gromacs_positions)
    gromacs_frames = [frame for time_fs, frame in gromacs_with_time if time_fs > 0.5 * SAMPLE_INTERVAL_FS]

    if len(gpumd_frames) != len(gromacs_frames):
        raise SystemExit(
            f"FAIL: trajectory frame counts differ: GPUMD={len(gpumd_frames)}, "
            f"GROMACS(nonzero time)={len(gromacs_frames)}"
        )

    bond_differences = []
    pair_differences = []
    angle_differences: list[float] = []
    dihedral_differences: list[float] = []
    rg_differences = []
    coordinate_differences = []
    for gpumd_frame, gromacs_frame in zip(gpumd_frames, gromacs_frames):
        bond_differences.extend(
            left - right for left, right in zip(bond_lengths(gpumd_frame), bond_lengths(gromacs_frame))
        )
        pair_differences.extend(
            left - right for left, right in zip(pair_distances(gpumd_frame), pair_distances(gromacs_frame))
        )
        if args.compare_angles:
            angle_differences.extend(
                left - right for left, right in zip(angles(gpumd_frame), angles(gromacs_frame))
            )
        if args.compare_dihedrals:
            dihedral_differences.extend(
                periodic_difference(left, right)
                for left, right in zip(dihedrals(gpumd_frame), dihedrals(gromacs_frame))
            )
        rg_differences.append(radius_of_gyration(gpumd_frame) - radius_of_gyration(gromacs_frame))
        coordinate_differences.extend(
            gpumd_frame[atom][axis] - gromacs_frame[atom][axis]
            for atom in range(NUMBER_OF_ATOMS)
            for axis in range(3)
        )

    gpumd_energy = read_gpumd_energy(args.gpumd_thermo)
    gromacs_energy_with_time = read_gromacs_energy(args.gromacs_energy)
    gromacs_energy = [row[1:] for row in gromacs_energy_with_time if row[0] > 0.5 * SAMPLE_INTERVAL_FS]
    if len(gpumd_energy) != len(gromacs_energy):
        raise SystemExit(
            f"FAIL: energy row counts differ: GPUMD={len(gpumd_energy)}, "
            f"GROMACS(nonzero time)={len(gromacs_energy)}"
        )

    potential_differences = [left[0] - right[0] for left, right in zip(gpumd_energy, gromacs_energy)]
    total_differences = [left[1] - right[1] for left, right in zip(gpumd_energy, gromacs_energy)]

    metrics = {
        "coordinate RMSD (A)": rms(coordinate_differences),
        "bond-length RMSD (A)": rms(bond_differences),
        "all-pair-distance RMSD (A)": rms(pair_differences),
        "radius-of-gyration RMSD (A)": rms(rg_differences),
        "potential-energy RMSD (eV)": rms(potential_differences),
        "total-energy RMSD (eV)": rms(total_differences),
    }
    thresholds = {
        "coordinate RMSD (A)": 2.0e-3,
        "bond-length RMSD (A)": 2.0e-3,
        "all-pair-distance RMSD (A)": 3.0e-3,
        "radius-of-gyration RMSD (A)": 2.0e-3,
        "potential-energy RMSD (eV)": 3.0e-3,
        "total-energy RMSD (eV)": 2.0e-3,
    }
    if args.compare_angles:
        metrics["angle RMSD (rad)"] = rms(angle_differences)
        thresholds["angle RMSD (rad)"] = 2.0e-3
    if args.compare_dihedrals:
        metrics["signed-dihedral RMSD (rad)"] = rms(dihedral_differences)
        thresholds["signed-dihedral RMSD (rad)"] = 5.0e-3
    for name, value in metrics.items():
        print(f"{name}: {value:.8e}")

    failures = [name for name, value in metrics.items() if value > thresholds[name]]
    if failures:
        for name in failures:
            print(f"FAIL: {name} exceeds tolerance {thresholds[name]:.3e}")
        return 1

    final_bonds_gpumd = bond_lengths(gpumd_frames[-1])
    final_bonds_gromacs = bond_lengths(gromacs_frames[-1])
    print("Final bond lengths (A):")
    for index, (gpumd_value, gromacs_value) in enumerate(
        zip(final_bonds_gpumd, final_bonds_gromacs), start=1
    ):
        print(f"  {index}-{index + 1}: GPUMD={gpumd_value:.8f}, GROMACS={gromacs_value:.8f}")
    print(f"PASS: GPUMD and GROMACS {args.case_name} NVE trajectories agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
