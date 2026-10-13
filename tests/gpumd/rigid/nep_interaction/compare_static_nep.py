#!/usr/bin/env python3
"""Compare forces/energies from rigid NVE against an independent static GPUMD NEP run.

Place in tests/gpumd/rigid/nep_interaction and execute from any directory:
    python3 compare_static_nep.py

All generated reference calculations live in nep_interaction/static_reference/.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GPUMD = Path(os.environ.get("GPUMD_BIN", str(ROOT / "build/gpumd"))).expanduser().resolve()
FRAME_INDICES = (0, 49, -1)  # first, middle, last saved frame
FORCE_ATOL = 1.0e-4  # eV/angstrom
FORCE_RTOL = 1.0e-5
ENERGY_ATOL = 1.0e-4  # eV (total potential energy)
ENERGY_RTOL = 1.0e-6


def property_columns(header: str) -> dict[str, slice]:
    match = re.search(r"\bproperties=([^\s]+)", header, re.IGNORECASE)
    if not match:
        raise ValueError("Missing properties=... in XYZ header")
    parts = match.group(1).split(":")
    if len(parts) % 3:
        raise ValueError(f"Malformed XYZ properties: {match.group(1)}")
    result: dict[str, slice] = {}
    offset = 0
    for idx in range(0, len(parts), 3):
        name, _kind, count_text = parts[idx:idx + 3]
        count = int(count_text)
        result[name] = slice(offset, offset + count)
        offset += count
    return result


def read_xyz_frames(path: Path) -> list[dict]:
    frames = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            natoms = int(line.strip())
            header = f.readline()
            if not header:
                raise ValueError(f"Truncated header in {path}")
            columns = property_columns(header)
            rows = []
            for _ in range(natoms):
                row = f.readline()
                if not row:
                    raise ValueError(f"Truncated frame in {path}")
                rows.append(row.split())
            positions = np.array([
                [float(x) for x in row[columns["pos"]]] for row in rows
            ], dtype=float)
            forces_name = "forces" if "forces" in columns else "force"
            if forces_name not in columns:
                raise ValueError(f"No force property in {path}")
            forces = np.array([
                [float(x) for x in row[columns[forces_name]]] for row in rows
            ], dtype=float)
            energy_name = "energy_atom" if "energy_atom" in columns else "potential"
            if energy_name not in columns:
                raise ValueError(
                    f"No energy_atom property in {path}. Add 'potential' to dump_xyz."
                )
            energies = np.array([
                float(row[columns[energy_name]][0]) for row in rows
            ], dtype=float)
            symbols = [row[columns["species"]][0] for row in rows]
            time_match = re.search(r"\bTime=([^\s]+)", header)
            times = float(time_match.group(1)) if time_match else None
            frames.append({
                "positions": positions,
                "forces": forces,
                "energies": energies,
                "symbols": symbols,
                "time": times,
            })
    return frames


def parse_model(path: Path):
    lines = path.read_text().splitlines()
    natoms = int(lines[0].strip())
    header = lines[1]
    cols = property_columns(header)
    if len(lines) < natoms + 2:
        raise ValueError("model.xyz is truncated")
    rows = [x.split() for x in lines[2:2 + natoms]]
    masses = np.array([float(row[cols["mass"]][0]) for row in rows])
    groups = np.array([int(row[cols["group"]][0]) for row in rows])
    symbols = [row[cols["species"]][0] for row in rows]
    return natoms, header, cols, rows, masses, groups, symbols


def potential_from_run_in(path: Path) -> Path:
    for line in path.read_text().splitlines():
        tokens = shlex.split(line.partition("#")[0])
        if tokens and tokens[0] == "potential":
            return (path.parent / tokens[1]).resolve()
    raise ValueError("Could not find 'potential' in run.in")


def write_reference_model(
    dest: Path, original_header: str, cols: dict[str, slice],
    model_rows: list[list[str]], positions: np.ndarray,
) -> None:
    pos_col = cols["pos"]
    updated_rows = []
    for original, xyz in zip(model_rows, positions):
        row = original.copy()
        row[pos_col] = [f"{v:.17g}" for v in xyz]
        updated_rows.append(" ".join(row))
    dest.write_text(
        f"{len(model_rows)}\n{original_header}\n"
        + "\n".join(updated_rows) + "\n"
    )


def body_loads(
    position: np.ndarray, force: np.ndarray,
    masses: np.ndarray, groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    totals, torques = [], []
    for label in np.unique(groups):
        ids = np.flatnonzero(groups == label)
        com = np.average(position[ids], axis=0, weights=masses[ids])
        totals.append(force[ids].sum(axis=0))
        torques.append(np.cross(position[ids] - com, force[ids]).sum(axis=0))
    return np.array(totals), np.array(torques)


def main() -> None:
    if not GPUMD.is_file():
        raise FileNotFoundError(f"GPUMD executable not found: {GPUMD}")
    trajectory_path = HERE / "trajectory.xyz"
    if not trajectory_path.is_file():
        raise FileNotFoundError("Missing trajectory.xyz: first run ../run.sh nep_interaction")

    potential = potential_from_run_in(HERE / "run.in")
    if not potential.is_file():
        raise FileNotFoundError(f"NEP potential not found: {potential}")
    if not potential.read_text().split()[0].startswith("nep4"):
        raise ValueError("The test requires a standard NEP4 potential")

    natoms, header, cols, rows, masses, groups, symbols = parse_model(HERE / "model.xyz")
    rigid_frames = read_xyz_frames(trajectory_path)
    if not rigid_frames:
        raise ValueError("Empty rigid trajectory")
    targets = list(dict.fromkeys(i % len(rigid_frames) for i in FRAME_INDICES))
    output_base = HERE / "static_reference"
    output_base.mkdir(exist_ok=True)
    failures = []

    print(f"NEP: {potential}")
    print(f"GPUMD: {GPUMD}")
    print(f"Comparing saved frame indices: {targets}")
    print("Force tolerances: atol=%.1e eV/Å, rtol=%.1e" % (FORCE_ATOL, FORCE_RTOL))

    for i in targets:
        frame = rigid_frames[i]
        if frame["positions"].shape != (natoms, 3):
            raise ValueError(f"Atom count mismatch in rigid frame {i}")
        if frame["symbols"] != symbols:
            raise ValueError(f"Atom ordering/species mismatch in rigid frame {i}")

        work = output_base / f"frame_{i:04d}"
        work.mkdir(parents=True, exist_ok=True)
        write_reference_model(work / "model.xyz", header, cols, rows, frame["positions"])
        (work / "run.in").write_text(
            f"potential {potential}\n"
            "ensemble nve\n"
            "time_step 0\n"
            "dump_xyz 1 reference.xyz precision double force potential\n"
            "run 1\n"
        )
        (work / "reference.xyz").unlink(missing_ok=True)
        with (work / "gpumd.log").open("w") as log:
            proc = subprocess.run(
                [str(GPUMD)], cwd=work, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if proc.returncode:
            tail = "\n".join((work / "gpumd.log").read_text().splitlines()[-30:])
            raise RuntimeError(f"Static GPUMD failed for frame {i}:\n{tail}")

        references = read_xyz_frames(work / "reference.xyz")
        if len(references) != 1:
            raise ValueError(f"Expected one static reference frame; got {len(references)}")
        ref = references[0]
        coordinate_error = np.max(np.linalg.norm(
            frame["positions"] - ref["positions"], axis=1
        ))
        df = frame["forces"] - ref["forces"]
        force_abs = np.max(np.linalg.norm(df, axis=1))
        force_relative = force_abs / max(np.max(np.linalg.norm(ref["forces"], axis=1)), 1e-12)
        energy_abs = abs(frame["energies"].sum() - ref["energies"].sum())
        energy_rel = energy_abs / max(abs(ref["energies"].sum()), 1.0)
        per_atom_energy_abs = np.max(np.abs(frame["energies"] - ref["energies"]))

        rigid_forces, rigid_torques = body_loads(
            frame["positions"], frame["forces"], masses, groups
        )
        ref_forces, ref_torques = body_loads(
            ref["positions"], ref["forces"], masses, groups
        )
        body_force_abs = np.max(np.linalg.norm(rigid_forces - ref_forces, axis=1))
        body_torque_abs = np.max(np.linalg.norm(rigid_torques - ref_torques, axis=1))

        is_ok = (
            coordinate_error < 1e-10
            and np.allclose(frame["forces"], ref["forces"], atol=FORCE_ATOL, rtol=FORCE_RTOL)
            and np.isclose(frame["energies"].sum(), ref["energies"].sum(),
                           atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
        )
        print(f"\nFrame {i}  t={frame['time']} fs  {'PASS' if is_ok else 'FAIL'}")
        print(f"  max coordinate difference:  {coordinate_error:.8e} Å")
        print(f"  max per-bead |ΔF|:          {force_abs:.8e} eV/Å")
        print(f"  relative force discrepancy:{force_relative:13.8e}")
        print(f"  max per-bead |ΔU_i|:        {per_atom_energy_abs:.8e} eV")
        print(f"  total |ΔU|:                {energy_abs:.8e} eV")
        print(f"  relative total energy diff:{energy_rel:13.8e}")
        print(f"  max body |ΔF|:             {body_force_abs:.8e} eV/Å")
        print(f"  max body |Δτ|:             {body_torque_abs:.8e} eV")
        if not is_ok:
            failures.append(i)

    if failures:
        raise SystemExit(f"FAIL: mismatches at frames {failures}; inspect static_reference/")
    print("\nPASS: NEP interaction forces and potential energies match static NVE references.")


if __name__ == "__main__":
    main()
