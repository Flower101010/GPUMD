#!/usr/bin/env python3
"""Numerically verify a free rigid triangle initially wrapped across X and Y PBC.

The independent reference is a constant-COM-velocity, constant-principal-axis
rotation solution.  Trajectory positions are matched modulo lattice vectors.
"""

from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
CONV = 103.64269652680505  # amu * Å²/fs² -> eV
KB = 8.617333262145e-5    # eV/K
OMEGA = np.array([0.0, 0.0, 0.0015])  # rad/fs
TIME_STEP = 0.5             # fs
DUMP_INTERVAL = 10
STEPS = 1000


def mic(delta, box_lengths):
    return delta - box_lengths * np.rint(delta / box_lengths)


def parse_fields(header):
    match = re.search(r"\bProperties=([^\s]+)", header, flags=re.I)
    if not match:
        raise ValueError("Missing Properties= in XYZ header")
    tokens = match.group(1).split(":")
    if len(tokens) % 3:
        raise ValueError("Invalid extended-XYZ property layout")
    result = {}
    col = 0
    for k in range(0, len(tokens), 3):
        count = int(tokens[k + 2])
        result[tokens[k].lower()] = slice(col, col + count)
        col += count
    return result, col


def read_model():
    with (HERE / "model.xyz").open() as f:
        n = int(f.readline())
        header = f.readline()
        rows = [f.readline().split() for _ in range(n)]

    assert n == 3
    lattice = re.search(r'\blattice="([^"]+)"', header, flags=re.I)
    assert lattice, "Missing lattice"
    H = np.fromstring(lattice.group(1), sep=" ").reshape(3, 3)
    box = np.diag(H).copy()
    assert np.allclose(H, np.diag(box)) and np.all(box > 0)
    assert re.search(r'\bpbc="T T T"', header, flags=re.I)
    fields, ncol = parse_fields(header)
    assert all(k in fields for k in ("species", "pos", "mass", "vel", "group"))

    def col(row, key, dtype=float):
        assert len(row) == ncol, "Invalid model.xyz row length"
        return np.asarray(row[fields[key]], dtype=dtype)

    r_wrapped = np.array([col(row, "pos") for row in rows])
    v0 = np.array([col(row, "vel") for row in rows])
    mass = np.array([col(row, "mass")[0] for row in rows])
    groups = np.array([int(col(row, "group", int)[0]) for row in rows])
    assert np.all(groups == 0), "Expected one group of three beads"
    assert np.all(mass > 0) and np.all(np.isfinite(v0))
    assert np.all((r_wrapped >= 0) & (r_wrapped < box))

    # Test fixture genuinely crosses BOTH X and Y in its wrapped representation.
    assert np.all(np.ptp(r_wrapped, axis=0)[:2] > box[:2] / 2)
    # Reconstruct coherent initial images relative to bead 0 (span < half box).
    r0 = r_wrapped[0] + mic(r_wrapped - r_wrapped[0], box)
    assert np.all(np.ptp(r0, axis=0) < box / 2)

    com0 = np.average(r0, axis=0, weights=mass)
    vcom0 = np.average(v0, axis=0, weights=mass)
    expected_v0 = vcom0 + np.cross(OMEGA, r0 - com0)
    assert np.max(np.abs(expected_v0 - v0)) < 1e-12, (
        "Initial velocities must describe exact rigid rotation"
    )
    return box, r_wrapped, r0, v0, mass, groups, com0, vcom0


def read_trajectory():
    r, v, f, m, group, times = [], [], [], [], [], []
    with (HERE / "trajectory.xyz").open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line)
            assert n == 3
            header = stream.readline()
            tm = re.search(r"\bTime=([-+\deE.]+)", header)
            assert tm, "No Time= in trajectory header"
            times.append(float(tm.group(1)))
            fields, ncol = parse_fields(header)
            for required in ("species", "pos", "mass", "vel", "forces", "group"):
                assert required in fields, f"Trajectory missing {required}"

            frame_r, frame_v, frame_f, frame_m, frame_g = [], [], [], [], []
            for _ in range(n):
                row = stream.readline().split()
                assert len(row) == ncol, "Truncated trajectory frame"
                frame_r.append([float(x) for x in row[fields["pos"]]])
                frame_v.append([float(x) for x in row[fields["vel"]]])
                frame_f.append([float(x) for x in row[fields["forces"]]])
                frame_m.append(float(row[fields["mass"]][0]))
                frame_g.append(int(row[fields["group"]][0]))
            r.append(frame_r)
            v.append(frame_v)
            f.append(frame_f)
            m.append(frame_m)
            group.append(frame_g)

    return (np.asarray(r), np.asarray(v), np.asarray(f),
            np.asarray(m), np.asarray(group), np.asarray(times))


def main():
    box, r_wrapped, r0, v0, mass, groups, com0, vcom0 = read_model()
    r, v, forces, mout, gout, times = read_trajectory()

    expected_times = np.arange(1, STEPS // DUMP_INTERVAL + 1) * TIME_STEP * DUMP_INTERVAL
    assert len(times) == len(expected_times), f"Wrong number of frames: {len(times)}"
    assert np.allclose(times, expected_times, rtol=0, atol=1e-6)
    assert all(np.all(np.isfinite(x)) for x in (r, v, forces, mout))
    assert np.allclose(mout, mass, atol=0, rtol=0)
    assert np.all(gout == groups)
    assert np.all((r >= -1e-10) & (r < box + 1e-10)), "Positions not wrapped into [0,L)"

    # Exact free-rotation reference about principal Z axis.
    rho = r0 - com0
    theta = OMEGA[2] * times
    c, s = np.cos(theta), np.sin(theta)
    rho_t = np.empty((len(times), 3, 3))
    rho_t[:, :, 0] = c[:, None] * rho[None, :, 0] - s[:, None] * rho[None, :, 1]
    rho_t[:, :, 1] = s[:, None] * rho[None, :, 0] + c[:, None] * rho[None, :, 1]
    rho_t[:, :, 2] = rho[None, :, 2]
    r_ref = com0 + times[:, None, None] * vcom0 + rho_t
    v_ref = vcom0 + np.cross(OMEGA, rho_t)

    # A wrapped coordinate can differ by integer lattice translations.
    position_error = np.max(np.linalg.norm(mic(r - r_ref, box), axis=-1))
    velocity_error = np.max(np.linalg.norm(v - v_ref, axis=-1))

    # Use the independent reference to choose coherent periodic images for COM/L.
    r_unwrapped = r + box * np.rint((r_ref - r) / box)
    com = np.average(r_unwrapped, axis=1, weights=mass)
    vcom = np.average(v, axis=1, weights=mass)
    com_ref = com0 + times[:, None] * vcom0
    com_error = np.max(np.linalg.norm(com - com_ref, axis=1))
    vcom_error = np.max(np.linalg.norm(vcom - vcom0, axis=1))

    # Wrapped pair differences must reproduce the rigid shape.
    distance_error = 0.0
    velocity_constraint_error = 0.0
    for i, j in ((0, 1), (0, 2), (1, 2)):
        dr = mic(r[:, i] - r[:, j], box)
        dv = v[:, i] - v[:, j]
        d0 = np.linalg.norm(r0[i] - r0[j])
        distance_error = max(distance_error,
                             float(np.max(np.abs(np.linalg.norm(dr, axis=1) - d0))))
        velocity_constraint_error = max(
            velocity_constraint_error,
            float(np.max(np.abs(np.sum(dr * dv, axis=1))))
        )

    L0 = np.sum(mass[:, None] * np.cross(r0 - com0, v0 - vcom0), axis=0)
    L = np.sum(mass[None, :, None] * np.cross(
        r_unwrapped - com[:, None, :], v - vcom[:, None, :]), axis=1)
    L_error = np.max(np.linalg.norm(L - L0, axis=1))

    k0 = 0.5 * CONV * np.sum(mass[:, None] * v0**2)
    k = 0.5 * CONV * np.sum(mass[None, :, None] * v**2, axis=(1, 2))
    kinetic_error = np.max(np.abs(k - k0))
    thermo = np.atleast_2d(np.loadtxt(HERE / "thermo.out"))
    assert len(thermo) == len(times)
    assert np.all(np.isfinite(thermo))
    temperature_expected = 2 * k / (6 * KB)
    temp_error = np.max(np.abs(thermo[:, 0] - temperature_expected))
    thermo_k_error = np.max(np.abs(thermo[:, 1] - k))
    max_pe = np.max(np.abs(thermo[:, 2]))
    max_force = np.max(np.linalg.norm(forces, axis=-1))

    # Fixture should exercise further periodic crossings during propagation.
    first_cells = np.floor(r0 / box)
    last_cells = np.floor(r_ref[-1] / box)
    crossings = int(np.count_nonzero(first_cells != last_cells))

    print("\n=== Initially Wrapped PBC / Rigid Body ===")
    print(f"Frames:                              {len(times)}")
    print(f"Initial wrapped span (Å):           {np.ptp(r_wrapped,axis=0)}")
    print(f"Reconstructed coherent span (Å):    {np.ptp(r0,axis=0)}")
    print(f"Reference cell-index changes:       {crossings}")
    print(f"Analytical position error (Å):      {position_error:.8e}")
    print(f"Analytical velocity error (Å/fs):   {velocity_error:.8e}")
    print(f"COM position error (Å):             {com_error:.8e}")
    print(f"COM velocity error (Å/fs):          {vcom_error:.8e}")
    print(f"Internal distance error (Å):        {distance_error:.8e}")
    print(f"Velocity constraint residual:       {velocity_constraint_error:.8e}")
    print(f"Angular momentum error:             {L_error:.8e}")
    print(f"Kinetic energy error (eV):          {kinetic_error:.8e}")
    print(f"Thermo kinetic mismatch (eV):       {thermo_k_error:.8e}")
    print(f"Temperature mismatch (K):           {temp_error:.8e}")
    print(f"Maximum potential energy (eV):      {max_pe:.8e}")
    print(f"Maximum interaction force (eV/Å):   {max_force:.8e}")
    print(f"Reference initial KE (eV):          {k0:.12f}")
    print(f"Reference temperature (K):          {2*k0/(6*KB):.9f}")

    checks = {
        "Analytical positions": position_error < 1e-8,
        "Analytical velocities": velocity_error < 1e-9,
        "COM trajectory": com_error < 1e-8,
        "COM velocity": vcom_error < 1e-10,
        "Rigid distances": distance_error < 1e-8,
        "Rigid velocity constraints": velocity_constraint_error < 1e-9,
        "Angular momentum": L_error < 1e-8,
        "Kinetic energy": kinetic_error < 1e-6,
        "Thermo kinetic energy": thermo_k_error < 1e-6,
        "Temperature / 6 DOFs": temp_error < 1e-3,
        "Zero potential energy": max_pe < 1e-10,
        "Zero interaction force": max_force < 1e-10,
        "Crossing exercised": crossings > 0,
    }
    print("\n=== Validation ===")
    for label, ok in checks.items():
        print(f"{label:28s}: {'PASS' if ok else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit(1)
    print("\nPASS: Initially wrapped PBC rigid-body integration.")


if __name__ == "__main__":
    main()
