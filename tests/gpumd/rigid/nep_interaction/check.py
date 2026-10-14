#!/usr/bin/env python3
"""Regression checks for NEP + rigid_nve with the six-bead, two-body test model.

Requires compare_static_nep.py in the same directory for the shared XYZ reader.
Suggested run.in: time_step 0.025, run 4000, dump_thermo 40,
    dump_xyz 40 trajectory.xyz precision double velocity force potential group_labels
"""

import numpy as np
from compare_static_nep import read_xyz_frames, parse_model
from pathlib import Path

CONV = 103.64269652680505  # amu Angstrom^2/fs^2 -> eV
HERE = Path(__file__).resolve().parent

natoms, _, cols, model_rows, mass, groups, symbols = parse_model(HERE / "model.xyz")
r0 = np.array([[float(z) for z in row[cols["pos"]]] for row in model_rows])
v0 = np.array([[float(z) for z in row[cols["vel"]]] for row in model_rows])
assert natoms == 6 and np.array_equal(groups, [0, 0, 0, 1, 1, 1])

frames = read_xyz_frames(HERE / "trajectory.xyz")
r = np.array([frame["positions"] for frame in frames])
f = np.array([frame["forces"] for frame in frames])
times = np.array([frame["time"] for frame in frames])
if not frames or not all(frame["time"] is not None for frame in frames):
    raise ValueError("Expected time-stamped trajectory frames")
if len(frames) < 3 or np.any(np.diff(times) <= 0):
    raise ValueError("Need at least 3 strictly increasing output times")

# Read velocity columns separately; the static-reference reader deliberately
# focuses on positions, interaction forces and atomic energies.
def read_velocity_frames(path):
    import re
    from compare_static_nep import property_columns
    out = []
    with path.open() as stream:
        while line := stream.readline():
            if not line.strip():
                continue
            n = int(line)
            header = stream.readline()
            columns = property_columns(header)
            if "vel" not in columns:
                raise ValueError("Add 'velocity' to dump_xyz in run.in")
            out.append(np.array([
                [float(x) for x in stream.readline().split()[columns["vel"]]]
                for _ in range(n)
            ]))
    return np.array(out)

v = read_velocity_frames(HERE / "trajectory.xyz")
if r.shape != v.shape or r.shape != f.shape:
    raise ValueError("Incompatible positions, velocities and forces")
if not all(np.all(np.isfinite(x)) for x in (r, v, f)):
    raise ValueError("Non-finite values found")
if np.any((r <= 2) | (r >= 38)):
    raise ValueError("A bead approached the box boundary; this simple angular-momentum check needs unwrapped positions")

# Rigid constraints, including the velocity-level constraint.
distance_error = 0.0
velocity_constraint_error = 0.0
for label in np.unique(groups):
    ids = np.flatnonzero(groups == label)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            dr = r[:, a] - r[:, b]
            dv = v[:, a] - v[:, b]
            d0 = np.linalg.norm(r0[a] - r0[b])
            distance_error = max(distance_error, np.max(np.abs(np.linalg.norm(dr, axis=1) - d0)))
            velocity_constraint_error = max(
                velocity_constraint_error,
                np.max(np.abs(np.sum(dr * dv, axis=1))),
            )

P0 = np.sum(mass[:, None] * v0, axis=0)
P = np.sum(mass[None, :, None] * v, axis=1)
P_error = np.max(np.linalg.norm(P - P0, axis=1))


def angular_momentum(pos, vel, m):
    com = np.average(pos, axis=-2, weights=m)
    vcom = np.average(vel, axis=-2, weights=m)
    rel_pos = pos - np.expand_dims(com, axis=-2)
    rel_vel = vel - np.expand_dims(vcom, axis=-2)
    weights = m.reshape((1,) * (pos.ndim - 2) + (len(m), 1))
    return np.sum(weights * np.cross(rel_pos, rel_vel), axis=-2)

L0 = angular_momentum(r0, v0, mass)
L = angular_momentum(r, v, mass)
L_rel = np.max(np.linalg.norm(L - L0, axis=1)) / max(np.linalg.norm(L0), 1e-14)

thermo = np.atleast_2d(np.loadtxt(HERE / "thermo.out"))
if len(thermo) != len(frames):
    raise ValueError("thermo.out and trajectory.xyz have different frame counts")
K, U = thermo[:, 1], thermo[:, 2]
E = K + U
K0 = 0.5 * CONV * np.sum(mass[:, None] * v0**2)
E_abs_error = np.max(np.abs(E - E[0]))
E_relative_to_K0 = E_abs_error / K0

# Force and torque of each body. The values are interaction-only bead loads.
def body_quantities(ids):
    m = mass[ids]
    p = r[:, ids]
    vel = v[:, ids]
    forces = f[:, ids]
    M = m.sum()
    com = np.average(p, axis=1, weights=m)
    vcom = np.average(vel, axis=1, weights=m)
    body_force = forces.sum(axis=1)
    body_torque = np.cross(p - com[:, None, :], forces).sum(axis=1)
    spin = angular_momentum(p, vel, m)
    return M, vcom, body_force, body_torque, spin

ma, va, fa, taua, La = body_quantities(np.flatnonzero(groups == 0))
mb, vb, fb, taub, Lb = body_quantities(np.flatnonzero(groups == 1))
force_balance = np.max(np.linalg.norm(fa + fb, axis=1))
max_force = np.max(np.linalg.norm(fa, axis=1))
max_torque = np.max(np.linalg.norm(taua, axis=1))

# Centered differences are only diagnostics; they also have O(dump_interval^2) error.
central_dt = (times[2:] - times[:-2])[:, None]
acc_fd = (va[2:] - va[:-2]) / central_dt
acc_pred = fa[1:-1] / (ma * CONV)
spin_fd = (La[2:] - La[:-2]) / central_dt
spin_pred = taua[1:-1] / CONV


def residual(numerical, predicted):
    return np.max(np.linalg.norm(numerical - predicted, axis=1)) / max(
        np.max(np.linalg.norm(predicted, axis=1)), 1e-12
    )

acc_res = residual(acc_fd, acc_pred)
torque_res = residual(spin_fd, spin_pred)

print("\n=== NEP Interacting Rigid Bodies ===")
print(f"Frames:                    {len(frames)}")
print(f"Distance error (Å):        {distance_error:.8e}")
print(f"Velocity constraint error:{velocity_constraint_error:13.8e}")
print(f"Linear momentum error:    {P_error:.8e}")
print(f"Angular momentum rel err:{L_rel:13.8e}")
print(f"Maximum |E - E[0]|:       {E_abs_error:.8e} eV")
print(f"Energy deviation / K0:    {E_relative_to_K0:.8e}")
print(f"Final energy change:      {E[-1] - E[0]:.8e} eV")
print(f"Net-force imbalance:      {force_balance:.8e} eV/Å")
print(f"Maximum body force:       {max_force:.8e} eV/Å")
print(f"Maximum body torque:      {max_torque:.8e} eV")
print(f"COM acceleration residual:{acc_res:13.8e}")
print(f"Torque evolution residual:{torque_res:13.8e}")

checks = {
    "Rigid distances": distance_error < 1e-8,
    "Velocity constraints": velocity_constraint_error < 1e-8,
    "Linear momentum": P_error < 1e-5,
    "Angular momentum": L_rel < 1e-3,
    # Test-specific numerical gate for 100 fs at dt=0.025 fs.
    "NVE energy (test-specific)": E_abs_error < 1e-4,
    "Action-reaction": force_balance < 1e-5,
    "Nonzero force": max_force > 1e-4,
    "Nonzero torque": max_torque > 1e-4,
    "COM acceleration": acc_res < 0.1,
    "Torque evolution": torque_res < 0.1,
}
print("\n=== Validation ===")
for key, good in checks.items():
    print(f"{key:27s}: {'PASS' if good else 'FAIL'}")
if not all(checks.values()):
    raise SystemExit(1)
print("\nPASS: NEP rigid-body regression checks")
