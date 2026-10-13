import re
import numpy as np

C = 103.64269652680505  # amu Å²/fs² -> eV
KB = 8.617333262145e-5   # eV/K

def read_model():
    with open("model.xyz") as f:
        n = int(f.readline())
        f.readline()
        rows = [f.readline().split() for _ in range(n)]

    r = np.array([[float(x) for x in a[1:4]] for a in rows])
    m = np.array([float(a[4]) for a in rows])
    v = np.array([[float(x) for x in a[5:8]] for a in rows])
    g = np.array([int(a[8]) for a in rows])
    return r, v, m, g


def read_trajectory():
    positions, velocities, masses, labels, forces, times = (
        [], [], [], [], [], []
    )

    with open("trajectory.xyz") as f:
        while line := f.readline():
            if not line.strip():
                continue

            n = int(line)
            header = f.readline()
            match = re.search(r"\bTime=([0-9.eE+-]+)", header)
            assert match, "Missing Time field"
            times.append(float(match.group(1)))

            rr, vv, mm, gg, ff = [], [], [], [], []

            for _ in range(n):
                a = f.readline().split()
                assert len(a) == 12, "Unexpected XYZ columns"

                rr.append([float(x) for x in a[1:4]])
                mm.append(float(a[4]))
                vv.append([float(x) for x in a[5:8]])
                ff.append([float(x) for x in a[8:11]])
                gg.append(int(a[11]))

            positions.append(rr)
            velocities.append(vv)
            masses.append(mm)
            labels.append(gg)
            forces.append(ff)

    return (
        np.array(positions),
        np.array(velocities),
        np.array(masses),
        np.array(labels),
        np.array(forces),
        np.array(times),
    )


r0, v0, m, groups = read_model()
r, v, mout, gout, forces, t = read_trajectory()

assert len(m) == 9
assert np.array_equal(groups, [0, 0, 0, 1, 1, 1, 2, 3, 4])
assert len(t) == 100
assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))
assert np.allclose(mout, m)
assert np.all(gout == groups)
assert np.allclose(t, np.arange(1, 101) * 5.0)

# 1. Free particles: analytical straight-line trajectory
free = np.array([6, 7, 8])

r_expected = (
    r0[free][None, :, :]
    + t[:, None, None] * v0[free][None, :, :]
)

free_position_error = np.max(
    np.linalg.norm(r[:, free] - r_expected, axis=2)
)

free_velocity_error = np.max(
    np.linalg.norm(v[:, free] - v0[free], axis=2)
)

# 2. Rigid body COM trajectories and constraints
com_error = 0.0
distance_error = 0.0
constraint_velocity_error = 0.0
angular_momentum_error = 0.0

for ids in (np.arange(3), np.arange(3, 6)):
    mass = m[ids]
    M = mass.sum()

    rc0 = np.average(r0[ids], axis=0, weights=mass)
    vc0 = np.average(v0[ids], axis=0, weights=mass)

    rc = np.average(r[:, ids], axis=1, weights=mass)
    vc = np.average(v[:, ids], axis=1, weights=mass)

    expected_com = rc0 + t[:, None] * vc0

    com_error = max(
        com_error,
        np.max(np.linalg.norm(rc - expected_com, axis=1))
    )

    L0 = np.sum(
        mass[:, None]
        * np.cross(r0[ids] - rc0, v0[ids] - vc0),
        axis=0,
    )

    L = np.sum(
        mass[None, :, None]
        * np.cross(
            r[:, ids] - rc[:, None, :],
            v[:, ids] - vc[:, None, :],
        ),
        axis=1,
    )

    angular_momentum_error = max(
        angular_momentum_error,
        np.max(np.linalg.norm(L - L0, axis=1))
    )

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]

            dr = r[:, a] - r[:, b]
            dv = v[:, a] - v[:, b]

            d0 = np.linalg.norm(r0[a] - r0[b])

            distance_error = max(
                distance_error,
                np.max(np.abs(np.linalg.norm(dr, axis=1) - d0))
            )

            constraint_velocity_error = max(
                constraint_velocity_error,
                np.max(np.abs(np.sum(dr * dv, axis=1)))
            )

# 3. Kinetic energy and temperature DOFs
K = 0.5 * C * np.sum(
    m[None, :, None] * v**2, axis=(1, 2)
)

K0 = 0.5 * C * np.sum(m[:, None] * v0**2)

dof = 3 * 3 + 6 * 2
assert dof == 21

T_expected = 2.0 * K / (KB * dof)

thermo = np.atleast_2d(np.loadtxt("thermo.out"))
assert len(thermo) == len(t)

T_output = thermo[:, 0]
K_output = thermo[:, 1]
U_output = thermo[:, 2]

temperature_error = np.max(np.abs(T_output - T_expected))
kinetic_output_error = np.max(np.abs(K_output - K))
energy_error = np.max(np.abs(K + U_output - K0))
max_force = np.max(np.linalg.norm(forces, axis=2))

print("\n=== Mixed Free/Rigid Test ===")
print(f"Frames:                     {len(t)}")
print(f"Free particle position err: {free_position_error:.8e} Å")
print(f"Free particle velocity err: {free_velocity_error:.8e} Å/fs")
print(f"Rigid COM position err:     {com_error:.8e} Å")
print(f"Rigid distance err:         {distance_error:.8e} Å")
print(f"Velocity constraint err:    {constraint_velocity_error:.8e}")
print(f"Angular momentum error:     {angular_momentum_error:.8e}")
print(f"Kinetic energy error:       {energy_error:.8e} eV")
print(f"Thermo KE mismatch:         {kinetic_output_error:.8e} eV")
print(f"Temperature error:          {temperature_error:.8e} K")
print(f"Maximum force:              {max_force:.8e} eV/Å")
print(f"Expected thermal DOFs:      {dof}")
print(f"Expected initial KE:        {K0:.12f} eV")
print(f"Expected initial T:         {2*K0/(KB*dof):.8f} K")

checks = {
    "Free position": free_position_error < 1e-8,
    "Free velocity": free_velocity_error < 1e-10,
    "Rigid COM": com_error < 1e-8,
    "Rigid distances": distance_error < 1e-8,
    "Rigid velocity constraints": constraint_velocity_error < 1e-8,
    "Angular momentum": angular_momentum_error < 1e-8,
    "Energy": energy_error < 1e-6,
    "Thermo kinetic energy": kinetic_output_error < 1e-6,
    "Temperature / DOFs": temperature_error < 1e-3,
    "Zero interaction force": max_force < 1e-10,
}

print("\n=== Validation ===")
for name, ok in checks.items():
    print(f"{name:28s}: {'PASS' if ok else 'FAIL'}")

if not all(checks.values()):
    raise SystemExit(1)

print("\nAll tests passed.")
