import re
import numpy as np

CONV = 103.64269652680505  # amu Å²/fs² -> eV


def read_model():
    with open("model.xyz") as f:
        n = int(f.readline())
        f.readline()
        rows = [f.readline().split() for _ in range(n)]

    mass = np.array([float(x[4]) for x in rows])
    pos = np.array([[float(z) for z in x[1:4]] for x in rows])
    vel = np.array([[float(z) for z in x[5:8]] for x in rows])
    groups = np.array([int(x[8]) for x in rows])
    return mass, pos, vel, groups


def read_xyz():
    positions, velocities, forces, times = [], [], [], []

    with open("trajectory.xyz") as f:
        while line := f.readline():
            if not line.strip():
                continue

            n = int(line)
            header = f.readline()

            time = re.search(r"\bTime=([0-9.eE+-]+)", header)
            props = re.search(r"\bProperties=([^\s]+)", header)
            assert time and props

            times.append(float(time.group(1)))

            schema = props.group(1).split(":")
            columns = {}
            offset = 0
            for i in range(0, len(schema), 3):
                count = int(schema[i + 2])
                columns[schema[i]] = slice(offset, offset + count)
                offset += count

            assert all(k in columns for k in ("pos", "vel", "forces"))

            p, v, force = [], [], []
            for _ in range(n):
                row = f.readline().split()
                p.append([float(x) for x in row[columns["pos"]]])
                v.append([float(x) for x in row[columns["vel"]]])
                force.append([float(x) for x in row[columns["forces"]]])

            positions.append(p)
            velocities.append(v)
            forces.append(force)

    return (
        np.array(positions),
        np.array(velocities),
        np.array(forces),
        np.array(times),
    )


mass, r0, v0, groups = read_model()
r, v, f, t = read_xyz()

assert len(mass) == 6
assert np.array_equal(groups, [0, 0, 0, 1, 1, 1])
assert len(t) >= 3
assert np.all(np.diff(t) > 0)
assert all(np.all(np.isfinite(x)) for x in (r, v, f))

# These checks assume no periodic boundary crossing.
assert np.all((r > 2.0) & (r < 38.0)), \
    "Body approaches box boundary: use an unwrapped reference"

# 1. Rigid distances
distance_error = 0.0

for group in (0, 1):
    ids = np.flatnonzero(groups == group)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            d0 = np.linalg.norm(r0[i] - r0[j])
            d = np.linalg.norm(r[:, i] - r[:, j], axis=1)
            distance_error = max(
                distance_error, np.max(np.abs(d - d0))
            )

# 2. Total linear momentum
P0 = np.sum(mass[:, None] * v0, axis=0)
P = np.sum(mass[None, :, None] * v, axis=1)
P_error = np.max(np.linalg.norm(P - P0, axis=1))

# 3. Total angular momentum about the system COM
def angular_momentum(pos, vel, m):
    com = np.sum(pos * m[..., None], axis=-2) / np.sum(m)
    vc = np.sum(vel * m[..., None], axis=-2) / np.sum(m)

    rel_pos = pos - np.expand_dims(com, axis=-2)
    rel_vel = vel - np.expand_dims(vc, axis=-2)

    return np.sum(
        m[..., None] * np.cross(rel_pos, rel_vel),
        axis=-2,
    )


L0 = angular_momentum(r0, v0, mass)
L = angular_momentum(r, v, mass)

L_error = np.max(np.linalg.norm(L - L0, axis=1)) \
    / np.linalg.norm(L0)

# 4. Energy conservation
thermo = np.atleast_2d(np.loadtxt("thermo.out"))
assert len(thermo) == len(t)

K = thermo[:, 1]
U = thermo[:, 2]
E = K + U

K0 = 0.5 * CONV * np.sum(mass[:, None] * v0**2)
E_error = np.max(np.abs(E - E[0])) / K0

# 5. Force and torque on body A/B
ids_a = np.where(groups == 0)[0]
ids_b = np.where(groups == 1)[0]


def body_quantities(ids):
    m = mass[ids]
    total_mass = m.sum()
    pos = r[:, ids]
    vel = v[:, ids]
    forces = f[:, ids]

    com = np.sum(pos * m[None, :, None], axis=1) / total_mass
    vc = np.sum(vel * m[None, :, None], axis=1) / total_mass
    net_force = forces.sum(axis=1)

    rel_pos = pos - com[:, None]
    torque = np.cross(rel_pos, forces).sum(axis=1)

    spin = angular_momentum(pos, vel, m)

    return total_mass, vc, net_force, torque, spin


ma, va, fa, taua, La = body_quantities(ids_a)
mb, vb, fb, taub, Lb = body_quantities(ids_b)

newton_error = np.max(np.linalg.norm(fa + fb, axis=1))

force_magnitude = np.max(np.linalg.norm(fa, axis=1))
torque_magnitude = np.max(np.linalg.norm(taua, axis=1))

# 6. Numerical time derivatives: acceleration and spin evolution
dt = (t[2:] - t[:-2])[:, None]

acc_fd = (va[2:] - va[:-2]) / dt
acc_force = fa[1:-1] / (ma * CONV)

spin_fd = (La[2:] - La[:-2]) / dt
spin_torque = taua[1:-1] / CONV


def relative_residual(numerical, predicted):
    err = np.max(np.linalg.norm(numerical - predicted, axis=1))
    scale = max(np.max(np.linalg.norm(predicted, axis=1)), 1e-12)
    return err / scale


acc_error = relative_residual(acc_fd, acc_force)
torque_error = relative_residual(spin_fd, spin_torque)

print("\n=== Interacting Rigid Bodies ===")
print(f"Frames:                     {len(t)}")
print(f"Distance error (Å):         {distance_error:.8e}")
print(f"Linear momentum error:      {P_error:.8e}")
print(f"Angular momentum rel err:   {L_error:.8e}")
print(f"Total energy rel drift:     {E_error:.8e}")
print(f"Net-force balance (eV/Å):   {newton_error:.8e}")
print(f"Maximum body force (eV/Å):  {force_magnitude:.8e}")
print(f"Maximum body torque (eV):   {torque_magnitude:.8e}")
print(f"COM acceleration residual:  {acc_error:.8e}")
print(f"Torque evolution residual:  {torque_error:.8e}")

checks = {
    "Rigid distances": distance_error < 1e-8,
    "Linear momentum": P_error < 1e-5,
    "Angular momentum": L_error < 1e-3,
    "Total energy": E_error < 1e-3,
    "Action-reaction": newton_error < 1e-5,
    "Nonzero force": force_magnitude > 1e-4,
    "Nonzero torque": torque_magnitude > 1e-4,
    "COM acceleration": acc_error < 0.1,
    "Torque evolution": torque_error < 0.1,
}

print("\n=== Validation ===")
for name, passed in checks.items():
    print(f"{name:24s}: {'PASS' if passed else 'FAIL'}")

if not all(checks.values()):
    raise SystemExit(1)

print("\nAll tests passed.")
