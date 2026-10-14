import re
import numpy as np

CONVERSION = 103.64269652680505  # amu Å²/fs² -> eV

# Reference initial configuration
mass = np.array([12.0, 12.0, 12.0])

r0 = np.array([
    [19.0, 19.0, 20.0],
    [21.0, 19.0, 20.0],
    [20.0, 22.0, 20.0],
])

v0 = np.array([
    [0.003, -0.003,  0.001],
    [0.003,  0.003, -0.003],
    [-0.006,  0.000,  0.002],
])


def read_xyz(filename):
    positions = []
    velocities = []
    times = []

    with open(filename) as f:
        while line := f.readline():
            if not line.strip():
                continue

            n = int(line)
            header = f.readline()

            match = re.search(r"\bTime=([^\s]+)", header)
            if not match:
                raise ValueError("Missing Time field")
            times.append(float(match.group(1)))

            match = re.search(r"\bProperties=([^\s]+)", header)
            if not match:
                raise ValueError("Missing Properties field")

            schema = match.group(1).split(":")
            columns = {}
            offset = 0

            for i in range(0, len(schema), 3):
                name = schema[i]
                count = int(schema[i + 2])
                columns[name] = slice(offset, offset + count)
                offset += count

            if "pos" not in columns or "vel" not in columns:
                raise ValueError("XYZ must contain pos and vel")

            xyz, vel = [], []

            for _ in range(n):
                row = f.readline().split()
                xyz.append([float(x) for x in row[columns["pos"]]])
                vel.append([float(x) for x in row[columns["vel"]]])

            positions.append(xyz)
            velocities.append(vel)

    return (
        np.array(positions),
        np.array(velocities),
        np.array(times),
    )


r, v, times = read_xyz("trajectory.xyz")

assert r.ndim == 3 and r.shape[1:] == (3, 3)
assert len(r) >= 2
assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))
assert np.all(np.diff(times) > 0), "Non-increasing trajectory times"

# Center of mass
M = np.sum(mass)
com0 = np.average(r0, axis=0, weights=mass)
vcom0 = np.average(v0, axis=0, weights=mass)

com = np.average(r, axis=1, weights=mass)
vcom = np.average(v, axis=1, weights=mass)

com_error = np.max(np.linalg.norm(com - com0, axis=1))
vcom_error = np.max(np.linalg.norm(vcom - vcom0, axis=1))

# Rigid-body distances and velocity constraints
distance_error = 0.0
velocity_constraint_error = 0.0

for i, j in [(0, 1), (0, 2), (1, 2)]:
    dr = r[:, i] - r[:, j]
    dv = v[:, i] - v[:, j]
    d0 = np.linalg.norm(r0[i] - r0[j])

    distance_error = max(
        distance_error,
        np.max(np.abs(np.linalg.norm(dr, axis=1) - d0))
    )

    velocity_constraint_error = max(
        velocity_constraint_error,
        np.max(np.abs(np.sum(dr * dv, axis=1)))
    )

# Angular momentum in the laboratory frame
L = np.sum(
    mass[None, :, None]
    * np.cross(r - com[:, None], v - vcom[:, None]),
    axis=1,
)

L0 = np.sum(
    mass[:, None] * np.cross(r0 - com0, v0 - vcom0),
    axis=0,
)

L_error = np.max(
    np.linalg.norm(L - L0, axis=1)
) / np.linalg.norm(L0)

# Kinetic energy from bead velocities
K0 = 0.5 * CONVERSION * np.sum(mass[:, None] * v0**2)

K = 0.5 * CONVERSION * np.sum(
    mass[None, :, None] * v**2,
    axis=(1, 2)
)

K_error = np.max(np.abs(K - K0)) / K0

# Thermodynamic output
thermo = np.atleast_2d(np.loadtxt("thermo.out"))

K_thermo = thermo[:, 1]
U_thermo = thermo[:, 2]
E = K_thermo + U_thermo

E_error = np.max(np.abs(E - E[0])) / K0

# Results
print("\n=== General Free Rotation Test ===")
print(f"Frames:                    {len(r)}")
print(f"Distance error (Å):        {distance_error:.8e}")
print(f"COM drift (Å):             {com_error:.8e}")
print(f"COM velocity error:        {vcom_error:.8e}")
print(f"Velocity constraint err:   {velocity_constraint_error:.8e}")

print(f"\nInitial angular momentum:  {L0}")
print(f"Angular momentum rel err:  {L_error:.8e}")

print(f"\nExpected kinetic energy:   {K0:.12f} eV")
print(f"Kinetic energy rel err:    {K_error:.8e}")
print(f"Total energy rel drift:    {E_error:.8e}")

checks = {
    "Rigid distances": distance_error < 1e-8,
    "COM position": com_error < 1e-8,
    "COM velocity": vcom_error < 1e-8,
    "Velocity constraints": velocity_constraint_error < 1e-8,
    "Angular momentum": L_error < 1e-4,
    "Kinetic energy": K_error < 1e-4,
    "Total energy": E_error < 1e-4,
    "Zero potential": np.max(np.abs(U_thermo)) < 1e-10,
}

print("\n=== Validation ===")
for name, passed in checks.items():
    print(f"{name:24s}: {'PASS' if passed else 'FAIL'}")

if not all(checks.values()):
    raise SystemExit(1)

print("\nAll tests passed.")
