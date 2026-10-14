from pathlib import Path
import os
import re
import shutil
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "free_rotation_general"
ROOT = HERE.parents[3]
GPUMD = Path(
    os.environ.get("GPUMD_BIN", ROOT / "build/gpumd")
).expanduser().resolve()

CONVERSION = 103.64269652680505

# timestep (fs), number of steps, dump interval
CASES = [
    (1.0,   5000,  50),
    (0.5,  10000, 100),
    (0.25, 20000, 200),
    (0.125, 40000, 400),
]


def read_reference():
    lines = (SOURCE / "model.xyz").read_text().splitlines()
    n = int(lines[0])
    assert n == 3

    data = [line.split() for line in lines[2:2 + n]]
    mass = np.array([float(row[4]) for row in data])
    xyz = np.array([[float(x) for x in row[1:4]] for row in data])
    vel = np.array([[float(x) for x in row[5:8]] for row in data])

    return mass, xyz, vel


def read_trajectory(path):
    positions, velocities, times = [], [], []

    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue

            n = int(line)
            header = f.readline()

            match = re.search(r"\bTime=([0-9.eE+-]+)", header)
            assert match, "Missing Time in XYZ header"
            times.append(float(match.group(1)))

            match = re.search(r"\bProperties=([^\s]+)", header)
            assert match, "Missing Properties in XYZ header"

            schema = match.group(1).split(":")
            offsets = {}
            offset = 0

            for k in range(0, len(schema), 3):
                name = schema[k]
                count = int(schema[k + 2])
                offsets[name] = slice(offset, offset + count)
                offset += count

            assert "pos" in offsets and "vel" in offsets

            xyz, vel = [], []

            for _ in range(n):
                row = f.readline().split()
                xyz.append([float(x) for x in row[offsets["pos"]]])
                vel.append([float(x) for x in row[offsets["vel"]]])

            positions.append(xyz)
            velocities.append(vel)

    return (
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(times),
    )


def angular_momentum(xyz, vel, mass):
    com = np.average(xyz, weights=mass, axis=-2)
    vcom = np.average(vel, weights=mass, axis=-2)

    return np.sum(
        mass[..., None] * np.cross(
            xyz - np.expand_dims(com, axis=-2),
            vel - np.expand_dims(vcom, axis=-2),
        ),
        axis=-2,
    )


def run_case(dt, steps, interval):
    name = f"dt_{dt:g}".replace(".", "p")
    directory = HERE / name
    directory.mkdir(exist_ok=True)

    for filename in ("model.xyz", "zero_lj.txt"):
        shutil.copyfile(SOURCE / filename, directory / filename)

    run_in = f"""potential zero_lj.txt
ensemble rigid_nve 0
time_step {dt}
dump_thermo {interval}
dump_xyz {interval} trajectory.xyz precision double velocity force group_labels
run {steps}
"""
    (directory / "run.in").write_text(run_in)

    # Prevent append-mode output from contaminating results.
    for filename in ("trajectory.xyz", "thermo.out", "gpumd.log"):
        (directory / filename).unlink(missing_ok=True)

    print(f"Running {name}: dt={dt:g} fs, steps={steps}", flush=True)

    with open(directory / "gpumd.log", "w") as log:
        result = subprocess.run(
            [str(GPUMD)],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode:
        lines = (directory / "gpumd.log").read_text().splitlines()
        print("\n".join(lines[-30:]))
        raise RuntimeError(f"GPUMD failed for {name}")

    return directory


def main():
    assert GPUMD.is_file(), f"GPUMD not found: {GPUMD}"

    mass, r0, v0 = read_reference()

    K0 = 0.5 * CONVERSION * np.sum(mass[:, None] * v0**2)
    L0 = angular_momentum(r0, v0, mass)

    results = []

    for dt, steps, interval in CASES:
        directory = run_case(dt, steps, interval)
        r, v, times = read_trajectory(directory / "trajectory.xyz")

        assert len(r) == 100, f"Wrong frame count at dt={dt}"
        assert np.allclose(
            times, np.arange(1, 101) * 50.0, atol=1e-6
        ), f"Wrong output times at dt={dt}"
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))

        # Kinetic energy calculated independently from bead velocities
        K = (
            0.5 * CONVERSION
            * np.sum(mass[None, :, None] * v**2, axis=(1, 2))
        )
        kinetic_error = np.max(np.abs(K - K0)) / K0

        # Energy reported by GPUMD
        thermo = np.atleast_2d(
            np.loadtxt(directory / "thermo.out")
        )
        assert len(thermo) == 100

        E = thermo[:, 1] + thermo[:, 2]
        energy_drift = np.max(np.abs(E - E[0])) / K0

        # Thermo/trajectory kinetic-energy consistency
        consistency = np.max(np.abs(E - K)) / K0

        # Laboratory-frame angular momentum conservation
        L = angular_momentum(r, v, mass)
        angular_error = np.max(
            np.linalg.norm(L - L0, axis=1)
        ) / np.linalg.norm(L0)

        # Rigid distance conservation
        distance_error = 0.0
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            d = np.linalg.norm(r[:, i] - r[:, j], axis=1)
            d0 = np.linalg.norm(r0[i] - r0[j])
            distance_error = max(
                distance_error, np.max(np.abs(d - d0))
            )

        if np.max(np.abs(thermo[:, 2])) > 1e-10:
            raise RuntimeError(f"Nonzero potential at dt={dt}")
        if angular_error > 1e-8 or distance_error > 1e-7:
            raise RuntimeError(
                f"Constraint or angular momentum failure at dt={dt}"
            )

        results.append((
            dt, kinetic_error, energy_drift,
            consistency, angular_error
        ))

    results = np.asarray(results)

    print("\n=== Timestep Convergence ===")
    print(
        f"{'dt (fs)':>9} {'K error':>14} "
        f"{'E drift':>14} {'E-K mismatch':>14} {'L error':>14}"
    )

    for dt, ke, ed, ec, le in results:
        print(
            f"{dt:9.3f} {ke:14.6e} "
            f"{ed:14.6e} {ec:14.6e} {le:14.6e}"
        )

    dt = results[:, 0]

    for column, name in [
        (1, "Kinetic energy error"),
        (2, "Total energy drift"),
        (3, "Thermo/bead energy mismatch"),
    ]:
        errors = results[:, column]
        if np.any(errors <= 1e-15):
            print(f"\n{name}: below resolution for slope fitting")
            continue

        order = np.polyfit(np.log(dt), np.log(errors), 1)[0]
        ratios = errors[:-1] / errors[1:]

        print(f"\n{name}")
        print(f"  Observed order: {order:.4f}")
        print(f"  Halving ratios: {ratios}")

    print("\nConvergence analysis completed.")


if __name__ == "__main__":
    main()
