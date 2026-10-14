"""Exercise the production parser/driver/output path in small isolated cases.

Uses only the Python standard library. This checks numerical integration and
input/output contracts, not canonical sampling of a production material.
"""
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

KB = 8.617343e-5
TIME_UNIT_FS = 10.18051
POSITIONS = [(12, 12, 12), (15.5, 13.5, 13), (13.2, 12.1, 12), (12.4, 13, 12)]
MASSES = [8, 18, 12, 16]


def close(actual, expected, tolerance=2e-8):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) <= tolerance * (1 + abs(expected)), (actual, expected)


def frames(path):
    lines = path.read_text().splitlines()
    result = []
    while lines:
        count = int(lines.pop(0))
        header = lines.pop(0)
        fields = re.search(r"Properties=(\S+)", header).group(1).split(":")
        frame = []
        for line in lines[:count]:
            values = line.split()
            atom = {}
            offset = 0
            for name, kind, width in zip(fields[::3], fields[1::3], fields[2::3]):
                width = int(width)
                atom[name] = values[offset:offset + width] if kind == "S" else [
                    float(v) for v in values[offset:offset + width]]
                offset += width
            frame.append(atom)
        del lines[:count]
        result.append(frame)
    return result


def thermo_rows(path):
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines()
            if line and not line.startswith("#")]


def displacement(a, b):
    return [x - y - 30 * round((x - y) / 30) for x, y in zip(a, b)]


def check_output(case, dof, steps, rigid=True):
    trajectory = frames(case / "trajectory.xyz")
    thermo = thermo_rows(case / "thermo.out")
    assert len(trajectory) == len(thermo) == steps
    for frame, row in zip(trajectory, thermo):
        assert len(row) == 18 and all(math.isfinite(v) for v in row)
        kinetic = sum(0.5 * atom["mass"][0] * sum((v * TIME_UNIT_FS) ** 2
                      for v in atom["vel"]) for atom in frame)
        close(row[1], kinetic)
        close(row[0], 2 * kinetic / (dof * KB))
        if rigid:
            for i, j in [(0, 2), (0, 3), (2, 3)]:
                r = displacement(frame[i]["pos"], frame[j]["pos"])
                initial = displacement(POSITIONS[i], POSITIONS[j])
                close(sum(v * v for v in r), sum(v * v for v in initial))
                dv = [a - b for a, b in zip(frame[i]["vel"], frame[j]["vel"])]
                close(sum(a * b for a, b in zip(r, dv)), 0)
    return thermo


def main():
    binary, probe, source = map(lambda p: Path(p).resolve(), sys.argv[1:])
    result = subprocess.run([str(probe), "--probe"], capture_output=True, text=True, check=True)
    if result.stdout.startswith("SKIP:"):
        print("SKIP: no accessible GPU for rigid NVT run.in integration test.")
        return
    root = Path(tempfile.mkdtemp(prefix="gpumd-rigid-nvt-"))

    def run(name, commands, *, free=False, nep=False, error=None):
        case = root / name
        case.mkdir()
        labels = [0, 1, 2, 3] if free else [0, 1, 0, 0]
        model = ['4', 'Lattice="30 0 0 0 30 0 0 0 30" pbc="T T T" '
                 'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1']
        for i, (x, y, z) in enumerate(POSITIONS):
            # A common rigid translation + rotation for the triangle.
            vx = 0.003 + 0.001 * (z - 12) - 0.002 * (y - 12)
            vy = -0.002 + 0.002 * (x - 12) - 0.0015 * (z - 12)
            vz = 0.001 + 0.0015 * (y - 12) - 0.001 * (x - 12)
            model.append(f"C {x} {y} {z} {MASSES[i]} {vx} {vy} {vz} {labels[i]}")
        (case / "model.xyz").write_text("\n".join(model) + "\n")
        if nep:
            shutil.copyfile(source / "potentials/nep/C_2022_NEP4.txt", case / "potential.txt")
        else:
            (case / "potential.txt").write_text("lj 1 C\n0.0 1.0 9.0\n")
        (case / "run.in").write_text("potential potential.txt\n" + commands + "\n")
        result = subprocess.run([str(binary)], cwd=case, capture_output=True, text=True, timeout=60)
        log = result.stdout + result.stderr
        (case / "gpumd.log").write_text(log)
        if error is not None:
            assert result.returncode != 0 and error in log, (name, result.returncode, log)
        else:
            assert result.returncode == 0, (name, log)
        return case, log

    output = ("dump_thermo 1\n"
              "dump_xyz 1 trajectory.xyz precision double mass velocity force group_labels\n")
    try:
        case, log = run("consecutive", "ensemble rigid_nvt 300 400 1000 0\ntime_step 0.1\n" +
                        output + "run 20\ntime_step 0.05\n" + output + "run 20")
        check_output(case, 9, 40)
        assert log.count("Rigid NVT initialized: 1 bodies, 1 free particles, 9 thermal DOF.") == 2
        header = (case / "thermo.out").read_text()
        assert header.count("# ensemble rigid_nvt") == 2 and "# thermal_dof 9" in header

        a, _ = run("free_rigid", "ensemble rigid_nvt 250 450 1000 0\ntime_step 0.1\n" +
                   output + "run 30", free=True)
        b, _ = run("free_ordinary", "ensemble nvt_nhc 250 450 1000\ntime_step 0.1\n" +
                   output + "run 30", free=True)
        check_output(a, 12, 30, rigid=False)
        # Ordinary NHC leaves thermo at the pre-final-thermostat snapshot.
        # Compare actual phase-space states; rigid NVT output is checked above
        # against the final velocities, not that legacy thermo snapshot.
        assert len(thermo_rows(b / "thermo.out")) == 30
        for af, bf in zip(frames(a / "trajectory.xyz"), frames(b / "trajectory.xyz")):
            for aa, ba in zip(af, bf):
                for key in ("pos", "vel"):
                    for av, bv in zip(aa[key], ba[key]):
                        close(av, bv)

        case, log = run("switch", "time_step 0.1\n" +
                        "ensemble rigid_nvt 300 300 1000 0\n" + output + "run 3\n" +
                        "ensemble rigid_nve 0\n" + output + "run 3\n" +
                        "ensemble nvt_nhc 300 300 1000\n" + output + "run 3\n" +
                        "ensemble nve\n" + output + "run 3")
        assert log.count("Rigid NVT initialized:") == log.count("Rigid NVE initialized:") == 1
        assert (case / "thermo.out").read_text().count("# ensemble rigid_") == 2
        assert len(thermo_rows(case / "thermo.out")) == 12

        # Short NEP execution-path check with real, nonzero bead forces.
        case, _ = run("nep", "ensemble rigid_nvt 300 300 10000 0\ntime_step 0.01\n" +
                      output + "run 20", nep=True)
        check_output(case, 9, 20)
        assert any(abs(f) > 1e-5 for frame in frames(case / "trajectory.xyz")
                   for atom in frame for f in atom["forces"])

        invalid = [
            ("arity", "ensemble rigid_nvt 300 300 1000", "requires T_initial"),
            ("group", "ensemble rigid_nvt 300 300 1000 1", "valid nonnegative molecule"),
            ("negative_group", "ensemble rigid_nvt 300 300 1000 -1", "valid nonnegative molecule"),
            ("temperature", "ensemble rigid_nvt 0 300 1000 0", "Initial temperature should > 0"),
            ("coupling", "ensemble rigid_nvt 300 300 0.5 0", "Temperature coupling should >= 1"),
            ("nan", "ensemble rigid_nvt 300 nan 1000 0", "requires finite temperatures"),
            ("infinity", "ensemble rigid_nvt 300 300 inf 0", "requires finite temperatures"),
        ]
        for name, command, error in invalid:
            run(name, command + "\nrun 1", error=error)
        incompatible = [
            ("fix", "fix 0", "without fix, move, or deform"),
            ("deform", "deform 0.001 1 0 0", "has not been adapted to rigid ensembles"),
            ("adaptive", "time_step 0.1 0.01", "positive fixed time_step"),
            ("correction", "correct_velocity 10", "correct_velocity is not supported"),
            ("unwrapped", "dump_xyz 1 unwrapped.xyz unwrapped_position", "do not track unwrapped"),
        ]
        for name, command, error in incompatible:
            run(name, "ensemble rigid_nvt 300 300 1000 0\ntime_step 0.1\n" +
                command + "\nrun 1", error=error)
        print("PASS: rigid NVT parser, ramps, consecutive runs, ensemble switches, NEP, "
              "rigid constraints, thermo DOFs and invalid inputs.")
    except Exception:
        print(f"Failed test cases retained in {root}", file=sys.stderr)
        raise
    else:
        if os.environ.get("GPUMD_KEEP_TEST_DIR") == "1":
            print(f"Kept test cases in {root}")
        else:
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
