"""Short parser/driver/NEP checks; no NPT statistical acceptance claim."""
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from test_rigid_nvt_run_in import KB, TIME_UNIT_FS, MASSES, POSITIONS, close, frames, thermo_rows


def main():
    binary, probe, source = [Path(p).resolve() for p in sys.argv[1:]]
    result = subprocess.run([str(probe), "--probe"], capture_output=True, text=True, check=True)
    if result.stdout.startswith("SKIP:"):
        print("SKIP: no accessible GPU for rigid NPT run.in test.")
        return
    root = Path(tempfile.mkdtemp(prefix="gpumd-rigid-npt-"))
    output = ("dump_thermo 1\n"
              "dump_xyz 1 trajectory.xyz precision double mass velocity force group_labels\n")
    input_velocities = []
    for x, y, z in POSITIONS:
        input_velocities.append((0.003 + 0.001*(z-12) - 0.002*(y-12),
                                 -0.002 + 0.002*(x-12) - 0.0015*(z-12),
                                 0.001 + 0.0015*(y-12) - 0.001*(x-12)))

    def run(name, commands, *, free=False, nep=False, lattice="30 0 0 0 30 0 0 0 30",
            pbc="T T T", error=None):
        case = root / name
        case.mkdir()
        labels = [0, 1, 2, 3] if free else [0, 1, 0, 0]
        model = ["4", f'Lattice="{lattice}" pbc="{pbc}" '
                 'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1']
        for i, (r, v) in enumerate(zip(POSITIONS, input_velocities)):
            model.append("C " + " ".join(map(str, (*r, MASSES[i], *v, labels[i]))))
        (case / "model.xyz").write_text("\n".join(model) + "\n")
        if nep:
            shutil.copyfile(source / "potentials/nep/C_2022_NEP4.txt", case / "potential.txt")
        else:
            (case / "potential.txt").write_text("lj 1 C\n0.0 1.0 9.0\n")
        (case / "run.in").write_text("potential potential.txt\n" + commands + "\n")
        result = subprocess.run([str(binary)], cwd=case, capture_output=True, text=True, timeout=60)
        log = result.stdout + result.stderr
        (case / "gpumd.log").write_text(log)
        if error:
            assert result.returncode != 0 and error in log, (name, log)
        else:
            assert result.returncode == 0, (name, log)
        return case, log

    def check(case, dof, count, *, free=False, zero_force=False):
        trajectory, rows = frames(case / "trajectory.xyz"), thermo_rows(case / "thermo.out")
        assert len(trajectory) == len(rows) == count
        for frame, row in zip(trajectory, rows):
            assert len(row) == 18 and all(map(math.isfinite, row))
            lengths = [row[9], row[13], row[17]]
            assert min(lengths) > 0
            close(lengths[0], lengths[1]); close(lengths[0], lengths[2])
            kinetic = sum(0.5*a["mass"][0]*sum((v*TIME_UNIT_FS)**2 for v in a["vel"])
                          for a in frame)
            close(row[1], kinetic)
            close(row[0], 2*kinetic/(dof*KB))
            if free and zero_force:
                close(sum(row[3:6])/3, 2*kinetic/(3*math.prod(lengths))*160.2177)
            if not free:
                for i, j in [(0, 2), (0, 3), (2, 3)]:
                    r = [frame[i]["pos"][c]-frame[j]["pos"][c] for c in range(3)]
                    r = [d-L*round(d/L) for d, L in zip(r, lengths)]
                    initial = [a-b for a, b in zip(POSITIONS[i], POSITIONS[j])]
                    close(sum(d*d for d in r), sum(d*d for d in initial))
                    dv = [a-b for a, b in zip(frame[i]["vel"], frame[j]["vel"])]
                    close(sum(a*b for a, b in zip(r, dv)), 0)
        return rows

    try:
        case, log = run("mixed", "time_step 0.1\nensemble rigid_npt 300 300 1000 0.001 10000 0\n" +
                        output + "run 30\ntime_step 0.05\n" + output + "run 10")
        rows = check(case, 9, 40)
        assert log.count("Rigid NPT initialized: 1 bodies, 1 free particles, 9 thermal DOF.") == 2
        header = (case / "thermo.out").read_text()
        assert header.count("# ensemble rigid_npt") == 2
        assert "# rigid_constraint_estimator instantaneous_mechanical" in header
        assert abs(rows[-1][9] - 30) > 1e-8

        case, _ = run("free", "time_step 0.1\nensemble rigid_npt 300 300 1000 0.25 10000 0\n" +
                      output + "run 10", free=True)
        rows = check(case, 12, 10, free=True, zero_force=True)
        # Initial half piston kick also checks GPa -> eV/A^3 conversion and that
        # the first pressure is computed after initialization, from actual KE.
        dt = 0.1/TIME_UNIT_FS
        kt = sum(m*sum((v*TIME_UNIT_FS)**2 for v in vel)
                 for m, vel in zip(MASSES, input_velocities))
        mass = 15*KB*300*(10000*dt)**2
        nu = dt/2*(kt/3-0.25/160.2177*30**3+kt/12)/mass
        close(rows[0][9], 30*math.exp(dt*nu), tolerance=2e-10)

        case, _ = run("nep", "time_step 0.01\nensemble rigid_npt 300 300 10000 0 100000 0\n" +
                      output + "run 20", nep=True)
        rows = check(case, 9, 20)
        trajectory = frames(case / "trajectory.xyz")
        assert max(abs(v) for f in trajectory for a in f for v in a["forces"]) > 1e-8
        # Independently recover the scalar molecular pressure from the dumped
        # interaction virial, bead forces and COM/free velocities under NEP.
        headers = (case / "trajectory.xyz").read_text().splitlines()[1::6]
        members, body_mass = (0, 2, 3), MASSES[0]+MASSES[2]+MASSES[3]
        for frame, row, header in zip(trajectory, rows, headers):
            virial = [float(v) for v in re.search(r'virial="([^"]+)"', header).group(1).split()]
            vcm = [sum(MASSES[i]*frame[i]["vel"][c]*TIME_UNIT_FS for i in members)/body_mass
                   for c in range(3)]
            xcm = [sum(MASSES[i]*frame[i]["pos"][c] for i in members)/body_mass for c in range(3)]
            kt = body_mass*sum(v*v for v in vcm)
            kt += MASSES[1]*sum((v*TIME_UNIT_FS)**2 for v in frame[1]["vel"])
            internal = sum((frame[i]["pos"][c]-xcm[c])*frame[i]["forces"][c]
                           for i in members for c in range(3))
            expected = (kt+virial[0]+virial[4]+virial[8]-internal)/(3*row[9]*row[13]*row[17])
            close(sum(row[3:6])/3, expected*160.2177)

        case, _ = run("switch", "time_step 0.1\nensemble rigid_npt 300 300 1000 0 10000 0\n" +
                      output + "run 3\nensemble rigid_nve 0\n" + output +
                      "run 3\nensemble rigid_nvt 300 300 1000 0\n" + output + "run 3")
        check(case, 9, 9)

        for name, command, error in [
            ("ramp", "300 400 1000 0 10000 0", "constant temperature"),
            ("coupling", "300 300 1000 0 0 0", "pressure coupling >= 1"),
            ("nan", "300 300 1000 nan 10000 0", "finite pressure"),
            ("group", "300 300 1000 0 10000 1", "grouping-method index"),
        ]:
            run(name, f"ensemble rigid_npt {command}\ntime_step 0.1\nrun 1", error=error)
        for name, kwargs in [("triclinic", {"lattice": "30 0 0 1 30 0 0 0 30"}),
                             ("nonperiodic", {"pbc": "T T F"})]:
            run(name, "ensemble rigid_npt 300 300 1000 0 10000 0\ntime_step 0.1\nrun 1",
                error="fully periodic orthogonal box", **kwargs)
        print("Rigid NPT run.in, consecutive runs, pressure units, NEP and input guards passed.")
    except Exception:
        print(f"Preserved failing cases: {root}")
        raise
    else:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
