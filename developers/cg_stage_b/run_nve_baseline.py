#!/usr/bin/env python3
"""Short bonded-only NVE regression at matched duration and two time steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpumd", type=Path)
    args = parser.parse_args()
    binary = args.gpumd.resolve()
    inputs = HERE.parent / "cg_stage_a/input_draft"
    report = {"scope": "Bonded-only, zero initial velocities, 50 fs per case; not physical CG validation",
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "results": []}
    for frame in ("frame4", "frame6"):
        errors = []
        for dt in (0.05, 0.025):
            steps = round(50 / dt)
            with tempfile.TemporaryDirectory(prefix="gpumd-bonded-nve-") as tmp:
                tmp = Path(tmp)
                lines = (inputs / f"{frame}.model.xyz").read_text().splitlines()
                lines[1] += ":vel:R:3"
                lines[2:] = [line + " 0 0 0" for line in lines[2:]]
                (tmp / "model.xyz").write_text("\n".join(lines) + "\n")
                shutil.copyfile(inputs / f"{frame}.molecular_force.in", tmp / "molecular_force.in")
                (tmp / "zero_lj.txt").write_text("lj 2 C O\n" + "0.0 1.0 9.0\n" * 4)
                (tmp / "run.in").write_text(
                    "potential zero_lj.txt\nmolecular_force molecular_force.in\n"
                    f"time_step {dt}\nensemble nve\ndump_thermo 1\nrun {steps}\n")
                result = subprocess.run([str(binary)], cwd=tmp, text=True, capture_output=True, timeout=120)
                item = {"frame": frame, "time_step_fs": dt, "duration_fs": 50,
                        "steps": steps, "exit_code": result.returncode}
                if result.returncode:
                    item.update(status="FAIL", log=result.stdout + result.stderr)
                else:
                    rows = [list(map(float, line.split())) for line in (tmp / "thermo.out").read_text().splitlines()
                            if line.strip() and not line.startswith("#")]
                    energies = [row[1] + row[2] for row in rows]  # KE + PE in eV
                    reference = energies[0]
                    error = max(abs(x - reference) for x in energies)
                    relative = error / abs(reference)
                    valid = len(rows) == steps and all(math.isfinite(x) for x in energies) and relative < 1e-4
                    item.update(status="PASS" if valid else "FAIL", initial_total_energy_eV=reference,
                                max_energy_deviation_eV=error, relative_energy_deviation=relative)
                    errors.append(error)
                report["results"].append(item)
                print(item, flush=True)
        # The same trajectory duration at half dt should reduce integration error.
        trend = len(errors) == 2 and errors[1] <= max(errors[0] * 0.4, 1e-9)
        report["results"].append({"frame": frame, "check": "time_step_convergence",
                                  "status": "PASS" if trend else "FAIL"})
    report["status"] = "PASS" if all(x["status"] == "PASS" for x in report["results"]) else "FAIL"
    (HERE / "nve_baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
