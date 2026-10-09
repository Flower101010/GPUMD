#!/usr/bin/env python3
"""Check the two draft geometries using the existing v2 MD input and GPU kernels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def read_xyz(path):
    frames = []
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        count = int(lines[index])
        meta = dict(token.split("=", 1) for token in shlex.split(lines[index + 1]) if "=" in token)
        schema = meta["Properties"].split(":")
        properties, offset = {}, 0
        for p in range(0, len(schema), 3):
            width = int(schema[p + 2])
            properties[schema[p]] = slice(offset, offset + width)
            offset += width
        atoms = [line.split() for line in lines[index + 2:index + 2 + count]]
        if len(atoms) != count or any(len(row) != offset for row in atoms):
            raise ValueError("Invalid XYZ property count")
        frames.append((meta, properties, atoms))
        index += count + 2
    return frames


def maximum_error(a, b):
    if len(a) != len(b) or not all(math.isfinite(x) for x in a + b):
        raise ValueError("Incompatible/nonfinite output")
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpumd", type=Path)
    parser.add_argument("--report", type=Path, default=HERE / "fixture_md_baseline.json")
    args = parser.parse_args()
    executable = args.gpumd.resolve()
    report = {"binary": str(executable), "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "scope": "One NVE step at 1e-8 fs with zero velocities; near-static GPU E/F/W comparison, not stability testing",
              "tolerance": 1e-8, "results": []}
    inputs = HERE / "input_draft"
    for meta, prop, atoms in read_xyz(inputs / "train.xyz.draft"):
        name = meta["config_type"]
        with tempfile.TemporaryDirectory(prefix=f"gpumd-{name}-") as tmp:
            tmp = Path(tmp)
            lines = (inputs / f"{name}.model.xyz").read_text().splitlines()
            lines[1] += ":vel:R:3"
            lines[2:] = [line + " 0 0 0" for line in lines[2:]]
            (tmp / "model.xyz").write_text("\n".join(lines) + "\n")
            shutil.copyfile(inputs / f"{name}.molecular_force.in", tmp / "molecular_force.in")
            (tmp / "zero_lj.txt").write_text("lj 2 C O\n" + "0.0 1.0 9.0\n" * 4)
            (tmp / "run.in").write_text(
                "potential zero_lj.txt\nmolecular_force molecular_force.in\n"
                "time_step 0.00000001\nensemble nve\ndump_thermo 1\n"
                "dump_xyz 1 result.xyz force potential virial precision double\nrun 1\n")
            process = subprocess.run([str(executable)], cwd=tmp, text=True, capture_output=True, timeout=120)
            item = {"frame": name, "exit_code": process.returncode,
                    "run_in": (tmp / "run.in").read_text(), "log": process.stdout + process.stderr}
            try:
                if process.returncode != 0:
                    raise RuntimeError("GPUMD execution failed")
                outputs = read_xyz(tmp / "result.xyz")
                if len(outputs) != 1:
                    raise ValueError("Expected one output frame")
                _, output_prop, output_atoms = outputs[0]
                if len(output_atoms) != len(atoms):
                    raise ValueError("Bead count mismatch")
                expected_force = [float(x) for row in atoms for x in row[prop["force"]]]
                actual_force = [float(x) for row in output_atoms for x in row[output_prop["forces"]]]
                actual_energy = sum(float(row[output_prop["energy_atom"]][0]) for row in output_atoms)
                actual_virial = [sum(float(row[output_prop["virial"]][c]) for row in output_atoms) for c in range(9)]
                item.update(energy_error=abs(actual_energy - float(meta["energy"])),
                            max_force_error=maximum_error(actual_force, expected_force),
                            max_virial_error=maximum_error(actual_virial, list(map(float, meta["virial"].split()))),
                            output_xyz=(tmp / "result.xyz").read_text())
                errors = [item[key] for key in ("energy_error", "max_force_error", "max_virial_error")]
                if not all(math.isfinite(x) and x <= report["tolerance"] for x in errors):
                    raise ValueError("GPU E/F/W mismatch")
                item["status"] = "PASS"
            except Exception as error:
                item.update(status="FAIL", reason=str(error))
            report["results"].append(item)
            print(f"{item['status']}: {name}", flush=True)
    report["status"] = "PASS" if len(report["results"]) == 2 and all(
        item["status"] == "PASS" for item in report["results"]) else "FAIL"
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
