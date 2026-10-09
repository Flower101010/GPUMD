#!/usr/bin/env python3
"""Check ordinary NEP train/predict paths and refusal to silently ignore CG metadata."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


def run(executable, directory):
    return subprocess.run([str(executable)], cwd=directory, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)


def numeric_file(path):
    rows = [[float(token) for token in line.split()] for line in path.read_text().splitlines()
            if line.strip()]
    if not rows or any(not math.isfinite(value) for row in rows for value in row):
        raise RuntimeError(f"Empty or nonfinite output: {path.name}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nep", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("nep_smoke_baseline.json"))
    args = parser.parse_args()
    executable = args.nep.resolve()
    fixture = Path(__file__).resolve().parents[1] / "cg_stage_a/input_draft/train.xyz.draft"
    xyz = fixture.read_text()
    ordinary = re.sub(r'cg_topology_version=1 cg_bonds="[^"]*" cg_angles="[^"]*" cg_dihedrals="[^"]*" ', '', xyz)
    if ordinary == xyz:
        raise RuntimeError("Fixture CG fields were not removed")
    configuration = ("type 2 C O\ncutoff 4 3\nn_max 2 2\nbasis_size 2 2\nneuron 4\n"
                     "batch 1\npopulation 10\ngeneration 1\noutput_interval 1\nnep_compile off\n")
    checks = {}
    with tempfile.TemporaryDirectory(prefix="gpumd-stage-c-nep-") as temp:
        work = Path(temp)
        (work / "nep.in").write_text(configuration)
        (work / "train.xyz").write_text(ordinary)
        (work / "test.xyz").write_text(ordinary)
        result = run(executable, work)
        if result.returncode or "Finished running nep." not in result.stdout:
            raise RuntimeError(result.stdout)
        checks["ordinary_training"] = {"exit_code": result.returncode,
                                       "loss_rows": numeric_file(work / "loss.out")}
        if not (work / "nep.txt").is_file():
            raise RuntimeError("Training did not write nep.txt")
        (work / "nep.in").write_text(configuration + "prediction 1\n")
        result = run(executable, work)
        if result.returncode or "Finished running nep." not in result.stdout:
            raise RuntimeError(result.stdout)
        checks["ordinary_prediction"] = {
            "exit_code": result.returncode,
            "output_rows": {name: numeric_file(work / name) for name in
                            ("energy_train.out", "force_train.out", "virial_train.out",
                             "energy_test.out", "force_test.out", "virial_test.out")}}
        (work / "train.xyz").write_text(xyz)
        result = run(executable, work)
        if result.returncode == 0 or "CG topology requires explicit bonded parameters" not in result.stdout:
            raise RuntimeError("CG metadata was silently accepted: " + result.stdout)
        checks["incomplete_feature_guard"] = {"nonzero_exit": True,
            "diagnostic": "CG topology requires explicit bonded parameters"}
    report = {"checks": checks, "nep_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "scope": "ordinary NEP smoke and stage-C guard; not CG residual-training validation"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
