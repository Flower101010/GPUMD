#!/usr/bin/env python3
"""Run existing host tests and build the stage-A fixture; no production code changes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, default=ROOT.parent / "GPUMD-5.6_bondangledihedral_need_test")
    parser.add_argument("--report", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--fixture-dir", type=Path, default=HERE / "input_draft")
    args = parser.parse_args()
    donor = args.donor.resolve()
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    report = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository": str(ROOT), "donor": str(donor),
        "scope": "Host-only baseline; no CUDA kernels, NEP training or MD executed",
        "toolchain": {name: shutil.which(name) for name in ("g++", "nvcc", "cmake", "ctest")},
        "results": [], "source_sha256": {},
    }
    report["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["compiler"] = subprocess.check_output(compiler + ["--version"], text=True).splitlines()[0]
    sources = set()
    success = True
    with tempfile.TemporaryDirectory(prefix="gpumd-cg-stage-a-") as tmp:
        tmp = Path(tmp)

        def run_test(name, root, files, argv=(), defines=()):
            nonlocal success
            exe = tmp / name
            sources.update(root / f for f in files)
            command = compiler + ["-std=c++17", "-Wall", "-Wextra", "-pedantic", "-UNDEBUG",
                                  "-I" + str(root / "src"), *defines, "-x", "c++"]
            command += [str(root / f) for f in files] + ["-o", str(exe)]
            build = subprocess.run(command, cwd=tmp, text=True, capture_output=True)
            item = {"name": name, "compile_command": command, "compile_exit_code": build.returncode,
                    "compile_output": build.stdout + build.stderr}
            if build.returncode:
                item["status"] = "FAIL"
            else:
                invocation = [str(exe), *map(str, argv)]
                result = subprocess.run(invocation, cwd=tmp, text=True, capture_output=True)
                item.update(run_command=invocation, run_exit_code=result.returncode,
                            run_output=result.stdout + result.stderr,
                            status="PASS" if result.returncode == 0 else "FAIL")
            success = success and item["status"] == "PASS"
            report["results"].append(item)
            print(f"{item['status']}: {name}", flush=True)
            return item["status"] == "PASS"

        model = ["src/model/topology.cu", "src/model/force_field_parameters.cu"]
        run_test("own_topology", ROOT, ["tests/test_topology.cpp", model[0]])
        run_test("own_parameters", ROOT, ["tests/test_force_field_parameters.cpp", *model])
        run_test("own_reader", ROOT, ["tests/test_read_molecular_force.cpp", *model,
                                     "src/model/read_molecular_force.cu"])
        run_test("own_geometry", ROOT, ["tests/test_bonded_geometry.cu"],
                 defines=["-D__host__=", "-D__device__="])
        if donor.is_dir():
            run_test("donor_cg_topology", donor,
                     ["tests/cg_topology_unit.cpp", "src/utilities/coarse_grained.cu", "src/utilities/error.cu"],
                     argv=[donor / "tests/cg_topology.in", donor / "tests/cg_coefficients.in",
                           donor / "examples/coarse_grained_polymer/train_variable.xyz"])
        else:
            success = False
            report["results"].append({"name": "donor_cg_topology", "status": "NOT_RUN",
                                      "reason": "Donor directory missing"})
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        run_test("stage_a_fixture", ROOT, ["developers/cg_stage_a/make_fixture.cpp", *model,
                 "src/model/read_molecular_force.cu"], argv=[args.fixture_dir.resolve()],
                 defines=["-D__host__=", "-D__device__="])

    # Include production GPU and integration sources even though they were not executed.
    for name in ("src/force/harmonic_bond.cu", "src/force/harmonic_angle.cu",
                 "src/force/periodic_dihedral.cu", "src/force/bonded_geometry.cuh",
                 "src/force/molecular_force.cu", "src/force/force.cu",
                 "src/main_gpumd/run.cu", "src/main_nep/dataset.cu",
                 "src/main_nep/structure.cu", "src/main_nep/nep.cu",
                 "CMakeLists.txt", "tests/CMakeLists.txt"):
        sources.add(ROOT / name)
    for name in ("src/force/coarse_grained.cu", "src/utilities/coarse_grained.cuh",
                 "src/force/force.cu", "src/main_nep/dataset.cu", "src/main_nep/structure.cu",
                 "src/main_nep/nep.cu", "src/main_nep/parameters.cu"):
        if (donor / name).is_file():
            sources.add(donor / name)
    for source in sorted(sources):
        report["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["results"].append({"name": "gpu_md_training_baseline", "status": "NOT_RUN",
                              "reason": "This runner is host-only; GPU/MD/training need a separate CUDA build"})
    report["host_baseline_status"] = "PASS" if success else "FAIL"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {args.report.resolve()}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
