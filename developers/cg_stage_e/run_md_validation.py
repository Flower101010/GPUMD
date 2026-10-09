#!/usr/bin/env python3
"""Train/export a small package; compare NEP and MD, reject mismatches, run NVE."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cg_stage_a"))
from run_fixture_md import read_xyz, maximum_error
sys.path.insert(0, str(HERE.parent / "cg_stage_d"))
from run_training_validation import CONFIG, BONDED, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gpumd", type=Path)
    parser.add_argument("nep", type=Path)
    parser.add_argument("test_bonded_nep", type=Path)
    parser.add_argument("--example-directory", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "md_baseline.json")
    args = parser.parse_args()
    gpumd, nep, unit = (p.resolve() for p in (args.gpumd, args.nep, args.test_bonded_nep))
    example = args.example_directory.resolve() if args.example_directory else None
    if example and example.exists():
        raise RuntimeError("Example directory already exists; refusing to overwrite it")
    fixture = HERE.parent / "cg_stage_a/input_draft"
    report = {"comparison": [], "nve": [], "guards": [], "package_checks": {},
              "binary_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (gpumd, nep)}}
    with tempfile.TemporaryDirectory(prefix="gpumd-stage-e-") as temporary, (HERE / "validation.log").open("w") as log:
        root = Path(temporary)
        def run(command, work, expected=None):
            result = subprocess.run([str(x) for x in command], cwd=work, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            log.write(f"\nRUN {command} cwd={work}\n{result.stdout}"); log.flush()
            if expected:
                if result.returncode == 0 or expected not in result.stdout:
                    raise RuntimeError(f"Expected {expected}:\n{result.stdout}")
            elif result.returncode:
                raise RuntimeError(result.stdout)
            return result.stdout
        exported = root / "exported"
        run([unit, "--export", exported], root)
        train = root / "training"; train.mkdir()
        shutil.copyfile(exported / "teacher.xyz", train / "train.xyz")
        shutil.copyfile(exported / "teacher.xyz", train / "test.xyz")
        shutil.copyfile(exported / "initial.txt", train / "nep.txt")
        shutil.copyfile(exported / "initial.restart", train / "nep.restart")
        shutil.copyfile(fixture / "bonded_parameters.in.draft", train / "bonded.in")
        (train / "nep.in").write_text(CONFIG + BONDED + "import_q_scaler 1\nbatch 1\n"
            "generation 1\npopulation 10\noutput_interval 1\nsave_potential 1 1 0\nnep_compile off\n")
        run([nep], train)
        manifests = list(train.glob("*.json"))
        assert len(manifests) == 2, "Expected current and checkpoint manifests"
        for manifest in manifests:
            content = json.loads(manifest.read_text())
            for name, digest in (("nep_file", "nep_sha256"), ("bonded_parameters_file", "bonded_parameters_sha256")):
                assert hashlib.sha256((train / content[name]).read_bytes()).hexdigest() == content[digest]
            assert content["bead_types"] == ["C", "O"]
        report["package_checks"] = {"current_and_checkpoint_manifests": len(manifests),
            "hashlib_cross_check": True, "types": ["C", "O"]}
        for file in train.glob("*.out"): file.unlink()
        (train / "nep.in").write_text(CONFIG + BONDED + "prediction 1\nbatch 1\nstream_train 1\nnep_compile off\n")
        run([nep], train); reference = outputs(train)
        package = root / "package"; package.mkdir()
        content = json.loads((train / "cg_model.json").read_text())
        for name in ("cg_model.json", content["nep_file"], content["bonded_parameters_file"]):
            shutil.copyfile(train / name, package / name)
        # The exported package is independent of the original parameter path and training directory.
        (train / "bonded.in").write_text("corrupt original input after export\n")
        if example:
            shutil.copytree(package, example)
        atom_offset = 0
        for meta, props, atoms in read_xyz(train / "train.xyz"):
            n = len(atoms); name = f"frame{n}"
            work = root / name; work.mkdir()
            lines = [str(n), 'pbc="T T T" Lattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3']
            for row in atoms:
                # Feed MD exactly the float coordinate values used by the training loader.
                xyz = [struct.unpack("f", struct.pack("f", float(x)))[0] for x in row[props["pos"]]]
                mass = 12 if row[0] == "C" else 16
                lines.append(row[0] + " " + " ".join(format(x, ".17g") for x in xyz) + f" {mass} 0 0 0")
            (work / "model.xyz").write_text("\n".join(lines) + "\n")
            shutil.copyfile(fixture / f"{name}.topology.in.draft", work / "topology.in")
            shutil.copyfile(fixture / f"{name}.molecular_force.in", work / "combined.in")
            shutil.copyfile(fixture / "bonded_parameters.in.draft", work / "parameters.in")
            shutil.copyfile(package / content["nep_file"], work / "manual_nep.txt")
            bundle = "cg_model ../package/cg_model.json topology.in\n"
            md_tail = ("time_step 0.00000001\nensemble nve\ndump_thermo 1\n"
                       "dump_xyz 1 result.xyz force potential virial precision double\nrun 1\n")
            def static(declaration):
                for file in (work / "result.xyz", work / "thermo.out"):
                    if file.exists(): file.unlink()
                (work / "run.in").write_text(declaration + md_tail); run([gpumd], work)
                _, schema, rows = read_xyz(work / "result.xyz")[0]
                return {"energy": sum(float(row[schema["energy_atom"]][0]) for row in rows),
                        "force": [float(x) for row in rows for x in row[schema["forces"]]],
                        "virial": [sum(float(row[schema["virial"]][k]) for row in rows) for k in range(9)]}
            actual = static(bundle)
            for mode, declaration in (("split", "potential manual_nep.txt\nmolecular_force parameters.in topology.in\n"),
                                      ("legacy_v2", "potential manual_nep.txt\nmolecular_force combined.in\n")):
                other = static(declaration)
                assert abs(other["energy"] - actual["energy"]) < 2e-12
                assert maximum_error(other["force"], actual["force"]) < 2e-12
                assert maximum_error(other["virial"], actual["virial"]) < 2e-12
            nc = len(report["comparison"])
            expected_e = reference["energy_train"][nc][0] * n
            expected_f = [x for row in reference["force_train"][atom_offset:atom_offset+n] for x in row[:3]]
            expected_w = [x*n for x in reference["virial_train"][nc][:6]]
            md_six = [actual["virial"][k] for k in (0,4,8,1,5,6)]
            errors = {"energy_error_eV": abs(actual["energy"]-expected_e),
                      "max_force_error_eV_per_A": maximum_error(actual["force"], expected_f),
                      "max_virial_error_eV": maximum_error(md_six, expected_w)}
            assert max(errors.values()) < 3e-5, errors
            report["comparison"].append({"frame":name,"errors":errors,"legacy_and_split_equivalent":True})
            atom_offset += n
            print(f"PASS {name}: NEP/MD total comparison and legacy/split/bundle equivalence",flush=True)
            original_model = (work / "model.xyz").read_text()
            subset_lines = original_model.splitlines()
            for index in range(2, len(subset_lines)):
                columns = subset_lines[index].split()
                columns[0] = "O"
                columns[4] = "16"
                subset_lines[index] = " ".join(columns)
            (work / "model.xyz").write_text("\n".join(subset_lines)+"\n")
            subset = static(bundle)
            compact = static("potential manual_nep.txt\nmolecular_force parameters.in topology.in\n")
            assert abs(subset["energy"]-compact["energy"]) < 1e-10
            assert maximum_error(subset["force"],compact["force"]) < 1e-10
            assert maximum_error(subset["virial"],compact["virial"]) < 1e-10
            report["comparison"][-1]["absent_C_type_mapping_checked"] = True
            (work / "model.xyz").write_text(original_model)
            original_topology = (work / "topology.in").read_text()
            topology_lines = original_topology.splitlines()
            repeated = ["gpumd_topology 1", f"number_of_atoms {2*n}"]
            offset = 2
            for section in ("bonds", "angles", "dihedrals"):
                key, count = topology_lines[offset].split(); count = int(count)
                assert key == section
                entries = topology_lines[offset+1:offset+1+count]
                repeated.append(f"{section} {2*count}")
                repeated.extend(entries)
                for entry in entries:
                    values = [int(x) for x in entry.split()]
                    repeated.append(" ".join(str(x+n) for x in values[:-1])+f" {values[-1]}")
                offset += 1+count
            (work / "topology.in").write_text("\n".join(repeated)+"\n")
            replicated = static("replicate 2 1 1\n"+bundle)
            assert abs(replicated["energy"]-2*actual["energy"]) < 1e-10
            assert maximum_error(replicated["virial"],[2*x for x in actual["virial"]]) < 1e-10
            assert len(replicated["force"]) == 6*n
            (work / "topology.in").write_text(original_topology)
            static(bundle+"dump_restart 1\n")
            shutil.copyfile(work / "restart.xyz", work / "model.xyz")
            restarted = static(bundle)
            assert abs(restarted["energy"]-actual["energy"]) < 3e-5
            assert maximum_error(restarted["force"], actual["force"]) < 3e-5
            assert maximum_error(restarted["virial"], actual["virial"]) < 3e-5
            report["comparison"][-1]["restart_and_replicate_before_checked"] = True
            (work / "model.xyz").write_text(original_model)
            if example:
                dest = example / name; dest.mkdir()
                for file in ("model.xyz", "topology.in"): shutil.copyfile(work/file, dest/file)
                (dest / "run.in").write_text("cg_model ../cg_model.json topology.in\ntime_step 0.05\nensemble nve\ndump_thermo 1\nrun 1000\n")
            errors_dt = []
            for dt in (0.05,0.025):
                steps = round(50/dt)
                (work / "run.in").write_text(bundle + f"time_step {dt}\nensemble nve\ndump_thermo 1\nrun {steps}\n")
                (work / "thermo.out").unlink(); run([gpumd], work)
                thermo = [[float(x) for x in line.split()] for line in (work / "thermo.out").read_text().splitlines()
                          if line.strip() and not line.lstrip().startswith("#")]
                energies = [row[1]+row[2] for row in thermo]
                assert len(energies)==steps and all(math.isfinite(e) for e in energies)
                error = max(abs(e-energies[0]) for e in energies); relative = error/abs(energies[0])
                assert relative < 1e-4
                errors_dt.append(error)
                report["nve"].append({"frame":name,"dt_fs":dt,"duration_fs":50,
                    "max_energy_deviation_eV":error,"relative_deviation":relative})
            assert errors_dt[1] <= max(0.5*errors_dt[0],1e-8), errors_dt
            for key, declaration, diagnostic in (
                ("missing_topology", "cg_model ../package/cg_model.json\n", "requires manifest_file topology_file"),
                ("duplicate_model", bundle+bundle, "only potential/bonded model declaration"),
                ("mixed_model", bundle+"potential manual_nep.txt\n", "cannot be combined"),
                ("mixed_model_before", "potential manual_nep.txt\n"+bundle, "only potential/bonded model declaration"),
                ("replicate_after", bundle+"replicate 2 1 1\n", "replicate must appear before"),
                ("residual_only", "potential ../package/nep.txt\n", "companion manifest")):
                (work / "run.in").write_text(declaration + md_tail)
                run([gpumd],work,expected=diagnostic);report["guards"].append(name+":"+key)
            (work / "run.in").write_text(bundle+md_tail)
            original = (work / "topology.in").read_text()
            (work / "topology.in").write_text(original.replace(f"number_of_atoms {n}",f"number_of_atoms {n+1}"))
            run([gpumd],work,expected="number_of_atoms differs");report["guards"].append(name+":atom_count")
            (work / "topology.in").write_text(original)
            (work / "model.xyz").write_text(original_model.replace('pbc="T T T"','pbc="T T F"'))
            run([gpumd],work,expected="three-dimensional periodic");report["guards"].append(name+":pbc")
            (work / "model.xyz").write_text(original_model)
            member = package / content["bonded_parameters_file"]; source = member.read_bytes()
            member.write_bytes(source+b"# changed\n")
            run([gpumd],work,expected="checksum mismatch");report["guards"].append(name+":coefficients_checksum")
            member.write_bytes(source)
        report["scope"] = "two synthetic frames, one GPU, 50 fs; not real CG physical validation"
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("PASS stage E: package, training/MD, input guards, combined NVE",flush=True)


if __name__ == "__main__": main()
